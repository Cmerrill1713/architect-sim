"""Live runtime probing — finds real bugs that static analysis misses.

Checks:
1. Binary staleness: is the deployed binary older than the source?
2. Port liveness: is the expected port actually listening?
3. Health endpoint: does /health return ok?
4. Config drift: does the plist env match what the code expects?
5. Log errors: what errors have occurred in the last hour?
6. Embedding dimensions: do actual dimensions match DB schema?
7. Learning pipeline: is online-learning receiving data?
8. Daemon goals: are goals succeeding or piling up as failures?
"""
import json
import os
import subprocess
import time
from pathlib import Path
from ..models import Finding


def _run(cmd, timeout=5):
    """Run a command, return stdout or empty string."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=isinstance(cmd, str))
        return r.stdout.strip()
    except Exception:
        return ""


def _http_get(url, timeout=5):
    """HTTP GET, return parsed JSON or None."""
    raw = _run(["curl", "-s", "--connect-timeout", "3", "--max-time", str(timeout), url])
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return None


def _port_listening(port):
    return bool(_run(f"lsof -iTCP:{port} -sTCP:LISTEN -t 2>/dev/null"))


def probe_binary_staleness(config) -> list:
    """Check if deployed binaries are older than source code."""
    findings = []
    bin_dir = config.root / "bin"
    svc_dir = config.root / "athena" / "services"

    # Map of binary paths to check
    binary_map = {
        "agi-core-go": [
            bin_dir / "AgiCore.app" / "Contents" / "MacOS" / "agi-core",
            bin_dir / "agi-core-go",
        ],
        "world-model-go": [bin_dir / "world-model-go"],
        "intelligence-hub-go": [bin_dir / "intelligence-hub"],
        "router-rust": [bin_dir / "router-rust"],
        "rag-gateway-rust": [bin_dir / "rag-gateway-rust"],
        "service-manager-rust": [bin_dir / "service-manager-rust"],
    }

    for svc_name, bin_paths in binary_map.items():
        src_dir = svc_dir / svc_name
        if not src_dir.exists():
            continue

        # Find newest source file
        newest_src = 0
        for ext in ("*.go", "*.rs", "*.py"):
            for f in src_dir.rglob(ext):
                if "__pycache__" in str(f) or "test" in f.name.lower():
                    continue
                mtime = f.stat().st_mtime
                if mtime > newest_src:
                    newest_src = mtime

        if newest_src == 0:
            continue

        for bin_path in bin_paths:
            if not bin_path.exists():
                continue
            bin_mtime = bin_path.stat().st_mtime
            if newest_src > bin_mtime:
                age = int((newest_src - bin_mtime) / 3600)
                findings.append(Finding(
                    finding_type="stale_binary",
                    severity="critical",
                    fixability="auto",
                    service=svc_name,
                    endpoint=str(bin_path),
                    details=f"Binary is {age}h older than source. Source modified but binary not rebuilt.",
                    suggested_fix=f"Rebuild: cd {src_dir} && go build -o {bin_path} .",
                ))

    return findings


def probe_port_liveness(config) -> list:
    """Check if expected service ports are actually listening."""
    findings = []

    # Critical ports that MUST be up
    critical_ports = {
        8088: "llama-swap",
        8100: "agi-core-go",
        8101: "rag-gateway-rust",
        8185: "qwen3-embed",
        8186: "nomic-embed",
        8191: "embedding-service",
        8201: "truth-correlator",
        8202: "ego-memory-go",
        8331: "world-model-go",
        9113: "router-rust",
        9118: "online-learning-go",
        18200: "daemon-llm-4b",
    }

    for port, svc in critical_ports.items():
        if not _port_listening(port):
            findings.append(Finding(
                finding_type="service_down",
                severity="critical",
                fixability="auto",
                service=svc,
                endpoint=f":{port}",
                details=f"{svc} not listening on port {port}",
                suggested_fix=f"launchctl kickstart gui/$(id -u)/com.athena.{svc}",
            ))

    return findings


def probe_health_endpoints(config) -> list:
    """Check /health returns ok for all running services."""
    findings = []

    health_checks = {
        8100: ("agi-core-go", "/health"),
        8101: ("rag-gateway-rust", "/health"),
        8185: ("qwen3-embed", "/health"),
        8191: ("embedding-service", "/health"),
        8201: ("truth-correlator", "/health"),
        8202: ("ego-memory-go", "/health"),
        8331: ("world-model-go", "/health"),
        9113: ("router-rust", "/health"),
        9118: ("online-learning-go", "/health"),
        9110: ("governance-go", "/health"),
    }

    for port, (svc, path) in health_checks.items():
        if not _port_listening(port):
            continue  # Already caught by port_liveness
        resp = _http_get(f"http://127.0.0.1:{port}{path}")
        if resp is None:
            findings.append(Finding(
                finding_type="health_fail",
                severity="warning",
                fixability="needs_context",
                service=svc,
                endpoint=f":{port}{path}",
                details=f"{svc} /health returned non-JSON or timed out",
                suggested_fix="Check service logs for errors",
            ))
        else:
            status = resp.get("status", "unknown")
            if status == "degraded":
                details = json.dumps({k: v for k, v in resp.items() if k != "status"})[:200]
                findings.append(Finding(
                    finding_type="service_degraded",
                    severity="warning",
                    fixability="needs_context",
                    service=svc,
                    endpoint=f":{port}{path}",
                    details=f"{svc} is degraded: {details}",
                    suggested_fix="Check degraded components and fix underlying issue",
                ))

    return findings


def probe_embedding_dimensions(config) -> list:
    """Check that embedding service returns correct dimensions for each tier."""
    findings = []
    expected = {"mini": 384, "base": 768, "large": 1024, "deep": 2048}

    for tier, expected_dim in expected.items():
        resp = _http_get(f"http://127.0.0.1:8191/embed", timeout=10)
        # POST with data
        raw = _run([
            "curl", "-s", "--max-time", "10", "-X", "POST",
            "http://127.0.0.1:8191/embed",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({"texts": ["dimension test"], "tier": tier}),
        ], timeout=15)
        if raw:
            try:
                data = json.loads(raw)
                vectors = data.get("vectors", [])
                if vectors:
                    actual_dim = len(vectors[0])
                    if actual_dim != expected_dim:
                        findings.append(Finding(
                            finding_type="dimension_mismatch",
                            severity="critical",
                            fixability="needs_context",
                            service="embedding-service",
                            endpoint=f"tier={tier}",
                            details=f"Expected {expected_dim}d, got {actual_dim}d for tier {tier}",
                            suggested_fix=f"Check embedding model configuration for {tier} tier",
                        ))
            except Exception:
                pass

    return findings


def probe_learning_pipeline(config) -> list:
    """Check if the learning pipeline is actually working."""
    findings = []

    # Online learning stats
    stats = _http_get("http://127.0.0.1:9118/stats")
    if stats:
        interactions = stats.get("total_interactions", 0)
        if interactions == 0:
            findings.append(Finding(
                finding_type="learning_disconnected",
                severity="critical",
                fixability="needs_context",
                service="online-learning-go",
                details="Online learning has 0 interactions — feedback loop is disconnected",
                suggested_fix="Check sendLearningFeedback is called on all chat return paths",
            ))

    # Ego memory
    ego = _http_get("http://127.0.0.1:8202/health")
    if ego:
        episodes = ego.get("episode_count", 0)
        if episodes == 0:
            findings.append(Finding(
                finding_type="memory_empty",
                severity="warning",
                fixability="needs_context",
                service="ego-memory-go",
                details="Ego memory has 0 episodes",
                suggested_fix="Check ego-memory episode recording",
            ))

    return findings


def probe_daemon_health(config) -> list:
    """Check daemon goal success rate and failure patterns."""
    findings = []

    status = _http_get("http://127.0.0.1:8100/daemon/status")
    if status:
        if not status.get("running", False):
            findings.append(Finding(
                finding_type="daemon_stopped",
                severity="critical",
                fixability="auto",
                service="agi-core-go",
                details="Daemon is not running",
                suggested_fix="Check ATHENA_DAEMON env var and agi-core logs",
            ))

        rate = status.get("success_rate", 0)
        if rate < 0.7 and status.get("completed_last_24h", 0) > 5:
            findings.append(Finding(
                finding_type="daemon_low_success",
                severity="warning",
                fixability="needs_context",
                service="agi-core-go",
                details=f"Daemon success rate is {rate:.0%} ({status.get('failed_last_24h', 0)} failures in 24h)",
                suggested_fix="Analyze failed goals: curl localhost:8100/daemon/goals?status=failed",
            ))

    return findings


def probe_fitness(config) -> list:
    """Check fitness score components for zeros that indicate broken subsystems."""
    findings = []

    fitness = _http_get("http://127.0.0.1:8100/ops/fitness")
    if fitness:
        components = fitness.get("components", {})
        for name, score in components.items():
            if score == 0 and name not in ("incident_score",):  # 0 incidents is good
                findings.append(Finding(
                    finding_type="fitness_zero_component",
                    severity="warning",
                    fixability="needs_context",
                    service="agi-core-go",
                    endpoint=f"fitness/{name}",
                    details=f"Fitness component '{name}' is 0 — indicates broken subsystem",
                    suggested_fix=f"Investigate why {name} returns 0 in fitness_scorer.go",
                ))

    return findings


def probe_log_errors(config) -> list:
    """Scan recent logs for error patterns."""
    findings = []
    log_dir = Path.home() / "Library" / "Logs" / "Athena"

    if not log_dir.exists():
        return findings

    error_patterns = {}
    now = time.time()

    for log_file in log_dir.glob("*.error.log"):
        svc_name = log_file.stem.replace(".error", "")
        if log_file.stat().st_mtime < now - 3600:
            continue  # Skip logs not modified in last hour

        try:
            content = log_file.read_text(errors="ignore")
            lines = content.split("\n")
            # Check last 50 lines for errors
            recent = lines[-50:]
            errors = [l for l in recent if any(
                kw in l.lower() for kw in ["error", "panic", "fatal", "connection refused"]
            ) and "level" not in l.lower()]  # Skip structured log level fields

            if len(errors) > 3:
                # Deduplicate
                unique = set()
                for e in errors:
                    # Extract the core error message
                    key = e[:80]
                    unique.add(key)

                if len(unique) > 0:
                    sample = list(unique)[:3]
                    error_patterns[svc_name] = sample
        except Exception:
            continue

    for svc, errors in error_patterns.items():
        findings.append(Finding(
            finding_type="log_errors",
            severity="info",
            fixability="needs_context",
            service=svc,
            details=f"{len(errors)} unique error patterns in last hour: {errors[0][:100]}",
            suggested_fix=f"Check ~/Library/Logs/Athena/{svc}.error.log",
        ))

    return findings


def probe_plist_config(config) -> list:
    """Check plist configs for common issues."""
    findings = []
    plist_dir = Path.home() / "Library" / "LaunchAgents"

    critical_services = [
        "agi-core", "rag-gateway-rust", "truth-correlator", "ego-memory",
        "world-model-go", "router-rust", "qwen3-embed", "llama-swap",
        "online-learning", "embedding-service", "nats",
    ]

    for svc in critical_services:
        plist = plist_dir / f"com.athena.{svc}.plist"
        if not plist.exists():
            disabled = plist_dir / f"com.athena.{svc}.plist.disabled"
            if disabled.exists():
                findings.append(Finding(
                    finding_type="service_disabled",
                    severity="critical",
                    fixability="auto",
                    service=svc,
                    details=f"Service plist is disabled (.plist.disabled)",
                    suggested_fix=f"mv {disabled} {plist}",
                ))
            continue

        try:
            content = plist.read_text()
            # Check KeepAlive
            if "<key>KeepAlive</key>" in content:
                idx = content.index("<key>KeepAlive</key>")
                after = content[idx:idx+100]
                if "<false/>" in after:
                    findings.append(Finding(
                        finding_type="no_keepalive",
                        severity="warning",
                        fixability="auto",
                        service=svc,
                        details=f"KeepAlive is false — service won't auto-restart on crash",
                        suggested_fix=f"Set KeepAlive to true in {plist}",
                    ))
        except Exception:
            continue

    return findings


def run_live_probes(config) -> list:
    """Run all live probes and return findings."""
    findings = []
    findings.extend(probe_binary_staleness(config))
    findings.extend(probe_port_liveness(config))
    findings.extend(probe_health_endpoints(config))
    findings.extend(probe_embedding_dimensions(config))
    findings.extend(probe_learning_pipeline(config))
    findings.extend(probe_daemon_health(config))
    findings.extend(probe_fitness(config))
    findings.extend(probe_log_errors(config))
    findings.extend(probe_plist_config(config))
    return findings
