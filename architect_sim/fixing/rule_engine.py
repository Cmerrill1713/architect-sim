"""Rule-based decision engine for findings that need engineering judgment.

Pure deterministic IF/THEN logic. No LLM, no probabilities. Every decision
traces to a concrete rule with a verifiable condition.

Rules are ordered by specificity -- most specific match wins.
Every rule has:
  - IF: condition (inspectable, testable)
  - THEN: action
  - BECAUSE: engineering rationale
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from ..models import Finding, Endpoint, Call


@dataclass
class Decision:
    """A rule-based engineering decision."""
    finding: Finding
    action: str          # "register_port", "add_endpoint", "remove_call", "keep", "skip"
    reasoning: str       # Which rule fired and why
    confidence: float    # Always 1.0 for rule matches, 0.0 for no match
    rule_id: str = ""    # Which rule produced this decision
    mutations: dict = field(default_factory=dict)


# ============================================================
# RULE DEFINITIONS
# ============================================================


def _rule_port_zero(f, blueprints, config):
    """IF port is 0 THEN it's a failed resolution -- remove the call.
    BECAUSE port 0 means the discovery/ports system couldn't resolve the service name."""
    if f.finding_type == "unknown_service":
        port = _extract_port(f)
        if port == 0:
            return Decision(
                finding=f,
                action="remove_call",
                reasoning="Port 0 = failed service resolution. The caller has a typo or uses a service name not in port config.",
                confidence=1.0,
                rule_id="PORT_ZERO",
            )
    return None


def _rule_self_reference(f, blueprints, config):
    """IF a service has a circular dep with itself THEN it's a false positive.
    BECAUSE services commonly reference their own localhost URL in config."""
    if f.finding_type == "circular_dep":
        if "->" in f.details:
            parts = [p.strip() for p in f.details.split("->")]
            unique = set(parts)
            if len(unique) == 1:
                return Decision(
                    finding=f,
                    action="skip",
                    reasoning=f"Self-reference in {parts[0]} -- service references its own port. Not a real cycle.",
                    confidence=1.0,
                    rule_id="SELF_REFERENCE",
                )
    return None


def _rule_known_service_port(f, blueprints, config):
    """IF port is referenced in code AND has a matching service directory THEN register it.
    BECAUSE the service exists -- it just isn't in port config yet."""
    if f.finding_type != "unknown_service":
        return None

    port = _extract_port(f)
    if not port or port == 0:
        return None

    # Build port-to-directory map dynamically from service directories
    port_to_dir = _build_port_directory_map(config)

    if port in port_to_dir:
        svc_name = port_to_dir[port]
        return Decision(
            finding=f,
            action="register_port",
            reasoning=f"Port {port} has service directory '{svc_name}' -- service exists but isn't registered.",
            confidence=1.0,
            rule_id="KNOWN_SERVICE_DIR",
            mutations={"service_name": svc_name, "port": port},
        )
    return None


def _rule_port_no_directory(f, blueprints, config):
    """IF port maps to no known service AND no service directory exists THEN remove the call.
    BECAUSE a port with no service directory is dead code."""
    if f.finding_type != "unknown_service":
        return None

    port = _extract_port(f)
    if not port or port == 0:
        return None

    port_to_dir = _build_port_directory_map(config)
    if port in port_to_dir:
        return None

    return Decision(
        finding=f,
        action="remove_call",
        reasoning=f"Port {port} has no service directory and no port config entry. Dead code.",
        confidence=1.0,
        rule_id="NO_SERVICE_DIR",
    )


def _rule_doc_drift(f, blueprints, config):
    """IF documentation references something that doesn't exist in code THEN flag it.
    BECAUSE outdated docs cause confusion."""
    if f.finding_type != "doc_drift":
        return None

    return Decision(
        finding=f,
        action="skip",
        reasoning=f"Documentation drift detected -- doc references something not in current code.",
        confidence=1.0,
        rule_id="DOC_DRIFT",
    )


def _rule_unresolved_path(f, blueprints, config):
    """IF a phantom call has path '/' THEN it's an extraction artifact.
    BECAUSE the extractor found a service URL but couldn't extract the endpoint path."""
    if f.finding_type != "phantom_call":
        return None

    path = _extract_path(f)
    if path == "/" or (path.endswith("/") and path.count("/") == 1):
        return Decision(
            finding=f,
            action="skip",
            reasoning="Path '/' means the extractor captured the base URL but not the endpoint path. Not a real phantom call.",
            confidence=1.0,
            rule_id="UNRESOLVED_PATH",
        )
    return None


def _rule_endpoint_exists_different_path(f, blueprints, config):
    """IF a phantom call's path is close to an existing endpoint THEN it's a path typo.
    BECAUSE services frequently drift on path naming (/chat vs /v1/chat)."""
    if f.finding_type != "phantom_call":
        return None

    path = _extract_path(f)
    method = _extract_method(f)
    callee = _extract_callee(f)

    if callee not in blueprints:
        return None

    bp = blueprints[callee]
    last_segment = path.rstrip("/").split("/")[-1] if "/" in path else path
    if not last_segment or last_segment == "/":
        return None

    for ep in bp.endpoints:
        if ep.method == method and last_segment in ep.path and ep.path != path:
            return Decision(
                finding=f,
                action="skip",
                reasoning=f"Similar endpoint exists: {ep.method} {ep.path} (vs called {path}). Likely path format difference.",
                confidence=1.0,
                rule_id="SIMILAR_ENDPOINT",
                mutations={"actual_path": ep.path, "called_path": path},
            )
    return None


def _rule_callee_has_service_dir(f, blueprints, config):
    """IF phantom call targets a service that exists AND path looks real THEN add stub.
    BECAUSE the caller clearly expects this endpoint."""
    if f.finding_type != "phantom_call":
        return None

    path = _extract_path(f)
    method = _extract_method(f)
    callee = _extract_callee(f)

    if path == "/" or "%" in path:
        return None

    if callee in blueprints:
        return Decision(
            finding=f,
            action="add_endpoint",
            reasoning=f"Service {callee} exists and caller expects {method} {path}. Add stub to prevent runtime 404.",
            confidence=1.0,
            rule_id="ADD_STUB",
            mutations={"target_service": callee, "method": method, "path": path},
        )
    return None


def _rule_callee_no_blueprint(f, blueprints, config):
    """IF phantom call targets a service with no blueprint THEN remove the call.
    BECAUSE if the service has no extractable code, the call will fail at runtime."""
    if f.finding_type != "phantom_call":
        return None

    callee = _extract_callee(f)
    if callee not in blueprints:
        return Decision(
            finding=f,
            action="remove_call",
            reasoning=f"Service {callee} has no extractable code (no blueprint). Call will fail at runtime.",
            confidence=1.0,
            rule_id="NO_BLUEPRINT",
        )
    return None


def _rule_circular_dep_health_check(f, blueprints, config):
    """IF circular dep involves a /health call THEN it's not a real cycle.
    BECAUSE health checks are passive reads, not functional dependencies."""
    if f.finding_type != "circular_dep":
        return None

    if "->" in f.details:
        parts = [p.strip() for p in f.details.split("->")]
        for i in range(len(parts) - 1):
            caller = parts[i]
            callee = parts[i + 1]
            if caller in blueprints:
                bp = blueprints[caller]
                for call in bp.outbound_calls:
                    if (call.callee_service == callee and
                            call.path in ("/health", "/live", "/ready", "/metrics")):
                        return Decision(
                            finding=f,
                            action="skip",
                            reasoning=f"Cycle includes health check ({caller} -> {callee} via {call.path}). Not a functional dep.",
                            confidence=1.0,
                            rule_id="HEALTH_CHECK_CYCLE",
                        )
    return None


def _rule_circular_dep_real(f, blueprints, config):
    """IF circular dep has 2+ services and isn't a health check THEN flag it.
    BECAUSE real circular dependencies cause startup deadlocks."""
    if f.finding_type != "circular_dep":
        return None

    if "->" in f.details:
        parts = [p.strip() for p in f.details.split("->")]
        unique = set(parts)
        if len(unique) >= 2:
            return Decision(
                finding=f,
                action="skip",
                reasoning=f"Real circular dependency between {len(unique)} services. Needs architectural review.",
                confidence=1.0,
                rule_id="REAL_CYCLE",
            )
    return None


def _rule_sprintf_path(f, blueprints, config):
    """IF a phantom call path contains format specifiers (%s, %d) THEN add a parameterized stub.
    BECAUSE Go code uses fmt.Sprintf to build URLs -- these are real endpoints with path params."""
    if f.finding_type != "phantom_call":
        return None

    path = _extract_path(f)
    method = _extract_method(f)
    callee = _extract_callee(f)

    if "%" not in path:
        return None

    if callee not in blueprints:
        return None

    if path.startswith(callee + "/"):
        path = path[len(callee):]
    elif not path.startswith("/"):
        path = "/" + path

    clean_path = re.sub(r'\?.*$', '', path)
    clean_path = re.sub(r'%[sd]', ':param', clean_path)

    return Decision(
        finding=f,
        action="add_endpoint",
        reasoning=f"Path uses fmt.Sprintf params. Real parameterized endpoint on {callee}.",
        confidence=1.0,
        rule_id="SPRINTF_PATH",
        mutations={"target_service": callee, "method": method, "path": clean_path},
    )


def _rule_service_name_in_path(f, blueprints, config):
    """IF phantom call path is just the service name + '/' THEN it's an extraction artifact."""
    if f.finding_type != "phantom_call":
        return None

    path = _extract_path(f)
    callee = _extract_callee(f)

    if path == "/" or path == callee + "/" or path.rstrip("/") == "":
        return Decision(
            finding=f,
            action="skip",
            reasoning=f"Path '{path}' is the base URL of {callee}, not a specific endpoint. Extraction artifact.",
            confidence=1.0,
            rule_id="SERVICE_BASE_URL",
        )
    return None


def _rule_phantom_real_endpoint(f, blueprints, config):
    """IF phantom call has a real-looking path AND callee exists THEN create a stub endpoint."""
    if f.finding_type != "phantom_call":
        return None

    path = _extract_path(f)
    method = _extract_method(f)
    callee = _extract_callee(f)

    if path.startswith(callee + "/"):
        path = path[len(callee):]
    elif not path.startswith("/"):
        path = "/" + path

    if path == "/" or path == "":
        return None

    if callee in blueprints:
        return Decision(
            finding=f,
            action="add_endpoint",
            reasoning=f"Caller expects {method} {path} on {callee}. Service exists -- add stub.",
            confidence=1.0,
            rule_id="ADD_REAL_ENDPOINT",
            mutations={"target_service": callee, "method": method, "path": path},
        )
    return None


# Rules in priority order -- first match wins
RULES = [
    _rule_port_zero,
    _rule_self_reference,
    _rule_service_name_in_path,
    _rule_unresolved_path,
    _rule_known_service_port,
    _rule_port_no_directory,
    _rule_endpoint_exists_different_path,
    _rule_sprintf_path,
    _rule_callee_has_service_dir,
    _rule_phantom_real_endpoint,
    _rule_callee_no_blueprint,
    _rule_circular_dep_health_check,
    _rule_circular_dep_real,
    _rule_doc_drift,
]


def make_decision(finding: Finding, blueprints: dict, config) -> Decision:
    """Run the rule engine on a finding. First matching rule wins."""
    for rule in RULES:
        decision = rule(finding, blueprints, config)
        if decision is not None:
            return decision

    return Decision(
        finding=finding,
        action="skip",
        reasoning="No rule matched this finding.",
        confidence=0.0,
        rule_id="NO_MATCH",
    )


def make_decisions_batch(findings: list, blueprints: dict, config,
                         max_concurrent: int = 1) -> list:
    """Make decisions for a batch of findings."""
    return [make_decision(f, blueprints, config) for f in findings]


def apply_decision(blueprints: dict, decision: Decision, config) -> dict:
    """Apply a decision's action to the in-memory blueprint."""
    f = decision.finding

    if decision.action == "register_port":
        port = decision.mutations.get("port", _extract_port(f))
        if port:
            service_name = decision.mutations.get("service_name", f"service-{port}")
            config.ports[service_name] = port
            config.port_to_service[port] = service_name
            return {"type": "register_port", "port": port, "service": service_name}

    elif decision.action == "add_endpoint":
        target = decision.mutations.get("target_service", "")
        method = decision.mutations.get("method", "POST")
        path = decision.mutations.get("path", "/")
        if target in blueprints:
            bp = blueprints[target]
            already_exists = any(
                ep.path == path and ep.method == method for ep in bp.endpoints
            )
            if already_exists:
                return {"type": "already_fixed", "service": target, "method": method, "path": path}
            stub = Endpoint(
                service_name=target,
                port=bp.port,
                method=method,
                path=path,
                file_path="[rule-generated stub]",
                line_number=0,
                handler_name=f"stub_{path.replace('/', '_').strip('_')}",
                framework="stub",
            )
            bp.endpoints.append(stub)
            return {"type": "add_endpoint", "service": target, "method": method, "path": path}

    elif decision.action == "remove_call":
        caller = f.service
        if caller in blueprints:
            bp = blueprints[caller]
            path = _extract_path(f)
            before = len(bp.outbound_calls)
            bp.outbound_calls = [
                c for c in bp.outbound_calls
                if not (c.file_path == f.file_path and c.line_number == f.line_number)
            ]
            if len(bp.outbound_calls) == before:
                bp.outbound_calls = [
                    c for c in bp.outbound_calls
                    if not (c.path == path and c.caller_service == caller)
                ]
            removed = before - len(bp.outbound_calls)
            return {"type": "remove_call", "service": caller, "path": path, "removed": removed}

    return {"type": "noop"}


def undo_decision(blueprints: dict, decision: Decision, mutation: dict, config):
    """Undo a decision's mutations."""
    if mutation.get("type") == "register_port":
        port = mutation["port"]
        service = mutation["service"]
        config.ports.pop(service, None)
        config.port_to_service.pop(port, None)

    elif mutation.get("type") == "add_endpoint":
        target = mutation["service"]
        method = mutation["method"]
        path = mutation["path"]
        if target in blueprints:
            bp = blueprints[target]
            bp.endpoints = [
                ep for ep in bp.endpoints
                if not (ep.path == path and ep.method == method and ep.framework == "stub")
            ]


# ============================================================
# HELPERS
# ============================================================

def _extract_port(finding: Finding) -> int:
    match = re.search(r'port (\d+)', finding.details)
    return int(match.group(1)) if match else 0


def _extract_path(finding: Finding) -> str:
    parts = finding.endpoint.split(" ", 1)
    return parts[1] if len(parts) == 2 else "/"


def _extract_method(finding: Finding) -> str:
    parts = finding.endpoint.split(" ", 1)
    return parts[0] if parts else "GET"


def _extract_callee(finding: Finding) -> str:
    match = re.search(r'calls (?:GET|POST|PUT|DELETE|PATCH) (\S+?)/', finding.details)
    if match:
        return match.group(1)
    return finding.service


def _build_port_directory_map(config) -> dict:
    """Build mapping of ports to service directory names dynamically.

    Scans service directories for source files that reference ports
    (e.g., in main.go, server.py, etc.) and maps those ports to the
    directory name.
    """
    result = {}

    # First, include everything already in config.ports (reverse map)
    for name, port in config.ports.items():
        result[port] = name

    # Then scan service directories for port references in source
    if not config.services_dir or not config.services_dir.exists():
        return result

    port_in_source_re = re.compile(r'(?:listen|addr|port|PORT|:)[\s"=:]*(\d{4,5})')

    for d in config.services_dir.iterdir():
        if not d.is_dir() or d.name.startswith(".") or d.name.startswith("_"):
            continue

        # Check main source files for port references
        main_files = []
        for pattern in ["main.go", "server.py", "app.py", "main.rs", "server.go", "cmd/main.go"]:
            candidates = list(d.glob(pattern)) + list(d.glob(f"**/{pattern}"))
            main_files.extend(candidates[:1])

        for src_file in main_files:
            try:
                content = src_file.read_text(errors="replace")
            except (OSError, IOError):
                continue

            for match in port_in_source_re.finditer(content):
                port = int(match.group(1))
                if 1024 < port < 65535 and port not in result:
                    result[port] = d.name

    return result
