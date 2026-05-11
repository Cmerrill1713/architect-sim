"""Runtime log ingestion and correlation with static analysis findings.

Parses runtime error logs in GIN, structured JSON, bracketed plain-text,
and Python/stderr formats.  Correlates extracted events with static
analysis findings so phantom calls, unknown services, and missing
resilience show up with real failure counts.
"""

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RuntimeEvent:
    timestamp: str           # ISO-ish string as found in the log
    level: str               # INFO, WARN, ERROR
    source_file: str         # Log file path
    line_number: int         # Line in log file
    event_type: str          # http_error, connection_refused, timeout, llm_error, health_fail
    method: str = ""         # HTTP method if applicable
    path: str = ""           # URL path if applicable
    status_code: int = 0     # HTTP status if applicable
    port: int = 0            # Target port if extractable
    service: str = ""        # Target service if resolvable
    duration_ms: float = 0.0 # Request duration if available
    message: str = ""        # Raw log line


@dataclass
class RuntimeCorrelation:
    finding_type: str        # From static analysis (e.g. phantom_call)
    finding_details: str
    runtime_events: int      # How many runtime events match
    sample_events: list = field(default_factory=list)  # Up to 3 examples
    severity_boost: str = "" # "confirmed", "likely", "unobserved"


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

_TS_FORMATS = [
    "%Y/%m/%d - %H:%M:%S",   # GIN
    "%Y/%m/%d %H:%M:%S",     # bracketed plain-text
    "%Y-%m-%d %H:%M:%S",     # Python stderr
    "%Y-%m-%dT%H:%M:%S",     # structured JSON (no fractional)
]


def _parse_ts(raw: str) -> Optional[datetime]:
    """Best-effort parse of a timestamp string."""
    raw = raw.strip()
    # Strip fractional seconds / timezone suffix for matching
    clean = re.sub(r'[.,]\d+.*$', '', raw)
    clean = re.sub(r'Z$', '', clean)
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            continue
    return None


def _within_window(ts_raw: str, cutoff: Optional[datetime]) -> bool:
    if cutoff is None:
        return True
    dt = _parse_ts(ts_raw)
    if dt is None:
        return True  # keep events with unparseable timestamps
    return dt >= cutoff


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(
    r'(?P<val>[\d.]+)\s*(?P<unit>ns|µs|us|ms|s|m(?:in)?|h)'
)


def _duration_to_ms(raw: str) -> float:
    """Convert a human-readable duration to milliseconds."""
    raw = raw.strip()
    m = _DURATION_RE.search(raw)
    if not m:
        return 0.0
    val = float(m.group("val"))
    unit = m.group("unit")
    if unit == "ns":
        return val / 1_000_000
    if unit in ("µs", "us"):
        return val / 1_000
    if unit == "ms":
        return val
    if unit == "s":
        return val * 1_000
    if unit in ("m", "min"):
        return val * 60_000
    if unit == "h":
        return val * 3_600_000
    return 0.0


# ---------------------------------------------------------------------------
# Per-format parsers
# ---------------------------------------------------------------------------

# [GIN] 2026/05/10 - 23:35:39 | 404 | 173.25µs | 127.0.0.1 | GET "/daemon/stats"
_GIN_RE = re.compile(
    r'\[GIN\]\s+'
    r'(?P<ts>\d{4}/\d{2}/\d{2}\s*-\s*\d{2}:\d{2}:\d{2})\s*\|\s*'
    r'(?P<status>\d{3})\s*\|\s*'
    r'(?P<dur>\S+)\s*\|\s*'
    r'(?P<ip>\S+)\s*\|\s*'
    r'(?P<method>[A-Z]+)\s+'
    r'"(?P<path>[^"]*)"'
)


def _parse_gin(line: str) -> Optional[dict]:
    m = _GIN_RE.search(line)
    if not m:
        return None
    status = int(m.group("status"))
    duration = _duration_to_ms(m.group("dur"))
    ts = m.group("ts")
    method = m.group("method")
    path = m.group("path")

    event_type = _classify_http(status, duration, path)
    if event_type is None:
        return None

    level = "ERROR" if status >= 500 else "WARN" if status >= 400 else "INFO"
    return dict(
        timestamp=ts, level=level, event_type=event_type,
        method=method, path=path, status_code=status,
        duration_ms=duration,
    )


# Structured JSON: {"time":"...","level":"INFO","msg":"http_request","status":404,...}
def _parse_json_line(line: str) -> Optional[dict]:
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    # Must have at least a status or an error-class msg
    status = obj.get("status", obj.get("status_code", 0))
    msg = obj.get("msg", obj.get("message", ""))
    level = obj.get("level", "INFO").upper()
    ts = obj.get("time", obj.get("timestamp", obj.get("ts", "")))
    method = obj.get("method", "")
    path = obj.get("path", obj.get("url", ""))
    duration = obj.get("duration", obj.get("duration_ms", 0))
    if isinstance(duration, (int, float)):
        # Heuristic: if the value is huge it is probably nanoseconds
        if duration > 1_000_000:
            duration = duration / 1_000_000  # ns -> ms
    else:
        duration = _duration_to_ms(str(duration))

    # Classify
    event_type = None
    if isinstance(status, int) and status > 0:
        event_type = _classify_http(status, duration, path)
    if event_type is None and "connection" in msg.lower() and ("refused" in msg.lower() or "reset" in msg.lower()):
        event_type = "connection_refused"
    if event_type is None and ("timeout" in msg.lower() or "timed out" in msg.lower()):
        event_type = "timeout"
    if event_type is None:
        return None

    port = obj.get("port", 0)
    return dict(
        timestamp=str(ts), level=level, event_type=event_type,
        method=method, path=path, status_code=int(status) if status else 0,
        duration_ms=float(duration), port=int(port) if port else 0,
    )


# Plain-text with brackets:
# 2026/05/10 23:25:06 [DAEMON] /slots returned 404 (MLX?), assuming LLM available
_BRACKET_RE = re.compile(
    r'(?P<ts>\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})\s+'
    r'\[(?P<tag>[A-Z_]+)\]\s+'
    r'(?P<body>.*)'
)

_HTTP_STATUS_IN_TEXT = re.compile(r'\b(?:returned|status|code)\s+(?P<status>\d{3})\b', re.IGNORECASE)
_PORT_IN_TEXT = re.compile(r'(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(?P<port>\d{2,5})')
_PATH_IN_TEXT = re.compile(r'(?:GET|POST|PUT|DELETE|PATCH)\s+(?P<path>/\S+)')
_PATH_LEADING = re.compile(r'^(/\S+)\s')


def _parse_bracket(line: str) -> Optional[dict]:
    m = _BRACKET_RE.search(line)
    if not m:
        return None

    ts = m.group("ts")
    body = m.group("body")
    lower = body.lower()

    event_type = None
    status = 0
    port = 0
    path = ""
    method = ""
    level = "WARN"

    # Extract status code from body
    sm = _HTTP_STATUS_IN_TEXT.search(body)
    if sm:
        status = int(sm.group("status"))

    # Extract port
    pm = _PORT_IN_TEXT.search(body)
    if pm:
        port = int(pm.group("port"))

    # Extract path
    pa = _PATH_IN_TEXT.search(body)
    if pa:
        path = pa.group("path")
        method = body[:pa.start()].strip().split()[-1] if pa.start() > 0 else ""
    else:
        pa2 = _PATH_LEADING.search(body)
        if pa2:
            path = pa2.group(1)

    # Classify
    if status >= 400:
        event_type = _classify_http(status, 0, path)
    if event_type is None and ("connection refused" in lower or "connection reset" in lower):
        event_type = "connection_refused"
        level = "ERROR"
    if event_type is None and ("timeout" in lower or "timed out" in lower or "deadline exceeded" in lower):
        event_type = "timeout"
        level = "ERROR"
    if event_type is None and ("health" in lower and ("fail" in lower or "unhealthy" in lower or "down" in lower)):
        event_type = "health_fail"
        level = "ERROR"
    if event_type is None and status >= 400:
        event_type = "http_error"
    if event_type is None:
        return None

    if status >= 500:
        level = "ERROR"

    return dict(
        timestamp=ts, level=level, event_type=event_type,
        method=method, path=path, status_code=status,
        port=port,
    )


# Python/stderr:
# ERROR 2026-05-10 Connection refused: localhost:8085
# requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=8085)
_PYTHON_ERROR_RE = re.compile(
    r'(?:ERROR|CRITICAL|WARNING)\s+'
    r'(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\s+'
    r'(?P<body>.*)',
    re.IGNORECASE,
)
_CONNPOOL_RE = re.compile(
    r"HTTPConnectionPool\(host='(?P<host>[^']+)',\s*port=(?P<port>\d+)\)"
)
_CONN_REFUSED_RE = re.compile(
    r'[Cc]onnection\s+refused[:\s]+(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(?P<port>\d+)'
)


def _parse_python(line: str) -> Optional[dict]:
    # Try explicit ERROR/CRITICAL line
    m = _PYTHON_ERROR_RE.search(line)
    if m:
        ts = m.group("ts")
        body = m.group("body")
    else:
        ts = ""
        body = line

    lower = body.lower()
    event_type = None
    port = 0
    path = ""

    # Connection pool errors
    cp = _CONNPOOL_RE.search(line)
    if cp:
        port = int(cp.group("port"))
        event_type = "connection_refused"

    cr = _CONN_REFUSED_RE.search(line)
    if cr:
        port = int(cr.group("port"))
        event_type = "connection_refused"

    if event_type is None and ("timeout" in lower or "timed out" in lower or "read timed out" in lower):
        event_type = "timeout"
        pm = _PORT_IN_TEXT.search(line)
        if pm:
            port = int(pm.group("port"))

    if event_type is None and ("connectionerror" in lower or "connection refused" in lower or "connection reset" in lower):
        event_type = "connection_refused"
        pm = _PORT_IN_TEXT.search(line)
        if pm:
            port = int(pm.group("port"))

    if event_type is None:
        return None

    level_match = re.match(r'(ERROR|CRITICAL|WARNING)', line, re.IGNORECASE)
    level = level_match.group(1).upper() if level_match else "ERROR"
    if level == "CRITICAL":
        level = "ERROR"
    if level == "WARNING":
        level = "WARN"

    return dict(
        timestamp=ts, level=level, event_type=event_type,
        port=port, path=path,
    )


# ---------------------------------------------------------------------------
# HTTP classification helper
# ---------------------------------------------------------------------------

_LLM_PATHS = re.compile(
    r'/v1/(?:chat/completions|completions|embeddings)|'
    r'/slots|/tokenize|/completion|/chat'
)

_HEALTH_PATHS = re.compile(
    r'/health|/healthz|/readyz|/livez|/ops/snapshot|/status'
)


def _classify_http(status: int, duration_ms: float, path: str) -> Optional[str]:
    """Classify an HTTP event by status/path into an event_type, or None to skip."""
    if status == 0:
        return None

    # Timeouts (duration > 30s and non-200, or 504/408)
    if status in (504, 408) or (duration_ms > 30_000 and status != 200):
        if _LLM_PATHS.search(path):
            return "llm_error"
        return "timeout"

    if status == 503:
        if _LLM_PATHS.search(path):
            return "llm_error"
        if _HEALTH_PATHS.search(path):
            return "health_fail"
        return "http_error"

    if status == 404:
        if _HEALTH_PATHS.search(path):
            return "health_fail"
        return "http_error"

    if status >= 500:
        if _LLM_PATHS.search(path):
            return "llm_error"
        return "http_error"

    if status >= 400:
        return "http_error"

    # 2xx/3xx with very long duration on LLM paths is still notable
    if duration_ms > 60_000 and _LLM_PATHS.search(path):
        return "llm_error"

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_logs(log_paths: list, since_hours: int = 24) -> list[RuntimeEvent]:
    """Parse log files and extract runtime events.

    Args:
        log_paths: Paths to log files.
        since_hours: Only include events from the last N hours.
                     Set to 0 for no time filter.

    Returns:
        List of RuntimeEvent instances, sorted by source file and line.
    """
    cutoff = None
    if since_hours > 0:
        cutoff = datetime.now() - timedelta(hours=since_hours)

    events: list[RuntimeEvent] = []
    parsers = [_parse_gin, _parse_json_line, _parse_bracket, _parse_python]

    for log_path in log_paths:
        path = Path(log_path)
        if not path.is_file():
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue

        for line_no, raw_line in enumerate(lines, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            for parser in parsers:
                result = parser(stripped)
                if result is not None:
                    ts = result.get("timestamp", "")
                    if not _within_window(ts, cutoff):
                        break

                    # Extract port from path if not already set
                    port = result.get("port", 0)
                    if port == 0:
                        pm = _PORT_IN_TEXT.search(stripped)
                        if pm:
                            port = int(pm.group("port"))

                    events.append(RuntimeEvent(
                        timestamp=ts,
                        level=result.get("level", "WARN"),
                        source_file=str(path),
                        line_number=line_no,
                        event_type=result["event_type"],
                        method=result.get("method", ""),
                        path=result.get("path", ""),
                        status_code=result.get("status_code", 0),
                        port=port,
                        service="",
                        duration_ms=result.get("duration_ms", 0.0),
                        message=stripped,
                    ))
                    break  # first matching parser wins

    return events


def correlate_with_findings(
    events: list[RuntimeEvent],
    findings: list,
    config=None,
) -> list[RuntimeCorrelation]:
    """Match runtime events to static analysis findings.

    Correlation logic per finding type:
        phantom_call   -> HTTP 404 on the same path, or connection_refused on same port
        orphan_endpoint -> (no runtime signal expected; skip)
        missing_endpoint -> HTTP 404 on the endpoint path
        unknown_service / undeclared_dep -> connection_refused on the port
        timeout_issue  -> timeout events on the port
        port_conflict  -> any errors on the conflicting port
        field_mismatch -> 500 errors on the endpoint (server-side parse failures)

    Args:
        events:   Output of ingest_logs().
        findings: List of Finding dataclass instances (from models.py).
        config:   Optional Config instance for port->service resolution.

    Returns:
        List of RuntimeCorrelation, one per finding that could be correlated.
    """
    # Build indexes for fast lookup
    by_path: dict[str, list[RuntimeEvent]] = {}
    by_port: dict[int, list[RuntimeEvent]] = {}
    by_type: dict[str, list[RuntimeEvent]] = {}

    for ev in events:
        # Normalize path: strip trailing slash, ignore query string
        norm_path = ev.path.rstrip("/").split("?")[0] if ev.path else ""
        if norm_path:
            by_path.setdefault(norm_path, []).append(ev)
        if ev.port:
            by_port.setdefault(ev.port, []).append(ev)
        by_type.setdefault(ev.event_type, []).append(ev)

    # Resolve service names on events if config available
    if config is not None:
        for ev in events:
            if ev.port and not ev.service:
                ev.service = config.resolve_port(ev.port)

    correlations: list[RuntimeCorrelation] = []

    for finding in findings:
        matched: list[RuntimeEvent] = []

        ftype = finding.finding_type
        fpath = getattr(finding, "endpoint", "").rstrip("/").split("?")[0]
        fport = _extract_port_from_finding(finding)
        fservice = getattr(finding, "service", "")

        if ftype == "phantom_call":
            # A call to an endpoint that doesn't exist -> expect 404s on that path
            if fpath:
                matched.extend(
                    ev for ev in by_path.get(fpath, [])
                    if ev.status_code == 404
                )
            # Also check connection_refused on the port
            if fport:
                matched.extend(
                    ev for ev in by_port.get(fport, [])
                    if ev.event_type == "connection_refused"
                )

        elif ftype == "missing_endpoint":
            # Static analysis says the endpoint is needed but not defined
            if fpath:
                matched.extend(
                    ev for ev in by_path.get(fpath, [])
                    if ev.status_code in (404, 405)
                )

        elif ftype in ("unknown_service", "undeclared_dep"):
            # Unknown service on a port -> connection_refused or repeated errors
            if fport:
                matched.extend(
                    ev for ev in by_port.get(fport, [])
                    if ev.event_type in ("connection_refused", "http_error", "timeout")
                )

        elif ftype == "timeout_issue":
            # Missing retry/timeout config -> runtime timeouts on that port
            if fport:
                matched.extend(
                    ev for ev in by_port.get(fport, [])
                    if ev.event_type == "timeout"
                )
            elif fservice:
                matched.extend(
                    ev for ev in events
                    if ev.event_type == "timeout" and ev.service == fservice
                )

        elif ftype == "port_conflict":
            if fport:
                matched.extend(
                    ev for ev in by_port.get(fport, [])
                    if ev.event_type in ("connection_refused", "http_error")
                )

        elif ftype == "field_mismatch":
            # Schema mismatch -> 500s on the endpoint (deserialization failures)
            if fpath:
                matched.extend(
                    ev for ev in by_path.get(fpath, [])
                    if ev.status_code >= 500
                )
            elif fport:
                matched.extend(
                    ev for ev in by_port.get(fport, [])
                    if ev.status_code >= 500
                )

        elif ftype == "circular_dep":
            # Circular dependency -> timeouts (deadlock) on involved service
            if fport:
                matched.extend(
                    ev for ev in by_port.get(fport, [])
                    if ev.event_type == "timeout"
                )

        else:
            # Generic: look for any errors on the port or path
            if fport:
                matched.extend(
                    ev for ev in by_port.get(fport, [])
                    if ev.event_type in ("http_error", "connection_refused", "timeout")
                )
            if fpath:
                matched.extend(
                    ev for ev in by_path.get(fpath, [])
                    if ev.status_code >= 400
                )

        # Deduplicate (same event could match via both port and path)
        seen_ids: set[tuple[str, int]] = set()
        unique: list[RuntimeEvent] = []
        for ev in matched:
            key = (ev.source_file, ev.line_number)
            if key not in seen_ids:
                seen_ids.add(key)
                unique.append(ev)
        matched = unique

        # Determine severity boost
        count = len(matched)
        if count >= 5:
            boost = "confirmed"
        elif count >= 1:
            boost = "likely"
        else:
            boost = "unobserved"

        correlations.append(RuntimeCorrelation(
            finding_type=ftype,
            finding_details=getattr(finding, "details", ""),
            runtime_events=count,
            sample_events=matched[:3],
            severity_boost=boost,
        ))

    return correlations


def auto_discover_logs(root: str) -> list[str]:
    """Find log files in common locations.

    Searches:
        {root}/logs/
        {root}/**/logs/
        ~/Library/Logs/{project_name}/
        /var/log/{project_name}/
        /tmp/{project_name}*.log
    """
    root_path = Path(root).resolve()
    project_name = root_path.name
    found: list[str] = []

    _LOG_EXTENSIONS = {".log", ".txt", ".jsonl"}

    def _collect(directory: Path, recursive: bool = False):
        if not directory.is_dir():
            return
        pattern = "**/*" if recursive else "*"
        for p in directory.glob(pattern):
            if p.is_file() and (p.suffix in _LOG_EXTENSIONS or "log" in p.name.lower()):
                found.append(str(p))

    # {root}/logs/
    _collect(root_path / "logs")

    # {root}/**/logs/ (one level of nesting to avoid huge scans)
    for child in root_path.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            _collect(child / "logs")

    # ~/Library/Logs/{project_name}/
    lib_logs = Path.home() / "Library" / "Logs" / project_name
    _collect(lib_logs, recursive=True)

    # /var/log/{project_name}/
    var_log = Path("/var/log") / project_name
    _collect(var_log, recursive=True)

    # /tmp/{project_name}*.log
    tmp = Path("/tmp")
    if tmp.is_dir():
        for p in tmp.glob(f"{project_name}*.log"):
            if p.is_file():
                found.append(str(p))

    # Deduplicate, preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for f in found:
        real = os.path.realpath(f)
        if real not in seen:
            seen.add(real)
            unique.append(f)

    return sorted(unique)


def format_runtime_report(
    correlations: list[RuntimeCorrelation],
    events: list[RuntimeEvent],
) -> str:
    """Format runtime analysis as markdown.

    Shows:
        - Event type distribution
        - Top error endpoints
        - Correlations with static findings
        - Severity boost recommendations
    """
    lines: list[str] = []
    lines.append("# Runtime Log Analysis")
    lines.append("")

    # -- Summary --
    lines.append(f"**Total events:** {len(events)}")
    lines.append("")

    # -- Event type distribution --
    type_counts = Counter(ev.event_type for ev in events)
    if type_counts:
        lines.append("## Event Type Distribution")
        lines.append("")
        lines.append("| Type | Count |")
        lines.append("|------|-------|")
        for etype, count in type_counts.most_common():
            lines.append(f"| {etype} | {count} |")
        lines.append("")

    # -- Top error endpoints --
    path_counts = Counter(
        (ev.method or "?", ev.path, ev.status_code)
        for ev in events
        if ev.path and ev.status_code >= 400
    )
    if path_counts:
        lines.append("## Top Error Endpoints")
        lines.append("")
        lines.append("| Method | Path | Status | Count |")
        lines.append("|--------|------|--------|-------|")
        for (method, path, status), count in path_counts.most_common(15):
            lines.append(f"| {method} | `{path}` | {status} | {count} |")
        lines.append("")

    # -- Top error ports (connection failures) --
    port_counts = Counter(
        ev.port for ev in events
        if ev.port and ev.event_type in ("connection_refused", "timeout")
    )
    if port_counts:
        lines.append("## Top Failing Ports")
        lines.append("")
        lines.append("| Port | Failures |")
        lines.append("|------|----------|")
        for port, count in port_counts.most_common(10):
            svc = next((ev.service for ev in events if ev.port == port and ev.service), f"unknown:{port}")
            lines.append(f"| {port} ({svc}) | {count} |")
        lines.append("")

    # -- Correlations --
    confirmed = [c for c in correlations if c.severity_boost == "confirmed"]
    likely = [c for c in correlations if c.severity_boost == "likely"]
    unobserved = [c for c in correlations if c.severity_boost == "unobserved"]

    if confirmed or likely:
        lines.append("## Static Finding Correlations")
        lines.append("")
        lines.append(
            "Findings from static analysis matched against runtime logs. "
            "**confirmed** = 5+ matching events, **likely** = 1-4 events."
        )
        lines.append("")

        for label, group in [("Confirmed", confirmed), ("Likely", likely)]:
            if not group:
                continue
            lines.append(f"### {label}")
            lines.append("")
            for c in sorted(group, key=lambda x: -x.runtime_events):
                lines.append(
                    f"- **{c.finding_type}**: {c.finding_details}  "
                    f"({c.runtime_events} runtime events)"
                )
                for sample in c.sample_events[:3]:
                    ts_part = f" [{sample.timestamp}]" if sample.timestamp else ""
                    lines.append(
                        f"  - `{sample.source_file}:{sample.line_number}`{ts_part} "
                        f"  {sample.message[:120]}"
                    )
            lines.append("")

    # -- Severity boost summary --
    if confirmed:
        lines.append("## Severity Boost Recommendations")
        lines.append("")
        lines.append(
            "The following static findings are **confirmed by runtime evidence** "
            "and should be treated as higher severity:"
        )
        lines.append("")
        for c in confirmed:
            lines.append(
                f"- `{c.finding_type}` — {c.finding_details} "
                f"(**{c.runtime_events} failures**)"
            )
        lines.append("")

    # -- Unobserved note --
    if unobserved:
        lines.append(f"**{len(unobserved)} findings** had no matching runtime events "
                      "(may be latent bugs or untested paths).")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_port_from_finding(finding) -> int:
    """Best-effort port extraction from a Finding's details or endpoint."""
    # Check if there's a port mentioned in the endpoint field
    endpoint = getattr(finding, "endpoint", "")
    details = getattr(finding, "details", "")

    for text in (endpoint, details):
        m = _PORT_IN_TEXT.search(text)
        if m:
            return int(m.group("port"))
        # Also look for bare "port NNNN" or "port=NNNN"
        m2 = re.search(r'\bport\s*[=:]\s*(\d{2,5})\b', text, re.IGNORECASE)
        if m2:
            return int(m2.group(1))

    return 0
