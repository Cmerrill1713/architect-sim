"""Extract outbound inter-service HTTP calls from Go source files."""
import re
from pathlib import Path
from ..models import Call

# Pattern 1: serviceURLs map access
SERVICE_URL_MAP_RE = re.compile(
    r'serviceURLs\["(\w+)"\]\s*\+\s*"([^"]*)"',
    re.MULTILINE
)

# Pattern 2: Discovery / ports resolution
DISCOVERY_RE = re.compile(
    r'(?:discovery\.\w+\.)?GetServiceURL(?:FromEnv)?\(\s*"([^"]+)"(?:\s*,\s*"([^"]+)")?\s*\)',
    re.MULTILINE
)
PORTS_GET_RE = re.compile(
    r'ports\.Get(?:Port|ServiceURL)\(\s*"([^"]+)"\s*\)',
    re.MULTILINE
)

# Pattern 3: Hardcoded localhost URLs
HARDCODED_URL_RE = re.compile(
    r'"(https?://(?:127\.0\.0\.1|localhost):(\d+)(/[^"]*)?)"',
    re.MULTILINE
)

# Pattern 4: Environment variable URL resolution
ENV_URL_RE = re.compile(
    r'os\.Getenv\(\s*"([A-Z_]+(?:URL|HOST|PORT|ADDR)[A-Z_]*)"\s*\)',
    re.MULTILINE
)

# HTTP method detection near URL usage
HTTP_POST_RE = re.compile(r'\.(?:Post|Do)\(', re.MULTILINE)
HTTP_GET_RE = re.compile(r'\.(?:Get)\(', re.MULTILINE)

# ServiceURLs map initialization
SERVICE_URL_INIT_RE = re.compile(
    r'svcURLs\["(\w+)"\]\s*=\s*(?:resolve|getURL|discovery\.\w+\.GetServiceURLFromEnv)\(\s*"([^"]+)"(?:\s*,\s*"([^"]+)")?\s*\)',
    re.MULTILINE
)

# Retry wrapper detection
RETRY_WRAPPER_RE = re.compile(
    r'retry\.(?:Do|WithConfig|DefaultConfig)\(', re.MULTILINE
)

# Resilience patterns: circuit breakers, resilient HTTP clients, retries
RESILIENCE_PATTERNS = [
    re.compile(r'FastHTTPClient|DefaultHTTPClient|SlowHTTPClient|MultimodalHTTPClient|QuickHTTPClient', re.MULTILINE),
    re.compile(r'http\.Client\s*\{[^}]*Timeout\s*:', re.MULTILINE | re.DOTALL),
    re.compile(r'circuitbreaker|circuit_breaker|CircuitBreaker', re.MULTILINE),
    re.compile(r'\bretry\b|\battempt\b', re.MULTILINE),
    re.compile(r'httpGetJSON|httpPostJSON|httpGetWithContext|httpclient\.', re.MULTILINE),
    re.compile(r'WithTimeout|context\.WithTimeout|context\.WithDeadline', re.MULTILINE),
]


def extract_go_calls(service_name: str, source_dir: str, config) -> list:
    """Extract all outbound inter-service HTTP calls from Go source files.

    Args:
        service_name: Name of the service being analyzed
        source_dir: Path to service source directory
        config: Config object for port/service resolution

    Returns:
        List of Call objects representing outbound calls
    """
    calls = []
    source_path = Path(source_dir)

    for go_file in sorted(source_path.rglob("*.go")):
        if go_file.name.endswith("_test.go") or "vendor" in str(go_file):
            continue

        try:
            content = go_file.read_text(errors="replace")
        except (OSError, IOError):
            continue

        file_str = str(go_file)

        # Build serviceURLs map if present
        url_map = _build_service_url_map(content, config)

        # Pattern 1: serviceURLs map access
        for match in SERVICE_URL_MAP_RE.finditer(content):
            key, path = match.groups()
            line_num = content[:match.start()].count("\n") + 1
            method = _detect_http_method(content, match.start())

            callee_service = url_map.get(key, {}).get("service", key)
            callee_port = url_map.get(key, {}).get("port", config.resolve_service(callee_service))
            has_retry = _has_nearby_retry(content, match.start())
            has_resil = _has_resilience(content, match.start())

            calls.append(Call(
                caller_service=service_name,
                callee_service=callee_service if callee_service != key else config.resolve_port(callee_port) if callee_port else key,
                callee_port=callee_port or 0,
                method=method,
                path=path,
                file_path=file_str,
                line_number=line_num,
                resolution_method="serviceURLs",
                retry_config={"detected": True, "resilient": True} if (has_retry or has_resil) else None,
            ))

        # Pattern 2: Discovery calls
        for match in DISCOVERY_RE.finditer(content):
            env_key, service_hint = match.groups()
            line_num = content[:match.start()].count("\n") + 1

            resolved_name = service_hint or _env_to_service(env_key)
            resolved_port = config.resolve_service(resolved_name)

            path = _find_path_after_resolve(content, match.end())
            method = _detect_http_method(content, match.start())

            calls.append(Call(
                caller_service=service_name,
                callee_service=config.resolve_port(resolved_port) if resolved_port else resolved_name,
                callee_port=resolved_port,
                method=method,
                path=path,
                file_path=file_str,
                line_number=line_num,
                resolution_method="discovery",
            ))

        # Pattern 2b: ports.GetPort / ports.GetServiceURL
        for match in PORTS_GET_RE.finditer(content):
            svc_name = match.group(1)
            line_num = content[:match.start()].count("\n") + 1
            resolved_port = config.resolve_service(svc_name)
            path = _find_path_after_resolve(content, match.end())
            method = _detect_http_method(content, match.start())

            calls.append(Call(
                caller_service=service_name,
                callee_service=config.resolve_port(resolved_port) if resolved_port else svc_name,
                callee_port=resolved_port,
                method=method,
                path=path,
                file_path=file_str,
                line_number=line_num,
                resolution_method="ports",
            ))

        # Pattern 3: Hardcoded URLs
        for match in HARDCODED_URL_RE.finditer(content):
            full_url, port_str, path = match.groups()
            port = int(port_str)
            line_num = content[:match.start()].count("\n") + 1
            method = _detect_http_method(content, match.start())

            # Skip self-references (service calling its own port)
            own_port = config.resolve_service(service_name)
            if port == own_port:
                continue

            has_resil = _has_resilience(content, match.start())

            calls.append(Call(
                caller_service=service_name,
                callee_service=config.resolve_port(port),
                callee_port=port,
                method=method,
                path=path or "/",
                file_path=file_str,
                line_number=line_num,
                resolution_method="hardcoded",
                retry_config={"detected": True, "resilient": True} if has_resil else None,
            ))

    return calls


def _build_service_url_map(content: str, config) -> dict:
    """Build map from serviceURLs keys to resolved service info."""
    url_map = {}
    for match in SERVICE_URL_INIT_RE.finditer(content):
        key, env_or_func, service_hint = match.groups()
        service_name = service_hint or _env_to_service(env_or_func)
        port = config.resolve_service(service_name)
        url_map[key] = {"service": service_name, "port": port}

    # Also try simpler patterns: svcURLs["key"] = "http://..."
    simple_re = re.compile(r'svcURLs\["(\w+)"\]\s*=\s*"https?://[^:]+:(\d+)"')
    for match in simple_re.finditer(content):
        key, port_str = match.groups()
        port = int(port_str)
        url_map[key] = {"service": config.resolve_port(port), "port": port}

    return url_map


def _env_to_service(env_key: str) -> str:
    """Convert environment variable key to service name.

    AGI_CORE_URL -> agi-core
    RAG_GATEWAY_URL -> rag-gateway
    USER_SERVICE_PORT -> user-service
    """
    name = env_key.replace("_URL", "").replace("_PORT", "").replace("_HOST", "").replace("_ADDR", "")
    return name.lower().replace("_", "-")


def _detect_http_method(content: str, pos: int) -> str:
    """Detect the HTTP method used near a URL reference."""
    window_start = max(0, pos - 200)
    window_end = min(len(content), pos + 300)
    window = content[window_start:window_end]

    if ".Post(" in window or ".POST(" in window or "POST" in window:
        return "POST"
    if ".Put(" in window or ".PUT(" in window:
        return "PUT"
    if ".Delete(" in window or ".DELETE(" in window:
        return "DELETE"
    return "GET"  # default


def _find_path_after_resolve(content: str, pos: int) -> str:
    """Find the URL path concatenated after a service URL resolution.

    Looks for patterns like: + "/some/path"
    """
    window = content[pos:pos + 200]
    path_re = re.compile(r'\+\s*"(/[^"]*)"')
    match = path_re.search(window)
    if match:
        return match.group(1)
    return "/"


def _has_nearby_retry(content: str, pos: int) -> bool:
    """Check if there's a retry wrapper near this call."""
    window_start = max(0, pos - 1000)
    window = content[window_start:pos + 500]
    return bool(RETRY_WRAPPER_RE.search(window))


def _has_resilience(content: str, pos: int) -> bool:
    """Check if the call site has any resilience pattern nearby.

    Detects: FastHTTPClient, DefaultHTTPClient, http.Client with Timeout,
    circuit breakers, retry/attempt loops.
    """
    # Check nearby (1500 chars before, 500 after)
    window_start = max(0, pos - 1500)
    window_end = min(len(content), pos + 500)
    window = content[window_start:window_end]

    for pattern in RESILIENCE_PATTERNS:
        if pattern.search(window):
            return True

    # Also check file-level: if the file uses resilient patterns anywhere
    file_globals = ['FastHTTPClient', 'DefaultHTTPClient', 'SlowHTTPClient',
                    'MultimodalHTTPClient', 'QuickHTTPClient', 'serviceClient',
                    'httpGetJSON', 'httpPostJSON', 'httpclient.',
                    'context.WithTimeout', 'context.WithDeadline',
                    'http.Client{', 'Timeout:', 'timeout:']
    if any(g in content for g in file_globals):
        return True

    return False
