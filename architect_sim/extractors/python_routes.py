"""Extract HTTP route registrations from Python source files (FastAPI, Flask)."""
import re
from pathlib import Path
from ..models import Endpoint, Call

# FastAPI patterns
FASTAPI_ROUTE_RE = re.compile(
    r'@(?:\w+)\.(get|post|put|delete|patch)\(\s*["\']([^"\']+)["\']',
    re.MULTILINE
)
FASTAPI_ROUTER_RE = re.compile(
    r'@(?:\w+)\.(get|post|put|delete|patch)\(\s*["\']([^"\']+)["\']',
    re.MULTILINE
)
FASTAPI_INCLUDE_RE = re.compile(
    r'\.include_router\(\s*(\w+)\s*,\s*prefix\s*=\s*["\']([^"\']+)["\']',
    re.MULTILINE
)

# Flask patterns
FLASK_ROUTE_RE = re.compile(
    r'@(?:\w+)\.route\(\s*["\']([^"\']+)["\'](?:\s*,\s*methods\s*=\s*\[([^\]]+)\])?',
    re.MULTILINE
)

# Hardcoded localhost URLs in Python
PY_HARDCODED_URL_RE = re.compile(
    r'["\']+(https?://(?:127\.0\.0\.1|localhost):(\d+)(/[^"\']*)?)["\']',
    re.MULTILINE
)

# requests/httpx calls
PY_REQUEST_RE = re.compile(
    r'(?:requests|httpx|session|client)\.(get|post|put|delete|patch)\(\s*(?:f?["\']([^"\']+)["\']|(\w+))',
    re.MULTILINE
)


def extract_python_routes(service_name: str, port: int, source_dir: str) -> list:
    """Extract HTTP route registrations from Python source files.

    Returns list of Endpoint objects.
    """
    endpoints = []
    source_path = Path(source_dir)

    for py_file in sorted(source_path.rglob("*.py")):
        if _is_test_or_vendor(py_file):
            continue

        try:
            content = py_file.read_text(errors="replace")
        except (OSError, IOError):
            continue

        file_str = str(py_file)
        lines = content.split("\n")

        # Detect framework
        is_fastapi = "FastAPI" in content or "fastapi" in content
        is_flask = "Flask" in content or "flask" in content

        # Build router prefix map for FastAPI
        router_prefixes = {}
        for match in FASTAPI_INCLUDE_RE.finditer(content):
            router_var, prefix = match.groups()
            router_prefixes[router_var] = prefix

        if is_fastapi:
            for match in FASTAPI_ROUTE_RE.finditer(content):
                method, path = match.groups()
                line_num = content[:match.start()].count("\n") + 1

                # Find the handler function name on the next line
                handler = _find_python_handler(lines, line_num - 1)

                endpoints.append(Endpoint(
                    service_name=service_name,
                    port=port,
                    method=method.upper(),
                    path=path,
                    file_path=file_str,
                    line_number=line_num,
                    handler_name=handler,
                    framework="fastapi",
                ))

        if is_flask:
            for match in FLASK_ROUTE_RE.finditer(content):
                path = match.group(1)
                methods_str = match.group(2)
                line_num = content[:match.start()].count("\n") + 1
                handler = _find_python_handler(lines, line_num - 1)

                if methods_str:
                    # Parse methods list: ["GET", "POST"]
                    methods = re.findall(r'["\'](\w+)["\']', methods_str)
                else:
                    methods = ["GET"]

                for method in methods:
                    endpoints.append(Endpoint(
                        service_name=service_name,
                        port=port,
                        method=method.upper(),
                        path=path,
                        file_path=file_str,
                        line_number=line_num,
                        handler_name=handler,
                        framework="flask",
                    ))

    return endpoints


# Python resilience patterns
PY_RESILIENCE_PATTERNS = [
    'timeout=', 'timeout:', 'Timeout(',
    'retry', 'retries', '@retry', 'tenacity', 'backoff',
    'circuit_breaker', 'CircuitBreaker',
    'max_retries', 'max_attempts',
    'aiohttp.ClientTimeout', 'httpx.Timeout',
    'requests.adapters.HTTPAdapter', 'urllib3.util.retry',
]


def _has_python_resilience(content: str, pos: int) -> bool:
    """Check if file has resilience patterns. Uses full file for session/client patterns."""
    # Check nearby (1000 chars) for call-level patterns
    window = content[max(0, pos - 1000):pos + 1000]
    if any(p in window for p in PY_RESILIENCE_PATTERNS):
        return True
    # Also check entire file for session/client-level patterns (set once, used everywhere)
    file_patterns = ['httpx.Client(', 'httpx.AsyncClient(', 'requests.Session(',
                     'aiohttp.ClientSession(', 'ClientTimeout(', 'max_retries=']
    return any(p in content for p in file_patterns)


def extract_python_calls(service_name: str, source_dir: str, config) -> list:
    """Extract outbound HTTP calls from Python source files.

    Returns list of Call objects.
    """
    calls = []
    source_path = Path(source_dir)

    for py_file in sorted(source_path.rglob("*.py")):
        if _is_test_or_vendor(py_file):
            continue

        try:
            content = py_file.read_text(errors="replace")
        except (OSError, IOError):
            continue

        file_str = str(py_file)

        # Hardcoded URLs
        for match in PY_HARDCODED_URL_RE.finditer(content):
            if _inside_python_function(content, match.start(), "self_test"):
                continue
            full_url, port_str, path = match.groups()
            port = int(port_str)
            line_num = content[:match.start()].count("\n") + 1

            own_port = config.resolve_service(service_name)
            if port == own_port:
                continue

            method = _detect_python_method(content, match.start())

            call = Call(
                caller_service=service_name,
                callee_service=config.resolve_port(port),
                callee_port=port,
                method=method,
                path=path or "/",
                file_path=file_str,
                line_number=line_num,
                resolution_method="hardcoded",
            )
            if _has_python_resilience(content, match.start()):
                call.retry_config = {"type": "timeout_or_retry"}
            calls.append(call)

        # requests/httpx calls with URLs
        for match in PY_REQUEST_RE.finditer(content):
            if _inside_python_function(content, match.start(), "self_test"):
                continue
            method, url, var = match.groups()
            if not url:
                continue
            line_num = content[:match.start()].count("\n") + 1

            port_match = re.search(r':(\d{4,5})', url)
            if port_match:
                port = int(port_match.group(1))
                path_match = re.search(r':\d+(/[^"\'{}]*)', url)
                path = path_match.group(1) if path_match else "/"

                own_port = config.resolve_service(service_name)
                if port == own_port:
                    continue

                call = Call(
                    caller_service=service_name,
                    callee_service=config.resolve_port(port),
                    callee_port=port,
                    method=method.upper(),
                    path=path,
                    file_path=file_str,
                    line_number=line_num,
                    resolution_method="hardcoded",
                )
                if _has_python_resilience(content, match.start()):
                    call.retry_config = {"type": "timeout_or_retry"}
                calls.append(call)

    return calls


def _find_python_handler(lines: list, decorator_line: int) -> str:
    """Find the function name defined after a decorator line."""
    for i in range(decorator_line + 1, min(decorator_line + 5, len(lines))):
        match = re.match(r'\s*(?:async\s+)?def\s+(\w+)', lines[i])
        if match:
            return match.group(1)
    return ""


def _detect_python_method(content: str, pos: int) -> str:
    """Detect HTTP method near a URL in Python code.

    Uses line-based context to avoid false positives from neighboring code.
    """
    # Get the line and tight context (±3 lines)
    line_start = content.rfind("\n", 0, pos) + 1
    line_end = content.find("\n", pos)
    if line_end == -1:
        line_end = len(content)

    ctx_start = line_start
    for _ in range(3):
        prev = content.rfind("\n", 0, ctx_start - 1)
        if prev == -1:
            ctx_start = 0
            break
        ctx_start = prev + 1

    ctx_end = line_end
    for _ in range(3):
        nxt = content.find("\n", ctx_end + 1)
        if nxt == -1:
            ctx_end = len(content)
            break
        ctx_end = nxt

    near = content[ctx_start:ctx_end]

    # Priority 1: Explicit client method calls (requests.post, httpx.post, etc.)
    method_call_re = re.compile(r'\.(post|get|put|delete|patch)\s*\(')
    m = method_call_re.search(near)
    if m:
        return m.group(1).upper()

    # Priority 2: urllib.request.Request with data= implies POST
    if "Request(" in near and "data=" in near:
        return "POST"

    # Priority 3: method= keyword argument
    method_kwarg_re = re.compile(r'method\s*=\s*["\'](\w+)["\']')
    m = method_kwarg_re.search(near)
    if m:
        return m.group(1).upper()

    return "GET"


def _is_test_or_vendor(path: Path) -> bool:
    path_str = str(path)
    name = path.name
    return (
        "__pycache__" in path_str
        or "venv" in path_str
        or ".venv" in path_str
        or "tests" in path.parts
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _inside_python_function(content: str, pos: int, function_name: str) -> bool:
    prefix = content[:pos]
    match = list(re.finditer(rf"^def\s+{re.escape(function_name)}\s*\(", prefix, re.MULTILINE))
    if not match:
        return False
    start = match[-1].start()
    next_def = re.search(r"^(?:def|async\s+def)\s+\w+\s*\(", content[start + 1 :], re.MULTILINE)
    if next_def is None:
        return True
    return pos < start + 1 + next_def.start()
