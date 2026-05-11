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
        if "__pycache__" in str(py_file) or "venv" in str(py_file):
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


def extract_python_calls(service_name: str, source_dir: str, config) -> list:
    """Extract outbound HTTP calls from Python source files.

    Returns list of Call objects.
    """
    calls = []
    source_path = Path(source_dir)

    for py_file in sorted(source_path.rglob("*.py")):
        if "__pycache__" in str(py_file) or "venv" in str(py_file):
            continue

        try:
            content = py_file.read_text(errors="replace")
        except (OSError, IOError):
            continue

        file_str = str(py_file)

        # Hardcoded URLs
        for match in PY_HARDCODED_URL_RE.finditer(content):
            full_url, port_str, path = match.groups()
            port = int(port_str)
            line_num = content[:match.start()].count("\n") + 1

            own_port = config.resolve_service(service_name)
            if port == own_port:
                continue

            method = _detect_python_method(content, match.start())

            calls.append(Call(
                caller_service=service_name,
                callee_service=config.resolve_port(port),
                callee_port=port,
                method=method,
                path=path or "/",
                file_path=file_str,
                line_number=line_num,
                resolution_method="hardcoded",
            ))

        # requests/httpx calls with URLs
        for match in PY_REQUEST_RE.finditer(content):
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

                calls.append(Call(
                    caller_service=service_name,
                    callee_service=config.resolve_port(port),
                    callee_port=port,
                    method=method.upper(),
                    path=path,
                    file_path=file_str,
                    line_number=line_num,
                    resolution_method="hardcoded",
                ))

    return calls


def _find_python_handler(lines: list, decorator_line: int) -> str:
    """Find the function name defined after a decorator line."""
    for i in range(decorator_line + 1, min(decorator_line + 5, len(lines))):
        match = re.match(r'\s*(?:async\s+)?def\s+(\w+)', lines[i])
        if match:
            return match.group(1)
    return ""


def _detect_python_method(content: str, pos: int) -> str:
    """Detect HTTP method near a URL in Python code."""
    window_start = max(0, pos - 200)
    window_end = min(len(content), pos + 300)
    window = content[window_start:window_end]

    if ".post(" in window or "POST" in window:
        return "POST"
    if ".put(" in window or "PUT" in window:
        return "PUT"
    if ".delete(" in window or "DELETE" in window:
        return "DELETE"
    return "GET"
