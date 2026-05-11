"""Extract HTTP route registrations from Rust source files (axum, actix-web)."""
import re
from pathlib import Path
from ..models import Endpoint, Call

# Axum route patterns
AXUM_ROUTE_RE = re.compile(
    r'\.route\(\s*"([^"]+)"\s*,\s*(?:get|post|put|delete|patch)\(\s*(\w+)',
    re.MULTILINE
)
AXUM_METHOD_ROUTER_RE = re.compile(
    r'\.route\(\s*"([^"]+)"\s*,\s*(get|post|put|delete|patch)\(',
    re.MULTILINE
)

# Actix-web patterns
ACTIX_ROUTE_RE = re.compile(
    r'\.(?:resource|route)\(\s*"([^"]+)"\s*\)\s*\.(?:method\(\s*Method::(GET|POST|PUT|DELETE|PATCH)|to\(\s*(\w+))',
    re.MULTILINE
)
ACTIX_MACRO_RE = re.compile(
    r'#\[(get|post|put|delete|patch)\(\s*"([^"]+)"',
    re.MULTILINE
)

# Hardcoded localhost URLs in Rust
RUST_HARDCODED_URL_RE = re.compile(
    r'"(https?://(?:127\.0\.0\.1|localhost):(\d+)(/[^"]*)?)"',
    re.MULTILINE
)

# reqwest/hyper client calls
RUST_CLIENT_RE = re.compile(
    r'client\.(get|post|put|delete|patch)\(\s*(?:&?format!\(\s*"([^"]+)"|\s*"([^"]+)")',
    re.MULTILINE
)


def extract_rust_routes(service_name: str, port: int, source_dir: str) -> list:
    """Extract HTTP route registrations from Rust source files.

    Returns list of Endpoint objects.
    """
    endpoints = []
    source_path = Path(source_dir)

    for rs_file in sorted(source_path.rglob("*.rs")):
        if "target" in str(rs_file) or "vendor" in str(rs_file):
            continue

        try:
            content = rs_file.read_text(errors="replace")
        except (OSError, IOError):
            continue

        file_str = str(rs_file)

        # Axum routes
        for match in AXUM_METHOD_ROUTER_RE.finditer(content):
            path, method = match.groups()
            line_num = content[:match.start()].count("\n") + 1

            # Try to find handler name
            handler_match = re.search(
                r'\.route\(\s*"' + re.escape(path) + r'"\s*,\s*' + method + r'\(\s*(\w+)',
                content
            )
            handler = handler_match.group(1) if handler_match else ""

            endpoints.append(Endpoint(
                service_name=service_name,
                port=port,
                method=method.upper(),
                path=path,
                file_path=file_str,
                line_number=line_num,
                handler_name=handler,
                framework="axum",
            ))

        # Actix-web macro routes
        for match in ACTIX_MACRO_RE.finditer(content):
            method, path = match.groups()
            line_num = content[:match.start()].count("\n") + 1

            # Find the function name after the macro
            fn_match = re.search(
                r'#\[' + method + r'\(\s*"' + re.escape(path) + r'"[^)]*\)\]\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)',
                content[match.start():]
            )
            handler = fn_match.group(1) if fn_match else ""

            endpoints.append(Endpoint(
                service_name=service_name,
                port=port,
                method=method.upper(),
                path=path,
                file_path=file_str,
                line_number=line_num,
                handler_name=handler,
                framework="actix-web",
            ))

    return endpoints


def extract_rust_calls(service_name: str, source_dir: str, config) -> list:
    """Extract outbound HTTP calls from Rust source files.

    Returns list of Call objects.
    """
    calls = []
    source_path = Path(source_dir)

    for rs_file in sorted(source_path.rglob("*.rs")):
        if "target" in str(rs_file) or "vendor" in str(rs_file):
            continue

        try:
            content = rs_file.read_text(errors="replace")
        except (OSError, IOError):
            continue

        file_str = str(rs_file)

        # Hardcoded URLs
        for match in RUST_HARDCODED_URL_RE.finditer(content):
            full_url, port_str, path = match.groups()
            port = int(port_str)
            line_num = content[:match.start()].count("\n") + 1

            own_port = config.resolve_service(service_name)
            if port == own_port:
                continue

            method = _detect_rust_method(content, match.start())

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

        # Client method calls
        for match in RUST_CLIENT_RE.finditer(content):
            method, format_url, literal_url = match.groups()
            url = format_url or literal_url or ""
            line_num = content[:match.start()].count("\n") + 1

            port_match = re.search(r':(\d{4,5})', url)
            if port_match:
                port = int(port_match.group(1))
                path_match = re.search(r':\d+(/[^"{}]*)', url)
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


def _detect_rust_method(content: str, pos: int) -> str:
    """Detect HTTP method near a URL in Rust code."""
    window_start = max(0, pos - 200)
    window_end = min(len(content), pos + 300)
    window = content[window_start:window_end]

    if ".post(" in window or "Method::POST" in window:
        return "POST"
    if ".put(" in window or "Method::PUT" in window:
        return "PUT"
    if ".delete(" in window or "Method::DELETE" in window:
        return "DELETE"
    return "GET"
