"""Extract HTTP routes, outbound calls, and schemas from TypeScript/JavaScript files."""
import re
from pathlib import Path
from ..models import Endpoint, Call, Schema

# Directories to skip when scanning
_SKIP_DIRS = {"node_modules", "dist", "build", ".next", ".nuxt", "coverage", "__tests__"}

# File extensions to scan
_TS_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".mjs"}

# ---------------------------------------------------------------------------
# Route patterns
# ---------------------------------------------------------------------------

# Express / Hono (same shape: app.get('/path', handler))
EXPRESS_ROUTE_RE = re.compile(
    r"(?:app|router|server)\.(get|post|put|delete|patch|all|options|head)"
    r"\(\s*['\"`]([^'\"`]+)['\"`]",
    re.IGNORECASE,
)

# Fastify
FASTIFY_ROUTE_RE = re.compile(
    r"(?:fastify|server|app)\.(get|post|put|delete|patch)"
    r"\(\s*['\"`]([^'\"`]+)['\"`]",
    re.IGNORECASE,
)

# NestJS controller decorator
NEST_CONTROLLER_RE = re.compile(
    r"@Controller\(\s*['\"`]([^'\"`]*)['\"`]\s*\)",
)

# NestJS method decorators
NEST_METHOD_RE = re.compile(
    r"@(Get|Post|Put|Delete|Patch|Head|Options|All)"
    r"\(\s*(?:['\"`]([^'\"`]*)['\"`])?\s*\)",
)

# Vite dev-server proxy config
VITE_PROXY_RE = re.compile(
    r"['\"`](/[^'\"`]+)['\"`]\s*:\s*\{\s*target\s*:\s*['\"`](https?://[^'\"`]+)['\"`]",
)

# app.use('/prefix', router) — Express prefix mounts
EXPRESS_USE_RE = re.compile(
    r"(?:app|server)\.use\(\s*['\"`]([^'\"`]+)['\"`]\s*,\s*(\w+)",
)

# ---------------------------------------------------------------------------
# Outbound call patterns
# ---------------------------------------------------------------------------

# fetch('url') or fetch(`template`)
FETCH_PLAIN_RE = re.compile(
    r"fetch\(\s*['\"]([^'\"]+)['\"]",
)
FETCH_TEMPLATE_RE = re.compile(
    r"fetch\(\s*`([^`]+)`",
)
# fetch(variable + '/path')
FETCH_CONCAT_RE = re.compile(
    r"fetch\(\s*(\w+)\s*\+\s*['\"`]([^'\"`]+)['\"`]",
)

# axios.get/post/etc
AXIOS_METHOD_RE = re.compile(
    r"axios\.(get|post|put|delete|patch)\(\s*(?:['\"]([^'\"]+)['\"]|`([^`]+)`)",
)

# axios.create({ baseURL: '...' })
AXIOS_CREATE_RE = re.compile(
    r"axios\.create\(\s*\{[^}]*baseURL\s*:\s*['\"`]([^'\"`]+)['\"`]",
    re.DOTALL,
)

# got.get/post/etc
GOT_RE = re.compile(
    r"got\.(get|post|put|delete|patch)\(\s*(?:['\"]([^'\"]+)['\"]|`([^`]+)`)",
)

# Hardcoded localhost URLs assigned to const/let/var
URL_CONST_RE = re.compile(
    r"(?:const|let|var|export\s+const)\s+\w+\s*=\s*"
    r"['\"`](https?://(?:127\.0\.0\.1|localhost):(\d+)[^'\"`]*)['\"`]",
)

# process.env fallback:  process.env.X || 'http://localhost:8100'
ENV_FALLBACK_RE = re.compile(
    r"process\.env\.\w+\s*\|\|\s*['\"`](https?://(?:127\.0\.0\.1|localhost):(\d+)[^'\"`]*)['\"`]",
)

# WebSocket
WS_RE = re.compile(
    r"new\s+WebSocket\(\s*['\"`](wss?://[^'\"`]+)['\"`]",
)

# socket.io connect
SOCKETIO_RE = re.compile(
    r"io(?:\.connect)?\(\s*['\"`](https?://[^'\"`]+)['\"`]",
)

# Generic hardcoded localhost URL in any string (catch-all, used for calls)
HARDCODED_URL_RE = re.compile(
    r"""['"](https?://(?:127\.0\.0\.1|localhost):(\d+)(/[^'"]*)?)['""]""",
)
HARDCODED_URL_TEMPLATE_RE = re.compile(
    r"`(https?://(?:127\.0\.0\.1|localhost):(\d+)([^`]*)?)`",
)

# ---------------------------------------------------------------------------
# Schema patterns
# ---------------------------------------------------------------------------

TS_INTERFACE_RE = re.compile(
    r"(?:export\s+)?(?:interface|type)\s+(\w+)\s*(?:=\s*)?\{([^}]+)\}",
    re.DOTALL,
)

# Parse individual fields inside an interface body
TS_FIELD_RE = re.compile(
    r"(\w+)\s*\??\s*:\s*([^;\n,]+)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_ts_files(source_dir: str):
    """Yield Path objects for all TS/JS files, skipping excluded dirs."""
    root = Path(source_dir)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in _TS_EXTENSIONS:
            continue
        # Check if any parent is in skip list
        if _SKIP_DIRS.intersection(p.name for p in path.parents):
            continue
        # Also skip test files
        if ".test." in path.name or ".spec." in path.name or "__tests__" in str(path):
            continue
        yield path


def _read_file(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except (OSError, IOError):
        return ""


def _line_number(content: str, pos: int) -> int:
    return content[:pos].count("\n") + 1


def _detect_http_method(content: str, pos: int) -> str:
    """Detect HTTP method near a URL reference in TS/JS code."""
    window_start = max(0, pos - 300)
    window_end = min(len(content), pos + 300)
    window = content[window_start:window_end].lower()

    if "method:" in window:
        method_match = re.search(r"method\s*:\s*['\"`](\w+)['\"`]", window, re.IGNORECASE)
        if method_match:
            return method_match.group(1).upper()

    if ".post(" in window or "'post'" in window or '"post"' in window:
        return "POST"
    if ".put(" in window or "'put'" in window or '"put"' in window:
        return "PUT"
    if ".delete(" in window or "'delete'" in window or '"delete"' in window:
        return "DELETE"
    if ".patch(" in window or "'patch'" in window or '"patch"' in window:
        return "PATCH"
    return "GET"


def _extract_port_and_path(url: str):
    """Extract port number and path from a URL string.

    Returns (port, path) or (None, None) if no port found.
    Handles template literal placeholders like ${VAR}.
    """
    # Strip template literal expressions for port extraction
    cleaned = re.sub(r"\$\{[^}]+\}", "", url)
    match = re.search(r":(\d{4,5})(/[^\s'\"`,]*)?", cleaned)
    if match:
        port = int(match.group(1))
        path = match.group(2) or "/"
        # Re-extract path from original to preserve template expressions
        path_match = re.search(r":\d{4,5}(/[^\s'\"`,`]*)", url)
        if path_match:
            path = path_match.group(1)
        return port, path
    return None, None


def _find_handler_name(content: str, line_idx: int) -> str:
    """Try to find a handler/function name near a route definition."""
    lines = content.split("\n")
    # Check the same line and next few lines for function references
    for i in range(line_idx, min(line_idx + 3, len(lines))):
        # Named function reference: ..., handlerName)
        m = re.search(r",\s*(\w+)\s*\)", lines[i])
        if m and m.group(1) not in ("req", "res", "request", "reply", "c", "ctx", "next"):
            return m.group(1)
        # Arrow or regular function on same line
        m = re.search(r"(?:async\s+)?(?:function\s+)?(\w+)\s*\(", lines[i])
        if m and m.group(1) not in ("get", "post", "put", "delete", "patch", "app",
                                     "router", "server", "fastify", "async", "function"):
            return m.group(1)
    return ""


def _nextjs_path_from_file(file_path: Path, source_dir: str) -> str:
    """Convert a Next.js file path to an API route path.

    pages/api/users/[id].ts  -> /api/users/:id
    app/api/users/route.ts   -> /api/users
    """
    rel = str(file_path.relative_to(source_dir))

    # app router: app/api/.../route.ts
    app_match = re.match(r"(?:src/)?app/(api/.+)/route\.\w+$", rel)
    if app_match:
        route = "/" + app_match.group(1)
        # Convert [param] to :param
        route = re.sub(r"\[([^\]]+)\]", r":\1", route)
        return route

    # pages router: pages/api/...
    pages_match = re.match(r"(?:src/)?pages/(api/.+)\.\w+$", rel)
    if pages_match:
        route = "/" + pages_match.group(1)
        # Remove /index suffix
        route = re.sub(r"/index$", "", route)
        route = re.sub(r"\[([^\]]+)\]", r":\1", route)
        return route or "/"

    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_ts_routes(service_name: str, port: int, source_dir: str) -> list:
    """Extract HTTP route registrations from TypeScript/JavaScript files.

    Supports Express, Fastify, NestJS, Hono, Next.js file-based routes,
    and Vite dev-server proxy configs.

    Returns list of Endpoint objects.
    """
    endpoints = []

    # Track Express prefix mounts for resolving grouped routes
    # (simplified: we don't do full cross-file resolution)
    use_prefixes = {}  # router_var -> prefix

    for ts_file in _iter_ts_files(source_dir):
        content = _read_file(ts_file)
        if not content:
            continue

        file_str = str(ts_file)
        lines = content.split("\n")

        # --- Next.js file-based routes ---
        nextjs_path = _nextjs_path_from_file(ts_file, source_dir)
        if nextjs_path:
            # Detect which HTTP methods are exported
            exported_methods = set()
            for m in re.finditer(r"export\s+(?:async\s+)?function\s+(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b", content):
                exported_methods.add(m.group(1))
            # Also check for export default (pages router — handles all methods)
            if re.search(r"export\s+default", content):
                exported_methods.update({"GET", "POST"})
            if not exported_methods:
                exported_methods = {"GET"}

            for method in sorted(exported_methods):
                endpoints.append(Endpoint(
                    service_name=service_name,
                    port=port,
                    method=method,
                    path=nextjs_path,
                    file_path=file_str,
                    line_number=1,
                    handler_name="",
                    framework="nextjs",
                ))

        # --- Express / Hono routes ---
        # Collect app.use() prefix mounts first
        for match in EXPRESS_USE_RE.finditer(content):
            prefix, router_var = match.groups()
            use_prefixes[router_var] = prefix

        for match in EXPRESS_ROUTE_RE.finditer(content):
            method, path = match.groups()
            line_num = _line_number(content, match.start())
            handler = _find_handler_name(content, line_num - 1)

            # Detect framework
            framework = "express"
            if "Hono" in content or "hono" in content:
                framework = "hono"

            endpoints.append(Endpoint(
                service_name=service_name,
                port=port,
                method=method.upper(),
                path=path,
                file_path=file_str,
                line_number=line_num,
                handler_name=handler,
                framework=framework,
            ))

        # --- Fastify routes ---
        # Only match if this file looks like Fastify (avoid double-matching Express)
        if "fastify" in content.lower() or "Fastify" in content:
            for match in FASTIFY_ROUTE_RE.finditer(content):
                method, path = match.groups()
                line_num = _line_number(content, match.start())
                handler = _find_handler_name(content, line_num - 1)

                endpoints.append(Endpoint(
                    service_name=service_name,
                    port=port,
                    method=method.upper(),
                    path=path,
                    file_path=file_str,
                    line_number=line_num,
                    handler_name=handler,
                    framework="fastify",
                ))

        # --- NestJS routes ---
        if "@Controller" in content:
            # Find controller prefix
            ctrl_match = NEST_CONTROLLER_RE.search(content)
            ctrl_prefix = ""
            if ctrl_match:
                ctrl_prefix = ctrl_match.group(1)
                if ctrl_prefix and not ctrl_prefix.startswith("/"):
                    ctrl_prefix = "/" + ctrl_prefix

            for match in NEST_METHOD_RE.finditer(content):
                method_name, sub_path = match.groups()
                sub_path = sub_path or ""
                if sub_path and not sub_path.startswith("/"):
                    sub_path = "/" + sub_path

                full_path = ctrl_prefix + sub_path
                if not full_path:
                    full_path = "/"

                line_num = _line_number(content, match.start())

                # Find the method name on the next line
                handler = ""
                for i in range(line_num, min(line_num + 5, len(lines))):
                    m = re.match(r"\s*(?:async\s+)?(\w+)\s*\(", lines[i])
                    if m:
                        handler = m.group(1)
                        break

                endpoints.append(Endpoint(
                    service_name=service_name,
                    port=port,
                    method=method_name.upper(),
                    path=full_path,
                    file_path=file_str,
                    line_number=line_num,
                    handler_name=handler,
                    framework="nestjs",
                ))

        # --- Vite proxy config ---
        if "proxy" in content.lower() and ("vite" in ts_file.name.lower() or "config" in ts_file.name.lower()):
            for match in VITE_PROXY_RE.finditer(content):
                path, target = match.groups()
                line_num = _line_number(content, match.start())

                endpoints.append(Endpoint(
                    service_name=service_name,
                    port=port,
                    method="PROXY",
                    path=path,
                    file_path=file_str,
                    line_number=line_num,
                    handler_name="",
                    framework="vite-proxy",
                ))

    return endpoints


def extract_ts_calls(service_name: str, source_dir: str, config) -> list:
    """Extract outbound HTTP calls from TypeScript/JavaScript files.

    Detects fetch(), axios, got, WebSocket, socket.io, and hardcoded
    localhost URLs.

    Returns list of Call objects.
    """
    calls = []
    own_port = config.resolve_service(service_name)

    for ts_file in _iter_ts_files(source_dir):
        content = _read_file(ts_file)
        if not content:
            continue

        file_str = str(ts_file)
        seen_lines = set()  # Avoid duplicate calls on same line

        # TS/JS resilience detection
        ts_resilience_patterns = ['AbortController', 'signal:', 'timeout:', 'timeout=',
                                  'retry', 'retries', 'maxRetries', 'AbortSignal.timeout',
                                  'axios.create', 'got(', 'ky(']
        file_has_resilience = any(p in content for p in ts_resilience_patterns)

        def _add_call(port, path, method, line_num, resolution):
            if port == own_port:
                return
            if line_num in seen_lines:
                return
            seen_lines.add(line_num)
            call = Call(
                caller_service=service_name,
                callee_service=config.resolve_port(port),
                callee_port=port,
                method=method,
                path=path or "/",
                file_path=file_str,
                line_number=line_num,
                resolution_method=resolution,
            )
            if file_has_resilience:
                call.retry_config = {"type": "timeout_or_abort"}
            calls.append(call)

        # --- fetch() with plain strings ---
        for match in FETCH_PLAIN_RE.finditer(content):
            url = match.group(1)
            port, path = _extract_port_and_path(url)
            if port:
                line_num = _line_number(content, match.start())
                method = _detect_http_method(content, match.start())
                _add_call(port, path, method, line_num, "hardcoded")

        # --- fetch() with template literals ---
        for match in FETCH_TEMPLATE_RE.finditer(content):
            url = match.group(1)
            port, path = _extract_port_and_path(url)
            if port:
                line_num = _line_number(content, match.start())
                method = _detect_http_method(content, match.start())
                _add_call(port, path, method, line_num, "hardcoded")

        # --- fetch() with concatenation ---
        for match in FETCH_CONCAT_RE.finditer(content):
            var_name, path_suffix = match.groups()
            # Try to resolve the variable to a URL
            var_url = _resolve_variable_url(content, var_name)
            if var_url:
                port, _ = _extract_port_and_path(var_url)
                if port:
                    line_num = _line_number(content, match.start())
                    method = _detect_http_method(content, match.start())
                    _add_call(port, path_suffix, method, line_num, "hardcoded")

        # --- axios method calls ---
        for match in AXIOS_METHOD_RE.finditer(content):
            method_name, url_plain, url_template = match.groups()
            url = url_plain or url_template or ""
            port, path = _extract_port_and_path(url)
            if port:
                line_num = _line_number(content, match.start())
                _add_call(port, path, method_name.upper(), line_num, "hardcoded")

        # --- axios.create baseURL ---
        for match in AXIOS_CREATE_RE.finditer(content):
            base_url = match.group(1)
            port, path = _extract_port_and_path(base_url)
            if port:
                line_num = _line_number(content, match.start())
                _add_call(port, path, "GET", line_num, "hardcoded")

        # --- got calls ---
        for match in GOT_RE.finditer(content):
            method_name, url_plain, url_template = match.groups()
            url = url_plain or url_template or ""
            port, path = _extract_port_and_path(url)
            if port:
                line_num = _line_number(content, match.start())
                _add_call(port, path, method_name.upper(), line_num, "hardcoded")

        # --- Hardcoded URLs in const/let/var ---
        for match in URL_CONST_RE.finditer(content):
            url = match.group(1)
            port_str = match.group(2)
            port = int(port_str)
            line_num = _line_number(content, match.start())
            path = "/"
            path_m = re.search(r":\d+(/[^'\"`]*)", url)
            if path_m:
                path = path_m.group(1)
            _add_call(port, path, "GET", line_num, "hardcoded")

        # --- process.env fallback URLs ---
        for match in ENV_FALLBACK_RE.finditer(content):
            url = match.group(1)
            port = int(match.group(2))
            line_num = _line_number(content, match.start())
            path = "/"
            path_m = re.search(r":\d+(/[^'\"`]*)", url)
            if path_m:
                path = path_m.group(1)
            _add_call(port, path, "GET", line_num, "hardcoded")

        # --- WebSocket ---
        for match in WS_RE.finditer(content):
            url = match.group(1)
            port, path = _extract_port_and_path(url)
            if port:
                line_num = _line_number(content, match.start())
                _add_call(port, path, "WS", line_num, "hardcoded")

        # --- socket.io ---
        for match in SOCKETIO_RE.finditer(content):
            url = match.group(1)
            port, path = _extract_port_and_path(url)
            if port:
                line_num = _line_number(content, match.start())
                _add_call(port, path, "WS", line_num, "hardcoded")

        # --- Catch-all: any hardcoded localhost URL in plain strings ---
        for match in HARDCODED_URL_RE.finditer(content):
            full_url, port_str, path = match.groups()
            port = int(port_str)
            line_num = _line_number(content, match.start())
            method = _detect_http_method(content, match.start())
            _add_call(port, path or "/", method, line_num, "hardcoded")

        # --- Catch-all: hardcoded localhost in template literals ---
        for match in HARDCODED_URL_TEMPLATE_RE.finditer(content):
            full_url, port_str, path = match.groups()
            port = int(port_str)
            line_num = _line_number(content, match.start())
            method = _detect_http_method(content, match.start())
            _add_call(port, path or "/", method, line_num, "hardcoded")

    return calls


def _resolve_variable_url(content: str, var_name: str) -> str:
    """Try to find the value of a variable assigned to a URL.

    Searches for patterns like:
        const API_URL = 'http://localhost:8100'
        const API_URL = process.env.X || 'http://localhost:8100'
    """
    # Direct assignment
    m = re.search(
        rf"(?:const|let|var)\s+{re.escape(var_name)}\s*=\s*['\"`]([^'\"`]+)['\"`]",
        content,
    )
    if m:
        return m.group(1)

    # Env fallback
    m = re.search(
        rf"(?:const|let|var)\s+{re.escape(var_name)}\s*=\s*process\.env\.\w+\s*\|\|\s*['\"`]([^'\"`]+)['\"`]",
        content,
    )
    if m:
        return m.group(1)

    return ""


def extract_ts_schemas(source_dir: str) -> dict:
    """Extract TypeScript interfaces and type definitions.

    Returns {name: Schema} dict.
    """
    schemas = {}

    for ts_file in _iter_ts_files(source_dir):
        content = _read_file(ts_file)
        if not content:
            continue

        file_str = str(ts_file)

        for match in TS_INTERFACE_RE.finditer(content):
            name = match.group(1)
            body = match.group(2)
            line_num = _line_number(content, match.start())

            fields = {}
            json_tags = {}
            for field_match in TS_FIELD_RE.finditer(body):
                field_name = field_match.group(1)
                field_type = field_match.group(2).strip().rstrip(",;")
                fields[field_name] = field_type
                # In TS the JSON key is the field name (no separate tags)
                json_tags[field_name] = field_name

            if fields:
                schemas[name] = Schema(
                    name=name,
                    fields=fields,
                    json_tags=json_tags,
                    file_path=file_str,
                    line_number=line_num,
                )

    return schemas
