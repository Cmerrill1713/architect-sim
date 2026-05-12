"""Extract HTTP route registrations from Go source files."""
import re
from pathlib import Path
from ..models import Endpoint

# Gin patterns
GIN_ROUTE_RE = re.compile(
    r'(\w+)\.(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\(\s*"([^"]*)"\s*,\s*([\w.]+)',
    re.MULTILINE
)
GIN_GROUP_RE = re.compile(
    r'(\w+)\s*:?=\s*([\w.]+)\.Group\(\s*"([^"]+)"',
    re.MULTILINE
)

# Mux patterns — handler can be a simple identifier OR a wrapped call like s.withMiddleware(s.handler)
MUX_ROUTE_RE = re.compile(
    r'(\w+)\.HandleFunc\(\s*"([^"]+)"\s*,\s*([^)]+)\)',
    re.MULTILINE
)
MUX_METHOD_RE = re.compile(
    r'\.HandleFunc\(\s*"([^"]+)"\s*,\s*[^)]+\)\s*\.Methods\(\s*([^)]+)\)',
    re.MULTILINE
)

# http.HandleFunc (stdlib)
STDLIB_ROUTE_RE = re.compile(
    r'http\.HandleFunc\(\s*"([^"]+)"\s*,\s*([\w.]+)',
    re.MULTILINE
)


def extract_go_routes(service_name: str, port: int, source_dir: str) -> list:
    """Extract all HTTP route registrations from Go source files.

    Returns list of Endpoint objects.
    """
    endpoints = []
    source_path = Path(source_dir)

    for go_file in sorted(source_path.rglob("*.go")):
        # Skip test files and vendor
        if go_file.name.endswith("_test.go") or "vendor" in str(go_file):
            continue

        try:
            content = go_file.read_text(errors="replace")
        except (OSError, IOError):
            continue

        # Build group prefix map: variable_name -> prefix_path
        group_prefixes = _build_group_map(content)

        # Extract gin routes
        for match in GIN_ROUTE_RE.finditer(content):
            var_name, method, path, handler = match.groups()
            full_path = _resolve_group_path(var_name, path, group_prefixes)
            line_num = content[:match.start()].count("\n") + 1

            endpoints.append(Endpoint(
                service_name=service_name,
                port=port,
                method=method,
                path=full_path,
                file_path=str(go_file),
                line_number=line_num,
                handler_name=handler,
                framework="gin",
            ))

        # Extract mux routes
        # Use a combined regex that captures path, handler, and method together
        # to avoid the method_map overwrite problem (same path, different methods)
        MUX_FULL_RE = re.compile(
            r'(\w+)\.HandleFunc\(\s*"([^"]+)"\s*,\s*([^)]+)\)(?:\s*\.Methods\(\s*([^)]+)\))?',
            re.MULTILINE
        )
        for match in MUX_FULL_RE.finditer(content):
            var_name, path, handler, methods_str = match.groups()
            line_num = content[:match.start()].count("\n") + 1

            if methods_str:
                # Explicit .Methods() constraint — parse all methods
                methods = [m.strip().strip('"').strip("'")
                           for m in methods_str.split(",")]
                for method in methods:
                    if method:
                        endpoints.append(Endpoint(
                            service_name=service_name,
                            port=port,
                            method=method.upper(),
                            path=path,
                            file_path=str(go_file),
                            line_number=line_num,
                            handler_name=handler,
                            framework="mux",
                        ))
            else:
                # No .Methods() — mux accepts all methods
                for method in ["GET", "POST"]:
                    endpoints.append(Endpoint(
                        service_name=service_name,
                        port=port,
                        method=method,
                        path=path,
                        file_path=str(go_file),
                        line_number=line_num,
                        handler_name=handler,
                        framework="mux",
                    ))

        # Extract stdlib http.HandleFunc routes
        # stdlib handlers accept ALL methods — check if the handler body
        # restricts to a specific method, otherwise register for all common methods
        for match in STDLIB_ROUTE_RE.finditer(content):
            path, handler = match.groups()
            line_num = content[:match.start()].count("\n") + 1

            # Check if handler restricts method (r.Method != http.MethodPost, etc.)
            restricted_method = _detect_stdlib_method_restriction(content, handler)

            if restricted_method:
                endpoints.append(Endpoint(
                    service_name=service_name,
                    port=port,
                    method=restricted_method,
                    path=path,
                    file_path=str(go_file),
                    line_number=line_num,
                    handler_name=handler,
                    framework="stdlib",
                ))
            else:
                # stdlib handles all methods — register both GET and POST
                for method in ["GET", "POST"]:
                    endpoints.append(Endpoint(
                        service_name=service_name,
                        port=port,
                        method=method,
                        path=path,
                        file_path=str(go_file),
                        line_number=line_num,
                        handler_name=handler,
                        framework="stdlib",
                    ))

    return endpoints


def _build_group_map(content: str) -> dict:
    """Build a map of variable names to their group path prefixes.

    Handles nested groups by tracking parent relationships.
    For dotted parents like 's.router', uses the last segment ('router')
    as the parent variable name for lookup.
    """
    groups = {}  # var_name -> (parent_var, prefix)

    for match in GIN_GROUP_RE.finditer(content):
        child_var, parent_var, prefix = match.groups()
        # For dotted expressions like 's.router', use last segment
        if "." in parent_var:
            parent_var = parent_var.rsplit(".", 1)[-1]
        groups[child_var] = (parent_var, prefix)

    return groups


def _resolve_group_path(var_name: str, path: str, groups: dict, depth: int = 0) -> str:
    """Resolve the full path including group prefixes.

    Follows the parent chain: child.POST("/execute") -> parent.Group("/agi") -> r (root)
    Result: /agi/execute
    """
    if depth > 10:  # prevent infinite loops
        return path

    if var_name in groups:
        parent_var, prefix = groups[var_name]
        parent_path = _resolve_group_path(parent_var, prefix, groups, depth + 1)
        if parent_path.endswith("/") and path.startswith("/"):
            return parent_path + path[1:]
        return parent_path + path

    return path


def _detect_stdlib_method_restriction(content: str, handler_name: str):
    """Check if a stdlib handler function restricts to a specific HTTP method.

    Looks for patterns like:
        if r.Method != http.MethodPost { ... }
        if r.Method != "POST" { ... }

    Returns the restricted method string, or None if handler accepts all methods.
    """
    # Find the handler function body
    handler_re = re.compile(
        rf'func\s+(?:\(\w+\s+\*?\w+\)\s+)?{re.escape(handler_name)}\s*\(',
        re.MULTILINE,
    )
    match = handler_re.search(content)
    if not match:
        return None

    # Look at first 500 chars of the handler body for method checks
    body_start = match.end()
    body_window = content[body_start:body_start + 500]

    # Pattern: r.Method != http.MethodPost
    const_re = re.compile(r'r\.Method\s*!=\s*http\.Method(\w+)')
    m = const_re.search(body_window)
    if m:
        return m.group(1).upper()

    # Pattern: r.Method != "POST"
    str_re = re.compile(r'r\.Method\s*!=\s*"(\w+)"')
    m = str_re.search(body_window)
    if m:
        return m.group(1).upper()

    return None
