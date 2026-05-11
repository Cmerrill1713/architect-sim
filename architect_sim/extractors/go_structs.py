"""Extract Go struct definitions and associate them with HTTP endpoints."""
import re
from pathlib import Path
from ..models import Schema

# Struct definition (handles multi-line bodies)
STRUCT_RE = re.compile(
    r'type\s+(\w+)\s+struct\s*\{([^}]*)\}',
    re.MULTILINE | re.DOTALL,
)

# Struct field with json tag
FIELD_RE = re.compile(
    r'(\w+)\s+([\w\[\]*.\{\}]+(?:\s*\{[^}]*\})?)\s+`[^`]*json:"([^"]+)"[^`]*`'
)

# Handler binding (gin / echo / stdlib)
BIND_RE = re.compile(
    r'(?:ShouldBindJSON|BindJSON|ShouldBind|Bind)\(\s*&(\w+)\s*\)'
)

# JSON response: c.JSON(status, variable)
RESPONSE_RE = re.compile(
    r'c\.JSON\(\s*\w+\s*,\s*(\w+)\s*\)'
)

# json.Marshal for outbound calls
MARSHAL_RE = re.compile(
    r'json\.Marshal\(\s*(?:&)?(\w+)\s*\)'
)

# Variable declaration with type: var req SomeRequest
VAR_DECL_RE = re.compile(
    r'var\s+(\w+)\s+(\w+)'
)

# Short variable declaration with struct literal: req := SomeRequest{...}
SHORT_DECL_RE = re.compile(
    r'(\w+)\s*:=\s*(\w+)\s*\{'
)

# Struct literal field assignments: FieldName: value
LITERAL_FIELD_RE = re.compile(
    r'(\w+)\s*:'
)


def extract_go_structs(source_dir: str) -> dict:
    """Extract all Go struct definitions from a service.

    Returns: {struct_name: Schema}
    """
    schemas = {}
    source_path = Path(source_dir)

    for go_file in sorted(source_path.rglob("*.go")):
        if go_file.name.endswith("_test.go") or "vendor" in str(go_file):
            continue

        try:
            content = go_file.read_text(errors="replace")
        except (OSError, IOError):
            continue

        file_str = str(go_file)

        for match in STRUCT_RE.finditer(content):
            struct_name = match.group(1)
            body = match.group(2)
            line_num = content[:match.start()].count("\n") + 1

            fields = {}
            json_tags = {}

            for field_match in FIELD_RE.finditer(body):
                field_name = field_match.group(1)
                field_type = field_match.group(2).strip()
                tag_value = field_match.group(3)

                # Parse json tag: "name,omitempty" -> tag_name="name"
                tag_parts = tag_value.split(",")
                tag_name = tag_parts[0]
                if tag_name == "-":
                    continue  # Skip json:"-" fields

                fields[field_name] = field_type
                json_tags[field_name] = tag_value  # Keep full tag including omitempty

            if fields:
                schemas[struct_name] = Schema(
                    name=struct_name,
                    fields=fields,
                    json_tags=json_tags,
                    file_path=file_str,
                    line_number=line_num,
                )

    return schemas


def associate_schemas_with_endpoints(schemas: dict, endpoints: list) -> dict:
    """Associate request/response schemas with endpoints.

    Returns: {endpoint_key: {"request": Schema|None, "response": Schema|None}}

    Uses both naming conventions and handler body analysis.
    """
    associations = {}

    # Build handler -> endpoint map
    handler_endpoints = {}
    for ep in endpoints:
        if ep.handler_name:
            # Strip receiver prefix: s.handleChat -> handleChat
            handler_base = ep.handler_name.split(".")[-1]
            handler_endpoints[handler_base] = ep

    # Strategy 1: Analyze handler bodies for ShouldBindJSON / c.JSON
    handler_schemas = _analyze_handler_bodies(endpoints, schemas)

    # Strategy 2: Naming convention matching
    convention_schemas = _match_by_naming(schemas, handler_endpoints)

    # Merge: handler body analysis takes priority
    for ep in endpoints:
        ep_key = f"{ep.method} {ep.path}"
        handler_base = ep.handler_name.split(".")[-1] if ep.handler_name else ""

        req_schema = None
        resp_schema = None

        # From handler body analysis
        if handler_base in handler_schemas:
            req_schema = handler_schemas[handler_base].get("request")
            resp_schema = handler_schemas[handler_base].get("response")

        # Fill gaps from naming conventions
        if handler_base in convention_schemas:
            if req_schema is None:
                req_schema = convention_schemas[handler_base].get("request")
            if resp_schema is None:
                resp_schema = convention_schemas[handler_base].get("response")

        if req_schema or resp_schema:
            associations[ep_key] = {
                "request": req_schema,
                "response": resp_schema,
            }

    return associations


def _analyze_handler_bodies(endpoints: list, schemas: dict) -> dict:
    """Analyze handler function bodies to find request/response types.

    Looks for ShouldBindJSON(&req) and c.JSON(200, resp) patterns,
    then resolves the variable types to schema names.
    """
    results = {}

    # Group endpoints by file
    files = {}
    for ep in endpoints:
        if ep.file_path and ep.handler_name:
            files.setdefault(ep.file_path, []).append(ep)

    for file_path, eps in files.items():
        try:
            content = Path(file_path).read_text(errors="replace")
        except (OSError, IOError):
            continue

        # Build variable type map for entire file
        var_types = {}
        for m in VAR_DECL_RE.finditer(content):
            var_types[m.group(1)] = m.group(2)
        for m in SHORT_DECL_RE.finditer(content):
            var_types[m.group(1)] = m.group(2)

        for ep in eps:
            handler_base = ep.handler_name.split(".")[-1]

            # Find the handler function body
            handler_body = _extract_handler_body(content, handler_base)
            if not handler_body:
                continue

            # Build local variable type map
            local_vars = dict(var_types)  # start with file-level
            for m in VAR_DECL_RE.finditer(handler_body):
                local_vars[m.group(1)] = m.group(2)
            for m in SHORT_DECL_RE.finditer(handler_body):
                local_vars[m.group(1)] = m.group(2)

            req_schema = None
            resp_schema = None

            # Find request schema via ShouldBindJSON(&varName)
            bind_match = BIND_RE.search(handler_body)
            if bind_match:
                var_name = bind_match.group(1)
                type_name = local_vars.get(var_name)
                if type_name and type_name in schemas:
                    req_schema = schemas[type_name]

            # Find response schema via c.JSON(200, varName)
            resp_match = RESPONSE_RE.search(handler_body)
            if resp_match:
                var_name = resp_match.group(1)
                type_name = local_vars.get(var_name)
                if type_name and type_name in schemas:
                    resp_schema = schemas[type_name]

            if req_schema or resp_schema:
                results[handler_base] = {
                    "request": req_schema,
                    "response": resp_schema,
                }

    return results


def _extract_handler_body(content: str, handler_name: str) -> str:
    """Extract the body of a handler function by name.

    Finds `func ... handlerName(` and returns text until the matching
    closing brace.
    """
    # Match function definition: func (s *Server) handlerName(...)
    # or: func handlerName(...)
    pattern = re.compile(
        r'func\s+(?:\([^)]*\)\s+)?' + re.escape(handler_name) + r'\s*\([^)]*\)\s*(?:\([^)]*\)\s*)?\{',
    )
    match = pattern.search(content)
    if not match:
        return ""

    # Find matching closing brace
    start = match.end()
    depth = 1
    pos = start
    while pos < len(content) and depth > 0:
        ch = content[pos]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        pos += 1

    return content[start:pos]


def _match_by_naming(schemas: dict, handler_endpoints: dict) -> dict:
    """Match schemas to handlers by naming convention.

    Patterns:
        ChatRequest / ChatResponse -> handler handleChat or chat
        ProposePlanRequest -> handler proposePlan
    """
    results = {}

    for handler_name in handler_endpoints:
        # Normalize: handleChat -> Chat, proposePlan -> ProposePlan
        base = handler_name
        if base.startswith("handle"):
            base = base[6:]  # strip "handle"
        if base.startswith("Handle"):
            base = base[6:]

        # Capitalize first letter for struct name matching
        if base:
            base_cap = base[0].upper() + base[1:]
        else:
            continue

        req_schema = None
        resp_schema = None

        # Try {Base}Request, {Base}Req
        for suffix in ["Request", "Req"]:
            candidate = base_cap + suffix
            if candidate in schemas:
                req_schema = schemas[candidate]
                break

        # Try {Base}Response, {Base}Resp
        for suffix in ["Response", "Resp"]:
            candidate = base_cap + suffix
            if candidate in schemas:
                resp_schema = schemas[candidate]
                break

        if req_schema or resp_schema:
            results[handler_name] = {
                "request": req_schema,
                "response": resp_schema,
            }

    return results


def extract_caller_fields(source_dir: str, calls: list) -> dict:
    """For each outbound call, find what fields the caller sends.

    Looks for json.Marshal() or struct literal near the call site.
    Returns: {call_key: {"fields": {field_name: field_type}, "struct_name": str}}

    call_key is "{file_path}:{line_number}"
    """
    results = {}
    source_path = Path(source_dir)

    # Pre-extract all structs for type resolution
    schemas = extract_go_structs(source_dir)

    # Index calls by file
    calls_by_file = {}
    for call in calls:
        calls_by_file.setdefault(call.file_path, []).append(call)

    for file_path, file_calls in calls_by_file.items():
        try:
            content = Path(file_path).read_text(errors="replace")
        except (OSError, IOError):
            continue

        lines = content.split("\n")

        # Build variable type map for the file
        var_types = {}
        for m in VAR_DECL_RE.finditer(content):
            var_types[m.group(1)] = m.group(2)
        for m in SHORT_DECL_RE.finditer(content):
            var_types[m.group(1)] = m.group(2)

        for call in file_calls:
            call_key = f"{call.file_path}:{call.line_number}"

            # Search a window around the call site for json.Marshal or struct literals
            window_start = max(0, call.line_number - 30)
            window_end = min(len(lines), call.line_number + 10)
            window = "\n".join(lines[window_start:window_end])

            # Strategy 1: json.Marshal(&varName) or json.Marshal(varName)
            marshal_match = MARSHAL_RE.search(window)
            if marshal_match:
                var_name = marshal_match.group(1)
                type_name = var_types.get(var_name)
                if type_name and type_name in schemas:
                    schema = schemas[type_name]
                    results[call_key] = {
                        "fields": dict(schema.fields),
                        "struct_name": type_name,
                        "json_tags": dict(schema.json_tags),
                    }
                    continue

            # Strategy 2: struct literal: TypeName{Field1: val, Field2: val}
            for m in SHORT_DECL_RE.finditer(window):
                type_name = m.group(2)
                if type_name in schemas:
                    # Extract literal fields from the struct literal
                    literal_start = m.end() - 1  # at the {
                    literal_body = _extract_brace_content(window, literal_start)
                    if literal_body:
                        literal_fields = {}
                        for fm in LITERAL_FIELD_RE.finditer(literal_body):
                            field_name = fm.group(1)
                            if field_name in schemas[type_name].fields:
                                literal_fields[field_name] = schemas[type_name].fields[field_name]
                        if literal_fields:
                            results[call_key] = {
                                "fields": literal_fields,
                                "struct_name": type_name,
                                "json_tags": {
                                    k: schemas[type_name].json_tags[k]
                                    for k in literal_fields
                                    if k in schemas[type_name].json_tags
                                },
                            }
                    break

    return results


def _extract_brace_content(text: str, start: int) -> str:
    """Extract content between { and matching } starting at position start."""
    if start >= len(text) or text[start] != '{':
        return ""
    depth = 0
    pos = start
    while pos < len(text):
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:pos]
        pos += 1
    return ""
