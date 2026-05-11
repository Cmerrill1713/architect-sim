"""Schema-level contract checker: cross-reference caller fields against callee schemas."""
from ..models import Finding, Schema
from ..extractors.go_structs import (
    extract_go_structs,
    associate_schemas_with_endpoints,
    extract_caller_fields,
)


def check_schemas(blueprints: dict, config) -> list:
    """Check request/response field compatibility between callers and callees.

    For each inter-service call, compares the fields the caller sends against
    the fields the callee endpoint expects (from its ShouldBindJSON struct).

    Returns list of Finding objects for schema violations.
    """
    findings = []

    # Phase 1: Extract schemas and endpoint associations for all Go services
    service_schemas = {}    # {service_name: {struct_name: Schema}}
    endpoint_schemas = {}   # {service_name: {endpoint_key: {"request": Schema, "response": Schema}}}

    for name, bp in blueprints.items():
        if bp.language != "go" or not bp.source_dir:
            continue

        schemas = extract_go_structs(bp.source_dir)
        if schemas:
            service_schemas[name] = schemas
            bp.schemas = schemas

        assoc = associate_schemas_with_endpoints(schemas, bp.endpoints)
        if assoc:
            endpoint_schemas[name] = assoc

    # Phase 2: Extract caller fields for each service's outbound calls
    caller_field_maps = {}  # {service_name: {call_key: {...}}}
    for name, bp in blueprints.items():
        if bp.language != "go" or not bp.source_dir or not bp.outbound_calls:
            continue
        caller_fields = extract_caller_fields(bp.source_dir, bp.outbound_calls)
        if caller_fields:
            caller_field_maps[name] = caller_fields

    # Phase 3: Cross-reference caller fields against callee schemas
    for bp in blueprints.values():
        if bp.language != "go":
            continue

        for call in bp.outbound_calls:
            callee = call.callee_service
            callee_base = _base_name(callee)

            # Find the callee's endpoint schema
            callee_ep_schemas = (
                endpoint_schemas.get(callee) or
                endpoint_schemas.get(callee_base)
            )
            if not callee_ep_schemas:
                continue

            ep_key = f"{call.method} {call.path}"
            ep_assoc = callee_ep_schemas.get(ep_key)
            if not ep_assoc or not ep_assoc.get("request"):
                continue

            callee_schema = ep_assoc["request"]

            # Find caller fields for this call
            call_key = f"{call.file_path}:{call.line_number}"
            caller_info = (caller_field_maps.get(bp.name) or {}).get(call_key)
            if not caller_info:
                continue

            # Compare
            mismatches = _find_field_mismatches(
                caller_info, callee_schema, bp.name, callee, call
            )
            findings.extend(mismatches)

    return findings


def _find_field_mismatches(
    caller_info: dict,
    callee_schema: Schema,
    caller_service: str,
    callee_service: str,
    call,
) -> list:
    """Compare caller's sent fields against callee's expected fields.

    Checks for:
    - Missing required fields (callee expects, caller doesn't send)
    - Extra fields (caller sends, callee doesn't expect)
    - Possible renames (case/naming convention mismatches)
    - Type mismatches

    Returns list of Finding objects.
    """
    findings = []

    caller_fields = caller_info.get("fields", {})
    caller_tags = caller_info.get("json_tags", {})
    caller_struct = caller_info.get("struct_name", "?")

    # Build callee's expected fields: json_tag_name -> (field_name, field_type, required)
    callee_expected = {}
    callee_required = set()
    for field_name, tag_value in callee_schema.json_tags.items():
        parts = tag_value.split(",")
        json_name = parts[0]
        is_omitempty = "omitempty" in parts[1:]
        field_type = callee_schema.fields.get(field_name, "")
        callee_expected[json_name] = {
            "field_name": field_name,
            "type": field_type,
            "omitempty": is_omitempty,
        }
        if not is_omitempty:
            callee_required.add(json_name)

    # Build caller's sent fields: json_tag_name -> (field_name, field_type)
    caller_sent = {}
    for field_name, tag_value in caller_tags.items():
        parts = tag_value.split(",")
        json_name = parts[0]
        field_type = caller_fields.get(field_name, "")
        caller_sent[json_name] = {
            "field_name": field_name,
            "type": field_type,
        }

    callee_json_names = set(callee_expected.keys())
    caller_json_names = set(caller_sent.keys())

    # Check 1: Missing required fields
    missing_required = callee_required - caller_json_names
    if missing_required:
        missing_str = ", ".join(sorted(missing_required))
        findings.append(Finding(
            finding_type="field_mismatch",
            severity="warning",
            fixability="needs_context",
            service=caller_service,
            endpoint=f"{call.method} {call.path}",
            file_path=call.file_path,
            line_number=call.line_number,
            details=(
                f"{caller_service} sends {caller_struct} to "
                f"{callee_service} {call.method} {call.path} "
                f"({callee_schema.name}) but is missing required fields: {missing_str}"
            ),
            suggested_fix=f"Add missing fields ({missing_str}) to {caller_struct} or mark them omitempty on the callee",
        ))

    # Check 2: Extra fields (caller sends but callee doesn't expect)
    extra_fields = caller_json_names - callee_json_names
    if extra_fields:
        # Check for possible renames (fuzzy match)
        renames = []
        truly_extra = []
        for extra in sorted(extra_fields):
            rename_target = _find_rename_candidate(extra, callee_json_names - caller_json_names)
            if rename_target:
                renames.append(f"{extra} -> {rename_target}")
            else:
                truly_extra.append(extra)

        if renames:
            rename_str = ", ".join(renames)
            findings.append(Finding(
                finding_type="field_mismatch",
                severity="warning",
                fixability="auto",
                service=caller_service,
                endpoint=f"{call.method} {call.path}",
                file_path=call.file_path,
                line_number=call.line_number,
                details=(
                    f"{caller_service} sends fields with possible naming mismatches "
                    f"to {callee_service} {call.method} {call.path}: {rename_str}"
                ),
                suggested_fix=f"Rename fields to match callee's expected names: {rename_str}",
            ))

        if truly_extra:
            extra_str = ", ".join(truly_extra)
            findings.append(Finding(
                finding_type="field_mismatch",
                severity="info",
                fixability="needs_context",
                service=caller_service,
                endpoint=f"{call.method} {call.path}",
                file_path=call.file_path,
                line_number=call.line_number,
                details=(
                    f"{caller_service} sends extra fields to "
                    f"{callee_service} {call.method} {call.path} "
                    f"that {callee_schema.name} doesn't define: {extra_str}"
                ),
                suggested_fix=f"Remove extra fields ({extra_str}) or add them to {callee_schema.name}",
            ))

    # Check 3: Type mismatches on shared fields
    common = caller_json_names & callee_json_names
    for json_name in sorted(common):
        caller_type = caller_sent[json_name]["type"]
        callee_type = callee_expected[json_name]["type"]
        if caller_type and callee_type and not _types_compatible(caller_type, callee_type):
            findings.append(Finding(
                finding_type="field_mismatch",
                severity="warning",
                fixability="needs_context",
                service=caller_service,
                endpoint=f"{call.method} {call.path}",
                file_path=call.file_path,
                line_number=call.line_number,
                details=(
                    f"Type mismatch on field '{json_name}': "
                    f"{caller_service} sends {caller_type} but "
                    f"{callee_service} expects {callee_type}"
                ),
                suggested_fix=f"Align types for field '{json_name}' between {caller_service} and {callee_service}",
            ))

    return findings


def _find_rename_candidate(field: str, candidates: set) -> str:
    """Find a likely rename candidate for a field name.

    Detects common patterns:
    - camelCase vs snake_case: topK vs top_k
    - Capitalization: TopK vs topK
    """
    field_lower = field.lower().replace("_", "")
    for candidate in candidates:
        candidate_lower = candidate.lower().replace("_", "")
        if field_lower == candidate_lower:
            return candidate
    return ""


def _types_compatible(type_a: str, type_b: str) -> bool:
    """Check if two Go types are compatible.

    Considers interface{} and any as wildcards.
    Considers *T compatible with T.
    """
    # Normalize
    a = type_a.strip().lstrip("*")
    b = type_b.strip().lstrip("*")

    if a == b:
        return True

    # interface{} and any are compatible with everything
    wildcards = {"interface{}", "any"}
    if a in wildcards or b in wildcards:
        return True

    # map[string]interface{} is compatible with map[string]any
    if "interface{}" in a:
        a = a.replace("interface{}", "any")
    if "interface{}" in b:
        b = b.replace("interface{}", "any")
    if a == b:
        return True

    return False


def _base_name(name: str) -> str:
    """Strip language suffix from service name."""
    for suffix in ["-go", "-rust", "-py", "-python", "-service"]:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name
