"""Contract verification: check that inter-service calls land on real endpoints."""
from ..models import Finding


def check_contracts(blueprints: dict, config) -> list:
    """Verify all inter-service calls match actual endpoints.

    Args:
        blueprints: {service_name: ServiceBlueprint}
        config: Config object

    Returns:
        List of Finding objects for contract violations
    """
    findings = []

    # Build endpoint index: (service, method, path) -> Endpoint
    endpoint_index = {}
    for bp in blueprints.values():
        for ep in bp.endpoints:
            key = (bp.name, ep.method, ep.path)
            endpoint_index[key] = ep
            normalized = ep.path.rstrip("/") if ep.path != "/" else "/"
            key_norm = (bp.name, ep.method, normalized)
            endpoint_index[key_norm] = ep

    # Check each outbound call
    for bp in blueprints.values():
        for call in bp.outbound_calls:
            callee = call.callee_service

            # Skip calls with unresolved paths
            if call.path == "/" and call.resolution_method in ("discovery", "ports"):
                continue

            # Skip calls to unknown services (reported separately)
            if callee.startswith("unknown:"):
                findings.append(Finding(
                    finding_type="unknown_service",
                    severity="critical",
                    fixability="needs_user",
                    service=bp.name,
                    endpoint=f"{call.method} {call.path}",
                    file_path=call.file_path,
                    line_number=call.line_number,
                    details=f"{bp.name} calls port {call.callee_port} which maps to no known service",
                    suggested_fix=f"Add service definition for port {call.callee_port} to port config, or remove this call",
                ))
                continue

            # Check if callee service exists in blueprints
            if callee not in blueprints:
                findings.append(Finding(
                    finding_type="phantom_call",
                    severity="warning",
                    fixability="needs_context",
                    service=bp.name,
                    endpoint=f"{call.method} {call.path}",
                    file_path=call.file_path,
                    line_number=call.line_number,
                    details=f"{bp.name} calls {callee}{call.path} but {callee} has no extracted blueprint (may not be scanned yet)",
                    suggested_fix=f"Ensure {callee} source directory is included in scan",
                ))
                continue

            # Check if the specific endpoint exists
            path_normalized = call.path.rstrip("/") if call.path != "/" else "/"
            key_exact = (callee, call.method, call.path)
            key_norm = (callee, call.method, path_normalized)

            if key_exact not in endpoint_index and key_norm not in endpoint_index:
                any_method_match = any(
                    (callee, m, call.path) in endpoint_index or (callee, m, path_normalized) in endpoint_index
                    for m in ["GET", "POST", "PUT", "DELETE", "PATCH"]
                )

                if any_method_match:
                    findings.append(Finding(
                        finding_type="method_mismatch",
                        severity="critical",
                        fixability="auto",
                        service=bp.name,
                        endpoint=f"{call.method} {call.path}",
                        file_path=call.file_path,
                        line_number=call.line_number,
                        details=f"{bp.name} calls {call.method} {callee}{call.path} but endpoint exists with different HTTP method",
                        suggested_fix=f"Update the HTTP method in the caller to match the endpoint definition",
                    ))
                else:
                    findings.append(Finding(
                        finding_type="phantom_call",
                        severity="critical",
                        fixability="needs_user",
                        service=bp.name,
                        endpoint=f"{call.method} {call.path}",
                        file_path=call.file_path,
                        line_number=call.line_number,
                        details=f"{bp.name} calls {call.method} {callee}{call.path} but this endpoint does not exist on {callee}",
                        suggested_fix=f"Either create {call.method} {call.path} on {callee}, remove the call, or redirect to the correct endpoint",
                        impact=1,
                    ))

    # Check for orphan endpoints (defined but never called)
    called_endpoints = set()
    for bp in blueprints.values():
        for call in bp.outbound_calls:
            called_endpoints.add((call.callee_service, call.method, call.path))
            path_norm = call.path.rstrip("/") if call.path != "/" else "/"
            called_endpoints.add((call.callee_service, call.method, path_norm))

    SKIP_PATHS = {"/health", "/live", "/ready", "/metrics", "/capabilities", "/debug", "/version"}

    for bp in blueprints.values():
        for ep in bp.endpoints:
            path_norm = ep.path.rstrip("/") if ep.path != "/" else "/"
            if path_norm in SKIP_PATHS or ep.path in SKIP_PATHS:
                continue

            key = (bp.name, ep.method, ep.path)
            key_norm = (bp.name, ep.method, path_norm)

            if key not in called_endpoints and key_norm not in called_endpoints:
                findings.append(Finding(
                    finding_type="orphan_endpoint",
                    severity="info",
                    fixability="needs_context",
                    service=bp.name,
                    endpoint=f"{ep.method} {ep.path}",
                    file_path=ep.file_path,
                    line_number=ep.line_number,
                    details=f"{bp.name} defines {ep.method} {ep.path} but no other service calls it",
                    suggested_fix="Verify this is a public API or remove if dead code",
                ))

    return findings


def check_dependency_drift(blueprints: dict, config) -> list:
    """Compare code-level dependencies vs declared dependencies.

    Finds:
    - Undeclared deps (code calls service not in depends_on)
    - Phantom deps (depends_on lists service that code never calls)
    """
    findings = []

    for bp in blueprints.values():
        actual_deps = set()
        for call in bp.outbound_calls:
            if not call.callee_service.startswith("unknown:"):
                actual_deps.add(call.callee_service)

        declared_deps = set(config.get_declared_deps(bp.name))

        for dep in actual_deps - declared_deps:
            if not any(_fuzzy_match(dep, d) for d in declared_deps):
                findings.append(Finding(
                    finding_type="undeclared_dep",
                    severity="warning",
                    fixability="auto",
                    service=bp.name,
                    details=f"{bp.name} calls {dep} in code but does not declare it in contracts config depends_on",
                    suggested_fix=f"Add '{dep}' to depends_on list for {bp.name}",
                ))

        for dep in declared_deps - actual_deps:
            if not any(_fuzzy_match(dep, a) for a in actual_deps):
                findings.append(Finding(
                    finding_type="phantom_dep",
                    severity="info",
                    fixability="needs_context",
                    service=bp.name,
                    details=f"{bp.name} declares dependency on {dep} but never calls it in code",
                    suggested_fix=f"Remove '{dep}' from depends_on if truly unused, or the call may be indirect",
                ))

    return findings


def check_circular_deps(blueprints: dict) -> list:
    """Detect circular dependencies in the service graph."""
    findings = []

    # Build adjacency list
    graph = {}
    for bp in blueprints.values():
        deps = set()
        for call in bp.outbound_calls:
            if not call.callee_service.startswith("unknown:"):
                deps.add(call.callee_service)
        graph[bp.name] = deps

    # DFS cycle detection
    visited = set()
    rec_stack = set()
    cycles = []

    def dfs(node, path):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor, path)
            elif neighbor in rec_stack:
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)

        path.pop()
        rec_stack.discard(node)

    for node in graph:
        if node not in visited:
            dfs(node, [])

    for cycle in cycles:
        cycle_str = " -> ".join(cycle)
        findings.append(Finding(
            finding_type="circular_dep",
            severity="warning",
            fixability="needs_user",
            service=cycle[0],
            details=f"Circular dependency detected: {cycle_str}",
            suggested_fix="Consider breaking the cycle with async messaging or restructuring",
            impact=len(cycle) - 1,
        ))

    return findings


def _fuzzy_match(a: str, b: str) -> bool:
    """Check if two service names refer to the same service."""
    suffixes = ["-go", "-rust", "-py", "-python", "-service"]
    a_base = a
    b_base = b
    for s in suffixes:
        a_base = a_base.replace(s, "")
        b_base = b_base.replace(s, "")
    return a_base == b_base
