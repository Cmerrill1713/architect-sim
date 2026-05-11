"""Contract verification: check that inter-service calls land on real endpoints."""
import re
from ..models import Finding

# Infrastructure ports that are NOT service dependencies — these are databases,
# message brokers, model servers, and other shared infrastructure.
INFRASTRUCTURE_PORTS = frozenset({
    5432,   # postgres
    6379,   # redis
    4222,   # nats
    9090,   # prometheus
    11434,  # ollama
    18088,  # internal llama-server
    18100,  # internal backend
    18200,  # internal backend
})


def check_contracts(blueprints: dict, config) -> list:
    """Verify all inter-service calls match actual endpoints."""
    findings = []

    # Build endpoint index with fuzzy service name matching
    # Index by both exact name AND base name (strip -go/-rust/-py)
    endpoint_index = {}
    for bp in blueprints.values():
        for ep in bp.endpoints:
            # Exact service name
            key = (bp.name, ep.method, ep.path)
            endpoint_index[key] = ep
            normalized = ep.path.rstrip("/") if ep.path != "/" else "/"
            endpoint_index[(bp.name, ep.method, normalized)] = ep
            # Base name (strip language suffix)
            base = _base_name(bp.name)
            if base != bp.name:
                endpoint_index[(base, ep.method, ep.path)] = ep
                endpoint_index[(base, ep.method, normalized)] = ep

    # Build name->blueprint lookup with fuzzy matching
    bp_lookup = {}
    for name, bp in blueprints.items():
        bp_lookup[name] = bp
        bp_lookup[_base_name(name)] = bp

    # Check each outbound call
    for bp in blueprints.values():
        for call in bp.outbound_calls:
            callee = call.callee_service
            path = call.path

            # Skip root path "/" calls — these are base URL references,
            # not real endpoint calls
            if path == "/":
                continue

            # Skip base URL artifacts: "servicename/" with no real path
            if path.endswith("/") and "/" not in path.rstrip("/"):
                continue

            # Skip calls to infrastructure ports (databases, brokers, etc.)
            if call.callee_port in INFRASTRUCTURE_PORTS:
                continue

            # Skip calls to unknown services (reported separately)
            if callee.startswith("unknown:"):
                findings.append(Finding(
                    finding_type="unknown_service",
                    severity="critical",
                    fixability="needs_user",
                    service=bp.name,
                    endpoint=f"{call.method} {path}",
                    file_path=call.file_path,
                    line_number=call.line_number,
                    details=f"{bp.name} calls port {call.callee_port} which maps to no known service",
                    suggested_fix=f"Add service definition for port {call.callee_port} to port config, or remove this call",
                ))
                continue

            # Resolve callee with fuzzy matching
            resolved_bp = bp_lookup.get(callee) or bp_lookup.get(_base_name(callee))

            if resolved_bp is None:
                findings.append(Finding(
                    finding_type="phantom_call",
                    severity="warning",
                    fixability="needs_context",
                    service=bp.name,
                    endpoint=f"{call.method} {path}",
                    file_path=call.file_path,
                    line_number=call.line_number,
                    details=f"{bp.name} calls {callee}{path} but {callee} has no extracted blueprint",
                    suggested_fix=f"Ensure {callee} source directory is included in scan",
                ))
                continue

            # Use the resolved service name for endpoint lookup
            resolved_name = resolved_bp.name
            callee_base = _base_name(callee)

            # Strip query params for matching: /facts?limit=1 -> /facts
            path_no_query = path.split("?")[0] if "?" in path else path
            path_normalized = path_no_query.rstrip("/") if path_no_query != "/" else "/"

            # Check if the specific endpoint exists (try exact, base, and fuzzy)
            found = False
            for svc_key in [resolved_name, callee_base, callee]:
                for p in [path, path_no_query, path_normalized]:
                    if (svc_key, call.method, p) in endpoint_index:
                        found = True
                        break
                if found:
                    break

            if not found:
                # Check if endpoint exists with different method
                any_method_match = False
                for m in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    for svc_key in [resolved_name, callee_base]:
                        for p in [path, path_no_query, path_normalized]:
                            if (svc_key, m, p) in endpoint_index:
                                any_method_match = True
                                break
                        if any_method_match:
                            break
                    if any_method_match:
                        break

                if any_method_match:
                    findings.append(Finding(
                        finding_type="method_mismatch",
                        severity="critical",
                        fixability="auto",
                        service=bp.name,
                        endpoint=f"{call.method} {path}",
                        file_path=call.file_path,
                        line_number=call.line_number,
                        details=f"{bp.name} calls {call.method} {resolved_name}{path} but endpoint exists with different HTTP method",
                        suggested_fix=f"Update the HTTP method in the caller to match the endpoint definition",
                    ))
                else:
                    findings.append(Finding(
                        finding_type="phantom_call",
                        severity="critical",
                        fixability="needs_user",
                        service=bp.name,
                        endpoint=f"{call.method} {path}",
                        file_path=call.file_path,
                        line_number=call.line_number,
                        details=f"{bp.name} calls {call.method} {resolved_name}{path} but endpoint not found on {resolved_name}",
                        suggested_fix=f"Either create {call.method} {path} on {resolved_name}, remove the call, or redirect",
                        impact=1,
                    ))

    # Check for orphan endpoints
    # Build set of all called endpoints (with fuzzy service names + query param stripping)
    called_endpoints = set()
    for bp in blueprints.values():
        for call in bp.outbound_calls:
            callee = call.callee_service
            callee_base = _base_name(callee)
            path_no_query = call.path.split("?")[0] if "?" in call.path else call.path
            path_norm = path_no_query.rstrip("/") if path_no_query != "/" else "/"
            for svc in [callee, callee_base]:
                for p in [call.path, path_no_query, path_norm]:
                    called_endpoints.add((svc, call.method, p))
                    # Also add with any method (some calls have wrong method detection)
                    for m in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                        called_endpoints.add((svc, m, p))

    # Paths that are typically public APIs / infrastructure, not inter-service
    SKIP_PATHS = {"/health", "/live", "/ready", "/metrics", "/capabilities",
                  "/debug", "/version", "/swagger", "/docs", "/openapi.json"}
    SKIP_PREFIXES = ("/health", "/live", "/ready", "/metrics", "/debug",
                     "/ops/", "/api/health", "/api/v1/", "/v1/", "/swagger", "/docs")

    for bp in blueprints.values():
        base = _base_name(bp.name)
        for ep in bp.endpoints:
            path_norm = ep.path.rstrip("/") if ep.path != "/" else "/"

            # Skip standard infra/public endpoints
            if path_norm in SKIP_PATHS or ep.path in SKIP_PATHS:
                continue
            if any(path_norm.startswith(p) for p in SKIP_PREFIXES):
                continue

            # Check if called (by exact name or base name)
            is_called = False
            for svc in [bp.name, base]:
                if (svc, ep.method, ep.path) in called_endpoints:
                    is_called = True
                    break
                if (svc, ep.method, path_norm) in called_endpoints:
                    is_called = True
                    break

            if not is_called:
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

    Only reports for services that HAVE contract entries.
    Services with no contract entry are skipped — you can't have
    "undeclared deps" if you have no declarations at all.
    """
    findings = []

    for bp in blueprints.values():
        # Skip services with no contract entry
        declared_deps = set(config.get_declared_deps(bp.name))
        if not declared_deps and bp.name not in config.contracts:
            continue

        # Actual deps from code (excluding infrastructure)
        actual_deps = set()
        for call in bp.outbound_calls:
            if not call.callee_service.startswith("unknown:"):
                if call.callee_port in INFRASTRUCTURE_PORTS:
                    continue
                actual_deps.add(call.callee_service)

        # Undeclared: in code but not in YAML
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

        # Phantom declared: in YAML but not in code
        for dep in declared_deps - actual_deps:
            if not any(_fuzzy_match(dep, a) for a in actual_deps):
                findings.append(Finding(
                    finding_type="phantom_dep",
                    severity="info",
                    fixability="needs_context",
                    service=bp.name,
                    details=f"{bp.name} declares dependency on {dep} but never calls it in code",
                    suggested_fix=f"Remove '{dep}' from depends_on if truly unused",
                ))

    return findings


def check_circular_deps(blueprints: dict) -> list:
    """Detect circular dependencies in the service graph."""
    findings = []

    # Build adjacency list (skip self-references and infrastructure)
    graph = {}
    for bp in blueprints.values():
        deps = set()
        for call in bp.outbound_calls:
            callee = call.callee_service
            if not callee.startswith("unknown:"):
                # Skip self-references
                if callee == bp.name or _base_name(callee) == _base_name(bp.name):
                    continue
                # Skip infrastructure ports
                if call.callee_port in INFRASTRUCTURE_PORTS:
                    continue
                deps.add(callee)
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

    # Deduplicate cycles (same cycle can be found starting from different nodes)
    seen_cycles = set()
    for cycle in cycles:
        unique = set(cycle)
        if len(unique) <= 1:
            continue  # Self-reference, already filtered
        cycle_key = frozenset(unique)
        if cycle_key in seen_cycles:
            continue
        seen_cycles.add(cycle_key)

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


def _base_name(name: str) -> str:
    """Strip language suffix from service name: 'rag-gateway-rust' -> 'rag-gateway'."""
    for suffix in ["-go", "-rust", "-py", "-python", "-service"]:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def _fuzzy_match(a: str, b: str) -> bool:
    """Check if two service names refer to the same service."""
    return _base_name(a) == _base_name(b)
