"""Auto-discover and trace request flows through the service dependency graph.

Instead of hardcoded pipeline phases, this module:
1. Identifies entry points (services with no inbound calls)
2. Traces all reachable paths from each entry point
3. Verifies each call in every path lands on a real endpoint
"""
from ..models import FlowTrace, Finding


def discover_flows(blueprints: dict, config) -> list:
    """Auto-discover request flows by finding entry points and tracing paths.

    Entry points are services that have endpoints but no inbound calls
    from other services (i.e., they are the first point of contact).

    Returns:
        list of (FlowTrace, list[Finding]) tuples
    """
    results = []

    # Build inbound call map: which services receive calls?
    inbound = set()
    for bp in blueprints.values():
        for call in bp.outbound_calls:
            if not call.callee_service.startswith("unknown:"):
                inbound.add(call.callee_service)

    # Entry points: services with endpoints but not called by other services
    entry_points = []
    for name, bp in blueprints.items():
        if bp.endpoints and name not in inbound:
            entry_points.append(name)

    # If no clear entry points found, use services with the most endpoints
    if not entry_points:
        by_endpoints = sorted(blueprints.items(), key=lambda x: len(x[1].endpoints), reverse=True)
        entry_points = [name for name, bp in by_endpoints[:3] if bp.endpoints]

    # Trace from each entry point
    for entry in entry_points:
        trace, findings = _trace_from_entry(entry, blueprints, config)
        results.append((trace, findings))

    return results


def trace_all_flows(blueprints: dict, config) -> tuple:
    """Trace all auto-discovered flows.

    Returns:
        (list[FlowTrace], list[Finding])
    """
    all_traces = []
    all_findings = []

    flow_results = discover_flows(blueprints, config)
    for trace, findings in flow_results:
        all_traces.append(trace)
        all_findings.extend(findings)

    return all_traces, all_findings


def _trace_from_entry(entry_service: str, blueprints: dict, config, max_depth: int = 10) -> tuple:
    """Trace all reachable paths from an entry point service.

    Returns:
        (FlowTrace, list[Finding])
    """
    trace = FlowTrace(name=f"flow_from_{entry_service}")
    findings = []
    visited = set()

    def _trace_service(service_name: str, depth: int):
        if depth > max_depth or service_name in visited:
            return
        visited.add(service_name)

        bp = blueprints.get(service_name)
        if bp is None:
            return

        for call in bp.outbound_calls:
            callee = call.callee_service
            if callee.startswith("unknown:"):
                continue

            step = {
                "caller": service_name,
                "callee": callee,
                "method": call.method,
                "path": call.path,
                "status": "unknown",
            }

            # Check if callee exists
            callee_bp = blueprints.get(callee)
            if callee_bp is None:
                step["status"] = "missing_service"
                trace.steps.append(step)
                continue

            # Check if endpoint exists on callee
            endpoint_found = any(
                ep.method == call.method and _path_matches(ep.path, call.path)
                for ep in callee_bp.endpoints
            )

            if endpoint_found:
                step["status"] = "ok"
            else:
                # Check for any method match
                any_path_match = any(
                    _path_matches(ep.path, call.path)
                    for ep in callee_bp.endpoints
                )
                if any_path_match:
                    step["status"] = "method_mismatch"
                else:
                    step["status"] = "endpoint_missing"
                    if call.path != "/":
                        findings.append(Finding(
                            finding_type="phantom_call",
                            severity="warning",
                            fixability="needs_context",
                            service=service_name,
                            endpoint=f"{call.method} {call.path}",
                            details=f"Flow from {trace.name}: {service_name} calls {call.method} {callee}{call.path} but endpoint not found",
                        ))

            trace.steps.append(step)

            # Continue tracing into the callee
            _trace_service(callee, depth + 1)

    _trace_service(entry_service, 0)

    trace.success = all(
        s.get("status") in ("ok", "missing_service") for s in trace.steps
    )

    return trace, findings


def _path_matches(actual: str, pattern: str) -> bool:
    """Check if an actual endpoint path matches a pattern.

    Handles parameterized paths like /tools/{toolName}/execute
    matching /tools/:toolName/execute (gin) or /tools/{toolName}/execute (mux).
    """
    if actual == pattern:
        return True

    a = actual.rstrip("/") if actual != "/" else "/"
    p = pattern.rstrip("/") if pattern != "/" else "/"
    if a == p:
        return True

    # Handle gin :param and mux {param} patterns
    a_parts = a.split("/")
    p_parts = p.split("/")

    if len(a_parts) != len(p_parts):
        return False

    for ap, pp in zip(a_parts, p_parts):
        if ap == pp:
            continue
        if ap.startswith(":") or ap.startswith("{"):
            continue
        if pp.startswith(":") or pp.startswith("{"):
            continue
        if ap.startswith("*") or pp.startswith("*"):
            continue
        return False

    return True
