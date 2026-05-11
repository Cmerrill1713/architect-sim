"""Score the system across 6 dimensions and produce an overall grade."""
from ..models import GradeReport, Finding


WEIGHTS = {
    "completeness": 0.25,
    "contract_fidelity": 0.25,
    "resilience": 0.20,
    "dependency_hygiene": 0.15,
    "coverage": 0.10,
    "security": 0.05,
}

GRADE_THRESHOLDS = [
    (90, "A"),
    (75, "B"),
    (60, "C"),
    (40, "D"),
    (0, "F"),
]


def score_system(blueprints: dict, findings: list, flow_traces: list, config) -> GradeReport:
    """Score the system across all dimensions.

    Args:
        blueprints: {service_name: ServiceBlueprint}
        findings: All findings from simulation
        flow_traces: Flow traces
        config: Config object

    Returns:
        GradeReport with per-dimension and overall scores
    """
    report = GradeReport()

    # Count findings by type
    type_counts = {}
    for f in findings:
        type_counts[f.finding_type] = type_counts.get(f.finding_type, 0) + 1

    severity_counts = {"critical": 0, "warning": 0, "info": 0}
    for f in findings:
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

    # 1. Completeness: phantom calls + missing endpoints reduce score
    #    Normalized by total call count so large systems aren't unfairly punished.
    #    Floor of 10% if there are ANY valid calls.
    phantom = type_counts.get("phantom_call", 0)
    missing = type_counts.get("missing_endpoint", 0)
    unknown = type_counts.get("unknown_service", 0)
    total_calls = sum(len(bp.outbound_calls) for bp in blueprints.values())
    total_endpoints = sum(len(bp.endpoints) for bp in blueprints.values())
    total_items = max(total_calls + total_endpoints, 1)
    bad_items = phantom + missing + unknown
    completeness_ratio = max(0, 1.0 - (bad_items / total_items))
    report.completeness = completeness_ratio * 100
    # Floor: if there are some correct items, never go below 10
    if total_items > bad_items and report.completeness < 10:
        report.completeness = 10.0

    # 2. Contract Fidelity: field mismatches + method mismatches
    #    Cap penalty per finding to prevent one category driving score to 0.
    field_mm = type_counts.get("field_mismatch", 0)
    method_mm = type_counts.get("method_mismatch", 0)
    fidelity_penalty = min(field_mm * 5, 45) + min(method_mm * 10, 45)
    report.contract_fidelity = max(0, 100 - fidelity_penalty)
    # Floor: if there are any endpoints, never go below 10
    if total_endpoints > 0 and (field_mm + method_mm) < total_endpoints and report.contract_fidelity < 10:
        report.contract_fidelity = 10.0

    # 3. Resilience: % of outbound calls that have retry + timeout
    #    Cap failopen penalty so it can't nuke the whole dimension.
    resilient_calls = 0
    for bp in blueprints.values():
        for call in bp.outbound_calls:
            if call.retry_config:
                resilient_calls += 1
    report.resilience = (resilient_calls / max(total_calls, 1)) * 100

    failopen_mm = type_counts.get("failopen_mismatch", 0)
    report.resilience = max(0, report.resilience - min(failopen_mm * 5, 40))
    # Floor: if any calls have retry config, never go below 10
    if resilient_calls > 0 and report.resilience < 10:
        report.resilience = 10.0

    # 4. Dependency Hygiene: undeclared deps + phantom deps + circular deps
    #    Cap each category so no single issue type can drive to 0.
    undeclared = type_counts.get("undeclared_dep", 0)
    phantom_dep = type_counts.get("phantom_dep", 0)
    circular = type_counts.get("circular_dep", 0)
    hygiene_penalty = min(undeclared * 10, 40) + min(phantom_dep * 5, 30) + min(circular * 20, 40)
    report.dependency_hygiene = max(0, 100 - hygiene_penalty)
    # Floor: if there are services with valid deps, never go below 10
    total_deps = undeclared + phantom_dep + circular
    if len(blueprints) > total_deps and report.dependency_hygiene < 10:
        report.dependency_hygiene = 10.0

    # 5. Coverage: % of endpoints reachable from entry points
    orphan_count = type_counts.get("orphan_endpoint", 0)
    reachable = total_endpoints - orphan_count
    report.coverage = (reachable / max(total_endpoints, 1)) * 100
    # Floor: if any endpoints are reachable, never go below 10
    if reachable > 0 and report.coverage < 10:
        report.coverage = 10.0

    # 6. Security (placeholder -- will be expanded)
    #    Cap so a few findings don't obliterate the dimension.
    security_findings = type_counts.get("security_issue", 0)
    report.security = max(0, 100 - min(security_findings * 15, 80))

    # Overall score (weighted average)
    report.overall = (
        report.completeness * WEIGHTS["completeness"] +
        report.contract_fidelity * WEIGHTS["contract_fidelity"] +
        report.resilience * WEIGHTS["resilience"] +
        report.dependency_hygiene * WEIGHTS["dependency_hygiene"] +
        report.coverage * WEIGHTS["coverage"] +
        report.security * WEIGHTS["security"]
    )

    # Assign letter grade
    for threshold, letter in GRADE_THRESHOLDS:
        if report.overall >= threshold:
            report.grade = letter
            break

    # Details
    report.details = {
        "type_counts": type_counts,
        "severity_counts": severity_counts,
        "total_services": len(blueprints),
        "total_endpoints": total_endpoints,
        "total_calls": total_calls,
        "resilient_calls": resilient_calls,
        "orphan_endpoints": orphan_count,
        "flows_success": all(t.success for t in flow_traces),
    }

    return report
