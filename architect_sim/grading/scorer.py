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
    phantom = type_counts.get("phantom_call", 0)
    missing = type_counts.get("missing_endpoint", 0)
    unknown = type_counts.get("unknown_service", 0)
    report.completeness = max(0, 100 - (phantom * 10) - (missing * 10) - (unknown * 15))

    # 2. Contract Fidelity: field mismatches + method mismatches
    field_mm = type_counts.get("field_mismatch", 0)
    method_mm = type_counts.get("method_mismatch", 0)
    report.contract_fidelity = max(0, 100 - (field_mm * 5) - (method_mm * 10))

    # 3. Resilience: % of outbound calls that have retry + timeout
    total_calls = 0
    resilient_calls = 0
    for bp in blueprints.values():
        for call in bp.outbound_calls:
            total_calls += 1
            if call.retry_config:
                resilient_calls += 1
    report.resilience = (resilient_calls / max(total_calls, 1)) * 100

    failopen_mm = type_counts.get("failopen_mismatch", 0)
    report.resilience = max(0, report.resilience - (failopen_mm * 5))

    # 4. Dependency Hygiene: undeclared deps + phantom deps + circular deps
    undeclared = type_counts.get("undeclared_dep", 0)
    phantom_dep = type_counts.get("phantom_dep", 0)
    circular = type_counts.get("circular_dep", 0)
    report.dependency_hygiene = max(0, 100 - (undeclared * 10) - (phantom_dep * 5) - (circular * 20))

    # 5. Coverage: % of endpoints reachable from entry points
    total_endpoints = sum(len(bp.endpoints) for bp in blueprints.values())
    orphan_count = type_counts.get("orphan_endpoint", 0)
    reachable = total_endpoints - orphan_count
    report.coverage = (reachable / max(total_endpoints, 1)) * 100

    # 6. Security (placeholder -- will be expanded)
    security_findings = type_counts.get("security_issue", 0)
    report.security = max(0, 100 - (security_findings * 15))

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
