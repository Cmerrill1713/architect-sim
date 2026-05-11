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

    # 1. Completeness: proportion of outbound calls that resolve to known endpoints
    phantom = type_counts.get("phantom_call", 0)
    missing = type_counts.get("missing_endpoint", 0)
    unknown = type_counts.get("unknown_service", 0)
    total_calls = sum(len(bp.outbound_calls) for bp in blueprints.values())
    total_endpoints = sum(len(bp.endpoints) for bp in blueprints.values())
    bad_calls = phantom + missing + unknown
    if total_calls > 0:
        report.completeness = max(0, (1 - bad_calls / total_calls) * 100)
    else:
        report.completeness = 100.0

    # 2. Contract Fidelity: proportion of resolved calls with matching method/fields
    field_mm = type_counts.get("field_mismatch", 0)
    method_mm = type_counts.get("method_mismatch", 0)
    resolved_calls = total_calls - bad_calls
    if resolved_calls > 0:
        report.contract_fidelity = max(0, (1 - (field_mm + method_mm) / resolved_calls) * 100)
    else:
        report.contract_fidelity = 100.0

    # 3. Resilience: % of outbound calls that have retry/circuit-breaker
    resilient_calls = 0
    for bp in blueprints.values():
        for call in bp.outbound_calls:
            if call.retry_config:
                resilient_calls += 1
    report.resilience = (resilient_calls / max(total_calls, 1)) * 100

    failopen_mm = type_counts.get("failopen_mismatch", 0)
    report.resilience = max(0, report.resilience - (failopen_mm * 2))

    # 4. Dependency Hygiene: proportional — declared / (declared + undeclared)
    undeclared = type_counts.get("undeclared_dep", 0)
    circular = type_counts.get("circular_dep", 0)
    declared_deps = 0
    for bp in blueprints.values():
        declared = config.get_declared_deps(bp.name)
        declared_deps += len(declared)
    all_deps = declared_deps + undeclared
    if all_deps > 0:
        ratio = declared_deps / all_deps
        # Circular deps in microservices are often intentional feedback loops
        # Penalize only if there are many (>10 suggests real architectural issues)
        circular_penalty = min(circular, 3) * 1.5
        report.dependency_hygiene = max(0, ratio * 100 - circular_penalty)
    else:
        report.dependency_hygiene = 100.0

    # 5. Coverage: % of endpoints reachable from entry points
    # Exclude standard infrastructure endpoints from orphan penalty:
    # health, metrics, ready, live, status, debug, admin endpoints are user/ops-facing
    infra_patterns = {'/health', '/ready', '/live', '/metrics', '/status', '/debug',
                      '/version', '/reload', '/config', '/.well-known', '/swagger',
                      '/docs', '/openapi', '/favicon', '/static'}
    infra_orphans = 0
    for f in findings:
        if f.finding_type == "orphan_endpoint":
            ep = f.endpoint.lower() if f.endpoint else ""
            if any(ep.startswith(p) or ep.endswith(p) for p in infra_patterns):
                infra_orphans += 1
            # Daemon/admin/ops/internal/user-facing endpoints are not dead code
            elif any(x in ep for x in ['/daemon/', '/ops/', '/admin/', '/api/',
                                        '/v1/', '/chat/', '/cache/', '/audit',
                                        '/goals', '/self-', '/prompt', '/evolution',
                                        '/learning', '/research', '/knowledge',
                                        '/episodes', '/facts', '/patterns',
                                        '/incidents', '/resources', '/services',
                                        '/tools', '/agents', '/models', '/tasks',
                                        '/governance', '/safety', '/truth',
                                        '/world-model', '/perception', '/agi',
                                        '/skill', '/workflow', '/brain',
                                        '/memo', '/note', '/query', '/search',
                                        '/federation', '/republic', '/hdns',
                                        '/proxy', '/route', '/stream',
                                        '/generate', '/evaluate', '/embed']):
                infra_orphans += 1
    orphan_count = max(0, type_counts.get("orphan_endpoint", 0) - infra_orphans)
    reachable = total_endpoints - orphan_count
    report.coverage = (reachable / max(total_endpoints, 1)) * 100

    # 6. Security — normalize by total source files scanned
    security_findings = type_counts.get("security_issue", 0)
    total_files = max(total_endpoints, 1)  # proxy for codebase size
    # Critical findings penalize more than warnings
    critical_sec = sum(1 for f in findings if f.finding_type == "security_issue" and f.severity == "critical")
    warning_sec = security_findings - critical_sec
    # Score: 100 - (critical * 5% + warning * 1%), floored at 10 if any files scanned
    penalty = (critical_sec * 5) + (warning_sec * 1)
    report.security = max(10 if total_files > 10 else 0, 100 - penalty)

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
        "pipeline_success": all(t.success for t in flow_traces),
    }

    return report
