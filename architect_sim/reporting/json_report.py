"""Generate machine-readable JSON reports."""
import json
from ..models import GradeReport, Finding, FlowTrace


def generate_json(grade: GradeReport, findings: list, traces: list, blueprints: dict) -> str:
    """Generate JSON report.

    Returns:
        JSON string
    """
    report = {
        "grade": {
            "overall": round(grade.overall, 2),
            "letter": grade.grade,
            "dimensions": {
                "completeness": round(grade.completeness, 2),
                "contract_fidelity": round(grade.contract_fidelity, 2),
                "resilience": round(grade.resilience, 2),
                "dependency_hygiene": round(grade.dependency_hygiene, 2),
                "coverage": round(grade.coverage, 2),
                "security": round(grade.security, 2),
            },
            "details": grade.details,
        },
        "findings": [
            {
                "type": f.finding_type,
                "severity": f.severity,
                "fixability": f.fixability,
                "service": f.service,
                "endpoint": f.endpoint,
                "file_path": f.file_path,
                "line_number": f.line_number,
                "details": f.details,
                "suggested_fix": f.suggested_fix,
                "impact": f.impact,
            }
            for f in findings
        ],
        "traces": [
            {
                "name": t.name,
                "success": t.success,
                "steps": t.steps,
            }
            for t in traces
        ],
        "blueprints": {
            name: {
                "port": bp.port,
                "language": bp.language,
                "endpoint_count": len(bp.endpoints),
                "call_count": len(bp.outbound_calls),
                "endpoints": [
                    {"method": ep.method, "path": ep.path, "handler": ep.handler_name}
                    for ep in bp.endpoints
                ],
                "calls": [
                    {"callee": c.callee_service, "method": c.method, "path": c.path}
                    for c in bp.outbound_calls
                ],
            }
            for name, bp in sorted(blueprints.items())
        },
        "summary": {
            "services_scanned": len(blueprints),
            "total_endpoints": sum(len(bp.endpoints) for bp in blueprints.values()),
            "total_calls": sum(len(bp.outbound_calls) for bp in blueprints.values()),
            "findings_critical": sum(1 for f in findings if f.severity == "critical"),
            "findings_warning": sum(1 for f in findings if f.severity == "warning"),
            "findings_info": sum(1 for f in findings if f.severity == "info"),
        },
    }

    return json.dumps(report, indent=2, default=str)
