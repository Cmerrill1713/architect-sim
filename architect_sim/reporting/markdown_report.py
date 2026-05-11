"""Generate human-readable markdown reports."""
import datetime
from ..models import GradeReport, Finding, FlowTrace


def generate_report(grade: GradeReport, findings: list, traces: list, blueprints: dict) -> str:
    """Generate a full markdown report.

    Args:
        grade: GradeReport from scorer
        findings: All findings
        traces: Flow traces
        blueprints: Service blueprints

    Returns:
        Markdown string
    """
    lines = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"# Architecture Simulator Report")
    lines.append(f"Generated: {now}\n")

    # Overall Grade
    lines.append(f"## Overall Grade: {grade.grade} ({grade.overall:.1f}/100)\n")
    lines.append("| Dimension | Score | Weight |")
    lines.append("|-----------|-------|--------|")
    lines.append(f"| Completeness | {grade.completeness:.1f} | 25% |")
    lines.append(f"| Contract Fidelity | {grade.contract_fidelity:.1f} | 25% |")
    lines.append(f"| Resilience | {grade.resilience:.1f} | 20% |")
    lines.append(f"| Dependency Hygiene | {grade.dependency_hygiene:.1f} | 15% |")
    lines.append(f"| Coverage | {grade.coverage:.1f} | 10% |")
    lines.append(f"| Security | {grade.security:.1f} | 5% |")
    lines.append("")

    # Summary Stats
    details = grade.details
    lines.append("## Summary\n")
    lines.append(f"- **Services scanned:** {details.get('total_services', 0)}")
    lines.append(f"- **Endpoints found:** {details.get('total_endpoints', 0)}")
    lines.append(f"- **Inter-service calls:** {details.get('total_calls', 0)}")
    lines.append(f"- **Resilient calls (with retry):** {details.get('resilient_calls', 0)}")
    lines.append(f"- **Orphan endpoints:** {details.get('orphan_endpoints', 0)}")
    lines.append(f"- **Flows healthy:** {'PASS' if details.get('flows_success') else 'FAIL'}")
    lines.append("")

    # Findings by severity
    critical = [f for f in findings if f.severity == "critical"]
    warnings = [f for f in findings if f.severity == "warning"]
    info = [f for f in findings if f.severity == "info"]

    lines.append(f"## Findings: {len(critical)} critical, {len(warnings)} warning, {len(info)} info\n")

    if critical:
        lines.append("### Critical\n")
        for f in critical:
            lines.append(f"- **[{f.finding_type}]** {f.details}")
            if f.file_path:
                lines.append(f"  - Location: `{f.file_path}:{f.line_number}`")
            if f.suggested_fix:
                lines.append(f"  - Fix: {f.suggested_fix}")
            lines.append("")

    if warnings:
        lines.append("### Warnings\n")
        for f in warnings:
            lines.append(f"- **[{f.finding_type}]** {f.details}")
            if f.file_path:
                lines.append(f"  - Location: `{f.file_path}:{f.line_number}`")
            lines.append("")

    if info:
        lines.append("### Info\n")
        for f in info[:20]:
            lines.append(f"- **[{f.finding_type}]** {f.details}")
        if len(info) > 20:
            lines.append(f"\n*... and {len(info) - 20} more info findings*\n")

    # Flow Traces
    if traces:
        lines.append("## Flow Traces\n")
        for trace in traces:
            lines.append(f"### {trace.name} {'PASS' if trace.success else 'FAIL'}\n")
            if trace.steps:
                lines.append("| Caller | Callee | Endpoint | Status |")
                lines.append("|--------|--------|----------|--------|")
                for step in trace.steps:
                    caller = step.get("caller", step.get("service", "?"))
                    callee = step.get("callee", step.get("resolved_service", "?"))
                    endpoint = f"{step.get('method', '')} {step.get('path', '')}"
                    status = step.get("status", "unknown")
                    status_icon = "OK" if status == "ok" else "MISSING" if "missing" in status else "?"
                    lines.append(f"| {caller} | {callee} | `{endpoint}` | {status_icon} |")
            lines.append("")

    # Dependency Graph (Mermaid)
    lines.append("## Service Dependency Graph\n")
    lines.append("```mermaid")
    lines.append("graph LR")

    for name, bp in sorted(blueprints.items()):
        node_id = name.replace("-", "_").replace(".", "_")
        lines.append(f"    {node_id}[{name}]")

    seen_edges = set()
    for bp in blueprints.values():
        src = bp.name.replace("-", "_").replace(".", "_")
        for call in bp.outbound_calls:
            if call.callee_service.startswith("unknown:"):
                continue
            dst = call.callee_service.replace("-", "_").replace(".", "_")
            edge = (src, dst)
            if edge not in seen_edges and src != dst:
                seen_edges.add(edge)
                lines.append(f"    {src} --> {dst}")

    lines.append("```\n")

    # Finding type breakdown
    lines.append("## Finding Types\n")
    type_counts = details.get("type_counts", {})
    if type_counts:
        lines.append("| Type | Count |")
        lines.append("|------|-------|")
        for ftype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {ftype} | {count} |")

    return "\n".join(lines)
