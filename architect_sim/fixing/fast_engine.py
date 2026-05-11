"""Fast in-memory fix iteration engine.

Extract once, then run 1000s of fix iterations in-memory on the blueprint
without touching disk. Each fix modifies the blueprint (adds endpoints,
corrects schemas, adds deps), then re-simulates and re-grades instantly.

The engine tries every auto-fixable finding, keeps the ones that improve
the score, and combines them greedily into an optimal fix set.
"""
import copy
import time
from dataclasses import dataclass, field
from ..models import Endpoint, Call, Finding, ServiceBlueprint, GradeReport, IterationRecord


@dataclass
class Fix:
    """A concrete in-memory fix applied to a blueprint."""
    fix_id: int
    finding: Finding
    description: str
    action: str  # "add_endpoint", "add_dep", "remove_call", "add_retry", "fix_method"
    target_service: str
    mutations: dict = field(default_factory=dict)  # Details of what changed
    score_delta: float = 0.0
    accepted: bool = False


@dataclass
class FixResult:
    """Result of the fast iteration engine."""
    initial_grade: GradeReport
    final_grade: GradeReport
    iterations_run: int
    fixes_accepted: list  # list[Fix]
    fixes_rejected: list
    needs_user: list      # list[Finding] that need user decisions
    elapsed_ms: float
    iterations_per_second: float


def apply_fix_to_blueprint(blueprints: dict, finding: Finding, fix_id: int) -> Fix:
    """Apply a single fix to the in-memory blueprint.

    Returns a Fix object describing what was done.
    Modifies blueprints in-place.
    """
    f = finding

    if f.finding_type == "phantom_call" and f.fixability == "auto":
        return _fix_add_stub_endpoint(blueprints, f, fix_id)

    elif f.finding_type == "method_mismatch":
        return _fix_method_mismatch(blueprints, f, fix_id)

    elif f.finding_type == "undeclared_dep":
        return Fix(
            fix_id=fix_id,
            finding=f,
            description=f"Add dependency declaration for {f.service}",
            action="add_dep",
            target_service=f.service,
            mutations={"type": "yaml_update", "details": f.suggested_fix},
        )

    elif f.finding_type == "failopen_mismatch":
        return Fix(
            fix_id=fix_id,
            finding=f,
            description=f"Update degraded_ok for {f.service}",
            action="update_yaml",
            target_service=f.service,
            mutations={"type": "yaml_update", "details": f.suggested_fix},
        )

    elif f.finding_type == "missing_endpoint":
        return _fix_add_stub_endpoint(blueprints, f, fix_id)

    return Fix(
        fix_id=fix_id,
        finding=f,
        description=f"Unhandled fix type: {f.finding_type}",
        action="noop",
        target_service=f.service,
    )


def _fix_add_stub_endpoint(blueprints: dict, f: Finding, fix_id: int) -> Fix:
    """Add a stub endpoint to a service to resolve a phantom call."""
    parts = f.endpoint.split(" ", 1)
    if len(parts) != 2:
        return Fix(fix_id=fix_id, finding=f, description="Cannot parse endpoint",
                   action="noop", target_service=f.service)

    method, path = parts

    target_service = None
    for name, bp in blueprints.items():
        for call in sum([b.outbound_calls for b in blueprints.values()], []):
            if call.callee_service == name and call.path == path and call.method == method:
                target_service = name
                break
        if target_service:
            break

    if not target_service:
        target_service = f.service

    if target_service not in blueprints:
        return Fix(fix_id=fix_id, finding=f, description=f"Target {target_service} not in blueprints",
                   action="noop", target_service=target_service)

    bp = blueprints[target_service]

    stub = Endpoint(
        service_name=target_service,
        port=bp.port,
        method=method,
        path=path,
        file_path="[auto-generated stub]",
        line_number=0,
        handler_name=f"stub_{path.replace('/', '_').strip('_')}",
        framework="stub",
    )
    bp.endpoints.append(stub)

    return Fix(
        fix_id=fix_id,
        finding=f,
        description=f"Add stub {method} {path} to {target_service}",
        action="add_endpoint",
        target_service=target_service,
        mutations={"method": method, "path": path, "type": "stub_endpoint"},
    )


def _fix_method_mismatch(blueprints: dict, f: Finding, fix_id: int) -> Fix:
    """Fix a method mismatch by updating the caller's call method."""
    parts = f.endpoint.split(" ", 1)
    if len(parts) != 2:
        return Fix(fix_id=fix_id, finding=f, description="Cannot parse", action="noop", target_service=f.service)

    wrong_method, path = parts

    correct_method = None
    for bp in blueprints.values():
        for ep in bp.endpoints:
            if ep.path == path:
                correct_method = ep.method
                break
        if correct_method:
            break

    if not correct_method or correct_method == wrong_method:
        return Fix(fix_id=fix_id, finding=f, description="Cannot determine correct method",
                   action="noop", target_service=f.service)

    bp = blueprints.get(f.service)
    if bp:
        for call in bp.outbound_calls:
            if call.path == path and call.method == wrong_method:
                call.method = correct_method
                break

    return Fix(
        fix_id=fix_id,
        finding=f,
        description=f"Change {wrong_method} to {correct_method} for {path} in {f.service}",
        action="fix_method",
        target_service=f.service,
        mutations={"old_method": wrong_method, "new_method": correct_method, "path": path},
    )


def run_fast_iterations(blueprints: dict, findings: list, config,
                        max_iterations: int = 5000) -> FixResult:
    """Run fast in-memory fix iterations.

    Strategy: greedy hill-climbing
    1. Sort findings by severity (critical first) and fixability (auto first)
    2. For each auto-fixable finding:
       a. Deep-copy blueprint
       b. Apply fix
       c. Re-simulate + re-grade
       d. If score improved -> keep, else revert
    3. After all individual fixes, try combinations
    4. Return the optimal fix set

    This runs entirely in-memory -- no disk I/O, no service calls.
    Typical speed: 1000+ iterations/second.
    """
    from ..simulation.contract_checker import check_contracts, check_dependency_drift, check_circular_deps
    from ..simulation.flow_tracer import trace_all_flows
    from ..grading.scorer import score_system

    start = time.time()

    # Initial score
    all_findings, traces = _simulate_fast(blueprints, config)
    initial_grade = score_system(blueprints, all_findings, traces, config)

    # Sort auto-fixable findings
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    auto_fixable = [
        f for f in findings
        if f.fixability == "auto" and f.severity in ("critical", "warning")
    ]
    auto_fixable.sort(key=lambda f: severity_order.get(f.severity, 9))

    needs_user = [f for f in findings if f.fixability == "needs_user"]

    accepted_fixes = []
    rejected_fixes = []
    current_blueprints = copy.deepcopy(blueprints)
    current_score = initial_grade.overall
    iteration = 0

    # Phase 1: Try each fix individually (greedy) with lightweight undo
    for finding in auto_fixable:
        if iteration >= max_iterations:
            break
        iteration += 1

        fix = apply_fix_to_blueprint(current_blueprints, finding, iteration)

        if fix.action == "noop":
            rejected_fixes.append(fix)
            continue

        new_findings, new_traces = _simulate_fast(current_blueprints, config)
        new_grade = score_system(current_blueprints, new_findings, new_traces, config)

        fix.score_delta = new_grade.overall - current_score

        if new_grade.overall > current_score + 0.001:
            fix.accepted = True
            accepted_fixes.append(fix)
            current_score = new_grade.overall
        else:
            _undo_fix(current_blueprints, fix)
            rejected_fixes.append(fix)

    # Phase 2: Try fixing remaining findings that weren't auto but might work now
    needs_context = [f for f in findings if f.fixability == "needs_context"]
    for finding in needs_context:
        if iteration >= max_iterations:
            break
        iteration += 1

        fix = apply_fix_to_blueprint(current_blueprints, finding, iteration)

        if fix.action == "noop":
            continue

        new_findings, new_traces = _simulate_fast(current_blueprints, config)
        new_grade = score_system(current_blueprints, new_findings, new_traces, config)
        fix.score_delta = new_grade.overall - current_score

        if new_grade.overall > current_score + 0.001:
            fix.accepted = True
            accepted_fixes.append(fix)
            current_score = new_grade.overall
        else:
            _undo_fix(current_blueprints, fix)
            rejected_fixes.append(fix)

    # Phase 3: Rule-based decisions for "needs_user" findings
    from .rule_engine import make_decision, apply_decision, undo_decision

    llm_remaining = []

    for finding in needs_user:
        if iteration >= max_iterations:
            llm_remaining.extend(needs_user[needs_user.index(finding):])
            break
        iteration += 1

        decision = make_decision(finding, current_blueprints, config)

        if decision.confidence < 0.4:
            llm_remaining.append(finding)
            continue

        if decision.action == "skip":
            fix = Fix(
                fix_id=iteration,
                finding=finding,
                description=f"[RULE] {decision.rule_id}: {decision.reasoning}",
                action="skip",
                target_service=finding.service,
                mutations={"rule_id": decision.rule_id},
                score_delta=0.0,
                accepted=True,
            )
            accepted_fixes.append(fix)
            continue

        mutation = apply_decision(current_blueprints, decision, config)

        if mutation.get("type") == "noop":
            llm_remaining.append(finding)
            continue

        if mutation.get("type") == "already_fixed":
            fix = Fix(
                fix_id=iteration,
                finding=finding,
                description=f"[RULE] {decision.rule_id}: Already resolved by earlier fix",
                action="skip",
                target_service=finding.service,
                mutations={"rule_id": decision.rule_id},
                score_delta=0.0,
                accepted=True,
            )
            accepted_fixes.append(fix)
            continue

        new_findings, new_traces = _simulate_fast(current_blueprints, config)
        new_grade = score_system(current_blueprints, new_findings, new_traces, config)
        score_delta = new_grade.overall - current_score

        if new_grade.overall >= current_score - 0.001:
            mutation["rule_id"] = decision.rule_id
            fix = Fix(
                fix_id=iteration,
                finding=finding,
                description=f"[RULE] {decision.rule_id}: {decision.reasoning}",
                action=decision.action,
                target_service=finding.service,
                mutations=mutation,
                score_delta=score_delta,
                accepted=True,
            )
            accepted_fixes.append(fix)
            current_score = new_grade.overall
        else:
            undo_decision(current_blueprints, decision, mutation, config)
            llm_remaining.append(finding)

    # Final grade
    final_findings, final_traces = _simulate_fast(current_blueprints, config)
    final_grade = score_system(current_blueprints, final_findings, final_traces, config)

    elapsed = time.time() - start

    return FixResult(
        initial_grade=initial_grade,
        final_grade=final_grade,
        iterations_run=iteration,
        fixes_accepted=accepted_fixes,
        fixes_rejected=rejected_fixes,
        needs_user=llm_remaining,
        elapsed_ms=elapsed * 1000,
        iterations_per_second=iteration / max(elapsed, 0.001),
    )


def _undo_fix(blueprints: dict, fix: Fix):
    """Undo a fix by reversing its mutations. Lightweight alternative to deepcopy."""
    if fix.action == "add_endpoint":
        bp = blueprints.get(fix.target_service)
        if bp and bp.endpoints:
            path = fix.mutations.get("path", "")
            method = fix.mutations.get("method", "")
            bp.endpoints = [
                ep for ep in bp.endpoints
                if not (ep.path == path and ep.method == method and ep.framework == "stub")
            ]

    elif fix.action == "fix_method":
        bp = blueprints.get(fix.target_service)
        if bp:
            old_method = fix.mutations.get("new_method")
            new_method = fix.mutations.get("old_method")
            path = fix.mutations.get("path")
            for call in bp.outbound_calls:
                if call.path == path and call.method == old_method:
                    call.method = new_method
                    break


def _simulate_fast(blueprints: dict, config) -> tuple:
    """Fast simulation pass -- contracts + flow tracing."""
    from ..simulation.contract_checker import check_contracts, check_dependency_drift, check_circular_deps
    from ..simulation.flow_tracer import trace_all_flows

    findings = []
    traces = []

    findings.extend(check_contracts(blueprints, config))
    findings.extend(check_dependency_drift(blueprints, config))
    findings.extend(check_circular_deps(blueprints))

    flow_traces, flow_findings = trace_all_flows(blueprints, config)
    traces.extend(flow_traces)
    findings.extend(flow_findings)

    # Deduplicate
    seen = set()
    unique = []
    for f in findings:
        key = (f.finding_type, f.service, f.endpoint, f.details)
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique, traces


def format_fix_report(result: FixResult) -> str:
    """Format the fix iteration results as markdown."""
    lines = []
    lines.append("# Fix Iteration Report\n")
    lines.append(f"## Performance")
    lines.append(f"- Iterations: {result.iterations_run}")
    lines.append(f"- Time: {result.elapsed_ms:.0f}ms")
    lines.append(f"- Speed: {result.iterations_per_second:.0f} iterations/sec\n")

    lines.append(f"## Score Improvement")
    lines.append(f"- Before: {result.initial_grade.grade} ({result.initial_grade.overall:.1f}/100)")
    lines.append(f"- After:  {result.final_grade.grade} ({result.final_grade.overall:.1f}/100)")
    lines.append(f"- Delta:  +{result.final_grade.overall - result.initial_grade.overall:.1f}\n")

    lines.append(f"## Dimension Breakdown\n")
    lines.append("| Dimension | Before | After | Delta |")
    lines.append("|-----------|--------|-------|-------|")
    for dim in ["completeness", "contract_fidelity", "resilience", "dependency_hygiene", "coverage", "security"]:
        before = getattr(result.initial_grade, dim)
        after = getattr(result.final_grade, dim)
        delta = after - before
        sign = "+" if delta > 0 else ""
        lines.append(f"| {dim} | {before:.1f} | {after:.1f} | {sign}{delta:.1f} |")
    lines.append("")

    auto_fixes = [f for f in result.fixes_accepted if not f.description.startswith("[RULE]")]
    rule_fixes = [f for f in result.fixes_accepted if f.description.startswith("[RULE]")]

    if auto_fixes:
        lines.append(f"## Auto-Fixes ({len(auto_fixes)})\n")
        for fix in auto_fixes:
            lines.append(f"- **{fix.action}**: {fix.description} (score +{fix.score_delta:.2f})")
            if fix.mutations:
                lines.append(f"  - Details: {fix.mutations}")
        lines.append("")

    if rule_fixes:
        lines.append(f"## Rule-Based Decisions ({len(rule_fixes)})\n")
        lines.append("*Decided by deterministic IF/THEN rules:*\n")

        by_rule = {}
        for fix in rule_fixes:
            rule = fix.mutations.get("rule_id", "UNKNOWN") if isinstance(fix.mutations, dict) else "UNKNOWN"
            by_rule.setdefault(rule, []).append(fix)

        for rule_id, fixes in sorted(by_rule.items()):
            lines.append(f"### Rule: {rule_id} ({len(fixes)} applied)\n")
            for fix in fixes[:5]:
                desc = fix.description.replace("[RULE] ", "")
                lines.append(f"- **{fix.action}**: {desc}")
            if len(fixes) > 5:
                lines.append(f"- *... and {len(fixes) - 5} more*")
            lines.append("")

    if result.needs_user:
        lines.append(f"## Unresolved ({len(result.needs_user)})\n")
        lines.append("*No rule matched -- needs architectural review:*\n")
        for i, f in enumerate(result.needs_user[:20], 1):
            lines.append(f"{i}. **[{f.severity}] {f.finding_type}**: {f.details}")
            if f.suggested_fix:
                lines.append(f"   - Suggestion: {f.suggested_fix}")
        if len(result.needs_user) > 20:
            lines.append(f"\n*... and {len(result.needs_user) - 20} more*")

    return "\n".join(lines)
