"""Generate prioritized, actionable engineering recommendations.

Synthesizes findings from all analysis phases into a ranked action plan.
Each recommendation has: what to do, why, expected impact, and effort.
"""
from ..models import Finding, GradeReport


def generate_recommendations(grade, findings, blueprints, llm_calls=None,
                             runtime_events=None, correlations=None,
                             hardware=None, model_recs=None,
                             temporal_profiles=None) -> str:
    """Generate a prioritized action plan from all analysis data.

    Returns markdown string.
    """
    actions = []

    # ============================================================
    # 1. RUNTIME-CONFIRMED ISSUES (highest priority — causing real failures)
    # ============================================================
    if correlations:
        confirmed = [c for c in correlations if c.severity_boost == "confirmed"]
        if confirmed:
            # Group by port/service
            port_failures = {}
            for c in confirmed:
                # Extract port from finding
                import re
                m = re.search(r'port (\d+)', c.finding_details)
                port = int(m.group(1)) if m else 0
                if port:
                    if port not in port_failures:
                        port_failures[port] = {"count": 0, "details": c.finding_details[:80]}
                    port_failures[port]["count"] = max(port_failures[port]["count"], c.runtime_events)

            for port, info in sorted(port_failures.items(), key=lambda x: -x[1]["count"]):
                if info["count"] >= 100:
                    actions.append({
                        "priority": 1,
                        "category": "RUNTIME FAILURE",
                        "action": f"Remove or fix references to port {port}",
                        "reason": f"{info['count']:,} runtime failures in 24h — actively burning CPU and filling logs",
                        "impact": "HIGH — eliminates noise, reduces error log volume, frees resources",
                        "effort": "Small — grep for :{port} across codebase, remove dead references",
                        "details": info["details"],
                    })

    # ============================================================
    # 2. LLM OPTIMIZATION (second priority — direct performance gains)
    # ============================================================
    if llm_calls:
        simple_calls = [c for c in llm_calls if c.estimated_complexity == "simple"]
        complex_calls = [c for c in llm_calls if c.estimated_complexity == "complex"]
        no_model = [c for c in llm_calls if not c.model]
        no_temp = [c for c in llm_calls if c.temperature < 0]
        classification_calls = [c for c in llm_calls if c.task_type == "classification"]

        if simple_calls and hardware:
            # Can we run a small model alongside the main one?
            small_models = [m for m in (model_recs or []) if m.params_b <= 8 and m.fits]
            if small_models:
                best_small = small_models[0]
                actions.append({
                    "priority": 2,
                    "category": "LLM OPTIMIZATION",
                    "action": f"Add {best_small.model_name} ({best_small.quantization}) for {len(simple_calls)} simple calls",
                    "reason": f"{len(simple_calls)} classification/routing calls use the main model unnecessarily",
                    "impact": f"HIGH — frees main model for complex tasks, ~{best_small.estimated_vram_gb:.1f}GB VRAM, ~{best_small.estimated_tps:.0f} tok/s",
                    "effort": "Medium — add model to serving config, update routing logic",
                })

        if no_model:
            actions.append({
                "priority": 3,
                "category": "LLM OPTIMIZATION",
                "action": f"Add explicit model selection to {len(no_model)} LLM call sites",
                "reason": "Calls without explicit model can't be routed intelligently",
                "impact": "MEDIUM — enables per-task model routing and cost optimization",
                "effort": "Small — add model parameter to each call site",
            })

        if classification_calls and no_temp:
            deterministic_needed = [c for c in classification_calls if c.temperature < 0 or c.temperature > 0]
            if deterministic_needed:
                actions.append({
                    "priority": 3,
                    "category": "LLM OPTIMIZATION",
                    "action": f"Set temperature=0 for {len(deterministic_needed)} classification/extraction calls",
                    "reason": "Non-deterministic classification causes inconsistent routing and flaky behavior",
                    "impact": "MEDIUM — deterministic outputs, consistent behavior, easier debugging",
                    "effort": "Small — add temperature parameter",
                })

    # ============================================================
    # 3. ARCHITECTURE ISSUES (third priority — structural correctness)
    # ============================================================
    type_counts = {}
    for f in findings:
        type_counts[f.finding_type] = type_counts.get(f.finding_type, 0) + 1

    # Critical phantom calls (real missing endpoints)
    critical_phantoms = [f for f in findings
                         if f.finding_type == "phantom_call" and f.severity == "critical"]
    if critical_phantoms:
        # Group by callee service
        by_callee = {}
        for f in critical_phantoms:
            by_callee.setdefault(f.service, []).append(f)

        top_service = max(by_callee.items(), key=lambda x: len(x[1]))
        actions.append({
            "priority": 3,
            "category": "MISSING ENDPOINTS",
            "action": f"Add {len(critical_phantoms)} missing endpoints across {len(by_callee)} services",
            "reason": f"Callers expect these endpoints but they don't exist — causes runtime 404s",
            "impact": "HIGH — prevents request failures at service boundaries",
            "effort": f"Medium — worst: {top_service[0]} needs {len(top_service[1])} endpoints",
        })

    # Method mismatches
    mismatches = type_counts.get("method_mismatch", 0)
    if mismatches:
        actions.append({
            "priority": 2,
            "category": "CONTRACT MISMATCH",
            "action": f"Fix {mismatches} HTTP method mismatches",
            "reason": "Caller uses GET but endpoint expects POST (or vice versa) — guaranteed runtime failure",
            "impact": "HIGH — each mismatch is a guaranteed 405 error",
            "effort": "Small — change the method in the caller or endpoint",
        })

    # Resilience
    if grade.resilience < 30:
        total_calls = sum(len(bp.outbound_calls) for bp in blueprints.values())
        resilient = sum(1 for bp in blueprints.values()
                       for c in bp.outbound_calls if c.retry_config)
        actions.append({
            "priority": 4,
            "category": "RESILIENCE",
            "action": f"Add retry wrappers to inter-service calls ({resilient}/{total_calls} have retries)",
            "reason": f"Resilience score is {grade.resilience:.0f}% — most calls fail permanently on first error",
            "impact": "HIGH — prevents cascading failures from transient network issues",
            "effort": "Medium — wrap each HTTP call with exponential backoff retry",
        })

    # Circular dependencies
    circular = type_counts.get("circular_dep", 0)
    if circular:
        actions.append({
            "priority": 5,
            "category": "ARCHITECTURE",
            "action": f"Break {circular} circular dependencies",
            "reason": "Circular deps cause startup deadlocks and tight coupling",
            "impact": "MEDIUM — enables independent deployment and faster startups",
            "effort": "Large — requires async messaging or service restructuring",
        })

    # ============================================================
    # 4. DEAD CODE / CLEANUP (lower priority — maintenance burden)
    # ============================================================
    unknown_ports = type_counts.get("unknown_service", 0)
    if unknown_ports:
        actions.append({
            "priority": 5,
            "category": "DEAD CODE",
            "action": f"Remove {unknown_ports} references to unregistered ports",
            "reason": "Code calls services that don't exist — dead code from removed features",
            "impact": "LOW — reduces confusion, smaller codebase, faster comprehension",
            "effort": "Small — grep and remove",
        })

    orphans = type_counts.get("orphan_endpoint", 0)
    if orphans > 50:
        actions.append({
            "priority": 6,
            "category": "DEAD CODE",
            "action": f"Audit {orphans} orphan endpoints — remove dead ones, document public APIs",
            "reason": "Endpoints defined but never called by any service",
            "impact": "LOW — many are likely public APIs, but genuine dead code wastes maintenance effort",
            "effort": "Medium — requires judgment on each endpoint",
        })

    # ============================================================
    # 5. STALE SERVICES (from temporal analysis)
    # ============================================================
    if temporal_profiles:
        stale = [name for name, p in temporal_profiles.items()
                 if p.staleness in ("stale", "abandoned")]
        if stale:
            actions.append({
                "priority": 6,
                "category": "MAINTENANCE",
                "action": f"Review {len(stale)} stale/abandoned services: {', '.join(stale[:5])}",
                "reason": "Not modified in 30+ days — may be dead code or need ownership",
                "impact": "LOW — reduces maintenance surface",
                "effort": "Small — decide keep/archive for each",
            })

    # ============================================================
    # FORMAT OUTPUT
    # ============================================================
    if not actions:
        return "## Recommendations\n\nNo actionable recommendations at this time.\n"

    actions.sort(key=lambda a: a["priority"])

    lines = ["# Engineering Action Plan\n"]
    lines.append(f"**{len(actions)} recommendations** prioritized by impact and runtime evidence.\n")

    current_priority = None
    priority_labels = {1: "CRITICAL", 2: "HIGH", 3: "IMPORTANT", 4: "MODERATE", 5: "LOW", 6: "CLEANUP"}

    for i, action in enumerate(actions, 1):
        p = action["priority"]
        if p != current_priority:
            current_priority = p
            label = priority_labels.get(p, f"P{p}")
            lines.append(f"\n## Priority: {label}\n")

        lines.append(f"### {i}. [{action['category']}] {action['action']}\n")
        lines.append(f"**Why:** {action['reason']}")
        lines.append(f"**Impact:** {action['impact']}")
        lines.append(f"**Effort:** {action['effort']}")
        if action.get("details"):
            lines.append(f"**Context:** {action['details']}")
        lines.append("")

    # Summary table
    lines.append("\n## Summary\n")
    lines.append("| # | Priority | Category | Action |")
    lines.append("|---|----------|----------|--------|")
    for i, a in enumerate(actions, 1):
        label = priority_labels.get(a["priority"], f"P{a['priority']}")
        lines.append(f"| {i} | {label} | {a['category']} | {a['action'][:60]} |")

    return "\n".join(lines)
