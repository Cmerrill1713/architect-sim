"""Strategic architecture assessment -- the holistic "where should this system go" engine.

Synthesizes ALL analysis data (blueprints, findings, grades, temporal profiles,
latency estimates, LLM calls, hardware) into 7 strategic assessments:

1. Architecture Health Map — classify services by structural role
2. Complexity Hotspots — find dangerously complex files/services
3. Service Maturity Model — grade services against tier expectations
4. Dependency Flow Analysis — detect layer violations
5. Path to A Roadmap — sequenced steps from current grade to 90+
6. Consolidation Analysis — find merge candidates
7. Over-Engineering Detection — find unnecessary complexity
"""
from collections import defaultdict
from pathlib import Path

from ..models import ServiceBlueprint, Finding, GradeReport, LLMCall
from ..grading.scorer import WEIGHTS, GRADE_THRESHOLDS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SKIP_DIRS = {
    "vendor", "node_modules", "venv", ".venv", "__pycache__", ".git",
    "dist", "build", "target", ".tox", ".mypy_cache",
}

_SOURCE_EXTENSIONS = {
    ".go", ".py", ".rs", ".ts", ".tsx", ".js", ".jsx",
}

MATURITY_CRITERIA = {
    "tier0_spine": {
        "has_health_endpoint": True,
        "has_retries": True,
        "has_timeout": True,
        "has_circuit_breaker": False,
        "has_tests": True,
        "no_critical_findings": True,
        "no_security_issues": True,
    },
    "tier1_core": {
        "has_health_endpoint": True,
        "has_retries": True,
        "has_timeout": True,
        "no_critical_findings": True,
    },
    "tier2_extended": {
        "has_health_endpoint": True,
    },
}


# ===================================================================
# 1. Architecture Health Map
# ===================================================================

def classify_services(blueprints, findings, temporal_profiles=None):
    """Classify each service by its structural role.

    Roles:
    - "god_service": >100 endpoints OR >50 outbound calls OR >20 inbound callers
    - "spine": called by >5 other services, <50 endpoints
    - "leaf": calls others but nobody calls it (entry point)
    - "island": no inbound, no outbound calls
    - "pass_through": few endpoints (<10), mostly forwards to others
    - "workhorse": moderate endpoints, moderate calls, actively developed

    Returns:
        {service_name: {"role": str, "reason": str, "recommendation": str,
         "inbound_callers": int, "endpoints": int, "outbound_calls": int}}
    """
    # Build inbound caller map: callee -> set of unique callers
    inbound_callers = defaultdict(set)
    for bp in blueprints.values():
        for call in bp.outbound_calls:
            callee = call.callee_service
            if not callee.startswith("unknown:") and callee != bp.name:
                inbound_callers[callee].add(bp.name)

    result = {}
    for name, bp in blueprints.items():
        ep_count = len(bp.endpoints)
        out_count = len(bp.outbound_calls)
        in_count = len(inbound_callers.get(name, set()))

        # Classify
        if ep_count > 100 or out_count > 50 or in_count > 20:
            role = "god_service"
            reason = []
            if ep_count > 100:
                reason.append(f"{ep_count} endpoints")
            if out_count > 50:
                reason.append(f"{out_count} outbound calls")
            if in_count > 20:
                reason.append(f"{in_count} inbound callers")
            reason_str = f"Too large: {', '.join(reason)}"
            recommendation = (
                f"Split into focused services. Group endpoints by domain "
                f"(e.g., separate chat, daemon, tools, admin into distinct services)."
            )
        elif in_count > 5 and ep_count < 50:
            role = "spine"
            reason_str = f"Called by {in_count} services with {ep_count} endpoints"
            recommendation = (
                "Critical path service. Needs highest reliability: health checks, "
                "retries, circuit breakers, monitoring."
            )
        elif out_count > 0 and in_count == 0:
            role = "leaf"
            reason_str = f"Entry point: {out_count} outbound calls, no inbound callers"
            recommendation = "User-facing surface. Focus on input validation and UX."
        elif out_count == 0 and in_count == 0:
            role = "island"
            reason_str = "No inbound or outbound service calls"
            recommendation = "Evaluate if this service is still needed or if it runs standalone."
        elif ep_count < 10 and out_count > ep_count:
            role = "pass_through"
            reason_str = f"Only {ep_count} endpoints but {out_count} outbound calls"
            recommendation = (
                "May be unnecessary indirection. Consider merging into the "
                "caller or callee if it adds no business logic."
            )
        else:
            role = "workhorse"
            reason_str = f"{ep_count} endpoints, {out_count} outbound, {in_count} inbound"
            recommendation = "Healthy service. Monitor for growth toward god_service thresholds."

        result[name] = {
            "role": role,
            "reason": reason_str,
            "recommendation": recommendation,
            "inbound_callers": in_count,
            "endpoints": ep_count,
            "outbound_calls": out_count,
        }

    return result


# ===================================================================
# 2. Complexity Hotspots
# ===================================================================

def _count_file_lines(file_path):
    """Count lines in a source file. Returns 0 on error."""
    try:
        with open(file_path, "r", errors="replace") as f:
            return sum(1 for _ in f)
    except (OSError, IOError):
        return 0


def _collect_source_files(source_dir):
    """Collect source files from a service directory, skipping vendor/node_modules."""
    files = []
    if not source_dir:
        return files
    root = Path(source_dir)
    if not root.exists():
        return files

    for path in root.rglob("*"):
        if any(skip in path.parts for skip in _SKIP_DIRS):
            continue
        if path.is_file() and path.suffix in _SOURCE_EXTENSIONS:
            files.append(path)
    return files


def find_complexity_hotspots(blueprints, temporal_profiles=None):
    """Find files and services that are dangerously complex.

    Metrics per service:
    - total_files: count of source files
    - largest_file_lines: line count of biggest file
    - avg_file_lines: average file size
    - endpoint_density: endpoints per file
    - call_density: outbound calls per file
    - change_frequency: commits in last 30 days (from temporal)

    Hotspot = large file x high change frequency x high endpoint count

    Returns:
        list of dicts with file paths, metrics, and refactor suggestions,
        sorted by severity (worst first).
    """
    hotspots = []

    for name, bp in blueprints.items():
        source_files = _collect_source_files(bp.source_dir)
        if not source_files:
            continue

        file_lines = {}
        for f in source_files:
            lines = _count_file_lines(f)
            if lines > 0:
                file_lines[str(f)] = lines

        total_files = len(file_lines)
        if total_files == 0:
            continue

        largest_file = max(file_lines, key=file_lines.get) if file_lines else ""
        largest_lines = file_lines.get(largest_file, 0)
        avg_lines = sum(file_lines.values()) / total_files
        ep_count = len(bp.endpoints)
        call_count = len(bp.outbound_calls)
        ep_density = ep_count / total_files if total_files else 0
        call_density = call_count / total_files if total_files else 0

        change_freq = 0
        if temporal_profiles and name in temporal_profiles:
            change_freq = temporal_profiles[name].commit_count_30d

        # Compute a composite severity score for sorting
        severity = 0
        flags = []

        if largest_lines > 2000:
            severity += 3
            flags.append(f"File > 2000 lines: {Path(largest_file).name} ({largest_lines} lines)")

        if ep_count > 50:
            severity += 2
            flags.append(f"Service has {ep_count} endpoints (> 50)")

        if change_freq > 10 and largest_lines > 1000:
            severity += 2
            flags.append(
                f"High churn ({change_freq} commits/30d) on large file "
                f"({largest_lines} lines)"
            )

        if largest_lines > 1000 and largest_lines <= 2000:
            severity += 1
            flags.append(f"Large file: {Path(largest_file).name} ({largest_lines} lines)")

        if not flags:
            continue

        # Suggest refactoring based on what's wrong
        suggestions = []
        if largest_lines > 2000:
            suggestions.append(
                f"Split {Path(largest_file).name} into focused modules "
                f"(handlers, middleware, models)."
            )
        if ep_count > 50:
            suggestions.append(
                "Group endpoints by domain and split into sub-services or "
                "route packages."
            )
        if change_freq > 10 and largest_lines > 1000:
            suggestions.append(
                "High churn on large files increases merge conflicts. "
                "Extract hot-path logic into a separate module."
            )

        hotspots.append({
            "service": name,
            "severity": severity,
            "flags": flags,
            "suggestions": suggestions,
            "metrics": {
                "total_files": total_files,
                "largest_file": largest_file,
                "largest_file_lines": largest_lines,
                "avg_file_lines": round(avg_lines, 0),
                "endpoint_density": round(ep_density, 2),
                "call_density": round(call_density, 2),
                "change_frequency": change_freq,
            },
        })

    hotspots.sort(key=lambda h: h["severity"], reverse=True)
    return hotspots


# ===================================================================
# 3. Service Maturity Model
# ===================================================================

def _has_health_endpoint(bp):
    """Check if service has a health endpoint."""
    for ep in bp.endpoints:
        path_lower = ep.path.lower()
        if any(p in path_lower for p in ("/health", "/healthz", "/ready", "/live")):
            return True
    return False


def _has_retries(bp):
    """Check if service has retry wrappers on outbound calls."""
    if not bp.outbound_calls:
        return True  # No outbound calls = nothing to retry
    retried = sum(1 for c in bp.outbound_calls if c.retry_config)
    return retried > 0 and retried >= len(bp.outbound_calls) * 0.3


def _has_timeout(bp):
    """Check if service has timeout configuration."""
    for call in bp.outbound_calls:
        if call.retry_config and call.retry_config.get("timeout"):
            return True
    # Check retry configs at service level
    if bp.retry_configs:
        for cfg in bp.retry_configs.values():
            if isinstance(cfg, dict) and cfg.get("timeout"):
                return True
    return not bp.outbound_calls  # No calls = no timeout needed


def _has_tests(bp):
    """Check if service has test files."""
    if not bp.source_dir:
        return False
    root = Path(bp.source_dir)
    if not root.exists():
        return False
    for path in root.rglob("*"):
        if any(skip in path.parts for skip in _SKIP_DIRS):
            continue
        name = path.name
        if (name.endswith("_test.go") or name.startswith("test_") or
                name.endswith(".test.ts") or name.endswith(".test.js") or
                name.endswith("_test.py") or name.endswith(".spec.ts") or
                name.endswith(".spec.js")):
            return True
    return False


def _has_critical_findings(name, findings):
    """Check if service has critical findings."""
    return any(
        f.service == name and f.severity == "critical"
        for f in findings
    )


def _has_security_issues(name, findings):
    """Check if service has security findings."""
    return any(
        f.service == name and f.finding_type == "security_issue"
        for f in findings
    )


def _determine_tier(name, bp, config, inbound_count):
    """Determine the maturity tier a service should meet."""
    # Check contracts for startup policy
    if config and config.contracts:
        contract = config.contracts.get(name, {})
        if isinstance(contract, dict):
            policy = contract.get("startup_policy", "")
            if policy in ("always_on", "required"):
                return "tier0_spine"

    # High inbound = spine
    if inbound_count > 5:
        return "tier0_spine"
    if inbound_count > 2:
        return "tier1_core"
    return "tier2_extended"


def assess_service_maturity(blueprints, findings, config):
    """Grade each service against its tier's maturity criteria.

    Returns:
        {service: {tier, criteria_met, criteria_missing, maturity_score, recommendation}}
    """
    # Build inbound count
    inbound_callers = defaultdict(set)
    for bp in blueprints.values():
        for call in bp.outbound_calls:
            callee = call.callee_service
            if not callee.startswith("unknown:") and callee != bp.name:
                inbound_callers[callee].add(bp.name)

    result = {}
    for name, bp in blueprints.items():
        in_count = len(inbound_callers.get(name, set()))
        tier = _determine_tier(name, bp, config, in_count)
        criteria = MATURITY_CRITERIA[tier]

        checks = {}
        if "has_health_endpoint" in criteria:
            checks["has_health_endpoint"] = _has_health_endpoint(bp)
        if "has_retries" in criteria:
            checks["has_retries"] = _has_retries(bp)
        if "has_timeout" in criteria:
            checks["has_timeout"] = _has_timeout(bp)
        if "has_circuit_breaker" in criteria:
            checks["has_circuit_breaker"] = False  # Not auto-detectable yet
        if "has_tests" in criteria:
            checks["has_tests"] = _has_tests(bp)
        if "no_critical_findings" in criteria:
            checks["no_critical_findings"] = not _has_critical_findings(name, findings)
        if "no_security_issues" in criteria:
            checks["no_security_issues"] = not _has_security_issues(name, findings)

        # Only count required criteria (value=True in template)
        required = {k: v for k, v in criteria.items() if v is True}
        met = [k for k in required if checks.get(k, False)]
        missing = [k for k in required if not checks.get(k, False)]
        score = (len(met) / len(required) * 100) if required else 100.0

        # Recommendation
        if not missing:
            rec = f"Meets all {tier} criteria."
        else:
            missing_readable = [k.replace("_", " ").replace("has ", "add ").replace("no ", "fix ")
                                for k in missing]
            rec = f"Missing for {tier}: {', '.join(missing_readable)}."

        result[name] = {
            "tier": tier,
            "criteria_met": met,
            "criteria_missing": missing,
            "maturity_score": round(score, 1),
            "recommendation": rec,
        }

    return result


# ===================================================================
# 4. Dependency Flow Analysis
# ===================================================================

_LAYER_KEYWORDS = {
    5: ("db", "redis", "postgres", "mongo", "embed", "llm", "cache",
        "llama", "mlx", "rerank"),
    4: ("memory", "model", "truth", "store", "ego", "world",
        "correlat"),
    3: ("core", "orchestrat", "handler", "loop", "daemon",
        "pipeline", "service-manager"),
    2: ("router", "gateway", "proxy", "ingress", "nginx", "caddy",
        "vite"),
    1: ("ui", "cli", "frontend", "web", "app", "dashboard"),
}


def _assign_layer(name, bp, inbound_callers):
    """Assign a dependency layer to a service.

    Layer 1 = entry (UI, CLI), Layer 5 = infrastructure (DB, LLM).
    """
    name_lower = name.lower()

    # Keyword match (check from infrastructure up to entry)
    for layer in (5, 4, 3, 2, 1):
        keywords = _LAYER_KEYWORDS[layer]
        if any(kw in name_lower for kw in keywords):
            return layer

    # Heuristic: no inbound callers = entry point (layer 1)
    if not inbound_callers.get(name):
        return 1

    # Default: application layer
    return 3


def analyze_dependency_flow(blueprints, config):
    """Detect dependency flow violations.

    Clean architecture layers (top -> bottom):
    1. Entry (UI, CLI, external API)
    2. Gateway (router, proxy, API gateway)
    3. Application (business logic services)
    4. Core (memory, truth, world model)
    5. Infrastructure (database, cache, LLM, embedding)

    Flag violations: lower layers calling upper layers.

    Returns:
        {
            "layers": {service: layer_number},
            "violations": [(caller, callee, description)],
            "clean_ratio": float,
            "recommendation": str,
        }
    """
    # Build inbound map
    inbound_callers = defaultdict(set)
    for bp in blueprints.values():
        for call in bp.outbound_calls:
            callee = call.callee_service
            if not callee.startswith("unknown:") and callee != bp.name:
                inbound_callers[callee].add(bp.name)

    # Assign layers
    layers = {}
    for name, bp in blueprints.items():
        layers[name] = _assign_layer(name, bp, inbound_callers)

    # Check all calls for violations
    violations = []
    clean_count = 0
    total_count = 0

    layer_names = {1: "entry", 2: "gateway", 3: "application", 4: "core", 5: "infrastructure"}

    for bp in blueprints.values():
        caller = bp.name
        caller_layer = layers.get(caller, 3)
        for call in bp.outbound_calls:
            callee = call.callee_service
            if callee.startswith("unknown:") or callee == caller:
                continue
            callee_layer = layers.get(callee, 3)
            total_count += 1

            if callee_layer < caller_layer:
                # Lower layer calling upper layer = violation
                violations.append((
                    caller,
                    callee,
                    f"{layer_names.get(caller_layer, '?')} (L{caller_layer}) calls "
                    f"{layer_names.get(callee_layer, '?')} (L{callee_layer})",
                ))
            else:
                clean_count += 1

    clean_ratio = (clean_count / total_count * 100) if total_count else 100.0

    # Recommendation
    if clean_ratio >= 95:
        recommendation = "Dependency flow is clean. Minor violations are acceptable."
    elif clean_ratio >= 80:
        recommendation = (
            f"{len(violations)} violations detected. Focus on breaking upward "
            f"calls from core/infrastructure into application/gateway layers."
        )
    else:
        recommendation = (
            f"Significant layering issues: {len(violations)} violations "
            f"({100 - clean_ratio:.0f}% of calls go the wrong direction). "
            f"Consider introducing an event bus or async messaging to decouple layers."
        )

    return {
        "layers": layers,
        "violations": violations,
        "clean_ratio": round(clean_ratio, 1),
        "recommendation": recommendation,
    }


# ===================================================================
# 5. Path to A Roadmap
# ===================================================================

def generate_roadmap(grade, findings, blueprints, scenario_results=None,
                     llm_calls=None, hardware=None):
    """Generate a sequenced roadmap from current grade to A (90+).

    Strategy: fix dimensions in order of weight x gap.
    Each step predicts cumulative score improvement.

    Returns:
        list of step dicts with predicted cumulative score.
    """
    target = 95.0
    if grade.overall >= target:
        return [{
            "step": 1,
            "action": "Maintain current A grade",
            "dimension": "all",
            "score_before": round(grade.overall, 1),
            "score_after": round(grade.overall, 1),
            "delta": 0,
            "cumulative": round(grade.overall, 1),
        }]

    # Current dimension scores
    dimensions = {
        "completeness": grade.completeness,
        "contract_fidelity": grade.contract_fidelity,
        "resilience": grade.resilience,
        "dependency_hygiene": grade.dependency_hygiene,
        "coverage": grade.coverage,
        "security": grade.security,
    }

    # Count fixable items per dimension
    type_to_dimension = {
        "phantom_call": "completeness",
        "missing_endpoint": "completeness",
        "unknown_service": "completeness",
        "field_mismatch": "contract_fidelity",
        "method_mismatch": "contract_fidelity",
        "undeclared_dep": "dependency_hygiene",
        "circular_dep": "dependency_hygiene",
        "orphan_endpoint": "coverage",
        "security_issue": "security",
    }

    finding_counts = defaultdict(int)
    for f in findings:
        dim = type_to_dimension.get(f.finding_type)
        if dim:
            finding_counts[dim] += 1

    # Calculate weighted gap for each dimension
    gaps = []
    for dim, score in dimensions.items():
        gap = 100 - score
        if gap <= 0:
            continue
        weight = WEIGHTS.get(dim, 0.1)
        weighted_gap = gap * weight
        gaps.append((dim, score, gap, weight, weighted_gap))

    # Sort by weighted gap descending (biggest improvement opportunity first)
    gaps.sort(key=lambda x: x[4], reverse=True)

    # Also account for resilience which is about retry counts, not findings
    total_calls = sum(len(bp.outbound_calls) for bp in blueprints.values())
    resilient_calls = sum(
        1 for bp in blueprints.values()
        for c in bp.outbound_calls if c.retry_config
    )
    calls_needing_retries = total_calls - resilient_calls

    steps = []
    cumulative = grade.overall

    for dim, current_score, gap, weight, weighted_gap in gaps:
        # Estimate how much we can improve this dimension
        if dim == "resilience":
            if calls_needing_retries > 0:
                new_score = 100.0
                action = f"Add retries to {calls_needing_retries} calls"
            else:
                continue
        elif dim == "completeness":
            count = finding_counts.get(dim, 0)
            if count == 0:
                continue
            new_score = min(100, current_score + gap * 0.9)
            action = f"Fix {count} missing/phantom endpoints"
        elif dim == "contract_fidelity":
            count = finding_counts.get(dim, 0)
            if count == 0:
                continue
            new_score = min(100, current_score + gap * 0.9)
            action = f"Fix {count} method/field mismatches"
        elif dim == "dependency_hygiene":
            count = finding_counts.get(dim, 0)
            if count == 0:
                continue
            new_score = min(100, current_score + gap * 0.7)
            action = f"Fix {count} undeclared/circular dependencies"
        elif dim == "coverage":
            count = finding_counts.get(dim, 0)
            if count == 0:
                continue
            new_score = min(100, current_score + gap * 0.5)
            action = f"Remove/document {count} orphan endpoints"
        elif dim == "security":
            count = finding_counts.get(dim, 0)
            if count == 0:
                continue
            new_score = min(100, current_score + gap * 0.8)
            action = f"Fix {count} security issues"
        else:
            continue

        # Calculate score impact
        delta = (new_score - current_score) * weight
        new_cumulative = cumulative + delta

        steps.append({
            "step": len(steps) + 1,
            "action": action,
            "dimension": dim,
            "dimension_before": round(current_score, 0),
            "dimension_after": round(new_score, 0),
            "score_before": round(cumulative, 1),
            "score_after": round(new_cumulative, 1),
            "delta": round(delta, 1),
            "cumulative": round(new_cumulative, 1),
        })
        cumulative = new_cumulative

    # Determine final grade
    if steps:
        final_score = steps[-1]["cumulative"]
        final_grade = "F"
        for threshold, letter in GRADE_THRESHOLDS:
            if final_score >= threshold:
                final_grade = letter
                break
        steps.append({
            "step": "TOTAL",
            "action": f"{grade.grade}({grade.overall:.0f}) -> {final_grade}({final_score:.0f}) in {len(steps)} steps",
            "dimension": "all",
            "score_before": round(grade.overall, 1),
            "score_after": round(final_score, 1),
            "delta": round(final_score - grade.overall, 1),
            "cumulative": round(final_score, 1),
        })

    return steps


# ===================================================================
# 6. Consolidation Analysis
# ===================================================================

def find_consolidation_candidates(blueprints, findings, temporal_profiles=None):
    """Find services that should be merged to reduce operational complexity.

    Returns:
        list of dicts with candidates, merge_into, reason, etc.
    """
    candidates = []
    seen = set()

    # Build call graph for mutual-call detection
    calls_to = defaultdict(set)    # service -> set of callees
    called_by = defaultdict(set)   # service -> set of callers
    for bp in blueprints.values():
        for call in bp.outbound_calls:
            callee = call.callee_service
            if not callee.startswith("unknown:") and callee != bp.name:
                calls_to[bp.name].add(callee)
                called_by[callee].add(bp.name)

    # Pattern 1: Too small to justify a service
    for name, bp in blueprints.items():
        ep_count = len(bp.endpoints)
        out_count = len(bp.outbound_calls)
        if ep_count < 3 and out_count < 5 and name not in seen:
            # Find who calls this service -- merge into the primary caller
            callers = called_by.get(name, set())
            if callers:
                merge_into = max(callers,
                                 key=lambda c: len(blueprints.get(c, ServiceBlueprint(name=c, port=0, language="")).endpoints))
            else:
                # No callers -- find who this service calls most
                callees = calls_to.get(name, set())
                if callees:
                    merge_into = max(callees,
                                     key=lambda c: len(blueprints.get(c, ServiceBlueprint(name=c, port=0, language="")).endpoints))
                else:
                    merge_into = None

            if merge_into and merge_into != name:
                seen.add(name)
                candidates.append({
                    "candidates": [name],
                    "merge_into": merge_into,
                    "reason": f"Too small: {ep_count} endpoints, {out_count} outbound calls",
                    "endpoints_saved": ep_count,
                    "services_eliminated": 1,
                })

    # Pattern 2: Stale and small
    if temporal_profiles:
        for name, bp in blueprints.items():
            if name in seen:
                continue
            tp = temporal_profiles.get(name)
            if not tp:
                continue
            ep_count = len(bp.endpoints)
            if tp.days_since_modified > 10 and ep_count < 10:
                callers = called_by.get(name, set())
                if callers:
                    merge_into = max(callers,
                                     key=lambda c: len(blueprints.get(c, ServiceBlueprint(name=c, port=0, language="")).endpoints))
                    seen.add(name)
                    candidates.append({
                        "candidates": [name],
                        "merge_into": merge_into,
                        "reason": (
                            f"Stale ({tp.days_since_modified}d since modified) "
                            f"and small ({ep_count} endpoints)"
                        ),
                        "endpoints_saved": ep_count,
                        "services_eliminated": 1,
                    })

    # Pattern 3: Two services that ONLY call each other
    for name, callees in calls_to.items():
        if name in seen:
            continue
        for callee in callees:
            if callee in seen or callee == name:
                continue
            callee_callees = calls_to.get(callee, set())
            # Check: name only calls callee, callee only calls name
            if callees == {callee} and callee_callees == {name}:
                larger = name if len(blueprints.get(name, ServiceBlueprint(name=name, port=0, language="")).endpoints) >= \
                                 len(blueprints.get(callee, ServiceBlueprint(name=callee, port=0, language="")).endpoints) else callee
                smaller = callee if larger == name else name
                seen.add(smaller)
                ep_saved = len(blueprints.get(smaller, ServiceBlueprint(name=smaller, port=0, language="")).endpoints)
                candidates.append({
                    "candidates": [name, callee],
                    "merge_into": larger,
                    "reason": "These two services only call each other -- tight coupling",
                    "endpoints_saved": ep_saved,
                    "services_eliminated": 1,
                })

    # Pattern 4: Services with identical dependency sets
    dep_groups = defaultdict(list)
    for name in blueprints:
        if name in seen:
            continue
        deps = frozenset(calls_to.get(name, set()))
        if deps:  # Only group non-empty dep sets
            dep_groups[deps].append(name)

    for deps, group in dep_groups.items():
        if len(group) < 2:
            continue
        # These services call the exact same set of downstream services
        largest = max(group,
                      key=lambda n: len(blueprints.get(n, ServiceBlueprint(name=n, port=0, language="")).endpoints))
        others = [n for n in group if n != largest and n not in seen]
        if others:
            total_ep = sum(
                len(blueprints.get(n, ServiceBlueprint(name=n, port=0, language="")).endpoints)
                for n in others
            )
            for n in others:
                seen.add(n)
            candidates.append({
                "candidates": group,
                "merge_into": largest,
                "reason": f"Identical dependency set: {', '.join(sorted(deps))}",
                "endpoints_saved": total_ep,
                "services_eliminated": len(others),
            })

    return candidates


# ===================================================================
# 7. Over-Engineering Detection
# ===================================================================

def detect_over_engineering(blueprints, findings, llm_calls=None):
    """Find infrastructure that is more complex than needed.

    Returns:
        list of dicts with service, pattern, description, simplification.
    """
    issues = []

    # Build call graph
    called_by = defaultdict(set)
    for bp in blueprints.values():
        for call in bp.outbound_calls:
            callee = call.callee_service
            if not callee.startswith("unknown:") and callee != bp.name:
                called_by[callee].add(bp.name)

    # Build set of actually-called endpoints
    called_endpoints = set()
    for bp in blueprints.values():
        for call in bp.outbound_calls:
            called_endpoints.add((call.callee_service, call.method, call.path))

    for name, bp in blueprints.items():
        in_count = len(called_by.get(name, set()))
        ep_count = len(bp.endpoints)
        out_count = len(bp.outbound_calls)

        # Pattern: Retry wrappers but 0 callers
        has_retries = any(c.retry_config for c in bp.outbound_calls)
        if has_retries and in_count == 0 and out_count > 0:
            issues.append({
                "service": name,
                "pattern": "retry_without_callers",
                "description": (
                    f"Has retry wrappers on outbound calls but no service calls {name}. "
                    f"Retries protect callers, but there are none."
                ),
                "simplification": "Retries are fine to keep as defense-in-depth, but don't invest more here.",
            })

        # Pattern: >50 endpoints but <5 are called
        if ep_count > 50:
            called_count = sum(
                1 for ep in bp.endpoints
                if (name, ep.method, ep.path) in called_endpoints
            )
            if called_count < 5:
                dead_pct = ((ep_count - called_count) / ep_count) * 100
                issues.append({
                    "service": name,
                    "pattern": "mostly_dead_endpoints",
                    "description": (
                        f"{ep_count} endpoints defined but only {called_count} are called "
                        f"by other services ({dead_pct:.0f}% potentially unused)."
                    ),
                    "simplification": (
                        "Audit endpoints: remove truly dead ones, mark public APIs explicitly. "
                        "Many may be user-facing (not service-to-service) which is fine."
                    ),
                })

        # Pattern: Separate service for <3 endpoints
        if ep_count < 3 and ep_count > 0 and in_count <= 1:
            issues.append({
                "service": name,
                "pattern": "micro_service",
                "description": (
                    f"Only {ep_count} endpoint(s) with {in_count} caller(s). "
                    f"This could be a function or module, not a service."
                ),
                "simplification": (
                    "Merge into the calling service to reduce operational overhead "
                    "(deployment, monitoring, networking)."
                ),
            })

    # Pattern: LLM temperature/top_p tuning on classification tasks
    if llm_calls:
        for lc in llm_calls:
            if lc.task_type == "classification" and lc.temperature > 0:
                issues.append({
                    "service": lc.service_name,
                    "pattern": "llm_overfit_classification",
                    "description": (
                        f"Classification call with temperature={lc.temperature} "
                        f"(file: {Path(lc.file_path).name}:{lc.line_number}). "
                        f"Classification should be deterministic (temperature=0)."
                    ),
                    "simplification": "Set temperature=0 for classification tasks.",
                })

        # Pattern: Multiple embedding tiers but only 1 used
        embedding_services = set()
        for lc in llm_calls:
            if "embed" in lc.call_type.lower() or "embed" in lc.task_type.lower():
                embedding_services.add(lc.service_name)
        # Check if any service has multiple embedding calls to different models
        embed_models = defaultdict(set)
        for lc in llm_calls:
            if "embed" in lc.model.lower():
                embed_models[lc.service_name].add(lc.model)
        for svc, models in embed_models.items():
            if len(models) > 2:
                issues.append({
                    "service": svc,
                    "pattern": "multiple_embedding_tiers",
                    "description": (
                        f"Uses {len(models)} different embedding models: "
                        f"{', '.join(sorted(models))}. "
                        f"Multiple tiers add operational complexity."
                    ),
                    "simplification": (
                        "Evaluate if all tiers are needed. Often one good embedding "
                        "model with appropriate dimensionality covers all use cases."
                    ),
                })

    return issues


# ===================================================================
# Main synthesis: generate_strategic_assessment
# ===================================================================

def generate_strategic_assessment(blueprints, findings, grade, config,
                                  temporal_profiles=None, llm_calls=None,
                                  hardware=None, scenario_results=None,
                                  flow_latencies=None):
    """Generate the full strategic architecture assessment.

    This is the "forest view" -- not individual bugs but system-level direction.

    Args:
        blueprints: {service_name: ServiceBlueprint}
        findings: list[Finding]
        grade: GradeReport
        config: Config object
        temporal_profiles: {service_name: TemporalProfile} (optional)
        llm_calls: list[LLMCall] (optional)
        hardware: HardwareProfile (optional)
        scenario_results: list (optional)
        flow_latencies: list[FlowLatency] (optional)

    Returns:
        Markdown string with the full strategic assessment.
    """
    lines = []

    # Run all analyses
    health_map = classify_services(blueprints, findings, temporal_profiles)
    hotspots = find_complexity_hotspots(blueprints, temporal_profiles)
    maturity = assess_service_maturity(blueprints, findings, config)
    dep_flow = analyze_dependency_flow(blueprints, config)
    roadmap = generate_roadmap(grade, findings, blueprints, scenario_results,
                               llm_calls, hardware)
    consolidation = find_consolidation_candidates(blueprints, findings, temporal_profiles)
    over_eng = detect_over_engineering(blueprints, findings, llm_calls)

    # Derived stats
    total_services = len(blueprints)
    total_endpoints = sum(len(bp.endpoints) for bp in blueprints.values())
    total_calls = sum(len(bp.outbound_calls) for bp in blueprints.values())
    god_services = [n for n, h in health_map.items() if h["role"] == "god_service"]
    spine_services = [n for n, h in health_map.items() if h["role"] == "spine"]
    island_services = [n for n, h in health_map.items() if h["role"] == "island"]
    active_count = 0
    if temporal_profiles:
        active_count = sum(1 for p in temporal_profiles.values()
                           if p.staleness == "active")

    # ---------------------------------------------------------------
    # Executive Summary
    # ---------------------------------------------------------------
    lines.append("# Strategic Architecture Assessment\n")
    lines.append("## Executive Summary\n")

    biggest_risk = "unknown"
    if god_services:
        biggest_risk = f"god service{'s' if len(god_services) > 1 else ''} ({', '.join(god_services[:3])})"
    elif grade.resilience < 30:
        biggest_risk = f"resilience at {grade.resilience:.0f}% (most calls have no retry)"
    elif grade.security < 60:
        biggest_risk = f"security at {grade.security:.0f}%"
    elif dep_flow["violations"]:
        biggest_risk = f"{len(dep_flow['violations'])} dependency layer violations"

    lines.append(
        f"The system comprises **{total_services} services** with "
        f"**{total_endpoints} endpoints** and **{total_calls} inter-service calls**. "
        f"Current grade: **{grade.grade} ({grade.overall:.0f})**. "
    )
    if temporal_profiles:
        lines.append(
            f"Of these, **{active_count}** are actively developed "
            f"(commits in last 7 days). "
        )
    lines.append(f"Biggest risk: **{biggest_risk}**.\n")

    # ---------------------------------------------------------------
    # Architecture Health Map
    # ---------------------------------------------------------------
    lines.append("## Architecture Health Map\n")

    role_groups = defaultdict(list)
    for name, info in health_map.items():
        role_groups[info["role"]].append((name, info))

    role_order = ["god_service", "spine", "leaf", "workhorse", "pass_through", "island"]
    role_labels = {
        "god_service": "GOD SERVICE (split urgently)",
        "spine": "SPINE (critical path)",
        "leaf": "LEAF (entry point)",
        "workhorse": "WORKHORSE (healthy)",
        "pass_through": "PASS-THROUGH (evaluate need)",
        "island": "ISLAND (standalone/dead?)",
    }

    lines.append("| Service | Role | Endpoints | Outbound | Inbound | Recommendation |")
    lines.append("|---------|------|----------:|---------:|--------:|----------------|")

    for role in role_order:
        group = role_groups.get(role, [])
        for name, info in sorted(group, key=lambda x: -x[1]["endpoints"]):
            role_short = role.upper().replace("_", " ")
            lines.append(
                f"| {name} | {role_short} | {info['endpoints']} "
                f"| {info['outbound_calls']} | {info['inbound_callers']} "
                f"| {info['recommendation'][:80]} |"
            )
    lines.append("")

    if god_services:
        lines.append(
            f"**{len(god_services)} god service(s) detected.** "
            f"These are the highest-priority architectural risk.\n"
        )

    # ---------------------------------------------------------------
    # Critical Path
    # ---------------------------------------------------------------
    lines.append("## Critical Path\n")
    lines.append(
        "Services that everything depends on. Failure here cascades everywhere.\n"
    )

    critical = sorted(
        [(n, h) for n, h in health_map.items()
         if h["role"] in ("spine", "god_service")],
        key=lambda x: -x[1]["inbound_callers"],
    )[:5]

    if critical:
        lines.append("| Service | Inbound Callers | Endpoints | Maturity |")
        lines.append("|---------|----------------:|----------:|----------|")
        for name, info in critical:
            mat = maturity.get(name, {})
            mat_score = mat.get("maturity_score", 0)
            tier = mat.get("tier", "?")
            lines.append(
                f"| {name} | {info['inbound_callers']} | {info['endpoints']} "
                f"| {mat_score:.0f}% ({tier}) |"
            )
        lines.append("")
    else:
        lines.append("No clear spine services detected (flat architecture).\n")

    # ---------------------------------------------------------------
    # Path to A(95)
    # ---------------------------------------------------------------
    lines.append("## Path to A (95)\n")

    if roadmap:
        lines.append("| Step | Action | Dimension | Before | After | Delta |")
        lines.append("|-----:|--------|-----------|-------:|------:|------:|")
        for step in roadmap:
            step_num = step["step"]
            dim = step.get("dimension", "")
            dim_before = step.get("dimension_before", "")
            dim_after = step.get("dimension_after", "")
            dim_label = ""
            if dim_before != "" and dim != "all":
                dim_label = f"{dim} {dim_before}->{dim_after}"
            elif dim == "all":
                dim_label = ""
            lines.append(
                f"| {step_num} | {step['action'][:60]} | {dim_label} "
                f"| {step['score_before']} | {step['score_after']} | +{step['delta']} |"
            )
        lines.append("")
    else:
        lines.append("Already at target grade.\n")

    # ---------------------------------------------------------------
    # Consolidation Opportunities
    # ---------------------------------------------------------------
    lines.append("## Consolidation Opportunities\n")

    if consolidation:
        total_eliminated = sum(c["services_eliminated"] for c in consolidation)
        total_ep_saved = sum(c["endpoints_saved"] for c in consolidation)
        lines.append(
            f"**{total_eliminated} services** could be eliminated, "
            f"removing **{total_ep_saved} endpoints** from operational surface.\n"
        )
        lines.append("| Merge Into | Candidates | Reason | Endpoints Saved |")
        lines.append("|------------|------------|--------|----------------:|")
        for c in consolidation:
            cands = ", ".join(c["candidates"][:3])
            if len(c["candidates"]) > 3:
                cands += f" (+{len(c['candidates']) - 3} more)"
            lines.append(
                f"| {c['merge_into']} | {cands} | {c['reason'][:50]} "
                f"| {c['endpoints_saved']} |"
            )
        lines.append("")
    else:
        lines.append("No obvious consolidation candidates found.\n")

    # ---------------------------------------------------------------
    # Complexity Hotspots
    # ---------------------------------------------------------------
    lines.append("## Complexity Hotspots\n")

    top_hotspots = hotspots[:5]
    if top_hotspots:
        for h in top_hotspots:
            m = h["metrics"]
            lines.append(f"### {h['service']}\n")
            lines.append(
                f"- **Largest file:** {Path(m['largest_file']).name} "
                f"({m['largest_file_lines']} lines)"
            )
            lines.append(f"- **Total files:** {m['total_files']}")
            lines.append(f"- **Avg file size:** {m['avg_file_lines']:.0f} lines")
            if m["change_frequency"]:
                lines.append(f"- **Change frequency:** {m['change_frequency']} commits/30d")
            for flag in h["flags"]:
                lines.append(f"- **Flag:** {flag}")
            for sug in h["suggestions"]:
                lines.append(f"- **Action:** {sug}")
            lines.append("")
    else:
        lines.append("No complexity hotspots detected.\n")

    # ---------------------------------------------------------------
    # Dependency Flow
    # ---------------------------------------------------------------
    lines.append("## Dependency Flow\n")

    layer_names = {1: "Entry", 2: "Gateway", 3: "Application", 4: "Core", 5: "Infrastructure"}
    layers = dep_flow["layers"]

    # Layer summary
    layer_counts = defaultdict(list)
    for svc, layer in layers.items():
        layer_counts[layer].append(svc)

    for layer_num in sorted(layer_counts.keys()):
        svcs = layer_counts[layer_num]
        label = layer_names.get(layer_num, f"L{layer_num}")
        lines.append(f"- **L{layer_num} {label}** ({len(svcs)}): {', '.join(sorted(svcs)[:5])}")
        if len(svcs) > 5:
            lines[-1] += f" (+{len(svcs) - 5} more)"

    lines.append("")
    lines.append(f"**Clean ratio:** {dep_flow['clean_ratio']:.0f}% of calls flow downward.")
    lines.append(f"**Violations:** {len(dep_flow['violations'])}")

    if dep_flow["violations"]:
        lines.append("")
        lines.append("| Caller | Callee | Violation |")
        lines.append("|--------|--------|-----------|")
        for caller, callee, desc in dep_flow["violations"][:10]:
            lines.append(f"| {caller} | {callee} | {desc} |")
        if len(dep_flow["violations"]) > 10:
            lines.append(f"| ... | ... | +{len(dep_flow['violations']) - 10} more |")
    lines.append("")
    lines.append(f"**Recommendation:** {dep_flow['recommendation']}\n")

    # ---------------------------------------------------------------
    # Over-Engineering
    # ---------------------------------------------------------------
    lines.append("## Over-Engineering\n")

    if over_eng:
        for issue in over_eng:
            lines.append(f"- **{issue['service']}** ({issue['pattern']}): {issue['description']}")
            lines.append(f"  - Simplify: {issue['simplification']}")
        lines.append("")
    else:
        lines.append("No over-engineering patterns detected.\n")

    # ---------------------------------------------------------------
    # Service Maturity
    # ---------------------------------------------------------------
    lines.append("## Service Maturity\n")

    # Show services not meeting their tier expectations
    failing = sorted(
        [(n, m) for n, m in maturity.items() if m["criteria_missing"]],
        key=lambda x: x[1]["maturity_score"],
    )

    if failing:
        lines.append("| Service | Tier | Score | Missing |")
        lines.append("|---------|------|------:|---------|")
        for name, m in failing[:15]:
            missing = ", ".join(m["criteria_missing"][:3])
            lines.append(
                f"| {name} | {m['tier']} | {m['maturity_score']:.0f}% | {missing} |"
            )
        if len(failing) > 15:
            lines.append(f"| ... | ... | ... | +{len(failing) - 15} more |")
        lines.append("")
    else:
        lines.append("All services meet their tier's maturity criteria.\n")

    # ---------------------------------------------------------------
    # 6-Month Direction
    # ---------------------------------------------------------------
    lines.append("## 6-Month Direction\n")

    directions = []

    # Consolidation direction
    if consolidation:
        total_eliminated = sum(c["services_eliminated"] for c in consolidation)
        if total_eliminated > 5:
            directions.append(
                f"You have {total_services} services but consolidation analysis suggests "
                f"eliminating {total_eliminated}. Consolidate to "
                f"~{total_services - total_eliminated} services to reduce operational overhead."
            )

    # God service splitting
    for gs in god_services:
        info = health_map[gs]
        ep = info["endpoints"]
        directions.append(
            f"**{gs}** is a god service with {ep} endpoints. Split into focused services "
            f"by domain (e.g., separate handler groups into distinct services with clear contracts)."
        )

    # Activity vs. service count
    if temporal_profiles and active_count < total_services * 0.3:
        stale_count = total_services - active_count
        directions.append(
            f"Only {active_count}/{total_services} services are actively developed. "
            f"The {stale_count} inactive services are maintenance burden. "
            f"Archive or merge them."
        )

    # Resilience
    if grade.resilience < 50:
        directions.append(
            "Resilience is critically low. Add a shared HTTP client with built-in "
            "retry, timeout, and circuit breaker as a library. All services adopt it."
        )

    # Layer violations
    violations = dep_flow["violations"]
    if len(violations) > 5:
        # Find the most common violation pair
        pair_counts = defaultdict(int)
        for caller, callee, _ in violations:
            pair_counts[(caller, callee)] += 1
        worst_pair = max(pair_counts, key=pair_counts.get) if pair_counts else None
        if worst_pair:
            directions.append(
                f"The dependency graph has {len(violations)} layer violations. "
                f"Breaking the {worst_pair[0]} <-> {worst_pair[1]} coupling should be "
                f"priority because it is in the critical path."
            )

    # LLM strategy
    if llm_calls:
        simple = sum(1 for lc in llm_calls if lc.estimated_complexity == "simple")
        complex_ = sum(1 for lc in llm_calls if lc.estimated_complexity == "complex")
        if simple > 5:
            directions.append(
                f"The LLM layer has {simple} simple calls (classification/routing) "
                f"that do not need the main model. Add a small model (e.g., 4B) for "
                f"these tasks so the system survives main model restarts."
            )

    # Security
    if grade.security < 60:
        sec_count = sum(1 for f in findings if f.finding_type == "security_issue")
        directions.append(
            f"Security score is {grade.security:.0f}% with {sec_count} issues. "
            f"Implement a security review gate before merging: "
            f"input validation, auth checks, path traversal prevention."
        )

    # Island services
    if len(island_services) > 3:
        directions.append(
            f"{len(island_services)} island services have no inbound or outbound calls: "
            f"{', '.join(island_services[:5])}. "
            f"Either integrate them into the service mesh or remove them."
        )

    # Fallback
    if not directions:
        directions.append(
            "Architecture is in good shape. Focus on maintaining test coverage, "
            "monitoring service maturity, and preventing god service emergence."
        )

    for d in directions:
        lines.append(f"- {d}")
    lines.append("")

    return "\n".join(lines)
