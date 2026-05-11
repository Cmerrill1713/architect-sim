"""What-if scenario simulation engine.

Predicts the impact of hypothetical changes without touching any code.

Workflow:
    1. Take the current system state (blueprints, findings, grade, LLM calls, hardware)
    2. Apply a hypothetical scenario as an in-memory mutation
    3. Re-run simulation + grading on the mutated state
    4. Compare before/after to predict impact
    5. Calculate probability of success based on benchmark data
"""
import copy
from dataclasses import dataclass, field
from typing import Optional

from ..models import (
    Call,
    Endpoint,
    Finding,
    FlowTrace,
    GradeReport,
    LLMCall,
    ServiceBlueprint,
)
from ..grading.scorer import score_system
from ..profiling.hardware import (
    HardwareProfile,
    ModelFit,
    check_model_fits,
    estimate_vram_gb,
)
from ..simulation.contract_checker import check_contracts, check_dependency_drift, check_circular_deps
from ..simulation.flow_tracer import trace_all_flows
try:
    from ..simulation.schema_checker import check_schemas
except ImportError:
    check_schemas = None
try:
    from ..simulation.security_scan import check_security
except ImportError:
    check_security = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Mutation:
    """A single atomic change to apply to the system state."""
    mutation_type: str   # "add_model", "add_retry", "add_endpoint", etc.
    params: dict = field(default_factory=dict)


@dataclass
class Scenario:
    """A hypothetical change composed of one or more mutations."""
    name: str
    description: str
    mutations: list = field(default_factory=list)   # list[Mutation]
    expected_improvement: str = ""
    effort: str = "medium"       # "small", "medium", "large"
    risk: str = "medium"         # "low", "medium", "high"


@dataclass
class ScenarioResult:
    """The predicted outcome of applying a scenario."""
    scenario: Scenario
    grade_before: GradeReport
    grade_after: GradeReport
    score_delta: float = 0.0
    dimension_deltas: dict = field(default_factory=dict)
    findings_resolved: int = 0
    findings_new: int = 0
    probability_success: float = 0.5
    risk_factors: list = field(default_factory=list)
    tradeoffs: list = field(default_factory=list)
    estimated_time: str = ""
    model_fit: Optional[ModelFit] = None  # For add_model scenarios


# ---------------------------------------------------------------------------
# Effort-to-time mapping
# ---------------------------------------------------------------------------

_EFFORT_TIME = {
    "small": "5-30 minutes",
    "medium": "2-8 hours",
    "large": "1-3 days",
}


# ---------------------------------------------------------------------------
# Preset scenarios
# ---------------------------------------------------------------------------

PRESET_SCENARIOS: dict[str, Scenario] = {
    "add_small_model": Scenario(
        name="Add small model for classification",
        description="Run a 4B model alongside the main model for simple tasks like classification, routing, and extraction.",
        mutations=[Mutation("add_model", {
            "name": "qwen3-4b",
            "params_b": 4.0,
            "quant": "Q4_K_M",
            "port": 8089,
            "aliases": ["qwen3-4b-classifier"],
            "task_types": ["classification", "routing", "extraction"],
        })],
        expected_improvement="3-10x faster on simple tasks",
        effort="small",
        risk="low",
    ),
    "full_resilience": Scenario(
        name="Add retry + circuit breaker to all services",
        description="Wrap every inter-service call with retry (3 attempts, exponential backoff) and circuit breaker.",
        mutations=[
            Mutation("add_retry", {
                "service": "*",
                "max_attempts": 3,
                "initial_delay_ms": 100,
            }),
            Mutation("add_circuit_breaker", {
                "service": "*",
            }),
        ],
        expected_improvement="Resilience score from current to near-100%",
        effort="medium",
        risk="low",
    ),
    "kill_dead_ports": Scenario(
        name="Remove all dead port references",
        description="Remove calls to ports that have no running service. Auto-populated from findings.",
        mutations=[],  # Auto-populated from findings by _auto_populate
        expected_improvement="Eliminate unknown_service and phantom_call findings for dead ports",
        effort="small",
        risk="low",
    ),
    "fix_all_contracts": Scenario(
        name="Fix all contract violations",
        description="Add missing endpoints, fix method mismatches, resolve phantom calls.",
        mutations=[],  # Auto-populated from findings by _auto_populate
        expected_improvement="Contract fidelity and completeness near 100%",
        effort="medium",
        risk="medium",
    ),
    "upgrade_serving": Scenario(
        name="Switch to vLLM/SGLang",
        description="Replace llama.cpp with vLLM for higher throughput on NVIDIA hardware, or SGLang for Apple Silicon.",
        mutations=[Mutation("upgrade_tech", {
            "category": "llm_serving",
            "from_tech": "llama.cpp",
            "to_tech": "vllm",
            "improvement_factor": 1.7,
        })],
        expected_improvement="1.7-2.3x throughput improvement",
        effort="medium",
        risk="medium",
    ),
    "add_caching_layer": Scenario(
        name="Add response caching for repeated LLM calls",
        description="Cache identical LLM requests to avoid redundant inference. Most effective for classification and extraction.",
        mutations=[Mutation("upgrade_tech", {
            "category": "llm_caching",
            "from_tech": "none",
            "to_tech": "semantic_cache",
            "improvement_factor": 5.0,
        })],
        expected_improvement="5-50x faster for repeated queries (cache hit rate dependent)",
        effort="medium",
        risk="low",
    ),
    "split_monolith": Scenario(
        name="Split oversized service",
        description="Split a service that handles too many endpoints into two focused services.",
        mutations=[Mutation("split_service", {
            "service": "",       # Must be filled in by caller
            "new_service": "",
            "endpoints_to_move": [],
        })],
        expected_improvement="Better isolation, independent scaling, clearer ownership",
        effort="large",
        risk="high",
    ),
}


# ---------------------------------------------------------------------------
# Mutation application functions
# ---------------------------------------------------------------------------

def _apply_mutation(
    mutation: Mutation,
    blueprints: dict[str, ServiceBlueprint],
    findings: list[Finding],
    llm_calls: list[LLMCall],
    hardware: Optional[HardwareProfile],
) -> tuple[list[str], list[str], Optional[ModelFit]]:
    """Apply a single mutation to the in-memory state.

    Returns:
        (risk_factors, tradeoffs, model_fit_or_none)
    """
    dispatch = {
        "add_model": _apply_add_model,
        "add_retry": _apply_add_retry,
        "add_endpoint": _apply_add_endpoint,
        "remove_dead_port": _apply_remove_dead_port,
        "upgrade_tech": _apply_upgrade_tech,
        "add_circuit_breaker": _apply_add_circuit_breaker,
        "split_service": _apply_split_service,
    }
    fn = dispatch.get(mutation.mutation_type)
    if fn is None:
        return [f"Unknown mutation type: {mutation.mutation_type}"], [], None
    return fn(mutation, blueprints, findings, llm_calls, hardware)


def _apply_add_model(
    mutation: Mutation,
    blueprints: dict[str, ServiceBlueprint],
    findings: list[Finding],
    llm_calls: list[LLMCall],
    hardware: Optional[HardwareProfile],
) -> tuple[list[str], list[str], Optional[ModelFit]]:
    """Simulate adding a new LLM model.

    - Check hardware fit
    - Reclassify simple LLM calls to the new model
    - Predict latency improvement for reclassified calls
    """
    risks = []
    tradeoffs = []
    model_fit = None

    name = mutation.params.get("name", "unknown")
    params_b = mutation.params.get("params_b", 4.0)
    quant = mutation.params.get("quant", "Q4_K_M")
    port = mutation.params.get("port", 0)
    task_types = set(mutation.params.get("task_types", []))

    # Check hardware fit
    if hardware:
        model_fit = check_model_fits(name, params_b, quant, hardware)
        if not model_fit.fits:
            risks.append(
                f"Model {name} ({quant}) needs {model_fit.estimated_vram_gb:.1f} GB "
                f"but only {hardware.gpu_vram_gb:.1f} GB available"
            )
        elif "Tight" in model_fit.notes:
            risks.append(f"Tight fit for {name}: may swap under memory pressure")
    else:
        risks.append("No hardware profile available; cannot verify model fits")

    # Reclassify LLM calls: simple/matching tasks go to the new smaller model
    reclassified = 0
    for call in llm_calls:
        if task_types and call.task_type in task_types:
            reclassified += 1
        elif not task_types and call.estimated_complexity == "simple":
            reclassified += 1

    if reclassified == 0:
        tradeoffs.append(f"No LLM calls match task types {task_types or 'simple'}; model would be idle")
    else:
        tradeoffs.append(
            f"{reclassified} LLM calls would be reclassified to {name} "
            f"(smaller model = faster but potentially lower quality)"
        )

    # Add the model as a pseudo-service in blueprints so port resolves
    if port:
        bp = ServiceBlueprint(
            name=name,
            port=port,
            language="config",
            endpoints=[Endpoint(
                service_name=name, port=port, method="POST",
                path="/v1/chat/completions", file_path="(simulated)",
                line_number=0, framework="openai-api",
            )],
        )
        blueprints[name] = bp

    return risks, tradeoffs, model_fit


def _apply_add_retry(
    mutation: Mutation,
    blueprints: dict[str, ServiceBlueprint],
    findings: list[Finding],
    llm_calls: list[LLMCall],
    hardware: Optional[HardwareProfile],
) -> tuple[list[str], list[str], Optional[ModelFit]]:
    """Simulate adding retries to service calls.

    Marks all outbound calls for the target service (or all services if "*")
    as having retry_config, which improves the resilience score.
    """
    risks = []
    tradeoffs = []
    target = mutation.params.get("service", "*")
    max_attempts = mutation.params.get("max_attempts", 3)
    initial_delay = mutation.params.get("initial_delay_ms", 100)

    retry_config = {
        "max_attempts": max_attempts,
        "initial_delay_ms": initial_delay,
        "backoff": "exponential",
    }

    modified = 0
    for bp in blueprints.values():
        if target != "*" and bp.name != target:
            continue
        for call in bp.outbound_calls:
            if call.retry_config is None:
                call.retry_config = retry_config
                modified += 1

    if modified == 0:
        tradeoffs.append("All calls already have retry configs")
    else:
        tradeoffs.append(
            f"Added retry ({max_attempts} attempts) to {modified} calls; "
            f"worst-case latency increases by ~{max_attempts}x on failures"
        )
        risks.append("Retries on non-idempotent endpoints can cause duplicate side effects")

    return risks, tradeoffs, None


def _apply_add_endpoint(
    mutation: Mutation,
    blueprints: dict[str, ServiceBlueprint],
    findings: list[Finding],
    llm_calls: list[LLMCall],
    hardware: Optional[HardwareProfile],
) -> tuple[list[str], list[str], Optional[ModelFit]]:
    """Simulate adding a missing endpoint to a service."""
    risks = []
    tradeoffs = []
    service = mutation.params.get("service", "")
    method = mutation.params.get("method", "GET")
    path = mutation.params.get("path", "")

    bp = blueprints.get(service)
    if bp is None:
        risks.append(f"Service {service} not found in blueprints")
        return risks, tradeoffs, None

    ep = Endpoint(
        service_name=service,
        port=bp.port,
        method=method,
        path=path,
        file_path="(simulated)",
        line_number=0,
    )
    bp.endpoints.append(ep)

    # Remove phantom_call / missing_endpoint findings that this would fix
    resolved = []
    for f in findings:
        if f.finding_type in ("phantom_call", "missing_endpoint"):
            if f"{method} " in f.endpoint and path in f.endpoint:
                if service in f.details:
                    resolved.append(f)
    for f in resolved:
        findings.remove(f)

    tradeoffs.append(f"New endpoint {method} {path} on {service} requires implementation")

    return risks, tradeoffs, None


def _apply_remove_dead_port(
    mutation: Mutation,
    blueprints: dict[str, ServiceBlueprint],
    findings: list[Finding],
    llm_calls: list[LLMCall],
    hardware: Optional[HardwareProfile],
) -> tuple[list[str], list[str], Optional[ModelFit]]:
    """Remove all outbound calls to a specific port.

    Also removes corresponding unknown_service findings.
    """
    risks = []
    tradeoffs = []
    port = mutation.params.get("port", 0)

    removed = 0
    for bp in blueprints.values():
        before = len(bp.outbound_calls)
        bp.outbound_calls = [c for c in bp.outbound_calls if c.callee_port != port]
        removed += before - len(bp.outbound_calls)

    # Remove corresponding findings
    resolved = []
    for f in findings:
        if f.finding_type == "unknown_service" and f"port {port}" in f.details:
            resolved.append(f)
    for f in resolved:
        findings.remove(f)

    if removed == 0:
        tradeoffs.append(f"No calls to port {port} found")
    else:
        tradeoffs.append(
            f"Removed {removed} calls to port {port}; "
            f"features depending on that port will break until migrated"
        )
        risks.append(f"Any feature that relied on port {port} will lose functionality")

    return risks, tradeoffs, None


def _apply_upgrade_tech(
    mutation: Mutation,
    blueprints: dict[str, ServiceBlueprint],
    findings: list[Finding],
    llm_calls: list[LLMCall],
    hardware: Optional[HardwareProfile],
) -> tuple[list[str], list[str], Optional[ModelFit]]:
    """Simulate a technology upgrade.

    Applies the improvement_factor as a note but does not change the
    blueprint structure (the serving layer is infrastructure, not a
    service endpoint).
    """
    risks = []
    tradeoffs = []

    category = mutation.params.get("category", "")
    from_tech = mutation.params.get("from_tech", "")
    to_tech = mutation.params.get("to_tech", "")
    factor = mutation.params.get("improvement_factor", 1.0)

    if category == "llm_serving":
        tradeoffs.append(
            f"Switching from {from_tech} to {to_tech}: "
            f"~{factor:.1f}x throughput improvement expected"
        )
        if to_tech == "vllm" and hardware and hardware.gpu_type == "apple_silicon":
            risks.append("vLLM does not support Apple Silicon / Metal; requires NVIDIA GPU")
        if to_tech in ("vllm", "sglang"):
            risks.append(f"{to_tech} requires different model format and config migration")
            tradeoffs.append("Lose llama.cpp GGUF flexibility; must use safetensors/AWQ formats")
    elif category == "llm_caching":
        tradeoffs.append(
            f"Adding {to_tech}: up to {factor:.0f}x improvement on cache hits"
        )
        risks.append("Cache invalidation complexity; stale results on prompt changes")
        tradeoffs.append("Additional memory usage for cache storage")
    else:
        tradeoffs.append(f"Upgrade {category}: {from_tech} -> {to_tech} ({factor:.1f}x)")

    return risks, tradeoffs, None


def _apply_add_circuit_breaker(
    mutation: Mutation,
    blueprints: dict[str, ServiceBlueprint],
    findings: list[Finding],
    llm_calls: list[LLMCall],
    hardware: Optional[HardwareProfile],
) -> tuple[list[str], list[str], Optional[ModelFit]]:
    """Simulate adding circuit breakers to service calls.

    Circuit breakers are modeled as an enhancement to retry_config.
    """
    risks = []
    tradeoffs = []
    target = mutation.params.get("service", "*")

    modified = 0
    for bp in blueprints.values():
        if target != "*" and bp.name != target:
            continue
        for call in bp.outbound_calls:
            if call.retry_config is None:
                call.retry_config = {"circuit_breaker": True}
            else:
                call.retry_config["circuit_breaker"] = True
            modified += 1

    if modified == 0:
        tradeoffs.append("No calls found to add circuit breakers to")
    else:
        tradeoffs.append(
            f"Added circuit breaker to {modified} calls; "
            f"failing services will be short-circuited after threshold"
        )
        risks.append(
            "Circuit breakers can cause cascading open-circuits if thresholds are too aggressive"
        )

    return risks, tradeoffs, None


def _apply_split_service(
    mutation: Mutation,
    blueprints: dict[str, ServiceBlueprint],
    findings: list[Finding],
    llm_calls: list[LLMCall],
    hardware: Optional[HardwareProfile],
) -> tuple[list[str], list[str], Optional[ModelFit]]:
    """Simulate splitting a service into two.

    Moves specified endpoints to a new service and rewires callers.
    """
    risks = []
    tradeoffs = []

    service = mutation.params.get("service", "")
    new_service = mutation.params.get("new_service", "")
    endpoints_to_move = set(mutation.params.get("endpoints_to_move", []))

    bp = blueprints.get(service)
    if bp is None:
        risks.append(f"Service {service} not found in blueprints")
        return risks, tradeoffs, None

    if not new_service:
        risks.append("No new_service name specified")
        return risks, tradeoffs, None

    # Create the new service blueprint
    moved_endpoints = []
    remaining_endpoints = []
    for ep in bp.endpoints:
        ep_key = f"{ep.method} {ep.path}"
        if ep_key in endpoints_to_move or ep.path in endpoints_to_move:
            moved_ep = copy.deepcopy(ep)
            moved_ep.service_name = new_service
            moved_endpoints.append(moved_ep)
        else:
            remaining_endpoints.append(ep)

    if not moved_endpoints:
        risks.append(f"No matching endpoints found to move from {service}")
        return risks, tradeoffs, None

    # Assign a new port (use max existing + 1 as a rough heuristic)
    max_port = max((b.port for b in blueprints.values()), default=9000)
    new_port = max_port + 1

    new_bp = ServiceBlueprint(
        name=new_service,
        port=new_port,
        language=bp.language,
        endpoints=moved_endpoints,
        outbound_calls=[],  # Simplified: don't split outbound calls
    )
    blueprints[new_service] = new_bp
    bp.endpoints = remaining_endpoints

    # Rewire callers: calls to the moved endpoints now go to new_service
    moved_paths = {ep.path for ep in moved_endpoints}
    for other_bp in blueprints.values():
        for call in other_bp.outbound_calls:
            if call.callee_service == service and call.path in moved_paths:
                call.callee_service = new_service
                call.callee_port = new_port

    tradeoffs.append(
        f"Split {len(moved_endpoints)} endpoints from {service} into {new_service}; "
        f"requires new deployment, service discovery update, and monitoring"
    )
    risks.append("Split may break shared state or transactions between the two halves")
    risks.append(f"All callers of moved endpoints must update service discovery to {new_service}")

    return risks, tradeoffs, None


# ---------------------------------------------------------------------------
# Auto-populate mutations from findings
# ---------------------------------------------------------------------------

def _auto_populate(
    scenario: Scenario,
    findings: list[Finding],
) -> None:
    """Auto-populate mutations for scenarios that derive from findings.

    Modifies scenario.mutations in place.
    """
    if scenario.name == "Remove all dead port references":
        # Find all unique dead ports from unknown_service findings
        dead_ports = set()
        for f in findings:
            if f.finding_type == "unknown_service":
                # Extract port from details like "calls port 8999 which maps to no known service"
                for word in f.details.split():
                    if word.isdigit():
                        port = int(word)
                        if 1024 < port < 65535:
                            dead_ports.add(port)
                            break
        for port in sorted(dead_ports):
            scenario.mutations.append(Mutation("remove_dead_port", {"port": port}))

    elif scenario.name == "Fix all contract violations":
        # Add missing endpoints for phantom_call findings
        seen = set()
        for f in findings:
            if f.finding_type == "phantom_call" and "endpoint not found" in f.details:
                # Parse: "service calls METHOD target_service/path but endpoint not found"
                parts = f.endpoint.split(" ", 1)
                if len(parts) == 2:
                    method, path = parts
                    # Extract target service from details
                    target = _extract_target_service(f.details)
                    if target:
                        key = (target, method, path)
                        if key not in seen:
                            seen.add(key)
                            scenario.mutations.append(Mutation("add_endpoint", {
                                "service": target,
                                "method": method,
                                "path": path,
                            }))


def _extract_target_service(details: str) -> str:
    """Extract target service name from a finding details string.

    Handles patterns like:
        "svc-a calls POST svc-b/path but endpoint not found on svc-b"
    """
    marker = "not found on "
    idx = details.find(marker)
    if idx >= 0:
        return details[idx + len(marker):].strip()
    # Fallback: look for "calls METHOD service/path"
    if " calls " in details:
        after_calls = details.split(" calls ", 1)[1]
        # Skip method
        parts = after_calls.split()
        if len(parts) >= 2:
            target_path = parts[1]  # "service/path"
            service = target_path.split("/")[0]
            if service and not service.startswith("/"):
                return service
    return ""


# ---------------------------------------------------------------------------
# Probability calculation
# ---------------------------------------------------------------------------

def calculate_probability(
    scenario: Scenario,
    grade_before: GradeReport,
    grade_after: GradeReport,
    hardware: Optional[HardwareProfile],
    llm_calls: list[LLMCall],
    model_fit: Optional[ModelFit] = None,
) -> float:
    """Calculate probability of success (0.0-1.0).

    Factors:
        1. Hardware fit: Does the change fit on available hardware?
        2. Benchmark confidence: How well-established are the numbers?
        3. Complexity: How many files/services need to change?
        4. Score improvement: Does the simulation predict improvement?
        5. Risk factors: Penalty per identified risk
    """
    probability = 0.5

    # --- Hardware fit (for add_model scenarios) ---
    has_model_mutation = any(m.mutation_type == "add_model" for m in scenario.mutations)
    if has_model_mutation and model_fit:
        if model_fit.fits:
            vram_headroom = hardware.gpu_vram_gb - model_fit.estimated_vram_gb if hardware else 0
            if vram_headroom > 4.0:
                probability += 0.20
            elif vram_headroom > 0:
                probability += 0.10
        else:
            probability -= 0.30
    elif has_model_mutation and not model_fit:
        # No hardware info, uncertain
        probability -= 0.10

    # --- Benchmark confidence ---
    has_upgrade = any(m.mutation_type == "upgrade_tech" for m in scenario.mutations)
    if has_upgrade:
        # Published benchmarks for vLLM/SGLang exist
        for m in scenario.mutations:
            if m.mutation_type == "upgrade_tech":
                to_tech = m.params.get("to_tech", "")
                if to_tech in ("vllm", "sglang"):
                    probability += 0.15  # Well-benchmarked
                else:
                    probability += 0.05  # Estimated

    # --- Complexity (based on effort) ---
    effort = scenario.effort
    if effort == "small":
        probability += 0.15  # Config change or minor code
    elif effort == "medium":
        probability += 0.05
    else:  # large
        probability -= 0.10

    # --- Score improvement ---
    delta = grade_after.overall - grade_before.overall
    if delta > 15:
        probability += 0.15
    elif delta > 5:
        probability += 0.10
    elif delta > 0:
        probability += 0.05
    elif delta < -5:
        probability -= 0.15

    # --- Risk penalty ---
    risk_map = {"low": 0, "medium": 1, "high": 2}
    risk_level = risk_map.get(scenario.risk, 1)
    probability -= risk_level * 0.05

    return min(1.0, max(0.1, round(probability, 2)))


# ---------------------------------------------------------------------------
# Time estimation
# ---------------------------------------------------------------------------

def _estimate_time(scenario: Scenario) -> str:
    """Estimate implementation time based on effort and mutation count."""
    base = _EFFORT_TIME.get(scenario.effort, "unknown")
    n = len(scenario.mutations)
    if n == 0:
        return base
    if n == 1 and scenario.effort == "small":
        return "5-15 minutes"
    if n > 5:
        return "1-3 days"  # Many mutations = integration work
    return base


# ---------------------------------------------------------------------------
# Core simulation functions
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: Scenario,
    blueprints: dict[str, ServiceBlueprint],
    findings: list[Finding],
    grade: GradeReport,
    llm_calls: list[LLMCall],
    hardware: Optional[HardwareProfile],
    config,
) -> ScenarioResult:
    """Run a single what-if scenario simulation.

    1. Deep copy blueprints, findings, and config
    2. Auto-populate mutations from findings if needed
    3. Apply each mutation to the copied state
    4. Re-run simulation + grading on the mutated state
    5. Compare before/after, calculate probability
    """
    # Deep copy mutable state so mutations are isolated
    bp_copy = copy.deepcopy(blueprints)
    findings_copy = copy.deepcopy(findings)
    llm_copy = copy.deepcopy(llm_calls)
    scenario_copy = copy.deepcopy(scenario)

    # Auto-populate mutations for findings-driven scenarios
    if not scenario_copy.mutations:
        _auto_populate(scenario_copy, findings_copy)

    # Apply all mutations
    all_risks = []
    all_tradeoffs = []
    model_fit = None
    for mutation in scenario_copy.mutations:
        risks, tradeoffs, mf = _apply_mutation(
            mutation, bp_copy, findings_copy, llm_copy, hardware,
        )
        all_risks.extend(risks)
        all_tradeoffs.extend(tradeoffs)
        if mf is not None:
            model_fit = mf

    # Re-run full simulation on mutated state (same checks as simulate_all)
    new_findings = []
    new_findings.extend(check_contracts(bp_copy, config))
    if check_schemas:
        new_findings.extend(check_schemas(bp_copy, config))
    new_findings.extend(check_dependency_drift(bp_copy, config))
    new_findings.extend(check_circular_deps(bp_copy))
    if check_security:
        new_findings.extend(check_security(bp_copy, config))

    flow_traces, flow_findings = trace_all_flows(bp_copy, config)
    new_findings.extend(flow_findings)

    # Deduplicate (same logic as simulate_all)
    seen = set()
    deduped = []
    for f in new_findings:
        key = (f.finding_type, f.service, f.endpoint)
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    new_findings = deduped

    # Re-grade
    grade_after = score_system(bp_copy, new_findings, flow_traces, config)

    # Calculate deltas
    score_delta = grade_after.overall - grade.overall
    dimensions = [
        "completeness", "contract_fidelity", "resilience",
        "dependency_hygiene", "coverage", "security",
    ]
    dimension_deltas = {}
    for dim in dimensions:
        before_val = getattr(grade, dim, 0.0)
        after_val = getattr(grade_after, dim, 0.0)
        dimension_deltas[dim] = round(after_val - before_val, 2)

    # Count resolved vs new findings
    old_types = _finding_signature_set(findings)
    new_types = _finding_signature_set(new_findings)
    findings_resolved = len(old_types - new_types)
    findings_new = len(new_types - old_types)

    # Calculate probability
    probability = calculate_probability(
        scenario_copy, grade, grade_after, hardware, llm_calls, model_fit,
    )

    return ScenarioResult(
        scenario=scenario_copy,
        grade_before=grade,
        grade_after=grade_after,
        score_delta=round(score_delta, 2),
        dimension_deltas=dimension_deltas,
        findings_resolved=findings_resolved,
        findings_new=findings_new,
        probability_success=probability,
        risk_factors=all_risks,
        tradeoffs=all_tradeoffs,
        estimated_time=_estimate_time(scenario_copy),
        model_fit=model_fit,
    )


def _finding_signature_set(findings: list[Finding]) -> set[tuple]:
    """Create a set of (type, service, endpoint) tuples for comparison."""
    sigs = set()
    for f in findings:
        sigs.add((f.finding_type, f.service, f.endpoint))
    return sigs


def run_all_presets(
    blueprints: dict[str, ServiceBlueprint],
    findings: list[Finding],
    grade: GradeReport,
    llm_calls: list[LLMCall],
    hardware: Optional[HardwareProfile],
    config,
) -> list[ScenarioResult]:
    """Run all preset scenarios and return results ranked by impact."""
    results = []
    for key, scenario in PRESET_SCENARIOS.items():
        result = run_scenario(
            scenario, blueprints, findings, grade, llm_calls, hardware, config,
        )
        results.append(result)

    # Rank by score_delta descending, then probability descending
    results.sort(key=lambda r: (r.score_delta, r.probability_success), reverse=True)
    return results


def compare_scenarios(results: list[ScenarioResult]) -> dict:
    """Compare multiple scenario results side-by-side.

    Returns a summary dict with rankings across different axes.
    """
    if not results:
        return {"rankings": {}, "best_overall": None, "best_roi": None}

    # Rank by different criteria
    by_impact = sorted(results, key=lambda r: r.score_delta, reverse=True)
    by_probability = sorted(results, key=lambda r: r.probability_success, reverse=True)
    by_findings = sorted(results, key=lambda r: r.findings_resolved, reverse=True)

    # ROI = score_delta * probability / effort_weight
    effort_weights = {"small": 1, "medium": 3, "large": 10}
    by_roi = sorted(
        results,
        key=lambda r: (
            r.score_delta * r.probability_success /
            effort_weights.get(r.scenario.effort, 3)
        ),
        reverse=True,
    )

    return {
        "by_impact": [r.scenario.name for r in by_impact],
        "by_probability": [r.scenario.name for r in by_probability],
        "by_findings_resolved": [r.scenario.name for r in by_findings],
        "by_roi": [r.scenario.name for r in by_roi],
        "best_overall": by_roi[0].scenario.name if by_roi else None,
        "best_quick_win": next(
            (r.scenario.name for r in by_roi if r.scenario.effort == "small"),
            None,
        ),
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_scenario_report(results: list[ScenarioResult]) -> str:
    """Format scenario comparison as a markdown report."""
    if not results:
        return "No scenarios to report."

    lines = []
    lines.append("# What-If Scenario Analysis")
    lines.append("")

    # Summary table
    lines.append("## Scenario Comparison")
    lines.append("")
    lines.append(
        "| Scenario | Score Delta | Grade | Findings Fixed | "
        "New Issues | Probability | Effort | Risk | Time |"
    )
    lines.append(
        "|----------|------------|-------|---------------|"
        "-----------|------------|--------|------|------|"
    )
    for r in results:
        delta_str = f"+{r.score_delta:.1f}" if r.score_delta >= 0 else f"{r.score_delta:.1f}"
        grade_change = f"{r.grade_before.grade} -> {r.grade_after.grade}"
        prob_pct = f"{r.probability_success:.0%}"
        lines.append(
            f"| {r.scenario.name} | {delta_str} | {grade_change} | "
            f"{r.findings_resolved} | {r.findings_new} | {prob_pct} | "
            f"{r.scenario.effort} | {r.scenario.risk} | {r.estimated_time} |"
        )
    lines.append("")

    # Rankings
    comparison = compare_scenarios(results)
    lines.append("## Rankings")
    lines.append("")
    if comparison.get("best_overall"):
        lines.append(f"- **Best overall (ROI)**: {comparison['best_overall']}")
    if comparison.get("best_quick_win"):
        lines.append(f"- **Best quick win**: {comparison['best_quick_win']}")
    lines.append("")

    # Detailed breakdown per scenario
    lines.append("## Detailed Breakdown")
    lines.append("")
    for r in results:
        lines.append(f"### {r.scenario.name}")
        lines.append("")
        lines.append(f"_{r.scenario.description}_")
        lines.append("")
        lines.append(f"**Expected**: {r.scenario.expected_improvement}")
        lines.append("")

        # Dimension deltas
        nonzero_dims = {k: v for k, v in r.dimension_deltas.items() if v != 0}
        if nonzero_dims:
            lines.append("**Dimension changes:**")
            for dim, delta in sorted(nonzero_dims.items(), key=lambda x: -abs(x[1])):
                sign = "+" if delta > 0 else ""
                lines.append(f"  - {dim}: {sign}{delta:.1f}")
            lines.append("")

        # Model fit info
        if r.model_fit:
            fit_str = "fits" if r.model_fit.fits else "DOES NOT FIT"
            lines.append(
                f"**Model**: {r.model_fit.model_name} ({r.model_fit.quantization}) - "
                f"{fit_str}, {r.model_fit.estimated_vram_gb:.1f} GB VRAM"
            )
            if r.model_fit.estimated_tps > 0:
                lines.append(f"  - Estimated throughput: {r.model_fit.estimated_tps:.0f} tok/s")
            lines.append("")

        # Risks
        if r.risk_factors:
            lines.append("**Risks:**")
            for risk in r.risk_factors:
                lines.append(f"  - {risk}")
            lines.append("")

        # Tradeoffs
        if r.tradeoffs:
            lines.append("**Tradeoffs:**")
            for tradeoff in r.tradeoffs:
                lines.append(f"  - {tradeoff}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)
