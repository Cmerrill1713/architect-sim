"""DAG-walk latency estimator for service call graphs.

Walks the service dependency graph and estimates end-to-end latency for
request flows by combining:
1. Runtime log data (actual observed latencies per endpoint)
2. LLM latency formulas (analytical estimation for LLM calls)
3. Heuristic defaults (when no data available)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ..models import ServiceBlueprint, LLMCall
from .hardware import (
    HardwareProfile,
    QUANT_BITS,
    _TPS_PER_ACTIVE_B,
    _chip_key,
    estimate_vram_gb,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HopLatency:
    caller: str
    callee: str
    method: str
    path: str
    estimated_ms: float       # Expected latency for this hop
    source: str               # "runtime_p50", "runtime_p95", "llm_formula", "heuristic"
    confidence: str           # "measured", "estimated", "guess"


@dataclass
class FlowLatency:
    flow_name: str
    hops: list = field(default_factory=list)          # list[HopLatency]
    total_ms: float = 0.0                             # End-to-end estimated latency
    critical_path_ms: float = 0.0                     # Longest sequential chain
    bottleneck: str = ""                              # Which hop is the bottleneck
    breakdown: dict = field(default_factory=dict)     # {category: ms}


# ---------------------------------------------------------------------------
# Heuristic latency defaults
# ---------------------------------------------------------------------------

HEURISTIC_LATENCY_MS: dict[str, float] = {
    "health_check": 5,
    "db_read": 10,
    "db_write": 25,
    "cache_hit": 2,
    "cache_miss": 15,
    "http_internal": 20,
    "http_external": 200,
    "embedding": 50,
    "reranking": 100,
    "llm_simple": 500,
    "llm_moderate": 2000,
    "llm_complex": 5000,
    "file_io": 5,
    "search": 300,
}

# Map heuristic categories to breakdown buckets.
_CATEGORY_MAP: dict[str, str] = {
    "health_check": "network",
    "db_read": "db",
    "db_write": "db",
    "cache_hit": "db",
    "cache_miss": "db",
    "http_internal": "network",
    "http_external": "network",
    "embedding": "llm",
    "reranking": "llm",
    "llm_simple": "llm",
    "llm_moderate": "llm",
    "llm_complex": "llm",
    "file_io": "compute",
    "search": "compute",
}


# ---------------------------------------------------------------------------
# 1. Runtime latency table
# ---------------------------------------------------------------------------

def build_latency_table(runtime_events: list) -> dict:
    """Build {(method, path): {p50: ms, p95: ms, count: N}} from runtime events.

    Only uses successful requests (status < 500) with a positive duration.
    """
    by_endpoint: dict[tuple[str, str], list[float]] = defaultdict(list)
    for event in runtime_events:
        if event.duration_ms > 0 and event.status_code < 500:
            key = (event.method, event.path)
            by_endpoint[key].append(event.duration_ms)

    table: dict[tuple[str, str], dict] = {}
    for key, durations in by_endpoint.items():
        durations.sort()
        n = len(durations)
        table[key] = {
            "p50": durations[n // 2] if n > 0 else 0,
            "p95": durations[int(n * 0.95)] if n > 0 else 0,
            "count": n,
        }
    return table


# ---------------------------------------------------------------------------
# 2. LLM latency formula
# ---------------------------------------------------------------------------

_HTTP_OVERHEAD_MS = 100  # HTTP + serialization overhead for LLM calls
_PREFILL_DECODE_RATIO = 20  # Prefill is ~20x faster than decode on average


def estimate_llm_latency_ms(
    params_b: float,
    quant: str,
    hardware: Optional[HardwareProfile],
    input_tokens: int = 500,
    output_tokens: int = 200,
    active_params_b: Optional[float] = None,
) -> float:
    """Estimate LLM call latency from model config + hardware.

    Formula:
    - decode_tps from hardware.py TPS tables, adjusted for model size
    - prefill_tps = decode_tps * prefill_decode_ratio
    - prefill_latency = input_tokens / prefill_tps * 1000
    - decode_latency = output_tokens / decode_tps * 1000
    - total = prefill + decode + http_overhead

    For MoE models, uses active_params for compute estimation.
    """
    if hardware is None:
        return 0.0

    active = active_params_b if active_params_b else params_b

    key = _chip_key(hardware)
    base_tps_per_b = _TPS_PER_ACTIVE_B.get(key, 0.0)
    if base_tps_per_b <= 0 or active <= 0:
        return 0.0

    # Quantization adjustment (base rate is for Q4_K_M).
    bits = QUANT_BITS.get(quant, 4.85)
    quant_factor = 4.85 / bits

    decode_tps = (base_tps_per_b / active) * quant_factor
    if decode_tps <= 0:
        return 0.0

    prefill_tps = decode_tps * _PREFILL_DECODE_RATIO

    prefill_ms = (input_tokens / prefill_tps) * 1000
    decode_ms = (output_tokens / decode_tps) * 1000
    total_ms = prefill_ms + decode_ms + _HTTP_OVERHEAD_MS
    return round(total_ms, 1)


def _estimate_tokens_for_task(task_type: str) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) estimates by task type."""
    estimates = {
        "classification": (200, 20),
        "routing": (300, 30),
        "extraction": (500, 150),
        "generation": (400, 300),
        "planning": (800, 500),
        "code": (600, 400),
        "unknown": (500, 200),
    }
    return estimates.get(task_type, (500, 200))


# ---------------------------------------------------------------------------
# 3. Hop classifier
# ---------------------------------------------------------------------------

def classify_hop(
    method: str,
    path: str,
    callee: str,
    llm_calls: Optional[list] = None,
) -> str:
    """Classify a hop into a latency category based on path patterns."""
    path_lower = path.lower()

    if any(p in path_lower for p in ("/health", "/live", "/ready", "/metrics")):
        return "health_check"
    if any(p in path_lower for p in ("/v1/chat/completions", "/completions", "/generate")):
        # Refine: if we have LLM call metadata for this callee, use complexity.
        if llm_calls:
            for lc in llm_calls:
                if lc.service_name == callee:
                    complexity = lc.estimated_complexity
                    return f"llm_{complexity}"
        return "llm_moderate"
    if any(p in path_lower for p in ("/embed", "/v1/embeddings")):
        return "embedding"
    if any(p in path_lower for p in ("/rerank", "/v1/rerank")):
        return "reranking"
    if any(p in path_lower for p in ("/search", "/query", "/kb/")):
        return "search"
    if method == "GET":
        return "db_read"
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        return "http_internal"
    return "http_internal"


# ---------------------------------------------------------------------------
# 4. Single-hop latency estimation
# ---------------------------------------------------------------------------

def _estimate_hop(
    caller: str,
    callee: str,
    method: str,
    path: str,
    latency_table: Optional[dict],
    hardware: Optional[HardwareProfile],
    llm_calls: Optional[list],
    percentile: str = "p95",
) -> HopLatency:
    """Estimate latency for a single hop, trying sources in priority order."""

    # --- Priority 1: runtime log data ---
    if latency_table:
        key = (method, path)
        entry = latency_table.get(key)
        if entry and entry["count"] >= 3:
            ms = entry.get(percentile, entry["p95"])
            return HopLatency(
                caller=caller,
                callee=callee,
                method=method,
                path=path,
                estimated_ms=round(ms, 1),
                source=f"runtime_{percentile}",
                confidence="measured",
            )

    # --- Priority 2: LLM formula (for LLM endpoints) ---
    category = classify_hop(method, path, callee, llm_calls)
    if category.startswith("llm_") and hardware and llm_calls:
        # Find the LLM call metadata for this callee.
        for lc in llm_calls:
            if lc.service_name == callee:
                in_tok, out_tok = _estimate_tokens_for_task(lc.task_type)
                if lc.max_tokens > 0:
                    out_tok = min(lc.max_tokens, out_tok)

                # Extract model params from COMMON_MODELS via name, or fall back.
                params_b, active_b, quant = _resolve_model_params(lc.model)
                if params_b > 0:
                    ms = estimate_llm_latency_ms(
                        params_b, quant, hardware, in_tok, out_tok, active_b
                    )
                    if ms > 0:
                        return HopLatency(
                            caller=caller,
                            callee=callee,
                            method=method,
                            path=path,
                            estimated_ms=ms,
                            source="llm_formula",
                            confidence="estimated",
                        )

    # --- Priority 3: heuristic ---
    ms = HEURISTIC_LATENCY_MS.get(category, HEURISTIC_LATENCY_MS["http_internal"])
    return HopLatency(
        caller=caller,
        callee=callee,
        method=method,
        path=path,
        estimated_ms=ms,
        source="heuristic",
        confidence="guess",
    )


def _resolve_model_params(model_name: str) -> tuple[float, float, str]:
    """Resolve model name to (params_b, active_params_b, quant).

    Tries to match against COMMON_MODELS from hardware.py, falling back to
    rough heuristics extracted from the name itself.
    """
    from .hardware import COMMON_MODELS

    if not model_name:
        return (0.0, 0.0, "Q4_K_M")

    name_lower = model_name.lower()

    # Direct match against known models.
    for m in COMMON_MODELS:
        if m["name"].lower() in name_lower or name_lower in m["name"].lower():
            params = m["params"]
            active = m.get("active_params", params)
            return (params, active, "Q4_K_M")

    # Heuristic: extract number from name (e.g. "qwen3-8b" -> 8).
    import re
    match = re.search(r'(\d+)[bB]', model_name)
    if match:
        params = float(match.group(1))
        return (params, params, "Q4_K_M")

    return (0.0, 0.0, "Q4_K_M")


# ---------------------------------------------------------------------------
# 5. DAG walk
# ---------------------------------------------------------------------------

_MAX_DEPTH = 10


def estimate_flow_latency(
    flow_name: str,
    entry_service: str,
    blueprints: dict,
    latency_table: Optional[dict] = None,
    hardware: Optional[HardwareProfile] = None,
    llm_calls: Optional[list] = None,
    percentile: str = "p95",
) -> FlowLatency:
    """Estimate end-to-end latency for a request flow.

    Algorithm:
    1. Start from entry_service
    2. For each outbound call, estimate hop latency (runtime > formula > heuristic)
    3. All calls from a single service are treated as sequential (conservative)
    4. Recursive: follow the call chain through downstream services
    5. Track the critical path (longest sequential chain)
    6. Identify bottleneck (highest single hop)

    Cycle detection: stop if we revisit a service.
    Depth limit: 10 hops deep.
    """
    hops: list[HopLatency] = []
    # service -> total latency for that subtree (for critical path)
    subtree_latency: dict[str, float] = {}
    visited: set[str] = set()

    def _walk(service_name: str, depth: int) -> float:
        """Walk from service_name, return total latency of this subtree."""
        if depth > _MAX_DEPTH or service_name in visited:
            return 0.0
        visited.add(service_name)

        bp = blueprints.get(service_name)
        if bp is None:
            return 0.0

        service_total = 0.0
        for call in bp.outbound_calls:
            callee = call.callee_service
            if callee.startswith("unknown:"):
                continue

            hop = _estimate_hop(
                caller=service_name,
                callee=callee,
                method=call.method,
                path=call.path,
                latency_table=latency_table,
                hardware=hardware,
                llm_calls=llm_calls,
                percentile=percentile,
            )
            hops.append(hop)

            # Recurse into callee to get downstream latency.
            downstream = _walk(callee, depth + 1)

            # Sequential: sum hop latency + downstream latency.
            service_total += hop.estimated_ms + downstream

        subtree_latency[service_name] = service_total
        return service_total

    total_ms = _walk(entry_service, 0)

    # Identify bottleneck: single hop with highest latency.
    bottleneck = ""
    if hops:
        worst = max(hops, key=lambda h: h.estimated_ms)
        bottleneck = f"{worst.caller} -> {worst.callee} {worst.method} {worst.path} ({worst.estimated_ms:.0f}ms)"

    # Build breakdown by category.
    breakdown: dict[str, float] = defaultdict(float)
    for hop in hops:
        cat = classify_hop(hop.method, hop.path, hop.callee, llm_calls)
        bucket = _CATEGORY_MAP.get(cat, "compute")
        breakdown[bucket] += hop.estimated_ms

    # Critical path: the longest single chain from entry to a leaf.
    critical_path_ms = _compute_critical_path(entry_service, hops, blueprints)

    return FlowLatency(
        flow_name=flow_name,
        hops=hops,
        total_ms=round(total_ms, 1),
        critical_path_ms=round(critical_path_ms, 1),
        bottleneck=bottleneck,
        breakdown=dict(breakdown),
    )


def _compute_critical_path(
    entry: str,
    hops: list[HopLatency],
    blueprints: dict,
) -> float:
    """Find the longest sequential chain through the DAG.

    Uses a simple DFS from the entry. For each service, among its outbound
    hops, the critical path goes through the one whose subtotal (hop + downstream)
    is largest.
    """
    # Build adjacency: caller -> [(callee, hop_ms)]
    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for hop in hops:
        adj[hop.caller].append((hop.callee, hop.estimated_ms))

    visited: set[str] = set()

    def _longest(service: str) -> float:
        if service in visited:
            return 0.0
        visited.add(service)
        children = adj.get(service, [])
        if not children:
            return 0.0
        # Take the maximum (hop + downstream) among children.
        return max(hop_ms + _longest(callee) for callee, hop_ms in children)

    return _longest(entry)


# ---------------------------------------------------------------------------
# 6. Discover and estimate all flows
# ---------------------------------------------------------------------------

def estimate_all_flows(
    blueprints: dict,
    config=None,
    runtime_events: Optional[list] = None,
    hardware: Optional[HardwareProfile] = None,
    llm_calls: Optional[list] = None,
    percentile: str = "p95",
) -> list[FlowLatency]:
    """Estimate latency for all discoverable flows.

    Finds entry points (services with inbound calls but no callers),
    traces each flow, and estimates latency.
    """
    latency_table = None
    if runtime_events:
        latency_table = build_latency_table(runtime_events)

    # Find entry points: services that have endpoints but nobody calls them.
    called_services: set[str] = set()
    for bp in blueprints.values():
        for call in bp.outbound_calls:
            if not call.callee_service.startswith("unknown:"):
                called_services.add(call.callee_service)

    entry_points = [
        name for name, bp in blueprints.items()
        if bp.endpoints and name not in called_services
    ]

    # Fallback: if no clear entry points, use services with the most outbound calls.
    if not entry_points:
        by_calls = sorted(
            blueprints.items(),
            key=lambda x: len(x[1].outbound_calls),
            reverse=True,
        )
        entry_points = [name for name, bp in by_calls[:3] if bp.outbound_calls]

    flows: list[FlowLatency] = []
    for entry in entry_points:
        flow = estimate_flow_latency(
            flow_name=f"flow_from_{entry}",
            entry_service=entry,
            blueprints=blueprints,
            latency_table=latency_table,
            hardware=hardware,
            llm_calls=llm_calls,
            percentile=percentile,
        )
        flows.append(flow)

    # Sort: slowest first.
    flows.sort(key=lambda f: f.total_ms, reverse=True)
    return flows


# ---------------------------------------------------------------------------
# 7. Report formatting
# ---------------------------------------------------------------------------

def format_latency_report(flows: list[FlowLatency]) -> str:
    """Format latency estimates as markdown.

    Shows per-flow totals, breakdown tables, and optimization suggestions.
    """
    lines: list[str] = []
    lines.append("# Latency Estimation Report")
    lines.append("")

    if not flows:
        lines.append("No flows discovered to estimate.")
        return "\n".join(lines)

    # Summary table.
    lines.append("## Flow Summary")
    lines.append("")
    lines.append("| Flow | Total (ms) | Critical Path (ms) | Hops | Bottleneck |")
    lines.append("|------|-----------|--------------------:|-----:|------------|")
    for flow in flows:
        lines.append(
            f"| {flow.flow_name} | {flow.total_ms:,.0f} | {flow.critical_path_ms:,.0f} "
            f"| {len(flow.hops)} | {flow.bottleneck} |"
        )
    lines.append("")

    # Per-flow details.
    for flow in flows:
        lines.append(f"## {flow.flow_name}")
        lines.append("")

        if flow.hops:
            lines.append("### Hop Detail")
            lines.append("")
            lines.append("| # | Caller | Callee | Endpoint | Latency (ms) | Source | Confidence |")
            lines.append("|--:|--------|--------|----------|-------------:|--------|------------|")
            for i, hop in enumerate(flow.hops, 1):
                lines.append(
                    f"| {i} | {hop.caller} | {hop.callee} "
                    f"| {hop.method} {hop.path} | {hop.estimated_ms:,.1f} "
                    f"| {hop.source} | {hop.confidence} |"
                )
            lines.append("")

        # Breakdown.
        if flow.breakdown:
            lines.append("### Latency Breakdown")
            lines.append("")
            total = sum(flow.breakdown.values()) or 1
            lines.append("| Category | Time (ms) | Share |")
            lines.append("|----------|----------:|------:|")
            for cat in sorted(flow.breakdown, key=lambda c: flow.breakdown[c], reverse=True):
                ms = flow.breakdown[cat]
                pct = (ms / total) * 100
                lines.append(f"| {cat} | {ms:,.0f} | {pct:.0f}% |")
            lines.append("")

    # Optimization suggestions.
    suggestions = _generate_suggestions(flows)
    if suggestions:
        lines.append("## Optimization Suggestions")
        lines.append("")
        for s in suggestions:
            lines.append(f"- {s}")
        lines.append("")

    # Confidence summary.
    all_hops = [h for f in flows for h in f.hops]
    measured = sum(1 for h in all_hops if h.confidence == "measured")
    estimated = sum(1 for h in all_hops if h.confidence == "estimated")
    guessed = sum(1 for h in all_hops if h.confidence == "guess")
    total_hops = len(all_hops)
    if total_hops > 0:
        lines.append("## Confidence")
        lines.append("")
        lines.append(
            f"- **Measured** (from runtime logs): {measured}/{total_hops} hops "
            f"({measured * 100 // total_hops}%)"
        )
        lines.append(
            f"- **Estimated** (LLM formula): {estimated}/{total_hops} hops "
            f"({estimated * 100 // total_hops}%)"
        )
        lines.append(
            f"- **Guess** (heuristic defaults): {guessed}/{total_hops} hops "
            f"({guessed * 100 // total_hops}%)"
        )
        if guessed > total_hops * 0.5:
            lines.append("")
            lines.append(
                "> **Low confidence**: Over half the hops use heuristic defaults. "
                "Feed runtime logs via `--logs` to improve accuracy."
            )
        lines.append("")

    return "\n".join(lines)


def _generate_suggestions(flows: list[FlowLatency]) -> list[str]:
    """Generate optimization suggestions based on latency breakdown patterns."""
    suggestions: list[str] = []

    # Aggregate breakdown across all flows.
    total_by_cat: dict[str, float] = defaultdict(float)
    grand_total = 0.0
    for flow in flows:
        for cat, ms in flow.breakdown.items():
            total_by_cat[cat] += ms
            grand_total += ms

    if grand_total <= 0:
        return suggestions

    # LLM dominance.
    llm_ms = total_by_cat.get("llm", 0)
    llm_pct = (llm_ms / grand_total) * 100
    if llm_pct > 50:
        suggestions.append(
            f"LLM calls account for {llm_pct:.0f}% of total latency ({llm_ms:,.0f}ms). "
            f"Consider model routing: use a smaller model (e.g. 4B) for "
            f"classification/routing tasks and reserve larger models for complex generation."
        )
    elif llm_pct > 30:
        suggestions.append(
            f"LLM calls account for {llm_pct:.0f}% of total latency. "
            f"Consider caching repeated prompts or reducing output token limits."
        )

    # Network dominance.
    network_ms = total_by_cat.get("network", 0)
    network_pct = (network_ms / grand_total) * 100
    if network_pct > 30:
        suggestions.append(
            f"Network/inter-service calls account for {network_pct:.0f}% of total latency "
            f"({network_ms:,.0f}ms). Consider batching calls, adding response caching, "
            f"or merging tightly-coupled services."
        )

    # DB dominance.
    db_ms = total_by_cat.get("db", 0)
    db_pct = (db_ms / grand_total) * 100
    if db_pct > 30:
        suggestions.append(
            f"Database calls account for {db_pct:.0f}% of total latency ({db_ms:,.0f}ms). "
            f"Consider adding a Redis cache layer or optimizing query patterns."
        )

    # Bottleneck concentration: if one service appears as bottleneck in multiple flows.
    bottleneck_services: dict[str, int] = defaultdict(int)
    for flow in flows:
        if flow.bottleneck:
            # Extract callee from "caller -> callee ..." format.
            parts = flow.bottleneck.split(" -> ")
            if len(parts) >= 2:
                callee = parts[1].split()[0]
                bottleneck_services[callee] += 1

    for svc, count in bottleneck_services.items():
        if count >= 2:
            suggestions.append(
                f"**{svc}** is the bottleneck in {count} flows. "
                f"Consider scaling it horizontally or optimizing its hot paths."
            )

    # Guess-heavy warning.
    all_hops = [h for f in flows for h in f.hops]
    guess_count = sum(1 for h in all_hops if h.confidence == "guess")
    if all_hops and guess_count > len(all_hops) * 0.6:
        suggestions.append(
            f"{guess_count}/{len(all_hops)} hops use heuristic guesses. "
            f"Collect runtime logs with duration data to improve accuracy."
        )

    return suggestions
