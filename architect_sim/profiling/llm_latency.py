"""Analytical LLM inference latency estimation.

Predicts decode/prefill throughput and end-to-end latency from model config
and hardware specs — no inference required.  Based on the llm-analysis approach
(github.com/cli99/llm-analysis).

Core insight: autoregressive LLM decode is memory-bandwidth bound.  The
fundamental formula is:

    decode_tps = memory_bandwidth_bytes_per_sec / model_size_bytes

For Apple Silicon unified memory, bandwidth is the single bottleneck.
For NVIDIA GPUs, HBM bandwidth dominates decode while tensor cores dominate
prefill.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .hardware import (
    COMMON_MODELS,
    QUANT_BITS,
    HardwareProfile,
    estimate_vram_gb,
    _chip_key,
    _recommend_backend,
)


# ---------------------------------------------------------------------------
# Memory bandwidth table (GB/s)
# ---------------------------------------------------------------------------

MEMORY_BANDWIDTH_GB_S: dict[str, float] = {
    # Apple Silicon — unified memory
    "Apple M1": 68.25,
    "Apple M1 Pro": 200,
    "Apple M1 Max": 400,
    "Apple M1 Ultra": 800,
    "Apple M2": 100,
    "Apple M2 Pro": 200,
    "Apple M2 Max": 400,
    "Apple M2 Ultra": 800,
    "Apple M3": 100,
    "Apple M3 Pro": 150,
    "Apple M3 Max": 400,
    "Apple M4": 120,
    "Apple M4 Pro": 273,
    "Apple M4 Max": 546,
    # NVIDIA GPUs — HBM / GDDR6X bandwidth
    "NVIDIA A100 40GB": 1555,
    "NVIDIA A100 80GB": 2039,
    "NVIDIA H100": 3350,
    "NVIDIA RTX 4090": 1008,
    "NVIDIA RTX 3090": 936,
    "NVIDIA RTX 4080": 717,
    "NVIDIA RTX 3080": 760,
    "NVIDIA RTX 4070": 504,
}

# Map _chip_key() values to bandwidth table keys for lookup.
_CHIP_KEY_TO_BW: dict[str, str] = {
    "m1": "Apple M1",
    "m1_pro": "Apple M1 Pro",
    "m1_max": "Apple M1 Max",
    "m1_ultra": "Apple M1 Ultra",
    "m2": "Apple M2",
    "m2_pro": "Apple M2 Pro",
    "m2_max": "Apple M2 Max",
    "m2_ultra": "Apple M2 Ultra",
    "m3": "Apple M3",
    "m3_pro": "Apple M3 Pro",
    "m3_max": "Apple M3 Max",
    "m4": "Apple M4",
    "m4_pro": "Apple M4 Pro",
    "m4_max": "Apple M4 Max",
    "rtx_4090": "NVIDIA RTX 4090",
    "rtx_4080": "NVIDIA RTX 4080",
    "rtx_3090": "NVIDIA RTX 3090",
    "rtx_3080": "NVIDIA RTX 3080",
    "rtx_4070": "NVIDIA RTX 4070",
    "a100": "NVIDIA A100 80GB",
    "h100": "NVIDIA H100",
}

# Typical transformer architecture constants (Llama-family defaults).
# These are used for KV cache sizing when exact config is unavailable.
# Ratio: hidden_dim / params_b  (approximate, works for 1B-70B range).
_HIDDEN_DIM_PER_B: float = 512  # e.g. 8B model ~ 4096 hidden
_NUM_LAYERS_PER_B: float = 4.0  # e.g. 8B model ~ 32 layers

# Backend overhead: HTTP round-trip + tokenization + sampling (ms).
_OVERHEAD_MS: float = 15.0

# Memory bandwidth efficiency — real-world vs theoretical peak.
# llama.cpp / MLX typically achieve 75-85% of peak on Apple Silicon,
# slightly less on NVIDIA due to kernel launch overhead.
_BW_EFFICIENCY: dict[str, float] = {
    "apple_silicon": 0.82,
    "nvidia": 0.78,
    "amd": 0.70,
    "none": 0.60,  # CPU-only fallback
}

# Prefill speedup over decode (compute-bound vs memory-bound).
# Depends on batch size; these are for single-request (batch=1).
_PREFILL_SPEEDUP: float = 10.0

# Speculative decoding defaults.
_SPEC_DRAFT_LENGTH: int = 5
_SPEC_ACCEPTANCE_DEFAULT: float = 0.80


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LLMLatencyEstimate:
    model_name: str
    params_b: float
    active_params_b: float          # For MoE; otherwise same as params_b
    quantization: str
    hardware: str                   # CPU/GPU model name

    # Throughput
    decode_tps: float               # Tokens/sec (generation)
    prefill_tps: float              # Tokens/sec (prompt processing)

    # Latency for a specific request shape
    input_tokens: int
    output_tokens: int
    prefill_ms: float               # Time to process input
    decode_ms: float                # Time to generate output
    overhead_ms: float              # HTTP + serialization
    total_ms: float                 # End-to-end

    # Speculative decoding
    spec_decode_tps: float          # Effective TPS with draft model
    spec_total_ms: float            # Total with speculative decoding

    # Context scaling
    context_penalty_ms: float       # Extra latency at max_context vs 2K baseline
    max_context: int                # Before OOM

    # Memory
    model_size_gb: float            # Weight memory footprint
    vram_required_gb: float         # Total (weights + KV + overhead)

    confidence: str = "formula"     # "formula" or "calibrated"


@dataclass
class LatencyComparison:
    """Compare latency across model options for a task type."""
    task_type: str                  # "classification", "generation", etc.
    input_tokens: int
    output_tokens: int
    estimates: list[LLMLatencyEstimate] = field(default_factory=list)
    recommended: str = ""           # Best model for this task
    speedup_vs_current: float = 1.0 # vs current setup


# ---------------------------------------------------------------------------
# Task type definitions
# ---------------------------------------------------------------------------

TASK_TOKEN_BUDGETS: dict[str, tuple[int, int]] = {
    "classification": (200, 20),
    "extraction": (500, 100),
    "generation": (500, 500),
    "planning": (1000, 500),
    "code": (2000, 1000),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_bandwidth_gb_s(hw: HardwareProfile) -> float:
    """Resolve effective memory bandwidth for hardware in GB/s."""
    key = _chip_key(hw)
    bw_name = _CHIP_KEY_TO_BW.get(key, "")
    raw_bw = MEMORY_BANDWIDTH_GB_S.get(bw_name, 0.0)

    if raw_bw == 0.0:
        # Fallback: try direct match against cpu_model or gpu_model
        combined = hw.cpu_model + " " + hw.gpu_model
        for name, bw in MEMORY_BANDWIDTH_GB_S.items():
            if name.lower() in combined.lower():
                raw_bw = bw
                break

    if raw_bw == 0.0:
        # Last resort: estimate from RAM size (very rough, CPU-only)
        # DDR5 ~50 GB/s dual-channel, DDR4 ~35 GB/s
        raw_bw = 50.0

    efficiency = _BW_EFFICIENCY.get(hw.gpu_type, 0.65)
    return raw_bw * efficiency


def _model_size_bytes(params_b: float, quant: str) -> float:
    """Weight memory in bytes."""
    bits = QUANT_BITS.get(quant, 4.85)
    return params_b * 1e9 * bits / 8


def _kv_cache_bytes_per_token(params_b: float) -> float:
    """Approximate KV cache bytes per token (fp16 KV).

    KV cache per token = 2 (K+V) * num_layers * hidden_dim * 2 (fp16 bytes).
    We approximate num_layers and hidden_dim from param count.
    """
    num_layers = max(1, int(params_b * _NUM_LAYERS_PER_B))
    hidden_dim = max(64, int(params_b * _HIDDEN_DIM_PER_B))
    return 2 * num_layers * hidden_dim * 2  # fp16


def _kv_overhead_ms(
    params_b: float,
    context_length: int,
    bandwidth_bytes_per_sec: float,
) -> float:
    """Extra decode latency from KV cache reads at given context length.

    Returns the per-token overhead in ms for reading the KV cache.
    At short contexts this is negligible; at 128K it can add 20-40%.
    """
    if bandwidth_bytes_per_sec <= 0 or context_length <= 0:
        return 0.0
    kv_bytes = _kv_cache_bytes_per_token(params_b) * context_length
    # This is per-token overhead: time to read KV cache for one decode step.
    return (kv_bytes / bandwidth_bytes_per_sec) * 1000


def _context_penalty_ms(
    params_b: float,
    max_context: int,
    output_tokens: int,
    bandwidth_bytes_per_sec: float,
) -> float:
    """Total extra latency at max_context vs a 2K baseline, over all output tokens."""
    baseline_ctx = 2048
    if max_context <= baseline_ctx:
        return 0.0
    # Average context during generation: roughly (max_context + max_context + output_tokens) / 2
    # but the penalty is the difference vs the baseline average.
    avg_ctx_max = max_context + output_tokens / 2
    avg_ctx_base = baseline_ctx + output_tokens / 2
    delta_per_token = _kv_overhead_ms(params_b, avg_ctx_max, bandwidth_bytes_per_sec) - \
                      _kv_overhead_ms(params_b, avg_ctx_base, bandwidth_bytes_per_sec)
    return max(0.0, delta_per_token * output_tokens)


# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------

def estimate_llm_latency(
    model_name: str,
    params_b: float,
    quant: str,
    hardware: HardwareProfile,
    input_tokens: int = 500,
    output_tokens: int = 200,
    active_params_b: Optional[float] = None,
    draft_model: bool = False,
    draft_acceptance: float = _SPEC_ACCEPTANCE_DEFAULT,
    context_length: Optional[int] = None,
) -> LLMLatencyEstimate:
    """Estimate LLM inference latency from model + hardware specs.

    This is pure math -- no inference required.

    For MoE models, pass *active_params_b* (the expert subset loaded per
    token).  The full *params_b* is used for weight memory and KV cache;
    *active_params_b* is used for decode throughput (only active weights
    stream through memory each step).

    Parameters
    ----------
    model_name : str
        Human-readable model identifier.
    params_b : float
        Total parameters in billions.
    quant : str
        Quantization format key (must be in QUANT_BITS).
    hardware : HardwareProfile
        Target hardware from ``detect_hardware()`` or manually constructed.
    input_tokens, output_tokens : int
        Request shape for latency estimate.
    active_params_b : float, optional
        Active parameters per token for MoE models.  Defaults to *params_b*.
    draft_model : bool
        Whether speculative decoding is available.
    draft_acceptance : float
        Expected acceptance rate for speculative decoding (0-1).
    context_length : int, optional
        Context length for VRAM / KV sizing.  Defaults to
        ``input_tokens + output_tokens``.
    """
    if active_params_b is None:
        active_params_b = params_b

    if context_length is None:
        context_length = input_tokens + output_tokens

    # --- Memory bandwidth ---
    bw_gb_s = _get_bandwidth_gb_s(hardware)
    bw_bytes_s = bw_gb_s * 1e9

    # --- Model sizes ---
    # Full model in memory (all params for MoE, since weights must be resident)
    full_model_bytes = _model_size_bytes(params_b, quant)
    # Active params per decode step (MoE uses only active experts)
    active_model_bytes = _model_size_bytes(active_params_b, quant)

    model_size_gb = full_model_bytes / 1e9

    # --- Decode throughput ---
    # Time per token = active_model_bytes / bandwidth
    if bw_bytes_s > 0:
        time_per_token_s = active_model_bytes / bw_bytes_s
        # Add KV cache overhead for average context during generation
        avg_context = input_tokens + output_tokens / 2
        kv_overhead_s = _kv_overhead_ms(params_b, avg_context, bw_bytes_s) / 1000
        effective_time_per_token_s = time_per_token_s + kv_overhead_s
        decode_tps = 1.0 / effective_time_per_token_s if effective_time_per_token_s > 0 else 0.0
    else:
        decode_tps = 0.0

    # --- Prefill throughput ---
    # Prefill is compute-bound and parallelized; much faster per token.
    prefill_tps = decode_tps * _PREFILL_SPEEDUP

    # --- Request latency ---
    prefill_ms = (input_tokens / prefill_tps * 1000) if prefill_tps > 0 else 0.0
    decode_ms = (output_tokens / decode_tps * 1000) if decode_tps > 0 else 0.0
    total_ms = prefill_ms + decode_ms + _OVERHEAD_MS

    # --- Speculative decoding ---
    if draft_model and draft_acceptance > 0:
        # Effective throughput: each draft step proposes draft_length tokens,
        # acceptance_rate fraction are accepted on average.
        accepted_per_step = draft_acceptance * _SPEC_DRAFT_LENGTH
        # But we also pay the cost of verifying (one forward pass of big model).
        # Net speedup ~ accepted_per_step / (1 + overhead_ratio).
        # Simplified: effective_tps ~ decode_tps * (1 + accepted_per_step - 1)
        # More precisely, per step we get accepted_per_step tokens for the cost
        # of ~1 big-model forward + draft_length small-model forwards (cheap).
        spec_speedup = max(1.0, accepted_per_step)
        spec_decode_tps = decode_tps * spec_speedup
    else:
        spec_decode_tps = decode_tps

    spec_decode_ms = (output_tokens / spec_decode_tps * 1000) if spec_decode_tps > 0 else 0.0
    spec_total_ms = prefill_ms + spec_decode_ms + _OVERHEAD_MS

    # --- VRAM and context ---
    vram_required_gb = estimate_vram_gb(params_b, quant, context_length)
    vram_budget = hardware.gpu_vram_gb if hardware.gpu_vram_gb > 0 else hardware.ram_total_gb
    # Max context: solve for context where VRAM is exhausted.
    bits = QUANT_BITS.get(quant, 4.85)
    model_gb = params_b * bits / 8
    overhead_gb = 0.5
    remaining = vram_budget - model_gb - overhead_gb
    if remaining > 0 and params_b > 0:
        max_context = int((remaining / (params_b * 0.002)) * 1024)
        max_context = max(512, min(max_context, 131_072))
    else:
        max_context = 0

    # --- Context scaling penalty ---
    ctx_penalty = _context_penalty_ms(params_b, max_context, output_tokens, bw_bytes_s)

    hw_label = hardware.gpu_model if hardware.gpu_model != "none" else hardware.cpu_model

    return LLMLatencyEstimate(
        model_name=model_name,
        params_b=params_b,
        active_params_b=active_params_b,
        quantization=quant,
        hardware=hw_label,
        decode_tps=round(decode_tps, 1),
        prefill_tps=round(prefill_tps, 1),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        prefill_ms=round(prefill_ms, 1),
        decode_ms=round(decode_ms, 1),
        overhead_ms=_OVERHEAD_MS,
        total_ms=round(total_ms, 1),
        spec_decode_tps=round(spec_decode_tps, 1),
        spec_total_ms=round(spec_total_ms, 1),
        context_penalty_ms=round(ctx_penalty, 1),
        max_context=max_context,
        model_size_gb=round(model_size_gb, 2),
        vram_required_gb=round(vram_required_gb, 1),
    )


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_models_for_task(
    task_type: str,
    hardware: HardwareProfile,
    current_model: Optional[str] = None,
) -> LatencyComparison:
    """Compare all fitting models for a specific task type.

    *task_type* determines input/output token budget (see TASK_TOKEN_BUDGETS).
    If *current_model* is provided, the result includes speedup vs that model.
    """
    if task_type not in TASK_TOKEN_BUDGETS:
        raise ValueError(
            f"Unknown task type {task_type!r}. "
            f"Choose from: {', '.join(TASK_TOKEN_BUDGETS)}"
        )

    input_tokens, output_tokens = TASK_TOKEN_BUDGETS[task_type]
    context = input_tokens + output_tokens
    vram_budget = hardware.gpu_vram_gb if hardware.gpu_vram_gb > 0 else hardware.ram_total_gb

    estimates: list[LLMLatencyEstimate] = []
    current_estimate: Optional[LLMLatencyEstimate] = None

    for model in COMMON_MODELS:
        active = model.get("active_params", model["params"])
        # Pick the best quantization that fits.
        for quant in ("fp16", "Q8_0", "Q6_K", "Q4_K_M"):
            vram_needed = estimate_vram_gb(model["params"], quant, context)
            if vram_needed <= vram_budget:
                est = estimate_llm_latency(
                    model_name=model["name"],
                    params_b=model["params"],
                    quant=quant,
                    hardware=hardware,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    active_params_b=active,
                )
                estimates.append(est)
                if current_model and model["name"] == current_model:
                    current_estimate = est
                break  # best quant that fits

    # Sort by total latency (fastest first).
    estimates.sort(key=lambda e: e.total_ms)

    recommended = estimates[0].model_name if estimates else ""
    speedup = 1.0
    if current_estimate and estimates:
        speedup = current_estimate.total_ms / estimates[0].total_ms if estimates[0].total_ms > 0 else 1.0

    return LatencyComparison(
        task_type=task_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimates=estimates,
        recommended=recommended,
        speedup_vs_current=round(speedup, 2),
    )


# ---------------------------------------------------------------------------
# Routing table
# ---------------------------------------------------------------------------

def build_routing_table(
    comparisons: list[LatencyComparison],
    hardware: HardwareProfile,
) -> dict[str, dict]:
    """Build optimal model-to-task routing table.

    Returns a dict keyed by task_type, each value containing the recommended
    model, expected throughput, latency, and VRAM.

    Constraint: the set of distinct models across all tasks must fit in VRAM
    simultaneously (or the table notes which models must be swapped).
    """
    vram_budget = hardware.gpu_vram_gb if hardware.gpu_vram_gb > 0 else hardware.ram_total_gb
    table: dict[str, dict] = {}

    # First pass: pick the fastest model per task.
    model_vram: dict[str, float] = {}  # model_name -> vram_gb
    for comp in comparisons:
        if not comp.estimates:
            table[comp.task_type] = {
                "model": "none",
                "quantization": "none",
                "expected_tps": 0,
                "expected_latency_ms": 0,
                "vram_gb": 0,
            }
            continue

        best = comp.estimates[0]
        table[comp.task_type] = {
            "model": best.model_name,
            "quantization": best.quantization,
            "expected_tps": best.decode_tps,
            "expected_latency_ms": best.total_ms,
            "vram_gb": best.vram_required_gb,
        }
        model_vram[best.model_name] = best.vram_required_gb

    # Check total VRAM for all distinct models.
    total_vram = sum(model_vram.values())
    if total_vram > vram_budget:
        # Mark that hot-swapping is required.
        for task_type in table:
            if table[task_type]["model"] != "none":
                table[task_type]["hot_swap_required"] = True
                table[task_type]["total_model_vram_gb"] = round(total_vram, 1)
    else:
        for task_type in table:
            if table[task_type]["model"] != "none":
                table[task_type]["hot_swap_required"] = False

    return table


# ---------------------------------------------------------------------------
# Calibration with runtime data
# ---------------------------------------------------------------------------

def calibrate_with_runtime(
    estimates: list[LLMLatencyEstimate],
    runtime_events: list[dict],
) -> list[LLMLatencyEstimate]:
    """Adjust formula estimates using actual runtime observations.

    Each runtime event should be a dict with at minimum::

        {
            "model_name": str,
            "hardware": str,         # GPU/CPU model
            "input_tokens": int,
            "output_tokens": int,
            "latency_ms": float,     # Actual end-to-end latency
            "decode_tps": float,     # Actual measured decode TPS (optional)
        }

    Groups events by (model_name, hardware), computes correction factors
    from the median ratio of actual/predicted, and applies them.

    Returns new LLMLatencyEstimate objects with confidence="calibrated".
    """
    if not runtime_events:
        return estimates

    # Group runtime observations.
    from collections import defaultdict
    obs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for ev in runtime_events:
        key = (ev.get("model_name", ""), ev.get("hardware", ""))
        obs[key].append(ev)

    # Compute correction factors per (model, hardware).
    corrections: dict[tuple[str, str], dict[str, float]] = {}
    for key, events in obs.items():
        latency_ratios = []
        tps_ratios = []
        for ev in events:
            # Find matching estimate.
            for est in estimates:
                if est.model_name == ev.get("model_name") and est.hardware == ev.get("hardware"):
                    if est.total_ms > 0 and ev.get("latency_ms", 0) > 0:
                        latency_ratios.append(ev["latency_ms"] / est.total_ms)
                    if est.decode_tps > 0 and ev.get("decode_tps", 0) > 0:
                        tps_ratios.append(ev["decode_tps"] / est.decode_tps)
                    break

        latency_corr = _median(latency_ratios) if latency_ratios else 1.0
        tps_corr = _median(tps_ratios) if tps_ratios else 1.0
        corrections[key] = {"latency": latency_corr, "tps": tps_corr}

    # Apply corrections.
    calibrated: list[LLMLatencyEstimate] = []
    for est in estimates:
        key = (est.model_name, est.hardware)
        corr = corrections.get(key)
        if corr is None:
            calibrated.append(est)
            continue

        lc = corr["latency"]
        tc = corr["tps"]

        calibrated.append(LLMLatencyEstimate(
            model_name=est.model_name,
            params_b=est.params_b,
            active_params_b=est.active_params_b,
            quantization=est.quantization,
            hardware=est.hardware,
            decode_tps=round(est.decode_tps * tc, 1),
            prefill_tps=round(est.prefill_tps * tc, 1),
            input_tokens=est.input_tokens,
            output_tokens=est.output_tokens,
            prefill_ms=round(est.prefill_ms * lc, 1),
            decode_ms=round(est.decode_ms * lc, 1),
            overhead_ms=est.overhead_ms,
            total_ms=round(est.total_ms * lc, 1),
            spec_decode_tps=round(est.spec_decode_tps * tc, 1),
            spec_total_ms=round(est.spec_total_ms * lc, 1),
            context_penalty_ms=round(est.context_penalty_ms * lc, 1),
            max_context=est.max_context,
            model_size_gb=est.model_size_gb,
            vram_required_gb=est.vram_required_gb,
            confidence="calibrated",
        ))

    return calibrated


def _median(values: list[float]) -> float:
    """Compute median of a list of floats."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_llm_latency_report(
    comparisons: list[LatencyComparison],
    current_config: Optional[dict] = None,
) -> str:
    """Format latency comparisons as markdown.

    Shows per-task-type best model, expected latency, and speedup.
    If *current_config* is provided (``{"model": "...", "tps": ...}``),
    includes comparison against current setup.
    """
    lines: list[str] = []
    lines.append("# LLM Latency Analysis")
    lines.append("")

    if current_config:
        lines.append(f"**Current model**: {current_config.get('model', 'unknown')}")
        lines.append("")

    for comp in comparisons:
        lines.append(f"## {comp.task_type.title()} ({comp.input_tokens} in / {comp.output_tokens} out)")
        lines.append("")

        if not comp.estimates:
            lines.append("No models fit for this task on available hardware.")
            lines.append("")
            continue

        lines.append("| Model | Quant | Decode TPS | Prefill TPS | Total ms | VRAM GB | Confidence |")
        lines.append("|-------|-------|-----------|-------------|----------|---------|------------|")
        for est in comp.estimates[:8]:  # Top 8
            marker = " **" if est.model_name == comp.recommended else ""
            lines.append(
                f"| {est.model_name}{marker} | {est.quantization} "
                f"| {est.decode_tps:.0f} | {est.prefill_tps:.0f} "
                f"| {est.total_ms:.0f} | {est.vram_required_gb:.1f} "
                f"| {est.confidence} |"
            )

        best = comp.estimates[0]
        rec = f"**Recommended**: {best.model_name} ({best.quantization}) -- {best.total_ms:.0f}ms"
        if comp.speedup_vs_current > 1.05:
            rec += f" ({comp.speedup_vs_current:.1f}x faster than current)"
        lines.append("")
        lines.append(rec)
        lines.append("")

    return "\n".join(lines)


def format_routing_table(
    table: dict[str, dict],
    hardware: HardwareProfile,
) -> str:
    """Format a routing table as markdown."""
    lines: list[str] = []
    hw_label = hardware.gpu_model if hardware.gpu_model != "none" else hardware.cpu_model
    lines.append(f"# Optimal Model Routing Table ({hw_label})")
    lines.append("")
    lines.append("| Task Type | Model | Quant | Decode TPS | Latency ms | VRAM GB | Swap? |")
    lines.append("|-----------|-------|-------|-----------|------------|---------|-------|")

    for task_type, entry in table.items():
        swap = "yes" if entry.get("hot_swap_required") else "no"
        lines.append(
            f"| {task_type} | {entry['model']} | {entry.get('quantization', '')} "
            f"| {entry['expected_tps']:.0f} | {entry['expected_latency_ms']:.0f} "
            f"| {entry['vram_gb']:.1f} | {swap} |"
        )

    # Check for hot-swap note.
    any_swap = any(e.get("hot_swap_required") for e in table.values())
    if any_swap:
        total = max(
            (e.get("total_model_vram_gb", 0) for e in table.values()), default=0
        )
        budget = hardware.gpu_vram_gb if hardware.gpu_vram_gb > 0 else hardware.ram_total_gb
        lines.append("")
        lines.append(
            f"*Hot-swap required: distinct models need {total:.0f} GB "
            f"but only {budget:.0f} GB available. Use llama-swap or similar.*"
        )

    lines.append("")
    return "\n".join(lines)
