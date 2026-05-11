"""Hardware profiler for LLM model fitting.

Detects available hardware (CPU, RAM, GPU) and determines which LLM models
can fit in memory. Primary target: macOS Apple Silicon with Metal/unified memory.
Also supports Linux with NVIDIA GPUs.
"""
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HardwareProfile:
    platform: str                   # "darwin", "linux"
    cpu_model: str                  # "Apple M4 Max" or "AMD Ryzen 9 7950X"
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float

    # GPU info
    gpu_type: str                   # "apple_silicon", "nvidia", "amd", "none"
    gpu_model: str                  # "Apple M4 Max GPU" or "NVIDIA RTX 4090"
    gpu_vram_gb: float              # Dedicated VRAM (or unified memory for Apple)
    gpu_vram_available_gb: float
    gpu_compute: str                # "metal", "cuda", "rocm", "none"

    # Inference backends available
    available_backends: list = field(default_factory=list)


@dataclass
class ModelFit:
    model_name: str
    params_b: float                 # Billion parameters
    quantization: str               # "Q4_K_M", "Q8_0", "fp16", etc.
    estimated_vram_gb: float        # How much VRAM it needs
    fits: bool                      # Can it fit on this hardware?
    estimated_tps: float            # Rough tokens/sec estimate
    recommended_backend: str        # "llama.cpp", "mlx", "vllm"
    context_limit: int              # Max context at this VRAM budget
    notes: str = ""                 # "Tight fit, may swap" etc.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANT_BITS: dict[str, float] = {
    "fp16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.5,
    "Q5_K_M": 5.5,
    "Q4_K_M": 4.85,
    "Q4_0": 4.5,
    "Q3_K_M": 3.9,
    "Q2_K": 3.35,
    "IQ4_XS": 4.3,
    "IQ2_M": 2.7,
}

COMMON_MODELS: list[dict] = [
    {"name": "Qwen3-0.6B", "params": 0.6},
    {"name": "Qwen3-1.7B", "params": 1.7},
    {"name": "Qwen3-4B", "params": 4.0},
    {"name": "Qwen3-8B", "params": 8.0},
    {"name": "Qwen3-14B", "params": 14.0},
    {"name": "Qwen3-30B-A3B", "params": 30.0, "active_params": 3.0},
    {"name": "Qwen3-32B", "params": 32.0},
    {"name": "Qwen3.6-35B-A3B", "params": 35.0, "active_params": 3.0},
    {"name": "Qwen3.6-27B", "params": 27.0},
    {"name": "Llama-3.3-70B", "params": 70.0},
    {"name": "Mistral-7B", "params": 7.0},
    {"name": "Phi-4-14B", "params": 14.0},
    {"name": "DeepSeek-R1-7B", "params": 7.0},
    {"name": "DeepSeek-R1-32B", "params": 32.0},
    {"name": "Gemma-3-9B", "params": 9.0},
    {"name": "Gemma-3-27B", "params": 27.0},
]

QUANTIZATIONS: list[str] = ["Q4_K_M", "Q6_K", "Q8_0", "fp16"]

# Rough single-core token/sec baselines per billion active params, by chip family.
# These are *very* approximate; real throughput depends on batch size, context
# length, memory bandwidth, quantisation kernel quality, etc.
_TPS_PER_ACTIVE_B: dict[str, float] = {
    # Apple Silicon — memory-bandwidth-bound; numbers are for Q4_K_M llama.cpp
    "m1":       4.0,
    "m1_pro":   5.5,
    "m1_max":   7.0,
    "m1_ultra": 10.0,
    "m2":       5.0,
    "m2_pro":   6.5,
    "m2_max":   8.5,
    "m2_ultra": 12.0,
    "m3":       5.5,
    "m3_pro":   7.0,
    "m3_max":   9.5,
    "m3_ultra": 13.0,
    "m4":       6.0,
    "m4_pro":   8.0,
    "m4_max":   11.0,
    "m4_ultra": 15.0,
    # NVIDIA — CUDA, tensor cores, much higher bandwidth
    "rtx_4090": 25.0,
    "rtx_4080": 18.0,
    "rtx_3090": 15.0,
    "rtx_3080": 12.0,
    "a100":     30.0,
    "h100":     45.0,
}


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------

def estimate_vram_gb(params_b: float, quant: str, context_length: int = 8192) -> float:
    """Estimate VRAM needed for a model.

    Formula:
        model_gb  = params_b * bits_per_param / 8
        kv_cache  = (context_length / 1024) * params_b * 0.002
        overhead  = 0.5 GB (CUDA/Metal framework)
        total     = model_gb + kv_cache + overhead

    The KV-cache estimate (~2 MB per 1K context per 1B params at Q4) is a
    simplification that works reasonably well across transformer architectures
    with GQA.  It slightly overestimates for MoE models (where only active
    params contribute to KV) but we use total params for the *weight* size
    and only use active params for throughput, so this stays conservative.
    """
    bits = QUANT_BITS.get(quant, 4.85)
    model_gb = params_b * bits / 8
    kv_cache_gb = (context_length / 1024) * params_b * 0.002
    overhead_gb = 0.5
    return model_gb + kv_cache_gb + overhead_gb


def _max_context_for_budget(params_b: float, quant: str, vram_budget_gb: float) -> int:
    """Return the largest context length (in tokens) that fits within *vram_budget_gb*.

    Inverts the estimate_vram_gb formula and clamps to [512, 131072].
    """
    bits = QUANT_BITS.get(quant, 4.85)
    model_gb = params_b * bits / 8
    overhead_gb = 0.5
    remaining = vram_budget_gb - model_gb - overhead_gb
    if remaining <= 0:
        return 0
    # remaining = (ctx / 1024) * params_b * 0.002
    ctx = (remaining / (params_b * 0.002)) * 1024
    ctx = int(ctx)
    return max(512, min(ctx, 131_072))


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def detect_backends() -> list[str]:
    """Detect which inference backends are installed."""
    backends: list[str] = []

    # llama.cpp
    for name in ("llama-server", "llama-cli", "llama.cpp"):
        if shutil.which(name):
            backends.append("llama.cpp")
            break

    # MLX (Python package or standalone server)
    try:
        import importlib
        importlib.import_module("mlx")
        backends.append("mlx")
    except Exception:
        if shutil.which("mlx_lm.server"):
            backends.append("mlx")

    # Ollama
    if shutil.which("ollama"):
        backends.append("ollama")

    # vLLM
    try:
        import importlib
        importlib.import_module("vllm")
        backends.append("vllm")
    except Exception:
        pass

    return backends


# ---------------------------------------------------------------------------
# Hardware detection — helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 10) -> str:
    """Run a subprocess and return stripped stdout, or '' on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def _detect_apple_silicon() -> HardwareProfile:
    """Detect hardware on macOS (Apple Silicon or Intel)."""
    cpu_model = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if not cpu_model:
        cpu_model = "Unknown Mac CPU"

    cpu_cores = os.cpu_count() or 1

    # Total RAM
    mem_bytes_str = _run(["sysctl", "-n", "hw.memsize"])
    ram_total_gb = int(mem_bytes_str) / (1024 ** 3) if mem_bytes_str.isdigit() else 0.0

    # Available RAM — vm_stat gives pages
    available_gb = 0.0
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        available_gb = (page_size * avail_pages) / (1024 ** 3)
    except Exception:
        # Fallback: parse vm_stat
        vm = _run(["vm_stat"])
        if vm:
            page_size = 16384  # default on Apple Silicon
            free = 0
            for line in vm.splitlines():
                if "Pages free" in line:
                    free += int(line.split(":")[1].strip().rstrip("."))
                elif "Pages inactive" in line:
                    free += int(line.split(":")[1].strip().rstrip("."))
            available_gb = (free * page_size) / (1024 ** 3)

    # GPU info from system_profiler
    gpu_model = ""
    gpu_cores = 0
    sp = _run(["system_profiler", "SPDisplaysDataType", "-json"])
    if sp:
        try:
            data = json.loads(sp)
            displays = data.get("SPDisplaysDataType", [])
            if displays:
                gpu_info = displays[0]
                gpu_model = gpu_info.get("sppci_model", "")
                cores_str = gpu_info.get("sppci_cores", "0")
                gpu_cores = int(cores_str) if str(cores_str).isdigit() else 0
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    is_apple_silicon = any(
        chip in cpu_model.lower() for chip in ("apple m", "m1", "m2", "m3", "m4")
    )

    if is_apple_silicon:
        gpu_type = "apple_silicon"
        gpu_compute = "metal"
        if not gpu_model:
            gpu_model = cpu_model + " GPU"
        # Unified memory: GPU can use ~75% of total RAM in practice
        # (the OS and other apps take the rest).  We report total and
        # available separately so callers can decide their own margin.
        gpu_vram_gb = ram_total_gb
        gpu_vram_available_gb = available_gb
    else:
        gpu_type = "none"
        gpu_compute = "none"
        gpu_model = gpu_model or "Unknown"
        gpu_vram_gb = 0.0
        gpu_vram_available_gb = 0.0

    return HardwareProfile(
        platform="darwin",
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        ram_total_gb=round(ram_total_gb, 1),
        ram_available_gb=round(available_gb, 1),
        gpu_type=gpu_type,
        gpu_model=gpu_model,
        gpu_vram_gb=round(gpu_vram_gb, 1),
        gpu_vram_available_gb=round(gpu_vram_available_gb, 1),
        gpu_compute=gpu_compute,
        available_backends=detect_backends(),
    )


def _detect_linux() -> HardwareProfile:
    """Detect hardware on Linux."""
    # CPU
    cpu_model = "Unknown Linux CPU"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    cpu_cores = os.cpu_count() or 1

    # RAM
    ram_total_gb = 0.0
    ram_available_gb = 0.0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram_total_gb = int(line.split()[1]) / (1024 ** 2)  # kB -> GB
                elif line.startswith("MemAvailable:"):
                    ram_available_gb = int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass

    # GPU — try nvidia-smi first
    gpu_type = "none"
    gpu_model = "none"
    gpu_vram_gb = 0.0
    gpu_vram_available_gb = 0.0
    gpu_compute = "none"

    nv = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ])
    if nv:
        try:
            parts = nv.splitlines()[0].split(", ")
            gpu_model = parts[0].strip()
            gpu_vram_gb = float(parts[1].strip()) / 1024  # MiB -> GiB
            gpu_vram_available_gb = float(parts[2].strip()) / 1024
            gpu_type = "nvidia"
            gpu_compute = "cuda"
        except (IndexError, ValueError):
            pass

    # Fallback: check for AMD ROCm
    if gpu_type == "none" and shutil.which("rocm-smi"):
        gpu_type = "amd"
        gpu_compute = "rocm"
        gpu_model = "AMD GPU (ROCm)"
        # rocm-smi parsing is complex; leave VRAM at 0 for now

    return HardwareProfile(
        platform="linux",
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        ram_total_gb=round(ram_total_gb, 1),
        ram_available_gb=round(ram_available_gb, 1),
        gpu_type=gpu_type,
        gpu_model=gpu_model,
        gpu_vram_gb=round(gpu_vram_gb, 1),
        gpu_vram_available_gb=round(gpu_vram_available_gb, 1),
        gpu_compute=gpu_compute,
        available_backends=detect_backends(),
    )


# ---------------------------------------------------------------------------
# TPS estimation
# ---------------------------------------------------------------------------

def _chip_key(hw: HardwareProfile) -> str:
    """Derive a lookup key into _TPS_PER_ACTIVE_B from the hardware profile."""
    model_lower = hw.gpu_model.lower() + " " + hw.cpu_model.lower()

    # Apple Silicon
    for gen in ("m4", "m3", "m2", "m1"):
        for tier in ("ultra", "max", "pro", ""):
            tag = f"{gen}_{tier}" if tier else gen
            search = f"{gen} {tier}" if tier else gen
            if search in model_lower:
                return tag

    # NVIDIA
    for card in ("h100", "a100", "rtx_4090", "rtx_4080", "rtx_3090", "rtx_3080"):
        if card.replace("_", " ") in model_lower or card.replace("_", "") in model_lower:
            return card

    return ""


def _estimate_tps(hw: HardwareProfile, active_params_b: float, quant: str) -> float:
    """Rough tokens/sec estimate for the given hardware and model size.

    Returns 0.0 if no baseline is available for this hardware.
    """
    key = _chip_key(hw)
    base = _TPS_PER_ACTIVE_B.get(key, 0.0)
    if base == 0.0 or active_params_b <= 0:
        return 0.0

    # Base rate is for Q4_K_M. Adjust for quantisation: higher bits = slower
    # (more memory bandwidth per token).
    bits = QUANT_BITS.get(quant, 4.85)
    quant_factor = 4.85 / bits  # >1 for smaller quants, <1 for larger

    tps = (base / active_params_b) * quant_factor
    return round(tps, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_hardware() -> HardwareProfile:
    """Auto-detect available hardware."""
    system = platform.system().lower()
    if system == "darwin":
        return _detect_apple_silicon()
    elif system == "linux":
        return _detect_linux()
    else:
        # Minimal fallback for unsupported platforms
        return HardwareProfile(
            platform=system,
            cpu_model=platform.processor() or "Unknown",
            cpu_cores=os.cpu_count() or 1,
            ram_total_gb=0.0,
            ram_available_gb=0.0,
            gpu_type="none",
            gpu_model="none",
            gpu_vram_gb=0.0,
            gpu_vram_available_gb=0.0,
            gpu_compute="none",
            available_backends=detect_backends(),
        )


def check_model_fits(
    model_name: str,
    params_b: float,
    quant: str,
    hardware: HardwareProfile,
    context: int = 8192,
    active_params_b: Optional[float] = None,
) -> ModelFit:
    """Check if a specific model fits on the detected hardware."""
    vram_needed = estimate_vram_gb(params_b, quant, context)
    vram_budget = hardware.gpu_vram_gb if hardware.gpu_vram_gb > 0 else hardware.ram_total_gb

    fits = vram_needed <= vram_budget
    active = active_params_b if active_params_b else params_b

    # Context limit: how much context can we actually afford?
    ctx_limit = _max_context_for_budget(params_b, quant, vram_budget)
    if ctx_limit < context:
        fits = False

    # Notes
    notes = ""
    if fits:
        headroom = vram_budget - vram_needed
        if headroom < 2.0:
            notes = "Tight fit, may swap under pressure"
        elif headroom < 4.0:
            notes = "Fits with moderate headroom"
        else:
            notes = "Comfortable fit"
    else:
        notes = f"Needs {vram_needed:.1f} GB, only {vram_budget:.1f} GB available"

    # Backend recommendation
    backend = _recommend_backend(hardware, params_b, quant)

    # TPS estimate
    tps = _estimate_tps(hardware, active, quant)

    return ModelFit(
        model_name=model_name,
        params_b=params_b,
        quantization=quant,
        estimated_vram_gb=round(vram_needed, 1),
        fits=fits,
        estimated_tps=tps,
        recommended_backend=backend,
        context_limit=min(ctx_limit, 131_072),
        notes=notes,
    )


def _recommend_backend(hw: HardwareProfile, params_b: float, quant: str) -> str:
    """Pick the best available backend for this hardware + model combo."""
    backends = hw.available_backends

    if hw.gpu_type == "apple_silicon":
        # MLX is native and often fastest for Apple Silicon, but only
        # supports its own format.  llama.cpp is the universal fallback
        # with excellent Metal support.
        if "mlx" in backends:
            return "mlx"
        if "llama.cpp" in backends:
            return "llama.cpp"
        if "ollama" in backends:
            return "ollama"
    elif hw.gpu_type == "nvidia":
        # vLLM gives best throughput on NVIDIA if available
        if "vllm" in backends and quant == "fp16":
            return "vllm"
        if "llama.cpp" in backends:
            return "llama.cpp"
        if "ollama" in backends:
            return "ollama"
    else:
        # CPU-only or unknown GPU
        if "llama.cpp" in backends:
            return "llama.cpp"
        if "ollama" in backends:
            return "ollama"

    return backends[0] if backends else "none"


def recommend_models(
    hardware: HardwareProfile,
    task_types: Optional[dict] = None,
) -> list[ModelFit]:
    """Recommend models that fit, ranked by capability (largest first).

    If *task_types* is provided (e.g. from an LLM call extractor), it maps
    complexity levels to counts::

        {"simple": 12, "moderate": 5, "complex": 3}

    The recommendations will then include task-specific notes suggesting
    smaller models for simple tasks and the largest fitting model for complex
    tasks.
    """
    results: list[ModelFit] = []

    for model in COMMON_MODELS:
        active = model.get("active_params", model["params"])
        for quant in QUANTIZATIONS:
            fit = check_model_fits(
                model_name=model["name"],
                params_b=model["params"],
                quant=quant,
                hardware=hardware,
                context=8192,
                active_params_b=active,
            )
            if fit.fits:
                results.append(fit)

    # Sort: prefer larger models, then smaller quant (better quality)
    results.sort(key=lambda m: (m.params_b, -QUANT_BITS.get(m.quantization, 5)), reverse=True)

    # Deduplicate: keep only the best quant per model name
    seen: set[str] = set()
    deduped: list[ModelFit] = []
    for r in results:
        if r.model_name not in seen:
            seen.add(r.model_name)
            deduped.append(r)

    # Annotate with task-type suggestions if provided
    if task_types and deduped:
        has_complex = task_types.get("complex", 0) > 0
        has_simple = task_types.get("simple", 0) > 0

        if has_complex:
            deduped[0].notes += " | Recommended for complex tasks (planning, code)"
        if has_simple and len(deduped) > 1:
            smallest = deduped[-1]
            smallest.notes += " | Recommended for simple tasks (classification, extraction)"

    return deduped


def format_hardware_report(
    hardware: HardwareProfile,
    recommendations: list[ModelFit],
) -> str:
    """Format hardware analysis as markdown."""
    lines: list[str] = []
    lines.append("# Hardware Profile")
    lines.append("")
    lines.append(f"- **Platform**: {hardware.platform}")
    lines.append(f"- **CPU**: {hardware.cpu_model} ({hardware.cpu_cores} cores)")
    lines.append(f"- **RAM**: {hardware.ram_total_gb:.1f} GB total, {hardware.ram_available_gb:.1f} GB available")
    lines.append(f"- **GPU**: {hardware.gpu_model} ({hardware.gpu_type})")
    lines.append(f"- **VRAM**: {hardware.gpu_vram_gb:.1f} GB total, {hardware.gpu_vram_available_gb:.1f} GB available")
    lines.append(f"- **Compute**: {hardware.gpu_compute}")
    lines.append(f"- **Backends**: {', '.join(hardware.available_backends) or 'none detected'}")
    lines.append("")

    if not recommendations:
        lines.append("## No models fit on this hardware")
        return "\n".join(lines)

    lines.append("## Model Recommendations")
    lines.append("")
    lines.append("| Model | Quant | VRAM (GB) | TPS | Context | Backend | Notes |")
    lines.append("|-------|-------|-----------|-----|---------|---------|-------|")
    for m in recommendations:
        tps_str = f"{m.estimated_tps:.0f}" if m.estimated_tps > 0 else "n/a"
        ctx_str = f"{m.context_limit // 1024}K" if m.context_limit >= 1024 else str(m.context_limit)
        lines.append(
            f"| {m.model_name} | {m.quantization} | {m.estimated_vram_gb:.1f} "
            f"| {tps_str} | {ctx_str} | {m.recommended_backend} | {m.notes} |"
        )
    lines.append("")
    return "\n".join(lines)
