import os
import sys
import time
import math
import tracemalloc
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn


N_UAV           = 40
T               = 10

# import the model from st_gcn_cat.py in same directory
sys.path.insert(0, os.path.dirname(__file__))
from st_gcn_cat import STGCNTurbulence, F_TOTAL
from src.graph_builder import build_uav_graph



# ─────────────────────────────────────────────────────────────
# Device profiles
# ─────────────────────────────────────────────────────────────

@dataclass
class DeviceProfile:
    name:             str
    display_name:     str
    n_threads:        int       # PyTorch intraop thread count
    cpu_freq_ghz:     float     # simulated effective clock (informational)
    ram_mb:           int       # total RAM in MB
    ram_budget_mb:    int       # MB available for this process
    throttle_factor:  float     # sleep multiplier to simulate slower CPU
                                # 1.0 = no throttle, 2.0 = 2x slower
    realtime_budget_s: float    # max allowed inference time [s]
    notes:            str = ""

DEVICE_PROFILES = {
    "rpi4": DeviceProfile(
        name             = "rpi4",
        display_name     = "Raspberry Pi 4B (4 GB)",
        n_threads        = 4,
        cpu_freq_ghz     = 1.5,
        ram_mb           = 4096,
        ram_budget_mb    = 512,   # OS + other processes take the rest
        throttle_factor  = 8.0,   # RPi 4 is ~8x slower than modern laptop
        realtime_budget_s= 10.0,
        notes            = "ARM Cortex-A72, no NEON BLAS acceleration for PyTorch",
    ),
    "rpi4_2gb": DeviceProfile(
        name             = "rpi4_2gb",
        display_name     = "Raspberry Pi 4B (2 GB)",
        n_threads        = 4,
        cpu_freq_ghz     = 1.5,
        ram_mb           = 2048,
        ram_budget_mb    = 256,
        throttle_factor  = 8.0,
        realtime_budget_s= 10.0,
        notes            = "Memory-constrained — may OOM with large batches",
    ),
    "rpi5": DeviceProfile(
        name             = "rpi5",
        display_name     = "Raspberry Pi 5 (4 GB)",
        n_threads        = 4,
        cpu_freq_ghz     = 2.4,
        ram_mb           = 4096,
        ram_budget_mb    = 1024,
        throttle_factor  = 3.5,   # A76 core is ~2.5x faster than A72
        realtime_budget_s= 10.0,
        notes            = "ARM Cortex-A76, significantly faster than RPi 4",
    ),
    "jetson_nano": DeviceProfile(
        name             = "jetson_nano",
        display_name     = "NVIDIA Jetson Nano",
        n_threads        = 4,
        cpu_freq_ghz     = 1.43,
        ram_mb           = 4096,
        ram_budget_mb    = 2048,
        throttle_factor  = 6.0,
        realtime_budget_s= 2.0,   # GPU available — tighter target
        notes            = "Has GPU but this benchmark is CPU-only path",
    ),
    "laptop": DeviceProfile(
        name             = "laptop",
        display_name     = "Development laptop (baseline)",
        n_threads        = 4,     # match RPi thread count for fair comparison
        cpu_freq_ghz     = 3.5,
        ram_mb           = 16384,
        ram_budget_mb    = 4096,
        throttle_factor  = 1.0,   # no throttle
        realtime_budget_s= 2.0,
        notes            = "Baseline — no throttling applied",
    ),
}


# ─────────────────────────────────────────────────────────────
# Benchmark result dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    device:               str
    model_variant:        str        # "fp32" or "int8"
    n_uavs:               int
    n_edges:              int
    n_params:             int
    model_size_mb:        float

    # timing (seconds)
    warmup_runs:          int
    timed_runs:           int
    latency_mean_s:       float
    latency_std_s:        float
    latency_min_s:        float
    latency_max_s:        float
    latency_p95_s:        float
    throughput_hz:        float      # inferences per second

    # memory
    peak_ram_mb:          float
    model_ram_mb:         float

    # budget
    realtime_budget_s:    float
    meets_budget:         bool
    headroom_pct:         float      # (budget - mean) / budget * 100

    # throttle info
    throttle_factor:      float
    simulated:            bool       # True = throttled, False = native

    warnings:             list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Model size helpers
# ─────────────────────────────────────────────────────────────

def model_size_mb(model: nn.Module) -> float:
    """Total size of model parameters in MB."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / 1024 / 1024


def activation_memory_mb(model: nn.Module, example_input, edge_index, edge_weight) -> float:
    """
    Estimate peak activation memory during a forward pass using
    tracemalloc (Python-level allocations only — underestimates
    true C++/ATen memory but gives a useful lower bound).
    """
    tracemalloc.start()
    with torch.no_grad():
        _ = model(example_input, edge_index, edge_weight)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


# ─────────────────────────────────────────────────────────────
# Quantisation helper
# ─────────────────────────────────────────────────────────────

def quantise_model(model: nn.Module) -> nn.Module:
    """
    Apply PyTorch dynamic INT8 quantisation to Linear layers.

    Dynamic quantisation:
      - Weights are quantised to INT8 at quantisation time
      - Activations are quantised dynamically per-batch at runtime
      - No calibration dataset needed (unlike static quantisation)
      - Typically 2-4x speedup on CPU for linear layers
      - ~4x model size reduction

    Note: Conv1d quantisation is not supported by torch dynamic
    quant — only Linear layers are quantised here.
    """
    q_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
    )
    return q_model

# ─────────────────────────────────────────────────────────────
# Throttled inference runner
# ─────────────────────────────────────────────────────────────

def run_inference(
    model:       nn.Module,
    x:           torch.Tensor,
    edge_index:  torch.Tensor,
    edge_weight: torch.Tensor,
    throttle:    float = 1.0,
) -> tuple[tuple, float]:
    """
    Run one forward pass and return (output, wall_time_seconds).
    throttle > 1 adds a sleep proportional to the forward time
    to simulate a slower CPU.
    """
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x, edge_index, edge_weight)
    t1 = time.perf_counter()
    elapsed = t1 - t0

    # simulate slower device: add extra sleep
    if throttle > 1.0:
        extra = elapsed * (throttle - 1.0)
        time.sleep(extra)
        elapsed = elapsed * throttle   # report as-if the whole pass was slow

    return out, elapsed


# ─────────────────────────────────────────────────────────────
# Main benchmark function
# ─────────────────────────────────────────────────────────────

def benchmark(
    profile:      DeviceProfile,
    model:        nn.Module,
    x:            torch.Tensor,
    edge_index:   torch.Tensor,
    edge_weight:  torch.Tensor,
    model_variant: str = "fp32",
    warmup_runs:  int  = 5,
    timed_runs:   int  = 20,
) -> BenchmarkResult:
    """
    Benchmark a single (device_profile, model_variant) combination.
    """
    warnings = []

    # set thread count to match device
    torch.set_num_threads(profile.n_threads)
    torch.set_num_interop_threads(1)   # no inter-op parallelism on edge

    model.eval()
    n_params   = sum(p.numel() for p in model.parameters())
    size_mb    = model_size_mb(model)

    # memory check
    if size_mb > profile.ram_budget_mb * 0.5:
        warnings.append(
            f"Model ({size_mb:.1f} MB) exceeds 50% of RAM budget "
            f"({profile.ram_budget_mb} MB) — may cause OOM on device."
        )

    # peak activation memory
    act_mb = activation_memory_mb(model, x, edge_index, edge_weight)

    # warmup (fills caches, JIT compilation if any)
    for _ in range(warmup_runs):
        run_inference(model, x, edge_index, edge_weight,
                      throttle=profile.throttle_factor)

    # timed runs
    times = []
    for _ in range(timed_runs):
        _, elapsed = run_inference(
            model, x, edge_index, edge_weight,
            throttle=profile.throttle_factor,
        )
        times.append(elapsed)

    times_t = torch.tensor(times)
    lat_mean  = times_t.mean().item()
    lat_std   = times_t.std().item()
    lat_min   = times_t.min().item()
    lat_max   = times_t.max().item()
    lat_p95   = times_t.quantile(0.95).item()
    throughput = 1.0 / lat_mean

    meets = lat_mean <= profile.realtime_budget_s
    headroom = (profile.realtime_budget_s - lat_mean) / profile.realtime_budget_s * 100

    if lat_p95 > profile.realtime_budget_s:
        warnings.append(
            f"P95 latency ({lat_p95:.3f}s) exceeds budget "
            f"({profile.realtime_budget_s}s) — will miss deadlines under load."
        )

    return BenchmarkResult(
        device            = profile.name,
        model_variant     = model_variant,
        n_uavs            = x.size(0),
        n_edges           = edge_index.size(1),
        n_params          = n_params,
        model_size_mb     = size_mb,
        warmup_runs       = warmup_runs,
        timed_runs        = timed_runs,
        latency_mean_s    = lat_mean,
        latency_std_s     = lat_std,
        latency_min_s     = lat_min,
        latency_max_s     = lat_max,
        latency_p95_s     = lat_p95,
        throughput_hz     = throughput,
        peak_ram_mb       = act_mb + size_mb,
        model_ram_mb      = size_mb,
        realtime_budget_s = profile.realtime_budget_s,
        meets_budget      = meets,
        headroom_pct      = headroom,
        throttle_factor   = profile.throttle_factor,
        simulated         = profile.throttle_factor != 1.0,
        warnings          = warnings,
    )

# ─────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────

def print_result(r: BenchmarkResult, profile: DeviceProfile):
    ok    = "PASS" if r.meets_budget else "FAIL"
    sim   = " [simulated]" if r.simulated else " [native]"
    print(f"\n  {profile.display_name}{sim}  |  {r.model_variant.upper()}")
    print(f"  {'─'*54}")
    print(f"  UAVs:           {r.n_uavs}   edges: {r.n_edges}")
    print(f"  Parameters:     {r.n_params:,}   model: {r.model_size_mb:.2f} MB")
    print(f"  Latency mean:   {r.latency_mean_s*1000:.1f} ms")
    print(f"  Latency std:    {r.latency_std_s*1000:.1f} ms")
    print(f"  Latency P95:    {r.latency_p95_s*1000:.1f} ms")
    print(f"  Throughput:     {r.throughput_hz:.2f} Hz")
    print(f"  Peak RAM est:   {r.peak_ram_mb:.1f} MB")
    budget_str = f"{r.realtime_budget_s*1000:.0f} ms"
    head_str   = f"{r.headroom_pct:.1f}%"
    print(f"  Budget:         {budget_str}   [{ok}]   headroom: {head_str}")
    for w in r.warnings:
        print(f"  WARNING: {w}")


if __name__ == "__main__":
    torch.manual_seed(0)

    # ── build graph once ──
    side = max(1, int(math.ceil(math.sqrt(N_UAV))))
    positions = torch.tensor([
        [ix * 5000.0, iy * 5000.0, 150.0]
        for ix in range(side)
        for iy in range(side)
    ])[:N_UAV]

    edge_index, edge_weight = build_uav_graph(positions)
    x = torch.randn(N_UAV, T, F_TOTAL)

    print(f"\n  Graph: {N_UAV} nodes, {edge_index.shape[1]} edges")

    # ── build models ──
    model_fp32 = STGCNTurbulence(hidden_dim=64, T=10)
    model_fp32.eval()

    model_int8 = quantise_model(model_fp32)
    model_int8.eval()
    print(f"\n  FP32 size: {model_size_mb(model_fp32):.2f} MB")
    print(f"  INT8 size: {model_size_mb(model_int8):.2f} MB  ")

    # ── run benchmarks ──
    all_results = []

    dev_name = "rpi4"
    profile = DEVICE_PROFILES[dev_name]

    # FP32
    r_fp32 = benchmark(
        profile, model_fp32, x, edge_index, edge_weight,
        model_variant="fp32",
        warmup_runs=3, timed_runs=10,
    )
    print_result(r_fp32, profile)
    all_results.append(r_fp32)

    # INT8
    r_int8 = benchmark(
        profile, model_int8, x, edge_index, edge_weight,
        model_variant="int8",
        warmup_runs=3, timed_runs=10,
    )
    print_result(r_int8, profile)
    all_results.append(r_int8)

    speedup = r_fp32.latency_mean_s / r_int8.latency_mean_s
    print(f"  INT8 speedup: {speedup:.2f}x")



