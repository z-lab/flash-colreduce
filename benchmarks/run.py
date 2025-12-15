import time
from typing import Callable

import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from flash_colreduce import flash_colreduce, naive_colreduce

CONFIGS = (
    [
        {
            "b": b,
            "h": 32,
            "m": 1024,
            "n": 1024,
            "d": 128,
            "is_causal": False,
        }
        for b in [1, 4, 16, 64, 256]
    ]
    + [
        {
            "b": 1,
            "h": 32,
            "m": n,
            "n": n,
            "d": 128,
            "is_causal": False,
        }
        for n in [1024, 2048, 4096, 8192, 16384]
    ]
    + [
        {
            "b": b,
            "h": 32,
            "m": 64,
            "n": 16384,
            "d": 128,
            "is_causal": True,
        }
        for b in [1, 4, 16, 64, 256]
    ]
    + [
        {
            "b": 1,
            "h": 32,
            "m": 512,
            "n": n,
            "d": 128,
            "is_causal": True,
        }
        for n in [16384, 32768, 65536, 131072, 262144]
    ]
)


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def measure_latency(fn: Callable, min_iters: int = 10) -> float:
    torch.cuda.empty_cache()
    fn()

    # Calibration run
    start = cuda_time()
    for _ in range(min_iters):
        fn()
    num_iters = max(int(min_iters / (cuda_time() - start)), min_iters)

    # Warmup run
    for _ in range(num_iters):
        fn()

    # Measurement run
    start = cuda_time()
    for _ in range(num_iters):
        fn()
    return (cuda_time() - start) / num_iters


@torch.inference_mode()
def measure_memory(fn: Callable) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def chunked_run(
    colreduce: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    reduction: str,
    is_causal: bool = False,
    chunk_b: int | None = None,
    chunk_m: int | None = None,
) -> torch.Tensor:
    b, h, m, d = q.shape
    n = k.shape[2]

    key = (colreduce, b, h, m, n, d, is_causal)
    if not hasattr(chunked_run, "chunk_size"):
        chunked_run.chunk_size = {}
    if key in chunked_run.chunk_size:
        chunk_b, chunk_m = chunked_run.chunk_size[key]
    chunk_b = chunk_b or b
    chunk_m = chunk_m or m

    try:
        o = torch.zeros((b, h, n), device=q.device, dtype=q.dtype)
        for bl in range(0, b, chunk_b):
            br = min(bl + chunk_b, b)
            for ml in range(0, m, chunk_m):
                mr = min(ml + chunk_m, m)
                o[bl:br] += colreduce(q[bl:br, :, ml:mr], k[bl:br], reduction=reduction, is_causal=is_causal)
        chunked_run.chunk_size[key] = (chunk_b, chunk_m)
        return o
    except torch.OutOfMemoryError:
        if chunk_b > 1:
            chunk_b //= 2
        elif chunk_m > 1:
            chunk_m //= 2
        else:
            raise
        return chunked_run(colreduce, q, k, reduction=reduction, is_causal=is_causal, chunk_b=chunk_b, chunk_m=chunk_m)


def benchmark(b: int, h: int, m: int, n: int, d: int, is_causal: bool) -> dict:
    q = torch.randn(b, h, m, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)

    flash_fn = lambda: chunked_run(flash_colreduce, q, k, reduction="sum", is_causal=is_causal)
    naive_fn = lambda: chunked_run(naive_colreduce, q, k, reduction="sum", is_causal=is_causal)

    flash_latency = measure_latency(flash_fn) * 1000
    flash_memory = measure_memory(flash_fn) / (1024**2)

    naive_latency = measure_latency(naive_fn) * 1000
    naive_memory = measure_memory(naive_fn) / (1024**2)

    return {
        "flash_latency": flash_latency,
        "flash_memory": flash_memory,
        "naive_latency": naive_latency,
        "naive_memory": naive_memory,
    }


def main() -> None:
    table = Table(title=f"Benchmark Results on {torch.cuda.get_device_name()}", show_header=True, title_justify="left")

    table.add_column("Configuration", no_wrap=True)
    table.add_column("Flash (ms)", justify="right")
    table.add_column("Naive (ms)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Flash (MB)", justify="right")
    table.add_column("Naive (MB)", justify="right")
    table.add_column("Memory Saving", justify="right")

    for c in tqdm(CONFIGS, leave=False):
        r = benchmark(**c)
        table.add_row(
            ",".join(f"{k}={v}" for k, v in c.items()),
            f"{r['flash_latency']:.3f}",
            f"{r['naive_latency']:.3f}",
            f"{r['naive_latency'] / r['flash_latency']:.2f}",
            f"{r['flash_memory']:.1f}",
            f"{r['naive_memory']:.1f}",
            f"{r['naive_memory'] / r['flash_memory']:.2f}",
        )

    console = Console()
    console.print(table)


if __name__ == "__main__":
    main()
