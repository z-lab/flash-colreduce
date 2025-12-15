#!/usr/bin/env python3
"""Flash-ColReduce Benchmark Suite"""

import time
from dataclasses import dataclass
from typing import Callable

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from flash_colreduce import flash_colreduce, naive_colreduce

console = Console()


@dataclass
class BenchConfig:
    b: int
    h: int
    m: int
    n: int
    d: int
    is_causal: bool

    @property
    def label(self) -> str:
        mode = "causal" if self.is_causal else "dense"
        return f"B={self.b}, H={self.h}, M={self.m}, N={self.n}, D={self.d} [{mode}]"

    @property
    def short_label(self) -> str:
        return f"{self.b}×{self.h}×{self.m}×{self.n}"


@dataclass
class BenchResult:
    config: BenchConfig
    flash_time_ms: float
    naive_time_ms: float
    flash_mem_mb: float
    naive_mem_mb: float

    @property
    def speedup(self) -> float:
        return self.naive_time_ms / self.flash_time_ms if self.flash_time_ms > 0 else float("inf")

    @property
    def mem_reduction(self) -> float:
        return self.naive_mem_mb / self.flash_mem_mb if self.flash_mem_mb > 0 else float("inf")


def cuda_sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def measure_latency(fn: Callable, warmup: int = 10, min_iters: int = 10) -> float:
    """Measure average latency with adaptive iteration count."""
    torch.cuda.empty_cache()

    # Warmup
    for _ in range(warmup):
        fn()

    # Calibration run
    start = cuda_sync_time()
    for _ in range(min_iters):
        fn()
    calibration = cuda_sync_time() - start

    # Target ~1 second of measurement
    iters = max(int(min_iters / calibration), min_iters)

    # Timed run
    for _ in range(iters):
        fn()
    start = cuda_sync_time()
    for _ in range(iters):
        fn()
    elapsed = cuda_sync_time() - start

    return elapsed / iters


@torch.inference_mode()
def measure_peak_memory(fn: Callable) -> int:
    """Measure peak GPU memory in bytes."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


# Chunking state for OOM recovery
_chunk_cache: dict = {}


def chunked_naive(q, k, is_causal=False, batch_chunk=None, seq_chunk=None):
    """Naive implementation with chunking to avoid OOM."""
    b, h, m, d = q.shape
    n = k.shape[2]
    key = (b, h, m, d, n)

    if key in _chunk_cache:
        batch_chunk, seq_chunk = _chunk_cache[key]
    batch_chunk = batch_chunk or b
    seq_chunk = seq_chunk or m

    try:
        out = torch.zeros((b, h, n), device=q.device, dtype=q.dtype)
        for bs in range(0, b, batch_chunk):
            be = min(bs + batch_chunk, b)
            for qs in range(0, m, seq_chunk):
                qe = min(qs + seq_chunk, m)
                out[bs:be] += naive_colreduce(
                    q[bs:be, :, qs:qe], k[bs:be], is_causal=is_causal
                )
        _chunk_cache[key] = (batch_chunk, seq_chunk)
        return out
    except torch.OutOfMemoryError:
        if batch_chunk > 1:
            return chunked_naive(q, k, is_causal, batch_chunk // 2, seq_chunk)
        elif seq_chunk > 1:
            return chunked_naive(q, k, is_causal, batch_chunk, seq_chunk // 2)
        raise


def run_benchmark(config: BenchConfig, device: torch.device, dtype: torch.dtype) -> BenchResult:
    """Run benchmark for a single configuration."""
    q = torch.randn(config.b, config.h, config.m, config.d, device=device, dtype=dtype)
    k = torch.randn(config.b, config.h, config.n, config.d, device=device, dtype=dtype)

    flash_fn = lambda: flash_colreduce(q, k, is_causal=config.is_causal)
    naive_fn = lambda: chunked_naive(q, k, is_causal=config.is_causal)

    flash_time = measure_latency(flash_fn) * 1000  # ms
    flash_mem = measure_peak_memory(flash_fn) / (1024 ** 2)  # MB

    naive_time = measure_latency(naive_fn) * 1000
    naive_mem = measure_peak_memory(naive_fn) / (1024 ** 2)

    return BenchResult(config, flash_time, naive_time, flash_mem, naive_mem)


def get_configs() -> list[BenchConfig]:
    """Generate benchmark configurations."""
    configs = []

    configs += [
        BenchConfig(b=b, h=32, m=1024, n=1024, d=128, is_causal=False)
        for b in [1, 4, 16, 64, 256]
    ]

    configs += [
        BenchConfig(b=1, h=32, m=n, n=n, d=128, is_causal=False)
        for n in [1024, 2048, 4096, 8192, 16384]
    ]

    configs += [
        BenchConfig(b=b, h=32, m=64, n=16384, d=128, is_causal=True)
        for b in [1, 4, 16, 64, 256]
    ]

    configs += [
        BenchConfig(b=1, h=32, m=512, n=n, d=128, is_causal=True)
        for n in [16384, 32768, 65536, 131072, 262144]
    ]

    return configs


def format_speedup(x: float) -> Text:
    """Format speedup with color coding."""
    if x >= 10:
        return Text(f"{x:.1f}×", style="bold green")
    elif x >= 2:
        return Text(f"{x:.1f}×", style="green")
    elif x >= 1:
        return Text(f"{x:.2f}×", style="yellow")
    else:
        return Text(f"{x:.2f}×", style="red")


def format_mem_reduction(x: float) -> Text:
    """Format memory reduction with color coding."""
    if x >= 100:
        return Text(f"{x:.0f}×", style="bold cyan")
    elif x >= 10:
        return Text(f"{x:.1f}×", style="cyan")
    elif x >= 2:
        return Text(f"{x:.1f}×", style="blue")
    else:
        return Text(f"{x:.2f}×", style="dim")


def print_results(results: list[BenchResult]) -> None:
    """Print results in tables."""
    console.print()

    # Split by causal/non-causal
    non_causal = [r for r in results if not r.config.is_causal]
    causal = [r for r in results if r.config.is_causal]

    for group_name, group in [("Non-Causal Attention", non_causal), ("Causal Attention", causal)]:
        if not group:
            continue

        table = Table(
            title=f"[bold]{group_name}[/bold]",
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
            title_justify="left",
        )

        table.add_column("Config", style="cyan", no_wrap=True)
        table.add_column("Flash (ms)", justify="right")
        table.add_column("Naive (ms)", justify="right")
        table.add_column("Speedup", justify="right")
        table.add_column("Flash (MB)", justify="right")
        table.add_column("Naive (MB)", justify="right")
        table.add_column("Mem Savings", justify="right")

        for r in group:
            table.add_row(
                r.config.short_label,
                f"{r.flash_time_ms:.3f}",
                f"{r.naive_time_ms:.3f}",
                format_speedup(r.speedup),
                f"{r.flash_mem_mb:.1f}",
                f"{r.naive_mem_mb:.1f}",
                format_mem_reduction(r.mem_reduction),
            )

        console.print(table)
        console.print()

    # Summary statistics
    if results:
        avg_speedup = sum(r.speedup for r in results) / len(results)
        max_speedup = max(r.speedup for r in results)
        avg_mem = sum(r.mem_reduction for r in results) / len(results)
        max_mem = max(r.mem_reduction for r in results)

        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column(justify="right")

        summary.add_row("Average Speedup:", format_speedup(avg_speedup))
        summary.add_row("Maximum Speedup:", format_speedup(max_speedup))
        summary.add_row("Average Memory Reduction:", format_mem_reduction(avg_mem))
        summary.add_row("Maximum Memory Reduction:", format_mem_reduction(max_mem))

        console.print(Panel(summary, title="[bold green]Summary[/bold green]", border_style="green"))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    if not torch.cuda.is_available():
        console.print("[red]CUDA not available. Exiting.[/red]")
        return

    gpu_name = torch.cuda.get_device_name(0)
    console.print(Panel(f"[bold]Flash-ColReduce Benchmark[/bold]\nDevice: {gpu_name}", border_style="blue"))

    configs = get_configs()
    results: list[BenchResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Running benchmarks...", total=len(configs))

        for config in configs:
            progress.update(task, description=f"[cyan]{config.label}")
            result = run_benchmark(config, device, dtype)
            results.append(result)
            progress.advance(task)

    print_results(results)


if __name__ == "__main__":
    main()
