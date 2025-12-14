import json
import time
from typing import Callable

import torch

from flash_colreduce import flash_colreduce, naive_colreduce


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def measure_latency(fn: Callable) -> float:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fn()
    start = cuda_time()
    for _ in range(10):
        fn()
    duration = cuda_time() - start

    iters = max(int(1 / duration), 10)

    print(iters)

    for _ in range(iters):
        fn()
    start = cuda_time()
    for _ in range(iters):
        fn()
    duration = cuda_time() - start
    return duration / iters


@torch.inference_mode()
def peak_memory(fn: Callable) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


__chunk_size = {}


def chunked_colreduce(q, k, is_causal=False, reduction="sum", batch_chunk_size=None, seq_chunk_size=None):
    b, h, m, d = q.shape
    _, _, n, _ = k.shape

    if (b, h, m, d, n) in __chunk_size:
        batch_chunk_size, seq_chunk_size = __chunk_size[(b, h, m, d, n)]

    if batch_chunk_size is None:
        batch_chunk_size = b
    if seq_chunk_size is None:
        seq_chunk_size = m

    try:
        output = torch.zeros((b, h, n), device=q.device, dtype=q.dtype)
        for b_start in range(0, b, batch_chunk_size):
            b_end = min(b_start + batch_chunk_size, b)
            for q_start in range(0, m, seq_chunk_size):
                q_end = min(q_start + seq_chunk_size, m)
                q_chunk = q[b_start:b_end, :, q_start:q_end, :]
                k_chunk = k[b_start:b_end, :, :, :]
                output[b_start:b_end, :, :] += naive_colreduce(
                    q_chunk, k_chunk, is_causal=is_causal, reduction=reduction
                )
    except torch.OutOfMemoryError as e:
        print(f"OOM: {e}, batch_chunk_size={batch_chunk_size}, seq_chunk_size={seq_chunk_size}")
        if batch_chunk_size > 1:
            batch_chunk_size //= 2
        elif seq_chunk_size > 1:
            seq_chunk_size //= 2
        else:
            raise e
        return chunked_colreduce(
            q,
            k,
            is_causal=is_causal,
            reduction=reduction,
            batch_chunk_size=batch_chunk_size,
            seq_chunk_size=seq_chunk_size,
        )

    __chunk_size[(b, h, m, d, n)] = (batch_chunk_size, seq_chunk_size)

    return output


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    configs = [
        {"b": b, "h": 32, "m": 1024, "n": 1024, "d": 128, "is_causal": False, "reduction": "sum"}
        for b in [1, 4, 16, 64, 256]
    ]
    configs += [
        {"b": 1, "h": 32, "m": n, "n": n, "d": 128, "is_causal": False, "reduction": "sum"}
        for n in [1024, 2048, 4096, 8192, 16384]
    ]
    configs += [
        {"b": b, "h": 32, "m": 64, "n": 16384, "d": 128, "is_causal": True, "reduction": "sum"}
        for b in [1, 4, 16, 64, 256]
    ]
    configs += [
        {"b": 1, "h": 32, "m": 512, "n": n, "d": 128, "is_causal": True, "reduction": "sum"}
        for n in [16384, 32768, 65536, 131072, 262144]
    ]

    # configs = [
    #     {"b": b, "h": 16, "m": 1024, "n": 1024, "d": 64, "is_causal": False, "reduction": "sum"}
    #     for b in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # ]
    # configs += [
    #     {"b": 1, "h": 16, "m": n, "n": n, "d": 64, "is_causal": False, "reduction": "sum"}
    #     for n in [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    # ]
    # configs += [
    #     {"b": b, "h": 32, "m": 128, "n": 65536, "d": 128, "is_causal": True, "reduction": "sum"}
    #     for b in [1, 2, 4, 8, 16, 32]
    # ]
    # configs += [
    #     {"b": 1, "h": 32, "m": 128, "n": n, "d": 128, "is_causal": True, "reduction": "sum"}
    #     for n in [16384, 32768, 65536, 131072, 262144]
    # ]

    results = []
    for config in configs:
        print(config)
        b, h, m, n, d, is_causal, reduction = config.values()
        q = torch.randn(b, h, m, d, device=device, dtype=dtype)
        k = torch.randn(b, h, n, d, device=device, dtype=dtype)
        for fn in [flash_colreduce, chunked_colreduce]:
            f = lambda: fn(q, k, is_causal=is_causal, reduction=reduction)
            t = measure_latency(f)
            mem = peak_memory(f)
            print(fn.__name__, t, mem)
            results.append({**config, "fn": fn.__name__, "t": t, "mem": mem})

    with open("results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

    with open("results.json", "r") as f:
        results = json.load(f)

    for result in results:
        if result["fn"] == "flash_colreduce":
            result["t"] = result["t"] * 1000
            result["mem"] = result["mem"] / 1024 / 1024 / 1024
            print(result)

    for result in results:
        if result["fn"] == "chunked_colreduce":
            result["t"] = result["t"] * 1000
            result["mem"] = result["mem"] / 1024 / 1024 / 1024
            print(result)
