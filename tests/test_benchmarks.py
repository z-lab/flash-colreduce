import os
import sys
from pathlib import Path
import pytest
import torch

# Ensure repository root is on sys.path so 'benchmarks' can be imported
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
	sys.path.insert(0, str(_ROOT))

from benchmarks import benchmark_colsum as bc


def _has_cuda_triton() -> bool:
	try:
		import triton  # noqa: F401
	except Exception:
		return False
	return torch.cuda.is_available()


@pytest.mark.skipif(not _has_cuda_triton(), reason="CUDA+Triton required for benchmark")
def test_generate_benchmark_report():
	"""
	Run sweep benchmarks and generate unified plot + CSVs.
	Gated by env var to avoid accidental long runs in default test passes.
	"""
	if os.getenv("FLASH_COLSUM_RUN_BENCH", "0") != "1":
		pytest.skip("Set FLASH_COLSUM_RUN_BENCH=1 to run benchmark sweeps")

	device = torch.device("cuda")
	out_dir = os.getenv("FLASH_COLSUM_BENCH_OUT", "benchmarks/out")
	print(f"\nBenchmark output directory: {out_dir}")

	# Run all benchmarks and generate unified plot
	# Uses default iterations: vision benchmarks (warmup=25, iters=250), text causal (warmup=100, iters=1000)
	bc.sweep_all_unified(device=device, out_dir=out_dir)
	print(f"\n✓ Saved unified benchmark plot and CSVs to {out_dir}")


