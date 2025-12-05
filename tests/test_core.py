import math
import pytest
import torch

from flash_colsum import flash_colsum, naive_colsum


def _has_cuda() -> bool:
	return torch.cuda.is_available()


def _has_triton() -> bool:
	try:
		import triton  # noqa: F401
		return True
	except Exception:
		return False


@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
def test_correctness_batched_non_causal():
	"""Test batched non-causal attention."""
	device = torch.device("cuda")
	dtype = torch.float16
	B, H, S, D = 4, 8, 512, 64
	torch.manual_seed(0)
	Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
	K = torch.randn(B, H, S, D, device=device, dtype=dtype)
	
	out_ref = naive_colsum(Q, K)
	out_fast = flash_colsum(Q, K)
	
	assert out_ref.shape == out_fast.shape == (B, S)
	assert torch.allclose(out_ref, out_fast, atol=1e-3, rtol=1e-3), \
		f"Max diff: {(out_ref - out_fast).abs().max():.6f}"


@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
def test_correctness_non_batched():
	"""Test non-batched (B=1, large sequence)."""
	device = torch.device("cuda")
	dtype = torch.float16
	B, H, S, D = 1, 16, 8192, 64
	torch.manual_seed(0)
	Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
	K = torch.randn(B, H, S, D, device=device, dtype=dtype)
	
	out_ref = naive_colsum(Q, K)
	out_fast = flash_colsum(Q, K)
	
	assert out_ref.shape == out_fast.shape == (B, S)
	assert torch.allclose(out_ref, out_fast, atol=1e-3, rtol=1e-3), \
		f"Max diff: {(out_ref - out_fast).abs().max():.6f}"


@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
def test_correctness_batched_non_causal_cls_prefix():
	"""Test batched non-causal attention with CLS prefix averaging."""
	device = torch.device("cuda")
	dtype = torch.float16
	B, H, S, D = 4, 8, 512, 64
	cls_len = 1
	torch.manual_seed(0)
	Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
	K = torch.randn(B, H, S, D, device=device, dtype=dtype)
	
	out_ref = naive_colsum(Q, K, cls_len=cls_len)
	out_fast = flash_colsum(Q, K, cls_len=cls_len)
	
	assert out_ref.shape == out_fast.shape == (B, S)
	assert torch.allclose(out_ref, out_fast, atol=1e-3, rtol=1e-3), \
		f"Max diff: {(out_ref - out_fast).abs().max():.6f}"


@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
def test_correctness_non_batched_cls_prefix():
	"""Test non-batched (B=1) with CLS prefix averaging."""
	device = torch.device("cuda")
	dtype = torch.float16
	B, H, S, D = 1, 16, 4096, 64
	cls_len = 2
	torch.manual_seed(0)
	Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
	K = torch.randn(B, H, S, D, device=device, dtype=dtype)
	
	out_ref = naive_colsum(Q, K, cls_len=cls_len)
	out_fast = flash_colsum(Q, K, cls_len=cls_len)
	
	assert out_ref.shape == out_fast.shape == (B, S)
	assert torch.allclose(out_ref, out_fast, atol=1e-3, rtol=1e-3), \
		f"Max diff: {(out_ref - out_fast).abs().max():.6f}"


@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
def test_correctness_causal():
	"""Test causal attention."""
	device = torch.device("cuda")
	dtype = torch.float16
	B, H, Q_len, K_len, D = 1, 16, 128, 16384, 128
	torch.manual_seed(0)
	Q = torch.randn(B, H, Q_len, D, device=device, dtype=dtype)
	K = torch.randn(B, H, K_len, D, device=device, dtype=dtype)
	
	out_ref = naive_colsum(Q, K, is_causal=True)
	out_fast = flash_colsum(Q, K, is_causal=True)
	
	assert out_ref.shape == out_fast.shape == (B, K_len)
	assert torch.allclose(out_ref, out_fast, atol=1e-3, rtol=1e-3), \
		f"Max diff: {(out_ref - out_fast).abs().max():.6f}"


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required")
def test_error_handling():
	"""Test error handling for invalid inputs."""
	device = torch.device("cuda")
	dtype = torch.float16
	
	# Test causal with B > 1
	Q = torch.randn(2, 8, 128, 64, device=device, dtype=dtype)
	K = torch.randn(2, 8, 1024, 64, device=device, dtype=dtype)
	with pytest.raises(ValueError, match="batch size 1"):
		flash_colsum(Q, K, is_causal=True)
	
	# Test non-CUDA tensor
	Q_cpu = torch.randn(1, 8, 128, 64)
	K_cpu = torch.randn(1, 8, 128, 64)
	with pytest.raises(ValueError, match="CUDA"):
		flash_colsum(Q_cpu, K_cpu)
	
	# Test shape mismatch for non-causal
	Q = torch.randn(2, 8, 128, 64, device=device, dtype=dtype)
	K = torch.randn(2, 8, 256, 64, device=device, dtype=dtype)
	with pytest.raises(ValueError, match="same shape"):
		flash_colsum(Q, K)


