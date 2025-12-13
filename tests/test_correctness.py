import math
import pytest
import torch

from flash_colsum import flash_colsum, flash_colmean, naive_colsum, naive_colmean


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


tests = [
    (1, 16, 8192, 8192, 64, False),
    (4, 8, 512, 512, 64, False),
    (4, 8, 256, 1024, 64, False),
    (1, 16, 1024, 1024, 64, True),
    (4, 8, 128, 1024, 64, True),
    (1, 16, 128, 2048, 64, True),
    (1, 16, 1024, 1025, 64, True),
]


@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
@pytest.mark.parametrize("b, h, m, n, d, is_causal", tests)
def test_colsum_correctness(b: int, h: int, m: int, n: int, d: int, is_causal: bool):
    query = torch.randn(b, h, m, d, dtype=torch.float16, device="cuda")
    key = torch.randn(b, h, n, d, dtype=torch.float16, device="cuda")

    target = naive_colsum(query, key, is_causal=is_causal)
    output = flash_colsum(query, key, is_causal=is_causal)

    assert target.shape == output.shape == (b, n)
    assert torch.allclose(target, output, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not (_has_cuda() and _has_triton()), reason="CUDA+Triton required")
@pytest.mark.parametrize("b, h, m, n, d, is_causal", tests)
def test_colmean_correctness(b: int, h: int, m: int, n: int, d: int, is_causal: bool):
    query = torch.randn(b, h, m, d, dtype=torch.float16, device="cuda")
    key = torch.randn(b, h, n, d, dtype=torch.float16, device="cuda")

    target = naive_colmean(query, key, is_causal=is_causal)
    output = flash_colmean(query, key, is_causal=is_causal)

    assert target.shape == output.shape == (b, n)
    assert torch.allclose(target, output, atol=1e-3, rtol=1e-3)
