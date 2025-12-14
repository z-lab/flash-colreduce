import pytest
import torch

from flash_colreduce import flash_colreduce, naive_colreduce


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for these tests")
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("d", [16, 32, 64])
@pytest.mark.parametrize("m,n", [(128, 128), (131, 131), (128, 1024), (131, 1027)])
@pytest.mark.parametrize("h", [8, 13])
@pytest.mark.parametrize("b", [1, 4, 7])
def test_correctness(
    b: int,
    h: int,
    m: int,
    n: int,
    d: int,
    is_causal: bool,
    reduction: str,
):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    query = torch.randn(b, h, m, d, device="cuda", dtype=torch.float16)
    key = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)

    output = flash_colreduce(query, key, is_causal=is_causal, reduction=reduction)
    target = naive_colreduce(query, key, is_causal=is_causal, reduction=reduction)

    torch.testing.assert_close(output, target, atol=1e-3, rtol=1e-3)
