import math

import torch

__all__ = ["naive_colreduce"]


def naive_colreduce(
    query: torch.Tensor,
    key: torch.Tensor,
    is_causal: bool = False,
    scale: float | None = None,
    reduction: str = "sum",
) -> torch.Tensor:
    m, n = query.shape[2], key.shape[2]
    if scale is None:
        scale = 1 / math.sqrt(query.shape[-1])

    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if is_causal:
        c = max(n - m, 0)
        q = torch.arange(m, device=query.device).view(1, 1, -1, 1)
        k = torch.arange(n, device=query.device).view(1, 1, 1, -1)
        scores = scores.masked_fill(q + c < k, -float("inf"))
    scores = torch.softmax(scores, dim=-1)

    if reduction == "sum":
        scores = scores.sum(dim=2)
    elif reduction == "mean":
        scores = scores.sum(dim=2)
        if is_causal:
            scores[..., :c] /= m
            scores[..., c:] /= torch.arange(n - c, 0, -1, device=query.device)
        else:
            scores /= m
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'sum', or 'mean'.")
    return scores.to(query.dtype)
