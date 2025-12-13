import math

import torch

__all__ = ["naive_colsum", "naive_colmean"]


def naive_colsum(
    query: torch.Tensor, key: torch.Tensor, scale: float = None, is_causal: bool = False
) -> torch.Tensor:
    _, _, m, _ = query.shape
    _, _, n, _ = key.shape

    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale

    if is_causal:
        # Right-aligned causal mask: q_idx + K_PAST >= k_idx
        K_PAST = max(0, n - m)
        q_idx = torch.arange(m, device=query.device).unsqueeze(1) + K_PAST
        k_idx = torch.arange(n, device=query.device).unsqueeze(0)
        mask = q_idx >= k_idx  # (m, n)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return attn.sum(dim=(1, 2)).to(query.dtype)


def naive_colmean(
    query: torch.Tensor,
    key: torch.Tensor,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    _, h, m, _ = query.shape
    _, _, n, _ = key.shape

    output = naive_colsum(
        query,
        key,
        is_causal=is_causal,
        scale=scale,
    )
    output /= h

    if is_causal:
        c = max(0, n - m)
        output[:, :c] /= m
        output[:, c:] /= torch.arange(n - c, 0, -1, device=query.device)
    else:
        output /= m
    return output
