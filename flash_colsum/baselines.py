"""
Naive baseline implementations for attention column-sum operations.

These implementations materialize the full attention matrix and are provided
for correctness testing and comparison. They are numerically stable but
memory- and latencyintensive for large sequences.
"""

import torch
from typing import Optional


def naive_colsum(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
    cls_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Naive baseline for attention column-sum computation.
    
    Args:
        query: Query tensor (B, H, S_q, D) or (B, H, Q_len, D) for causal
        key: Key tensor (B, H, S_k, D) or (B, H, K_len, D) for causal
        scale: Attention scale. If None, uses 1/sqrt(D)
        is_causal: Whether to apply causal masking
        cls_len: Optional number of leading query positions (CLS-style tokens)
            to average over in the non-causal case. If None (default), averages
            over all query positions.
    
    Returns:
        Column means of attention matrix: (B, S_k) or (B, K_len)
    """
    B, H, S_q, D = query.shape
    _, _, S_k, _ = key.shape
    
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    
    # Compute attention scores: (B, H, S_q, S_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Apply causal mask if needed
    if is_causal:
        # Match the retrieval-style causal semantics used by the Triton kernel:
        # when S_k > S_q, allow all queries to see the "prefix" keys and apply
        # a standard causal (triangular) mask only over the most recent window.
        if S_k >= S_q:
            Kpast = S_k - S_q
            mask = torch.zeros(S_q, S_k, device=query.device, dtype=torch.bool)
            # Prefix keys (0:Kpast) are visible to all queries
            if Kpast > 0:
                mask[:, :Kpast] = True
            # Last window uses standard causal masking
            window = torch.tril(torch.ones(S_q, S_q, device=query.device, dtype=torch.bool))
            mask[:, Kpast:] = window[:, : (S_k - Kpast)]
        else:
            # Fallback: standard causal mask when S_k <= S_q
            q_indices = torch.arange(S_q, device=query.device).unsqueeze(1)
            k_indices = torch.arange(S_k, device=key.device).unsqueeze(0)
            mask = k_indices <= q_indices
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Softmax and compute column means
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    # Non-causal: optionally restrict to first cls_len query positions
    if not is_causal and cls_len is not None:
        if cls_len <= 0:
            raise ValueError(f"cls_len must be positive, got {cls_len}")
        cls_n_eff = min(int(cls_len), S_q)
        attn_reduced = attn[:, :, :cls_n_eff, :]
        col_mean = attn_reduced.mean(dim=(1, 2))  # (B, S_k)
    else:
    # Mean over query positions (rows) and heads
        col_mean = attn.mean(dim=(1, 2))  # (B, S_k)
    
    return col_mean


__all__ = ['naive_colsum']

