"""
Flash-ColSum: Efficient attention column-sum operations.
"""

import torch
from typing import Optional

from .kernel_noncausal_batched import _flash_colsum_batched
from .kernel_noncausal import _flash_colsum_non_batched
from .kernel_causal import _flash_colsum_causal


def flash_colsum(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
    cls_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute attention column means efficiently without materializing full attention matrix.
    
    Args:
        query: Query tensor (B, H, S, D) for non-causal or (1, H, Q_len, D) for causal
        key: Key tensor (same shape as query for non-causal or (1, H, K_len, D) for causal)
        scale: Attention scale factor. If None, uses 1/sqrt(D). Default: None
        is_causal: Whether to apply causal masking. Default: False
        cls_len: Optional number of leading query positions (CLS-style tokens)
            to average over in the non-causal case. If None (default), averages
            over all query positions.
    
    Returns:
        Column means of attention matrix:
            - Non-causal (no cls_len): (B, S) - mean over heads and all query positions
            - Non-causal with cls_len: (B, S) - mean over heads and first cls_len positions
            - Causal: (1, K_len) - mean over heads and query positions
    
    Raises:
        ValueError: If causal attention requested with B > 1
        ValueError: If input shapes are invalid
    
    Examples:
        >>> # Batched non-causal (ViT, BERT, etc.)
        >>> Q = torch.randn(8, 16, 2048, 64, device='cuda', dtype=torch.float16)
        >>> K = Q.clone()
        >>> col_mean = flash_colsum(Q, K)  # (8, 2048)
        
        >>> # Non-batched (single sample with large sequence)
        >>> Q = torch.randn(1, 16, 65536, 64, device='cuda', dtype=torch.float16)
        >>> K = Q.clone()
        >>> col_mean = flash_colsum(Q, K)  # (1, 65536)
        
        >>> # Causal attention (GPT, retrieval, etc.)
        >>> Q = torch.randn(1, 32, 128, 128, device='cuda', dtype=torch.float16)
        >>> K = torch.randn(1, 32, 4096, 128, device='cuda', dtype=torch.float16)
        >>> col_mean = flash_colsum(Q, K, is_causal=True)  # (1, 4096)
    """
    # Validate inputs
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError(f"Expected 4D tensors, got query.ndim={query.ndim}, key.ndim={key.ndim}")
    
    if query.device.type != 'cuda':
        raise ValueError(f"Flash attention only supports CUDA tensors, got {query.device}")
    
    if cls_len is not None and is_causal:
        raise ValueError("cls_len is only supported for non-causal attention")
    if cls_len is not None and cls_len <= 0:
        raise ValueError(f"cls_len must be positive, got {cls_len}")
    
    B_q, H_q, S_q, D_q = query.shape
    B_k, H_k, S_k, D_k = key.shape
    
    # Compute scale
    if scale is None:
        scale = 1.0 / (D_q ** 0.5)
    
    if is_causal:
        # Causal attention: short query to long key retrieval
        if B_q != 1 or B_k != 1:
            raise ValueError(
                f"Causal attention currently only supports batch size 1, got B_q={B_q}, B_k={B_k}"
            )
        if H_q != H_k or D_q != D_k:
            raise ValueError(
                f"Query and key must have same heads and head_dim, got "
                f"H_q={H_q}, H_k={H_k}, D_q={D_q}, D_k={D_k}"
            )
        
        return _flash_colsum_causal(query, key, scale)
    
    else:
        # Non-causal attention
        if B_q != B_k or H_q != H_k or S_q != S_k or D_q != D_k:
            raise ValueError(
                f"For non-causal attention, query and key must have same shape, got "
                f"query: {query.shape}, key: {key.shape}"
            )
        
        cls_n = -1 if cls_len is None else int(cls_len)
        
        # Choose kernel: non-batched (B=1) vs batched
        if B_q == 1:
            return _flash_colsum_non_batched(query, key, scale, CLS_N=cls_n)
        else:
            return _flash_colsum_batched(query, key, scale, CLS_N=cls_n)


__all__ = ['flash_colsum']

