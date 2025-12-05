import torch
import triton
import triton.language as tl

__all__ = ["_flash_colsum_causal"]

@triton.jit
def col_mean_attn_qsmall_kernel(
    Q_ptr, K_ptr, Out_ptr,
    H, Q, K, D, K_PAST, SCALE,
    stride_qh, stride_qs, stride_qd,
    stride_kh, stride_ks, stride_kd,
    stride_os,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # program ids: heads × query blocks
    pid_h = tl.program_id(0)
    pid_q = tl.program_id(1)

    d_offsets = tl.arange(0, BLOCK_D)
    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < Q
    d_mask = d_offsets < D

    # Load Q block [BLOCK_Q, D]
    Q_tile = tl.load(
        Q_ptr + pid_h * stride_qh
               + q_offsets[:, None] * stride_qs
               + d_offsets[None, :] * stride_qd,
        mask=q_mask[:, None] & d_mask[None, :],
        other=0.0
    ).to(tl.float32) * SCALE

    # Init row_max, row_sum for softmax normalization
    row_max = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    # Precompute per-query causal thresholds to avoid recomputing additions
    q_thresholds = K_PAST + q_offsets  # shape [BLOCK_Q]

    # ------------ Pass 1: row_max, row_sum over all keys ------------
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        K_tile = tl.load(
            K_ptr + pid_h * stride_kh
                   + k_offsets[:, None] * stride_ks
                   + d_offsets[None, :] * stride_kd,
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0
        ).to(tl.float32)

        scores = tl.dot(Q_tile, tl.trans(K_tile))  # [BLOCK_Q, BLOCK_K]
        # Apply causal mask: allow keys where k_idx <= q_threshold
        # causal_mask = k_offsets[None, :] <= q_thresholds[:, None]
        causal_mask = (k_mask[None, :]) & (q_mask[:, None]) & (k_offsets[None, :] <= q_thresholds[:, None])
        scores = tl.where(causal_mask, scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)

        exp_row_max = tl.exp(row_max - new_max)
        exp_tile = tl.exp(scores - new_max[:, None])

        row_sum = exp_row_max * row_sum + tl.sum(exp_tile, axis=1)
        row_max = new_max

    # ------------ Pass 2: accumulate contributions ------------
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        K_tile = tl.load(
            K_ptr + pid_h * stride_kh
                   + k_offsets[:, None] * stride_ks
                   + d_offsets[None, :] * stride_kd,
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0
        ).to(tl.float32)

        scores = tl.dot(Q_tile, tl.trans(K_tile))  # [BLOCK_Q, BLOCK_K]
        # Apply causal mask again for the accumulation pass
        causal_mask = (k_mask[None, :]) & (q_mask[:, None]) & (k_offsets[None, :] <= q_thresholds[:, None])
        scores = tl.where(causal_mask, scores, -float("inf"))
        scores_exp = tl.exp(scores - row_max[:, None])
        attn_block = scores_exp / row_sum[:, None]

        col_sum_tile = tl.sum(attn_block, axis=0)  # [BLOCK_K]

        out_ptrs = Out_ptr + k_offsets * stride_os
        tl.atomic_add(out_ptrs, col_sum_tile, mask=k_mask)

def _flash_colsum_causal(Q: torch.Tensor, K: torch.Tensor, scale: float, K_past=None):
    """
    Q: (1, H, Q, D)  -- short queries
    K: (1, H, K, D)  -- long keys
    K_past: optional number of prefill keys (excluding current query tokens). If None,
            it is inferred as K_len - Q_len (query-prefill layout: [past | query-keys]).
    returns: (K,)
    """
    _, H, Q_len, D = Q.shape
    _, _, K_len, _ = K.shape

    out = torch.zeros((K_len,), device=Q.device, dtype=torch.float32)
    BLOCK_Q = min(16, Q_len)   
    BLOCK_K = 64               
    BLOCK_D = 64

    grid = (H, triton.cdiv(Q_len, BLOCK_Q))

    # infer K_past if not provided (assumes K contains [past, query-keys])
    K_past_val = K_past if K_past is not None else max(0, K_len - Q_len)

    col_mean_attn_qsmall_kernel[grid](
        Q, K, out,
        H, Q_len, K_len, D, K_past_val, scale,
        Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(1), K.stride(2), K.stride(3),
        out.stride(0),
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
    )

    out = (out / (H * Q_len)).to(torch.float16)
    return out.view(1, -1)


if __name__ == "__main__":
    B, H, Q_len, K_len, D = 1, 32, 128, 64000, 128
    Q = torch.randn(B, H, Q_len, D, device='cuda', dtype=torch.float16)
    K = torch.randn(B, H, K_len, D, device='cuda', dtype=torch.float16)
    scale = 1.0 / (D ** 0.5)

    # -----------------------------
    # Eager Mode (Reference)
    # -----------------------------
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  
    Kpast = K_len - Q_len
    mask = torch.ones(Q_len, K_len, device=Q.device, dtype=torch.bool)
    mask[:, Kpast:] = torch.tril(torch.ones(Q_len, Q_len, device=Q.device, dtype=torch.bool))
    attn = attn_scores.masked_fill(~mask[None, None], float("-inf"))
    attn_weights = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(Q.dtype)
    attn_col_mean = attn_weights.mean(dim=-2).mean(dim=1).to(torch.float16)
    assert attn_col_mean.shape == (B, K_len)
    # -----------------------------
    # Triton Kernel (Optimized)
    # -----------------------------
    col_sum_new = _flash_colsum_causal(Q, K, scale)
    assert col_sum_new.shape == (B, K_len)
    # -----------------------------
    # Check Results
    # -----------------------------
    # Correctness Assertion
    max_diff = (attn_col_mean - col_sum_new).abs().max()
    print(f"Max diff = {max_diff:.6f}")
    assert torch.allclose(attn_col_mean, col_sum_new, atol=1e-3, rtol=1e-3), \
        f"Mismatch: max diff = {max_diff}"
    print(f"Correctness check passed! Max diff = {max_diff:.6f}")