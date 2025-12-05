import torch
import triton
import triton.language as tl
import time

__all__ = ["_flash_colsum_batched"]

@triton.jit
def _col_mean_attn_streaming_kernel(
    Q_ptr, K_ptr, Out_ptr,
    B, H, S, D, SCALE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_ob, stride_os,
    CLS_N: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    s_offsets = tl.arange(0, BLOCK_S)
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    row_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    row_mask = row_offsets < S

    # Load Q tile [BLOCK_Q, D]
    Q_tile = tl.load(
        Q_ptr + pid_b * stride_qb
        + pid_h * stride_qh
        + row_offsets[:, None] * stride_qs
        + d_offsets[None, :] * stride_qd,
        mask=row_mask[:, None] & d_mask[None, :],
        other=0.0
    ).to(tl.float32) * SCALE

    # Pass 1: Streaming row_max and row_sum for normalization
    row_max = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    for k0 in range(0, S, BLOCK_S):
        k_offsets = k0 + s_offsets
        k_mask = k_offsets < S

        K_tile = tl.load(
            K_ptr + pid_b * stride_kb
            + pid_h * stride_kh
            + k_offsets[:, None] * stride_ks
            + d_offsets[None, :] * stride_kd,
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0
        ).to(tl.float32)

        scores = tl.dot(Q_tile, tl.trans(K_tile))  # [BLOCK_Q, BLOCK_S]
        tile_max = tl.max(scores, axis=1)

        new_max = tl.maximum(row_max, tile_max)
         
        # Update row_sum with logsumexp accumulation
        exp_row_max = tl.exp(row_max - new_max)
        exp_tile = tl.exp(scores - new_max[:, None])
        row_sum = exp_row_max * row_sum + tl.sum(exp_tile, axis=1)
        row_max = new_max

    # Pass 2: Accumulate column sums (mean over rows, heads)
    
    # Before the k-loop
    if CLS_N == -1:
        cls_row_mask = row_mask
    else:
        CLS_N_EFF = tl.minimum(CLS_N, S)
        cls_row_mask = (row_offsets < CLS_N_EFF) & row_mask
        
    for k0 in range(0, S, BLOCK_S):
        k_offsets = k0 + s_offsets
        k_mask = k_offsets < S

        K_tile = tl.load(
            K_ptr + pid_b * stride_kb
            + pid_h * stride_kh
            + k_offsets[:, None] * stride_ks
            + d_offsets[None, :] * stride_kd,
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0
        ).to(tl.float32)

        scores = tl.dot(Q_tile, tl.trans(K_tile))  # [BLOCK_Q, BLOCK_S]
        scores_exp = tl.exp(scores - row_max[:, None])
        attn = scores_exp / row_sum[:, None]
        
        attn_masked = attn * cls_row_mask[:, None]
        col_sum_tile = tl.sum(attn_masked, axis=0)    # [BLOCK_S]
        
        out_ptrs = Out_ptr + pid_b * stride_ob + k_offsets * stride_os
        tl.atomic_add(out_ptrs, col_sum_tile, mask=k_mask)
        
        


def _flash_colsum_batched(Q: torch.Tensor, K: torch.Tensor, scale: float, CLS_N: int = -1):
    B, H, S, D = Q.shape
    out = torch.zeros((B, S), device=Q.device, dtype=torch.float32)

    BLOCK_Q = 64
    BLOCK_S = 64
    BLOCK_D = 64

    grid = (B, H, triton.cdiv(S, BLOCK_Q))

    _col_mean_attn_streaming_kernel[grid](
        Q, K, out,
        B, H, S, D, scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        out.stride(0), out.stride(1),
        CLS_N=CLS_N,
        BLOCK_Q=BLOCK_Q, BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
    )

    norm_factor = H * S if CLS_N == -1 else H * CLS_N
    # Normalize by number of heads * number of keys (to get mean over rows and heads)
    out = (out / norm_factor).to(torch.float16)
    return out


if __name__ == "__main__":
    B, H, S, D = 16, 32, 1024, 64
    Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    scale = 1.0 / (D ** 0.5)

    # -----------------------------
    # Eager Mode (Reference)
    # -----------------------------
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = torch.nn.functional.softmax(attn_scores, dtype=torch.float32, dim=-1).to(Q.dtype)
    attn_col_mean = attn_weights.mean(dim=-2).mean(dim=1).to(torch.float16)
    assert attn_col_mean.shape == (B, S)
    # -----------------------------
    # Triton Kernel (Optimized)
    # -----------------------------
    col_sum_new = _flash_colsum_batched(Q, K, scale)
    assert col_sum_new.shape == (B, S)
    # -----------------------------
    # Check Results
    # -----------------------------
    # Correctness Assertion
    max_diff = (attn_col_mean - col_sum_new).abs().max()
    print(f"Max diff = {max_diff:.6f}")
    assert torch.allclose(attn_col_mean, col_sum_new, atol=1e-3, rtol=1e-3), \
        f"Mismatch: max diff = {max_diff}"
    print(f"Correctness check passed! Max diff = {max_diff:.6f}")

    #* Check cls token comparison
    CLS_N = 1
    # -----------------------------
    # Eager Mode (Reference)
    # -----------------------------
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = torch.nn.functional.softmax(attn_scores, dtype=torch.float32, dim=-1).to(Q.dtype)
    attn_col_mean = attn_weights[:, :, :CLS_N,:].mean(dim=-2).mean(dim=1).to(torch.float16)
    assert attn_col_mean.shape == (B, S)
    # -----------------------------
    # Triton Kernel (Optimized)
    # -----------------------------
    col_sum_new = _flash_colsum_batched(Q, K, scale, CLS_N)
    assert col_sum_new.shape == (B, S)
    # -----------------------------
    # Check Results
    # -----------------------------
    # Correctness Assertion
    max_diff = (attn_col_mean - col_sum_new).abs().max()
    print(f"Max diff = {max_diff:.6f}")
    assert torch.allclose(attn_col_mean, col_sum_new, atol=1e-3, rtol=1e-3), \
        f"Mismatch: max diff = {max_diff}"
    print(f"Correctness check passed! Max diff = {max_diff:.6f}")
