import torch
import triton
import triton.language as tl
import math
import time

__all__ = ["_flash_colsum_non_batched"]

@triton.jit
def _col_mean_attn_streaming_kernel_b1(
    Q_ptr, K_ptr, Out_ptr,
    H, S, D, SCALE,
    stride_qh, stride_qs, stride_qd,
    stride_kh, stride_ks, stride_kd,
    stride_os,
    CLS_N: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_q = tl.program_id(1)

    s_offsets = tl.arange(0, BLOCK_S)
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    row_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    row_mask = row_offsets < S

    # Q tile [BLOCK_Q, D] in fp16, accumulate in fp32
    Q_tile_f16 = tl.load(
        Q_ptr + pid_h * stride_qh
              + row_offsets[:, None] * stride_qs
              + d_offsets[None, :] * stride_qd,
        mask=row_mask[:, None] & d_mask[None, :],
        other=0.0
    ).to(tl.float16)

    # Pass 1: streaming row_max, row_sum
    row_max = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    
    inv_ln2 = 1.4426950408889634  # add once near row_sum/row_max init

    for k0 in range(0, S, BLOCK_S):
        k_offsets = k0 + s_offsets
        k_mask = k_offsets < S

        K_tile_f16 = tl.load(
            K_ptr + pid_h * stride_kh
                  + k_offsets[:, None] * stride_ks
                  + d_offsets[None, :] * stride_kd,
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0
        ).to(tl.float16)

        # fp16×fp16 -> fp32 matmul
        scores = tl.dot(Q_tile_f16, tl.trans(K_tile_f16), out_dtype=tl.float32, input_precision='tf32')
        scores = scores * SCALE
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)

        # streaming LSE
        row_sum = tl.exp2((row_max - new_max) * inv_ln2) * row_sum \
        + tl.sum(tl.exp2((scores - new_max[:, None]) * inv_ln2), axis=1)
        row_max = new_max

    # Pass 2: accumulate column sums into fp32
    # Determine which query rows contribute (CLS prefix or all)
    if CLS_N == -1:
        cls_row_mask = row_mask
    else:
        CLS_N_EFF = tl.minimum(CLS_N, S)
        cls_row_mask = (row_offsets < CLS_N_EFF) & row_mask

    for k0 in range(0, S, BLOCK_S):
        k_offsets = k0 + s_offsets
        k_mask = k_offsets < S

        K_tile_f16 = tl.load(
            K_ptr + pid_h * stride_kh
                  + k_offsets[:, None] * stride_ks
                  + d_offsets[None, :] * stride_kd,
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0
        ).to(tl.float16)

        scores = tl.dot(Q_tile_f16, tl.trans(K_tile_f16), out_dtype=tl.float32, input_precision='tf32')
        scores = scores * SCALE
        attn = tl.exp2((scores - row_max[:, None]) * inv_ln2) / row_sum[:, None]
        attn_masked = attn * cls_row_mask[:, None]
        col_sum_tile = tl.sum(attn_masked, axis=0).to(tl.float32)  # [BS]

        tl.atomic_add(Out_ptr + k_offsets * stride_os, col_sum_tile, mask=k_mask)
        
def _flash_colsum_non_batched(Q: torch.Tensor, K: torch.Tensor, scale: float, CLS_N: int = -1):
    B, H, S, D = Q.shape
    
    out32 = torch.zeros((S,), device=Q.device, dtype=torch.float32)
    BLOCK_Q = 128
    BLOCK_S = 128
    BLOCK_D = 64

    grid = (H, triton.cdiv(S, BLOCK_Q))  # no batch axis

    _col_mean_attn_streaming_kernel_b1[grid](
        Q, K, out32,
        H, S, D, scale,
        Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(1), K.stride(2), K.stride(3),
        out32.stride(0),
        CLS_N=CLS_N,
        BLOCK_Q=BLOCK_Q, BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )
    # Mean over heads and rows: divide by H * (# contributing query positions)
    eff_rows = S if CLS_N == -1 else min(CLS_N, S)
    return (out32 / (H * eff_rows)).to(torch.float16).unsqueeze(0)


if __name__ == "__main__":
    B, H, S, D = 1, 16, 4096, 64
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
    col_sum_new = _flash_colsum_non_batched(Q, K, scale)
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
    col_sum_new = _flash_colsum_non_batched(Q, K, scale, CLS_N)
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
    
    