// Copyright (C) 2026 Tencent.
//
// Qwen3-TTS MTP code-predictor NeoX RoPE.
//
// This is a dedicated kernel for the MTP re-prefill path where q/k are already
// split and normalized as [B, H, S, D], cos/sin are [B, S, D], and D=128.

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#include "src/rope/rope.h"

namespace hpc {
namespace rope {
namespace qwen3_mtp_rope_kernels {

template <int kHeadDim, int kRowsPerBlock, int kThreadsPerRow>
__global__ void qwen3_mtp_rope_kernel(__nv_bfloat16 *q_out, __nv_bfloat16 *k_out,
                                      const __nv_bfloat16 *q, const __nv_bfloat16 *k,
                                      const __nv_bfloat16 *cos, const __nv_bfloat16 *sin,
                                      int batch_size, int seq_len, int num_q_heads,
                                      int num_kv_heads, int64_t q_s0, int64_t q_s1,
                                      int64_t q_s2, int64_t k_s0, int64_t k_s1,
                                      int64_t k_s2, int64_t cos_s0, int64_t cos_s1,
                                      int64_t sin_s0, int64_t sin_s1) {
  constexpr int kHalfDim = kHeadDim / 2;
  static_assert(kHalfDim == 64, "qwen3_mtp_rope expects head_dim=128");
  static_assert(kHalfDim % kThreadsPerRow == 0, "threads per row must divide half dim");
  constexpr int kPairsPerThread = kHalfDim / kThreadsPerRow;

  const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kThreadsPerRow;
  const int lane = threadIdx.x % kThreadsPerRow;
  const int total_q_rows = batch_size * num_q_heads * seq_len;
  const int total_rows = total_q_rows + batch_size * num_kv_heads * seq_len;
  if (row >= total_rows) return;

  // === [perf mtp-rope-rowpack 2026-06-25] 32 threads/row x 8 rows/CTA ========
  // For bs>=8, 32 threads/row halves CTA count versus 64 threads/row while each
  // thread handles two rotate_half pairs.  The launcher keeps a bs<=1 branch
  // with 64 threads/row, which was slightly faster for the tiny-batch case.
  if (row < total_q_rows) {
    const int s = row % seq_len;
    const int tmp = row / seq_len;
    const int h = tmp % num_q_heads;
    const int b = tmp / num_q_heads;
    const __nv_bfloat16 *q_row = q + b * q_s0 + h * q_s1 + s * q_s2;
    __nv_bfloat16 *out_row = q_out + static_cast<int64_t>(row) * kHeadDim;
    const __nv_bfloat16 *cos_row = cos + b * cos_s0 + s * cos_s1;
    const __nv_bfloat16 *sin_row = sin + b * sin_s0 + s * sin_s1;
#pragma unroll
    for (int r = 0; r < kPairsPerThread; ++r) {
      const int i = lane + r * kThreadsPerRow;
      const float x1 = __bfloat162float(q_row[i]);
      const float x2 = __bfloat162float(q_row[i + kHalfDim]);
      const float c1 = __bfloat162float(cos_row[i]);
      const float s1v = __bfloat162float(sin_row[i]);
      const float c2 = __bfloat162float(cos_row[i + kHalfDim]);
      const float s2v = __bfloat162float(sin_row[i + kHalfDim]);
      out_row[i] = __float2bfloat16(x1 * c1 - x2 * s1v);
      out_row[i + kHalfDim] = __float2bfloat16(x2 * c2 + x1 * s2v);
    }
  } else {
    const int krow = row - total_q_rows;
    const int s = krow % seq_len;
    const int tmp = krow / seq_len;
    const int h = tmp % num_kv_heads;
    const int b = tmp / num_kv_heads;
    const __nv_bfloat16 *k_row = k + b * k_s0 + h * k_s1 + s * k_s2;
    __nv_bfloat16 *out_row = k_out + static_cast<int64_t>(krow) * kHeadDim;
    const __nv_bfloat16 *cos_row = cos + b * cos_s0 + s * cos_s1;
    const __nv_bfloat16 *sin_row = sin + b * sin_s0 + s * sin_s1;
#pragma unroll
    for (int r = 0; r < kPairsPerThread; ++r) {
      const int i = lane + r * kThreadsPerRow;
      const float x1 = __bfloat162float(k_row[i]);
      const float x2 = __bfloat162float(k_row[i + kHalfDim]);
      const float c1 = __bfloat162float(cos_row[i]);
      const float s1v = __bfloat162float(sin_row[i]);
      const float c2 = __bfloat162float(cos_row[i + kHalfDim]);
      const float s2v = __bfloat162float(sin_row[i + kHalfDim]);
      out_row[i] = __float2bfloat16(x1 * c1 - x2 * s1v);
      out_row[i + kHalfDim] = __float2bfloat16(x2 * c2 + x1 * s2v);
    }
  }
}

}  // namespace qwen3_mtp_rope_kernels

void qwen3_mtp_rope_async(__nv_bfloat16 *q_out, __nv_bfloat16 *k_out, const __nv_bfloat16 *q,
                          const __nv_bfloat16 *k, const __nv_bfloat16 *cos,
                          const __nv_bfloat16 *sin, int batch_size, int seq_len,
                          int num_q_heads, int num_kv_heads, int head_dim, int64_t q_s0,
                          int64_t q_s1, int64_t q_s2, int64_t k_s0, int64_t k_s1,
                          int64_t k_s2, int64_t cos_s0, int64_t cos_s1, int64_t sin_s0,
                          int64_t sin_s1, cudaStream_t stream) {
  if (head_dim != 128) return;

  const int total_rows = batch_size * seq_len * (num_q_heads + num_kv_heads);
  if (batch_size <= 1) {
    // === [perf mtp-rope-bs1 2026-06-25] small-batch branch ==================
    // 64 threads/row x 4 rows/CTA was slightly faster for bs=1 in profiling.
    constexpr int kRowsPerBlock = 4;
    constexpr int kThreadsPerRow = 64;
    dim3 grid((total_rows + kRowsPerBlock - 1) / kRowsPerBlock);
    dim3 block(kThreadsPerRow * kRowsPerBlock);
    qwen3_mtp_rope_kernels::qwen3_mtp_rope_kernel<128, kRowsPerBlock, kThreadsPerRow>
        <<<grid, block, 0, stream>>>(q_out, k_out, q, k, cos, sin, batch_size, seq_len,
                                     num_q_heads, num_kv_heads, q_s0, q_s1, q_s2, k_s0,
                                     k_s1, k_s2, cos_s0, cos_s1, sin_s0, sin_s1);
  } else {
    constexpr int kRowsPerBlock = 8;
    constexpr int kThreadsPerRow = 32;
    dim3 grid((total_rows + kRowsPerBlock - 1) / kRowsPerBlock);
    dim3 block(kThreadsPerRow * kRowsPerBlock);
    qwen3_mtp_rope_kernels::qwen3_mtp_rope_kernel<128, kRowsPerBlock, kThreadsPerRow>
        <<<grid, block, 0, stream>>>(q_out, k_out, q, k, cos, sin, batch_size, seq_len,
                                     num_q_heads, num_kv_heads, q_s0, q_s1, q_s2, k_s0,
                                     k_s1, k_s2, cos_s0, cos_s1, sin_s0, sin_s1);
  }
}

}  // namespace rope
}  // namespace hpc
