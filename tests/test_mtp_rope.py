# Copyright (C) 2026 Tencent.
"""Tests for the Qwen3-TTS MTP code-predictor RoPE operator."""

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc  # noqa: E402


SEQ_LEN = 16
HEAD_DIM = 128
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
ROPE_THETA = 1_000_000.0


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def ref_mtp_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def make_inputs(batch_size: int):
    q = torch.randn(
        (batch_size, NUM_Q_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    k = torch.randn(
        (batch_size, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )

    inv_freq = 1.0 / (
        ROPE_THETA
        ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device="cuda") / HEAD_DIM)
    )
    position_ids = torch.arange(SEQ_LEN, device="cuda", dtype=torch.long).unsqueeze(0)
    position_ids = position_ids.expand(batch_size, -1).contiguous()
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    with torch.autocast(device_type="cuda", enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=torch.bfloat16)
        sin = emb.sin().to(dtype=torch.bfloat16)
    return q, k, cos, sin


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_qwen3_mtp_rope_golden(batch_size):
    torch.manual_seed(0x20260709 + batch_size)
    q, k, cos, sin = make_inputs(batch_size)

    ref_q, ref_k = ref_mtp_rope(q, k, cos, sin)
    out_q, out_k = hpc.qwen3_mtp_rope(q, k, cos, sin)

    assert out_q.shape == q.shape
    assert out_k.shape == k.shape
    assert out_q.dtype == torch.bfloat16
    assert out_k.dtype == torch.bfloat16
    assert out_q.is_contiguous()
    assert out_k.is_contiguous()
    assert torch.allclose(out_q.float(), ref_q.float(), atol=5e-2, rtol=5e-2)
    assert torch.allclose(out_k.float(), ref_k.float(), atol=5e-2, rtol=5e-2)


def test_qwen3_mtp_rope_rejects_bad_head_dim():
    q = torch.randn((1, NUM_Q_HEADS, SEQ_LEN, 64), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((1, NUM_KV_HEADS, SEQ_LEN, 64), dtype=torch.bfloat16, device="cuda")
    cos = torch.randn((1, SEQ_LEN, 64), dtype=torch.bfloat16, device="cuda")
    sin = torch.randn_like(cos)
    with pytest.raises((RuntimeError, ValueError)):
        hpc.qwen3_mtp_rope(q, k, cos, sin)
