# Copyright (C) 2026 Tencent.
"""Tests and microbenchmarks for the Qwen3-TTS MTP RoPE operator.

The CUDA op is specialized for NeoX-style rotate_half layout with head_dim=128,
but the test matrix covers a wider set of TTS-like MTP shapes: different batch
sizes, code-group sequence lengths, attention head counts, and GQA ratios.

Run correctness:
    pytest tests/test_mtp_rope.py -q

Run the microbenchmark table:
    python3 tests/test_mtp_rope.py --bench
    python3 tests/test_mtp_rope.py --bench --shapes qwen3_tts,gqa1,no_gqa --bs 1,8,32
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc  # noqa: E402


ROPE_THETA = 1_000_000.0


@dataclass(frozen=True)
class MtpRopeShape:
    name: str
    seq_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int = 128


TTS_MTP_SHAPES = (
    MtpRopeShape("qwen3_tts", seq_len=16, num_q_heads=16, num_kv_heads=8),
    MtpRopeShape("qwen3_tts_full_groups", seq_len=17, num_q_heads=16, num_kv_heads=8),
    MtpRopeShape("small_tts", seq_len=8, num_q_heads=8, num_kv_heads=4),
    MtpRopeShape("gqa1", seq_len=17, num_q_heads=16, num_kv_heads=1),
    MtpRopeShape("no_gqa", seq_len=16, num_q_heads=8, num_kv_heads=8),
    MtpRopeShape("wide_heads", seq_len=32, num_q_heads=32, num_kv_heads=8),
)


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


def make_inputs(batch_size: int, shape: MtpRopeShape):
    q = torch.randn(
        (batch_size, shape.num_q_heads, shape.seq_len, shape.head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        (batch_size, shape.num_kv_heads, shape.seq_len, shape.head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )

    inv_freq = 1.0 / (
        ROPE_THETA
        ** (
            torch.arange(0, shape.head_dim, 2, dtype=torch.float32, device="cuda")
            / shape.head_dim
        )
    )
    position_ids = torch.arange(shape.seq_len, device="cuda", dtype=torch.long).unsqueeze(0)
    position_ids = position_ids.expand(batch_size, -1).contiguous()
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    with torch.autocast(device_type="cuda", enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=torch.bfloat16)
        sin = emb.sin().to(dtype=torch.bfloat16)
    return q, k, cos, sin


def _shape_ids(shape: MtpRopeShape) -> str:
    return (
        f"{shape.name}_S{shape.seq_len}_Hq{shape.num_q_heads}_"
        f"Hkv{shape.num_kv_heads}_D{shape.head_dim}"
    )


@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("shape", TTS_MTP_SHAPES, ids=_shape_ids)
def test_qwen3_mtp_rope_tts_shapes_golden(batch_size, shape):
    torch.manual_seed(0x20260709 + batch_size + shape.seq_len + shape.num_q_heads)
    q, k, cos, sin = make_inputs(batch_size, shape)

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
    bad_shape = MtpRopeShape(
        "bad_head_dim", seq_len=16, num_q_heads=16, num_kv_heads=8, head_dim=64
    )
    q, k, cos, sin = make_inputs(1, bad_shape)
    with pytest.raises((RuntimeError, ValueError)):
        hpc.qwen3_mtp_rope(q, k, cos, sin)


def _time_event(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3


def _warmup(*fns, iters: int) -> None:
    for _ in range(iters):
        for fn in fns:
            fn()
    torch.cuda.synchronize()


def _bench_shape(batch_size: int, shape: MtpRopeShape, iters: int, warmup: int) -> None:
    q, k, cos, sin = make_inputs(batch_size, shape)
    compiled_ref = torch.compile(ref_mtp_rope, dynamic=False, options={"epilogue_fusion": False})

    def eager_call():
        ref_mtp_rope(q, k, cos, sin)

    def compile_call():
        compiled_ref(q, k, cos, sin)

    def hpc_call():
        hpc.qwen3_mtp_rope(q, k, cos, sin)

    _warmup(eager_call, compile_call, hpc_call, iters=warmup)
    eager_us = _time_event(eager_call, iters)
    compile_us = _time_event(compile_call, iters)
    hpc_us = _time_event(hpc_call, iters)
    speedup = compile_us / hpc_us if hpc_us > 0 else float("nan")
    rows = batch_size * shape.seq_len * (shape.num_q_heads + shape.num_kv_heads)
    print(
        f"{shape.name:22s} bs={batch_size:3d} S={shape.seq_len:2d} "
        f"Hq={shape.num_q_heads:2d} Hkv={shape.num_kv_heads:2d} rows={rows:5d} | "
        f"eager={eager_us:8.2f}us compile={compile_us:8.2f}us "
        f"hpc={hpc_us:8.2f}us compile/hpc={speedup:5.2f}x"
    )


def _select_shapes(names: str) -> list[MtpRopeShape]:
    if not names or names == "all":
        return list(TTS_MTP_SHAPES)
    want = {name.strip() for name in names.split(",") if name.strip()}
    by_name = {shape.name: shape for shape in TTS_MTP_SHAPES}
    missing = sorted(want - set(by_name))
    if missing:
        raise ValueError(f"unknown shape names: {missing}; available={sorted(by_name)}")
    return [by_name[name] for name in want]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="run timing table instead of pytest")
    parser.add_argument("--shapes", default="all", help="comma-separated shape names or 'all'")
    parser.add_argument("--bs", default="1,8,32", help="comma-separated batch sizes")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if not args.bench:
        parser.print_help()
        return 0

    batches = [int(x) for x in args.bs.split(",") if x]
    for shape in _select_shapes(args.shapes):
        for batch_size in batches:
            _bench_shape(batch_size, shape, args.iters, args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
