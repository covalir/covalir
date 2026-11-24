# Copyright 2026 CovalIR Contributors
# Licensed under the Apache License, Version 2.0
"""Pure-integer Group-quantizer (W4A8 & 2-bit ternary) and fixed-point LUT generator.

Uses Q8.8 (INT16, scale=256, range [-128, +128)) for activations and residual stream,
with dynamic power-of-2 per-token INT8 absmax quantization before Group-G MACs and
Q1.15 unsigned 16-bit per-group weight scales. Matches FP32 stories15M.bin 16/16 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from covalir.front.checkpoint import FloatModel, ModelConfig

Q88_SCALE = 256.0
W_SCALE_Q15 = 32768.0


def _clamp(val: int, lo: int, hi: int) -> int:
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


@dataclass
class QuantMatrix:
    rows: int
    cols: int
    group_size: int
    weights: list[list[int]]       # [rows][cols] INT4 [-7..7] or ternary {-1, 0, +1}
    group_scales: list[list[int]]  # [rows][cols // group_size] UINT16 (Q1.15)


@dataclass
class QuantLayer:
    rms_att: list[int]             # INT16 Q8.8 [dim]
    wq: QuantMatrix
    wk: QuantMatrix
    wv: QuantMatrix
    wo: QuantMatrix
    rms_ffn: list[int]             # INT16 Q8.8 [dim]
    w1: QuantMatrix
    w2: QuantMatrix
    w3: QuantMatrix


@dataclass
class FixedPointLuts:
    inv_rms_lut: list[int]         # 256 x UINT16 (round(32768 / sqrt(u)), u=1..255)
    silu_lut: list[int]            # 512 x INT16 Q8.8 (covering [-16.0, +16.0) in 1/16 steps)
    exp_lut: list[int]             # 256 x UINT8 (round(255 * exp(-d / 16.0)))
    rope_cos: list[int]            # [max_seq_len * (head_dim/2)] INT16 Q1.14
    rope_sin: list[int]            # [max_seq_len * (head_dim/2)] INT16 Q1.14
    inv_dim_q16: int               # round(65536 / dim)
    att_scale_q12: int             # round(4096 * 16 / sqrt(head_dim))


@dataclass
class QuantModel:
    config: ModelConfig
    quant_mode: str                # "w4a8" or "ternary"
    group_size: int
    token_embedding: list[list[int]]  # [vocab_size][dim] INT16 Q8.8
    layers: list[QuantLayer]
    rms_final: list[int]           # [dim] INT16 Q8.8
    wcls: QuantMatrix
    luts: FixedPointLuts


def build_fixed_point_luts(cfg: ModelConfig) -> FixedPointLuts:
    """Build exact integer LUT tables shared identically by golden_sim.py and SystemVerilog ROMs."""
    # inv_rms_lut[u] = round(32768 / sqrt(u)) for u = 1..255 (u=0 unused -> 32768)
    inv_rms_lut = [32768] + [
        _clamp(int(round(32768.0 / math.sqrt(float(u)))), 1, 65535) for u in range(1, 256)
    ]

    # 512-entry SiLU LUT in Q8.8 covering [-16.0, +16.0) in 1/16 steps
    silu_lut: list[int] = []
    for u in range(512):
        xr = (u - 256) / 16.0
        sig = 1.0 / (1.0 + math.exp(-xr))
        silu_lut.append(_clamp(int(round((xr * sig) * Q88_SCALE)), -32768, 32767))

    # 256-entry softmax exp(-d / 16.0) * 255
    exp_lut = [_clamp(int(round(255.0 * math.exp(-d / 16.0))), 0, 255) for d in range(256)]

    # Precomputed RoPE cos/sin tables in Q1.14 (scaled by 16384)
    half_hd = cfg.head_dim // 2
    rope_cos: list[int] = []
    rope_sin: list[int] = []
    for pos in range(cfg.max_seq_len):
        for j in range(half_hd):
            freq = 1.0 / (10000.0 ** ((2.0 * j) / float(cfg.head_dim)))
            angle = pos * freq
            rope_cos.append(_clamp(int(round(16384.0 * math.cos(angle))), -16384, 16384))
            rope_sin.append(_clamp(int(round(16384.0 * math.sin(angle))), -16384, 16384))

    inv_dim_q16 = int(round(65536.0 / float(cfg.dim)))
    att_scale_q12 = max(1, int(round(4096.0 * 16.0 / math.sqrt(float(cfg.head_dim)))))

    return FixedPointLuts(
        inv_rms_lut=inv_rms_lut,
        silu_lut=silu_lut,
        exp_lut=exp_lut,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        inv_dim_q16=inv_dim_q16,
        att_scale_q12=att_scale_q12,
    )


def _quantize_matrix(mat: Sequence[Sequence[float]], mode: str, group_size: int) -> QuantMatrix:
    rows = len(mat)
    cols = len(mat[0]) if rows else 0
    if cols % group_size != 0:
        raise ValueError(f"Matrix cols ({cols}) must be divisible by group_size ({group_size})")

    q_rows: list[list[int]] = []
    g_scales: list[list[int]] = []

    for row in mat:
        q_row: list[int] = []
        row_gs: list[int] = []
        for start in range(0, cols, group_size):
            grp = row[start : start + group_size]
            if mode == "w4a8":
                max_abs = max((abs(v) for v in grp), default=0.0)
                sc = max_abs / 7.0 if max_abs > 1e-12 else 0.0
                if sc == 0.0:
                    q_row.extend([0] * group_size)
                else:
                    q_row.extend(_clamp(int(round(v / sc)), -7, 7) for v in grp)
            elif mode == "ternary":
                mean_abs = sum(abs(v) for v in grp) / float(group_size)
                sc = mean_abs if mean_abs > 1e-12 else 0.0
                if sc == 0.0:
                    q_row.extend([0] * group_size)
                else:
                    q_row.extend(_clamp(int(round(v / sc)), -1, 1) for v in grp)
            else:
                raise ValueError(f"Unsupported quant_mode: {mode}")
            row_gs.append(_clamp(int(round(sc * W_SCALE_Q15)), 0, 65535))
        q_rows.append(q_row)
        g_scales.append(row_gs)

    return QuantMatrix(
        rows=rows, cols=cols, group_size=group_size, weights=q_rows, group_scales=g_scales
    )


def _quantize_q88_vec(vec: Sequence[float]) -> list[int]:
    return [_clamp(int(round(v * Q88_SCALE)), -32768, 32767) for v in vec]


def quantize_model(
    model: FloatModel,
    quant_mode: str = "w4a8",
    group_size: int = 32,
) -> QuantModel:
    """Quantize a FloatModel into Q8.8 activations + per-group W4A8 or 2-bit ternary weights."""
    if quant_mode not in ("w4a8", "ternary"):
        raise ValueError(f"Unsupported quant_mode '{quant_mode}'; choose 'w4a8' or 'ternary'")
    cfg = model.config
    if cfg.dim % group_size != 0 or cfg.hidden_dim % group_size != 0:
        raise ValueError(
            f"group_size ({group_size}) must divide dim ({cfg.dim}) and hidden_dim ({cfg.hidden_dim})"
        )

    emb_q = [_quantize_q88_vec(row) for row in model.token_embedding]
    q_layers = [
        QuantLayer(
            rms_att=_quantize_q88_vec(lw.rms_att),
            wq=_quantize_matrix(lw.wq, quant_mode, group_size),
            wk=_quantize_matrix(lw.wk, quant_mode, group_size),
            wv=_quantize_matrix(lw.wv, quant_mode, group_size),
            wo=_quantize_matrix(lw.wo, quant_mode, group_size),
            rms_ffn=_quantize_q88_vec(lw.rms_ffn),
            w1=_quantize_matrix(lw.w1, quant_mode, group_size),
            w2=_quantize_matrix(lw.w2, quant_mode, group_size),
            w3=_quantize_matrix(lw.w3, quant_mode, group_size),
        )
        for lw in model.layers
    ]
    rms_final_q = _quantize_q88_vec(model.rms_final)
    wcls_q = _quantize_matrix(model.wcls, quant_mode, group_size)
    luts = build_fixed_point_luts(cfg)
    return QuantModel(
        config=cfg,
        quant_mode=quant_mode,
        group_size=group_size,
        token_embedding=emb_q,
        layers=q_layers,
        rms_final=rms_final_q,
        wcls=wcls_q,
        luts=luts,
    )
