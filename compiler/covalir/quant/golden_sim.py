# Copyright 2026 CovalIR Contributors
# Licensed under the Apache License, Version 2.0
"""Bit-exact pure-integer Transformer simulator matching covalir_llm_core.sv.

Implements Q8.8 (INT16) activations, CLZ-normalized RMSNorm, power-of-2 dynamic INT8
activation quantization, per-group W4A8/2-bit-ternary MACs with Q1.15 group scales,
Q1.14 RoPE, causal MHA with exact restoring division, and Q8.8 SwiGLU.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import mul
from typing import Sequence

from covalir.quant.golden import dot_ternary, dot_w4a8, wrap_signed
from covalir.quant.quantize_model import FixedPointLuts, QuantMatrix, QuantModel


def _clamp(val: int, lo: int, hi: int) -> int:
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


def _trunc_div_signed(num: int, den: int) -> int:
    if den <= 0:
        raise ValueError(f"Denominator must be > 0, got {den}")
    if num >= 0:
        return num // den
    return -((-num) // den)


def dyn_quant_q88_to_i8(vec_q88: Sequence[int]) -> tuple[list[int], int]:
    """Dynamic power-of-2 absmax INT8 quantization (zero divider in hardware)."""
    max_abs = max((abs(v) for v in vec_q88), default=0)
    if max_abs <= 127:
        e = 0
        qx = [_clamp(v, -128, 127) for v in vec_q88]
    else:
        e = max(0, max_abs.bit_length() - 7)
        bias = 1 << (e - 1)
        qx = [_clamp((v + bias) >> e, -128, 127) for v in vec_q88]
    return qx, e


def _rmsnorm_q88(x_q88: Sequence[int], w_q88: Sequence[int], luts: FixedPointLuts) -> list[int]:
    sum_sq = sum(v * v for v in x_q88)
    msq = max(1, (sum_sq * luts.inv_dim_q16) >> 16)
    k = 0
    mv = msq
    while mv >= 256:
        mv >>= 2
        k += 1
    inv_q15 = luts.inv_rms_lut[mv]
    sh = 15 + k
    bias = 1 << (sh - 1)
    out: list[int] = []
    for xi, wi in zip(x_q88, w_q88):
        prod48 = wrap_signed(xi * inv_q15 * wi, 48)
        out.append(_clamp((prod48 + bias) >> sh, -32768, 32767))
    return out


def _linear_group_q88(
    x_q88: Sequence[int],
    mat: QuantMatrix,
    quant_mode: str,
    return_raw48: bool = False,
) -> list[int]:
    qx, e = dyn_quant_q88_to_i8(x_q88)
    gs = mat.group_size
    n_groups = mat.cols // gs
    sh = 15 - e  # e in [0..9] -> sh in [6..15] > 0 always
    bias = 1 << (sh - 1)

    # Fast exact C-level inner loop when neither per-group dot32 nor row acc48 can wrap.
    # Max |dot| per group is gs * 8 * 128; max |acc48| is n_groups * (gs * 1024) * 65535.
    no_wrap = (gs * 1024 < (1 << 31)) and (n_groups * gs * 1024 * 65535 < (1 << 47))
    if no_wrap:
        qx_groups = [qx[g * gs : (g + 1) * gs] for g in range(n_groups)]
        offs = [g * gs for g in range(n_groups)]
        out: list[int] = []
        weights = mat.weights
        scales = mat.group_scales
        for r in range(mat.rows):
            w_row = weights[r]
            s_row = scales[r]
            acc48 = 0
            for g in range(n_groups):
                off = offs[g]
                acc48 += sum(map(mul, w_row[off : off + gs], qx_groups[g])) * s_row[g]
            if return_raw48:
                out.append(acc48)
            else:
                v = (acc48 + bias) >> sh
                out.append(-32768 if v < -32768 else (32767 if v > 32767 else v))
        return out

    dot_fn = dot_ternary if quant_mode == "ternary" else dot_w4a8
    out = []
    for r in range(mat.rows):
        w_row = mat.weights[r]
        s_row = mat.group_scales[r]
        acc48 = 0
        for g in range(n_groups):
            off = g * gs
            dot32 = dot_fn(w_row[off : off + gs], qx[off : off + gs], acc_width=32)
            acc48 = wrap_signed(acc48 + dot32 * s_row[g], 48)
        if return_raw48:
            out.append(acc48)
        else:
            out.append(_clamp((acc48 + bias) >> sh, -32768, 32767))
    return out


def _apply_rope_q88(
    vec: list[int], pos: int, n_heads: int, head_dim: int, luts: FixedPointLuts
) -> list[int]:
    half_hd = head_dim // 2
    out = list(vec)
    base = pos * half_hd
    for h in range(n_heads):
        hoff = h * head_dim
        for j in range(half_hd):
            c = luts.rope_cos[base + j]
            s = luts.rope_sin[base + j]
            a = out[hoff + 2 * j]
            b = out[hoff + 2 * j + 1]
            r0 = _clamp((a * c - b * s + 8192) >> 14, -32768, 32767)
            r1 = _clamp((a * s + b * c + 8192) >> 14, -32768, 32767)
            out[hoff + 2 * j] = r0
            out[hoff + 2 * j + 1] = r1
    return out


@dataclass
class StepTrace:
    token_in: int
    pos: int
    next_token: int
    max_logit: int
    logits: list[int]


class GoldenTransformerSim:
    """Stateful bit-exact autoregressive integer Transformer matching covalir_llm_core.sv."""

    def __init__(self, model: QuantModel) -> None:
        self.model = model
        cfg = model.config
        if cfg.n_heads != cfg.n_kv_heads:
            raise ValueError("Phase 1 core requires n_heads == n_kv_heads")
        self.k_cache: list[list[list[int]]] = [
            [[0] * cfg.dim for _ in range(cfg.max_seq_len)] for _ in range(cfg.n_layers)
        ]
        self.v_cache: list[list[list[int]]] = [
            [[0] * cfg.dim for _ in range(cfg.max_seq_len)] for _ in range(cfg.n_layers)
        ]

    def step(self, token_id: int, pos: int) -> StepTrace:
        cfg = self.model.config
        luts = self.model.luts
        qmode = self.model.quant_mode
        if not (0 <= token_id < cfg.vocab_size):
            raise ValueError(f"token_id {token_id} out of range [0, {cfg.vocab_size})")
        if not (0 <= pos < cfg.max_seq_len):
            raise ValueError(f"pos {pos} out of range [0, {cfg.max_seq_len})")

        # 1. Token embedding lookup (Q8.8 INT16)
        x = list(self.model.token_embedding[token_id])

        # 2. Transformer layers
        for l_idx, layer in enumerate(self.model.layers):
            xb = _rmsnorm_q88(x, layer.rms_att, luts)

            q = _linear_group_q88(xb, layer.wq, qmode)
            k = _linear_group_q88(xb, layer.wk, qmode)
            v = _linear_group_q88(xb, layer.wv, qmode)

            q = _apply_rope_q88(q, pos, cfg.n_heads, cfg.head_dim, luts)
            k = _apply_rope_q88(k, pos, cfg.n_heads, cfg.head_dim, luts)

            self.k_cache[l_idx][pos] = k
            self.v_cache[l_idx][pos] = v

            # Causal multi-head self-attention (Q8.8)
            att_out = [0] * cfg.dim
            for h in range(cfg.n_heads):
                hoff = h * cfg.head_dim
                scores: list[int] = []
                for t in range(pos + 1):
                    dot48 = 0
                    for d in range(cfg.head_dim):
                        dot48 = wrap_signed(
                            dot48 + q[hoff + d] * self.k_cache[l_idx][t][hoff + d], 48
                        )
                    score = (wrap_signed(dot48 * luts.att_scale_q12, 48) + (1 << 27)) >> 28
                    scores.append(wrap_signed(score, 32))

                max_s = max(scores)
                ew: list[int] = []
                sum_ew = 0
                for s in scores:
                    diff = max_s - s
                    idx = 255 if diff > 255 else (0 if diff < 0 else diff)
                    w_val = luts.exp_lut[idx]
                    ew.append(w_val)
                    sum_ew += w_val

                for d in range(cfg.head_dim):
                    wsum40 = 0
                    for t in range(pos + 1):
                        wsum40 = wrap_signed(
                            wsum40 + ew[t] * self.v_cache[l_idx][t][hoff + d], 40
                        )
                    att_out[hoff + d] = _clamp(_trunc_div_signed(wsum40, sum_ew), -32768, 32767)

            # Output projection + saturating Q8.8 residual add
            xb2 = _linear_group_q88(att_out, layer.wo, qmode)
            x = [_clamp(a + b, -32768, 32767) for a, b in zip(x, xb2)]

            # FFN RMSNorm + SwiGLU MLP + residual add
            xb = _rmsnorm_q88(x, layer.rms_ffn, luts)
            hb = _linear_group_q88(xb, layer.w1, qmode)
            hb2 = _linear_group_q88(xb, layer.w3, qmode)
            h_act = [
                _clamp(
                    (luts.silu_lut[_clamp((g >> 4) + 256, 0, 511)] * u + 128) >> 8,
                    -32768,
                    32767,
                )
                for g, u in zip(hb, hb2)
            ]
            xb_ffn = _linear_group_q88(h_act, layer.w2, qmode)
            x = [_clamp(a + b, -32768, 32767) for a, b in zip(x, xb_ffn)]

        # 3. Final RMSNorm + LM Head raw 48-bit logits + deterministic argmax
        x_norm = _rmsnorm_q88(x, self.model.rms_final, luts)
        logits = _linear_group_q88(x_norm, self.model.wcls, qmode, return_raw48=True)

        best_tok = 0
        best_logit = logits[0]
        for tok in range(1, cfg.vocab_size):
            if logits[tok] > best_logit:
                best_logit = logits[tok]
                best_tok = tok

        return StepTrace(
            token_in=token_id,
            pos=pos,
            next_token=best_tok,
            max_logit=best_logit,
            logits=logits,
        )

    def generate(self, prompt_tokens: Sequence[int], gen_steps: int) -> list[StepTrace]:
        """Run prefill on prompt_tokens then autoregressive decode for gen_steps tokens."""
        traces: list[StepTrace] = []
        pos = 0
        last_tok = prompt_tokens[0]
        for tok in prompt_tokens:
            tr = self.step(tok, pos)
            traces.append(tr)
            last_tok = tr.next_token
            pos += 1
        for _ in range(gen_steps):
            tr = self.step(last_tok, pos)
            traces.append(tr)
            last_tok = tr.next_token
            pos += 1
        return traces
