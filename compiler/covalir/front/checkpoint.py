# Copyright 2026 CovalIR Contributors
# Licensed under the Apache License, Version 2.0
"""Zero-dependency checkpoint loaders for Llama-2 / TinyStories models.

Supports:
  1. Karpathy llama2.c `.bin` checkpoints (e.g. `stories15M.bin`)
  2. HuggingFace `.safetensors` checkpoints (F32 / F16 / BF16 decoded in stdlib Python)
  3. Deterministic synthetic Llama-2 checkpoints for fast CI & Verilator regressions
"""

from __future__ import annotations

import array
from dataclasses import dataclass
import json
import math
import pathlib
import random
import struct
from typing import BinaryIO


@dataclass(frozen=True)
class ModelConfig:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    max_seq_len: int

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.head_dim * self.n_kv_heads


@dataclass
class LayerWeights:
    rms_att: list[float]          # [dim]
    wq: list[list[float]]         # [dim][dim]
    wk: list[list[float]]         # [kv_dim][dim]
    wv: list[list[float]]         # [kv_dim][dim]
    wo: list[list[float]]         # [dim][dim]
    rms_ffn: list[float]          # [dim]
    w1: list[list[float]]         # [hidden_dim][dim]
    w2: list[list[float]]         # [dim][hidden_dim]
    w3: list[list[float]]         # [hidden_dim][dim]


@dataclass
class FloatModel:
    config: ModelConfig
    token_embedding: list[list[float]]  # [vocab_size][dim]
    layers: list[LayerWeights]
    rms_final: list[float]              # [dim]
    wcls: list[list[float]]             # [vocab_size][dim]


def _read_f32_flat(f: BinaryIO, count: int) -> list[float]:
    raw = f.read(count * 4)
    if len(raw) != count * 4:
        raise ValueError(f"Unexpected EOF reading {count} float32 values")
    arr = array.array("f")
    arr.frombytes(raw)
    return arr.tolist()


def _reshape_2d(flat: list[float], rows: int, cols: int) -> list[list[float]]:
    if len(flat) != rows * cols:
        raise ValueError(f"Cannot reshape {len(flat)} elements into ({rows}, {cols})")
    return [flat[r * cols : (r + 1) * cols] for r in range(rows)]


def _decode_bf16(raw: bytes) -> list[float]:
    u16s = struct.unpack(f"<{len(raw) // 2}H", raw)
    f32_bytes = bytearray(len(u16s) * 4)
    for i, h in enumerate(u16s):
        f32_bytes[i * 4 + 2] = h & 0xFF
        f32_bytes[i * 4 + 3] = (h >> 8) & 0xFF
    arr = array.array("f")
    arr.frombytes(f32_bytes)
    return arr.tolist()


def _decode_f16(raw: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(raw) // 2}e", raw))


def load_llama2c_bin(path: pathlib.Path) -> FloatModel:
    """Load a Karpathy llama2.c `.bin` checkpoint (e.g. stories15M.bin)."""
    with path.open("rb") as f:
        hdr = struct.unpack("<7i", f.read(28))
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_raw, seq_len = hdr
        shared_weights = vocab_raw > 0
        vocab_size = abs(vocab_raw)
        cfg = ModelConfig(
            dim=dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=vocab_size,
            max_seq_len=seq_len,
        )
        emb = _reshape_2d(_read_f32_flat(f, vocab_size * dim), vocab_size, dim)
        rms_att_all = [_read_f32_flat(f, dim) for _ in range(n_layers)]
        wq_all = [_reshape_2d(_read_f32_flat(f, dim * dim), dim, dim) for _ in range(n_layers)]
        wk_all = [
            _reshape_2d(_read_f32_flat(f, cfg.kv_dim * dim), cfg.kv_dim, dim)
            for _ in range(n_layers)
        ]
        wv_all = [
            _reshape_2d(_read_f32_flat(f, cfg.kv_dim * dim), cfg.kv_dim, dim)
            for _ in range(n_layers)
        ]
        wo_all = [_reshape_2d(_read_f32_flat(f, dim * dim), dim, dim) for _ in range(n_layers)]
        rms_ffn_all = [_read_f32_flat(f, dim) for _ in range(n_layers)]
        w1_all = [
            _reshape_2d(_read_f32_flat(f, hidden_dim * dim), hidden_dim, dim)
            for _ in range(n_layers)
        ]
        w2_all = [
            _reshape_2d(_read_f32_flat(f, dim * hidden_dim), dim, hidden_dim)
            for _ in range(n_layers)
        ]
        w3_all = [
            _reshape_2d(_read_f32_flat(f, hidden_dim * dim), hidden_dim, dim)
            for _ in range(n_layers)
        ]
        rms_final = _read_f32_flat(f, dim)
        # Skip legacy freq_cis_real and freq_cis_imag tables
        f.read(seq_len * (cfg.head_dim // 2) * 4 * 2)
        if shared_weights:
            wcls = emb
        else:
            wcls = _reshape_2d(_read_f32_flat(f, vocab_size * dim), vocab_size, dim)

    layers = [
        LayerWeights(
            rms_att=rms_att_all[i],
            wq=wq_all[i],
            wk=wk_all[i],
            wv=wv_all[i],
            wo=wo_all[i],
            rms_ffn=rms_ffn_all[i],
            w1=w1_all[i],
            w2=w2_all[i],
            w3=w3_all[i],
        )
        for i in range(n_layers)
    ]
    return FloatModel(config=cfg, token_embedding=emb, layers=layers, rms_final=rms_final, wcls=wcls)


def load_safetensors(model_dir: pathlib.Path) -> FloatModel:
    """Load a HuggingFace LlamaForCausalLM directory (`config.json` + `model.safetensors`)."""
    cfg_json = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    dim = int(cfg_json["hidden_size"])
    hidden_dim = int(cfg_json["intermediate_size"])
    n_layers = int(cfg_json["num_hidden_layers"])
    n_heads = int(cfg_json["num_attention_heads"])
    n_kv_heads = int(cfg_json.get("num_key_value_heads", n_heads))
    vocab_size = int(cfg_json["vocab_size"])
    max_seq_len = int(cfg_json.get("max_position_embeddings", 256))
    cfg = ModelConfig(
        dim=dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )

    st_path = model_dir / "model.safetensors"
    with st_path.open("rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hdr_len).decode("utf-8"))
        data_start = 8 + hdr_len

        def get_tensor(name: str) -> list[float]:
            meta = header[name]
            dtype = meta["dtype"]
            start, end = meta["data_offsets"]
            f.seek(data_start + start)
            raw = f.read(end - start)
            if dtype == "F32":
                arr = array.array("f")
                arr.frombytes(raw)
                return arr.tolist()
            if dtype == "F16":
                return _decode_f16(raw)
            if dtype == "BF16":
                return _decode_bf16(raw)
            raise ValueError(f"Unsupported safetensors dtype {dtype} for {name}")

        emb = _reshape_2d(get_tensor("model.embed_tokens.weight"), vocab_size, dim)
        layers: list[LayerWeights] = []
        for i in range(n_layers):
            p = f"model.layers.{i}"
            layers.append(
                LayerWeights(
                    rms_att=get_tensor(f"{p}.input_layernorm.weight"),
                    wq=_reshape_2d(get_tensor(f"{p}.self_attn.q_proj.weight"), dim, dim),
                    wk=_reshape_2d(get_tensor(f"{p}.self_attn.k_proj.weight"), cfg.kv_dim, dim),
                    wv=_reshape_2d(get_tensor(f"{p}.self_attn.v_proj.weight"), cfg.kv_dim, dim),
                    wo=_reshape_2d(get_tensor(f"{p}.self_attn.o_proj.weight"), dim, dim),
                    rms_ffn=get_tensor(f"{p}.post_attention_layernorm.weight"),
                    w1=_reshape_2d(get_tensor(f"{p}.mlp.gate_proj.weight"), hidden_dim, dim),
                    w2=_reshape_2d(get_tensor(f"{p}.mlp.down_proj.weight"), dim, hidden_dim),
                    w3=_reshape_2d(get_tensor(f"{p}.mlp.up_proj.weight"), hidden_dim, dim),
                )
            )
        rms_final = get_tensor("model.norm.weight")
        if "lm_head.weight" in header:
            wcls = _reshape_2d(get_tensor("lm_head.weight"), vocab_size, dim)
        else:
            wcls = emb

    return FloatModel(config=cfg, token_embedding=emb, layers=layers, rms_final=rms_final, wcls=wcls)


def make_synthetic_model(
    dim: int = 32,
    hidden_dim: int = 64,
    n_layers: int = 2,
    n_heads: int = 2,
    vocab_size: int = 64,
    max_seq_len: int = 32,
    seed: int = 0xC0FA1101,
) -> FloatModel:
    """Create a deterministic Llama-2 float model for fast end-to-end verification."""
    rng = random.Random(seed)
    cfg = ModelConfig(
        dim=dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )

    def mat(r: int, c: int) -> list[list[float]]:
        std = 1.0 / math.sqrt(c)
        return [[rng.gauss(0.0, std) for _ in range(c)] for _ in range(r)]

    def vec_ones(n: int) -> list[float]:
        return [1.0 + rng.uniform(-0.15, 0.15) for _ in range(n)]

    emb = [[rng.gauss(0.0, 0.8) for _ in range(dim)] for _ in range(vocab_size)]
    layers = [
        LayerWeights(
            rms_att=vec_ones(dim),
            wq=mat(dim, dim),
            wk=mat(cfg.kv_dim, dim),
            wv=mat(cfg.kv_dim, dim),
            wo=mat(dim, dim),
            rms_ffn=vec_ones(dim),
            w1=mat(hidden_dim, dim),
            w2=mat(dim, hidden_dim),
            w3=mat(hidden_dim, dim),
        )
        for _ in range(n_layers)
    ]
    rms_final = vec_ones(dim)
    wcls = mat(vocab_size, dim)
    return FloatModel(config=cfg, token_embedding=emb, layers=layers, rms_final=rms_final, wcls=wcls)


def save_llama2c_bin(model: FloatModel, path: pathlib.Path) -> None:
    """Serialize a FloatModel to standard Karpathy llama2.c `.bin` format."""
    cfg = model.config
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        # Negative vocab_size signals untied lm_head weights
        f.write(
            struct.pack(
                "<7i",
                cfg.dim,
                cfg.hidden_dim,
                cfg.n_layers,
                cfg.n_heads,
                cfg.n_kv_heads,
                -cfg.vocab_size,
                cfg.max_seq_len,
            )
        )

        def write_flat(vals: list[float]) -> None:
            arr = array.array("f", vals)
            f.write(arr.tobytes())

        def write_2d(m: list[list[float]]) -> None:
            flat = [v for row in m for v in row]
            write_flat(flat)

        write_2d(model.token_embedding)
        for lw in model.layers:
            write_flat(lw.rms_att)
        for lw in model.layers:
            write_2d(lw.wq)
        for lw in model.layers:
            write_2d(lw.wk)
        for lw in model.layers:
            write_2d(lw.wv)
        for lw in model.layers:
            write_2d(lw.wo)
        for lw in model.layers:
            write_flat(lw.rms_ffn)
        for lw in model.layers:
            write_2d(lw.w1)
        for lw in model.layers:
            write_2d(lw.w2)
        for lw in model.layers:
            write_2d(lw.w3)
        write_flat(model.rms_final)
        # Legacy freq_cis_real / freq_cis_imag placeholders
        write_flat([0.0] * (cfg.max_seq_len * (cfg.head_dim // 2) * 2))
        write_2d(model.wcls)
