# Copyright 2026 CovalIR Contributors
# Licensed under the Apache License, Version 2.0
"""Bit-exact Golden Reference arithmetic and test-vector generator for CovalIR RTL."""

from __future__ import annotations

import pathlib
import random
from typing import Sequence


def wrap_signed(val: int, width: int = 32) -> int:
    """Wrap an arbitrary Python integer to two's-complement signed `width`-bit range."""
    if width < 2:
        raise ValueError(f"Signed width must be >= 2, got {width}")
    mask = (1 << width) - 1
    sign_bit = 1 << (width - 1)
    wrapped = val & mask
    return wrapped - (1 << width) if (wrapped & sign_bit) else wrapped


def _wrap_int32(val: int) -> int:
    return wrap_signed(val, 32)


def dot_w4a8(weights: Sequence[int], activations: Sequence[int], acc_width: int = 32) -> int:
    """Bit-exact reference for hw/ip/mac_w4a8/rtl/mac_w4a8.sv."""
    if len(weights) != len(activations):
        raise ValueError("weights and activations must have identical length")
    acc = 0
    for w, a in zip(weights, activations):
        if not (-8 <= w <= 7):
            raise ValueError(f"INT4 weight out of range [-8, 7]: {w}")
        if not (-128 <= a <= 127):
            raise ValueError(f"INT8 activation out of range [-128, 127]: {a}")
        acc = wrap_signed(acc + w * a, acc_width)
    return acc


def dot_ternary(weights: Sequence[int], activations: Sequence[int], acc_width: int = 32) -> int:
    """Bit-exact reference for hw/ip/mac_ternary/rtl/mac_ternary.sv (2-bit packed ternary)."""
    if len(weights) != len(activations):
        raise ValueError("weights and activations must have identical length")
    acc = 0
    for w, a in zip(weights, activations):
        if w not in (-1, 0, 1):
            raise ValueError(f"Ternary weight must be in {{-1, 0, +1}}: {w}")
        if not (-128 <= a <= 127):
            raise ValueError(f"INT8 activation out of range [-128, 127]: {a}")
        if w == 1:
            acc = wrap_signed(acc + a, acc_width)
        elif w == -1:
            acc = wrap_signed(acc - a, acc_width)
    return acc


def frame_dot_w4a8(
    beats_w: Sequence[Sequence[int]],
    beats_a: Sequence[Sequence[int]],
    acc_width: int = 32,
) -> int:
    """Multi-beat frame accumulation matching mac_w4a8 RTL (modular signed arithmetic)."""
    if len(beats_w) != len(beats_a) or len(beats_w) == 0:
        raise ValueError("beats_w and beats_a must be non-empty and have identical length")
    acc = 0
    for ws, acts in zip(beats_w, beats_a):
        acc = wrap_signed(acc + dot_w4a8(ws, acts, acc_width=acc_width), acc_width)
    return acc


def frame_dot_ternary(
    beats_w: Sequence[Sequence[int]],
    beats_a: Sequence[Sequence[int]],
    acc_width: int = 32,
) -> int:
    """Multi-beat frame accumulation matching mac_ternary RTL (modular signed arithmetic)."""
    if len(beats_w) != len(beats_a) or len(beats_w) == 0:
        raise ValueError("beats_w and beats_a must be non-empty and have identical length")
    acc = 0
    for ws, acts in zip(beats_w, beats_a):
        acc = wrap_signed(acc + dot_ternary(ws, acts, acc_width=acc_width), acc_width)
    return acc


def _pack_w4(ws: Sequence[int]) -> int:
    val = 0
    for k, w in enumerate(ws):
        val |= (w & 0xF) << (k * 4)
    return val


def _pack_trit2(ws: Sequence[int]) -> int:
    val = 0
    for k, w in enumerate(ws):
        code = 1 if w > 0 else (3 if w < 0 else 0)
        val |= code << (k * 2)
    return val


def _pack_a8(acts: Sequence[int]) -> int:
    val = 0
    for k, a in enumerate(acts):
        val |= (a & 0xFF) << (k * 8)
    return val


def export_mac_vectors(
    mode: str,
    out_path: pathlib.Path,
    lanes: int,
    acc_width: int = 32,
    num_random_frames: int = 500,
    seed: int = 0xC0FA1101,
) -> tuple[int, int]:
    """Generate deterministic golden test vectors consumed directly by Verilator C++ TBs.

    File format (ASCII lines):
      Header: `LANES <lanes> ACC_WIDTH <acc_width> FRAMES <n_frames>`
      Per beat: `B <last:0|1> <w_hex> <a_hex>`
      Per frame end: `E <expected_signed_int>`
    """
    if mode not in ("w4a8", "ternary"):
        raise ValueError(f"Unsupported mode: {mode}")

    rng = random.Random(seed)
    frames: list[tuple[list[list[int]], list[list[int]], int]] = []

    def add_frame(bw: list[list[int]], ba: list[list[int]]) -> None:
        exp = (
            frame_dot_w4a8(bw, ba, acc_width=acc_width)
            if mode == "w4a8"
            else frame_dot_ternary(bw, ba, acc_width=acc_width)
        )
        frames.append((bw, ba, exp))

    if mode == "w4a8":
        # Corner cases
        add_frame([[-8] * lanes], [[-128] * lanes])
        add_frame([[7] * lanes], [[127] * lanes])
        add_frame([[0] * lanes], [[0] * lanes])
        add_frame([[-8] * lanes], [[127] * lanes])
        # Explicit accumulator wrap-around tests (positive overflow & negative underflow)
        if acc_width <= 20:
            per_beat = 1024 * lanes
            n_wrap = max(2, ((1 << (acc_width - 1)) // per_beat) + 2)
            add_frame([[-8] * lanes for _ in range(n_wrap)], [[-128] * lanes for _ in range(n_wrap)])
            add_frame([[-8] * lanes for _ in range(n_wrap)], [[127] * lanes for _ in range(n_wrap)])
        # Randomized multi-beat frames
        for _ in range(num_random_frames):
            nb = rng.randint(1, 8)
            bw = [[rng.randint(-8, 7) for _ in range(lanes)] for _ in range(nb)]
            ba = [[rng.randint(-128, 127) for _ in range(lanes)] for _ in range(nb)]
            add_frame(bw, ba)
    else:
        # Critical ternary corner cases including exact -(-128) = +128
        add_frame([[-1] * lanes], [[-128] * lanes])
        add_frame([[1] * lanes], [[-128] * lanes])
        add_frame([[0] * lanes], [[127] * lanes])
        if acc_width <= 18:
            per_beat = 128 * lanes
            n_wrap = max(2, ((1 << (acc_width - 1)) // per_beat) + 2)
            add_frame([[-1] * lanes for _ in range(n_wrap)], [[-128] * lanes for _ in range(n_wrap)])
            add_frame([[1] * lanes for _ in range(n_wrap)], [[-128] * lanes for _ in range(n_wrap)])
        for _ in range(num_random_frames):
            nb = rng.randint(1, 8)
            bw = [[rng.choice((-1, 0, 1)) for _ in range(lanes)] for _ in range(nb)]
            ba = [[rng.randint(-128, 127) for _ in range(lanes)] for _ in range(nb)]
            add_frame(bw, ba)

    w_bits = lanes * (4 if mode == "w4a8" else 2)
    a_bits = lanes * 8
    w_hex_len = (w_bits + 3) // 4
    a_hex_len = (a_bits + 3) // 4

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_beats = 0
    lines = [f"LANES {lanes} ACC_WIDTH {acc_width} FRAMES {len(frames)}"]
    for bw, ba, exp in frames:
        for idx, (ws, acts) in enumerate(zip(bw, ba)):
            last = 1 if idx + 1 == len(bw) else 0
            wp = _pack_w4(ws) if mode == "w4a8" else _pack_trit2(ws)
            ap = _pack_a8(acts)
            lines.append(f"B {last} {wp:0{w_hex_len}x} {ap:0{a_hex_len}x}")
            total_beats += 1
        lines.append(f"E {exp}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(frames), total_beats


def requant_int32_to_int8(acc: int, scale: int, rshift: int) -> int:
    """Bit-exact reference for hw/ip/requant_group/rtl/requant_group.sv."""
    if not (-0x80000000 <= acc <= 0x7FFFFFFF):
        raise ValueError(f"INT32 accumulator out of range: {acc}")
    if not (0 <= scale <= 0xFFFF):
        raise ValueError(f"UINT16 scale out of range [0, 65535]: {scale}")
    if not (0 <= rshift <= 31):
        raise ValueError(f"rshift out of range [0, 31]: {rshift}")
    prod = acc * scale
    if rshift == 0:
        rounded = prod
    else:
        rounded = (prod + (1 << (rshift - 1))) >> rshift
    if rounded > 127:
        return 127
    if rounded < -128:
        return -128
    return rounded


def export_requant_vectors(
    out_path: pathlib.Path,
    num_random: int = 1000,
    seed: int = 0xAE0A117,
) -> int:
    """Generate deterministic golden vectors for hw/ip/requant_group/dv/tb_requant_group.cpp."""
    rng = random.Random(seed)
    cases: list[tuple[int, int, int]] = [
        (0, 0, 0),
        (0, 65535, 31),
        (127, 1, 0),
        (-128, 1, 0),
        (128, 1, 0),
        (-129, 1, 0),
        (0x7FFFFFFF, 65535, 31),
        (-0x80000000, 65535, 31),
        (0x7FFFFFFF, 65535, 0),
        (-0x80000000, 65535, 0),
        (1, 1, 1),   # exact +0.5 -> +1
        (-1, 1, 1),  # exact -0.5 -> 0 (round half toward +inf)
        (-3, 1, 1),  # exact -1.5 -> -1
        (3, 1, 1),   # exact +1.5 -> +2
    ]
    for _ in range(num_random):
        rshift = rng.randint(0, 31)
        scale = rng.randint(0, 65535)
        # Mix full-range and non-saturating in-band values
        if rng.random() < 0.65:
            mag = max(1, (1 << max(0, rshift - 4)))
            acc = rng.randint(-mag, mag)
            acc = max(-0x80000000, min(0x7FFFFFFF, acc))
        else:
            acc = rng.randint(-0x80000000, 0x7FFFFFFF)
        cases.append((acc, scale, rshift))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# acc scale rshift expected_int8"]
    for acc, scale, rshift in cases:
        exp = requant_int32_to_int8(acc, scale, rshift)
        lines.append(f"{acc} {scale} {rshift} {exp}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(cases)
