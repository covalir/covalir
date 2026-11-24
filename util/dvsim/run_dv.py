#!/usr/bin/env python3
# Copyright 2026 CovalIR Contributors
# Licensed under the Apache License, Version 2.0
"""CovalIR DV regression runner: strict Verilator lint, negative parameter guard tests,
golden.py differential sims for all IPs, and end-to-end compiled Transformer bit-exact proofs."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "compiler"))

from covalir.front.checkpoint import make_synthetic_model  # noqa: E402
from covalir.gen.emit_sv import emit_compiled_model  # noqa: E402
from covalir.quant.golden import export_mac_vectors, export_requant_vectors  # noqa: E402
from covalir.quant.quantize_model import quantize_model  # noqa: E402

FIFO_SV = "hw/ip/prim/rtl/prim_fifo_sync.sv"
SRAM_SV = "hw/ip/prim_generic/rtl/prim_generic_sram_2p.sv"
MAC_W4A8_SV = "hw/ip/mac_w4a8/rtl/mac_w4a8.sv"
MAC_TERN_SV = "hw/ip/mac_ternary/rtl/mac_ternary.sv"
REQ_SV = "hw/ip/requant_group/rtl/requant_group.sv"
CORE_SV = "hw/ip/transformer_core/rtl/covalir_llm_core.sv"

LINT_TARGETS = [
    ("prim_fifo_sync", [FIFO_SV]),
    ("prim_generic_sram_2p", [SRAM_SV]),
    ("mac_w4a8", [FIFO_SV, MAC_W4A8_SV]),
    ("mac_ternary", [FIFO_SV, MAC_TERN_SV]),
    ("requant_group", [FIFO_SV, REQ_SV]),
    ("covalir_llm_core", [FIFO_SV, MAC_W4A8_SV, MAC_TERN_SV, CORE_SV]),
]

NEGATIVE_GUARD_CASES = [
    ("prim_fifo_sync_depth1", "prim_fifo_sync", [FIFO_SV], ["-GDepth=1"]),
    ("prim_fifo_sync_depth3", "prim_fifo_sync", [FIFO_SV], ["-GDepth=3"]),
    ("prim_generic_sram_2p_depth1", "prim_generic_sram_2p", [SRAM_SV], ["-GDepth=1"]),
    ("prim_generic_sram_2p_depth1000", "prim_generic_sram_2p", [SRAM_SV], ["-GDepth=1000"]),
    ("prim_generic_sram_2p_width12", "prim_generic_sram_2p", [SRAM_SV], ["-GWidth=12"]),
    ("mac_w4a8_acc8", "mac_w4a8", [FIFO_SV, MAC_W4A8_SV], ["-GAccWidth=8"]),
    ("mac_ternary_acc8", "mac_ternary", [FIFO_SV, MAC_TERN_SV], ["-GAccWidth=8"]),
    ("requant_group_acc4", "requant_group", [FIFO_SV, REQ_SV], ["-GAccWidth=4"]),
]


def run(cmd: list[str], cwd: pathlib.Path) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def run_negative_guard(name: str, top: str, svs: list[str], params: list[str]) -> None:
    cmd = [
        "verilator",
        "--lint-only",
        "-Wall",
        "--top-module",
        top,
        *params,
        *[str(REPO_ROOT / s) for s in svs],
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode == 0:
        raise RuntimeError(f"Negative guard FAILED: {name} unexpectedly succeeded!")
    combined = res.stdout + res.stderr
    if "%Error" not in combined or "Internal Error" in combined:
        raise RuntimeError(
            f"Negative guard FAILED for {name}: expected clean elaboration $error, got:\n{combined}"
        )
    print(f"PASS [guard:{name}] (rejected cleanly at elaboration)")


def main() -> int:
    if shutil.which("verilator") is None:
        print("ERROR: 'verilator' not found on PATH.", file=sys.stderr)
        return 1

    build_dir = REPO_ROOT / "build" / "dv"
    vec_dir = build_dir / "vectors"
    build_dir.mkdir(parents=True, exist_ok=True)
    vec_dir.mkdir(parents=True, exist_ok=True)

    print("=== Stage 1: Verilator Strict Lint (-Wall -Werror) ===")
    for top, svs in LINT_TARGETS:
        run(
            [
                "verilator",
                "--lint-only",
                "-Wall",
                "-Werror-WIDTH",
                "-Werror-UNUSED",
                "-Werror-UNOPTFLAT",
                "--top-module",
                top,
                *[str(REPO_ROOT / s) for s in svs],
            ],
            cwd=REPO_ROOT,
        )
        print(f"PASS [lint:{top}]")

    print("\n=== Stage 2: Elaboration Parameter Guard Tests ===")
    for name, top, svs, params in NEGATIVE_GUARD_CASES:
        run_negative_guard(name, top, svs, params)

    print("\n=== Stage 3: Primitive Differential Simulations (FIFO & SRAM) ===")
    for depth in (2, 4, 16):
        mdir = build_dir / f"obj_fifo_d{depth}"
        bin_name = f"sim_fifo_d{depth}"
        run(
            [
                "verilator",
                "--cc",
                "--exe",
                "--build",
                "-j",
                "0",
                "-Wall",
                "-Werror-WIDTH",
                "-Werror-UNUSED",
                "-O3",
                f"-GDepth={depth}",
                "-CFLAGS",
                f"-DTB_DEPTH={depth}",
                "--Mdir",
                str(mdir),
                "--top-module",
                "prim_fifo_sync",
                "-o",
                bin_name,
                str(REPO_ROOT / FIFO_SV),
                str(REPO_ROOT / "hw/ip/prim/dv/tb_prim_fifo_sync.cpp"),
            ],
            cwd=REPO_ROOT,
        )
        run([str(mdir / bin_name)], cwd=REPO_ROOT)

    for depth in (16, 64):
        mdir = build_dir / f"obj_sram_d{depth}"
        bin_name = f"sim_sram_d{depth}"
        run(
            [
                "verilator",
                "--cc",
                "--exe",
                "--build",
                "-j",
                "0",
                "-Wall",
                "-Werror-WIDTH",
                "-Werror-UNUSED",
                "-O3",
                f"-GDepth={depth}",
                "-CFLAGS",
                f"-DTB_DEPTH={depth}",
                "--Mdir",
                str(mdir),
                "--top-module",
                "prim_generic_sram_2p",
                "-o",
                bin_name,
                str(REPO_ROOT / SRAM_SV),
                str(REPO_ROOT / "hw/ip/prim_generic/dv/tb_prim_generic_sram_2p.cpp"),
            ],
            cwd=REPO_ROOT,
        )
        run([str(mdir / bin_name)], cwd=REPO_ROOT)

    print("\n=== Stage 4: MAC & Requant Differential Simulations (golden.py Vectors) ===")
    mac_sweeps = [
        ("w4a8", "mac_w4a8", MAC_W4A8_SV, "hw/ip/mac_w4a8/dv/tb_mac_w4a8.cpp", 4, 16, 400),
        ("w4a8", "mac_w4a8", MAC_W4A8_SV, "hw/ip/mac_w4a8/dv/tb_mac_w4a8.cpp", 16, 32, 1000),
        ("ternary", "mac_ternary", MAC_TERN_SV, "hw/ip/mac_ternary/dv/tb_mac_ternary.cpp", 8, 16, 400),
        ("ternary", "mac_ternary", MAC_TERN_SV, "hw/ip/mac_ternary/dv/tb_mac_ternary.cpp", 32, 32, 1000),
    ]
    for mode, top, sv, tb, lanes, acc_w, n_frames in mac_sweeps:
        vec_file = vec_dir / f"{top}_L{lanes}_A{acc_w}.vec"
        nf, nb = export_mac_vectors(
            mode=mode,
            out_path=vec_file,
            lanes=lanes,
            acc_width=acc_w,
            num_random_frames=n_frames,
            seed=0xC0FA1101 + lanes * 13 + acc_w,
        )
        print(f"Generated {nf} golden.py frames ({nb} beats) -> {vec_file.relative_to(REPO_ROOT)}")
        mdir = build_dir / f"obj_{top}_L{lanes}_A{acc_w}"
        bin_name = f"sim_{top}_L{lanes}_A{acc_w}"
        run(
            [
                "verilator",
                "--cc",
                "--exe",
                "--build",
                "-j",
                "0",
                "-Wall",
                "-Werror-WIDTH",
                "-Werror-UNUSED",
                "-O3",
                f"-GLanes={lanes}",
                f"-GAccWidth={acc_w}",
                "-CFLAGS",
                f"-DTB_LANES={lanes} -DTB_ACC_WIDTH={acc_w}",
                "--Mdir",
                str(mdir),
                "--top-module",
                top,
                "-o",
                bin_name,
                str(REPO_ROOT / FIFO_SV),
                str(REPO_ROOT / sv),
                str(REPO_ROOT / tb),
            ],
            cwd=REPO_ROOT,
        )
        run([str(mdir / bin_name), "--vectors", str(vec_file)], cwd=REPO_ROOT)

    req_vec = vec_dir / "requant_group.vec"
    n_req = export_requant_vectors(req_vec, num_random=1000)
    print(f"Generated {n_req} golden.py requant vectors -> {req_vec.relative_to(REPO_ROOT)}")
    req_mdir = build_dir / "obj_requant_group"
    run(
        [
            "verilator",
            "--cc",
            "--exe",
            "--build",
            "-j",
            "0",
            "-Wall",
            "-Werror-WIDTH",
            "-Werror-UNUSED",
            "-O3",
            "--Mdir",
            str(req_mdir),
            "--top-module",
            "requant_group",
            "-o",
            "sim_requant_group",
            str(REPO_ROOT / FIFO_SV),
            str(REPO_ROOT / REQ_SV),
            str(REPO_ROOT / "hw/ip/requant_group/dv/tb_requant_group.cpp"),
        ],
        cwd=REPO_ROOT,
    )
    run([str(req_mdir / "sim_requant_group"), "--vectors", str(req_vec)], cwd=REPO_ROOT)

    print("\n=== Stage 5: End-to-End Compiled Transformer Bit-Exact Proof (W4A8 & Ternary) ===")
    fmodel = make_synthetic_model(dim=32, hidden_dim=64, n_layers=2, n_heads=2, vocab_size=64, max_seq_len=16)
    for qmode in ("w4a8", "ternary"):
        dist_dir = build_dir / f"e2e_{qmode}"
        qmodel = quantize_model(fmodel, quant_mode=qmode)
        emit_compiled_model(qmodel, out_dir=dist_dir, lanes=16, prompt_tokens=(1, 7, 13), gen_steps=5)
        mdir = dist_dir / "obj_dir"
        run(
            [
                "verilator",
                "--cc",
                "--exe",
                "--build",
                "-j",
                "0",
                "-Wall",
                "-Werror-WIDTH",
                "-Werror-UNUSED",
                "-O3",
                "--Mdir",
                str(mdir),
                "--top-module",
                "covalir_top",
                "-o",
                "sim_covalir_top",
                str(REPO_ROOT / FIFO_SV),
                str(REPO_ROOT / MAC_W4A8_SV),
                str(REPO_ROOT / MAC_TERN_SV),
                str(REPO_ROOT / CORE_SV),
                str(dist_dir / "covalir_top.sv"),
                str(dist_dir / "tb_covalir_top.cpp"),
            ],
            cwd=dist_dir,
        )
        run(
            [str(mdir / "sim_covalir_top"), "--trace", str(dist_dir / "golden_trace.txt")],
            cwd=dist_dir,
        )

    print("\nALL RTL LINT, GUARDS, IP DV, AND END-TO-END TRANSFORMER PROOFS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
