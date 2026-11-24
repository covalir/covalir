# Copyright 2026 CovalIR Contributors
# Licensed under the Apache License, Version 2.0
"""CLI entrypoint for CovalIR Phase 1 compiler, golden simulator, and Verilator runner."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

from covalir import __version__
from covalir.front.checkpoint import load_llama2c_bin, load_safetensors, make_synthetic_model
from covalir.front.tokenizer import Llama2Tokenizer
from covalir.gen.emit_sv import emit_compiled_model
from covalir.quant.golden import export_mac_vectors, export_requant_vectors
from covalir.quant.quantize_model import quantize_model


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _run_verilator_sim(dist_dir: pathlib.Path) -> int:
    if shutil.which("verilator") is None:
        print("ERROR: 'verilator' not found on PATH.", file=sys.stderr)
        return 1
    root = _repo_root()
    dist_dir = dist_dir.resolve()
    mdir = dist_dir / "obj_dir"
    rtl_files = [
        root / "hw/ip/prim/rtl/prim_fifo_sync.sv",
        root / "hw/ip/mac_w4a8/rtl/mac_w4a8.sv",
        root / "hw/ip/mac_ternary/rtl/mac_ternary.sv",
        root / "hw/ip/transformer_core/rtl/covalir_llm_core.sv",
        dist_dir / "covalir_top.sv",
    ]
    cmd = [
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
        *[str(p) for p in rtl_files],
        str(dist_dir / "tb_covalir_top.cpp"),
    ]
    subprocess.run(cmd, cwd=dist_dir, check=True)
    subprocess.run(
        [str(mdir / "sim_covalir_top"), "--trace", str(dist_dir / "golden_trace.txt")],
        cwd=dist_dir,
        check=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="covalir",
        description="CovalIR — Open-Source PyTorch/Llama2-to-SystemVerilog FPGA Compiler",
    )
    parser.add_argument("--version", action="version", version=f"covalir {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("info", help="Print currently implemented RTL IPs and compiler flows")

    gv = sub.add_parser(
        "gen-vectors",
        help="Generate bit-exact golden test vectors for Verilator differential DV",
    )
    gv.add_argument("--mode", choices=["w4a8", "ternary", "requant"], required=True)
    gv.add_argument("--lanes", type=int, default=16)
    gv.add_argument("--acc-width", type=int, default=32)
    gv.add_argument("--frames", type=int, default=500)
    gv.add_argument("--seed", type=int, default=0xC0FA1101)
    gv.add_argument("--out", type=pathlib.Path, required=True)

    comp = sub.add_parser(
        "compile",
        help="Compile a Llama-2 / TinyStories checkpoint into verified SystemVerilog + hex weights",
    )
    comp.add_argument(
        "checkpoint",
        nargs="?",
        default="synthetic",
        help="Path to llama2.c `.bin`, HuggingFace SafeTensors dir, or `synthetic` micro-model",
    )
    comp.add_argument("--quant", choices=["w4a8", "ternary"], default="w4a8")
    comp.add_argument("--lanes", type=int, default=16)
    comp.add_argument("--group-size", type=int, default=32)
    comp.add_argument("--tokenizer", type=pathlib.Path, default=None, help="Optional llama2.c tokenizer.bin")
    comp.add_argument("--prompt", default="1,2,3", help="Comma-separated prompt token IDs")
    comp.add_argument("--gen-steps", type=int, default=4, help="Autoregressive decode steps")
    comp.add_argument("--out", type=pathlib.Path, required=True)
    comp.add_argument("--sim", action="store_true", help="Immediately verify bit-exact in Verilator")

    sim_cmd = sub.add_parser(
        "sim",
        help="Run bit-exact Verilator simulation on a compiled CovalIR output directory",
    )
    sim_cmd.add_argument("dist_dir", type=pathlib.Path)

    args = parser.parse_args(argv)

    if args.cmd == "info":
        print(f"CovalIR {__version__} (Phase 1 — End-to-End Integer Transformer Compiler)")
        print("Verified RTL IPs (Verilator strict lint + golden.py differential DV):")
        print("  - prim_fifo_sync       : Low-latency sync FIFO (SymbiYosys formal prove + cover)")
        print("  - prim_generic_sram_2p : Behavioral 1R1W SRAM (byte mask, rvalid_o, read-first)")
        print("  - mac_w4a8             : Pipelined signed INT4 x INT8 -> signed AccWidth MAC")
        print("  - mac_ternary          : Pipelined 2-bit ternary ({-1,0,+1}) x INT8 zero-mult MAC")
        print("  - requant_group        : Pipelined INT32 -> INT8 fixed-point scaler & saturator")
        print("  - covalir_llm_core     : Full autoregressive Llama-2 / TinyStories Transformer core")
        print("Compiler flows: llama2.c `.bin` / SafeTensors / synthetic -> W4A8 or 2-bit ternary SV")
        print("See STATUS.md for local vs. hardware bring-up status.")
        return 0

    if args.cmd == "gen-vectors":
        if args.mode == "requant":
            n_items = export_requant_vectors(args.out, num_random=args.frames, seed=args.seed)
            print(f"Generated {n_items} requant vectors -> {args.out}")
            return 0
        n_frames, n_beats = export_mac_vectors(
            mode=args.mode,
            out_path=args.out,
            lanes=args.lanes,
            acc_width=args.acc_width,
            num_random_frames=args.frames,
            seed=args.seed,
        )
        print(f"Generated {n_frames} frames ({n_beats} beats) -> {args.out}")
        return 0

    if args.cmd == "compile":
        if args.checkpoint == "synthetic":
            fmodel = make_synthetic_model()
        else:
            ckpt = pathlib.Path(args.checkpoint)
            if ckpt.is_dir():
                fmodel = load_safetensors(ckpt)
            else:
                fmodel = load_llama2c_bin(ckpt)
        qmodel = quantize_model(fmodel, quant_mode=args.quant, group_size=args.group_size)
        prompt_toks = [int(x.strip()) for x in args.prompt.split(",") if x.strip()]
        manifest = emit_compiled_model(
            qmodel,
            out_dir=args.out,
            lanes=args.lanes,
            prompt_tokens=prompt_toks,
            gen_steps=args.gen_steps,
        )
        print(
            f"Compiled {args.checkpoint} ({args.quant}, lanes={args.lanes}, group={args.group_size}) -> "
            f"{args.out} [beats={manifest['total_weight_beats']}, tokens={manifest['generated_tokens']}]"
        )
        if args.tokenizer is not None:
            tok = Llama2Tokenizer(args.tokenizer, fmodel.config.vocab_size)
            print(f"Decoded text: {tok.decode(manifest['generated_tokens'])!r}")
        if args.sim:
            return _run_verilator_sim(args.out)
        return 0

    if args.cmd == "sim":
        return _run_verilator_sim(args.dist_dir)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
