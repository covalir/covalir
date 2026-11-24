# Copyright 2026 CovalIR Contributors
# Licensed under the Apache License, Version 2.0
"""Unit tests for CovalIR golden math, checkpoint loaders, quantizer, simulator & compiler CLI."""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from covalir.cli import main as cli_main
from covalir.front.checkpoint import load_llama2c_bin, make_synthetic_model, save_llama2c_bin
from covalir.gen.emit_sv import emit_compiled_model
from covalir.quant.golden import (
    dot_ternary,
    dot_w4a8,
    export_mac_vectors,
    export_requant_vectors,
    frame_dot_ternary,
    frame_dot_w4a8,
    requant_int32_to_int8,
    wrap_signed,
)
from covalir.quant.golden_sim import GoldenTransformerSim
from covalir.quant.quantize_model import quantize_model


class TestGoldenArithmetic(unittest.TestCase):
    def test_w4a8_extremes(self) -> None:
        self.assertEqual(dot_w4a8([-8] * 16, [-128] * 16), 16 * 1024)
        self.assertEqual(dot_w4a8([7] * 16, [127] * 16), 16 * 889)
        self.assertEqual(dot_w4a8([-8] * 16, [127] * 16), -16 * 1016)
        self.assertEqual(dot_w4a8([0] * 16, [127] * 16), 0)

    def test_ternary_neg_128_exact(self) -> None:
        self.assertEqual(dot_ternary([-1] * 32, [-128] * 32), 4096)
        self.assertEqual(dot_ternary([1] * 32, [-128] * 32), -4096)
        self.assertEqual(dot_ternary([0] * 32, [127] * 32), 0)

    def test_signed_wrap_overflow_and_underflow(self) -> None:
        self.assertEqual(wrap_signed(0x7FFFFFFF + 1, 32), -0x80000000)
        self.assertEqual(wrap_signed(-0x80000000 - 1, 32), 0x7FFFFFFF)
        self.assertEqual(dot_w4a8([-8] * 32, [-128] * 32, acc_width=16), -32768)
        self.assertEqual(
            frame_dot_w4a8([[-8] * 16, [-8] * 16], [[-128] * 16, [-128] * 16], acc_width=16),
            -32768,
        )
        self.assertEqual(
            frame_dot_ternary([[-1] * 1024, [1] * 1024], [[-128] * 1024, [64] * 1024]),
            1024 * 128 + 1024 * 64,
        )

    def test_requant_int32_to_int8(self) -> None:
        self.assertEqual(requant_int32_to_int8(0, 32768, 15), 0)
        self.assertEqual(requant_int32_to_int8(100, 32768, 15), 100)
        self.assertEqual(requant_int32_to_int8(500, 32768, 15), 127)
        self.assertEqual(requant_int32_to_int8(-500, 32768, 15), -128)
        self.assertEqual(requant_int32_to_int8(1, 1, 1), 1)
        self.assertEqual(requant_int32_to_int8(-1, 1, 1), 0)
        self.assertEqual(requant_int32_to_int8(-3, 1, 1), -1)

    def test_range_guards(self) -> None:
        with self.assertRaises(ValueError):
            dot_w4a8([8], [1])
        with self.assertRaises(ValueError):
            dot_ternary([2], [1])
        with self.assertRaises(ValueError):
            wrap_signed(0, 1)
        with self.assertRaises(ValueError):
            requant_int32_to_int8(0, 0, 32)

    def test_vector_export_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "vec.txt"
            n_frames, n_beats = export_mac_vectors(
                mode="w4a8", out_path=p, lanes=4, acc_width=16, num_random_frames=10, seed=42
            )
            self.assertGreaterEqual(n_frames, 14)
            self.assertGreaterEqual(n_beats, n_frames)
            self.assertEqual(export_requant_vectors(pathlib.Path(td) / "req.vec", num_random=20), 34)
            self.assertEqual(cli_main(["info"]), 0)

    def test_checkpoint_roundtrip_quant_sim_and_emit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            fmodel = make_synthetic_model(dim=16, hidden_dim=32, n_layers=1, n_heads=2, vocab_size=32, max_seq_len=8)
            bin_path = root / "tiny.bin"
            save_llama2c_bin(fmodel, bin_path)
            loaded = load_llama2c_bin(bin_path)
            self.assertEqual(loaded.config, fmodel.config)
            self.assertAlmostEqual(loaded.token_embedding[0][0], fmodel.token_embedding[0][0], places=5)

            for qmode in ("w4a8", "ternary"):
                qmodel = quantize_model(loaded, quant_mode=qmode, group_size=16)
                sim = GoldenTransformerSim(qmodel)
                traces = sim.generate((1, 2), gen_steps=3)
                self.assertEqual(len(traces), 5)
                out_dir = root / f"dist_{qmode}"
                manifest = emit_compiled_model(qmodel, out_dir=out_dir, lanes=8, prompt_tokens=(1, 2), gen_steps=3)
                self.assertEqual(manifest["generated_tokens"], [t.next_token for t in traces])
                self.assertTrue((out_dir / "covalir_top.sv").exists())
                self.assertTrue((out_dir / "mem" / "weights.hex").exists())

            rc = cli_main(
                [
                    "compile",
                    str(bin_path),
                    "--quant",
                    "ternary",
                    "--lanes",
                    "8",
                    "--group-size",
                    "16",
                    "--out",
                    str(root / "cli_dist"),
                ]
            )
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
