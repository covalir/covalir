# CovalIR — Implementation Status

> **Policy:** CovalIR never claims capabilities, formal proofs, or hardware throughput numbers that have not been run and reproduced. Every item below reflects exact current state.

## 1. Implemented & Verified Locally

| Component | Path | What Is Verified |
|---|---|---|
| **Low-latency synchronous FIFO** | [prim_fifo_sync.sv](hw/ip/prim/rtl/prim_fifo_sync.sv) | Verilator strict lint (`-Wall -Werror-WIDTH -Werror-UNUSED -Werror-UNOPTFLAT`), elaboration guards (`Depth >= 2`, power-of-two), and 5,000-cycle differential C++ DV across `Depth ∈ {2, 4, 16}` (incl. `clr_i` gating, stall stability, mid-stream reset). |
| **Behavioral 1R1W SRAM** | [prim_generic_sram_2p.sv](hw/ip/prim_generic/rtl/prim_generic_sram_2p.sv) | Verilator strict lint, elaboration guards (`Depth >= 2` power-of-two, `Width` multiple of 8), and 5,000-cycle differential C++ DV across `Depth ∈ {16, 64}` (byte masks, `rvalid_o`, read-first collisions, sync reset). |
| **Pipelined W4A8 group MAC** | [mac_w4a8.sv](hw/ip/mac_w4a8/rtl/mac_w4a8.sv) | 2-stage pipelined `INT4 × INT8 → signed AccWidth` MAC with `prim_fifo_sync #(Depth=2)` decoupling. Verified bit-exact against `golden.py` across `(Lanes=4, AccWidth=16)` (incl. wrap) & `(16, 32)`. |
| **Pipelined 2-bit ternary MAC** | [mac_ternary.sv](hw/ip/mac_ternary/rtl/mac_ternary.sv) | 2-stage zero-multiplier `{-1, 0, +1} × INT8` MAC (2 bits/trit, `2'b10` assertion, exact `-(-128)=+128`), `prim_fifo_sync #(Depth=2)` decoupling. Verified bit-exact against `golden.py` across `(8, 16)` & `(32, 32)`. |
| **Pipelined INT32 → INT8 Requantizer** | [requant_group.sv](hw/ip/requant_group/rtl/requant_group.sv) | 2-stage fixed-point `(acc * scale_u16 + bias) >>> rshift_u5` saturating requantizer with `prim_fifo_sync #(Depth=2)` decoupling. Verified bit-exact across 1,014 `golden.py` vectors + `clr_i` + reset + stall stability. |
| **End-to-End Integer Transformer Core** | [covalir_llm_core.sv](hw/ip/transformer_core/rtl/covalir_llm_core.sv) | Full autoregressive pure-integer Llama-2 / TinyStories engine instantiating `mac_w4a8`/`mac_ternary`: Q8.8 (`INT16`) activations & KV-cache, CLZ-normalized RMSNorm LUT, dynamic power-of-2 INT8 absmax quantizer, Group-32 `Q1.15` scaled 48-bit MAC accumulation, `Q1.14` RoPE, causal MHA + exact 40-cycle restoring divider, 512-entry `Q8.8` SwiGLU MLP, LM head, deterministic 48-bit argmax. Verified **100% bit-exact** (logits & tokens) vs. `golden_sim.py` in Verilator across: (1) synthetic micro-model in both **`w4a8`** & **`ternary`** modes (`7,341 cycles/token`), and (2) **real Karpathy `stories15M.bin` (15.19M params, 6L/288d/768h/32k vocab, `w4a8`, `Lanes=16`, `GroupSize=32`) across 16 autoregressive steps (`32,547,278 cycles`, avg `2,034,204 cycles/token`) producing `' Once upon a time, there was a little girl named Lily. She loved'` — matching FP32 16/16 tokens with zero floating-point hardware.** |
| **Checkpoint Loader, Tokenizer, Quantizer, Golden Sim & SV Emitter** | [checkpoint.py](compiler/covalir/front/checkpoint.py), [tokenizer.py](compiler/covalir/front/tokenizer.py), [quantize_model.py](compiler/covalir/quant/quantize_model.py), [golden_sim.py](compiler/covalir/quant/golden_sim.py), [emit_sv.py](compiler/covalir/gen/emit_sv.py), [cli.py](compiler/covalir/cli.py) | Zero-dependency loaders for Karpathy `llama2.c` `.bin`, `tokenizer.bin`, HuggingFace `.safetensors` (F32/F16/BF16), and synthetic models; Group-G symmetric W4A8 & AbsMean 2-bit ternary quantizers (`Q8.8` activations, `Q1.15` group scales); bit-exact `GoldenTransformerSim`; one-command `covalir compile ... --sim` emitting packed `.hex` images, `covalir_top.sv`, `tb_covalir_top.cpp`, `synth_yosys.ys`, and SHA-256 `manifest.json`. |
| **FuseSoC Manifests & FOSS Synth Script** | [fusesoc.conf](fusesoc.conf), `hw/ip/*/*.core`, [build_yosys.sh](targets/fpga_foss/build_yosys.sh) | All 6 cores discovered & linted via FuseSoC; `build_yosys.sh` supports `generic`, `ecp5`, and `gowin` Yosys flows. |

## 2. CI-Gated in GitHub Actions

Wired in [.github/workflows/ci.yml](.github/workflows/ci.yml):
* **Python tests:** `unittest` suite (golden math, requant, checkpoint `.bin` roundtrip, quantizer, `GoldenTransformerSim`, SV emitter, CLI).
* **Verilator + FuseSoC:** Strict lint, 8 negative elaboration guards, all 5 IP differential sims, AND both end-to-end compiled Transformer bit-exact proofs (`w4a8` + `ternary`).
* **SymbiYosys Formal:** `prove` (k-inductive) + `cover` on `prim_fifo_sync.sby`.
* **Yosys Generic Synthesis:** Synthesizability & cell stats for all 5 IP modules + a compiled `covalir_top` ternary Transformer core.
* **Verible Style Lint:** All `hw/ip/*/rtl/*.sv` against `hw/lint/verible.rules`.

## 3. Planned Next (Physical Board & Cloud Targets)

* **Physical Edge FPGA Bring-Up & Board Power/TPS Measurement:** Flashing a physical Artix-7 / ECP5 / Tang Nano 20K board with a UART host streamer and recording real hardware tok/s, TTFT, and Watts.
* **AWS EC2 F2 AFI Build & Live HBM Benchmark:** Building an AWS `f2.6xlarge` AFI and measuring standalone HBM2 sustained bandwidth.
* **Phase 2 (Gated on Standalone F2 HBM ≥70% BW):** `prim_xilinx_usp` (`DSP48E2` 2× packing, `URAM288`, SLR slices), HBM2 streamer, and 1.7B–8B models.
