# Hardware IP Catalog (`hw/ip/`)

Modeled on OpenTitan's `hw/ip/` and `prim_*` conventions, every CovalIR RTL module uses synchronous active-low reset (`rst_ni`), strict elaboration-time parameter guards (`$error`), latency-insensitive `valid/ready` handshakes gated by `!clr_i`, and FuseSoC `CAPI=2` `.core` manifests.

---

## Verified RTL Modules

| Module | FuseSoC VLNV | Pipeline / Microarchitecture | Verification Status |
|---|---|---|---|
| **`prim_fifo_sync`** | `covalir:prim:fifo_sync:0.1` | Low-latency FWFT sync FIFO (`Depth >= 2`, power-of-two). `wready_o` has zero combinational dependency on `rready_i`; both handshakes gated by `!clr_i`. | Verilator lint + guards (`Depth=1,3`) + 5k-cycle DV (`Depth=2,4,16`) + SymbiYosys `prove` (k-inductive 2-element ordering) & `cover` + Yosys synth + Verible |
| **`prim_generic_sram_2p`** | `covalir:prim_generic:sram_2p:0.1` | Behavioral 1R1W synchronous SRAM (`Depth >= 2` power-of-two, `Width` multiple of 8). Byte write masks (`wmask_i`), 1-cycle registered `rvalid_o`, documented read-first collision semantics. | Verilator lint + guards (`Depth=1,1000`, `Width=12`) + 5k-cycle DV (`Depth=16,64`, ~1,000 read-first collisions) + Yosys synth + Verible |
| **`mac_w4a8`** | `covalir:ip:mac_w4a8:0.1` | 2-stage pipelined signed `INT4 × INT8 → signed AccWidth` (`>= 12`) group dot-product MAC + `prim_fifo_sync #(Depth=2)` decoupling buffer. | Verilator lint + guard (`AccWidth=8`) + `golden.py` differential DV across `(4,16)` (incl. wrap) & `(16,32)` + `clr_i` + stall stability + Yosys synth + Verible |
| **`mac_ternary`** | `covalir:ip:mac_ternary:0.1` | 2-stage zero-multiplier 2-bit packed ternary (`{-1, 0, +1}`) × `INT8 → signed AccWidth` (`>= 9`) MAC. Exact 9-bit `-(-128)=+128`, simulation `$error` on illegal `2'b10`, `prim_fifo_sync #(Depth=2)`. | Verilator lint + guard (`AccWidth=8`) + `golden.py` differential DV across `(8,16)` & `(32,32)` + `clr_i` + stall stability + Yosys synth + Verible |
| **`requant_group`** | `covalir:ip:requant_group:0.1` | 2-stage fixed-point `INT32 → INT8` scaler: `clamp((acc * scale_u16 + round_bias) >>> rshift_u5, -128, 127)` + `prim_fifo_sync #(Depth=2)` decoupling. | Verilator lint + guard (`AccWidth=4`) + 1,014 `golden.py` vectors + `clr_i` + stall stability + Yosys synth + Verible |
| **`covalir_llm_core`** | `covalir:ip:transformer_core:0.1` | Full pure-integer autoregressive Llama-2 / TinyStories inference engine instantiating `mac_w4a8` or `mac_ternary`. Features `Q8.8` (`INT16`) activations & KV cache, CLZ-normalized RMSNorm LUT, dynamic power-of-2 INT8 absmax quantizer, Group-32 `Q1.15` scaled 48-bit accumulation, `Q1.14` RoPE, causal MHA + exact 40-cycle restoring divider, 512-entry `Q8.8` SwiGLU, LM head, deterministic 48-bit argmax. | Verilator lint + end-to-end bit-exact differential proof vs `golden_sim.py` (`w4a8` & `ternary` synthetic + real 15.19M `stories15M.bin` matching FP32 16/16 tokens) + Yosys synth (`covalir_top`) + Verible |

---

## Running Verification Locally

```bash
# Full Verilator lint, 8 negative parameter guards, 5 IP sims, and 2 E2E Transformer proofs
python3 util/dvsim/run_dv.py

# FuseSoC core discovery & per-core Verilator lint
fusesoc --config fusesoc.conf core list
fusesoc --config fusesoc.conf run --target=lint covalir:ip:transformer_core:0.1
```

## Planned Technology Primitives (Phase 2)

* `hw/ip/prim_xilinx_usp/` — Explicit UltraScale+ `DSP48E2` 2× W4A8 packing, `URAM288` KV banks, and SLR register slices (gated on standalone AWS F2 HBM benchmark reaching ≥70% of 460 GB/s).
* `hw/ip/prim_asic/` — Foundry SRAM macro wrappers (deferred until a concrete ASIC target exists; cross-target synthesizability is checked today via Yosys generic standard-cell synthesis in CI).
