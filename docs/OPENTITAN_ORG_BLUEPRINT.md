# CovalIR — Repository Layout & Open-Source EDA Standards

## 1. Monorepo Architecture

CovalIR maintains its compiler, verified RTL IP library, formal harnesses, synthesis targets, and regression tooling atomically inside a single monorepo (`github.com/covalir/covalir`):

* `hw/` — Verified SystemVerilog primitives (`prim/`, `prim_generic/`), compute IPs (`mac_w4a8/`, `mac_ternary/`, `requant_group/`), end-to-end Transformer core (`transformer_core/`), lint rules (`lint/`), and formal harnesses (`formal/`).
* `compiler/` — Pure-Python checkpoint loaders, tokenizer, quantizer, bit-exact `GoldenTransformerSim`, and SystemVerilog/memory emitter.
* `targets/` — Open-source FPGA (`Yosys`/`nextpnr`/`OpenXC7`) and cloud FPGA (`aws_f2/`) target configurations.
* `sw/` — Host runtime and driver notes.
* `util/` — Regression runner (`dvsim/`).
* `docs/` — Architecture specification, hardware IP catalog, compiler CLI reference, and status ledger.

---

## 2. Open-Source EDA Standards Adopted

CovalIR follows OpenTitan and CHIPS Alliance hardware engineering practices:

| Layer | Open-Source Tool / Standard | Status in CovalIR |
|---|---|---|
| **Hardware Packaging** | **FuseSoC (`CAPI=2` `.core` files)** | Active (`fusesoc.conf` + 6 `.core` manifests with `depend:` links). |
| **Technology Separation** | **OpenTitan `prim_*` pattern** | Active for `prim_fifo_sync` & `prim_generic_sram_2p`; vendor `prim_xilinx_usp` planned for Phase 2. |
| **Style & Linting** | **Verilator (`-Wall -Werror`) + Verible** | Verilator strict lint runs locally & in CI; Verible runs in GitHub Actions CI. |
| **Differential Verification** | **Verilator 5 + `golden.py` / `golden_sim.py`** | Active — single Python golden reference drives IP C++ testbenches and end-to-end Transformer proofs. |
| **Formal Verification** | **SymbiYosys (`sby`) + Z3** | Active for `prim_fifo_sync` (`prove` + `cover`), gated in GitHub Actions CI. |
| **Cross-Target Synthesis** | **Yosys generic standard-cell synth** | Gated in GitHub Actions CI across all 5 IP modules and compiled `covalir_top`. |
