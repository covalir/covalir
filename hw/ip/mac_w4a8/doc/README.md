# `mac_w4a8` — Pipelined Signed `INT4 × INT8` Group Dot-Product MAC

Pipelined 2-stage vector MAC computing `Lanes` parallel signed `INT4` (`[-8, 7]`) × signed `INT8` (`[-128, 127]`) products accumulated into a signed `AccWidth`-bit (`>= 12`) register until `last_i` terminates a quantization group.

## Architecture Notes
* **Stage 1 (registered):** `Lanes` parallel behavioral signed multipliers (`prod_q`). *(UltraScale+ DSP48E2 2× packing is not implemented yet; planned as a Phase 2 `prim_xilinx_usp` backend.)*
* **Stage 2 (registered):** Reduction adder tree + running signed accumulator pushing completed group sums into a 2-entry `prim_fifo_sync` decoupling buffer.
* **Zero combinational ready loop:** `in_ready_o` depends only on registered stage/FIFO state and `!clr_i` (never combinationally on `out_ready_i`).
* **Per-group dequantization:** `last_i` marks quantization group boundaries (e.g. `group_size = 32..128`); per-group scale multiplication is performed by a downstream requantization unit (Phase 1 Week 1).
* **Golden reference:** Verified bit-exact against `compiler/covalir/quant/golden.py` vectors in Verilator.
