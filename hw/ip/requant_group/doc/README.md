# `requant_group` — Pipelined Fixed-Point `INT32 → INT8` Requantization Unit

Applies a per-group/per-row unsigned 16-bit fixed-point scale multiplier and 5-bit arithmetic right shift with half-up rounding, then saturates to signed `INT8` (`[-128, 127]`).

* **Stage 1 (registered):** Signed `AccWidth`-bit × unsigned 16-bit multiplier.
* **Stage 2:** Rounding bias + arithmetic right shift + clamp into a 2-entry `prim_fifo_sync` decoupling buffer (`in_ready_o` has zero combinational dependency on `out_ready_i` and is gated by `!clr_i`).
* **Golden reference:** Matches `compiler/covalir/quant/golden.py::requant_int32_to_int8` bit-for-bit.
