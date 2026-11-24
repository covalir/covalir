# `mac_ternary` — Pipelined 2-Bit Packed Ternary (`{-1, 0, +1}`) Zero-Multiplier MAC

Pipelined 2-stage zero-multiplier vector MAC computing `Lanes` parallel ternary (`{-1, 0, +1}`) × signed `INT8` (`[-128, 127]`) dot products accumulated into a signed `AccWidth`-bit (`>= 9`) register until `last_i`.

## Encoding & Packing Contract
Each weight uses a **2-bit code** (2× weight bandwidth reduction vs. INT4; 5-trit/byte dense unpacking is not implemented):
* `2'b00` → `0`
* `2'b01` → `+1` (`+a_lane`)
* `2'b11` → `-1` (`-a_lane`, full signed 9-bit negation so `-(-128) = +128` is exact)
* `2'b10` → **Illegal** (triggers `$error` in simulation; hardware clamps to `0`)

## Architecture Notes
* **Stage 1 (registered):** `Lanes` parallel add/sub/zero muxes (`term_q`).
* **Stage 2 (registered):** Reduction adder tree + running accumulator into a 2-entry `prim_fifo_sync` decoupling buffer (`in_ready_o` has zero combinational dependency on `out_ready_i`, and is gated by `!clr_i`).
* **Golden reference:** Verified bit-exact against `compiler/covalir/quant/golden.py` vectors in Verilator.
