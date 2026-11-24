# Phase 1 Specification — End-to-End Integer Transformer (`TinyStories-15M`)

## 1. Locked Scope

* **Model family:** Llama-2 decoder-only architecture (matching Karpathy's `stories15M.bin` and HuggingFace SafeTensors `LlamaForCausalLM`).
  * Full `TinyStories-15M` geometry: `dim=288`, `n_layers=6`, `n_heads=6`, `n_kv_heads=6`, `head_dim=48`, `hidden_dim=768`, `vocab_size=32000`, `max_seq_len=256`.
  * Fast CI / Verilator micro-geometry (same exact RTL, parameterized): `dim=32`, `n_layers=2`, `n_heads=2`, `n_kv_heads=2`, `head_dim=16`, `hidden_dim=64`, `vocab_size=64`, `max_seq_len=32`.
* **Quantization modes (both supported by the same compiler & RTL generator):**
  1. **`w4a8`:** Group-32 symmetric signed `INT4` weights (`[-7, 7]` quantized, `[-8, 7]` hardware range) with unsigned `Q1.15` (`UINT16`) per-group scales × dynamically quantized `INT8` activations (`[-128, 127]`) → `INT32` per-group MAC → 48-bit scaled row accumulator → `Q8.8` (`INT16`) output.
  2. **`ternary`:** Group-32 AbsMean 2-bit packed ternary weights (`{-1, 0, +1}`, zero hardware multipliers) with unsigned `Q1.15` (`UINT16`) per-group scales × dynamically quantized `INT8` activations → `INT32` per-group MAC → 48-bit scaled row accumulator → `Q8.8` (`INT16`) output.
* **Pure-integer arithmetic contract (zero floating-point hardware, 100% bit-exact between `golden_sim.py` and SystemVerilog RTL):**
  * **Residual & Activation Representation:** Signed `Q8.8` (`INT16`, scale `256`, range `[-128.0, +127.996]`) across token embeddings, RMSNorm outputs, RoPE, KV-cache, attention context, SwiGLU, and residual additions — capturing real `stories15M.bin` activations (up to `±78`) without clipping.
  * **Dynamic Power-of-2 INT8 Activation Quantizer (zero hardware divider):**
    * Scans `max_abs = max(|x_i|)` over the `Q8.8` input vector; computes `e = (max_abs <= 127) ? 0 : (bit_length(max_abs) - 7)` (`e ∈ [0..9]`).
    * Quantizes each activation via exact rounding right shift: `qx_i = clamp((x_i + round_bias) >>> e, -128, 127)`.
  * **Grouped Linear Projection (`y = W @ x`):**
    1. Streams `Lanes`-wide beats into `mac_w4a8` or `mac_ternary` to compute exact signed `INT32` per-group dot products (`GroupSize` elements, default `32`).
    2. Multiplies each group's `INT32` dot product by its `Q1.15` unsigned 16-bit scale and accumulates in signed 48-bit (`acc48 += dot32 * scale_q15`).
    3. Requantizes to `Q8.8` (`INT16`) via exact rounding shift `sh = 15 - e` (`clamp((acc48 + (1 << (sh - 1))) >>> sh, -32768, 32767)`), or outputs raw signed 48-bit logits for the LM head.
  * **CLZ-Normalized RMSNorm (`Q8.8` → `Q8.8`):**
    * Computes `sum_sq = sum(x_i^2)` in `UINT48`, `msq = max(1, (sum_sq * inv_dim_q16) >> 16)`.
    * Shifts `msq` right by 2 bits `k` times until `mv < 256`, looks up `inv_q15 = inv_rms_lut[mv]` (`round(32768 / sqrt(u))`), and computes `y_i = clamp((x_i * inv_q15 * w_norm_i + round_bias) >>> (15 + k), -32768, 32767)`.
  * **RoPE (Rotary Position Embeddings):**
    * Precomputed `cos` / `sin` ROM in `Q1.14` (`INT16`, scaled by `16384`), exact pair rotation `(a*c - b*s + 8192) >>> 14`, `(a*s + b*c + 8192) >>> 14`, clamped to `Q8.8` (`INT16`).
  * **Causal Self-Attention + On-Chip KV Cache SRAM:**
    * Stores `K` and `V` (`Q8.8` `INT16`) directly in on-chip SRAM (`[layer][pos][dim]`).
    * Computes 48-bit dot product `score_t = (q · k_t * att_scale_q12 + 2^27) >>> 28` → max-subtracted 8-bit index into `exp_lut` (`Q0.8`, `0..255`) → 40-bit weighted sum over `v_t` normalized by an exact 40-cycle restoring integer divider → `Q8.8` (`INT16`) context vector.
  * **SwiGLU MLP:**
    * `gate = linear(W1, x_norm)`, `up = linear(W3, x_norm)` in `Q8.8`.
    * `silu_gate = silu_lut[clamp((gate >>> 4) + 256, 0, 511)]` (512-entry `Q8.8` ROM covering `[-16.0, +16.0)` in `1/16` steps).
    * `h = clamp((silu_gate * up + 128) >>> 8, -32768, 32767)` → `down = linear(W2, h)`.
  * **Residual Add:** Saturating signed `Q8.8` (`INT16`) addition (`clamp(a + b, -32768, 32767)`).
  * **LM Head & Sampler:** Projects final normalized hidden state against `vocab_size` rows and outputs both exact signed 48-bit `max_logit_o` and deterministic `argmax` next token ID.

## 2. Verification Gates

1. **RTL Primitives & Nonlinear Blocks:** Every IP (`prim_fifo_sync`, `prim_generic_sram_2p`, `mac_w4a8`, `mac_ternary`, `requant_group`, `covalir_llm_core`) passes Verilator strict lint (`-Wall -Werror`), Verible style lint, and Yosys generic synthesis in CI.
2. **Bit-Exact End-to-End Synthetic & Real `stories15M.bin` Proof:** `covalir compile` + `covalir sim` emits full-model SystemVerilog + packed `.hex` weights and proves **100% identical 48-bit logits and autoregressive token IDs** between `golden_sim.py` and Verilator across:
   * Synthetic micro-model (`w4a8` and `ternary`, `7,341 cycles/token`).
   * **Real Karpathy `stories15M.bin`** (`15.19M` params, `w4a8`, `Lanes=16`, `GroupSize=32`): 16 autoregressive steps verified bit-exact in `32,547,278` cycles (`2,034,204 cycles/token`), producing `" Once upon a time, there was a little girl named Lily. She loved"` — **matching FP32 16/16 tokens identically**.
3. **Hardware Bring-Up Target:** Edge FPGA bitstream (`Yosys` / `nextpnr` / `OpenXC7`) and AWS EC2 F2 HBM2 benchmark are gated on passing all local simulation and synthesis checks first.
