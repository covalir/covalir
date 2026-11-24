# CovalIR — Architecture & Roadmap

> **Status:** Active Development (Phase 1 compiler & bit-exact RTL verified in Verilator, SymbiYosys, Yosys, and Verible CI). See [STATUS](status.md) for what is implemented and verified today.
> **Project & Repo:** **CovalIR** (`github.com/covalir/covalir`, Apache-2.0)

---

## 1. Silicon Physics & Honesty Contract

Every claim in CovalIR must obey first-principles memory bandwidth and hardware resource limits.

### 1.1 Batch=1 Autoregressive Decode Roofline

For batch-1 autoregressive decode, throughput is strictly upper-bounded by:

$$\text{TPS}_{\max} = \min\left(\frac{\eta_{\text{mem}} \cdot \text{BW}_{\text{peak}}}{\text{Active Weight Bytes / Token} + \text{KV Read Bytes / Token}},\; \frac{f_{\text{clk}}}{\text{Cycles}_{\text{compute+overhead}}}\right)$$

Where realistic HBM/DRAM sustained efficiency $\eta_{\text{mem}} \approx 0.60\text{–}0.80$ after shell, AXI, and refresh overhead (and weights include quantization scale overhead: ~4.25 bits/param for W4 group-128, 2.0 bits/param for 2-bit packed ternary).

| Target Hardware | On-Chip SRAM | External Mem & Peak BW | Example Model & Format | Active Bytes / Token (Weights Only) | 100% Theoretical Cap | Realistic Sustained Range ($\eta = 60\text{–}80\%$) |
|---|---:|---|---|---:|---:|---:|
| **Tang Nano 20K** (GW2AR-18, ~$35) | ~100 KB | 8 MB SDRAM (~0.6 GB/s) | TinyStories-15M, W4 (~8 MB) | ~8 MB | ~75 tok/s | **10–45 tok/s** *(compute/SDRAM bound)* |
| **Artix-7 / ECP5** ($150–$400) | 1–13 MB | DDR3/DDR4 (12–19 GB/s) | TinyStories-15M, 2-bit Ternary (~3.8 MB) | ~3.8 MB (on-chip/DDR) | >3,000 tok/s (if SRAM) | **Measured in Phase 1** |
| **AWS `f2.6xlarge`** (1× VU47P) | ~43 MB *(total die; shell takes a slice)* | 16 GB HBM2 (460 GB/s peak) | 8B Dense, W4 (~4.25 GB) | ~4.25 GB | **108 tok/s** | **65–85 tok/s** *(Phase 2 target)* |
| **AWS `f2.6xlarge`** (1× VU47P) | ~43 MB *(total die)* | 16 GB HBM2 (460 GB/s peak) | 8B Native Ternary, 2-bit/trit (~2.0 GB) | ~2.0 GB | **230 tok/s** | **135–180 tok/s** *(Phase 2 target)* |
| **AMD Alveo V80** (~$10K) | ~47 MB | 32 GB HBM2e (819 GB/s peak) | 8B Dense, W4 (~4.25 GB) | ~4.25 GB | **192 tok/s** | **115–155 tok/s** |
| *(Ref)* **RTX 5090** (~$2K GPU) | 96 MB L2 | 32 GB GDDR7 (1,792 GB/s) | 8B Dense, W4 (~4.25 GB) | ~4.25 GB | **421 tok/s** | **250–340 tok/s** *(GPUs win dense W4!)* |

**Physical takeaways (no hype):**
* **GPUs crush single-FPGA cards on standard dense FP16/INT8/W4 serving** at both cost and bandwidth (`1,792 GB/s` on a $2K RTX 5090 vs `460 GB/s` on an AWS VU47P).
* **Notes on common traps:**
  * A 30B W4 MoE (`30.5B × 4.25 bits ≈ 16.2 GB`) does **not** fit in 16 GB HBM2 once KV cache and embeddings are included.
  * On-chip SRAM speculative decoding is constrained by vocabulary size (e.g., 128k × 512 × INT4 embeddings + untied LM head ≈ 32–64 MB, exceeding VU47P's entire 43 MB SRAM before draft weights or KV cache).
  * 2-bit ternary packing gives **2×** HBM savings over INT4 (not 2.5×; 5-trit/byte base-3 unpacking is not implemented).
  * FPGA behavioral RTL is **not** "ASIC tapeout in days" — real tapeouts require foundry SRAM macros, DFT/scan/MBIST, CTS/CDC, pad rings, IR-drop, and multi-corner signoff. CovalIR's honest cross-target guarantee is **lint-clean SystemVerilog that synthesizes cleanly under both FPGA flows and Yosys generic standard cells in CI**.

### 1.2 Mandatory Benchmark Disclosure Contract

No CovalIR hardware number may be published unless accompanied by a reproducible script and report listing:
exact model & parameter count, quantization format & packing bits/param, context length, TTFT, prefill TPS, **true single-stream autoregressive decode TPS**, perplexity / exact token match vs. golden reference, board power (W), Fmax / WNS / TNS, LUT / FF / DSP / BRAM / URAM utilization, and SHA-256 checkpoint hashes.

---

## 2. Pragmatic Phase 1 Architecture (Python Scheduler → Verified SystemVerilog)

Rather than spending years building custom MLIR/CIRCT dialects before generating a single token, Phase 1 uses a **simple, auditable 3-stage pipeline**:

```text
PyTorch Checkpoint (SafeTensors / TinyStories-15M)
                      │
      ┌───────────────▼───────────────┐
      │ 1. Quantizer & Golden Sim     │  compiler/covalir/quant/
      │    W4A8 (per-group) & 2-bit   │  Bit-exact Python/C++ reference
      │    Ternary + Fixed-Point LUTs │  Emits golden test vectors & token IDs
      └───────────────┬───────────────┘
                      │
      ┌───────────────▼───────────────┐
      │ 2. Python Scheduler & Codegen │  compiler/covalir/gen/
      │    Parameterized SV Emitter   │  Instantiates verified hw/ip/ primitives
      │    Weight/BRAM Memory Packing │  (mac_w4a8, mac_ternary, prim_fifo_sync, SRAM)
      └───────────────┬───────────────┘
                      │
      ┌───────────────▼───────────────┐
      │ 3. Verify & Build             │  Verilator + SymbiYosys + Yosys
      │    Bit-Exact Token Match      │  FOSS Bitstream (Yosys/nextpnr/OpenXC7)
      │    + Measured Hardware Run    │  + Single AWS F2 AFI (cached per shape)
      └───────────────────────────────┘
```

* **Why no MLIR/CIRCT in Phase 1?** Maintaining out-of-tree LLVM/MLIR dialects and CIRCT's moving APIs delays the first real hardware token by months. Re-evaluate MLIR/CIRCT at **v0.3** only when a second frontend or real multi-model graph optimization demands it.
* **Why no RapidWright `.dcp` stitching in Phase 1?** AWS F2 AFI creation runs asynchronously on AWS's service (~30–60 min outside user control) and pre-routed tiles are fragile across shell/Vivado versions. A **cache of pre-built AFIs per model shape + HBM weight reload** delivers fast iteration without brittle tile-stitching.

---

## 3. Realistic 8-Week Phase 1 Wedge: One Model, One Board, One Measured Number

**Goal:** Ship the first fully open-source, bit-exact **PyTorch → SystemVerilog → real FPGA bitstream** flow for **TinyStories-15M** (W4A8 and 2-bit ternary), verified against a golden reference and measured on real hardware.

| Week | Milestone | Hard Exit Gate |
|---:|---|---|
| **1** | **Freeze Phase 1 Spec & Requant Block** | One-page spec locking exact model (`TinyStories-15M`), quantization (`W4A8` group-32/64 + `2-bit ternary`), target boards (Tang Nano 20K / Artix-7 / ECP5 + 1 AWS F2 AFI), and pipelined per-group requantization RTL verified in Verilator. |
| **2** | **Bit-Exact Golden Simulator (`golden_sim`)** | PyTorch checkpoint → quantized weights + fixed-point RMSNorm/RoPE/SiLU/Softmax LUTs → bit-exact Python/C++ token generator with perplexity & top-1 match vs. FP16 on 20 prompts. |
| **3** | **Single-Layer Generated RTL** | Python emitter stitches `mac_w4a8`/`mac_ternary`, `prim_generic_sram_2p`, `prim_fifo_sync`, control FSM, and nonlinear LUTs for 1 Transformer layer; matches `golden_sim` bit-for-bit in Verilator. |
| **4** | **Full-Model Generated RTL** | All layers, embedding lookup, LM head, and argmax sampler simulated end-to-end in Verilator; 10 prompts produce identical token IDs to `golden_sim`. |
| **5** | **FOSS Synthesis & Timing Closure** | `Yosys` + `nextpnr` / `OpenXC7` synthesis & place-and-route closes timing cleanly; Fmax, LUT/FF/BRAM/DSP, and critical paths documented. |
| **6** | **Live Hardware Bring-Up & Measurement** | Real bitstream runs on edge FPGA board (plus 1 AWS F2 AFI); measured single-stream decode TPS, TTFT, Watts, and exact token output logged with reproducible scripts. |
| **7** | **One-Command CLI (`covalir compile`)** | Clean checkout runs `covalir compile <checkpoint> --quant w4a8|ternary --target <board>` through sim and bitstream generation. |
| **8** | **Public Release & Phase 2 Gate Review** | Publish full reproducible benchmark report + CI proofs; decide Phase 2 investment based on real hardware numbers. |

---

## 4. Phase 2 Entry Gate (1.7B–8B Models on AWS EC2 F2)

CovalIR will **only** begin Phase 2 (HBM-backed 1.7B–8B models on AWS `f2.6xlarge`) after:
1. Phase 1 ships bit-exact tokens and measured hardware telemetry.
2. A standalone AWS F2 HBM2 streamer microbenchmark measures **≥70% of 460 GB/s (≥320 GB/s)** sustained across the F2 shell's AXI interfaces on real hardware.
3. Vendor `prim_xilinx_usp` wrappers (explicit `DSP48E2` / `URAM288` / SLR register slices) pass Vivado post-route timing closure at ≥250 MHz.
