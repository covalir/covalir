# Compiler CLI & Synthesis Guide

CovalIR's Phase 1 compiler (`compiler/covalir/`) runs on **pure Python 3.11+ standard library** (no PyTorch or NumPy required).

---

## CLI Commands

### `covalir compile`

Compiles a Llama-2 / TinyStories checkpoint into packed `.hex` weights, LUT ROMs, a specialized `covalir_top.sv` wrapper, a bit-exact Verilator testbench (`tb_covalir_top.cpp`), a Yosys synthesis script (`synth_yosys.ys`), and a SHA-256 `manifest.json`.

```bash
# Synthetic 2-layer micro-model (fast local / CI proof)
PYTHONPATH=compiler python3 -m covalir.cli compile synthetic \
  --quant ternary \
  --lanes 16 \
  --prompt "1,7,13" \
  --gen-steps 5 \
  --out build/demo_ternary \
  --sim

# Real Karpathy llama2.c `stories15M.bin` (15.19M params — matches FP32 16/16 tokens in Verilator!)
PYTHONPATH=compiler python3 -m covalir.cli compile stories15M.bin \
  --quant w4a8 \
  --lanes 16 \
  --group-size 32 \
  --tokenizer tokenizer.bin \
  --prompt 1 \
  --gen-steps 15 \
  --out build/stories15m_w4a8 \
  --sim

# HuggingFace SafeTensors LlamaForCausalLM directory (`config.json` + `model.safetensors`)
PYTHONPATH=compiler python3 -m covalir.cli compile path/to/hf_model_dir \
  --quant ternary \
  --lanes 16 \
  --out build/hf_ternary
```

### `covalir sim`

Re-runs bit-exact Verilator differential verification against `golden_trace.txt` on any compiled directory:

```bash
PYTHONPATH=compiler python3 -m covalir.cli sim build/demo_ternary
```

### `covalir gen-vectors`

Exports deterministic golden test vectors for individual compute IPs (`w4a8`, `ternary`, `requant`):

```bash
PYTHONPATH=compiler python3 -m covalir.cli gen-vectors --mode requant --frames 1000 --out build/req.vec
```

---

## Open-Source FPGA Synthesis (`targets/fpga_foss/build_yosys.sh`)

Requires [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) (`yosys` on `PATH`):

```bash
# Generic standard-cell synthesis & area stats (also runs automatically in CI)
./targets/fpga_foss/build_yosys.sh build/demo_ternary generic

# Lattice ECP5 netlist JSON (for nextpnr-ecp5)
./targets/fpga_foss/build_yosys.sh build/demo_ternary ecp5

# Gowin / Tang Nano 20K netlist JSON (for nextpnr-himbaechel)
./targets/fpga_foss/build_yosys.sh build/demo_ternary gowin
```

---

## Emitted Artifact Layout

```text
<out_dir>/
├── covalir_top.sv       # Specialized top-level instantiating covalir_llm_core
├── tb_covalir_top.cpp   # Differential Verilator testbench checking logits & tokens
├── golden_trace.txt     # Bit-exact GoldenTransformerSim prompt + decode trace
├── synth_yosys.ys       # Standalone Yosys synthesis script
├── manifest.json        # Model dims, quant mode, generated tokens & SHA-256 hashes
└── mem/
    ├── emb.hex          # Q8.8 (INT16) token embedding table
    ├── rms.hex          # Q8.8 (INT16) RMSNorm weights (all layers + final)
    ├── weights.hex      # Packed Group-G W4 or 2-bit ternary beats in execution order
    ├── scales.hex       # Q1.15 (UINT16) per-group weight scale ROM
    ├── inv_rms.hex      # 256-entry Q1.15 CLZ-normalized inverse-RMS ROM
    ├── silu.hex         # 512-entry Q8.8 (INT16) SiLU ROM ([-16.0, +16.0))
    ├── exp.hex          # 256-entry Q0.8 softmax exp(-d/16) ROM
    ├── rope_cos.hex     # Q1.14 (INT16) RoPE cosine ROM
    └── rope_sin.hex     # Q1.14 (INT16) RoPE sine ROM
```
