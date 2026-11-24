# `hw/top_llm/` — Compiled Model Top-Level Output

`covalir compile <checkpoint> --quant w4a8|ternary --out <dir>` generates a model-specialized `covalir_top.sv` wrapping `hw/ip/transformer_core/rtl/covalir_llm_core.sv`, packed `.hex` weight/LUT memory images, a bit-exact Verilator testbench (`tb_covalir_top.cpp`), and a Yosys synthesis script (`synth_yosys.ys`).
