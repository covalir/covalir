# `transformer_core` — Parameterized End-to-End Integer Transformer Core

Instantiates CovalIR's verified compute IPs (`mac_w4a8` or `mac_ternary` + `requant_group` + `prim_fifo_sync`) inside a complete autoregressive Llama-2 / TinyStories inference engine with on-chip SRAM KV cache, fixed-point RMSNorm, RoPE, causal multi-head attention (with exact restoring integer divider), SwiGLU MLP, LM head, and deterministic argmax sampler.

Matches `compiler/covalir/quant/golden_sim.py` bit-for-bit across both `w4a8` and `ternary` modes.
