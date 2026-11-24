# `hw/formal/` — Formal Verification (SymbiYosys)

Formal verification harnesses live beside their modules (starting with `hw/ip/prim/formal/prim_fifo_sync.sby`, which proves bounded count, pointer/count consistency, `clr_i` handshake gating, stall stability, and 2-element FIFO ordering/data integrity).

Executed automatically in GitHub Actions CI via OSS CAD Suite (`sby -f hw/ip/prim/formal/prim_fifo_sync.sby`).
