// Copyright 2026 CovalIR Contributors
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Portable behavioral dual-port (1R1W) synchronous SRAM primitive.
//
// Contract:
// - Depth must be a power of two >= 2 (eliminates out-of-bounds addresses).
// - 1-cycle synchronous read latency with registered rvalid_o (cleared by rst_ni).
// - Read-during-write to the same address is READ-FIRST (returns prior stored word).
// - Byte/word write enable mask (wmask_i, 1 bit per byte; Width must be a multiple of 8).
// - Note: Vendor FPGA/ASIC SRAM macro wrappers (prim_xilinx_usp, prim_asic) are planned
//   for Phase 2; currently only this behavioral generic implementation exists.

module prim_generic_sram_2p #(
  parameter int unsigned Width = 64,
  parameter int unsigned Depth = 1024,
  localparam int unsigned AddrW = (Depth >= 2) ? $clog2(Depth) : 1,
  localparam int unsigned MaskW = (Width >= 8) ? (Width / 8) : 1
) (
  input  logic             clk_i,
  input  logic             rst_ni,

  // Write port
  input  logic             wreq_i,
  input  logic [AddrW-1:0] waddr_i,
  input  logic [Width-1:0] wdata_i,
  input  logic [MaskW-1:0] wmask_i,

  // Read port (1-cycle synchronous read latency, read-first on same-addr collision)
  input  logic             rreq_i,
  input  logic [AddrW-1:0] raddr_i,
  output logic             rvalid_o,
  output logic [Width-1:0] rdata_o
);

  if ((Width == 0) || ((Width % 8) != 0)) begin : gen_err_width
    $error("prim_generic_sram_2p: Width (%0d) must be a positive multiple of 8", Width);
  end
  if ((Depth < 2) || ((Depth & (Depth - 1)) != 0)) begin : gen_err_depth
    $error("prim_generic_sram_2p: Depth (%0d) must be a power of two >= 2", Depth);
  end

  logic [Width-1:0] mem [Depth];

  always_ff @(posedge clk_i) begin
    if (rreq_i) begin
      rdata_o <= mem[raddr_i];
    end
    if (wreq_i) begin
      for (int unsigned b = 0; b < MaskW; b++) begin
        if (wmask_i[b]) begin
          mem[waddr_i][b*8 +: 8] <= wdata_i[b*8 +: 8];
        end
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      rvalid_o <= 1'b0;
    end else begin
      rvalid_o <= rreq_i;
    end
  end

endmodule
