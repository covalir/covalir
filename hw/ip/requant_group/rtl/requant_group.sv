// Copyright 2026 CovalIR Contributors
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Pipelined fixed-point requantization unit: signed INT32 -> signed INT8.
//
// Exact arithmetic contract (matches compiler/covalir/quant/golden.py::requant_int32_to_int8):
//   prod    = signed'(acc_i) * signed'({1'b0, scale_i})      // 49-bit signed
//   bias    = (rshift_i == 0) ? 0 : (49'sd1 <<< (rshift_i - 1))
//   shifted = (prod + bias) >>> rshift_i                     // arithmetic right shift
//   out_q_o = clamp(shifted, -128, 127)
//
// Microarchitecture:
// - Stage 1 (registered): 32x16 signed*unsigned multiply + rshift_q register.
// - Stage 2: rounding shift + saturation into 2-entry prim_fifo_sync decoupling buffer
//   (zero combinational out_ready_i -> in_ready_o path; clr_i gates in_ready_o).

module requant_group #(
  parameter int unsigned AccWidth = 32
) (
  input  logic                       clk_i,
  input  logic                       rst_ni,
  input  logic                       clr_i,

  // Input stream: signed accumulator + unsigned Q-format multiplier + shift
  input  logic                       in_valid_i,
  output logic                       in_ready_o,
  input  logic signed [AccWidth-1:0] acc_i,
  input  logic [15:0]                scale_i,
  input  logic [4:0]                 rshift_i,

  // Output stream: clamped signed INT8 activation
  output logic                       out_valid_o,
  input  logic                       out_ready_i,
  output logic signed [7:0]          out_q_o
);

  if (AccWidth < 8) begin : gen_err_accwidth
    $error("requant_group: AccWidth (%0d) must be >= 8", AccWidth);
  end

  localparam int unsigned ProdW = AccWidth + 17;

  logic signed [ProdW-1:0] prod_d;
  logic signed [ProdW-1:0] prod_q;
  logic [4:0]              rshift_q;
  logic                    s1_valid_q;

  logic                    fifo_wvalid;
  logic                    fifo_wready;
  logic [7:0]              fifo_wdata;
  logic [7:0]              fifo_rdata;
  logic                    fifo_rvalid;
  logic [1:0]              fifo_depth_unused;

  logic s1_can_advance;
  assign s1_can_advance = !clr_i && (!s1_valid_q || fifo_wready);
  assign in_ready_o     = s1_can_advance;

  // Stage 1 combinational multiply (signed acc * unsigned 16-bit scale)
  logic signed [16:0] scale_signed;
  assign scale_signed = $signed({1'b0, scale_i});
  assign prod_d       = $signed(acc_i) * scale_signed;

  // Stage 2 combinational round-half-up + clamp to [-128, 127]
  logic signed [ProdW-1:0] round_bias;
  logic signed [ProdW-1:0] rounded;
  logic signed [7:0]       clamped;

  always_comb begin
    if (rshift_q == 5'd0) begin
      round_bias = '0;
      rounded    = prod_q;
    end else begin
      round_bias = ProdW'(1) <<< (rshift_q - 5'd1);
      rounded    = (prod_q + round_bias) >>> rshift_q;
    end

    if (rounded > ProdW'(127)) begin
      clamped = 8'sd127;
    end else if (rounded < -ProdW'(128)) begin
      clamped = -8'sd128;
    end else begin
      clamped = rounded[7:0];
    end
  end

  assign fifo_wvalid = !clr_i && s1_valid_q;
  assign fifo_wdata  = clamped;

  always_ff @(posedge clk_i) begin
    if (!rst_ni || clr_i) begin
      s1_valid_q <= 1'b0;
      prod_q     <= '0;
      rshift_q   <= '0;
    end else if (s1_can_advance) begin
      s1_valid_q <= in_valid_i;
      if (in_valid_i) begin
        prod_q   <= prod_d;
        rshift_q <= rshift_i;
      end
    end
  end

  prim_fifo_sync #(
    .Width (8),
    .Depth (2)
  ) u_out_fifo (
    .clk_i    (clk_i),
    .rst_ni   (rst_ni),
    .clr_i    (clr_i),
    .wvalid_i (fifo_wvalid),
    .wready_o (fifo_wready),
    .wdata_i  (fifo_wdata),
    .rvalid_o (fifo_rvalid),
    .rready_i (out_ready_i),
    .rdata_o  (fifo_rdata),
    .depth_o  (fifo_depth_unused)
  );

  assign out_valid_o = fifo_rvalid;
  assign out_q_o     = $signed(fifo_rdata);

  logic unused_fifo_depth;
  assign unused_fifo_depth = ^fifo_depth_unused;

endmodule
