// Copyright 2026 CovalIR Contributors
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Pipelined signed INT4 x INT8 -> INT32 group dot-product MAC engine.
//
// Microarchitecture:
// - Stage 1 (registered): Lanes parallel signed INT4 x INT8 lane multipliers.
//   (Uses behavioral signed multiply; DSP48E2 2x-packing is a future vendor prim.)
// - Stage 2 (registered): Adder tree + running signed accumulator across beats until
//   last_i marks the end of a quantization group (e.g. group_size = 32..128 elements).
//   Per-group scale multiplication is performed downstream by a requantization unit.
// - Output stage: 2-entry prim_fifo_sync decoupling buffer so in_ready_o has ZERO
//   combinational dependency on out_ready_i (safe to chain across long pipelines).
// - Synchronous active-low reset (rst_ni) for clean UltraScale+ DSP/BRAM packing.
// - clr_i gates in_ready_o so no input beat is ever acknowledged and silently dropped.

module mac_w4a8 #(
  parameter int unsigned Lanes    = 16,
  parameter int unsigned AccWidth = 32
) (
  input  logic                       clk_i,
  input  logic                       rst_ni,
  input  logic                       clr_i,

  // Input stream: packed Lanes x INT4 weights and Lanes x INT8 activations
  input  logic                       in_valid_i,
  output logic                       in_ready_o,
  input  logic [Lanes*4-1:0]         w_i,
  input  logic [Lanes*8-1:0]         a_i,
  input  logic                       last_i,

  // Output stream: completed signed INT32 group dot-product result
  output logic                       out_valid_o,
  input  logic                       out_ready_i,
  output logic signed [AccWidth-1:0] out_acc_o
);

  if (Lanes == 0) begin : gen_err_lanes
    $error("mac_w4a8: Lanes must be >= 1");
  end
  if (AccWidth < 12) begin : gen_err_accwidth
    $error("mac_w4a8: AccWidth (%0d) must be >= 12 to hold signed INT4 x INT8 products", AccWidth);
  end

  localparam int unsigned EffAccW = (AccWidth >= 12) ? AccWidth : 12;

  // Stage 1: lane products (combinational) -> registered prod_q
  logic signed [11:0] prod_d [Lanes];
  logic signed [11:0] prod_q [Lanes];
  logic               s1_valid_q;
  logic               s1_last_q;

  for (genvar k = 0; k < Lanes; k++) begin : gen_lanes
    logic signed [3:0] w_lane;
    logic signed [7:0] a_lane;
    assign w_lane    = $signed(w_i[k*4 +: 4]);
    assign a_lane    = $signed(a_i[k*8 +: 8]);
    assign prod_d[k] = w_lane * a_lane;
  end

  // Stage 2: reduction + running accumulator
  logic signed [EffAccW-1:0] dot_sum_s1;
  logic signed [EffAccW-1:0] run_acc_q;
  logic signed [EffAccW-1:0] next_acc_s1;

  // Output FIFO signals (Depth=2 breaks combinational out_ready_i -> in_ready_o path)
  logic                      fifo_wvalid;
  logic                      fifo_wready;
  logic [EffAccW-1:0]        fifo_wdata;
  logic [EffAccW-1:0]        fifo_rdata;
  logic                      fifo_rvalid;
  logic [1:0]                fifo_depth_unused;

  // Stage 2 can advance whenever it is not pushing to a full output FIFO
  logic s1_can_advance;
  assign s1_can_advance = !clr_i && (!s1_valid_q || !s1_last_q || fifo_wready);
  assign in_ready_o     = s1_can_advance;

  always_comb begin
    dot_sum_s1 = '0;
    for (int unsigned k = 0; k < Lanes; k++) begin
      dot_sum_s1 = dot_sum_s1 + $signed({{(EffAccW-12){prod_q[k][11]}}, prod_q[k]});
    end
    next_acc_s1 = run_acc_q + dot_sum_s1;
  end

  assign fifo_wvalid = !clr_i && s1_valid_q && s1_last_q;
  assign fifo_wdata  = next_acc_s1;

  // Stage 1 + Stage 2 registers (synchronous active-low reset)
  always_ff @(posedge clk_i) begin
    if (!rst_ni || clr_i) begin
      s1_valid_q <= 1'b0;
      s1_last_q  <= 1'b0;
      run_acc_q  <= '0;
    end else begin
      if (s1_can_advance) begin
        s1_valid_q <= in_valid_i;
        s1_last_q  <= last_i;
        if (in_valid_i) begin
          for (int unsigned k = 0; k < Lanes; k++) begin
            prod_q[k] <= prod_d[k];
          end
        end
      end

      if (s1_valid_q && (!s1_last_q || fifo_wready)) begin
        if (s1_last_q) begin
          run_acc_q <= '0;
        end else begin
          run_acc_q <= next_acc_s1;
        end
      end
    end
  end

  prim_fifo_sync #(
    .Width (EffAccW),
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
  assign out_acc_o   = $signed(fifo_rdata[AccWidth-1:0]);

  logic unused_fifo_depth;
  assign unused_fifo_depth = ^fifo_depth_unused;

endmodule
