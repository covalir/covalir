// Copyright 2026 CovalIR Contributors
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Pipelined BitNet 1.58-bit ({-1, 0, +1}) x INT8 -> INT32 ZERO-MULTIPLIER engine.
//
// Encoding & Packing Contract:
// - Each ternary weight uses a 2-bit code (2x weight bandwidth reduction vs INT4;
//   5-trit/byte dense unpacking is not yet implemented):
//     2'b00 ->  0
//     2'b01 -> +1  (+a_lane)
//     2'b11 -> -1  (-a_lane, 9-bit signed so -(-128) = +128 is exact)
//     2'b10 ->  ILLEGAL (checked by simulation/formal assertion; hardware clamps to 0)
//
// Microarchitecture:
// - Stage 1 (registered): Lanes parallel sign-extended ternary add/sub/zero lane muxes.
// - Stage 2 (registered): Adder tree + running signed accumulator until last_i.
// - Output stage: 2-entry prim_fifo_sync decoupling buffer (zero combinational
//   out_ready_i -> in_ready_o path).
// - Synchronous active-low reset (rst_ni); clr_i gates in_ready_o.

module mac_ternary #(
  parameter int unsigned Lanes    = 32,
  parameter int unsigned AccWidth = 32
) (
  input  logic                       clk_i,
  input  logic                       rst_ni,
  input  logic                       clr_i,

  // Input stream: packed Lanes x 2-bit trits and Lanes x INT8 activations
  input  logic                       in_valid_i,
  output logic                       in_ready_o,
  input  logic [Lanes*2-1:0]         w_i,
  input  logic [Lanes*8-1:0]         a_i,
  input  logic                       last_i,

  // Output stream: completed signed INT32 dot-product result
  output logic                       out_valid_o,
  input  logic                       out_ready_i,
  output logic signed [AccWidth-1:0] out_acc_o
);

  if (Lanes == 0) begin : gen_err_lanes
    $error("mac_ternary: Lanes must be >= 1");
  end
  if (AccWidth < 9) begin : gen_err_accwidth
    $error("mac_ternary: AccWidth (%0d) must be >= 9 to hold signed -(-128)=+128", AccWidth);
  end

  localparam int unsigned EffAccW = (AccWidth >= 9) ? AccWidth : 9;

  // Stage 1: multiplier-free lane selection (combinational) -> registered term_q
  logic signed [8:0] term_d [Lanes];
  logic signed [8:0] term_q [Lanes];
  logic              s1_valid_q;
  logic              s1_last_q;

  for (genvar k = 0; k < Lanes; k++) begin : gen_lanes
    logic [1:0]        trit;
    logic signed [7:0] a_lane;
    logic signed [8:0] a_ext;

    assign trit   = w_i[k*2 +: 2];
    assign a_lane = $signed(a_i[k*8 +: 8]);
    assign a_ext  = {a_lane[7], a_lane};  // 9-bit signed so -(-128) = +128 never overflows

    always_comb begin
      case (trit)
        2'b01:   term_d[k] = a_ext;
        2'b11:   term_d[k] = -a_ext;
        default: term_d[k] = 9'sd0;
      endcase
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk_i) begin
      if (rst_ni && !clr_i && in_valid_i && in_ready_o) begin
        if (trit == 2'b10) begin
          $error("mac_ternary: illegal reserved trit encoding 2'b10 on lane %0d", k);
        end
      end
    end
`endif
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

  logic s1_can_advance;
  assign s1_can_advance = !clr_i && (!s1_valid_q || !s1_last_q || fifo_wready);
  assign in_ready_o     = s1_can_advance;

  always_comb begin
    dot_sum_s1 = '0;
    for (int unsigned k = 0; k < Lanes; k++) begin
      dot_sum_s1 = dot_sum_s1 + $signed({{(EffAccW-9){term_q[k][8]}}, term_q[k]});
    end
    next_acc_s1 = run_acc_q + dot_sum_s1;
  end

  assign fifo_wvalid = !clr_i && s1_valid_q && s1_last_q;
  assign fifo_wdata  = next_acc_s1;

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
            term_q[k] <= term_d[k];
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
