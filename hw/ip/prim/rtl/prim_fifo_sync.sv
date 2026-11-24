// Copyright 2026 CovalIR Contributors
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Technology-independent low-latency synchronous FIFO with valid/ready handshake.
// Uses combinational read (distributed LUT-RAM / flip-flop storage) intended for
// shallow decoupling/skid buffers (Depth = 2..64). For large BRAM/URAM memories,
// use prim_generic_sram_2p instead.
//
// Synchronous active-low reset (rst_ni) is used throughout CovalIR RTL so registers
// pack cleanly into Xilinx UltraScale+ DSP48E2/BRAM/URAM primitives as well as ASIC
// standard cells.

module prim_fifo_sync #(
  parameter int unsigned Width = 32,
  parameter int unsigned Depth = 16
) (
  input  logic                       clk_i,
  input  logic                       rst_ni,
  input  logic                       clr_i,

  // Write (upstream) latency-insensitive channel
  input  logic                       wvalid_i,
  output logic                       wready_o,
  input  logic [Width-1:0]           wdata_i,

  // Read (downstream) latency-insensitive channel
  output logic                       rvalid_o,
  input  logic                       rready_i,
  output logic [Width-1:0]           rdata_o,

  output logic [$clog2(Depth+1)-1:0] depth_o
);

  // Elaboration-time parameter checks (enforced by Verilator and Yosys)
  if (Width == 0) begin : gen_err_width
    $error("prim_fifo_sync: Width must be >= 1");
  end
  if ((Depth < 2) || ((Depth & (Depth - 1)) != 0)) begin : gen_err_depth
    $error("prim_fifo_sync: Depth (%0d) must be a power of two >= 2", Depth);
  end

  localparam int unsigned PtrW = (Depth >= 2) ? $clog2(Depth) : 1;
  localparam int unsigned CntW = $clog2(Depth + 1);

  logic [Width-1:0] mem_q [Depth];
  logic [PtrW-1:0]  wptr_q, wptr_d;
  logic [PtrW-1:0]  rptr_q, rptr_d;
  logic [CntW-1:0]  cnt_q,  cnt_d;

  logic push;
  logic pop;

  // Gate both handshakes during clr_i so no beat is acknowledged and silently dropped.
  // Note: wready_o has zero combinational dependency on rready_i (breaks ready chains).
  assign wready_o = !clr_i && (cnt_q < CntW'(Depth));
  assign rvalid_o = !clr_i && (cnt_q > CntW'(0));
  assign rdata_o  = mem_q[rptr_q];
  assign depth_o  = cnt_q;

  assign push = wvalid_i && wready_o;
  assign pop  = rvalid_o && rready_i;

  always_comb begin
    wptr_d = wptr_q;
    rptr_d = rptr_q;
    cnt_d  = cnt_q;

    if (clr_i) begin
      wptr_d = '0;
      rptr_d = '0;
      cnt_d  = '0;
    end else begin
      if (push) begin
        wptr_d = wptr_q + PtrW'(1);
      end
      if (pop) begin
        rptr_d = rptr_q + PtrW'(1);
      end
      case ({push, pop})
        2'b10:   cnt_d = cnt_q + CntW'(1);
        2'b01:   cnt_d = cnt_q - CntW'(1);
        default: cnt_d = cnt_q;
      endcase
    end
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      wptr_q <= '0;
      rptr_q <= '0;
      cnt_q  <= '0;
    end else begin
      wptr_q <= wptr_d;
      rptr_q <= rptr_d;
      cnt_q  <= cnt_d;
      if (push) begin
        mem_q[wptr_q] <= wdata_i;
      end
    end
  end

`ifdef FORMAL
  // Formal verification contract (SymbiYosys):
  // 1. Bounded count and exact pointer-to-count invariant
  // 2. Output valid/data stability while downstream is stalled
  // 3. No handshake acknowledged during clr_i
  // 4. Two-element arbitrary-value FIFO data integrity and ordering contract
  logic f_past_valid;
  initial f_past_valid = 1'b0;
  always @(posedge clk_i) f_past_valid <= 1'b1;

  // Assume reset is active on initial cycle (required for k-induction / prove mode)
  always_comb begin
    if (!f_past_valid) begin
      assume (!rst_ni);
    end
  end

  // Structural pointer/count invariants (valid after initial synchronous reset cycle)
  logic [PtrW-1:0] f_ptr_diff;
  assign f_ptr_diff = wptr_q - rptr_q;

  always_comb begin
    if (f_past_valid) begin
      assert (cnt_q <= CntW'(Depth));
      if (cnt_q == CntW'(0) || cnt_q == CntW'(Depth)) begin
        assert (wptr_q == rptr_q);
      end else begin
        assert (f_ptr_diff == cnt_q[PtrW-1:0]);
      end
      if (clr_i) begin
        assert (!wready_o);
        assert (!rvalid_o);
      end
    end
  end

  // Reset and output-stability assertions
  always @(posedge clk_i) begin
    if (f_past_valid && !$past(rst_ni)) begin
      assert (cnt_q == '0);
      assert (wptr_q == '0);
      assert (rptr_q == '0);
    end
    if (f_past_valid && $past(rst_ni) && !$past(clr_i) &&
        $past(rvalid_o) && !$past(rready_i) && rst_ni && !clr_i) begin
      assert (rvalid_o);
      assert (rdata_o == $past(rdata_o));
    end
  end

  // Classic 2-element FIFO ordering & data-integrity proof (k-inductive)
  (* anyconst *) logic [PtrW-1:0]  f_const_addr;
  (* anyconst *) logic [Width-1:0] f_const_data1;
  (* anyconst *) logic [Width-1:0] f_const_data2;
  logic [PtrW-1:0] f_const_addr_next;
  assign f_const_addr_next = f_const_addr + PtrW'(1);

  logic f_addr1_in_fifo, f_addr2_in_fifo;
  logic [PtrW-1:0] f_dist1, f_dist2;
  assign f_dist1 = f_const_addr - rptr_q;
  assign f_dist2 = f_const_addr_next - rptr_q;
  assign f_addr1_in_fifo = (cnt_q > CntW'(0)) && ({1'b0, f_dist1} < cnt_q);
  assign f_addr2_in_fifo = (cnt_q > CntW'(0)) && ({1'b0, f_dist2} < cnt_q);

  logic [1:0] f_state;
  initial f_state = 2'd0;

  always @(posedge clk_i) begin
    if (!rst_ni || clr_i) begin
      f_state <= 2'd0;
    end else begin
      case (f_state)
        2'd0: begin
          if (push && (wptr_q == f_const_addr) && (wdata_i == f_const_data1)) begin
            f_state <= 2'd1;
          end
        end
        2'd1: begin
          if (pop && (rptr_q == f_const_addr)) begin
            f_state <= 2'd0;
          end else if (push) begin
            if ((wptr_q == f_const_addr_next) && (wdata_i == f_const_data2)) begin
              f_state <= 2'd2;
            end else begin
              f_state <= 2'd0;
            end
          end
        end
        2'd2: begin
          if (pop && (rptr_q == f_const_addr)) begin
            f_state <= 2'd3;
          end
        end
        2'd3: begin
          if (pop && (rptr_q == f_const_addr_next)) begin
            f_state <= 2'd0;
          end
        end
        default: f_state <= 2'd0;
      endcase
    end
  end

  always_comb begin
    if (f_past_valid) begin
      case (f_state)
        2'd1: begin
          assert (cnt_q >= CntW'(1));
          assert (f_addr1_in_fifo);
          assert (mem_q[f_const_addr] == f_const_data1);
          assert (wptr_q == f_const_addr_next);
        end
        2'd2: begin
          assert (cnt_q >= CntW'(2));
          assert (f_addr1_in_fifo);
          assert (f_addr2_in_fifo);
          assert (f_dist1 < f_dist2);
          assert (mem_q[f_const_addr] == f_const_data1);
          assert (mem_q[f_const_addr_next] == f_const_data2);
        end
        2'd3: begin
          assert (cnt_q >= CntW'(1));
          assert (f_addr2_in_fifo);
          assert (rptr_q == f_const_addr_next);
          assert (mem_q[f_const_addr_next] == f_const_data2);
        end
        default: ;
      endcase
    end
  end

  // Cover interesting reachable states
  always @(posedge clk_i) begin
    if (f_past_valid && rst_ni) begin
      cover (cnt_q == CntW'(Depth));
      cover (push && pop);
      cover (f_state == 2'd3 && pop);
    end
  end
`endif

endmodule
