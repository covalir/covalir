// Copyright 2026 CovalIR Contributors
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Parameterized end-to-end pure-integer Llama-2 / TinyStories Transformer core.
//
// Instantiates verified CovalIR MAC IPs:
//   - mac_w4a8 (UseTernary == 0) or mac_ternary (UseTernary == 1)
//
// Pure-integer Q8.8 + Group-G W4A8/Ternary contract (matches golden_sim.py bit-for-bit,
// verified 16/16 exact token match vs FP32 on real Karpathy stories15M.bin):
//   - Q8.8 (INT16) residual stream & nonlinear activations (range +/-128.0)
//   - CLZ-normalized 256-entry inv_sqrt RMSNorm (exact across 8 orders of magnitude)
//   - Dynamic power-of-2 absmax INT8 activation quantizer before every linear GEMV
//   - Per-group-G INT4 / 2-bit ternary MACs scaled by Q1.15 per-group scales -> 48-bit acc
//   - Q1.14 RoPE, on-chip Q8.8 KV cache, causal MHA + 40-cycle restoring divider, Q8.8 SwiGLU

module covalir_llm_core #(
  parameter int unsigned Dim         = 32,
  parameter int unsigned HiddenDim   = 64,
  parameter int unsigned NLayers     = 2,
  parameter int unsigned NHeads      = 2,
  parameter int unsigned VocabSize   = 64,
  parameter int unsigned MaxSeqLen   = 32,
  parameter int unsigned Lanes       = 16,
  parameter int unsigned GroupSize   = 32,
  parameter bit          UseTernary  = 1'b0,
  parameter int unsigned InvDimQ16   = 2048,
  parameter int unsigned AttScaleQ12 = 16384,
  parameter              EmbHex      = "",
  parameter              RmsHex      = "",
  parameter              WeightHex   = "",
  parameter              ScaleHex    = "",
  parameter              InvRmsHex   = "",
  parameter              SiluHex     = "",
  parameter              ExpHex      = "",
  parameter              RopeCosHex  = "",
  parameter              RopeSinHex  = ""
) (
  input  logic               clk_i,
  input  logic               rst_ni,

  // Token step interface
  input  logic               start_i,
  input  logic [15:0]        token_id_i,
  input  logic [15:0]        pos_i,
  output logic               busy_o,
  output logic               done_o,
  output logic [15:0]        next_token_o,
  output logic signed [47:0] max_logit_o
);

  // Elaboration guards
  if ((Lanes == 0) || (GroupSize < Lanes) || ((GroupSize % Lanes) != 0)) begin : gen_err_grp
    $error("covalir_llm_core: GroupSize must be a positive multiple of Lanes");
  end
  if ((Dim == 0) || ((Dim % GroupSize) != 0) || ((HiddenDim % GroupSize) != 0)) begin : gen_err_dim
    $error("covalir_llm_core: Dim and HiddenDim must be positive multiples of GroupSize");
  end
  if ((NHeads == 0) || ((Dim % NHeads) != 0) ||
      (((Dim / NHeads) % 2) != 0)) begin : gen_err_heads
    $error("covalir_llm_core: Dim must be divisible by NHeads and HeadDim must be even");
  end

  localparam int unsigned HeadDim      = Dim / NHeads;
  localparam int unsigned HalfHeadDim  = HeadDim / 2;
  localparam int unsigned BeatsPerGrp  = GroupSize / Lanes;
  localparam int unsigned GrpsDim      = Dim / GroupSize;
  localparam int unsigned GrpsHidden   = HiddenDim / GroupSize;
  localparam int unsigned BeatsDim     = Dim / Lanes;
  localparam int unsigned BeatsHidden  = HiddenDim / Lanes;
  localparam int unsigned WBits        = UseTernary ? 2 : 4;
  localparam int unsigned WBeatBits    = Lanes * WBits;
  localparam int unsigned ABeatBits    = Lanes * 8;

  localparam int unsigned DimW         = (Dim > 1) ? $clog2(Dim) : 1;
  localparam int unsigned HidW         = (HiddenDim > 1) ? $clog2(HiddenDim) : 1;
  localparam int unsigned SeqW         = (MaxSeqLen > 1) ? $clog2(MaxSeqLen) : 1;

  // Per-layer weight beat counts in unified w_mem
  localparam int unsigned BeatsQ       = Dim * BeatsDim;
  localparam int unsigned BeatsK       = Dim * BeatsDim;
  localparam int unsigned BeatsV       = Dim * BeatsDim;
  localparam int unsigned BeatsO       = Dim * BeatsDim;
  localparam int unsigned BeatsW1      = HiddenDim * BeatsDim;
  localparam int unsigned BeatsW3      = HiddenDim * BeatsDim;
  localparam int unsigned BeatsW2      = Dim * BeatsHidden;
  localparam int unsigned BeatsPerLyr  = BeatsQ + BeatsK + BeatsV + BeatsO +
                                         BeatsW1 + BeatsW3 + BeatsW2;
  localparam int unsigned BeatsCls     = VocabSize * BeatsDim;
  localparam int unsigned TotalWBeats  = NLayers * BeatsPerLyr + BeatsCls;

  // Per-layer group scale counts in unified scale_mem (Q1.15 UINT16 per group, incl. WCLS)
  localparam int unsigned GrpsQ        = Dim * GrpsDim;
  localparam int unsigned GrpsK        = Dim * GrpsDim;
  localparam int unsigned GrpsV        = Dim * GrpsDim;
  localparam int unsigned GrpsO        = Dim * GrpsDim;
  localparam int unsigned GrpsW1       = HiddenDim * GrpsDim;
  localparam int unsigned GrpsW3       = HiddenDim * GrpsDim;
  localparam int unsigned GrpsW2       = Dim * GrpsHidden;
  localparam int unsigned GrpsPerLyr   = GrpsQ + GrpsK + GrpsV + GrpsO +
                                         GrpsW1 + GrpsW3 + GrpsW2;
  localparam int unsigned GrpsCls      = VocabSize * GrpsDim;
  localparam int unsigned TotalGScales = NLayers * GrpsPerLyr + GrpsCls;

  localparam int unsigned EmbEntries   = VocabSize * Dim;
  localparam int unsigned RmsEntries   = (2 * NLayers + 1) * Dim;
  localparam int unsigned KvEntries    = NLayers * MaxSeqLen * Dim;
  localparam int unsigned RopeEntries  = MaxSeqLen * HalfHeadDim;
  localparam int unsigned MaxInLen     = (HiddenDim > Dim) ? HiddenDim : Dim;
  localparam int unsigned QxW          = (MaxInLen > 1) ? $clog2(MaxInLen) : 1;

  // =========================================================================
  // On-Chip Parameter ROMs & Q8.8 Activation / KV-Cache SRAMs
  // =========================================================================
  logic signed [15:0]   emb_mem      [EmbEntries];
  logic signed [15:0]   rms_mem      [RmsEntries];
  logic [WBeatBits-1:0] w_mem        [TotalWBeats];
  logic [15:0]          scale_mem    [TotalGScales];
  logic [15:0]          inv_rms_lut  [256];
  logic signed [15:0]   silu_lut     [512];
  logic [7:0]           exp_lut      [256];
  logic signed [15:0]   rope_cos_mem [RopeEntries];
  logic signed [15:0]   rope_sin_mem [RopeEntries];

  logic signed [15:0]   k_cache_mem  [KvEntries];
  logic signed [15:0]   v_cache_mem  [KvEntries];

  // Q8.8 (INT16) scratchpad vectors + dynamic INT8 quantized input vector
  logic signed [15:0]   x_buf        [Dim];
  logic signed [15:0]   xb_buf       [Dim];
  logic signed [15:0]   q_buf        [Dim];
  logic signed [15:0]   k_buf        [Dim];
  logic signed [15:0]   v_buf        [Dim];
  logic signed [15:0]   att_buf      [Dim];
  logic signed [15:0]   hb_buf       [HiddenDim];
  logic signed [15:0]   hb2_buf      [HiddenDim];
  logic signed [7:0]    qx_buf       [MaxInLen];
  logic signed [31:0]   score_buf    [MaxSeqLen];
  logic [7:0]           ew_buf       [MaxSeqLen];

  initial begin
    if (EmbHex != "")     $readmemh(EmbHex, emb_mem);
    if (RmsHex != "")     $readmemh(RmsHex, rms_mem);
    if (WeightHex != "")  $readmemh(WeightHex, w_mem);
    if (ScaleHex != "")   $readmemh(ScaleHex, scale_mem);
    if (InvRmsHex != "")  $readmemh(InvRmsHex, inv_rms_lut);
    if (SiluHex != "")    $readmemh(SiluHex, silu_lut);
    if (ExpHex != "")     $readmemh(ExpHex, exp_lut);
    if (RopeCosHex != "") $readmemh(RopeCosHex, rope_cos_mem);
    if (RopeSinHex != "") $readmemh(RopeSinHex, rope_sin_mem);
  end

  // =========================================================================
  // Verified Compute IP Instantiation: MAC (W4A8 or Ternary)
  // =========================================================================
  logic                 mac_clr;
  logic                 mac_in_valid;
  logic                 mac_in_ready;
  logic [WBeatBits-1:0] mac_w_beat;
  logic [ABeatBits-1:0] mac_a_beat;
  logic                 mac_last;
  logic                 mac_out_valid;
  logic                 mac_out_ready;
  logic signed [31:0]   mac_out_acc;

  if (UseTernary) begin : gen_mac_ternary
    mac_ternary #(
      .Lanes    (Lanes),
      .AccWidth (32)
    ) u_mac (
      .clk_i       (clk_i),
      .rst_ni      (rst_ni),
      .clr_i       (mac_clr),
      .in_valid_i  (mac_in_valid),
      .in_ready_o  (mac_in_ready),
      .w_i         (mac_w_beat),
      .a_i         (mac_a_beat),
      .last_i      (mac_last),
      .out_valid_o (mac_out_valid),
      .out_ready_i (mac_out_ready),
      .out_acc_o   (mac_out_acc)
    );
  end else begin : gen_mac_w4a8
    mac_w4a8 #(
      .Lanes    (Lanes),
      .AccWidth (32)
    ) u_mac (
      .clk_i       (clk_i),
      .rst_ni      (rst_ni),
      .clr_i       (mac_clr),
      .in_valid_i  (mac_in_valid),
      .in_ready_o  (mac_in_ready),
      .w_i         (mac_w_beat),
      .a_i         (mac_a_beat),
      .last_i      (mac_last),
      .out_valid_o (mac_out_valid),
      .out_ready_i (mac_out_ready),
      .out_acc_o   (mac_out_acc)
    );
  end

  // =========================================================================
  // Pure-Verilog Clamp Helpers (Yosys-friendly: no SV return statements)
  // =========================================================================
  function automatic logic signed [7:0] clamp_i8(input logic signed [31:0] v);
    if (v > 32'sd127)       clamp_i8 = 8'sd127;
    else if (v < -32'sd128) clamp_i8 = -8'sd128;
    else                    clamp_i8 = v[7:0];
  endfunction

  function automatic logic signed [15:0] clamp_i16_48(input logic signed [47:0] v);
    if (v > 48'sd32767)       clamp_i16_48 = 16'sd32767;
    else if (v < -48'sd32768) clamp_i16_48 = -16'sd32768;
    else                      clamp_i16_48 = v[15:0];
  endfunction

  function automatic logic signed [15:0] clamp_i16_32(input logic signed [31:0] v);
    if (v > 32'sd32767)       clamp_i16_32 = 16'sd32767;
    else if (v < -32'sd32768) clamp_i16_32 = -16'sd32768;
    else                      clamp_i16_32 = v[15:0];
  endfunction

  // =========================================================================
  // Main Control FSM State & Registers
  // =========================================================================
  typedef enum logic [4:0] {
    ST_IDLE         = 5'd0,
    ST_EMB_LOAD     = 5'd1,
    ST_RMS_SUM      = 5'd2,
    ST_RMS_NORM     = 5'd3,
    ST_RMS_APPLY    = 5'd4,
    ST_DYN_MAX      = 5'd5,
    ST_DYN_QUANT    = 5'd6,
    ST_GEMV_FEED    = 5'd7,
    ST_GEMV_WAIT    = 5'd8,
    ST_ROPE         = 5'd9,
    ST_KV_STORE     = 5'd10,
    ST_ATT_DOT      = 5'd11,
    ST_ATT_EXP      = 5'd12,
    ST_ATT_WSUM     = 5'd13,
    ST_ATT_DIV_INIT = 5'd14,
    ST_ATT_DIV_STEP = 5'd15,
    ST_SWIGLU       = 5'd16,
    ST_DONE         = 5'd17
  } state_e;

  typedef enum logic [3:0] {
    PH_RMS_ATT  = 4'd0,
    PH_PROJ_Q   = 4'd1,
    PH_PROJ_K   = 4'd2,
    PH_PROJ_V   = 4'd3,
    PH_PROJ_O   = 4'd4,
    PH_RMS_FFN  = 4'd5,
    PH_PROJ_W1  = 4'd6,
    PH_PROJ_W3  = 4'd7,
    PH_PROJ_W2  = 4'd8,
    PH_RMS_FIN  = 4'd9,
    PH_PROJ_CLS = 4'd10
  } phase_e;

  state_e      state_q;
  phase_e      phase_q;

  logic [15:0] tok_q;
  logic [15:0] pos_q;
  logic [15:0] layer_q;
  logic [15:0] idx_q;
  logic [15:0] grp_q;
  logic [15:0] beat_q;
  logic [15:0] head_q;
  logic [15:0] t_q;
  logic [15:0] d_q;

  // GEMV config & dynamic INT8 quantizer registers
  logic [31:0]        gemv_w_base_q;
  logic [31:0]        gemv_g_base_q;
  logic [15:0]        gemv_rows_q;
  logic [15:0]        gemv_grps_q;
  logic [15:0]        gemv_in_len_q;
  logic [1:0]         gemv_src_q;  // 0=xb_buf, 1=att_buf, 2=hb_buf
  logic               gemv_raw_q;  // 1=LM head raw 48-bit argmax
  logic [15:0]        dyn_max_q;
  logic [3:0]         dyn_e_q;     // 0..9
  logic signed [47:0] row_acc_q;

  // CLZ-normalized RMSNorm registers
  logic [47:0] rms_sum_sq_q;
  logic [47:0] rms_msq_q;
  logic [4:0]  rms_k_q;
  logic [15:0] rms_inv_q;
  logic [4:0]  rms_sh_q;
  logic [31:0] rms_base_q;

  // Attention & 40-bit exact restoring divider registers
  logic signed [47:0] att_dot_q;
  logic signed [31:0] att_max_s_q;
  logic [31:0]        att_sum_ew_q;
  logic signed [39:0] att_wsum_q;
  logic               div_neg_q;
  logic [39:0]        div_num_q;
  logic [39:0]        div_den_q;
  logic [39:0]        div_rem_q;
  logic [39:0]        div_quo_q;
  logic [5:0]         div_cnt_q;

  logic [15:0]        best_tok_q;
  logic signed [47:0] best_logit_q;
  logic               done_q;

  assign busy_o       = (state_q != ST_IDLE);
  assign done_o       = done_q;
  assign next_token_o = best_tok_q;
  assign max_logit_o  = best_logit_q;
  assign mac_clr      = 1'b0;

  // Exact bit-width index wires
  logic [DimW-1:0] idx_dim;
  logic [HidW-1:0] idx_hid;
  logic [QxW-1:0]  idx_qx;
  logic [SeqW-1:0] t_seq;
  assign idx_dim = idx_q[DimW-1:0];
  assign idx_hid = idx_q[HidW-1:0];
  assign idx_qx  = idx_q[QxW-1:0];
  assign t_seq   = t_q[SeqW-1:0];

  // Pack dynamic INT8 activation beat for current (grp_q, beat_q)
  logic [31:0] a_elem_idx;
  always_comb begin
    mac_a_beat = '0;
    a_elem_idx = '0;
    for (int unsigned k = 0; k < Lanes; k++) begin
      a_elem_idx = (32'(grp_q) * 32'(GroupSize)) + (32'(beat_q) * 32'(Lanes)) + k;
      if (a_elem_idx < MaxInLen) begin
        mac_a_beat[k*8 +: 8] = qx_buf[a_elem_idx[QxW-1:0]];
      end
    end
  end

  // Weight beat & per-group Q1.15 scale lookup
  logic [31:0] cur_w_addr;
  logic [31:0] cur_g_addr;
  logic [15:0] cur_scale_q15;

  always_comb begin
    cur_w_addr = gemv_w_base_q +
                 (32'(idx_q) * 32'(gemv_grps_q) * 32'(BeatsPerGrp)) +
                 (32'(grp_q) * 32'(BeatsPerGrp)) +
                 32'(beat_q);
    cur_g_addr = gemv_g_base_q + (32'(idx_q) * 32'(gemv_grps_q)) + 32'(grp_q);
    if (cur_w_addr < TotalWBeats) begin
      mac_w_beat = w_mem[cur_w_addr];
    end else begin
      mac_w_beat = '0;
    end
    if (cur_g_addr < TotalGScales) begin
      cur_scale_q15 = scale_mem[cur_g_addr];
    end else begin
      cur_scale_q15 = 16'd0;
    end
  end

  assign mac_in_valid  = (state_q == ST_GEMV_FEED);
  assign mac_last      = (beat_q == 16'(BeatsPerGrp - 1));
  assign mac_out_ready = (state_q == ST_GEMV_WAIT);

  // =========================================================================
  // Combinational Datapath Helpers (Yosys-friendly)
  // =========================================================================
  // 1. RMSNorm datapath
  logic signed [31:0] c_rms_x32;
  logic [47:0]        c_rms_sq;
  logic [47:0]        c_rms_tot_sq;
  logic [63:0]        c_rms_msq_mul;
  logic [47:0]        c_rms_msq_init;
  logic signed [47:0] c_rms_xi48;
  logic signed [47:0] c_rms_wi48;
  logic signed [47:0] c_rms_inv48;
  logic signed [47:0] c_rms_prod48;
  logic signed [47:0] c_rms_bias48;
  logic signed [15:0] c_rms_out16;

  always_comb begin
    c_rms_x32      = {{16{x_buf[idx_dim][15]}}, x_buf[idx_dim]};
    c_rms_sq       = 48'($signed(c_rms_x32) * $signed(c_rms_x32));
    c_rms_tot_sq   = rms_sum_sq_q + c_rms_sq;
    c_rms_msq_mul  = {16'd0, c_rms_tot_sq} * 64'(InvDimQ16);
    c_rms_msq_init = (c_rms_msq_mul[63:16] == 48'd0) ? 48'd1 : c_rms_msq_mul[63:16];

    c_rms_xi48     = {{32{x_buf[idx_dim][15]}}, x_buf[idx_dim]};
    c_rms_wi48     = {{32{rms_mem[rms_base_q + 32'(idx_q)][15]}},
                      rms_mem[rms_base_q + 32'(idx_q)]};
    c_rms_inv48    = $signed({32'd0, rms_inv_q});
    c_rms_prod48   = c_rms_xi48 * c_rms_inv48 * c_rms_wi48;
    c_rms_bias48   = 48'sd1 <<< (rms_sh_q - 5'd1);
    c_rms_out16    = clamp_i16_48((c_rms_prod48 + c_rms_bias48) >>> rms_sh_q);
  end

  // 2. Dynamic power-of-2 INT8 quantizer + Per-group GEMV output scaling
  logic signed [15:0] c_dyn_val;
  logic [15:0]        c_dyn_abs;
  logic [15:0]        c_dyn_max_next;
  logic [3:0]         c_dyn_e_calc;
  logic signed [31:0] c_dyn_val32;
  logic signed [31:0] c_dyn_bias32;
  logic signed [7:0]  c_dyn_q8;

  logic signed [47:0] c_grp_prod48;
  logic signed [47:0] c_row_acc_next;
  logic [4:0]         c_gemv_sh;
  logic signed [47:0] c_gemv_bias48;
  logic signed [15:0] c_gemv_out16;
  logic signed [31:0] c_res_sum32;

  always_comb begin
    if (gemv_src_q == 2'd0)      c_dyn_val = xb_buf[idx_dim];
    else if (gemv_src_q == 2'd1) c_dyn_val = att_buf[idx_dim];
    else                         c_dyn_val = hb_buf[idx_hid];

    c_dyn_abs      = c_dyn_val[15] ? (~c_dyn_val + 16'd1) : c_dyn_val;
    c_dyn_max_next = (c_dyn_abs > dyn_max_q) ? c_dyn_abs : dyn_max_q;

    if (c_dyn_max_next <= 16'd127)        c_dyn_e_calc = 4'd0;
    else if (c_dyn_max_next <= 16'd255)   c_dyn_e_calc = 4'd1;
    else if (c_dyn_max_next <= 16'd511)   c_dyn_e_calc = 4'd2;
    else if (c_dyn_max_next <= 16'd1023)  c_dyn_e_calc = 4'd3;
    else if (c_dyn_max_next <= 16'd2047)  c_dyn_e_calc = 4'd4;
    else if (c_dyn_max_next <= 16'd4095)  c_dyn_e_calc = 4'd5;
    else if (c_dyn_max_next <= 16'd8191)  c_dyn_e_calc = 4'd6;
    else if (c_dyn_max_next <= 16'd16383) c_dyn_e_calc = 4'd7;
    else if (c_dyn_max_next <= 16'd32767) c_dyn_e_calc = 4'd8;
    else                                  c_dyn_e_calc = 4'd9;

    c_dyn_val32  = {{16{c_dyn_val[15]}}, c_dyn_val};
    c_dyn_bias32 = (dyn_e_q == 4'd0) ? 32'sd0 : (32'sd1 <<< (dyn_e_q - 4'd1));
    c_dyn_q8     = (dyn_e_q == 4'd0) ? clamp_i8(c_dyn_val32) :
                   clamp_i8((c_dyn_val32 + c_dyn_bias32) >>> dyn_e_q);

    c_grp_prod48   = {{16{mac_out_acc[31]}}, mac_out_acc} * $signed({32'd0, cur_scale_q15});
    c_row_acc_next = row_acc_q + c_grp_prod48;
    c_gemv_sh      = 5'd15 - {1'b0, dyn_e_q};
    c_gemv_bias48  = 48'sd1 <<< (c_gemv_sh - 5'd1);
    c_gemv_out16   = clamp_i16_48((c_row_acc_next + c_gemv_bias48) >>> c_gemv_sh);
    c_res_sum32    = {{16{x_buf[idx_dim][15]}}, x_buf[idx_dim]} +
                     {{16{c_gemv_out16[15]}}, c_gemv_out16};
  end

  // 3. Q1.14 RoPE datapath
  logic [31:0]        c_rope_addr;
  logic [15:0]        c_rope_e0_16;
  logic [15:0]        c_rope_e1_16;
  logic [DimW-1:0]    c_rope_e0;
  logic [DimW-1:0]    c_rope_e1;
  logic signed [31:0] c_rc, c_rs, c_qa, c_qb, c_ka, c_kb;
  logic signed [15:0] c_q0_out, c_q1_out, c_k0_out, c_k1_out;

  always_comb begin
    c_rope_addr  = 32'(pos_q) * 32'(HalfHeadDim) + 32'(d_q);
    c_rope_e0_16 = head_q * 16'(HeadDim) + (d_q << 1);
    c_rope_e1_16 = c_rope_e0_16 + 16'd1;
    c_rope_e0    = c_rope_e0_16[DimW-1:0];
    c_rope_e1    = c_rope_e1_16[DimW-1:0];
    c_rc         = {{16{rope_cos_mem[c_rope_addr][15]}}, rope_cos_mem[c_rope_addr]};
    c_rs         = {{16{rope_sin_mem[c_rope_addr][15]}}, rope_sin_mem[c_rope_addr]};
    c_qa         = {{16{q_buf[c_rope_e0][15]}}, q_buf[c_rope_e0]};
    c_qb         = {{16{q_buf[c_rope_e1][15]}}, q_buf[c_rope_e1]};
    c_ka         = {{16{k_buf[c_rope_e0][15]}}, k_buf[c_rope_e0]};
    c_kb         = {{16{k_buf[c_rope_e1][15]}}, k_buf[c_rope_e1]};
    c_q0_out     = clamp_i16_32((c_qa * c_rc - c_qb * c_rs + 32'sd8192) >>> 14);
    c_q1_out     = clamp_i16_32((c_qa * c_rs + c_qb * c_rc + 32'sd8192) >>> 14);
    c_k0_out     = clamp_i16_32((c_ka * c_rc - c_kb * c_rs + 32'sd8192) >>> 14);
    c_k1_out     = clamp_i16_32((c_ka * c_rs + c_kb * c_rc + 32'sd8192) >>> 14);
  end

  // 4. Q8.8 Attention + 40-bit Restoring Divider + Q8.8 SwiGLU datapaths
  logic [31:0]        c_kv_store_addr;
  logic [15:0]        c_hd_idx16;
  logic [DimW-1:0]    c_hd_dim;
  logic [31:0]        c_kv_read_addr;
  logic signed [47:0] c_qv48, c_kv48, c_dot_next48, c_att_sc48;
  logic signed [31:0] c_att_sc32;
  logic signed [31:0] c_exp_diff;
  logic [7:0]         c_exp_idx, c_exp_val;
  logic signed [39:0] c_ev40, c_vv40, c_wsum_next40;
  logic [40:0]        c_div_shift;
  logic [39:0]        c_div_next_rem, c_div_next_quo;
  logic signed [47:0] c_div_signed_quo;
  logic signed [15:0] c_silu_shift;
  logic signed [15:0] c_silu_idx16;
  logic [8:0]         c_silu_idx;
  logic signed [31:0] c_sv32, c_u32;
  logic signed [15:0] c_swiglu_out;

  always_comb begin
    c_kv_store_addr = ((32'(layer_q) * 32'(MaxSeqLen) + 32'(pos_q)) * 32'(Dim)) + 32'(idx_q);
    c_hd_idx16      = head_q * 16'(HeadDim) + d_q;
    c_hd_dim        = c_hd_idx16[DimW-1:0];
    c_kv_read_addr  = ((32'(layer_q) * 32'(MaxSeqLen) + 32'(t_q)) * 32'(Dim)) + 32'(c_hd_idx16);

    c_qv48          = {{32{q_buf[c_hd_dim][15]}}, q_buf[c_hd_dim]};
    c_kv48          = {{32{k_cache_mem[c_kv_read_addr][15]}}, k_cache_mem[c_kv_read_addr]};
    c_dot_next48    = att_dot_q + (c_qv48 * c_kv48);
    c_att_sc48      = ((c_dot_next48 * $signed({16'd0, 32'(AttScaleQ12)})) +
                       48'sd134217728) >>> 28;
    c_att_sc32      = c_att_sc48[31:0];

    c_exp_diff      = att_max_s_q - score_buf[t_seq];
    c_exp_idx       = (c_exp_diff > 32'sd255) ? 8'd255 :
                      ((c_exp_diff < 32'sd0) ? 8'd0 : c_exp_diff[7:0]);
    c_exp_val       = exp_lut[c_exp_idx];

    c_ev40          = $signed({32'd0, ew_buf[t_seq]});
    c_vv40          = {{24{v_cache_mem[c_kv_read_addr][15]}}, v_cache_mem[c_kv_read_addr]};
    c_wsum_next40   = att_wsum_q + (c_ev40 * c_vv40);

    c_div_shift     = {div_rem_q, div_num_q[39]};
    if (c_div_shift >= {1'b0, div_den_q}) begin
      c_div_next_rem = c_div_shift[39:0] - div_den_q;
      c_div_next_quo = {div_quo_q[38:0], 1'b1};
    end else begin
      c_div_next_rem = c_div_shift[39:0];
      c_div_next_quo = {div_quo_q[38:0], 1'b0};
    end
    c_div_signed_quo = div_neg_q ? -$signed({8'd0, c_div_next_quo}) :
                                    $signed({8'd0, c_div_next_quo});

    c_silu_shift     = hb_buf[idx_hid] >>> 4;
    c_silu_idx16     = c_silu_shift + 16'sd256;
    c_silu_idx       = (c_silu_idx16 < 16'sd0) ? 9'd0 :
                       ((c_silu_idx16 > 16'sd511) ? 9'd511 : c_silu_idx16[8:0]);
    c_sv32           = {{16{silu_lut[c_silu_idx][15]}}, silu_lut[c_silu_idx]};
    c_u32            = {{16{hb2_buf[idx_hid][15]}}, hb2_buf[idx_hid]};
    c_swiglu_out     = clamp_i16_32((c_sv32 * c_u32 + 32'sd128) >>> 8);
  end

  // =========================================================================
  // Sequential Control FSM
  // =========================================================================
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      state_q       <= ST_IDLE;
      phase_q       <= PH_RMS_ATT;
      tok_q         <= '0;
      pos_q         <= '0;
      layer_q       <= '0;
      idx_q         <= '0;
      grp_q         <= '0;
      beat_q        <= '0;
      head_q        <= '0;
      t_q           <= '0;
      d_q           <= '0;
      gemv_w_base_q <= '0;
      gemv_g_base_q <= '0;
      gemv_rows_q   <= '0;
      gemv_grps_q   <= '0;
      gemv_in_len_q <= '0;
      gemv_src_q    <= '0;
      gemv_raw_q    <= 1'b0;
      dyn_max_q     <= '0;
      dyn_e_q       <= '0;
      row_acc_q     <= '0;
      rms_sum_sq_q  <= '0;
      rms_msq_q     <= '0;
      rms_k_q       <= '0;
      rms_inv_q     <= '0;
      rms_sh_q      <= 5'd15;
      rms_base_q    <= '0;
      att_dot_q     <= '0;
      att_max_s_q   <= '0;
      att_sum_ew_q  <= '0;
      att_wsum_q    <= '0;
      div_neg_q     <= 1'b0;
      div_num_q     <= '0;
      div_den_q     <= '0;
      div_rem_q     <= '0;
      div_quo_q     <= '0;
      div_cnt_q     <= '0;
      best_tok_q    <= '0;
      best_logit_q  <= '0;
      done_q        <= 1'b0;
    end else begin
      done_q <= 1'b0;

      case (state_q)
        ST_IDLE: begin
          if (start_i) begin
            tok_q   <= token_id_i;
            pos_q   <= pos_i;
            layer_q <= 16'd0;
            idx_q   <= 16'd0;
            state_q <= ST_EMB_LOAD;
          end
        end

        ST_EMB_LOAD: begin
          x_buf[idx_dim] <= emb_mem[32'(tok_q) * 32'(Dim) + 32'(idx_q)];
          if (idx_q == 16'(Dim - 1)) begin
            idx_q        <= 16'd0;
            rms_sum_sq_q <= 48'd0;
            rms_base_q   <= 32'd0;
            phase_q      <= PH_RMS_ATT;
            state_q      <= ST_RMS_SUM;
          end else begin
            idx_q <= idx_q + 16'd1;
          end
        end

        ST_RMS_SUM: begin
          rms_sum_sq_q <= c_rms_tot_sq;
          if (idx_q == 16'(Dim - 1)) begin
            rms_msq_q <= c_rms_msq_init;
            rms_k_q   <= 5'd0;
            idx_q     <= 16'd0;
            state_q   <= ST_RMS_NORM;
          end else begin
            idx_q <= idx_q + 16'd1;
          end
        end

        ST_RMS_NORM: begin
          if (rms_msq_q >= 48'd256) begin
            rms_msq_q <= rms_msq_q >> 2;
            rms_k_q   <= rms_k_q + 5'd1;
          end else begin
            rms_inv_q <= inv_rms_lut[rms_msq_q[7:0]];
            rms_sh_q  <= 5'd15 + rms_k_q;
            idx_q     <= 16'd0;
            state_q   <= ST_RMS_APPLY;
          end
        end

        ST_RMS_APPLY: begin
          xb_buf[idx_dim] <= c_rms_out16;
          if (idx_q == 16'(Dim - 1)) begin
            idx_q     <= 16'd0;
            dyn_max_q <= 16'd0;
            if (phase_q == PH_RMS_ATT) begin
              phase_q       <= PH_PROJ_Q;
              gemv_w_base_q <= 32'(layer_q) * 32'(BeatsPerLyr);
              gemv_g_base_q <= 32'(layer_q) * 32'(GrpsPerLyr);
              gemv_rows_q   <= 16'(Dim);
              gemv_grps_q   <= 16'(GrpsDim);
              gemv_in_len_q <= 16'(Dim);
              gemv_src_q    <= 2'd0;
              gemv_raw_q    <= 1'b0;
              state_q       <= ST_DYN_MAX;
            end else if (phase_q == PH_RMS_FFN) begin
              phase_q       <= PH_PROJ_W1;
              gemv_w_base_q <= 32'(layer_q) * 32'(BeatsPerLyr) +
                               32'(BeatsQ + BeatsK + BeatsV + BeatsO);
              gemv_g_base_q <= 32'(layer_q) * 32'(GrpsPerLyr) +
                               32'(GrpsQ + GrpsK + GrpsV + GrpsO);
              gemv_rows_q   <= 16'(HiddenDim);
              gemv_grps_q   <= 16'(GrpsDim);
              gemv_in_len_q <= 16'(Dim);
              gemv_src_q    <= 2'd0;
              gemv_raw_q    <= 1'b0;
              state_q       <= ST_DYN_MAX;
            end else begin
              phase_q       <= PH_PROJ_CLS;
              gemv_w_base_q <= 32'(NLayers * BeatsPerLyr);
              gemv_g_base_q <= 32'(NLayers * GrpsPerLyr);
              gemv_rows_q   <= 16'(VocabSize);
              gemv_grps_q   <= 16'(GrpsDim);
              gemv_in_len_q <= 16'(Dim);
              gemv_src_q    <= 2'd0;
              gemv_raw_q    <= 1'b1;
              best_tok_q    <= 16'd0;
              best_logit_q  <= $signed(48'h8000_0000_0000);
              state_q       <= ST_DYN_MAX;
            end
          end else begin
            idx_q <= idx_q + 16'd1;
          end
        end

        ST_DYN_MAX: begin
          dyn_max_q <= c_dyn_max_next;
          if (idx_q == (gemv_in_len_q - 16'd1)) begin
            dyn_e_q <= c_dyn_e_calc;
            idx_q   <= 16'd0;
            state_q <= ST_DYN_QUANT;
          end else begin
            idx_q <= idx_q + 16'd1;
          end
        end

        ST_DYN_QUANT: begin
          qx_buf[idx_qx] <= c_dyn_q8;
          if (idx_q == (gemv_in_len_q - 16'd1)) begin
            idx_q     <= 16'd0;
            grp_q     <= 16'd0;
            beat_q    <= 16'd0;
            row_acc_q <= 48'sd0;
            state_q   <= ST_GEMV_FEED;
          end else begin
            idx_q <= idx_q + 16'd1;
          end
        end

        ST_GEMV_FEED: begin
          if (mac_in_ready) begin
            if (beat_q == 16'(BeatsPerGrp - 1)) begin
              beat_q  <= 16'd0;
              state_q <= ST_GEMV_WAIT;
            end else begin
              beat_q <= beat_q + 16'd1;
            end
          end
        end

        ST_GEMV_WAIT: begin
          if (mac_out_valid) begin
            if (grp_q == (gemv_grps_q - 16'd1)) begin
              // Row completed across all groups!
              row_acc_q <= 48'sd0;
              grp_q     <= 16'd0;
              beat_q    <= 16'd0;

              if (gemv_raw_q) begin
                if (idx_q == 16'd0 || c_row_acc_next > best_logit_q) begin
                  best_logit_q <= c_row_acc_next;
                  best_tok_q   <= idx_q;
                end
                if (idx_q == (gemv_rows_q - 16'd1)) begin
                  state_q <= ST_DONE;
                end else begin
                  idx_q   <= idx_q + 16'd1;
                  state_q <= ST_GEMV_FEED;
                end
              end else begin
                case (phase_q)
                  PH_PROJ_Q:  q_buf[idx_dim]   <= c_gemv_out16;
                  PH_PROJ_K:  k_buf[idx_dim]   <= c_gemv_out16;
                  PH_PROJ_V:  v_buf[idx_dim]   <= c_gemv_out16;
                  PH_PROJ_O:  x_buf[idx_dim]   <= clamp_i16_32(c_res_sum32);
                  PH_PROJ_W1: hb_buf[idx_hid]  <= c_gemv_out16;
                  PH_PROJ_W3: hb2_buf[idx_hid] <= c_gemv_out16;
                  PH_PROJ_W2: x_buf[idx_dim]   <= clamp_i16_32(c_res_sum32);
                  default: ;
                endcase

                if (idx_q == (gemv_rows_q - 16'd1)) begin
                  idx_q <= 16'd0;
                  case (phase_q)
                    PH_PROJ_Q: begin
                      // Reuse qx_buf & dyn_e_q (same xb_buf input!)
                      phase_q       <= PH_PROJ_K;
                      gemv_w_base_q <= 32'(layer_q) * 32'(BeatsPerLyr) + 32'(BeatsQ);
                      gemv_g_base_q <= 32'(layer_q) * 32'(GrpsPerLyr) + 32'(GrpsQ);
                      gemv_rows_q   <= 16'(Dim);
                      gemv_grps_q   <= 16'(GrpsDim);
                      state_q       <= ST_GEMV_FEED;
                    end
                    PH_PROJ_K: begin
                      phase_q       <= PH_PROJ_V;
                      gemv_w_base_q <= 32'(layer_q) * 32'(BeatsPerLyr) + 32'(BeatsQ + BeatsK);
                      gemv_g_base_q <= 32'(layer_q) * 32'(GrpsPerLyr) + 32'(GrpsQ + GrpsK);
                      gemv_rows_q   <= 16'(Dim);
                      gemv_grps_q   <= 16'(GrpsDim);
                      state_q       <= ST_GEMV_FEED;
                    end
                    PH_PROJ_V: begin
                      head_q  <= 16'd0;
                      d_q     <= 16'd0;
                      state_q <= ST_ROPE;
                    end
                    PH_PROJ_O: begin
                      rms_sum_sq_q <= 48'd0;
                      rms_base_q   <= (32'(layer_q) * 32'd2 + 32'd1) * 32'(Dim);
                      phase_q      <= PH_RMS_FFN;
                      state_q      <= ST_RMS_SUM;
                    end
                    PH_PROJ_W1: begin
                      // Reuse qx_buf & dyn_e_q (same xb_buf input!)
                      phase_q       <= PH_PROJ_W3;
                      gemv_w_base_q <= 32'(layer_q) * 32'(BeatsPerLyr) +
                                       32'(BeatsQ + BeatsK + BeatsV + BeatsO + BeatsW1);
                      gemv_g_base_q <= 32'(layer_q) * 32'(GrpsPerLyr) +
                                       32'(GrpsQ + GrpsK + GrpsV + GrpsO + GrpsW1);
                      gemv_rows_q   <= 16'(HiddenDim);
                      gemv_grps_q   <= 16'(GrpsDim);
                      state_q       <= ST_GEMV_FEED;
                    end
                    PH_PROJ_W3: begin
                      idx_q   <= 16'd0;
                      state_q <= ST_SWIGLU;
                    end
                    PH_PROJ_W2: begin
                      if (layer_q == 16'(NLayers - 1)) begin
                        rms_sum_sq_q <= 48'd0;
                        rms_base_q   <= 32'(2 * NLayers * Dim);
                        phase_q      <= PH_RMS_FIN;
                        state_q      <= ST_RMS_SUM;
                      end else begin
                        layer_q      <= layer_q + 16'd1;
                        rms_sum_sq_q <= 48'd0;
                        rms_base_q   <= (32'(layer_q + 16'd1) * 32'd2) * 32'(Dim);
                        phase_q      <= PH_RMS_ATT;
                        state_q      <= ST_RMS_SUM;
                      end
                    end
                    default: state_q <= ST_IDLE;
                  endcase
                end else begin
                  idx_q   <= idx_q + 16'd1;
                  state_q <= ST_GEMV_FEED;
                end
              end
            end else begin
              row_acc_q <= c_row_acc_next;
              grp_q     <= grp_q + 16'd1;
              beat_q    <= 16'd0;
              state_q   <= ST_GEMV_FEED;
            end
          end
        end

        ST_ROPE: begin
          q_buf[c_rope_e0] <= c_q0_out;
          q_buf[c_rope_e1] <= c_q1_out;
          k_buf[c_rope_e0] <= c_k0_out;
          k_buf[c_rope_e1] <= c_k1_out;
          if (d_q == 16'(HalfHeadDim - 1)) begin
            d_q <= 16'd0;
            if (head_q == 16'(NHeads - 1)) begin
              head_q  <= 16'd0;
              idx_q   <= 16'd0;
              state_q <= ST_KV_STORE;
            end else begin
              head_q <= head_q + 16'd1;
            end
          end else begin
            d_q <= d_q + 16'd1;
          end
        end

        ST_KV_STORE: begin
          k_cache_mem[c_kv_store_addr] <= k_buf[idx_dim];
          v_cache_mem[c_kv_store_addr] <= v_buf[idx_dim];
          if (idx_q == 16'(Dim - 1)) begin
            idx_q       <= 16'd0;
            head_q      <= 16'd0;
            t_q         <= 16'd0;
            d_q         <= 16'd0;
            att_dot_q   <= 48'sd0;
            att_max_s_q <= $signed(32'h8000_0000);
            state_q     <= ST_ATT_DOT;
          end else begin
            idx_q <= idx_q + 16'd1;
          end
        end

        ST_ATT_DOT: begin
          if (d_q == 16'(HeadDim - 1)) begin
            score_buf[t_seq] <= c_att_sc32;
            if (t_q == 16'd0 || c_att_sc32 > att_max_s_q) begin
              att_max_s_q <= c_att_sc32;
            end
            att_dot_q <= 48'sd0;
            d_q       <= 16'd0;
            if (t_q == pos_q) begin
              t_q          <= 16'd0;
              att_sum_ew_q <= 32'd0;
              state_q      <= ST_ATT_EXP;
            end else begin
              t_q <= t_q + 16'd1;
            end
          end else begin
            att_dot_q <= c_dot_next48;
            d_q       <= d_q + 16'd1;
          end
        end

        ST_ATT_EXP: begin
          ew_buf[t_seq] <= c_exp_val;
          att_sum_ew_q  <= att_sum_ew_q + 32'(c_exp_val);
          if (t_q == pos_q) begin
            t_q        <= 16'd0;
            d_q        <= 16'd0;
            att_wsum_q <= 40'sd0;
            state_q    <= ST_ATT_WSUM;
          end else begin
            t_q <= t_q + 16'd1;
          end
        end

        ST_ATT_WSUM: begin
          att_wsum_q <= c_wsum_next40;
          if (t_q == pos_q) begin
            t_q     <= 16'd0;
            state_q <= ST_ATT_DIV_INIT;
          end else begin
            t_q <= t_q + 16'd1;
          end
        end

        ST_ATT_DIV_INIT: begin
          div_neg_q <= att_wsum_q[39];
          div_num_q <= att_wsum_q[39] ? (~att_wsum_q + 40'd1) : att_wsum_q;
          div_den_q <= {8'd0, att_sum_ew_q};
          div_rem_q <= 40'd0;
          div_quo_q <= 40'd0;
          div_cnt_q <= 6'd40;
          state_q   <= ST_ATT_DIV_STEP;
        end

        ST_ATT_DIV_STEP: begin
          div_rem_q <= c_div_next_rem;
          div_quo_q <= c_div_next_quo;
          div_num_q <= {div_num_q[38:0], 1'b0};
          div_cnt_q <= div_cnt_q - 6'd1;

          if (div_cnt_q == 6'd1) begin
            att_buf[c_hd_dim] <= clamp_i16_48(c_div_signed_quo);
            att_wsum_q        <= 40'sd0;
            if (d_q == 16'(HeadDim - 1)) begin
              d_q <= 16'd0;
              if (head_q == 16'(NHeads - 1)) begin
                head_q        <= 16'd0;
                idx_q         <= 16'd0;
                dyn_max_q     <= 16'd0;
                phase_q       <= PH_PROJ_O;
                gemv_w_base_q <= 32'(layer_q) * 32'(BeatsPerLyr) +
                                 32'(BeatsQ + BeatsK + BeatsV);
                gemv_g_base_q <= 32'(layer_q) * 32'(GrpsPerLyr) +
                                 32'(GrpsQ + GrpsK + GrpsV);
                gemv_rows_q   <= 16'(Dim);
                gemv_grps_q   <= 16'(GrpsDim);
                gemv_in_len_q <= 16'(Dim);
                gemv_src_q    <= 2'd1;
                gemv_raw_q    <= 1'b0;
                state_q       <= ST_DYN_MAX;
              end else begin
                head_q      <= head_q + 16'd1;
                t_q         <= 16'd0;
                att_dot_q   <= 48'sd0;
                att_max_s_q <= $signed(32'h8000_0000);
                state_q     <= ST_ATT_DOT;
              end
            end else begin
              d_q     <= d_q + 16'd1;
              t_q     <= 16'd0;
              state_q <= ST_ATT_WSUM;
            end
          end
        end

        ST_SWIGLU: begin
          hb_buf[idx_hid] <= c_swiglu_out;
          if (idx_q == 16'(HiddenDim - 1)) begin
            idx_q         <= 16'd0;
            dyn_max_q     <= 16'd0;
            phase_q       <= PH_PROJ_W2;
            gemv_w_base_q <= 32'(layer_q) * 32'(BeatsPerLyr) +
                             32'(BeatsQ + BeatsK + BeatsV + BeatsO + BeatsW1 + BeatsW3);
            gemv_g_base_q <= 32'(layer_q) * 32'(GrpsPerLyr) +
                             32'(GrpsQ + GrpsK + GrpsV + GrpsO + GrpsW1 + GrpsW3);
            gemv_rows_q   <= 16'(Dim);
            gemv_grps_q   <= 16'(GrpsHidden);
            gemv_in_len_q <= 16'(HiddenDim);
            gemv_src_q    <= 2'd2;
            gemv_raw_q    <= 1'b0;
            state_q       <= ST_DYN_MAX;
          end else begin
            idx_q <= idx_q + 16'd1;
          end
        end

        ST_DONE: begin
          done_q  <= 1'b1;
          state_q <= ST_IDLE;
        end

        default: state_q <= ST_IDLE;
      endcase
    end
  end

  logic unused_upper_bits;
  assign unused_upper_bits = ^{div_quo_q[39], c_rms_msq_mul[15:0], c_att_sc48[47:32],
                               c_rope_addr, c_rope_e0_16, c_rope_e1_16,
                               c_kv_store_addr, c_kv_read_addr};

endmodule
