// Copyright 2026 CovalIR Contributors
// Licensed under the Apache License, Version 2.0
//
// Differential Verilator testbench for prim_fifo_sync.
// Verifies FIFO ordering, full/empty boundaries, simultaneous push+pop,
// clr_i handshake gating (wready_o==0 & rvalid_o==0), output stall stability,
// and mid-stream synchronous reset across randomized 5,000-cycle runs.

#include "Vprim_fifo_sync.h"
#include "verilated.h"

#include <cstdint>
#include <iostream>
#include <queue>
#include <random>

#ifndef TB_DEPTH
#define TB_DEPTH 16
#endif

namespace {

constexpr size_t kDepth = TB_DEPTH;

void tick(Vprim_fifo_sync& dut) {
  dut.clk_i = 0;
  dut.eval();
  dut.clk_i = 1;
  dut.eval();
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vprim_fifo_sync dut;

  dut.rst_ni = 0;
  dut.clr_i = 0;
  dut.wvalid_i = 0;
  dut.wdata_i = 0;
  dut.rready_i = 0;
  for (int i = 0; i < 3; ++i) tick(dut);
  dut.rst_ni = 1;
  dut.eval();

  std::queue<uint32_t> model;
  std::mt19937 rng(0xF1F00000u + static_cast<uint32_t>(kDepth));
  std::uniform_int_distribution<uint32_t> data_dist(0, 0xFFFFFFFFu);
  std::uniform_int_distribution<int> prob_dist(0, 99);

  bool prev_stalled = false;
  uint32_t prev_stalled_data = 0;
  uint64_t pushes = 0, pops = 0, clears = 0, resets = 0;

  for (int cyc = 0; cyc < 5000; ++cyc) {
    bool do_rst = (cyc == 1500 || cyc == 3500);
    bool do_clr = !do_rst && (prob_dist(rng) < 3);  // ~3% random clear cycles
    bool wvalid = prob_dist(rng) < 60;
    bool rready = prob_dist(rng) < 55;
    uint32_t wdata = data_dist(rng);

    dut.rst_ni = do_rst ? 0 : 1;
    dut.clr_i = do_clr ? 1 : 0;
    dut.wvalid_i = wvalid ? 1 : 0;
    dut.wdata_i = wdata;
    dut.rready_i = rready ? 1 : 0;
    dut.eval();

    if (do_clr) {
      // Handshakes MUST be suppressed while clr_i is asserted
      if (dut.wready_o != 0 || dut.rvalid_o != 0) {
        std::cerr << "FAIL [prim_fifo_sync Depth=" << kDepth
                  << "]: wready_o or rvalid_o high during clr_i at cycle " << cyc << "\n";
        return 1;
      }
      prev_stalled = false;
    } else if (!do_rst) {
      if (dut.depth_o != model.size()) {
        std::cerr << "FAIL [prim_fifo_sync Depth=" << kDepth << "]: depth mismatch got="
                  << dut.depth_o << " exp=" << model.size() << " at cycle " << cyc << "\n";
        return 1;
      }
      if (prev_stalled) {
        if (!dut.rvalid_o || dut.rdata_o != prev_stalled_data) {
          std::cerr << "FAIL [prim_fifo_sync Depth=" << kDepth
                    << "]: output violated stall stability at cycle " << cyc << "\n";
          return 1;
        }
      }
      if (dut.rvalid_o) {
        if (model.empty() || dut.rdata_o != model.front()) {
          std::cerr << "FAIL [prim_fifo_sync Depth=" << kDepth << "]: data mismatch got=0x"
                    << std::hex << dut.rdata_o << " exp=0x"
                    << (model.empty() ? 0u : model.front()) << std::dec
                    << " at cycle " << cyc << "\n";
          return 1;
        }
      }
    }

    bool fire_w = !do_rst && dut.wvalid_i && dut.wready_o;
    bool fire_r = !do_rst && dut.rvalid_o && dut.rready_i;

    if (do_rst) {
      while (!model.empty()) model.pop();
      prev_stalled = false;
      ++resets;
    } else if (do_clr) {
      while (!model.empty()) model.pop();
      prev_stalled = false;
      ++clears;
    } else {
      if (fire_r) {
        model.pop();
        ++pops;
        prev_stalled = false;
      } else if (dut.rvalid_o && !dut.rready_i) {
        prev_stalled = true;
        prev_stalled_data = dut.rdata_o;
      } else {
        prev_stalled = false;
      }
      if (fire_w) {
        model.push(wdata);
        ++pushes;
      }
    }

    tick(dut);
  }

  std::cout << "PASS [prim_fifo_sync Depth=" << kDepth << "]: 5000 cycles (" << pushes
            << " pushes, " << pops << " pops, " << clears << " clears, " << resets
            << " mid-stream resets) bit-exact.\n";
  return 0;
}
