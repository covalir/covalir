// Copyright 2026 CovalIR Contributors
// Licensed under the Apache License, Version 2.0
//
// Differential Verilator testbench for prim_generic_sram_2p.
// Verifies byte-masked writes, 1-cycle rvalid_o latency, read-first same-address
// collision semantics, and synchronous reset of rvalid_o.

#include "Vprim_generic_sram_2p.h"
#include "verilated.h"

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#ifndef TB_DEPTH
#define TB_DEPTH 64
#endif

namespace {

constexpr size_t kDepth = TB_DEPTH;

void tick(Vprim_generic_sram_2p& dut) {
  dut.clk_i = 0;
  dut.eval();
  dut.clk_i = 1;
  dut.eval();
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Vprim_generic_sram_2p dut;

  dut.rst_ni = 0;
  dut.wreq_i = 0;
  dut.waddr_i = 0;
  dut.wdata_i = 0;
  dut.wmask_i = 0;
  dut.rreq_i = 0;
  dut.raddr_i = 0;
  for (int i = 0; i < 3; ++i) tick(dut);
  dut.rst_ni = 1;
  dut.eval();

  if (dut.rvalid_o != 0) {
    std::cerr << "FAIL [prim_generic_sram_2p]: rvalid_o high after reset\n";
    return 1;
  }

  // Initialize every entry deterministically so no X reads occur
  std::vector<uint64_t> mem(kDepth, 0);
  for (size_t a = 0; a < kDepth; ++a) {
    uint64_t init_v = 0xA5A5000000000000ULL ^ (a * 0x0102030405060708ULL);
    dut.wreq_i = 1;
    dut.waddr_i = static_cast<uint32_t>(a);
    dut.wdata_i = init_v;
    dut.wmask_i = 0xFF;
    dut.rreq_i = 0;
    mem[a] = init_v;
    tick(dut);
  }
  dut.wreq_i = 0;
  tick(dut);

  std::mt19937_64 rng(0x5A0A0000ULL + kDepth);
  std::uniform_int_distribution<size_t> addr_dist(0, kDepth - 1);
  std::uniform_int_distribution<int> prob_dist(0, 99);

  bool exp_rvalid = false;
  uint64_t exp_rdata = 0;
  uint64_t reads_checked = 0, collisions_checked = 0;

  for (int cyc = 0; cyc < 5000; ++cyc) {
    bool do_rst = (cyc == 2500);
    bool wreq = prob_dist(rng) < 60;
    bool rreq = prob_dist(rng) < 70;
    size_t waddr = addr_dist(rng);
    // Force ~20% same-address read-during-write collisions
    size_t raddr = (wreq && rreq && prob_dist(rng) < 20) ? waddr : addr_dist(rng);
    uint64_t wdata = rng();
    uint8_t wmask = static_cast<uint8_t>(rng() & 0xFFu);

    dut.rst_ni = do_rst ? 0 : 1;
    dut.wreq_i = wreq ? 1 : 0;
    dut.waddr_i = static_cast<uint32_t>(waddr);
    dut.wdata_i = wdata;
    dut.wmask_i = wmask;
    dut.rreq_i = rreq ? 1 : 0;
    dut.raddr_i = static_cast<uint32_t>(raddr);
    dut.eval();

    // Check registered outputs from previous cycle
    if (dut.rvalid_o != (exp_rvalid ? 1 : 0)) {
      std::cerr << "FAIL [prim_generic_sram_2p Depth=" << kDepth << "]: rvalid_o mismatch at cycle "
                << cyc << " got=" << int(dut.rvalid_o) << " exp=" << int(exp_rvalid) << "\n";
      return 1;
    }
    if (exp_rvalid) {
      if (dut.rdata_o != exp_rdata) {
        std::cerr << "FAIL [prim_generic_sram_2p Depth=" << kDepth << "]: rdata mismatch at cycle "
                  << cyc << " got=0x" << std::hex << dut.rdata_o << " exp=0x" << exp_rdata
                  << std::dec << "\n";
        return 1;
      }
      ++reads_checked;
    }

    // Compute next-cycle expected state (READ-FIRST semantics)
    bool next_rvalid = !do_rst && rreq;
    uint64_t next_rdata = rreq ? mem[raddr] : exp_rdata;
    if (!do_rst && wreq && rreq && waddr == raddr) {
      ++collisions_checked;
    }
    if (wreq) {
      uint64_t updated = mem[waddr];
      for (int b = 0; b < 8; ++b) {
        if ((wmask >> b) & 1u) {
          uint64_t byte_mask = 0xFFULL << (b * 8);
          updated = (updated & ~byte_mask) | (wdata & byte_mask);
        }
      }
      mem[waddr] = updated;
    }

    tick(dut);
    exp_rvalid = next_rvalid;
    exp_rdata = next_rdata;
  }

  std::cout << "PASS [prim_generic_sram_2p Depth=" << kDepth << "]: 5000 cycles (" << reads_checked
            << " reads, " << collisions_checked << " read-first collisions) bit-exact.\n";
  return 0;
}
