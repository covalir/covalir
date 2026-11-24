// Copyright 2026 CovalIR Contributors
// Licensed under the Apache License, Version 2.0
//
// Differential Verilator testbench for requant_group (INT32 -> INT8).
// Reads golden vectors generated directly by compiler/covalir/quant/golden.py and verifies:
//   1. Exact rounding & saturation across corner cases (INT32_MIN, INT32_MAX, rshift=0..31)
//   2. Randomized backpressure & output valid/data stability under stall
//   3. clr_i protocol safety (in_ready_o == 0 during clr_i)
//   4. Mid-stream synchronous reset recovery

#include "Vrequant_group.h"
#include "verilated.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct ReqVec {
  int32_t acc;
  uint16_t scale;
  uint8_t rshift;
  int8_t expected;
};

void tick(Vrequant_group& dut) {
  dut.clk_i = 0;
  dut.eval();
  dut.clk_i = 1;
  dut.eval();
}

void sync_reset(Vrequant_group& dut) {
  dut.rst_ni = 0;
  dut.clr_i = 0;
  dut.in_valid_i = 0;
  dut.acc_i = 0;
  dut.scale_i = 0;
  dut.rshift_i = 0;
  dut.out_ready_i = 0;
  for (int i = 0; i < 3; ++i) tick(dut);
  dut.rst_ni = 1;
  dut.eval();
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  std::string vec_path;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--vectors" && i + 1 < argc) {
      vec_path = argv[++i];
    }
  }
  if (vec_path.empty()) {
    std::cerr << "Usage: sim_requant_group --vectors <path>\n";
    return 2;
  }

  std::ifstream ifs(vec_path);
  if (!ifs) {
    std::cerr << "FAIL [requant_group]: cannot open " << vec_path << "\n";
    return 1;
  }

  std::vector<ReqVec> vecs;
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    int64_t acc = 0;
    unsigned scale = 0, rshift = 0;
    int exp = 0;
    ss >> acc >> scale >> rshift >> exp;
    vecs.push_back({static_cast<int32_t>(acc),
                    static_cast<uint16_t>(scale),
                    static_cast<uint8_t>(rshift),
                    static_cast<int8_t>(exp)});
  }
  if (vecs.empty()) {
    std::cerr << "FAIL [requant_group]: empty vector file\n";
    return 1;
  }

  Vrequant_group dut;
  sync_reset(dut);

  // Test 1: clr_i protocol safety
  dut.clr_i = 1;
  dut.in_valid_i = 1;
  dut.acc_i = 12345;
  dut.scale_i = 32768;
  dut.rshift_i = 15;
  dut.out_ready_i = 1;
  dut.eval();
  if (dut.in_ready_o != 0) {
    std::cerr << "FAIL [requant_group]: in_ready_o was 1 during clr_i\n";
    return 1;
  }
  tick(dut);
  dut.clr_i = 0;
  dut.in_valid_i = 0;
  tick(dut);

  // Test 2: Mid-stream reset
  dut.in_valid_i = 1;
  dut.out_ready_i = 0;
  tick(dut);
  sync_reset(dut);
  if (dut.out_valid_o != 0) {
    std::cerr << "FAIL [requant_group]: out_valid_o high after reset\n";
    return 1;
  }

  // Test 3: Full golden.py regression + random stalls + output stability
  std::queue<int8_t> expected_q;
  for (const auto& v : vecs) expected_q.push(v.expected);

  std::mt19937 rng(0xAE0A117u);
  std::uniform_int_distribution<int> prob(0, 99);

  size_t idx = 0;
  size_t checked = 0;
  uint64_t cycles = 0;
  bool prev_stalled = false;
  int8_t prev_stalled_val = 0;

  while (!expected_q.empty() && cycles < 200000) {
    bool dv = (idx < vecs.size()) && (prob(rng) < 75);
    bool dr = (prob(rng) < 60);

    dut.clr_i = 0;
    dut.in_valid_i = dv ? 1 : 0;
    dut.out_ready_i = dr ? 1 : 0;
    if (dv) {
      dut.acc_i = static_cast<uint32_t>(vecs[idx].acc);
      dut.scale_i = vecs[idx].scale;
      dut.rshift_i = vecs[idx].rshift;
    }

    dut.eval();

    if (prev_stalled) {
      int8_t cur = static_cast<int8_t>(dut.out_q_o);
      if (!dut.out_valid_o || cur != prev_stalled_val) {
        std::cerr << "FAIL [requant_group]: stall stability violated at cycle " << cycles << "\n";
        return 1;
      }
    }

    bool fire_in = dut.in_valid_i && dut.in_ready_o;
    bool fire_out = dut.out_valid_o && dut.out_ready_i;

    if (fire_out) {
      int8_t got = static_cast<int8_t>(dut.out_q_o);
      int8_t exp = expected_q.front();
      if (got != exp) {
        std::cerr << "FAIL [requant_group] item " << checked << ": got=" << int(got)
                  << " exp=" << int(exp) << " at cycle " << cycles << "\n";
        return 1;
      }
      expected_q.pop();
      ++checked;
      prev_stalled = false;
    } else if (dut.out_valid_o && !dut.out_ready_i) {
      prev_stalled = true;
      prev_stalled_val = static_cast<int8_t>(dut.out_q_o);
    } else {
      prev_stalled = false;
    }

    if (fire_in) ++idx;
    tick(dut);
    ++cycles;
  }

  if (!expected_q.empty()) {
    std::cerr << "FAIL [requant_group]: timed out\n";
    return 1;
  }

  std::cout << "PASS [requant_group]: verified " << checked
            << " golden.py vectors + clr_i + reset + stall-stability in " << cycles << " cycles.\n";
  return 0;
}
