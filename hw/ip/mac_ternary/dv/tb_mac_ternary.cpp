// Copyright 2026 CovalIR Contributors
// Licensed under the Apache License, Version 2.0
//
// Differential Verilator testbench for mac_ternary (2-bit packed ternary x INT8).
// Reads golden vectors generated directly by compiler/covalir/quant/golden.py and verifies:
//   1. Bit-exact dot-product (incl. -(-128)=+128 & modular wrap-around) under random stalls
//   2. Output valid/data stability under downstream stall
//   3. clr_i protocol safety (in_ready_o == 0 during clr_i; partial frame discarded)
//   4. Mid-stream synchronous reset recovery

#include "Vmac_ternary.h"
#include "verilated.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef TB_LANES
#define TB_LANES 32
#endif

#ifndef TB_ACC_WIDTH
#define TB_ACC_WIDTH 32
#endif

namespace {

constexpr int kLanes = TB_LANES;
constexpr int kAccWidth = TB_ACC_WIDTH;
constexpr int kWBits = kLanes * 2;
constexpr int kABits = kLanes * 8;
constexpr int kWWords = (kWBits + 31) / 32;
constexpr int kAWords = (kABits + 31) / 32;

struct Beat {
  std::vector<uint32_t> w_words;
  std::vector<uint32_t> a_words;
  bool last;
};

int64_t sign_extend(uint64_t raw, int width) {
  uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1ULL);
  raw &= mask;
  uint64_t sign = 1ULL << (width - 1);
  if (raw & sign) {
    return static_cast<int64_t>(raw | ~mask);
  }
  return static_cast<int64_t>(raw);
}

std::vector<uint32_t> parse_hex_words(const std::string& hex, int n_words) {
  std::vector<uint32_t> words(n_words, 0);
  int len = static_cast<int>(hex.size());
  for (int w = 0; w < n_words; ++w) {
    int end = len - w * 8;
    if (end <= 0) break;
    int start = std::max(0, end - 8);
    std::string chunk = hex.substr(start, end - start);
    words[w] = static_cast<uint32_t>(std::stoul(chunk, nullptr, 16));
  }
  return words;
}

template <typename T>
void assign_port(T& port, const std::vector<uint32_t>& words) {
  uint64_t v = words[0];
  if (words.size() > 1) v |= (static_cast<uint64_t>(words[1]) << 32);
  port = static_cast<T>(v);
}

template <size_t N>
void assign_port(VlWide<N>& port, const std::vector<uint32_t>& words) {
  for (size_t i = 0; i < N; ++i) {
    port[i] = (i < words.size()) ? words[i] : 0u;
  }
}

void tick(Vmac_ternary& dut) {
  dut.clk_i = 0;
  dut.eval();
  dut.clk_i = 1;
  dut.eval();
}

void sync_reset(Vmac_ternary& dut) {
  dut.rst_ni = 0;
  dut.clr_i = 0;
  dut.in_valid_i = 0;
  dut.last_i = 0;
  dut.out_ready_i = 0;
  assign_port(dut.w_i, std::vector<uint32_t>(kWWords, 0));
  assign_port(dut.a_i, std::vector<uint32_t>(kAWords, 0));
  for (int i = 0; i < 3; ++i) tick(dut);
  dut.rst_ni = 1;
  dut.eval();
}

bool load_vectors(const std::string& path,
                  std::vector<Beat>& stream,
                  std::vector<int64_t>& expected) {
  std::ifstream ifs(path);
  if (!ifs) {
    std::cerr << "FAIL [mac_ternary]: cannot open vector file: " << path << "\n";
    return false;
  }
  std::string line;
  if (!std::getline(ifs, line)) return false;
  std::istringstream hdr(line);
  std::string k1, k2, k3;
  int lanes = 0, acc_w = 0, n_frames = 0;
  hdr >> k1 >> lanes >> k2 >> acc_w >> k3 >> n_frames;
  if (lanes != kLanes || acc_w != kAccWidth) {
    std::cerr << "FAIL [mac_ternary]: vector header mismatch (file LANES=" << lanes
              << " ACC_WIDTH=" << acc_w << ", TB LANES=" << kLanes
              << " ACC_WIDTH=" << kAccWidth << ")\n";
    return false;
  }
  while (std::getline(ifs, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    char tag = 0;
    ss >> tag;
    if (tag == 'B') {
      int last = 0;
      std::string whex, ahex;
      ss >> last >> whex >> ahex;
      Beat b;
      b.last = (last != 0);
      b.w_words = parse_hex_words(whex, kWWords);
      b.a_words = parse_hex_words(ahex, kAWords);
      stream.push_back(std::move(b));
    } else if (tag == 'E') {
      int64_t exp = 0;
      ss >> exp;
      expected.push_back(exp);
    }
  }
  return !expected.empty();
}

}  // namespace

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);

  std::string vec_path;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--vectors" && i + 1 < argc) {
      vec_path = argv[++i];
    }
  }
  if (vec_path.empty()) {
    std::cerr << "Usage: sim_mac_ternary --vectors <path_from_golden_py>\n";
    return 2;
  }

  std::vector<Beat> stream;
  std::vector<int64_t> expected_vec;
  if (!load_vectors(vec_path, stream, expected_vec)) {
    return 1;
  }

  Vmac_ternary dut;
  sync_reset(dut);

  // -------------------------------------------------------------------------
  // Test 1: clr_i protocol & partial-frame discard check
  // -------------------------------------------------------------------------
  {
    std::vector<uint32_t> w_neg1(kWWords, 0xFFFFFFFFu);  // all 2'b11 = -1
    std::vector<uint32_t> a_m128(kAWords, 0x80808080u);  // all -128
    dut.in_valid_i = 1;
    dut.last_i = 0;
    dut.out_ready_i = 1;
    assign_port(dut.w_i, w_neg1);
    assign_port(dut.a_i, a_m128);
    dut.eval();
    tick(dut);
    dut.in_valid_i = 0;
    tick(dut);

    // Assert clr_i while valid is high -> in_ready_o MUST be 0
    dut.clr_i = 1;
    dut.in_valid_i = 1;
    dut.last_i = 1;
    assign_port(dut.w_i, stream[0].w_words);
    assign_port(dut.a_i, stream[0].a_words);
    dut.eval();
    if (dut.in_ready_o != 0) {
      std::cerr << "FAIL [mac_ternary]: clr_i protocol bug — in_ready_o was 1 during clr_i!\n";
      return 1;
    }
    tick(dut);

    dut.clr_i = 0;
    dut.eval();
    if (!dut.in_ready_o) {
      std::cerr << "FAIL [mac_ternary]: expected in_ready_o=1 immediately after clr_i\n";
      return 1;
    }
    tick(dut);
    dut.in_valid_i = 0;
    dut.last_i = 0;

    bool got_out = false;
    for (int c = 0; c < 10; ++c) {
      dut.eval();
      if (dut.out_valid_o) {
        int64_t got = sign_extend(dut.out_acc_o, kAccWidth);
        if (got != expected_vec[0]) {
          std::cerr << "FAIL [mac_ternary]: clr_i leaked prior state! got=" << got
                    << " expected=" << expected_vec[0] << "\n";
          return 1;
        }
        got_out = true;
        tick(dut);
        break;
      }
      tick(dut);
    }
    if (!got_out) {
      std::cerr << "FAIL [mac_ternary]: no output after clr_i recovery\n";
      return 1;
    }
  }

  // -------------------------------------------------------------------------
  // Test 2: Mid-stream synchronous reset recovery
  // -------------------------------------------------------------------------
  {
    dut.in_valid_i = 1;
    dut.last_i = 0;
    dut.out_ready_i = 0;
    assign_port(dut.w_i, stream[0].w_words);
    assign_port(dut.a_i, stream[0].a_words);
    tick(dut);
    sync_reset(dut);
    if (dut.out_valid_o != 0) {
      std::cerr << "FAIL [mac_ternary]: out_valid_o remained high after mid-stream reset\n";
      return 1;
    }
  }

  // -------------------------------------------------------------------------
  // Test 3: Full golden.py regression + randomized stalls + output stability
  // -------------------------------------------------------------------------
  std::queue<int64_t> expected_q;
  for (int64_t v : expected_vec) expected_q.push(v);

  std::mt19937 rng(0xB17E7158u + kLanes * 17 + kAccWidth);
  std::uniform_int_distribution<int> prob_dist(0, 99);

  size_t idx = 0;
  size_t checked = 0;
  uint64_t cycles = 0;
  constexpr uint64_t kMaxCycles = 500000;

  bool prev_stalled = false;
  int64_t prev_stalled_acc = 0;

  while (!expected_q.empty() && cycles < kMaxCycles) {
    bool drive_valid = (idx < stream.size()) && (prob_dist(rng) < 75);
    bool drive_ready = (prob_dist(rng) < 60);

    dut.clr_i = 0;
    dut.in_valid_i = drive_valid ? 1 : 0;
    dut.out_ready_i = drive_ready ? 1 : 0;
    if (drive_valid) {
      assign_port(dut.w_i, stream[idx].w_words);
      assign_port(dut.a_i, stream[idx].a_words);
      dut.last_i = stream[idx].last ? 1 : 0;
    } else {
      dut.last_i = 0;
    }

    dut.eval();

    if (prev_stalled) {
      int64_t cur_acc = sign_extend(dut.out_acc_o, kAccWidth);
      if (!dut.out_valid_o || cur_acc != prev_stalled_acc) {
        std::cerr << "FAIL [mac_ternary]: output violated stability under stall at cycle "
                  << cycles << "\n";
        return 1;
      }
    }

    bool fire_in = dut.in_valid_i && dut.in_ready_o;
    bool fire_out = dut.out_valid_o && dut.out_ready_i;

    if (fire_out) {
      int64_t got = sign_extend(dut.out_acc_o, kAccWidth);
      int64_t exp = expected_q.front();
      if (got != exp) {
        std::cerr << "FAIL [mac_ternary] frame " << checked << ": got=" << got
                  << " expected=" << exp << " at cycle " << cycles << "\n";
        return 1;
      }
      expected_q.pop();
      ++checked;
      prev_stalled = false;
    } else if (dut.out_valid_o && !dut.out_ready_i) {
      prev_stalled = true;
      prev_stalled_acc = sign_extend(dut.out_acc_o, kAccWidth);
    } else {
      prev_stalled = false;
    }

    if (fire_in) {
      ++idx;
    }

    tick(dut);
    ++cycles;
  }

  if (!expected_q.empty()) {
    std::cerr << "FAIL [mac_ternary]: timed out with " << expected_q.size()
              << " outputs remaining\n";
    return 1;
  }

  std::cout << "PASS [mac_ternary Lanes=" << kLanes << " AccWidth=" << kAccWidth
            << "]: verified " << checked << " golden.py frames (" << idx
            << " beats) + clr_i + reset + stall-stability in " << cycles << " cycles.\n";
  return 0;
}
