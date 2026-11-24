#!/usr/bin/env bash
# Copyright 2026 CovalIR Contributors
# Licensed under the Apache License, Version 2.0
#
# Open-source FPGA synthesis & place-and-route runner for a compiled CovalIR directory.
# Usage: ./targets/fpga_foss/build_yosys.sh <dist_dir> [generic|ecp5|gowin]

set -euo pipefail

DIST_DIR="${1:?Usage: build_yosys.sh <dist_dir> [generic|ecp5|gowin]}"
TARGET="${2:-generic}"

if ! command -v yosys >/dev/null 2>&1; then
  echo "ERROR: 'yosys' not found on PATH (install OSS CAD Suite)." >&2
  exit 1
fi

case "${TARGET}" in
  generic)
    yosys -s "${DIST_DIR}/synth_yosys.ys"
    ;;
  ecp5)
    yosys -p "read_verilog -sv $(grep '^read_verilog' "${DIST_DIR}/synth_yosys.ys" | awk '{print $3}' | tr '\n' ' '); synth_ecp5 -top covalir_top -json ${DIST_DIR}/covalir_ecp5.json"
    echo "Wrote ${DIST_DIR}/covalir_ecp5.json (ready for nextpnr-ecp5)."
    ;;
  gowin)
    yosys -p "read_verilog -sv $(grep '^read_verilog' "${DIST_DIR}/synth_yosys.ys" | awk '{print $3}' | tr '\n' ' '); synth_gowin -top covalir_top -json ${DIST_DIR}/covalir_gowin.json"
    echo "Wrote ${DIST_DIR}/covalir_gowin.json (ready for nextpnr-himbaechel)."
    ;;
  *)
    echo "ERROR: Unknown target '${TARGET}'. Choose generic, ecp5, or gowin." >&2
    exit 1
    ;;
esac
