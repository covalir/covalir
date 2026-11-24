# `targets/fpga_foss/` — Open-Source FPGA Synthesis Flow (`Yosys` / `nextpnr`)

Run Yosys generic, Lattice ECP5, or Gowin (Tang Nano 20K) synthesis on any directory emitted by `covalir compile`:

```bash
./targets/fpga_foss/build_yosys.sh build/demo_ternary generic
./targets/fpga_foss/build_yosys.sh build/demo_ternary ecp5
./targets/fpga_foss/build_yosys.sh build/demo_ternary gowin
```

Generic synthesis of a compiled `covalir_top` ternary core runs automatically in `.github/workflows/ci.yml`. Board-specific pin/clock constraints (`.lpf`/`.cst`/`.xdc`) will be added alongside physical board bring-up.
