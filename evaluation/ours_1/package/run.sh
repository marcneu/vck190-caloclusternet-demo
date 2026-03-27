#!/bin/bash
export XILINX_XRT=/usr

./host.exe -f dut.xclbin -i input.h5 -n 1

echo "Embedded host run completed"
