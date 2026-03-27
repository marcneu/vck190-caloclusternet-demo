#!/bin/bash
export XILINX_XRT=/usr

./host.exe -f dut.xclbin -i input.h5 -n 1024

echo "Embedded host run completed"
