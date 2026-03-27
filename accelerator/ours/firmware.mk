export LD_LIBRARY_PATH += :${VITIS_HLS_ROOT}/lib/lnx64.o:${VITIS_HLS_ROOT}/lib/lnx64.o/Rhel/8

HDF5_LIB       := $$HOME/bin/hdf5-1_14_0
HIGHFIVE_LIB   := $$HOME/bin/highfive-2_8_0
VITIS_HLS_ROOT := /tools/xilinx/Vitis/2024.2

BUILD_ROOT 	         ?= build
PCNHLSLIB_ROOT       := $(abspath ../../libraries/pcnhlslib)
PCNAIELIB_ROOT       := $(abspath ../../libraries/pcnaielib)
SRC_ROOT		     := $(abspath ./src)
CFG_ROOT		     := $(abspath ./config)

CXX := v++ 

CFLAG =  -I ${VITIS_HLS_ROOT}/aietools/include/
CFLAG += -I ${PCNAIELIB_ROOT}/aie
CFLAG += -I ${SRC_ROOT}
CFLAG += --aie.pl-freq=250
#CFLAG += --aie.event-trace=runtime
#CFLAG += --aie.event-trace-port=gmio
#CFLAG += --aie.num-trace-streams=4
CFLAG += --aie.broadcast-enable-core=true
CFLAG += --aie.xlopt=1
CFLAG += --aie.Xrouter=dmaFIFOsInFreeBankOnly
#CFLAG += --aie.evaluate-fifo-depth

PLATFORM = xilinx_vck190_base_202420_1
TARGET   = hw_emu #When switching to from hw_emu to hw, make sure to set the -g flag

.PHONY: clean csim link package all

clean:
	rm -f ${BUILD_ROOT}/*

$(BUILD_ROOT)/load.xo: ${SRC_ROOT}/pl/load/load.cpp ${SRC_ROOT}/pl/load/load.h ${SRC_ROOT}/pl/global.h ${CFG_ROOT}/load.hls.cfg
	mkdir -p $(BUILD_ROOT)
	cd $(BUILD_ROOT) && $(CXX) -c --mode hls -f $(PLATFORM) --freqhz=350000000 --config ${CFG_ROOT}/load.hls.cfg
	cp $(BUILD_ROOT)/load/load.xo $(BUILD_ROOT)/load.xo

$(BUILD_ROOT)/g.xo: ${SRC_ROOT}/pl/g/g.cpp ${SRC_ROOT}/pl/g/g.h ${SRC_ROOT}/pl/global.h ${SRC_ROOT}/pl/weights.h ${CFG_ROOT}/g.hls.cfg
	mkdir -p $(BUILD_ROOT)
	cd $(BUILD_ROOT) && $(CXX) -c --mode hls -f $(PLATFORM) --freqhz=350000000 --config ${CFG_ROOT}/g.hls.cfg
	cp $(BUILD_ROOT)/g/g.xo $(BUILD_ROOT)/g.xo

$(BUILD_ROOT)/e.xo: ${SRC_ROOT}/pl/e/e.cpp ${SRC_ROOT}/pl/e/e.h ${SRC_ROOT}/pl/global.h ${SRC_ROOT}/pl/weights.h ${CFG_ROOT}/e.hls.cfg
	mkdir -p $(BUILD_ROOT)
	cd $(BUILD_ROOT) && $(CXX) -c --mode hls -f $(PLATFORM) --freqhz=350000000 --config ${CFG_ROOT}/e.hls.cfg
	cp $(BUILD_ROOT)/e/e.xo $(BUILD_ROOT)/e.xo

$(BUILD_ROOT)/c.xo: ${SRC_ROOT}/pl/c/c.cpp ${SRC_ROOT}/pl/c/c.h ${SRC_ROOT}/pl/global.h ${SRC_ROOT}/pl/weights.h ${CFG_ROOT}/c.hls.cfg
	mkdir -p $(BUILD_ROOT)
	cd $(BUILD_ROOT) && $(CXX) -c --mode hls -f $(PLATFORM) --freqhz=350000000 --config ${CFG_ROOT}/c.hls.cfg
	cp $(BUILD_ROOT)/c/c.xo $(BUILD_ROOT)/c.xo

$(BUILD_ROOT)/a.xo: ${SRC_ROOT}/pl/a/a.cpp ${SRC_ROOT}/pl/a/a.h ${SRC_ROOT}/pl/global.h ${SRC_ROOT}/pl/weights.h ${CFG_ROOT}/a.hls.cfg
	mkdir -p $(BUILD_ROOT)
	cd $(BUILD_ROOT) && $(CXX) -c --mode hls -f $(PLATFORM) --freqhz=350000000 --config ${CFG_ROOT}/a.hls.cfg
	cp $(BUILD_ROOT)/a/a.xo $(BUILD_ROOT)/a.xo

$(BUILD_ROOT)/store.xo: ${SRC_ROOT}/pl/store/store.cpp ${SRC_ROOT}/pl/store/store.h ${SRC_ROOT}/pl/global.h ${CFG_ROOT}/store.hls.cfg
	mkdir -p $(BUILD_ROOT)
	cd $(BUILD_ROOT) && $(CXX) -c --mode hls -f $(PLATFORM) --freqhz=350000000 --config ${CFG_ROOT}/store.hls.cfg
	cp $(BUILD_ROOT)/store/store.xo $(BUILD_ROOT)/store.xo

$(BUILD_ROOT)/libadf.a: $(SRC_ROOT)/aie/graph.h $(SRC_ROOT)/aie/params.h
	mkdir -p $(BUILD_ROOT)
	cd $(BUILD_ROOT) && $(CXX) -c --mode aie -t hw -f $(PLATFORM) --work_dir aie $(CFLAG) $(SRC_ROOT)/aie/graph.cc

$(BUILD_ROOT)/dut.xsa: $(BUILD_ROOT)/store.xo $(BUILD_ROOT)/load.xo $(BUILD_ROOT)/a.xo $(BUILD_ROOT)/c.xo $(BUILD_ROOT)/e.xo $(BUILD_ROOT)/g.xo $(BUILD_ROOT)/libadf.a
	mkdir -p $(BUILD_ROOT)
	cd $(BUILD_ROOT) && $(CXX) -l -f $(PLATFORM) -t $(TARGET) -g store.xo a.xo load.xo c.xo e.xo g.xo libadf.a --config ${CFG_ROOT}/link.cfg --freqhz=350000000:load_1.ap_clk --freqhz=250000000:a_1.ap_clk --freqhz=250000000:c_1.ap_clk --freqhz=250000000:e_1.ap_clk --freqhz=250000000:g_1.ap_clk --freqhz=350000000:store_1.ap_clk -o dut.xsa

$(BUILD_ROOT)/dut.xclbin: $(BUILD_ROOT)/dut.xsa
	cd $(BUILD_ROOT) && v++ --package -t $(TARGET) -f $(PLATFORM) --config ${CFG_ROOT}/package.cfg libadf.a dut.xsa -o dut.xclbin

aie: $(BUILD_ROOT)/libadf.a

aiesim: 
	mkdir -p $(BUILD_ROOT)
	cd $(BUILD_ROOT) && aiesimulator --pkg-dir=aie --options-file=$(CFG_ROOT)/aiesim_options.txt  --online -wdb -text --hang-detect-time=10000 --profile
# --simulation-cycle-timeout=60000 or --hang-detect-time=10000 <- use either to debug AI engines 
# --profile <- Use this for enable per kernel profiling

link: $(BUILD_ROOT)/dut.xsa

package: $(BUILD_ROOT)/dut.xclbin

run: $(BUILD_ROOT)/package/launch_hw_emu.sh
	cd $(BUILD_ROOT)/package && ./launch_hw_emu.sh -g -aie-sim-options ../../config/aiesim_options.txt -xtlm-aximm-log -xtlm-axis-log  -user-pre-sim-script=../../../../../config/waveforms.tcl -add-env AIE_COMPILER_WORKDIR=../aie -forward-port 2222 22 
#-g for enabling the intertactive RTL simulation 
#-run-app=run.sh

all: package
