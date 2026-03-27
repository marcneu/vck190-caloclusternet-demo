export LD_LIBRARY_PATH += :${VITIS_HLS_ROOT}/lib/lnx64.o:${VITIS_HLS_ROOT}/lib/lnx64.o/Rhel/8

HDF5_LIB       := $$HOME/bin/hdf5-1_14_0
HIGHFIVE_LIB   := $$HOME/bin/highfive-2_8_0
VITIS_HLS_ROOT := /tools/xilinx/Vitis/2024.2

BUILD_ROOT 	         ?= build
PCNHLSLIB_ROOT       := $(abspath ../../libraries/pcnhlslib)
SRC_ROOT		     := $(abspath ./src)
CFG_ROOT		     := $(abspath ./config)

CXX := v++ 

IFLAG_SIM += -g
IFLAG_SIM += -I "${VITIS_HLS_ROOT}/include"
IFLAG_SIM += -D__SIM_FPO__ -D__SIM_OPENCV__ -D__SIM_FFT__ -D__SIM_FIR__ -D__SIM_DDS__ -D__DSP48E1__
IFLAG_SIM += -L"${VITIS_HLS_ROOT}/lnx64/lib/csim" -Wl,-rpath,"${VITIS_HLS_ROOT}/lnx64/lib/csim" -lhlsmc++-CLANG39
IFLAG_SIM += -L"${VITIS_HLS_ROOT}/lnx64/tools/fpo_v7_1" -Wl,-rpath,"${VITIS_HLS_ROOT}/lnx64/tools/fpo_v7_1" -lIp_floating_point_v7_1_bitacc_cmodel -lgmp -lmpfr
IFLAG_SIM += -fuse-ld=lld -lm -lpthread
IFLAG_SIM += -L"$(HDF5_LIB)/lib" -lhdf5 -Wl,-rpath,"$(HDF5_LIB)/lib"

CFLAG_SIM += -g
CFLAG_SIM += -I "${SRC_ROOT}/pl"
CFLAG_SIM += -I "${PCNHLSLIB_ROOT}"
CFLAG_SIM += -I "${VITIS_HLS_ROOT}/include"
CFLAG_SIM += -I "$(HDF5_LIB)/include"
CFLAG_SIM += -I "$(HIGHFIVE_LIB)/include"
CFLAG_SIM += -D__SIM_FPO__ -D__SIM_OPENCV__ -D__SIM_FFT__ -D__SIM_FIR__ -D__SIM_DDS__ -D__DSP48E1__

CFLAG_SIM += -fPIC -fPIE -Wno-unused-result
CFLAG_SIM += --gcc-toolchain=/tools/xilinx/Vitis/2024.2/tps/lnx64/gcc-8.3.0

CXX_SIM = ${VITIS_HLS_ROOT}/vcxx/libexec/clang++

PLATFORM = xilinx_vck190_base_202420_1
TARGET   = hw #hw_emu #When switching to from hw_emu to hw, make sure to set the -g flag

.PHONY: clean csim csynth link package all

clean:
	rm -f ${BUILD_ROOT}/*

${BUILD_ROOT}/csim.exe: ${SRC_ROOT}/csim.cpp ${SRC_ROOT}/pl/load/load.cpp ${SRC_ROOT}/pl/caloclusternet/caloclusternet.cpp ${SRC_ROOT}/pl/store/store.cpp ${SRC_ROOT}/pl/global.h ${SRC_ROOT}/pl/weights.h
	mkdir -p ${BUILD_ROOT}/csim
	$(CXX_SIM) -c ${SRC_ROOT}/pl/load/load.cpp -o ${BUILD_ROOT}/csim/load.o $(CFLAG_SIM)
	$(CXX_SIM) -c ${SRC_ROOT}/pl/caloclusternet/caloclusternet.cpp -o ${BUILD_ROOT}/csim/caloclusternet.o $(CFLAG_SIM)
	$(CXX_SIM) -c ${SRC_ROOT}/pl/store/store.cpp -o ${BUILD_ROOT}/csim/store.o $(CFLAG_SIM)
	$(CXX_SIM) -c ${SRC_ROOT}/csim.cpp -o ${BUILD_ROOT}/csim/csim.o $(CFLAG_SIM)
	$(CXX_SIM) ${BUILD_ROOT}/csim/*.o -o ${BUILD_ROOT}/csim.exe $(IFLAG_SIM)

$(BUILD_ROOT)/load.xo: ${SRC_ROOT}/pl/load/load.cpp ${SRC_ROOT}/pl/load/load.h ${SRC_ROOT}/pl/global.h ${CFG_ROOT}/load.hls.cfg
	mkdir -p $(BUILD_ROOT)
	cd $(BUILD_ROOT) && $(CXX) -c --mode hls -f $(PLATFORM) --freqhz=350000000 --config ${CFG_ROOT}/load.hls.cfg
	cp $(BUILD_ROOT)/load/load.xo $(BUILD_ROOT)/load.xo

$(BUILD_ROOT)/store.xo: ${SRC_ROOT}/pl/store/store.cpp ${SRC_ROOT}/pl/store/store.h ${SRC_ROOT}/pl/global.h ${CFG_ROOT}/store.hls.cfg
	mkdir -p $(BUILD_ROOT)
	cd $(BUILD_ROOT) && $(CXX) -c --mode hls -f $(PLATFORM) --freqhz=350000000 --config ${CFG_ROOT}/store.hls.cfg
	cp $(BUILD_ROOT)/store/store.xo $(BUILD_ROOT)/store.xo

$(BUILD_ROOT)/caloclusternet.xo: ${SRC_ROOT}/pl/caloclusternet/caloclusternet.cpp ${SRC_ROOT}/pl/caloclusternet/caloclusternet.h ${SRC_ROOT}/pl/global.h ${CFG_ROOT}/caloclusternet.hls.cfg
	mkdir -p $(BUILD_ROOT)
	cd $(BUILD_ROOT) && $(CXX) -c --mode hls -f $(PLATFORM) --freqhz=350000000 --config ${CFG_ROOT}/caloclusternet.hls.cfg
	cp $(BUILD_ROOT)/caloclusternet/caloclusternet.xo $(BUILD_ROOT)/caloclusternet.xo

$(BUILD_ROOT)/dut.xsa: $(BUILD_ROOT)/store.xo $(BUILD_ROOT)/load.xo $(BUILD_ROOT)/caloclusternet.xo
	mkdir -p $(BUILD_ROOT)
	cd $(BUILD_ROOT) && $(CXX) -l -f $(PLATFORM) -t $(TARGET) -g store.xo load.xo caloclusternet.xo --freqhz=350000000:load_1.ap_clk --freqhz=250000000:caloclusternet_1.ap_clk --freqhz=350000000:store_1.ap_clk --config ${CFG_ROOT}/link.cfg -o dut.xsa

$(BUILD_ROOT)/dut.xclbin: $(BUILD_ROOT)/dut.xsa
	cd $(BUILD_ROOT) && $(CXX) --package -t $(TARGET) -f $(PLATFORM) --config ${CFG_ROOT}/package.cfg dut.xsa -o dut.xclbin

$(BUILD_ROOT)/package/launch_hw_emu.sh: $(BUILD_ROOT)/dut.xclbin
	cd $(BUILD_ROOT)/package && ./launch_hw_emu.sh -g -user-pre-sim-script=../../../../../config/waveforms.tcl -forward-port 2222 22 

csim:  	  		$(BUILD_ROOT)/csim.exe
csynth:   		$(BUILD_ROOT)/load.xo $(BUILD_ROOT)/store.xo $(BUILD_ROOT)/caloclusternet.xo
link:     		$(BUILD_ROOT)/dut.xsa
package:  		$(BUILD_ROOT)/dut.xclbin
run:            $(BUILD_ROOT)/package/launch_hw_emu.sh

all: package
