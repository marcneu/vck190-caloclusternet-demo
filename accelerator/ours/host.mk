BUILD_ROOT ?= build

HIGHFIVE_LIB := $$HOME/bin/highfive-2_8_0

CXXFLAGS := -std=c++17 \
            -I${SDKTARGETSYSROOT}/usr/include/xrt/ \
            -I${SDKTARGETSYSROOT}/usr/include/xrt/experimental \
            -I$(HIGHFIVE_LIB)/include \
            -g -Wall -fmessage-length=0 -fPIC

LDFLAGS  := -lxrt_coreutil -lhdf5 -luuid

# ── Object files ─────────────────────────────────────────────────────────────
$(BUILD_ROOT)/host/host.o: src/host/host.cpp src/host/host.h
	mkdir -p $(BUILD_ROOT)/host
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_ROOT)/host/main.o: src/host/main.cpp src/host/host.h
	mkdir -p $(BUILD_ROOT)/host
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# ── Executable ───────────────────────────────────────────────────────────────
$(BUILD_ROOT)/host.exe: $(BUILD_ROOT)/host/host.o $(BUILD_ROOT)/host/main.o
	$(CXX) -o $@ $^ $(LDFLAGS)

# ── Phony targets ────────────────────────────────────────────────────────────
host: $(BUILD_ROOT)/host.exe

clean:
	rm -f $(BUILD_ROOT)/host.exe $(BUILD_ROOT)/host/*.o

.PHONY: host lib clean