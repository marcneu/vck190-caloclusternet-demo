BUILD_ROOT ?= build

CXXFLAGS := -std=c++17 \
            -I${SDKTARGETSYSROOT}/usr/include/xrt/ \
            -I${SDKTARGETSYSROOT}/usr/include/xrt/experimental \
            -g -Wall -fmessage-length=0 -fPIC

LDFLAGS  := -lxrt_coreutil -luuid

# ── Object files ─────────────────────────────────────────────────────────────
$(BUILD_ROOT)/host/host.o: host.cpp host.h
	mkdir -p $(BUILD_ROOT)/host
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_ROOT)/host/wrapper.o: wrapper.cpp wrapper.h host.h
	mkdir -p $(BUILD_ROOT)/host
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# ── Shared library (Python entry point) ──────────────────────────────────────
$(BUILD_ROOT)/libaccelerator.so: $(BUILD_ROOT)/host/host.o $(BUILD_ROOT)/host/wrapper.o
	$(CXX) -shared -o $@ $^ $(LDFLAGS)

build: $(BUILD_ROOT)/libaccelerator.so

clean:
	rm -f $(BUILD_ROOT)/host.exe $(BUILD_ROOT)/host/*.o

.PHONY: host lib clean