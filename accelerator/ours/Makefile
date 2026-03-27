BUILD_ROOT ?= build

host:
	make -f host.mk host

aie:
	make -f firmware.mk aie

aiesim:
	make -f firmware.mk aiesim

link:
	make -f firmware.mk link
	
package:
	make -f firmware.mk package

run:
	make -f firmware.mk run


clean:
	mkdir -p tmp
	cp $(BUILD_ROOT)/*.h5 tmp/ 2>/dev/null || :
	rm -rf $(BUILD_ROOT)
	mkdir -p $(BUILD_ROOT)
	cp tmp/*.h5 $(BUILD_ROOT)/
	rm -rf tmp
