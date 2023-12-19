.PHONY: clean default test compile install

include config/set_config

default: compile

compile: | build
		meson compile -C build

build:
		meson setup --native-file $(SYSTEM_CONFIG_FILE) build

install: 	
		meson install -C build

clean:
		rm -rf build
		rm -rf bin
		rm -rf lib

test:
		meson test -C build
