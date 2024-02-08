.PHONY: clean default test compile install

include config/set_config

default: install

compile: | build
		meson compile -C build

build:
		meson setup --native-file $(SYSTEM_CONFIG_FILE) build

install: compile
		meson install -C build

clean:
		rm -rf build
		rm -rf bin
		rm -rf lib

test:
		meson test -C build
