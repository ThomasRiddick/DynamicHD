.PHONY: clean default test compile

include config/set_config

default: compile

compile: | build
		meson compile -C build

build:
		meson setup --native-file $(SYSTEM_CONFIG_FILE) build

clean:
		rm -rf build

test:
		meson test -C build
