.PHONY: clean default

include config/set_config

default: | build
		meson compile -C build

build:
		meson setup --native-file $(SYSTEM_CONFIG_FILE) build

clean:
		rm -rf build
