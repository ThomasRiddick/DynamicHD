.PHONY: clean clean_examples default test compile install examples

include config/set_config

default: install examples

compile: | build meson_options.txt
		meson compile -C build

build: | meson_options.txt
		meson setup --native-file $(SYSTEM_CONFIG_FILE) build

install: compile | meson_options.txt
		meson install -C build

meson_options.txt:
		ln -s meson.options meson_options.txt

clean:
		rm -rf build
		rm -rf bin
		rm -rf lib

examples: | run/examples

run/examples:
		cd run && \
		mkdir examples && \
		export PYTHONPATH=../utils/run_utilities:${PYTHONPATH} && \
		python ../utils/run_utilities/generate_examples.py configs_for_examples.cfg

clean_examples:
	  rm -rf run/examples

test:
		meson test -C build
