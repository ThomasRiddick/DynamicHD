.PHONY: clean clean_examples default test compile install generate_examples

include config/set_config

default: install generate_examples

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

generate_examples: | run/examples

run/examples:
		cd run && \
		mkdir examples && \
		export PYTHONPATH=../utils/run_utilities:${PYTHONPATH} && \
		python ../utils/run_utilities/generate_examples.py configs_for_examples.cfg

clean_examples:
	  rm -rf run/examples

test:
		meson test -C build
