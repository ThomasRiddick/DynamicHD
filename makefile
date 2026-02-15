.PHONY: clean clean_examples default test compile install examples

default: install examples

compile: | build
		meson compile -C build

build: | meson_options.txt Dynamic_HD_Scripts/Dynamic_HD_Scripts/bin
		meson setup build

install: compile
		meson install -C build

meson_options.txt:
		ln -s meson.options meson_options.txt

Dynamic_HD_Scripts/Dynamic_HD_Scripts/bin:
		mkdir Dynamic_HD_Scripts/Dynamic_HD_Scripts/bin

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
