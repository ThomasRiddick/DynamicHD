RM := rm -rf
.PHONY: clean all

all :
	echo "Compiling Cython Modules" 1>&2
	cd interface/cpp_interface; python3 setup_fill_sinks.py clean --all
	cd interface/cpp_interface; python3 setup_fill_sinks.py build_ext --inplace -f
clean :
		-$(RM) ./bin/f2py_setup_wrapper.py
		-$(RM) ./libs/*.so
		-$(RM) ./build
		-$(RM)  __pycache__
		-$(RM) ./interface/cpp_interface/libs/*.so
		-$(RM) ./interface/cpp_interface/build
		-$(RM) ./interface/cpp_interface/__pycache__
		-$(RM) ./interface/cpp_interface/compute_catchments_wrapper.cpp
		-$(RM) ./interface/cpp_interface/create_connected_lsmask_wrapper.cpp
		-$(RM) ./interface/cpp_interface/create_orography_wrapper.cpp
		-$(RM) ./interface/cpp_interface/determine_river_directions_wrapper.cpp
		-$(RM) ./interface/cpp_interface/evaluate_basins_wrapper.cpp
		-$(RM) ./interface/cpp_interface/fill_sinks_wrapper.cpp
		-$(RM) ./interface/cpp_interface/fill_sinks_wrapper_low_mem.cpp
		-$(RM) ./interface/cpp_interface/follow_streams_wrapper.cpp
		-$(RM) ./interface/cpp_interface/lake_operators_wrapper.cpp
		-$(RM) ./interface/cpp_interface/upscale_orography_wrapper.cpp
		-$(RM) ./interface/fortran_interface/__pycache__
		-$(RM) ./utilities/__pycache__
		-$(RM) ./tools/__pycache__
		-$(RM) ./base/__pycache__
		-$(RM) ./dynamic_hd_and_dynamic_lake_drivers/__pycache__
		-$(RM) ./command_line_drivers/__pycache__
