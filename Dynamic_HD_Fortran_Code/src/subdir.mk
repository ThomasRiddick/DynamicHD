# Add inputs and outputs from these tool invocations to the build variables
F90_SRCS += \
$(FRUIT_LOC)/fruit.f90 \
../src/base/precision_mod.f90 \
../src/base/pointer_mod.f90 \
../src/base/unstructured_grid_mod.f90 \
../src/base/coords_mod.f90 \
../src/base/doubly_linked_list_link_mod.f90 \
../src/base/doubly_linked_list_mod.f90 \
../src/testing/doubly_linked_list_test_module.f90 \
../src/base/subfield_mod.f90 \
../src/testing/subfield_test_mod.f90 \
../src/base/field_section_mod.f90 \
../src/testing/field_section_test_mod.f90 \
../src/algorithms/map_non_coincident_grids_mod.f90 \
../src/base/area_mod.f90 \
../src/testing/area_test_mod.f90 \
../src/algorithms/break_loops_mod.f90 \
../src/drivers/break_loops_driver_mod.f90 \
../src/algorithms/cotat_parameters_mod.f90 \
../src/algorithms/cotat_plus.f90 \
../src/drivers/cotat_plus_driver_mod.f90 \
../src/testing/cotat_plus_test_mod.f90 \
../src/algorithms/flow.f90 \
../src/testing/flow_test_mod.f90 \
../src/algorithms/loop_breaker_mod.f90 \
../src/testing/loop_breaker_test_mod.f90 \
../src/algorithms/flow_accumulation_algorithm_mod.f90 \
../src/base/check_return_code_netcdf_mod.f90 \
../src/base/parameters_mod.f90 \
../src/algorithms/accumulate_flow_mod.f90 \
../src/drivers/accumulate_flow_driver_mod.f90 \
../src/testing/accumulate_flow_test_mod.f90 \
../src/latlon_hd_and_lake_model/latlon_lake_logger_mod.f90 \
../src/latlon_hd_and_lake_model/latlon_lake_model_mod.f90 \
../src/latlon_hd_and_lake_model/latlon_lake_model_interface_mod.f90 \
../src/latlon_hd_and_lake_model/latlon_lake_model_io_mod.f90 \
../src/latlon_hd_and_lake_model/latlon_hd_model_io_mod.f90 \
../src/latlon_hd_and_lake_model/latlon_hd_model_interface_mod.f90 \
../src/latlon_hd_and_lake_model/latlon_hd_model_mod.f90 \
../src/latlon_hd_and_lake_model/latlon_hd_model_driver.f90 \
../src/latlon_hd_and_lake_model/latlon_lake_model_retrieve_lake_numbers.f90 \
../src/testing/latlon_hd_and_lake_model_test_mod.f90 \
../src/testing/icosohedral_hd_and_lake_model_test_mod.f90 \
../src/command_line_drivers/cotat_plus_latlon_to_icon_simple_interface.f90 \
../src/testing/map_non_coincident_grids_test_mod.f90 \
../src/testing/manual_fruit_basket.f90 \
../src/testing/manual_fruit_basket_driver.f90 \
../src/algorithms/icon_to_latlon_landsea_downscaler.f90 \
../src/command_line_drivers/icon_to_latlon_landsea_downscaler_simple_interface.f90 \
../src/command_line_drivers/latlon_to_icon_loop_breaker_simple_interface.f90 \
../src/command_line_drivers/accumulate_flow_icon_simple_interface.f90 \
../src/command_line_drivers/cross_grid_mapper_latlon_to_icon_simple_interface.f90 \
../src/algorithms/cross_grid_mapper.f90 \
../src/base/convert_rdirs_to_indices.f90 \
../src/icosohedral_hd_and_lake_model/icosohedral_lake_model_mod.f90 \
../src/icosohedral_hd_and_lake_model/icosohedral_lake_model_io_mod.f90 \
../src/icosohedral_hd_and_lake_model/icosohedral_lake_model_interface_mod.f90 \
../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.f90 \
../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_io_mod.f90 \
../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_interface_mod.f90 \
../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_driver.f90 \
../src/icosohedral_hd_and_lake_model/grid_information_mod.f90

OBJS += \
./src/base/precision_mod.o \
./src/base/pointer_mod.o \
./src/base/unstructured_grid_mod.o \
./src/base/coords_mod.o \
./src/base/doubly_linked_list_link_mod.o \
./src/base/doubly_linked_list_mod.o \
./src/base/subfield_mod.o \
./src/base/field_section_mod.o \
./src/algorithms/map_non_coincident_grids_mod.o \
./src/base/area_mod.o \
./src/algorithms/break_loops_mod.o \
./src/drivers/break_loops_driver_mod.o \
./src/algorithms/cotat_parameters_mod.o \
./src/algorithms/cotat_plus.o \
./src/drivers/cotat_plus_driver_mod.o \
./src/algorithms/flow.o \
./src/algorithms/loop_breaker_mod.o \
./src/algorithms/flow_accumulation_algorithm_mod.o \
./src/algorithms/icon_to_latlon_landsea_downscaler.o \
./src/algorithms/accumulate_flow_mod.o \
./src/drivers/accumulate_flow_driver_mod.o \
./src/algorithms/cross_grid_mapper.o \
./src/base/convert_rdirs_to_indices.o

COTAT_PLUS_LATLON_TO_ICON_SIMPLE_INTERFACE_OBJS += \
./src/command_line_drivers/cotat_plus_latlon_to_icon_simple_interface.o \
./src/base/check_return_code_netcdf_mod.o

ACCUMULATE_FLOW_ICON_SIMPLE_INTERFACE_OBJS += \
./src/command_line_drivers/accumulate_flow_icon_simple_interface.o \
./src/base/check_return_code_netcdf_mod.o

ICON_TO_LATLON_LANDSEA_DOWNSCALER_SIMPLE_INTERFACE_OBJS += \
./src/command_line_drivers/icon_to_latlon_landsea_downscaler_simple_interface.o \
./src/base/check_return_code_netcdf_mod.o

LATLON_TO_ICON_LOOP_BREAKER_SIMPLE_INTERFACE_OBJS += \
./src/command_line_drivers/latlon_to_icon_loop_breaker_simple_interface.o \
./src/base/check_return_code_netcdf_mod.o

LATLON_TO_ICON_CROSS_GRID_MAPPER_SIMPLE_INTERFACE_OBJS += \
./src/command_line_drivers/cross_grid_mapper_latlon_to_icon_simple_interface.o \
./src/base/check_return_code_netcdf_mod.o

LATLON_HD_AND_LAKE_MODEL_OBJS += \
./src/base/parameters_mod.o \
./src/base/check_return_code_netcdf_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_logger_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_interface_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_io_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_io_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_interface_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_driver.o

LATLON_LAKE_NUMBER_RETRIEVAL_OBJS += \
./src/base/parameters_mod.o \
./src/base/check_return_code_netcdf_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_logger_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_interface_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_io_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_io_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_interface_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_retrieve_lake_numbers.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_lake_number_retrieval_driver.o

ICOSOHEDRAL_HD_AND_LAKE_MODEL_OBJS += \
./src/base/parameters_mod.o \
./src/base/check_return_code_netcdf_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_lake_model_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_lake_model_io_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_lake_model_interface_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_hd_model_io_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_hd_model_interface_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_hd_model_driver.o \
./src/icosohedral_hd_and_lake_model/grid_information_mod.o

TEST_OBJS += \
./src/testing/fruit.o \
./src/testing/doubly_linked_list_test_module.o \
./src/testing/subfield_test_mod.o \
./src/testing/field_section_test_mod.o \
./src/testing/area_test_mod.o \
./src/testing/cotat_plus_test_mod.o \
./src/testing/flow_test_mod.o \
./src/testing/loop_breaker_test_mod.o \
./src/testing/accumulate_flow_test_mod.o \
./src/base/parameters_mod.o \
./src/base/check_return_code_netcdf_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_logger_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_interface_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_io_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_io_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_interface_mod.o \
./src/testing/latlon_hd_and_lake_model_test_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_lake_model_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_lake_model_interface_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_lake_model_io_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_hd_model_io_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_hd_model_interface_mod.o \
./src/icosohedral_hd_and_lake_model/grid_information_mod.o \
./src/testing/icosohedral_hd_and_lake_model_test_mod.o \
./src/testing/map_non_coincident_grids_test_mod.o \
./src/testing/manual_fruit_basket.o \
./src/testing/manual_fruit_basket_driver.o

MODS += \
./fruit.mod \
./fruit_util.mod \
./unstructured_grid_mod.mod \
./precision_mod.mod \
./pointer_mod.mod \
./coords_mod.mod \
./doubly_linked_list_link_mod.mod \
./doubly_linked_list_mod.mod \
./doubly_linked_list_test_module.mod \
./subfield_mod.mod \
./subfield_test_mod.mod \
./field_section_mod.mod \
./field_section_test_mod.mod \
./map_non_coincident_grids_mod.mod \
./area_mod.mod \
./area_test_mod.mod \
./break_loops_mod.mod \
./break_loops_driver_mod.mod \
./cotat_parameters_mod.mod \
./cotat_plus.mod \
./cotat_plus_driver_mod.mod \
./cotat_plus_test_mod.mod \
./flow.mod \
./flow_test_mod.mod \
./loop_breaker_mod.mod \
./loop_breaker_test_mod.mod \
./accumulate_flow_icon_simple_interface.mod \
./accumulate_flow_test_mod.mod \
./accumulate_flow_mod.mod \
./accumulate_flow_driver_mod.mod \
./manual_fruit_basket.mod \
./manual_fruit_basket_driver.mod \
./flow_accumulation_algorithm_mod.mod \
./check_return_code_netcdf_mod.mod \
./parameters_mod.mod \
./latlon_hd_model_mod.mod \
./latlon_lake_logger_mod.mod \
./latlon_lake_model_mod.mod \
./latlon_lake_model_interface_mod.mod \
./latlon_lake_model_io_mod.mod \
./latlon_lake_model_retrieve_lake_numbers.mod \
./latlon_hd_model_io_mod.mod \
./latlon_hd_model_interface_mod.mod \
./latlon_hd_model_driver.mod \
./map_non_coincident_grids_test_mod.mod \
./cotat_plus_latlon_to_icon_simple_interface.mod \
./latlon_hd_and_lake_model_test_mod.mod \
./icosohedral_hd_and_lake_model_test_mod.mod \
./icon_to_latlon_landsea_downscaler.mod \
./icon_to_latlon_landsea_downscaler_simple_interface.mod \
./latlon_to_icon_loop_breaker_simple_interface.mod \
./cross_grid_mapper_latlon_to_icon_simple_interface.mod \
./cross_grid_mapper.mod \
./convert_rdirs_to_indices.mod \
./icosohedral_lake_model_mod.mod \
./icosohedral_lake_model_io_mod.mod \
./icosohedral_lake_model_interface_mod.mod \
./icosohedral_hd_model_interface_mod.mod \
./icosohedral_hd_model_io_mod.mod \
./icosohedral_hd_model_mod.mod \
./icosohedral_hd_model_driver.mod \
./grid_information_mod.mod

# Each subdirectory must supply rules for building sources it contributes
ifeq ($(FORTRAN),$(GFORTRAN))
src/testing/fruit.o: $(FRUIT_LOC)/fruit.f90
	@echo 'Building file: $<'
	@echo 'Invoking: GNU Fortran Compiler'
	$(FORTRAN) $(FLAGS) -funderscoring -cpp -O0 -g -Wall -c -fmessage-length=0 -fPIC -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '
else
src/testing/fruit.o: $(FRUIT_LOC)/fruit.f90
	@echo 'Building file: $<'
	@echo 'Invoking: NAG Fortran Compiler'
	$(FORTRAN) $(FLAGS) -O0 -g -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '
endif

src/testing/fruit.o: $(FRUIT_LOC)/fruit.f90

ifeq ($(FORTRAN),$(GFORTRAN))
src/%.o: ../src/%.f90
	@echo 'Building file: $<'
	@echo 'Invoking: GNU Fortran Compiler'
	$(FORTRAN) $(FLAGS) -funderscoring -cpp -O0 -g -Wall -c -fmessage-length=0 -fPIC -o "$@" "$<" "-I ../include" "-I $(NETCDF_F)/include"
	@echo 'Finished building: $<'
	@echo ' '
else ifeq ($(FORTRAN),$(INTELFORTRAN))
src/%.o: ../src/%.f90
	@echo 'Invoking: Intel Fortran Compiler'
	@echo 'Building file: $<'
	$(FORTRAN) $(FLAGS) -O0 -fpp -g -c -o "$@" "$<" "-I ../include" "-I $(NETCDF_F)/include"
	@echo 'Finished building: $<'
	@echo ' '
else
src/%.o: ../src/%.f90
	@echo 'Invoking: NAG Fortran Compiler'
	@echo 'Building file: $<'
	$(FORTRAN) $(FLAGS) -O0 -fpp -g -c -o "$@" "$<" "-I ../include" "-I $(NETCDF_F)/include"
	@echo 'Finished building: $<'
	@echo ' '
endif

src/base/area_mod.o: ../src/base/area_mod.f90 src/base/coords_mod.o src/algorithms/cotat_parameters_mod.o src/base/doubly_linked_list_mod.o src/base/field_section_mod.o src/base/precision_mod.o src/base/subfield_mod.o

src/testing/area_test_mod.o: ../src/testing/area_test_mod.f90 src/base/area_mod.o src/base/coords_mod.o src/algorithms/cotat_parameters_mod.o src/testing/fruit.o

src/algorithms/break_loops_mod.o: ../src/algorithms/break_loops_mod.f90 src/algorithms/loop_breaker_mod.o

src/algorithms/break_loops_driver_mod.o: ../src/drivers/break_loops_driver_mod.f90 src/algorithms/break_loops_mod.o

src/base/coords_mod.o: ../src/base/coords_mod.f90 src/base/unstructured_grid_mod.o

src/algorithms/cotat_parameters_mod.o: ../src/algorithms/cotat_parameters_mod.f90

src/algorithms/cotat_plus.o: ../src/algorithms/cotat_plus.f90 src/base/area_mod.o src/base/coords_mod.o src/algorithms/cotat_parameters_mod.o src/algorithms/map_non_coincident_grids_mod.o

src/algorithms/map_non_coincident_grids_mod.o: ../src/algorithms/map_non_coincident_grids_mod.f90 src/base/precision_mod.o src/base/coords_mod.o src/base/field_section_mod.o src/base/subfield_mod.o src/base/pointer_mod.o

src/drivers/cotat_plus_driver_mod.o: ../src/drivers/cotat_plus_driver_mod.f90 src/algorithms/cotat_plus.o src/algorithms/map_non_coincident_grids_mod.o

src/algorithms/cotat_plus_test_mod.o: ../src/testing/cotat_plus_test_mod.f90 src/algorithms/cotat_parameters_mod.o src/algorithms/cotat_plus.o src/testing/fruit.o

src/algorithms/flow.o: ../src/algorithms/flow.f90 src/base/area_mod.o src/base/coords_mod.o src/algorithms/cotat_parameters_mod.o

src/algorithms/flow_test_mod.o: ../src/algorithms/flow_test_mod.f90 src/algorithms/cotat_parameters_mod.o src/algorithms/flow.o src/testing/fruit.o

src/base/doubly_linked_list_link_mod.o: ../src/base/doubly_linked_list_link_mod.f90

src/base/doubly_linked_list_mod.o: ../src/base/doubly_linked_list_mod.f90 src/base/coords_mod.o src/base/doubly_linked_list_link_mod.o

src/testing/doubly_linked_list_test_module.o: ../src/testing/doubly_linked_list_test_module.f90 src/base/coords_mod.o src/base/doubly_linked_list_mod.o src/testing/fruit.o

src/base/field_section_mod.o: ../src/base/field_section_mod.f90 src/base/coords_mod.o

src/testing/field_section_test_mod.o: ../src/testing/field_section_test_mod.f90 src/base/coords_mod.o src/base/field_section_mod.o src/testing/fruit.o

src/algorithms/loop_breaker_mod.o: ../src/algorithms/loop_breaker_mod.f90 src/base/coords_mod.o src/base/doubly_linked_list_mod.o src/base/field_section_mod.o

src/testing/loop_breaker_test_mod.o: ../src/testing/loop_breaker_test_mod.f90 src/testing/fruit.o src/algorithms/loop_breaker_mod.o

src/algorithms/accumulate_flow_mod.o: ../src/algorithms/accumulate_flow_mod.f90 src/algorithms/flow_accumulation_algorithm_mod.o src/base/convert_rdirs_to_indices.o

src/drivers/accumulate_flow_driver_mod.o: ../src/drivers/accumulate_flow_driver_mod.f90 src/algorithms/accumulate_flow_mod.o

src/testing/accumulate_flow_test_mod.o: ../src/testing/accumulate_flow_test_mod.f90 src/algorithms/accumulate_flow_mod.o src/testing/fruit.o

src/testing/manual_fruit_basket.o: ../src/testing/manual_fruit_basket.f90 src/testing/area_test_mod.o src/testing/cotat_plus_test_mod.o src/testing/doubly_linked_list_test_module.o src/testing/field_section_test_mod.o src/testing/fruit.o src/testing/loop_breaker_test_mod.o src/testing/subfield_test_mod.o src/testing/latlon_hd_and_lake_model_test_mod.o src/testing/accumulate_flow_test_mod.o src/testing/icosohedral_hd_and_lake_model_test_mod.o

src/testing/manual_fruit_basket_driver.o: ../src/testing/manual_fruit_basket_driver.f90 src/testing/fruit.o src/testing/manual_fruit_basket.o

src/command_line_drivers/cotat_plus_latlon_to_icon_simple_interface.o: ../src/command_line_drivers/cotat_plus_latlon_to_icon_simple_interface.f90 src/algorithms/cotat_plus.o src/check_return_code_netcdf_mod.o

src/base/precision_mod.o: ../src/base/precision_mod.f90

src/base/subfield_mod.o: ../src/base/subfield_mod.f90 src/base/coords_mod.o

src/base/subfield_test_mod.o: ../src/base/subfield_test_mod.f90 src/base/coords_mod.o src/testing/fruit.o src/base/subfield_mod.o

src/algorithms/flow_accumulation_algorithm_mod.o: ../src/algorithms/flow_accumulation_algorithm_mod.f90 src/base/subfield_mod.o src/base/coords_mod.o

src/unstructured_grid_mod.o: ../src/unstructured_grid_mod.f90

src/base/pointer_mod.o: ../src/base/pointer_mod.f90

src/check_return_code_netcdf_mod.o: ../src/check_return_code_netcdf_mod.f90

src/parameters_mod.o: ../src/parameters_mod.f90

src/latlon_hd_and_lake_model/latlon_hd_model_driver.o: ../src/latlon_hd_and_lake_model/latlon_hd_model_driver.f90 src/latlon_hd_and_lake_model/latlon_hd_model_interface_mod.o src/parameters_mod.o

src/latlon_hd_and_lake_model/latlon_hd_model_interface_mod.o: ../src/latlon_hd_and_lake_model/latlon_hd_model_interface_mod.f90 src/latlon_hd_and_lake_model/latlon_hd_model_mod.o src/latlon_hd_and_lake_model/latlon_hd_model_io_mod.o

src/latlon_hd_and_lake_model/latlon_hd_model_io_mod.o: ../src/latlon_hd_and_lake_model/latlon_hd_model_io_mod.f90 src/latlon_hd_and_lake_model/latlon_hd_model_mod.o

src/latlon_hd_and_lake_model/atlon_lake_logger_mod.o: ../src/latlon_hd_and_lake_model/latlon_lake_logger_mod.f90

src/latlon_hd_and_lake_model/latlon_lake_model_mod.o: ../src/latlon_hd_and_lake_model/latlon_lake_model_mod.f90 src/latlon_hd_and_lake_model/latlon_lake_logger_mod.o

src/latlon_hd_and_lake_model/latlon_lake_model_interface_mod.o: ../src/latlon_hd_and_lake_model/latlon_lake_model_interface_mod.f90 src/latlon_hd_and_lake_model/latlon_lake_model_mod.o src/latlon_hd_and_lake_model/latlon_lake_model_io_mod.o

src/latlon_hd_and_lake_model/latlon_lake_model_io_mod.o: ../src/latlon_hd_and_lake_model/latlon_lake_model_io_mod.f90   src/latlon_hd_and_lake_model/latlon_lake_model_mod.o src/parameters_mod.o src/check_return_code_netcdf_mod.o

src/latlon_hd_and_lake_model/latlon_lake_model_retrieve_lake_numbers.o: ../src/latlon_hd_and_lake_model/latlon_lake_model_retrieve_lake_numbers.f90 src/latlon_hd_and_lake_model/latlon_lake_model_mod.o src/latlon_hd_and_lake_model/latlon_lake_model_io_mod.o src/parameters_mod.o

src/latlon_hd_and_lake_model/latlon_lake_model_lake_number_retrieval_driver.o: ../src/latlon_hd_and_lake_model/latlon_lake_model_lake_number_retrieval_driver.f90 src/latlon_hd_and_lake_model/latlon_lake_model_retrieve_lake_numbers.o src/parameters_mod.o

src/latlon_hd_and_lake_model/latlon_hd_model_mod.o: ../src/latlon_hd_and_lake_model/latlon_hd_model_mod.f90 src/latlon_hd_and_lake_model/latlon_lake_model_interface_mod.o

src/testing/latlon_hd_and_lake_model_test_mod.o: ../src/testing/latlon_hd_and_lake_model_test_mod.f90 src/latlon_hd_and_lake_model/latlon_hd_model_interface_mod.o src/latlon_hd_and_lake_model/latlon_hd_model_mod.o src/latlon_hd_and_lake_model/latlon_lake_model_retrieve_lake_numbers.o

src/testing/icosohedral_hd_and_lake_model_test_mod.o: ../src/testing/icosohedral_hd_and_lake_model_test_mod.f90 src/icosohedral_hd_and_lake_model/icosohedral_hd_model_interface_mod.o src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.o

src/testing/map_non_coincident_grids_test_mod.o: ../src/testing/map_non_coincident_grids_test_mod.f90 src/algorithms/map_non_coincident_grids_mod.o

src/algorithms/icon_to_latlon_landsea_downscaler.o: ../src/algorithms/icon_to_latlon_landsea_downscaler.f90

src/icon_to_latlon_landsea_downscaler_simple_interface.o: ../src/icon_to_latlon_landsea_downscaler_simple_interface.f90 src/icon_to_latlon_landsea_downscaler.o src/check_return_code_netcdf_mod.o

src/cross_grid_mapper.o: ../src/cross_grid_mapper.f90 src/base/precision_mod.o src/check_return_code_netcdf_mod.o src/base/coords_mod.o src/map_non_coincident_grids_mod.o

src/cross_grid_mapper_latlon_to_icon_simple_interface.o: ../src/cross_grid_mapper_latlon_to_icon_simple_interface.f90 src/check_return_code_netcdf_mod.o ../src/cross_grid_mapper.f90

src/accumulate_flow_icon_simple_interface.o: ../src/accumulate_flow_icon_simple_interface.f90  src/accumulate_flow_mod.o src/check_return_code_netcdf_mod.o

src/base/convert_rdirs_to_indices.o: ../src/base/convert_rdirs_to_indices.f90

src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.o: ../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.f90 src/icosohedral_hd_and_lake_model/icosohedral_lake_model_interface_mod.o

src/icosohedral_hd_and_lake_model/icosohedral_hd_model_io_mod.o: ../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_io_mod.f90 src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.o src/icosohedral_lake_model_mod/grid_information_mod.o

src/icosohedral_hd_and_lake_model/icosohedral_hd_model_interface_mod.o: ../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_interface_mod.f90 src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.o src/icosohedral_hd_and_lake_model/icosohedral_hd_model_io_mod.o src/icosohedral_lake_model_mod/grid_information_mod.o

src/icosohedral_hd_and_lake_model/icosohedral_lake_model_mod.o: ../src/icosohedral_hd_and_lake_model/icosohedral_lake_model_mod.f90

src/icosohedral_hd_and_lake_model/icosohedral_lake_model_io_mod.o: ../src/icosohedral_hd_and_lake_model/icosohedral_lake_model_io_mod.f90   src/icosohedral_hd_and_lake_model/icosohedral_lake_model_mod.o src/parameters_mod.o src/check_return_code_netcdf_mod.o src/icosohedral_hd_and_lake_model/grid_information_mod.o

src/icosohedral_hd_and_lake_model/icosohedral_lake_model_interface_mod.o: ../src/icosohedral_hd_and_lake_model/icosohedral_lake_model_interface_mod.f90 src/icosohedral_hd_and_lake_model/icosohedral_lake_model_mod.o src/icosohedral_hd_and_lake_model/icosohedral_lake_model_io_mod.o src/icosohedral_hd_and_lake_model/grid_information_mod.o

src/icosohedral_hd_and_lake_model/icosohedral_hd_model_driver.o: ../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_driver.f90 src/icosohedral_hd_and_lake_model/icosohedral_hd_model_interface_mod.o src/parameters_mod.o

src/icosohedral_lake_model_mod/grid_information_mod.o: ../src/icosohedral_hd_and_lake_model/grid_information_mod.f90
