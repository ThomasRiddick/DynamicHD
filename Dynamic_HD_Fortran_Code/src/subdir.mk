# Add inputs and outputs from these tool invocations to the build variables
F90_SRCS += \
$(FRUIT_LOC)/fruit.f90 \
../src/precision_mod.f90 \
../src/pointer_mod.f90 \
../src/unstructured_grid_mod.f90 \
../src/coords_mod.f90 \
../src/doubly_linked_list_link_mod.f90 \
../src/doubly_linked_list_mod.f90 \
../src/doubly_linked_list_test_module.f90 \
../src/subfield_mod.f90 \
../src/subfield_test_mod.f90 \
../src/field_section_mod.f90 \
../src/field_section_test_mod.f90 \
../src/map_non_coincident_grids_mod.f90 \
../src/area_mod.f90 \
../src/area_test_mod.f90 \
../src/break_loops_mod.f90 \
../src/break_loops_driver_mod.f90 \
../src/cotat_parameters_mod.f90 \
../src/cotat_plus.f90 \
../src/cotat_plus_driver_mod.f90 \
../src/cotat_plus_test_mod.f90 \
../src/flow.f90 \
../src/flow_test_mod.f90 \
../src/loop_breaker_mod.f90 \
../src/loop_breaker_test_mod.f90 \
../src/flow_accumulation_algorithm_mod.f90 \
../src/check_return_code_netcdf_mod.f90 \
../src/parameters_mod.f90 \
../src/accumulate_flow_mod.f90 \
../src/accumulate_flow_test_mod.f90 \
../src/latlon_hd_and_lake_model/latlon_lake_model_mod.f90 \
../src/latlon_hd_and_lake_model/latlon_lake_model_interface_mod.f90 \
../src/latlon_hd_and_lake_model/latlon_lake_model_io_mod.f90 \
../src/latlon_hd_and_lake_model/latlon_hd_model_io_mod.f90 \
../src/latlon_hd_and_lake_model/latlon_hd_model_interface_mod.f90 \
../src/latlon_hd_and_lake_model/latlon_hd_model_mod.f90 \
../src/latlon_hd_and_lake_model/latlon_hd_model_driver.f90 \
../src/latlon_hd_and_lake_model_test_mod.f90 \
../src/icosohedral_hd_and_lake_model_test_mod.f90 \
../src/cotat_plus_latlon_to_icon_simple_interface.f90 \
../src/map_non_coincident_grids_test_mod.f90 \
../src/manual_fruit_basket.f90 \
../src/manual_fruit_basket_driver.f90 \
../src/icon_to_latlon_landsea_downscaler.f90 \
../src/icon_to_latlon_landsea_downscaler_simple_interface.f90 \
../src/latlon_to_icon_loop_breaker_simple_interface.f90 \
../src/accumulate_flow_icon_simple_interface.f90 \
../src/cross_grid_mapper_latlon_to_icon_simple_interface.f90 \
../src/cross_grid_mapper.f90 \
../src/icosohedral_hd_and_lake_model/icosohedral_lake_model_mod.f90 \
../src/icosohedral_hd_and_lake_model/icosohedral_lake_model_io_mod.f90 \
../src/icosohedral_hd_and_lake_model/icosohedral_lake_model_interface_mod.f90 \
../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.f90 \
../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_io_mod.f90 \
../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_interface_mod.f90 \
../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_driver.f90 \
../src/icosohedral_hd_and_lake_model/grid_information_mod.f90

OBJS += \
./src/precision_mod.o \
./src/pointer_mod.o \
./src/unstructured_grid_mod.o \
./src/coords_mod.o \
./src/doubly_linked_list_link_mod.o \
./src/doubly_linked_list_mod.o \
./src/subfield_mod.o \
./src/field_section_mod.o \
./src/map_non_coincident_grids_mod.o \
./src/area_mod.o \
./src/break_loops_mod.o \
./src/break_loops_driver_mod.o \
./src/cotat_parameters_mod.o \
./src/cotat_plus.o \
./src/cotat_plus_driver_mod.o \
./src/flow.o \
./src/loop_breaker_mod.o \
./src/flow_accumulation_algorithm_mod.o \
./src/icon_to_latlon_landsea_downscaler.o \
./src/accumulate_flow_mod.o \
./src/cross_grid_mapper.o

COTAT_PLUS_LATLON_TO_ICON_SIMPLE_INTERFACE_OBJS += \
./src/cotat_plus_latlon_to_icon_simple_interface.o \
./src/check_return_code_netcdf_mod.o

ACCUMULATE_FLOW_ICON_SIMPLE_INTERFACE_OBJS += \
./src/accumulate_flow_icon_simple_interface.o \
./src/check_return_code_netcdf_mod.o

ICON_TO_LATLON_LANDSEA_DOWNSCALER_SIMPLE_INTERFACE_OBJS += \
./src/icon_to_latlon_landsea_downscaler_simple_interface.o \
./src/check_return_code_netcdf_mod.o

LATLON_TO_ICON_LOOP_BREAKER_SIMPLE_INTERFACE_OBJS += \
./src/latlon_to_icon_loop_breaker_simple_interface.o \
./src/check_return_code_netcdf_mod.o

LATLON_TO_ICON_CROSS_GRID_MAPPER_SIMPLE_INTERFACE_OBJS += \
./src/cross_grid_mapper_latlon_to_icon_simple_interface.o \
./src/check_return_code_netcdf_mod.o

LATLON_HD_AND_LAKE_MODEL_OBJS += \
./src/parameters_mod.o \
./src/check_return_code_netcdf_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_interface_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_io_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_io_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_interface_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_driver.o

ICOSOHEDRAL_HD_AND_LAKE_MODEL_OBJS += \
./src/parameters_mod.o \
./src/check_return_code_netcdf_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_lake_model_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_lake_model_io_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_lake_model_interface_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_hd_model_io_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_hd_model_interface_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_hd_model_driver.o \
./src/icosohedral_hd_and_lake_model/grid_information_mod.o

TEST_OBJS += \
./src/fruit.o \
./src/doubly_linked_list_test_module.o \
./src/subfield_test_mod.o \
./src/field_section_test_mod.o \
./src/area_test_mod.o \
./src/cotat_plus_test_mod.o \
./src/flow_test_mod.o \
./src/loop_breaker_test_mod.o \
./src/accumulate_flow_test_mod.o \
./src/parameters_mod.o \
./src/check_return_code_netcdf_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_interface_mod.o \
./src/latlon_hd_and_lake_model/latlon_lake_model_io_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_io_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_mod.o \
./src/latlon_hd_and_lake_model/latlon_hd_model_interface_mod.o \
./src/latlon_hd_and_lake_model_test_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_lake_model_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_lake_model_interface_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_lake_model_io_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_hd_model_io_mod.o \
./src/icosohedral_hd_and_lake_model/icosohedral_hd_model_interface_mod.o \
./src/icosohedral_hd_and_lake_model/grid_information_mod.o \
./src/icosohedral_hd_and_lake_model_test_mod.o \
./src/map_non_coincident_grids_test_mod.o \
./src/manual_fruit_basket.o \
./src/manual_fruit_basket_driver.o

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
./manual_fruit_basket.mod \
./manual_fruit_basket_driver.mod \
./flow_accumulation_algorithm_mod.mod \
./check_return_code_netcdf_mod.mod \
./parameters_mod.mod \
./latlon_hd_model_mod.mod \
./latlon_lake_model_mod.mod \
./latlon_lake_model_interface_mod.mod \
./latlon_lake_model_io_mod.mod \
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
./icosohedral_lake_model_mod.mod \
./icosohedral_lake_model_io_mod.mod \
./icosohedral_lake_model_interface_mod.mod \
./icosohedral_hd_model_interface_mod.mod \
./icosohedral_hd_model_io_mod.mod \
./icosohedral_hd_model_mod.mod \
./icosohedral_hd_model_driver.mod

# Each subdirectory must supply rules for building sources it contributes
ifeq ($(FORTRAN),$(GFORTRAN))
src/fruit.o: $(FRUIT_LOC)/fruit.f90
	@echo 'Building file: $<'
	@echo 'Invoking: GNU Fortran Compiler'
	$(FORTRAN) -funderscoring -cpp -O0 -g -Wall -c -fmessage-length=0 -fPIC -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '
else
src/fruit.o: $(FRUIT_LOC)/fruit.f90
	@echo 'Building file: $<'
	@echo 'Invoking: NAG Fortran Compiler'
	$(FORTRAN) -O0 -g -c -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '
endif

src/fruit.o: $(FRUIT_LOC)/fruit.f90

ifeq ($(FORTRAN),$(GFORTRAN))
src/%.o: ../src/%.f90
	@echo 'Building file: $<'
	@echo 'Invoking: GNU Fortran Compiler'
	$(FORTRAN) -funderscoring -cpp -O0 -g -Wall -c -fmessage-length=0 -fPIC -o "$@" "$<" "-I ../include" "-I $(NETCDF_F)/include"
	@echo 'Finished building: $<'
	@echo ' '
else ifeq ($(FORTRAN),$(INTELFORTRAN))
src/%.o: ../src/%.f90
	@echo 'Invoking: Intel Fortran Compiler'
	@echo 'Building file: $<'
	$(FORTRAN) -O0 -fpp -g -c -o "$@" "$<" -I ../include -I $(NETCDF_F)/include
	@echo 'Finished building: $<'
	@echo ' '
else
src/%.o: ../src/%.f90
	@echo 'Invoking: NAG Fortran Compiler'
	@echo 'Building file: $<'
	$(FORTRAN) -O0 -fpp -g -c -o "$@" "$<" -I ../include -I $(NETCDF_F)/include
	@echo 'Finished building: $<'
	@echo ' '
endif

src/area_mod.o: ../src/area_mod.f90 src/coords_mod.o src/cotat_parameters_mod.o src/doubly_linked_list_mod.o src/field_section_mod.o src/precision_mod.o src/subfield_mod.o

src/area_test_mod.o: ../src/area_test_mod.f90 src/area_mod.o src/coords_mod.o src/cotat_parameters_mod.o src/fruit.o

src/break_loops_mod.o: ../src/break_loops_mod.f90 src/loop_breaker_mod.o

src/break_loops_driver_mod.o: ../src/break_loops_driver_mod.f90 src/break_loops_mod.o

src/coords_mod.o: ../src/coords_mod.f90 src/unstructured_grid_mod.o

src/cotat_parameters_mod.o: ../src/cotat_parameters_mod.f90

src/cotat_plus.o: ../src/cotat_plus.f90 src/area_mod.o src/coords_mod.o src/cotat_parameters_mod.o src/map_non_coincident_grids_mod.o

src/map_non_coincident_grids_mod.o: ../src/map_non_coincident_grids_mod.f90 src/precision_mod.o src/coords_mod.o src/field_section_mod.o src/subfield_mod.o src/pointer_mod.o

src/cotat_plus_driver_mod.o: ../src/cotat_plus_driver_mod.f90 src/cotat_plus.o src/map_non_coincident_grids_mod.o

src/cotat_plus_test_mod.o: ../src/cotat_plus_test_mod.f90 src/cotat_parameters_mod.o src/cotat_plus.o src/fruit.o

src/flow.o: ../src/flow.f90 src/area_mod.o src/coords_mod.o src/cotat_parameters_mod.o

src/flow_test_mod.o: ../src/flow_test_mod.f90 src/cotat_parameters_mod.o src/flow.o src/fruit.o

src/doubly_linked_list_link_mod.o: ../src/doubly_linked_list_link_mod.f90

src/doubly_linked_list_mod.o: ../src/doubly_linked_list_mod.f90 src/coords_mod.o src/doubly_linked_list_link_mod.o

src/doubly_linked_list_test_module.o: ../src/doubly_linked_list_test_module.f90 src/coords_mod.o src/doubly_linked_list_mod.o src/fruit.o

src/field_section_mod.o: ../src/field_section_mod.f90 src/coords_mod.o

src/field_section_test_mod.o: ../src/field_section_test_mod.f90 src/coords_mod.o src/field_section_mod.o src/fruit.o

src/loop_breaker_mod.o: ../src/loop_breaker_mod.f90 src/coords_mod.o src/doubly_linked_list_mod.o src/field_section_mod.o

src/loop_breaker_test_mod.o: ../src/loop_breaker_test_mod.f90 src/fruit.o src/loop_breaker_mod.o

src/accumulate_flow_mod.o: ../src/accumulate_flow_mod.f90 src/flow_accumulation_algorithm_mod.o

src/accumulate_flow_test_mod.o: ../src/accumulate_flow_test_mod.f90 src/accumulate_flow_mod.o src/fruit.o

src/manual_fruit_basket.o: ../src/manual_fruit_basket.f90 src/area_test_mod.o src/cotat_plus_test_mod.o src/doubly_linked_list_test_module.o src/field_section_test_mod.o src/fruit.o src/loop_breaker_test_mod.o src/subfield_test_mod.o src/latlon_hd_and_lake_model_test_mod.o src/accumulate_flow_test_mod.o src/icosohedral_hd_and_lake_model_test_mod.o

src/manual_fruit_basket_driver.o: ../src/manual_fruit_basket_driver.f90 src/fruit.o src/manual_fruit_basket.o

src/cotat_plus_latlon_to_icon_simple_interface.o: ../src/cotat_plus_latlon_to_icon_simple_interface.f90 src/cotat_plus.o src/check_return_code_netcdf_mod.o

src/precision_mod.o: ../src/precision_mod.f90

src/subfield_mod.o: ../src/subfield_mod.f90 src/coords_mod.o

src/subfield_test_mod.o: ../src/subfield_test_mod.f90 src/coords_mod.o src/fruit.o src/subfield_mod.o

src/flow_accumulation_algorithm_mod.o: ../src/flow_accumulation_algorithm_mod.f90 src/subfield_mod.o src/coords_mod.o

src/unstructured_grid_mod.o: ../src/unstructured_grid_mod.f90

src/pointer_mod.o: ../src/pointer_mod.f90

src/check_return_code_netcdf_mod.o: ../src/check_return_code_netcdf_mod.f90

src/parameters_mod.o: ../src/parameters_mod.f90

src/latlon_hd_and_lake_model/latlon_hd_model_driver.o: ../src/latlon_hd_and_lake_model/latlon_hd_model_driver.f90 src/latlon_hd_and_lake_model/latlon_hd_model_interface_mod.o src/parameters_mod.o

src/latlon_hd_and_lake_model/latlon_hd_model_interface_mod.o: ../src/latlon_hd_and_lake_model/latlon_hd_model_interface_mod.f90 src/latlon_hd_and_lake_model/latlon_hd_model_mod.o src/latlon_hd_and_lake_model/latlon_hd_model_io_mod.o

src/latlon_hd_and_lake_model/latlon_hd_model_io_mod.o: ../src/latlon_hd_and_lake_model/latlon_hd_model_io_mod.f90 src/latlon_hd_and_lake_model/latlon_hd_model_mod.o

src/latlon_hd_and_lake_model/latlon_lake_model_mod.o: ../src/latlon_hd_and_lake_model/latlon_lake_model_mod.f90

src/latlon_hd_and_lake_model/latlon_lake_model_interface_mod.o: ../src/latlon_hd_and_lake_model/latlon_lake_model_interface_mod.f90 src/latlon_hd_and_lake_model/latlon_lake_model_mod.o src/latlon_hd_and_lake_model/latlon_lake_model_io_mod.o

src/latlon_hd_and_lake_model/latlon_lake_model_io_mod.o: ../src/latlon_hd_and_lake_model/latlon_lake_model_io_mod.f90   src/latlon_hd_and_lake_model/latlon_lake_model_mod.o src/parameters_mod.o src/check_return_code_netcdf_mod.o

src/latlon_hd_and_lake_model/latlon_hd_model_mod.o: ../src/latlon_hd_and_lake_model/latlon_hd_model_mod.f90 src/latlon_hd_and_lake_model/latlon_lake_model_interface_mod.o

src/latlon_hd_and_lake_model_test_mod.o: ../src/latlon_hd_and_lake_model_test_mod.f90 src/latlon_hd_and_lake_model/latlon_hd_model_interface_mod.o src/latlon_hd_and_lake_model/latlon_hd_model_mod.o

src/icosohedral_hd_and_lake_model_test_mod.o: ../src/icosohedral_hd_and_lake_model_test_mod.f90 src/icosohedral_hd_and_lake_model/icosohedral_hd_model_interface_mod.o src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.o

src/map_non_coincident_grids_mod.o: ../src/map_non_coincident_grids_mod.f90

src/map_non_coincident_grids_test_mod.o: ../src/map_non_coincident_grids_test_mod.f90 src/map_non_coincident_grids_mod.o

src/icon_to_latlon_landsea_downscaler.o: ../src/icon_to_latlon_landsea_downscaler.f90

src/icon_to_latlon_landsea_downscaler_simple_interface.o: ../src/icon_to_latlon_landsea_downscaler_simple_interface.f90 src/icon_to_latlon_landsea_downscaler.o src/check_return_code_netcdf_mod.o

src/cross_grid_mapper.o: ../src/cross_grid_mapper.f90 src/precision_mod.o src/check_return_code_netcdf_mod.o src/coords_mod.o src/map_non_coincident_grids_mod.o

src/cross_grid_mapper_latlon_to_icon_simple_interface.o: ../src/cross_grid_mapper_latlon_to_icon_simple_interface.f90 src/check_return_code_netcdf_mod.o ../src/cross_grid_mapper.f90

src/accumulate_flow_icon_simple_interface.o: ../src/accumulate_flow_icon_simple_interface.f90  src/accumulate_flow_mod.o src/check_return_code_netcdf_mod.o

src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.o: ../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.f90 src/icosohedral_hd_and_lake_model/icosohedral_lake_model_interface_mod.o

src/icosohedral_hd_and_lake_model/icosohedral_hd_model_io_mod.o: ../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_io_mod.f90 src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.o src/icosohedral_lake_model_mod/grid_information_mod.o

src/icosohedral_hd_and_lake_model/icosohedral_hd_model_interface_mod.o: ../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_interface_mod.f90 src/icosohedral_hd_and_lake_model/icosohedral_hd_model_mod.o src/icosohedral_hd_and_lake_model/icosohedral_hd_model_io_mod.o src/icosohedral_lake_model_mod/grid_information_mod.o

src/icosohedral_hd_and_lake_model/icosohedral_lake_model_mod.o: ../src/icosohedral_hd_and_lake_model/icosohedral_lake_model_mod.f90

src/icosohedral_hd_and_lake_model/icosohedral_lake_model_io_mod.o: ../src/icosohedral_hd_and_lake_model/icosohedral_lake_model_io_mod.f90   src/icosohedral_hd_and_lake_model/icosohedral_lake_model_mod.o src/parameters_mod.o src/check_return_code_netcdf_mod.o src/icosohedral_hd_and_lake_model/icosohedral_lake_model_io_mod.o src/icosohedral_hd_and_lake_model/grid_information_mod.o

src/icosohedral_hd_and_lake_model/icosohedral_lake_model_interface_mod.o: ../src/icosohedral_hd_and_lake_model/icosohedral_lake_model_interface_mod.f90 src/icosohedral_hd_and_lake_model/icosohedral_lake_model_mod.o src/icosohedral_hd_and_lake_model/icosohedral_lake_model_io_mod.o src/icosohedral_hd_and_lake_model/grid_information_mod.o

src/icosohedral_hd_and_lake_model/icosohedral_hd_model_driver.o: ../src/icosohedral_hd_and_lake_model/icosohedral_hd_model_driver.f90 src/icosohedral_hd_and_lake_model/icosohedral_hd_model_interface_mod.o src/parameters_mod.o

src/icosohedral_lake_model_mod/grid_information_mod.o: ../src/icosohedral_hd_and_lake_model/grid_information_mod.f90


