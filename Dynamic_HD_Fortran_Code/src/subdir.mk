# Add inputs and outputs from these tool invocations to the build variables
F90_SRCS += \
$(FRUIT_LOC)/fruit.f90 \
../src/precision_mod.f90 \
../src/coords_mod.f90 \
../src/doubly_linked_list_link_mod.f90 \
../src/doubly_linked_list_mod.f90 \
../src/doubly_linked_list_test_module.f90 \
../src/subfield_mod.f90 \
../src/subfield_test_mod.f90 \
../src/field_section_mod.f90 \
../src/field_section_test_mod.f90 \
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
../src/manual_fruit_basket.f90 \
../src/manual_fruit_basket_driver.f90

OBJS += \
./src/fruit.o \
./src/precision_mod.o \
./src/coords_mod.o \
./src/doubly_linked_list_link_mod.o \
./src/doubly_linked_list_mod.o \
./src/doubly_linked_list_test_module.o \
./src/subfield_mod.o \
./src/subfield_test_mod.o \
./src/field_section_mod.o \
./src/field_section_test_mod.o \
./src/area_mod.o \
./src/area_test_mod.o \
./src/break_loops_mod.o \
./src/break_loops_driver_mod.o \
./src/cotat_parameters_mod.o \
./src/cotat_plus.o \
./src/cotat_plus_driver_mod.o \
./src/cotat_plus_test_mod.o \
./src/flow.o \
./src/flow_test_mod.o \
./src/loop_breaker_mod.o \
./src/loop_breaker_test_mod.o \
./src/manual_fruit_basket.o \
./src/manual_fruit_basket_driver.o

MODS += \
./fruit.mod \
./fruit_util.mod \
./precision_mod.mod \
./coords_mod.mod \
./doubly_linked_list_link_mod.mod \
./doubly_linked_list_mod.mod \
./doubly_linked_list_test_module.mod \
./subfield_mod.mod \
./subfield_test_mod.mod \
./field_section_mod.mod \
./field_section_test_mod.mod \
./area_mod.mod \
./area_test_mod.mod \
./break_loops_mod.mod \
./break_loops_driver_mod.mod \
./cotat_parameters_mod.mod \
./cotat_plus.mod \
./cotat_plus_driver_mod.mod \
./cotat_plus_test_mod.mod \
./src/flow.mod \
./src/flow_test_mod.mod \
./loop_breaker_mod.mod \
./loop_breaker_test_mod.mod \
./manual_fruit_basket.mod \
./manual_fruit_basket_driver.mod

# Each subdirectory must supply rules for building sources it contributes
src/fruit.o: $(FRUIT_LOC)/fruit.f90
	@echo 'Building file: $<'
	@echo 'Invoking: GNU Fortran Compiler'
	$(GFORTRAN) -funderscoring -O0 -g -Wall -c -fmessage-length=0 -fPIC -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/fruit.o: $(FRUIT_LOC)/fruit.f90

src/%.o: ../src/%.f90
	@echo 'Building file: $<'
	@echo 'Invoking: GNU Fortran Compiler'
	$(GFORTRAN) -funderscoring -O0 -g -Wall -c -fmessage-length=0 -fPIC -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/area_mod.o: ../src/area_mod.f90 src/coords_mod.o src/cotat_parameters_mod.o src/doubly_linked_list_mod.o src/field_section_mod.o src/precision_mod.o src/subfield_mod.o

src/area_test_mod.o: ../src/area_test_mod.f90 src/area_mod.o src/coords_mod.o src/cotat_parameters_mod.o src/fruit.o

src/break_loops_mod.o: ../src/break_loops_mod.f90 src/loop_breaker_mod.o

src/break_loops_driver_mod.o: ../src/break_loops_driver_mod.f90 src/break_loops_mod.o

src/coords_mod.o: ../src/coords_mod.f90

src/cotat_parameters_mod.o: ../src/cotat_parameters_mod.f90

src/cotat_plus.o: ../src/cotat_plus.f90 src/area_mod.o src/coords_mod.o src/cotat_parameters_mod.o

src/cotat_plus_driver_mod.o: ../src/cotat_plus_driver_mod.f90 src/cotat_plus.o

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

src/manual_fruit_basket.o: ../src/manual_fruit_basket.f90 src/area_test_mod.o src/cotat_plus_test_mod.o src/doubly_linked_list_test_module.o src/field_section_test_mod.o src/fruit.o src/loop_breaker_test_mod.o src/subfield_test_mod.o

src/manual_fruit_basket_driver.o: ../src/manual_fruit_basket_driver.f90 src/fruit.o src/manual_fruit_basket.o

src/precision_mod.o: ../src/precision_mod.f90

src/subfield_mod.o: ../src/subfield_mod.f90 src/coords_mod.o

src/subfield_test_mod.o: ../src/subfield_test_mod.f90 src/coords_mod.o src/fruit.o src/subfield_mod.o


