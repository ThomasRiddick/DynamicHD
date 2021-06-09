# Add inputs and outputs from these tool invocations to the build variables
CPP_SRCS += \
../src/base/cell.cpp \
../src/base/grid.cpp \
../src/algorithms/connected_lsmask_generation_algorithm.cpp \
../src/algorithms/basin_evaluation_algorithm.cpp \
../src/algorithms/lake_filling_algorithm.cpp \
../src/algorithms/sink_filling_algorithm.cpp \
../src/algorithms/water_redistribution_algorithm.cpp \
../src/algorithms/catchment_computation_algorithm.cpp \
../src/algorithms/reduce_connected_areas_to_points_algorithm.cpp \
../src/algorithms/basin_post_processing_algorithm.cpp \
../src/algorithms/river_direction_determination_algorithm.cpp \
../src/algorithms/stream_following_algorithm.cpp \
../src/algorithms/carved_river_direction_burning_algorithm.cpp \
../src/algorithms/orography_creation_algorithm.cpp \
../src/drivers/filter_out_shallow_lakes.cpp \
../src/drivers/redistribute_water.cpp \
../src/drivers/determine_river_directions.cpp \
../src/drivers/create_connected_lsmask.cpp \
../src/drivers/fill_sinks.cpp \
../src/drivers/upscale_orography.cpp \
../src/drivers/burn_carved_rivers.cpp \
../src/drivers/reduce_connected_areas_to_points.cpp \
../src/drivers/fill_lakes.cpp \
../src/drivers/compute_catchments.cpp \
../src/drivers/evaluate_basins.cpp \
../src/drivers/follow_streams.cpp \
../src/drivers/create_orography.cpp \
../src/testing/test_fill_inks.cpp  \
../src/testing/test_lake_operators.cpp \
../src/testing/test_catchment_computation.cpp \
../src/testing/test_evaluate_basins.cpp \
../src/testing/test_grid.cpp \
../src/testing/test_determine_river_directions.cpp \
../src/testing/test_orography_creation.cpp \
../src/command_line_drivers/determine_river_directions_icon_simple_interface.cpp \
../src/command_line_drivers/compute_catchments_icon_simple_interface.cpp \
../src/command_line_drivers/sink_filling_icon_simple_interface.cpp \
../src/command_line_drivers/evaluate_basins_simple_interface.cpp

USER_OBJS += \
./src/base/cell.o \
./src/base/grid.o \
./src/algorithms/sink_filling_algorithm.o \
./src/algorithms/connected_lsmask_generation_algorithm.o \
./src/algorithms/carved_river_direction_burning_algorithm.o \
./src/algorithms/reduce_connected_areas_to_points_algorithm.o \
./src/algorithms/catchment_computation_algorithm.o \
./src/algorithms/lake_filling_algorithm.o \
./src/algorithms/basin_evaluation_algorithm.o \
./src/algorithms/basin_post_processing_algorithm.o \
./src/algorithms/river_direction_determination_algorithm.o \
./src/algorithms/water_redistribution_algorithm.o \
./src/algorithms/stream_following_algorithm.o \
./src/algorithms/orography_creation_algorithm.o \
./src/drivers/create_connected_lsmask.o \
./src/drivers/fill_sinks.o \
./src/drivers/upscale_orography.o \
./src/drivers/burn_carved_rivers.o \
./src/drivers/reduce_connected_areas_to_points.o \
./src/drivers/fill_lakes.o \
./src/drivers/compute_catchments.o \
./src/drivers/evaluate_basins.o \
./src/drivers/determine_river_directions.o \
./src/drivers/redistribute_water.o \
./src/drivers/filter_out_shallow_lakes.o \
./src/drivers/follow_streams.o \
./src/drivers/create_orography.o

FS_ICON_SI_OBJS += \
./src/command_line_drivers/sink_filling_icon_simple_interface.o

CC_ICON_SI_OBJS += \
./src/command_line_drivers/compute_catchments_icon_simple_interface.o

DRD_ICON_SI_OBJS += \
./src/command_line_drivers/determine_river_directions_icon_simple_interface.o

EB_ICON_SI_OBJS += \
./src/command_line_drivers/evaluate_basins_simple_interface.o

TEST_OBJS += \
./src/testing/test_fill_sinks.o \
./src/testing/test_lake_operators.o \
./src/testing/test_catchment_computation.o \
./src/testing/test_evaluate_basins.o \
./src/testing/test_grid.o \
./src/testing/test_determine_river_directions.o \
./src/testing/test_orography_creation.o

CPP_DEPS += \
./src/base/cell.d \
./src/base/grid.d \
./src/algorithms/connected_lsmask_generation_algorithm.d \
./src/algorithms/carved_river_direction_burning_algorithm.d \
./src/algorithms/reduce_connected_areas_to_points_algorithm.d \
./src/algorithms/sink_filling_algorithm.d \
./src/algorithms/catchment_computation_algorithm.d \
./src/algorithms/basin_evaluation_algorithm.d \
./src/algorithms/basin_post_processing_algorithm.d \
./src/algorithms/lake_filling_algorithm.d \
./src/algorithms/water_redistribution_algorithm.d \
./src/algorithms/river_direction_determination_algorithm.d \
./src/algorithms/orography_creation_algorithm.d \
./src/algorithms/stream_following_algorithm.d \
./src/drivers/create_connected_lsmask.d \
./src/drivers/fill_sinks.d \
./src/drivers/upscale_orography.d \
./src/drivers/burn_carved_rivers.d \
./src/drivers/reduce_connected_areas_to_points.d \
./src/drivers/fill_lakes.d \
./src/drivers/compute_catchments.d \
./src/drivers/determine_river_directions.d \
./src/drivers/evaluate_basins.d \
./src/drivers/redistribute_water.d \
./src/drivers/follow_streams.d \
./src/drivers/filter_out_shallow_lakes.d \
./src/drivers/create_orography.d \
./src/testing/test_lake_operators.d \
./src/testing/test_catchment_computation.d \
./src/testing/test_evaluate_basins.d \
./src/testing/test_grid.d \
./src/testing/test_determine_river_directions.d \
./src/testing/test_fill_sinks.d \
./src/testing/test_orography_creation.d \
./src/command_line_drivers/evaluate_basins_simple_interface.d \
./src/command_line_drivers/sink_filling_icon_simple_interface.d \
./src/command_line_drivers/compute_catchments_icon_simple_interface.d \
./src/command_line_drivers/determine_river_directions_icon_simple_interface.d

# Each subdirectory must supply rules for building sources it contributes
src/base/%.o: ../src/base/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	$(GPLUSPLUS) -I"../include" -I"../." -I"../src" $(INCLUDE) -O3 -Wall -c -fmessage-length=0 -std=gnu++11 $(FLAGS) $(STDLIB_OPT) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/algorithms/%.o: ../src/algorithms/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	$(GPLUSPLUS) -I"../include" -I"../." -I"../src" $(INCLUDE) -O3 -Wall -c -fmessage-length=0 -std=gnu++11 $(FLAGS) $(STDLIB_OPT) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/drivers/%.o: ../src/drivers/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	$(GPLUSPLUS) -I"../include" -I"../." -I"../src" $(INCLUDE) -O3 -Wall -c -fmessage-length=0 -std=gnu++11 $(FLAGS) $(STDLIB_OPT) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/testing/%.o: ../src/testing/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	$(GPLUSPLUS) -I"../include" -I"../." -I"../src" $(INCLUDE) -O3 -Wall -c -fmessage-length=0 -std=gnu++11 $(FLAGS) $(STDLIB_OPT) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/command_line_drivers/%.o: ../src/command_line_drivers/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	$(GPLUSPLUS) -I"../include" -I"../." -I"../src" $(INCLUDE) -O3 -Wall -c -fmessage-length=0 -std=gnu++11 $(FLAGS) $(STDLIB_OPT) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


