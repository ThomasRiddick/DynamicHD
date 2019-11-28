# Add inputs and outputs from these tool invocations to the build variables
CPP_SRCS += \
../src/cell.cpp \
../src/connected_lsmask_generation_algorithm.cpp \
../src/create_connected_lsmask.cpp \
../src/fill_sinks.cpp \
../src/grid.cpp \
../src/sink_filling_algorithm.cpp \
../src/test_fill_sinks.cpp  \
../src/upscale_orography.cpp \
../src/burn_carved_rivers.cpp \
../src/carved_river_direction_burning_algorithm.cpp \
../src/reduce_connected_areas_to_points.cpp \
../src/reduce_connected_areas_to_points_algorithm.cpp \
../src/fill_lakes.cpp \
../src/lake_filling_algorithm.cpp \
../src/compute_catchments.cpp \
../src/catchment_computation_algorithm.cpp \
../src/test_lake_operators.cpp \
../src/test_catchment_computation.cpp \
../src/test_evaluate_basins.cpp \
../src/basin_evaluation_algorithm.cpp \
../src/basin_post_processing_algorithm.cpp \
../src/evaluate_basins.cpp \
../src/determine_river_directions_icon_simple_interface.cpp \
../src/compute_catchments_icon_simple_interface.cpp \
../src/sink_filling_icon_simple_interface.cpp \
../src/test_grid.cpp \
../src/river_direction_determination_algorithm.cpp \
../src/determine_river_directions.cpp \
../src/test_determine_river_directions.cpp \
../src/water_redistribution_algorithm.cpp \
../src/redistribute_water.cpp \
../src/filter_out_shallow_lakes.cpp \
../src/evaluate_basins_simple_interface.cpp

USER_OBJS += \
./src/cell.o \
./src/connected_lsmask_generation_algorithm.o \
./src/create_connected_lsmask.o \
./src/fill_sinks.o \
./src/grid.o \
./src/sink_filling_algorithm.o \
./src/upscale_orography.o \
./src/burn_carved_rivers.o \
./src/carved_river_direction_burning_algorithm.o \
./src/reduce_connected_areas_to_points.o \
./src/reduce_connected_areas_to_points_algorithm.o \
./src/fill_lakes.o \
./src/lake_filling_algorithm.o \
./src/compute_catchments.o \
./src/catchment_computation_algorithm.o \
./src/basin_evaluation_algorithm.o \
./src/basin_post_processing_algorithm.o \
./src/evaluate_basins.o \
./src/river_direction_determination_algorithm.o \
./src/determine_river_directions.o \
./src/water_redistribution_algorithm.o \
./src/redistribute_water.o \
./src/filter_out_shallow_lakes.o

FS_ICON_SI_OBJS += \
./src/sink_filling_icon_simple_interface.o

CC_ICON_SI_OBJS += \
./src/compute_catchments_icon_simple_interface.o

DRD_ICON_SI_OBJS += \
./src/determine_river_directions_icon_simple_interface.o

EB_ICON_SI_OBJS += \
./src/evaluate_basins_simple_interface.o

TEST_OBJS += \
./src/test_fill_sinks.o \
./src/test_lake_operators.o \
./src/test_catchment_computation.o \
./src/test_evaluate_basins.o \
./src/test_grid.o \
./src/test_determine_river_directions.o

CPP_DEPS += \
./src/cell.d \
./src/connected_lsmask_generation_algorithm.d \
./src/create_connected_lsmask.d \
./src/fill_sinks.d \
./src/grid.d \
./src/sink_filling_algorithm.d \
./src/test_fill_sinks.d \
./src/upscale_orography.d \
./src/burn_carved_rivers.d \
./src/carved_river_direction_burning_algorithm.d \
./src/reduce_connected_areas_to_points.d \
./src/reduce_connected_areas_to_points_algorithm.d \
./src/fill_lakes.d \
./src/lake_filling_algorithm.d \
./src/compute_catchments.d \
./src/catchment_computation_algorithm.d \
./src/test_lake_operators.d \
./src/test_catchment_computation.d \
./src/test_evaluate_basins.d \
./src/basin_evaluation_algorithm.d \
./src/basin_post_processsing_algorithm.d \
./src/evaluate_basins.d \
./src/sink_filling_icon_simple_interface.d \
./src/compute_catchments_icon_simple_interface.d \
./src/determine_river_directions_icon_simple_interface.d \
./src/test_grid.d \
./src/river_direction_determination_algorithm.d \
./src/determine_river_directions.d \
./src/test_determine_river_directions.d \
./src/water_redistribution_algorithm.d \
./src/redistribute_water.d \
./src/filter_out_shallow_lakes.d \
./src/evaluate_basins_simple_interface.d

# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	$(GPLUSPLUS) -I"../include" -I"../." -I"../src" $(INCLUDE) -O3 -Wall -c -fmessage-length=0 -std=gnu++11 $(FLAGS) $(STDLIB_OPT) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


