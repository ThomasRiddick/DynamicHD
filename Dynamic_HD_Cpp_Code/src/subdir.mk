################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cell.cpp \
../src/connected_lsmask_generation_algorithm.cpp \
../src/create_connected_lsmask.cpp \
../src/fill_sinks.cpp \
../src/grid.cpp \
../src/sink_filling_algorithm.cpp \
../src/test_fill_sinks.cpp 

OBJS += \
./src/cell.o \
./src/connected_lsmask_generation_algorithm.o \
./src/create_connected_lsmask.o \
./src/fill_sinks.o \
./src/grid.o \
./src/sink_filling_algorithm.o \
./src/test_fill_sinks.o 

CPP_DEPS += \
./src/cell.d \
./src/connected_lsmask_generation_algorithm.d \
./src/create_connected_lsmask.d \
./src/fill_sinks.d \
./src/grid.d \
./src/sink_filling_algorithm.d \
./src/test_fill_sinks.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	/usr/bin/g++ -I"/Users/thomasriddick/Documents/workspace/Dynamic_HD_Code/Dynamic_HD_Cpp_Code/include" -I"/Users/thomasriddick/Documents/workspace/Dynamic_HD_Code/Dynamic_HD_Cpp_Code" -I"/Users/thomasriddick/Documents/workspace/Dynamic_HD_Code/Dynamic_HD_Cpp_Code/src" -O3 -Wall -c -fmessage-length=0 -std=gnu++11 $(STDLIB_OPT) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


