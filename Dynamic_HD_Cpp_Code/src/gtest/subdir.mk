################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../src/gtest/gtest-death-test.cc \
../src/gtest/gtest-filepath.cc \
../src/gtest/gtest-port.cc \
../src/gtest/gtest-printers.cc \
../src/gtest/gtest-test-part.cc \
../src/gtest/gtest-typed-test.cc \
../src/gtest/gtest_main.cc \
../src/gtest/gtest.cc 

CC_DEPS += \
./src/gtest/gtest-death-test.d \
./src/gtest/gtest-filepath.d \
./src/gtest/gtest-port.d \
./src/gtest/gtest-printers.d \
./src/gtest/gtest-test-part.d \
./src/gtest/gtest-typed-test.d \
./src/gtest/gtest_main.d \
./src/gtest/gtest.d 

OBJS += \
./src/gtest/gtest-death-test.o \
./src/gtest/gtest-filepath.o \
./src/gtest/gtest-port.o \
./src/gtest/gtest-printers.o \
./src/gtest/gtest-test-part.o \
./src/gtest/gtest-typed-test.o \
./src/gtest/gtest_main.o \
./src/gtest/gtest.o 


# Each subdirectory must supply rules for building sources it contributes
src/gtest/%.o: ../src/gtest/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	$(GPLUSPLUS) -I"../include" -I"../." -I"../src" -O3 -Wall -c -fmessage-length=0 -std=gnu++11 $(STDLIB_OPT) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


