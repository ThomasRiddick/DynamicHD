RM := rm -rf
ifeq ($(shell uname -s),Darwin)
GPLUSPLUS := /usr/bin/g++ 
STDLIB_OPT := -stdlib=libc++
FLAGS :=
else ifeq ($(shell hostname -d),hpc.dkrz.de)
GPLUSPLUS := /sw/rhel6-x64/gcc/gcc-6.2.0/bin/g++
STDLIB_OPT := 
FLAGS :=-pthread
else ifeq ($(shell uname -s),Linux)
GPLUSPLUS := $(shell which g++) 
STDLIB_OPT := 
FLAGS :=-pthread
else 
$(error OS $(uname -s) not recognised)
endif

# All of the sources participating in the build are defined here
-include ../sources.mk
-include ../src/gtest/subdir.mk
-include ../src/subdir.mk
-include ../subdir.mk
-include ../objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: Dynamic_HD_Cpp_Code

# Tool invocations
Dynamic_HD_Cpp_Code: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(GPLUSPLUS)  -o "Dynamic_HD_Cpp_Exec" $(OBJS) $(USER_OBJS) $(FLAGS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(CC_DEPS)$(C++_DEPS)$(EXECUTABLES)$(OBJS)$(C_UPPER_DEPS)$(CXX_DEPS)$(CPP_DEPS)$(C_DEPS) Dynamic_HD_Cpp_Exec
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY: