RM := rm -rf
USE_CLANG := True
ifeq ($(shell uname -s),Darwin)
ifeq ($(USE_CLANG),True)
GPLUSPLUS := clang++
else
GPLUSPLUS := /usr/local/Cellar/gcc@10/10.4.0/bin/g++-10
endif
STDLIB_OPT :=
FLAGS := -DPROCESSED_CELL_COUNTER  -DUSE_NETCDFCPP -DDARWIN -g  -fsanitize=address
NETCDFC :=  /usr/local/Cellar/netcdf/4.9.0
NETCDFCXX :=  /usr/local/Cellar/netcdf-cxx/4.3.1
INCLUDE := -isystem"$(NETCDFCXX)/include" -isystem"$(NETCDFC)/include"
else ifeq ($(shell hostname -d),lvt.dkrz.de)
STDLIB_OPT :=
FLAGS :=-pthread -DUSE_NETCDFCPP -g
INCLUDE := -isystem"$(NETCDFCXX)/include" -isystem"$(NETCDFC)/include"
ifneq ($(USE_SETTINGS_FROM_CONFIGURE),True)
GPLUSPLUS :=  /sw/spack-levante/gcc-11.2.0-7jcqrc/bin/g++
NETCDFC := $(HOME)/sw-spack/netcdf-c-4.8.1-khy3ru
NETCDFCXX := $(HOME)/sw-spack/netcdf-cxx4-4.3.1-d54zya
endif
else ifeq ($(shell uname -s),Linux)
STDLIB_OPT :=
FLAGS :=-pthread -DUSE_NETCDFCPP
INCLUDE := -isystem"$(NETCDFCXX)/include" -isystem"$(NETCDFC)/include"
ifneq ($(USE_SETTINGS_FROM_CONFIGURE),True)
GPLUSPLUS := $(shell which g++)
NETCDFC := /sw/stretch-x64/netcdf/netcdf_c-4.6.1
NETCDFCXX := /sw/stretch-x64/netcdf/netcdf_cxx-4.3.0-gccsys
endif
else
$(error OS $(uname -s) not recognised)
endif

# All of the sources participating in the build are defined here
-include ../objects.mk
-include ../sources.mk
-include ../src/gtest/subdir.mk
-include ../src/subdir.mk
-include ../subdir.mk

EXECUTABLES += Dynamic_HD_Cpp_Exec Fill_Sinks_Icon_SI_Exec Compute_Catchments_SI_Exec Determine_River_Directions_SI_Exec Bifurcate_River_Basic_SI_Exec

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
all: Dynamic_HD_Cpp_Code #Fill_Sinks_Icon_SI_Exec Compute_Catchments_SI_Exec Determine_River_Directions_SI_Exec Evaluate_Basins_SI_Exec Bifurcate_River_Basic_SI_Exec

# Tool invocations
tools_only: $(USER_OBJS) Fill_Sinks_Icon_SI_Exec Compute_Catchments_SI_Exec Determine_River_Directions_SI_Exec Bifurcate_River_Basic_SI_Exec
compile_only: $(USER_OBJS)

Dynamic_HD_Cpp_Code: $(OBJS) $(USER_OBJS) $(TEST_OBJS)
	@echo 'Building target: $@'
	$(GPLUSPLUS)  -o "Dynamic_HD_Cpp_Exec" $(OBJS) $(USER_OBJS) $(TEST_OBJS) $(FLAGS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

Fill_Sinks_Icon_SI_Exec: $(USER_OBJS) $(FS_ICON_SI_OBJS)
	@echo 'Building target: $@'
	$(GPLUSPLUS)  -o "Fill_Sinks_Icon_SI_Exec" $(USER_OBJS) $(FS_ICON_SI_OBJS) $(FLAGS) $(LIBS) $(INCLUDE)
	@echo 'Finished building target: $@'
	@echo ' '

Compute_Catchments_SI_Exec: $(USER_OBJS) $(CC_ICON_SI_OBJS)
	@echo 'Building target: $@'
	$(GPLUSPLUS)  -o "Compute_Catchments_SI_Exec" $(USER_OBJS) $(CC_ICON_SI_OBJS) $(FLAGS) $(LIBS) $(INCLUDE)
	@echo 'Finished building target: $@'
	@echo ' '

Determine_River_Directions_SI_Exec: $(USER_OBJS) $(DRD_ICON_SI_OBJS)
	@echo 'Building target: $@'
	$(GPLUSPLUS)  -o "Determine_River_Directions_SI_Exec" $(USER_OBJS) $(DRD_ICON_SI_OBJS) $(FLAGS) $(LIBS) $(INCLUDE)
	@echo 'Finished building target: $@'
	@echo ' '

Evaluate_Basins_SI_Exec: $(USER_OBJS) $(EB_ICON_SI_OBJS)
	@echo 'Building target: $@'
	$(GPLUSPLUS)  -o "Evaluate_Basins_SI_Exec" $(USER_OBJS) $(EB_ICON_SI_OBJS) $(FLAGS) $(LIBS) $(INCLUDE)
	@echo 'Finished building target: $@'
	@echo ' '

Bifurcate_River_Basic_SI_Exec:$(USER_OBJS) $(BRB_ICON_SI_OBJS)
	@echo 'Building target: $@'
	$(GPLUSPLUS)  -o "Bifurcate_River_Basic_SI_Exec" $(USER_OBJS) $(BRB_ICON_SI_OBJS) $(FLAGS) $(LIBS) $(INCLUDE)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(CC_DEPS)$(C++_DEPS)$(EXECUTABLES)$(OBJS)$(USER_OBJS)$(TEST_OBJS)$(FS_ICON_SI_OBJS)$(CC_ICON_SI_OBJS)$(DRD_ICON_SI_OBJS)$(EB_ICON_SI_OBJS)$(BRB_ICON_SI_OBJS)$(C_UPPER_DEPS)$(CXX_DEPS)$(CPP_DEPS)$(C_DEPS)
	-@echo ' '

.PHONY: all clean tools_only compile_only
.SECONDARY:
