RM := rm -rf
ifeq ($(shell uname -s),Darwin)
GFORTRAN := /usr/local/Cellar/gcc@6/6.5.0_2/bin/gfortran-6
EXT_SOURCE :=$(shell echo ${HOME} )/Documents/code/
NETCDF_F := /usr/local/Cellar/netcdf/4.6.3_1
NETCDF_F_LIB :=libnetcdff.dylib
else ifeq ($(shell uname -s),Linux)
GFORTRAN := $(shell which gfortran)
EXT_SOURCE :=$(shell echo ${HOME} )/HDancillarydata/code/
NETCDF_F := /sw/stretch-x64/netcdf/netcdf_fortran-4.4.4-gcc63/
NETCDF_F_LIB :=libnetcdff.so
else
$(error OS $(shell uname -s) not recognised)
endif
FRUIT_LOC := $(EXT_SOURCE)/fruit_3.4.1/src

# All of the sources participating in the build are defined here
-include ../sources.mk
-include ../src/subdir.mk
-include ../subdir.mk
-include ../objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

# Add inputs and outputs from these tool invocations to the build variables

# All Target
all: Dynamic_HD_Fortran_Code COTAT_Plus_LatLon_To_Icon_Simple_Interface_Code LatLon_Lake_Model_Driver_Code

# Tool invocations
compile_only: $(OBJS) $(USER_OBJS)

Dynamic_HD_Fortran_Code: $(OBJS) $(TEST_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(GFORTRAN) -o "Dynamic_HD_Fortran_Exec" $(OBJS) $(TEST_OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

COTAT_Plus_LatLon_To_Icon_Simple_Interface_Code: $(OBJS) $(COTAT_PLUS_LATLON_TO_ICON_SIMPLE_INTERFACE_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(GFORTRAN) -o "COTAT_Plus_LatLon_To_Icon_Fortran_Exec" $(OBJS) $(COTAT_PLUS_LATLON_TO_ICON_SIMPLE_INTERFACE_OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

LatLon_Lake_Model_Driver_Code: $(OBJS) $(LATLON_HD_AND_LAKE_MODEL_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(GFORTRAN) -o "LatLon_Lake_Model_Driver_Exec" $(OBJS) $(LATLON_HD_AND_LAKE_MODEL_OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(EXECUTABLES)$(OBJS)$(TEST_OBJS)$(LATLON_HD_AND_LAKE_MODEL_OBJS)$(C_DEPS) Dynamic_HD_Fortran_Exec
	-$(RM) $(COTAT_PLUS_LATLON_TO_ICON_SIMPLE_INTERFACE_OBJS) COTAT_Plus_LatLon_To_Icon_Fortran_Exec
	-$(RM) $(MODS)
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:
