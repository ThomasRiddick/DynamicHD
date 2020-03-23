RM := rm -rf
USE_NAG := False
ifeq ($(shell uname -s),Darwin)
GFORTRAN :=  /usr/local/Cellar/gcc/9.2.0_3/bin/gfortran-9
NAGFORTRAN :=  /usr/local/bin/nagfor
EXT_SOURCE :=$(shell echo ${HOME} )/Documents/code/
NETCDF_F := /usr/local/Cellar/netcdf/4.7.3
NETCDF_F_LIB :=libnetcdff.dylib
else ifeq ($(shell uname -s),Linux)
GFORTRAN := $(shell which gfortran)
EXT_SOURCE :=$(shell echo ${HOME} )/HDancillarydata/code/
NETCDF_F := /sw/stretch-x64/netcdf/netcdf_fortran-4.4.4-gcc63/
NETCDF_F_LIB :=libnetcdff.so
else
$(error OS $(shell uname -s) not recognised)
endif

ifeq ($(USE_NAG),True)
ifdef NAGFORTRAN
	FORTRAN := $(NAGFORTRAN)
else
	FORTRAN := $(GFORTRAN)
endif
else
	FORTRAN := $(GFORTRAN)
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
all: Dynamic_HD_Fortran_Code COTAT_Plus_LatLon_To_Icon_Simple_Interface_Code LatLon_Lake_Model_Driver_Code Icon_To_LatLon_Landsea_Downscaler_Simple_Interface LatLon_Icon_Loop_Breaker_Simple_Interface_Code

# Tool invocations
tools_only: $(OBJS) $(USER_OBJS) COTAT_Plus_LatLon_To_Icon_Simple_Interface_Code LatLon_Lake_Model_Driver_Code Icon_To_LatLon_Landsea_Downscaler_Simple_Interface LatLon_Icon_Loop_Breaker_Simple_Interface_Code
compile_only: $(OBJS) $(USER_OBJS)

Dynamic_HD_Fortran_Code: $(OBJS) $(TEST_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "Dynamic_HD_Fortran_Exec" $(OBJS) $(TEST_OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

COTAT_Plus_LatLon_To_Icon_Simple_Interface_Code: $(OBJS) $(COTAT_PLUS_LATLON_TO_ICON_SIMPLE_INTERFACE_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "COTAT_Plus_LatLon_To_Icon_Fortran_Exec" $(OBJS) $(COTAT_PLUS_LATLON_TO_ICON_SIMPLE_INTERFACE_OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

LatLon_Icon_Loop_Breaker_Simple_Interface_Code: $(OBJS) $(LATLON_TO_ICON_LOOP_BREAKER_SIMPLE_INTERFACE_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "LatLon_To_Icon_Loop_Breaker_Fortran_Exec" $(OBJS) $(LATLON_TO_ICON_LOOP_BREAKER_SIMPLE_INTERFACE_OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

Icon_To_LatLon_Landsea_Downscaler_Simple_Interface: $(OBJS) $(ICON_TO_LATLON_LANDSEA_DOWNSCALER_SIMPLE_INTERFACE_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "Icon_To_LatLon_Landsea_Downscaler_Simple_Interface" $(OBJS) $(ICON_TO_LATLON_LANDSEA_DOWNSCALER_SIMPLE_INTERFACE_OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

LatLon_Lake_Model_Driver_Code: $(OBJS) $(LATLON_HD_AND_LAKE_MODEL_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "LatLon_Lake_Model_Driver_Exec" $(OBJS) $(LATLON_HD_AND_LAKE_MODEL_OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(EXECUTABLES) $(OBJS) $(TEST_OBJS) $(LATLON_HD_AND_LAKE_MODEL_OBJS) $(C_DEPS) Dynamic_HD_Fortran_Exec
	-$(RM) $(COTAT_PLUS_LATLON_TO_ICON_SIMPLE_INTERFACE_OBJS) COTAT_Plus_LatLon_To_Icon_Fortran_Exec
	-$(RM) $(ICON_TO_LATLON_LANDSEA_DOWNSCALER_SIMPLE_INTERFACE_OBJS)
	-$(RM) Icon_To_LatLon_Landsea_Downscaler_Simple_Interface
	-$(RM) LatLon_Lake_Model_Driver_Exec
	-$(RM) $(MODS)
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:
