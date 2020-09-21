RM := rm -rf
USE_NAG := False
USE_INTEL := False
ifeq ($(shell uname -s),Darwin)
GFORTRAN :=  /usr/local/Cellar/gcc/9.2.0_3/bin/gfortran-9
NAGFORTRAN :=  /usr/local/bin/nagfor
EXT_SOURCE :=$(shell echo ${HOME} )/Documents/code/
GFORTRAN_NETCDF_F := /usr/local/Cellar/netcdf/4.7.3
NETCDF_F_LIB :=libnetcdff.dylib
else ifeq ($(shell hostname -d),hpc.dkrz.de)
GFORTRAN := $(shell which gfortran)
NAGFORTRAN :=  $(shell which nagfor)
INTELFORTRAN := $(shell which ifort)
EXT_SOURCE :=$(shell echo ${HOME} )/HDancillarydata/code/
GFORTRAN_NETCDF_F := /sw/rhel6-x64/netcdf/netcdf_fortran-4.4.4-gcc64
NAG_NETCDF_F := /sw/rhel6-x64/netcdf/netcdf_fortran-4.4.2-static-nag60
INTEL_NETCDF_F := /sw/rhel6-x64/netcdf/netcdf_fortran-4.4.3-intel14
NETCDF_F_LIB :=libnetcdff.so
else ifeq ($(shell uname -s),Linux)
GFORTRAN := $(shell which gfortran)
NAGFORTRAN :=  $(shell which nagfor)
INTELFORTRAN := $(shell which ifort)
EXT_SOURCE :=$(shell echo ${HOME} )/HDancillarydata/code/
GFORTRAN_NETCDF_F := /sw/stretch-x64/netcdf/netcdf_fortran-4.4.4-gcc63
NAG_NETCDF_F := /sw/stretch-x64/netcdf/netcdf_fortran-4.4.4-nag70
INTEL_NETCDF_F := /sw/stretch-x64/netcdf/netcdf_fortran-4.4.4-intel17
NETCDF_F_LIB :=libnetcdff.so
else
$(error OS $(shell uname -s) not recognised)
endif
ifeq ($(USE_INTEL),True)
ifeq ($(USE_NAG),True)
$(warning 'USE_INTEL option overrides USE_NAG option')
endif
ifdef INTELFORTRAN
FORTRAN := $(INTELFORTRAN)
NETCDF_F := $(INTEL_NETCDF_F)
else
$(error 'Intel compiler not set')
endif
else ifeq ($(USE_NAG),True)
ifdef NAGFORTRAN
FORTRAN := $(NAGFORTRAN)
NETCDF_F := $(NAG_NETCDF_F)
else
$(error 'NAG compiler not set')
endif
else
FORTRAN := $(GFORTRAN)
NETCDF_F := $(GFORTRAN_NETCDF_F)
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
all: Dynamic_HD_Fortran_Code COTAT_Plus_LatLon_To_Icon_Simple_Interface_Code LatLon_Lake_Model_Driver_Code Icon_To_LatLon_Landsea_Downscaler_Simple_Interface LatLon_Icon_Loop_Breaker_Simple_Interface_Code LatLon_To_Icon_Cross_Grid_Mapper_Simple_Interface Accumulate_Flow_Icon_Simple_Interface Icosohedral_Lake_Model_Driver_Code

# Tool invocations
tools_only: $(OBJS) $(USER_OBJS) COTAT_Plus_LatLon_To_Icon_Simple_Interface_Code LatLon_Lake_Model_Driver_Code Icon_To_LatLon_Landsea_Downscaler_Simple_Interface LatLon_Icon_Loop_Breaker_Simple_Interface_Code LatLon_To_Icon_Cross_Grid_Mapper_Simple_Interface Accumulate_Flow_Icon_Simple_Interface
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

LatLon_To_Icon_Cross_Grid_Mapper_Simple_Interface: $(OBJS) $(LATLON_TO_ICON_CROSS_GRID_MAPPER_SIMPLE_INTERFACE_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "LatLon_To_Icon_Cross_Grid_Mapper_Simple_Interface" $(OBJS) $(LATLON_TO_ICON_CROSS_GRID_MAPPER_SIMPLE_INTERFACE_OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

Accumulate_Flow_Icon_Simple_Interface: $(OBJS) $(ACCUMULATE_FLOW_ICON_SIMPLE_INTERFACE_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "Accumulate_Flow_Icon_Simple_Interface_Exec" $(OBJS) $(ACCUMULATE_FLOW_ICON_SIMPLE_INTERFACE_OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

LatLon_Lake_Model_Driver_Code: $(OBJS) $(LATLON_HD_AND_LAKE_MODEL_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "LatLon_Lake_Model_Driver_Exec" $(OBJS) $(LATLON_HD_AND_LAKE_MODEL_OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

Icosohedral_Lake_Model_Driver_Code: $(OBJS) $(ICOSOHEDRAL_HD_AND_LAKE_MODEL_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "Icosohedral_Lake_Model_Driver_Exec" $(OBJS) $(ICOSOHEDRAL_HD_AND_LAKE_MODEL_OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(EXECUTABLES) $(OBJS) $(TEST_OBJS) $(LATLON_HD_AND_LAKE_MODEL_OBJS) $(C_DEPS) Dynamic_HD_Fortran_Exec
	-$(RM) $(COTAT_PLUS_LATLON_TO_ICON_SIMPLE_INTERFACE_OBJS) COTAT_Plus_LatLon_To_Icon_Fortran_Exec
	-$(RM) $(ICON_TO_LATLON_LANDSEA_DOWNSCALER_SIMPLE_INTERFACE_OBJS)
	-$(RM) $(ACCUMULATE_FLOW_ICON_SIMPLE_INTERFACE_OBJS)
	-$(RM) $(LATLON_TO_ICON_LOOP_BREAKER_SIMPLE_INTERFACE_OBJS)
	-$(RM) $(ICOSOHEDRAL_HD_AND_LAKE_MODEL_OBJS)
	-$(RM) Icon_To_LatLon_Landsea_Downscaler_Simple_Interface
	-$(RM) LatLon_Lake_Model_Driver_Exec
	-$(RM) Accumulate_Flow_Icon_Simple_Interface_Exec
	-$(RM) LatLon_To_Icon_Loop_Breaker_Fortran_Exec
	-$(RM) LatLon_To_Icon_Cross_Grid_Mapper_Simple_Interface
	-$(RM) Icosohedral_Lake_Model_Driver_Exec
	-$(RM) $(MODS)
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:
