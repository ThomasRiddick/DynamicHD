RM := rm -rf
USE_NAG := False
USE_INTEL := False
FLAGS := -DUSE_LOGGING
USE_MPI ?= False
COMPILE_TRANSPOSED_LAKE_MODEL ?= False
ifeq ($(shell uname -s),Darwin)
ifeq ($(USE_MPI),True)
GFORTRAN :=  /usr/local/bin/mpifort
else
GFORTRAN :=  /usr/local/Cellar/gcc@10/10.4.0/bin/gfortran-10
endif
NAGFORTRAN :=  /usr/local/bin/nagfor
EXT_SOURCE :=$(shell echo ${HOME} )/Documents/code/
GFORTRAN_NETCDF_F := /usr/local/Cellar/netcdf-fortran/4.6.0
NETCDF_F_LIB :=libnetcdff.dylib
ifeq ($(COMPILE_TRANSPOSED_LAKE_MODEL),True)
TRANSPOSED_SOURCE_CODE_LOC := /Users/thomasriddick/Documents/Work/switched_code
FLAGS := -DTRANSPOSED_LAKE_MODEL
endif
ifeq ($(USE_MPI),True)
FLAGS := -DUSE_MPI -m64
endif
else ifeq ($(shell hostname -d),atos.local)
ifeq ($(USE_MPI),True)
GFORTRAN := $(shell which mpifort)
else
ifneq ($(USE_SETTINGS_FROM_CONFIGURE),True)
GFORTRAN := /sw/spack-levante/gcc-11.2.0-bcn7mb/bin/gfortran
endif
endif
NAGFORTRAN :=  $(shell which nagfor)
INTELFORTRAN := $(shell which ifort)
#EXT_SOURCE :=$(shell echo ${HOME} )/HDancillarydata_11Sept2019/code/
ifneq ($(USE_SETTINGS_FROM_CONFIGURE),True)
GFORTRAN_NETCDF_F :=/sw/spack-levante/netcdf-fortran-4.5.3-l2ulgp
NETCDF_F_LIB :=libnetcdff.so
endif
NAG_NETCDF_F :=..#/sw/rhel6-x64/netcdf/netcdf_fortran-4.4.2-static-nag60
INTEL_NETCDF_F :=..#/sw/rhel6-x64/netcdf/netcdf_fortran-4.4.3-intel14
ifeq ($(USE_MPI),True)
FLAGS := -DUSE_MPI -m64
endif
else ifeq ($(shell uname -s),Linux)
ifneq ($(USE_SETTINGS_FROM_CONFIGURE),True)
GFORTRAN := $(shell which gfortran)
GFORTRAN_NETCDF_F := /sw/stretch-x64/netcdf/netcdf_fortran-4.4.4-gcc63
NETCDF_F_LIB :=libnetcdff.so
endif
NAGFORTRAN :=  $(shell which nagfor)
INTELFORTRAN := $(shell which ifort)
EXT_SOURCE :=$(shell echo ${HOME} )/HDancillarydata/code/
NAG_NETCDF_F := /sw/stretch-x64/netcdf/netcdf_fortran-4.4.4-nag70
INTEL_NETCDF_F := /sw/stretch-x64/netcdf/netcdf_fortran-4.4.4-intel17
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
all: Dynamic_HD_Fortran_Code COTAT_Plus_LatLon_To_Icon_Simple_Interface_Code LatLon_Lake_Model_Driver_Code Icon_To_LatLon_Landsea_Downscaler_Simple_Interface LatLon_Icon_Loop_Breaker_Simple_Interface_Code LatLon_To_Icon_Cross_Grid_Mapper_Simple_Interface Accumulate_Flow_Icon_Simple_Interface Icosohedral_Lake_Model_Driver_Code LatLon_Lake_Number_Retrieval_Driver_Code

# Tool invocations
tools_only: $(OBJS) $(USER_OBJS) COTAT_Plus_LatLon_To_Icon_Simple_Interface_Code Icon_To_LatLon_Landsea_Downscaler_Simple_Interface LatLon_Icon_Loop_Breaker_Simple_Interface_Code LatLon_To_Icon_Cross_Grid_Mapper_Simple_Interface Accumulate_Flow_Icon_Simple_Interface
compile_only: $(OBJS) $(USER_OBJS)

Dynamic_HD_Fortran_Code: $(OBJS) $(TEST_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "Dynamic_HD_Fortran_Exec" $(OBJS) $(TEST_OBJS) $(USER_OBJS) $(FLAGS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

COTAT_Plus_LatLon_To_Icon_Simple_Interface_Code: $(OBJS) $(COTAT_PLUS_LATLON_TO_ICON_SIMPLE_INTERFACE_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "COTAT_Plus_LatLon_To_Icon_Fortran_Exec" $(OBJS) $(COTAT_PLUS_LATLON_TO_ICON_SIMPLE_INTERFACE_OBJS) $(USER_OBJS) $(FLAGS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

LatLon_Icon_Loop_Breaker_Simple_Interface_Code: $(OBJS) $(LATLON_TO_ICON_LOOP_BREAKER_SIMPLE_INTERFACE_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "LatLon_To_Icon_Loop_Breaker_Fortran_Exec" $(OBJS) $(LATLON_TO_ICON_LOOP_BREAKER_SIMPLE_INTERFACE_OBJS) $(USER_OBJS) $(FLAGS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

Icon_To_LatLon_Landsea_Downscaler_Simple_Interface: $(OBJS) $(ICON_TO_LATLON_LANDSEA_DOWNSCALER_SIMPLE_INTERFACE_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "Icon_To_LatLon_Landsea_Downscaler_Simple_Interface" $(OBJS) $(ICON_TO_LATLON_LANDSEA_DOWNSCALER_SIMPLE_INTERFACE_OBJS) $(USER_OBJS) $(FLAGS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

LatLon_To_Icon_Cross_Grid_Mapper_Simple_Interface: $(OBJS) $(LATLON_TO_ICON_CROSS_GRID_MAPPER_SIMPLE_INTERFACE_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "LatLon_To_Icon_Cross_Grid_Mapper_Simple_Interface" $(OBJS) $(LATLON_TO_ICON_CROSS_GRID_MAPPER_SIMPLE_INTERFACE_OBJS) $(USER_OBJS) $(FLAGS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

Accumulate_Flow_Icon_Simple_Interface: $(OBJS) $(ACCUMULATE_FLOW_ICON_SIMPLE_INTERFACE_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "Accumulate_Flow_Icon_Simple_Interface_Exec" $(OBJS) $(ACCUMULATE_FLOW_ICON_SIMPLE_INTERFACE_OBJS) $(USER_OBJS) $(FLAGS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

LatLon_Lake_Model_Driver_Code: $(OBJS) $(LATLON_HD_AND_LAKE_MODEL_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "LatLon_Lake_Model_Driver_Exec" $(OBJS) $(LATLON_HD_AND_LAKE_MODEL_OBJS) $(USER_OBJS) $(FLAGS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

LatLon_Lake_Number_Retrieval_Driver_Code: $(OBJS) $(LATLON_LAKE_NUMBER_RETRIEVAL_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "LatLon_Lake_Number_Retrieval_Driver_Exec" $(OBJS) $(LATLON_LAKE_NUMBER_RETRIEVAL_OBJS) $(USER_OBJS) $(LIBS) $(FLAGS)
	@echo 'Finished building target: $@'
	@echo ' '

Icosohedral_Lake_Model_Driver_Code: $(OBJS) $(ICOSOHEDRAL_HD_AND_LAKE_MODEL_OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(FORTRAN) -o "Icosohedral_Lake_Model_Driver_Exec" $(OBJS) $(ICOSOHEDRAL_HD_AND_LAKE_MODEL_OBJS) $(USER_OBJS) $(FLAGS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(EXECUTABLES) $(OBJS) $(TEST_OBJS) $(LATLON_HD_AND_LAKE_MODEL_OBJS) $(C_DEPS) Dynamic_HD_Fortran_Exec
	-$(RM) $(COTAT_PLUS_LATLON_TO_ICON_SIMPLE_INTERFACE_OBJS) COTAT_Plus_LatLon_To_Icon_Fortran_Exec
	-$(RM) $(ICON_TO_LATLON_LANDSEA_DOWNSCALER_SIMPLE_INTERFACE_OBJS)
	-$(RM) $(ACCUMULATE_FLOW_ICON_SIMPLE_INTERFACE_OBJS)
	-$(RM) $(LATLON_TO_ICON_LOOP_BREAKER_SIMPLE_INTERFACE_OBJS)
	-$(RM) $(LATLON_TO_ICON_CROSS_GRID_MAPPER_SIMPLE_INTERFACE_OBJS)
	-$(RM) $(ICOSOHEDRAL_HD_AND_LAKE_MODEL_OBJS)
	-$(RM) Icon_To_LatLon_Landsea_Downscaler_Simple_Interface
	-$(RM) LatLon_Lake_Model_Driver_Exec
	-$(RM) Accumulate_Flow_Icon_Simple_Interface_Exec
	-$(RM) LatLon_To_Icon_Loop_Breaker_Fortran_Exec
	-$(RM) LatLon_To_Icon_Cross_Grid_Mapper_Simple_Interface
	-$(RM) Icosohedral_Lake_Model_Driver_Exec
	-$(RM) LatLon_Lake_Number_Retrieval_Driver_Exec
	-$(RM) $(MODS)
	-@echo ' '

.PHONY: all clean compile_only tools_only
.SECONDARY:
