RM := rm -rf
ifeq ($(shell uname -s),Darwin)
GFORTRAN := /usr/local/bin/gfortran
EXT_SOURCE :=$(shell echo ${HOME} )/Documents/code/
else ifeq ($(shell uname -s),Linux)
GFORTRAN := $(shell which gfortran)
EXT_SOURCE :=$(shell echo ${HOME} )/external_code/
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
all: Dynamic_HD_Fortran_Code

# Tool invocations
Dynamic_HD_Fortran_Code: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	$(GFORTRAN) -o "Dynamic_HD_Fortran_Exec" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(EXECUTABLES)$(OBJS)$(C_DEPS) Dynamic_HD_Fortran_Exec
	-$(RM) $(MODS)
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY: