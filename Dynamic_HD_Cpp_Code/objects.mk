#OBSOLETE - USE MESON BUILD SYSTEM INSTEAD
# OBJS :=
# USER_OBJS :=
# TEST_OBJS :=
# SI_OBJS   :=
# FS_ICON_SI_OBJS :=
# CC_ICON_SI_OBJS :=
# DRD_ICON_SI_OBJS :=
# EB_ICON_SI_OBJS :=
# BRB_ICON_SI_OBJS :=


# ifeq ($(shell uname -s),Darwin)
# LIBS :=  -L"$(NETCDFCXX)/lib" -lnetcdf-cxx4 -L"$(NETCDFC)/lib" -lnetcdf
# else ifeq ($(shell hostname -d),lvt.dkrz.de)
# LIBS := -L"$(NETCDFCXX)/lib" -lnetcdf_c++4 -L"$(NETCDFC)/lib" -lnetcdf
# else ifeq ($(shell uname -s),Linux)
# LIBS := -L"$(NETCDFCXX)/lib" -lnetcdf_c++4
# else
# LIBS :=
endif

