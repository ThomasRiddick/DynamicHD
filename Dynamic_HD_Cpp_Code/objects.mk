OBJS :=
USER_OBJS :=
TEST_OBJS :=
SI_OBJS   :=
FS_ICON_SI_OBJS :=
CC_ICON_SI_OBJS :=
DRD_ICON_SI_OBJS :=


ifeq ($(shell uname -s),Darwin)
LIBS :=  -lnetcdf_c++
else
LIBS :=
endif

