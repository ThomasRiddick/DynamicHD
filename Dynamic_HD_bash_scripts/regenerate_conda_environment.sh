#!/bin/bash

no_modules=${1}

#Define module loading function
function load_module
{
module_name=$1
if [[ $(hostname -d) == "atos.local" ]]; then
  module load ${module_name}
else
  eval "eval `/usr/bin/tclsh /sw/share/Modules-4.2.1/libexec/modulecmd.tcl bash load ${module_name}`"
fi
}

#Define module unloading function (only works on hpc)
function unload_module
{
module_name=$1
if echo $LOADEDMODULES | fgrep -q ${module_name} ; then
  module unload $module_name
fi
}

if ! $no_modules ; then
  if [[ $(hostname -d) == "atos.local" ]]; then
    source /etc/profile
    unload_module netcdf_c
    unload_module imagemagick
    unload_module cdo/1.7.0-magicsxx-gcc48
    unload_module python
  else
    unload_module netcdf_c
    load_module anaconda3
  fi
fi

if ! $no_modules; then
  if [[ $(hostname -d) == "atos.local" ]]; then
    load_module python3
  else
    load_module anaconda3
  fi
fi

conda-env remove -n dyhdenv3
conda config --add channels conda-forge
conda create --yes -n dyhdenv3
source activate dyhdenv3
conda install --yes cdo=1.9.9 netcdf4 numpy matplotlib xarray cython scipy python-cdo configparser mpi4py
conda list -e
source deactivate
