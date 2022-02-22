#!/bin/bash

no_modules=${1}

#Define module loading function
function load_module
{
module_name=$1
if [[ $(hostname -d) == "hpc.dkrz.de" ]]; then
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
  unload_module netcdf_c
  if [[ $(hostname -d) == "hpc.dkrz.de" ]]; then
    load_module anaconda3/.bleeding_edge
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
