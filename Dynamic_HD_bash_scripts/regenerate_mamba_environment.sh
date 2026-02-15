#!/bin/bash

mamba env remove -n dyhdenv_mamba
mamba create --yes --override-channels -c conda-forge/linux-64 -c conda-forge/noarch -n dyhdenv_mamba python=3.11
source activate dyhdenv_mamba
mamba install --yes --override-channels -c conda-forge/linux-64 -c conda-forge/noarch cdo netcdf4 numpy matplotlib xarray cython scipy python-cdo mpi4py jinja2 meson cartopy
mamba list -e
source deactivate
