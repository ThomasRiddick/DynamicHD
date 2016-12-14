#!/bin/bash

#Set standard scripting options
set -e

#Get command line arguments
orog_nc_file_path=${1}
grids_file_path=${2}
weights_file_path=${3}
output_file_path=${4}
mask_nc_file_path=${5}

#add user installed application bin directory
export PATH="${PATH}:/usr/local/bin"

#create topography on hd grid 
cdo -f nc   -add -mul  -setmisstoc,0 -setrtomiss,-100000,0  -remap,${grids_file_path},${weights_file_path} ${orog_nc_file_path} ${mask_nc_file_path} ${mask_nc_file_path} ${output_file_path}