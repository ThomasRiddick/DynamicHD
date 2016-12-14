#!/bin/bash

#set standard scripting options
set -e

#add user installed application bin directory
export PATH="${PATH}:/usr/local/bin"

#Get command line arguments
orog_nc_file=${1}
grids_file=${2}
output_weights_file=${3}
method=${4}

#Set options
datapath="/Users/thomasriddick/Documents/data/HDdata"
orog_nc_file_path="${datapath}/orographys/${orog_nc_file}"
grids_file_path="${datapath}/grids/${grids_file}"
output_weights_file_path="${datapath}/remapweights/${output_weights_file}"

#generate weights
cdo -f nc -${method},${grids_file_path} ${orog_nc_file_path} ${output_weights_file_path}