#!/bin/bash
set -e

#Compile old fortran code from Stefan for production runs

#Get command line arguments
bin_dir=${1}
src_dir=${2}
paragen_source_filepath=${3}
paragen_bin_file=${4}

#Set compiler
comp=gfortran

#Create bin directory if it doesn't exist already
mkdir -p ${bin_dir}

#Clean
rm -f ${bin_dir}/hdfile
#Be careful how we delete the paragen binary to avoid mistakes
rm -f ${bin_dir}/paragen_nlat*_nlon* || true
rm -f ${bin_dir}/paragen || true

#Compile
${comp} -o ${bin_dir}/hdfile ${src_dir}/hdfile.f ${src_dir}/globuse.f
${comp} -o ${bin_dir}/${paragen_bin_file} ${paragen_source_filepath} ${src_dir}/globuse.f ${src_dir}/mathe.f ${src_dir}/modtime.f