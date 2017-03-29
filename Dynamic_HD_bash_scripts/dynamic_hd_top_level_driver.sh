#!/bin/bash
set -e

#Define module loading function
function load_module
{
module_name=$1
if [[ $(hostname -d) == "hpc.dkrz.de" ]]; then
	module load ${module_name}
else
	eval "eval `/usr/bin/tclsh /sw/share/Modules/modulecmd.tcl bash load ${module_name}`"
fi
}	

#Process command line arguments
first_timestep=${1}
input_orography_filepath=${2}
input_ls_mask_filepath=${3}
output_hdpara_filepath=${4}
ancillary_data_directory=${5}
diagostic_output_directory=${6}
output_hdstart_filepath=${7}

#Change first_timestep into a bash command for true or false
shopt -s nocasematch
if [[ $first_timestep == "true" ]] || [[ $first_timestep == "t" ]]; then 
	first_timestep=true
elif [[ $first_timestep == "false" ]] || [[ $first_timestep == "f" ]]; then
	first_timestep=false
else
	echo "Format of first_timestep flag is unknown, please use True/False or T/F" 1>&2
	exit 1
fi
shopt -u nocasematch

#Check number of arguments makes sense
if [[ $# -ne 6 ]] && [[ $# -ne 7 ]]; then
	echo "Wrong number of positional arguments ($# supplied), script only takes 6 or 7"	1>&2
	exit 1
fi 

if $first_timestep && [[ $# -eq 6 ]]; then
	echo "First timestep requires 7 arguments including output hdstart file path (6 supplied)" 1>&2
	exit 1
elif ! $first_timestep && [[ $# -eq 7 ]]; then
	echo "Timesteps other than the first requires 6 arguments (7 supplied)." 1>&2 
	echo "Specifying an output hdstart file path is not permitted." 1>&2
	exit 1
fi 

#Check the arguments have the correct file extensions
if ! [[ ${input_orography_filepath##*.} == "nc" ]] || ! [[ ${input_orography_filepath##*.} == "nc" ]] ; then
	echo "One or more input files has the wrong file extension" 1>&2
	exit 1
fi

if ! [[ ${output_hdpara_filepath##*.} == "nc" ]] ; then
	echo "Output hdpara file has the wrong file extension" 1>&2
	exit 1
fi

if  $first_timestep && ! [[ ${output_hdstart_filepath##*.} == "nc" ]] ; then
	echo "Output hdstart file has the wrong file extension" 1>&2
	exit 1
fi

#Setup conda environment
echo "Setting up environment"
if [[ $(hostname -d) == "hpc.dkrz.de" ]]; then
	source /sw/rhel6-x64/etc/profile.mistral
	module unload netcdf_c/4.3.2-gcc48
	module unload imagemagick/6.9.1-7-gcc48
	module unload cdo/1.7.0-magicsxx-gcc48
else
	export MODULEPATH="/sw/common/Modules:/client/Modules"
fi
load_module anaconda3

if $first_timestep && conda info -e | grep -q "dyhdenv"; then
	conda env remove --yes --name dyhdenv
fi
if ! conda info -e | grep -q "dyhdenv"; then
	#Use the txt file environment creation (not conda env create that requires a yml file)
	#Create a dynamic_hd_env.txt using conda list --export > filename.txt not 
	#conda env export which creates a yml file.
	conda create --file "${ancillary_data_directory}/dynamic_hd_env.txt" --yes --name "dyhdenv"
fi
source activate dyhdenv

#Load CDOs and reload version of python with CDOs included
load_module cdo
load_module python

#Load a new version of gcc that doesn't have the polymorphic variable bug
load_module gcc/6.2.0

#Check input files, ancillary data directory and diagnostic output directory exist

if ! [[ -e $input_ls_mask_filepath ]] || ! [[ -e $input_orography_filepath ]]; then
	echo "One or more input files does not exist" 1>&2
	exit 1
fi

if ! [[ -d $ancillary_data_directory ]]; then
	echo "Ancillary data directory does not exist" 1>&2
	exit 1	
fi

if ! [[ -d ${output_hdpara_filepath%/*} ]]; then
	echo "Filepath of output hdpara.nc does not exist" 1>&2
	exit 1
fi

if $first_timestep && ! [[ -d ${output_hdstart_filepath%/*} ]]; then
	echo "Filepath of output hdstart.nc does not exist" 1>&2
	exit 1
fi

# Define config file
config_file="${ancillary_data_directory}/top_level_driver.cfg"

# Check config file exists  and has correct format

if ! [[ -f ${config_file} ]]; then
	echo "Top level script config file doesn't exist!"	
	exit 1
fi

if egrep -v -q "^(#.*|.*=.*)$" ${config_file}; then
	echo "Config file has wrong format" 1>&2	
	exit 1
fi

# Read in source_directory, external_source_directory and working_directory
source ${config_file}

# Check we have actually read the variables correctly
if [[ -z ${source_directory} ]]; then
	echo "Source directory not set in config file or set to a blank string" 1>&2
	exit 1
fi

if [[ -z ${external_source_directory} ]]; then
	echo "External source directory not set in config file or set to a blank string" 1>&2
	exit 1
fi

if [[ -z ${working_directory} ]]; then
	echo "Working directory not set in config file or set to a blank string" 1>&2
	exit 1
fi

# Prepare a working directory if it is the first timestep and it doesn't already exist
if $first_timestep && ! [[ -e $working_directory ]]; then
	echo "Creating a working directory"	
	mkdir -p $working_directory
fi

#Check that the working directory, source directory and external source directory exist
if ! [[ -d $working_directory ]]; then
	echo "Working directory does not exist or is not a directory (even after attempt to create it)" 1>&2	
	exit 1
fi 

if ! [[ -d $source_directory ]]; then
	echo "Source directory does not exist." 1>&2
	
fi

if ! [[ -d $external_source_directory ]]; then
	echo "External Source directory does not exist." 1>&2
	
fi 

#Change to the working directory
cd ${working_directory}

#Set output_hdstart_filepath to blank if not the first timestep
if ! ${first_timestep}; then
	output_hdstart_filepath=""	
else 
	output_hdstart_filepath="-s ${output_hdstart_filepath}"
fi

#Setup correct python path
export PYTHONPATH=${source_directory}/Dynamic_HD_Scripts:${PYTHONPATH}

#Compile C++ and Fortran Code if this is the first timestep
if $first_timestep ; then
	#Normalise external source path; use a crude yet portable method
	cd $external_source_directory
	external_source_directory=$(pwd -P)
	cd -
	echo "Compiling C++ code" 1>&2
	mkdir -p ${source_directory}/Dynamic_HD_Cpp_Code/Release
	cd ${source_directory}/Dynamic_HD_Cpp_Code/Release
	make -f ../makefile clean
	make -f ../makefile all 
	cd - 2>&1 > /dev/null
	echo "Compiling Fortran code" 1>&2
	mkdir -p ${source_directory}/Dynamic_HD_Fortran_Code/Release
	cd ${source_directory}/Dynamic_HD_Fortran_Code/Release
	make -f ../makefile clean
	make -f ../makefile -e "EXT_SOURCE=${external_source_directory}" all 
	cd - 2>&1 > /dev/null
fi

# Clean shared libraries
if $first_timestep; then
	cd ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts
		make -f makefile clean
	cd - 2>&1 > /dev/null
fi

# Clean up paragen working directory if any

if [[ -d "${working_directory}/paragen" ]]; then
	cd ${working_directory}/paragen
	rm -f paragen.inp soil_partab.txt slope.dat riv_vel.dat riv_n.dat riv_k.dat over_vel.dat over_n.dat over_k.dat || true 
	rm -f hdpara.srv global.inp ddir.inp bas_k.dat || true
	cd - 2>&1 > /dev/null
	rmdir ${working_directory}/paragen 
fi 

#Setup cython interface between python and C++
if $first_timestep; then
	cd ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts
	echo "Compiling Cython Modules" 1>&2
	python2.7 ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/setup_fill_sinks.py clean --all
	python2.7 ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/setup_fill_sinks.py build_ext --inplace -f
	cd - 2>&1 > /dev/null
fi

#Run
echo "Running Dynamic HD Code" 1>&2 
python2.7 ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/dynamic_hd_production_run_driver.py ${input_orography_filepath} ${input_ls_mask_filepath} ${output_hdpara_filepath} ${ancillary_data_directory} ${working_directory} ${output_hdstart_filepath}

#Delete paragen directory
cd ${working_directory}/paragen
rm -f paragen.inp soil_partab.txt slope.dat riv_vel.dat riv_n.dat riv_k.dat over_vel.dat over_n.dat over_k.dat 
rm -f hdpara.srv global.inp ddir.inp bas_k.dat 
cd - 2>&1 > /dev/null
rmdir ${working_directory}/paragen
rm -f catchments.log loops.log 30minute_river_dirs.dat 
rm -f 30minute_ls_mask.dat 30minute_filled_orog.dat 30minute_river_dirs_temp.nc 30minute_filled_orog_temp.nc
rm -f 30minute_ls_mask_temp.nc

#Move diagnostic output to target location
if [[ $(ls ${working_directory}) ]]; then 
	mkdir -p ${diagostic_output_directory} || true 
	mv ${working_directory}/* ${diagostic_output_directory}
fi