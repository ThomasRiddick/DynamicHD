#!/bin/bash
set -e

#Process command line arguments
first_timestep=${1}
input_orography_filepath=${2}
input_ls_mask_filepath=${3}
output_hdpara_filepath=${4}
source_directory=${5}
external_source_directory=${6}
ancillary_data_directory=${7}
working_directory=${8}
diagostic_output_directory=${9}
output_hdstart_filepath=${10}

#Change first_timestep into a bash command for true or false
shopt -s nocasematch
if [[ $first_timestep -eq "true" ]] || [[ $first_timestep -eq "t" ]]; then 
	first_timestep=true
elif [[ $first_timestep -eq "false" ]] || [[ $first_timestep -eq "f" ]]; then
	first_timestep=false
else
	echo "Format of first_timestep flag is unknown, please use True/False or T/F" 1>&2
	exit 1
fi
shopt -u nocasematch

#Check number of arguments makes sense
if [[ $# -ne "9" ]] && [[ $# -ne "10" ]]; then
	echo "Wrong number of positional arguments ($# supplied), script only takes 9 or 10"	1>&2
	exit 1
fi 
if $first_timestep && [[ $# -eq "9" ]]; then
	echo "First timestep requires 10 arguments including output hdstart file path (9 supplied)" 1>&2
	exit 1
elif ! $first_timestep && [[ $# -eq "10" ]]; then
	echo "First timestep requires 9 arguments (10 supplied). Specifying an output hdstart file path is not permitted." 1>&2
	exit 1
fi 

#Check the arguments have the correct file extensions
if ! [[ ${input_orography_filepath##*.} -eq "nc" ]] || ! [[ ${input_orography_filepath##*.} -eq "nc" ]] ; then
	echo "One or more input files has the wrong file extension" 1>&2
	exit 1
fi

if ! [[ ${output_hdpara_filepath##*.} -eq "nc" ]] ; then
	echo "Output hdpara file has the wrong file extension" 1>&2
	exit 1
fi

if  $first_timestep && ! [[ ${output_hdstart_filepath##*.} -eq "nc" ]] ; then
	echo "Output hdstart file has the wrong file extension" 1>&2
	exit 1
fi

#Setup conda environment
echo "Setting up environment"
export MODULEPATH="/sw/common/Modules:/client/Modules"
eval `/usr/bin/tclsh /sw/share/Modules/modulecmd.tcl bash load anaconda3`
if ! conda info -e | grep -q "dyhdenv"; then
     conda create --yes --name dyhdenv --file dynamic_hd_env.txt
fi
source activate dyhdenv

#Setup correct python path
export PYTHONPATH=$(pwd)

#Load a new version of gcc that doesn't have the polymorphic variable bug
eval `/usr/bin/tclsh /sw/share/Modules/modulecmd.tcl bash load gcc/6.2.0`

# Prepare a working directory if it is the first timestep and it doesn't already exist
if $first_timestep && ! [[ -e $working_directory ]]; then
	echo "Creating a working directory"	
	mkdir -p $working_directory
fi

#Check input files, source directory ancillary data directory and working directory exist

if ! [[ -e $input_ls_mask_filepath ]] || ! [[ -e $input_orography_filepath ]]; then
	echo "One or more input files does not exist" 1>&2
	exit 1
fi

if ! [[ -d $source_directory ]]; then
	echo "Source directory does not exist." 1>&2
	
fi

if ! [[ -d $external_source_directory ]]; then
	echo "External Source directory does not exist." 1>&2
	
fi 

if ! [[ -d $ancillary_data_directory ]]; then
	echo "Ancillary data directory does not exist" 1>&2
	exit 1	
fi

if ! [[ -d $working_directory ]]; then
	echo "Working directory does not exist or is not a directory (even after attempt to create it)" 1>&2	
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

#Set output_hdstart_filepath to blank if not the first timestep
if ! ${first_timestep}; then
	output_hdstart_filepath=""	
else 
	output_hdstart_filepath="-s ${output_hdstart_filepath}"
fi

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
	cd - 2>&1 >/dev/null
	echo "Compiling Fortran code" 1>&2
	mkdir -p ${source_directory}/Dynamic_HD_Fortran_Code/Release
	cd ${source_directory}/Dynamic_HD_Fortran_Code/Release
	make -f ../makefile clean
	make -f ../makefile -e "EXT_SOURCE=${external_source_directory}" all 
	cd - 2>&1 >/dev/null
fi

#Run
echo "Running Dynamic HD Code" 1>&2 
python ${source_directory}/Dynamic_HD_Scripts/dynamic_hd_production_run_driver.py ${input_orography_filepath} ${input_ls_mask_filepath} ${output_hdpara_filepath} ${ancillary_data_directory} ${working_directory} ${output_hdstart_filepath}
#Move diagnostic output to target location
if [[ $(ls ${working_directory}) ]]; then 
	mv ${working_directory} ${diagostic_output_directory}
fi
rmdir ${working_directory}