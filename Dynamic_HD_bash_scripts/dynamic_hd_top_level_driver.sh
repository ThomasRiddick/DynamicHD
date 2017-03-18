#!/bin/bash
set -e

#Process command line arguments
first_timestep=${1}
input_orography_filepath=${2}
input_ls_mask_filepath=${3}
output_hdpara_filepath=${4}
ancillary_data_directory=${5}
working_directory=${6}
output_hdstart_filepath=${7}

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
if [[ $# -ne "6" ]] && [[ $# -ne "7" ]]; then
	echo "Wrong number of positional arguments ($# supplied), script only takes 6 or 7"	1>&2
	exit 1
fi 
if $first_timestep && [[ $# -eq "6" ]]; then
	echo "First timestep requires 7 arguments including output hdstart file path (6 supplied)" 1>&2
	exit 1
elif ! $first_timestep && [[ $# -eq "7" ]]; then
	echo "First timestep requires 6 arguments (7 supplied). Specifying an output hdstart file path is not permitted." 1>&2
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

if  $first_timestep || ! [[ ${output_hdstart_filepath##*.} -eq "nc" ]] ; then
	echo "Output hdstart file has the wrong file extension" 1>&2
	exit 1
fi

#Setup conda environment
echo "Setting up environment"
export MODULEPATH="/sw/common/Modules:/client/Modules"
eval `/usr/bin/tclsh /sw/share/Modules/modulecmd.tcl bash load anaconda`
if ! conda info -e | grep -q "dyhdenv"; then
     conda create --yes --name dyhdenv --file dynamic_hd_env.txt
fi
source activate dyhdenv

#Setup correct python path
export PYTHONPATH=$(pwd)

# Prepare a working directory if it is the first timestep and it doesn't already exist
exit 1
if $first_timestep && ! [[ -e $working_directory ]]; then
	echo "Creating a working directory"	
	mkdir -p $working_directory
fi

#Check input files, ancillary data directory and working directory exist

if ! [[ -e $input_ls_mask_filepath ]] || ! [[ -e $input_orography_filepath ]]; then
	echo "One or more input files does not exist" 1>&2
	exit 1
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
fi

if $first_timestep && ! [[ -d ${output_hdstart_filepath%/*} ]]; then
	echo "Filepath of output hdstart.nc does not exist" 1>&2
fi

#Compile C++ and Fortran Code if this is the first timestep
if $first_timestep ; then
	echo "Compiling C++ code"	
	echo "Error - Compilation not yet prepared" 1>&2
	exit 1
	echo "Compiling Fortran code"	
fi

#Run
python dynamic_hd_production_run_driver.py