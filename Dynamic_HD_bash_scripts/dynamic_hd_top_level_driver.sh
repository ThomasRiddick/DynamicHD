#!/bin/bash
set -e

echo "Running Version 2.3 of the Dynamic HD Parameters Generation Code"

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

#Define module unloading function (only works on hpc)
function unload_module
{
module_name=$1
if echo $LOADEDMODULES | fgrep -q ${module_name} ; then
	module unload $module_name
fi
}

#Define portable absolute path finding function	
function find_abs_path
{
relative_path=$1
perl -MCwd -e 'print Cwd::abs_path($ARGV[0]),qq<\n>' $relative_path
}

#Process command line arguments
first_timestep=${1}
input_orography_filepath=${2}
input_ls_mask_filepath=${3}
present_day_base_orography_filepath=${4}
glacier_mask_filepath=${5}
output_hdpara_filepath=${6}
ancillary_data_directory=${7}
working_directory=${8}
diagnostic_output_directory=${9}
diagnostic_output_exp_id_label=${10}
diagnostic_output_time_label=${11}
output_hdstart_filepath=${12}

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
if [[ $# -ne 11 ]] && [[ $# -ne 12 ]]; then
	echo "Wrong number of positional arguments ($# supplied), script only takes 11 or 12"	1>&2
	exit 1
fi 

if $first_timestep && [[ $# -eq 11 ]]; then
	echo "First timestep requires 12 arguments including output hdstart file path (11 supplied)" 1>&2
	exit 1
elif ! $first_timestep && [[ $# -eq 12 ]]; then
	echo "Timesteps other than the first requires 11 arguments (12 supplied)." 1>&2 
	echo "Specifying an output hdstart file path is not permitted." 1>&2
	exit 1
fi 

#Check the arguments have the correct file extensions
if ! [[ ${input_orography_filepath##*.} == "nc" ]] || ! [[ ${input_orography_filepath##*.} == "nc" ]] || ! [[ ${present_day_base_orography_filepath##*.} == "nc" ]] || ! [[ ${glacier_mask_filepath##*.} == "nc" ]] ; then
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

#Convert input filepaths from relative filepaths to absolute filepaths
input_ls_mask_filepath=$(find_abs_path $input_ls_mask_filepath)
input_orography_filepath=$(find_abs_path $input_orography_filepath)
present_day_base_orography_filepath=$(find_abs_path $present_day_base_orography_filepath)
glacier_mask_filepath=$(find_abs_path $glacier_mask_filepath)
ancillary_data_directory=$(find_abs_path $ancillary_data_directory)
working_directory=$(find_abs_path $working_directory)
diagnostic_output_directory=$(find_abs_path $diagnostic_output_directory)
output_hdpara_filepath=$(find_abs_path $output_hdpara_filepath)
if $first_timestep; then
	output_hdstart_filepath=$(find_abs_path $output_hdstart_filepath)
fi

#Check input files, ancillary data directory and diagnostic output directory exist

if ! [[ -e $input_ls_mask_filepath ]] || ! [[ -e $input_orography_filepath ]] || ! [[ -e $present_day_base_orography_filepath ]] || ! [[ -e $glacier_mask_filepath ]]; then
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

# Read in source_directory and external_source_directory
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

# Prepare a working directory if it is the first timestep and it doesn't already exist
if $first_timestep && ! [[ -e $working_directory ]]; then
	echo "Creating a working directory"	
	mkdir -p $working_directory
fi

#Check that the working directory, source directory and external source directory exist
if ! [[ -d $working_directory ]]; then
	if $first_timestep ; then
		echo "Working directory does not exist or is not a directory (even after attempt to create it)" 1>&2	
	else
		echo "Working directory does not exist or is not a directory" 1>&2	
	fi
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

#Check for locks if necesssary and set the compilation_required flag accordingly
exec 200>"${source_directory}/compilation.lock"
if $first_timestep ; then
	if flock -x -n 200 ; then
		compilation_required=true
	else
		flock -s 200
		compilation_required=false	
	fi
else
	flock -s 200
	compilation_required=false	
fi

#Setup conda environment
echo "Setting up environment"
if [[ $(hostname -d) == "hpc.dkrz.de" ]]; then
	source /sw/rhel6-x64/etc/profile.mistral
	unload_module netcdf_c
    unload_module imagemagick
	unload_module cdo/1.7.0-magicsxx-gcc48
    unload_module python
else
	export MODULEPATH="/sw/common/Modules:/client/Modules"
fi
load_module anaconda3

if $compilation_required && conda info -e | grep -q "dyhdenv"; then
	conda env remove --yes --name dyhdenv
fi
if ! conda info -e | grep -q "dyhdenv"; then
	#Use the txt file environment creation (not conda env create that requires a yml file)
	#Create a dynamic_hd_env.txt using conda list --export > filename.txt not 
	#conda env export which creates a yml file.
	conda create --file "${ancillary_data_directory}/dynamic_hd_env.txt" --yes --name "dyhdenv"
fi
source activate dyhdenv

#Load CDOs if required and reload version of python with CDOs included
if echo $LOADEDMODULES | fgrep -q -v "cdo" ; then
	load_module cdo
fi
if [[ $(hostname -d) == "hpc.dkrz.de" ]]; then
	load_module python/2.7.12
else
	load_module python
fi

#Load a new version of gcc that doesn't have the polymorphic variable bug
load_module gcc/6.2.0

#Setup correct python path
export PYTHONPATH=${source_directory}/Dynamic_HD_Scripts:${PYTHONPATH}

#Compile C++ and Fortran Code if this is the first timestep
if $compilation_required ; then
	#Normalise external source path; use a crude yet portable method
	cd $external_source_directory
	external_source_directory=$(pwd -P)
	cd - 2>&1 > /dev/null
	echo "Compiling C++ code" 1>&2
	mkdir -p ${source_directory}/Dynamic_HD_Cpp_Code/Release
	mkdir -p ${source_directory}/Dynamic_HD_Cpp_Code/Release/src
	mkdir -p ${source_directory}/Dynamic_HD_Cpp_Code/Release/src/gtest
	if ! [[ -f ${source_directory}/Dynamic_HD_Cpp_Code/src/gtest/*.cc ]]; then
		cp ${external_source_directory}/googletest-master/googletest/src/*.cc \
			${source_directory}/Dynamic_HD_Cpp_Code/src/gtest
	fi
	if ! [[ -d ${source_directory}/Dynamic_HD_Cpp_Code/include/gtest ]]; then
		cp -r ${external_source_directory}/googletest-master/googletest/include/gtest \
			${source_directory}/Dynamic_HD_Cpp_Code/include/gtest
	fi
	if ! [[ -f ${source_directory}/Dynamic_HD_Cpp_Code/src/gtest-internal-inl.h ]]; then
		cp -r ${external_source_directory}/googletest-master/googletest/src/gtest-internal-inl.h \
			${source_directory}/Dynamic_HD_Cpp_Code/src	
	fi
	cd ${source_directory}/Dynamic_HD_Cpp_Code/Release
	make -f ../makefile clean
	make -f ../makefile all 
	cd - 2>&1 > /dev/null
	echo "Compiling Fortran code" 1>&2
	mkdir -p ${source_directory}/Dynamic_HD_Fortran_Code/Release
	mkdir -p ${source_directory}/Dynamic_HD_Fortran_Code/Release/src
	cd ${source_directory}/Dynamic_HD_Fortran_Code/Release
	make -f ../makefile clean
	make -f ../makefile -e "EXT_SOURCE=${external_source_directory}" all 
	cd - 2>&1 > /dev/null
fi

# Clean shared libraries
if $compilation_required; then
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
if $compilation_required; then
	cd ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts
	echo "Compiling Cython Modules" 1>&2
	python2.7 ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/setup_fill_sinks.py clean --all
	python2.7 ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/setup_fill_sinks.py build_ext --inplace -f
	cd - 2>&1 > /dev/null
fi

#Prepare bin directory for python code and bash code
if $compilation_required; then
	mkdir -p ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/bin
	mkdir -p ${source_directory}/Dynamic_HD_bash_scripts/bin
fi

#Run
echo "Running Dynamic HD Code" 1>&2 
python2.7 ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/dynamic_hd_production_run_driver.py ${input_orography_filepath} ${input_ls_mask_filepath} ${present_day_base_orography_filepath} ${glacier_mask_filepath} ${output_hdpara_filepath} ${ancillary_data_directory} ${working_directory} ${output_hdstart_filepath}

#Delete paragen directory
cd ${working_directory}/paragen
rm -f paragen.inp soil_partab.txt slope.dat riv_vel.dat riv_n.dat riv_k.dat over_vel.dat over_n.dat over_k.dat 
rm -f hdpara.srv global.inp ddir.inp bas_k.dat 
cd - 2>&1 > /dev/null
rmdir ${working_directory}/paragen
rm -f catchments.log loops.log 30minute_river_dirs.dat 
rm -f 30minute_ls_mask.dat 30minute_filled_orog.dat 30minute_river_dirs_temp.nc 30minute_filled_orog_temp.nc
rm -f 30minute_filled_orog_temp.dat 30minute_ls_mask_temp.dat 30minute_river_dirs_temp.dat 
rm -f 30minute_ls_mask_temp.nc

#Generate full diagnostic output label
diagnostic_output_label="${diagnostic_output_exp_id_label}_${diagnostic_output_time_label}"

#Move diagnostic output to target location
if [[ $(ls ${working_directory}) ]]; then 
	mkdir -p ${diagnostic_output_directory} || true 
	for file in ${working_directory}/*.nc ; do
		mv  $file ${diagnostic_output_directory}/$(basename ${file} .nc)_${diagnostic_output_label}.nc
	done
fi