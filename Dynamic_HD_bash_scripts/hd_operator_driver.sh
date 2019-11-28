#!/bin/bash
set -e

echo "Running Version 3.7 of the HD Command Line Operator Code"

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

while getopts ":r:p:c:n" opt; do
	case $opt in
	r)
		hd_driver_config_filepath=$OPTARG
		;;
	p)
		hd_driver_print_info_target_filepath=$OPTARG
		;;
	c)
		config_file=$OPTARG
		;;
	n)
		no_compilation=true
		;;
	\?)
		echo "Invalid option: -$OPTARG" >&2
		exit 1
		;;
	:)
		echo "Option -$OPTARG requires an argument." >&2
		exit 1
		;;
	esac
done

if [[ -n ${hd_driver_config_filepath+x} ]] && [[ -n ${hd_driver_print_info_target_filepath+x} ]]; then
	echo "Incompatible options set" >&2
	exit 1
fi

if [[ -z ${hd_driver_config_filepath+x} ]] || [[ -z ${hd_driver_print_info_target_filepath+x} ]]; then
	echo "No options set; one option is required" >&2
	exit 1
fi

if ! [[ ${hd_driver_config_filepath##*.} == "ini" ]] ; then
	echo "HD driver config file has the wrong file extension (requires .ini extension)" 1>&2
	exit 1
fi

#Find the directory of this script
this_script_dir="$(dirname "$(readlink -f "$0")")"

#Thus find the default source directory and convert it to an absolute filepath
source_directory="${this_script_dir}/.."
source_directory=$(find_ab_path $source_directory)

#Allocate a default location for external source code
external_source_directory=${source_directory}/../dynamic_hd_external_source_code
external_source_directory=$(find_ab_path $external_source_directory)

#Convert input filepaths from relative filepaths to absolute filepaths
hd_driver_config_filepath=$(find_abs_path $hd_driver_config_filepath)
hd_driver_print_info_target_filepath=$(find_abs_path $hd_driver_print_info_target_filepath)

# Check config file exists and has correct format (if a config file has been specified)
if [[ -z ${config_file} ]]; then
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
fi

# Check we have actually read the variables correctly
if [[ -z ${source_directory} ]]; then
	echo "Source directory set to a blank string" 1>&2
	exit 1
fi

if [[ -z ${external_source_directory} ]]; then
	echo "External source directory not set in config file or set to a blank string" 1>&2
	exit 1
fi

if ! [[ -d $source_directory ]]; then
	echo "Source directory does not exist." 1>&2

fi

if ! [[ -d $external_source_directory ]]; then
	echo "External Source directory does not exist." 1>&2

fi

#Check for locks if necesssary and set the compilation_required flag accordingly
exec 200>"${source_directory}/compilation.lock"
if ! ${no_compilation} ; then
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
	conda create --file "${source_directory}/Dynamic_HD_Environmental_Settings/dynamic_hd_env.txt" --yes --name "dyhdenv"
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
	cd ${source_directory}/Dynamic_HD_Cpp_Code/Release
	make -f ../makefile clean
	make -f ../makefile compile_only
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

#Setup cython interface between python and C++
if $compilation_required; then
	cd ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts
	echo "Compiling Cython Modules" 1>&2
	python2.7 ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/setup_fill_sinks.py clean --all
	python2.7 ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/setup_fill_sinks.py build_ext --inplace -f
	cd - 2>&1 > /dev/null
fi

#Prepare bin directory for python code
if $compilation_required; then
	mkdir -p ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/bin
fi

#Prepare python script arguments
if [[ -n ${hd_driver_config_filepath+x} ]]; then
	python_script_arguments="-r ${hd_driver_config_filepath}"
elif [[ -n ${hd_driver_print_info_target_filepath+x} ]]; then
	python_script_arguments="-p ${hd_driver_print_info_target_filepath}"
else:
	echo "Incompatible options set or no options set. Early checks for this where ineffective!" >&2
fi

#Run
echo "Running HD Command Line Operator Code" 1>&2
python2.7 ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/hd_operator_driver.py ${python_script_arguments}
