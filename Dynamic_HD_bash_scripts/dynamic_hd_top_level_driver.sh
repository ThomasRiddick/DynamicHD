#!/bin/bash
set -e

echo "Running Version 3.11 of the Dynamic HD Parameters Generation Code"
start_time=$(date +%s%N)

#Define module loading function
function load_module
{
module_name=$1
if [[ $(hostname -d) == "hpc.dkrz.de" ]] || [[ $(hostname -d) == "atos.local" ]]; then
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

#Define portable absolute path finding function
function find_abs_path
{
relative_path=$1
perl -MCwd -e 'print Cwd::abs_path($ARGV[0]),qq<\n>' $relative_path
}

compile_only=false
no_compile=false
case ${1} in
	-c | --compile)
		compile_only=true
		first_timestep=true
		working_directory=$(pwd)
		shift
		ancillary_data_directory=${1}
		if [[ $# -ne 1 ]]; then
			echo "Wrong number of positional arguments ($# supplied)," 1>&2
			echo "script only takes 1 in compilation mode"	1>&2
			exit 1
		fi
		;;
	-n | --no-compile)
		no_compile=true
		shift
		;;
	*)
  		;;
esac

if ! ${compile_only} ; then
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
	working_directory=$(find_abs_path $working_directory)
	diagnostic_output_directory=$(find_abs_path $diagnostic_output_directory)
	output_hdpara_filepath=$(find_abs_path $output_hdpara_filepath)
	if $first_timestep; then
		output_hdstart_filepath=$(find_abs_path $output_hdstart_filepath)
	fi
fi
ancillary_data_directory=$(find_abs_path $ancillary_data_directory)

#Check input files, ancillary data directory and diagnostic output directory exist

if ! [[ -d $ancillary_data_directory ]]; then
	echo "Ancillary data directory does not exist" 1>&2
	exit 1
fi

if ! ${compile_only} ; then
	if ! [[ -e $input_ls_mask_filepath ]] || ! [[ -e $input_orography_filepath ]] || ! [[ -e $present_day_base_orography_filepath ]] || ! [[ -e $glacier_mask_filepath ]]; then
		echo "One or more input files does not exist" 1>&2
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
fi

# Define config file
config_file="${ancillary_data_directory}/top_level_driver.cfg"

# Check config file exists  and has correct format

if ! [[ -f ${config_file} ]]; then
	echo "Top level script config file (${config_file}) doesn't exist!"
	exit 1
fi

if egrep -v -q "^(#.*|.*=.*)$" ${config_file}; then
	echo "Config file has wrong format" 1>&2
	exit 1
fi

# Read in source_directory
source ${config_file}

# Check we have actually read the variables correctly
if [[ -z ${source_directory} ]]; then
	echo "Source directory not set in config file or set to a blank string - " 1>&2
	echo "defaulting to using base directory containing this top-level script" 1>&2
	source_directory=${$(dirname $0)%DynamicHD/Dynamic_HD_bash_scripts}
	source_directory=${source_directory%/}
fi

if ! ${compile_only} ; then
	# Prepare a working directory if it is the first timestep and it doesn't already exist
	if $first_timestep && ! [[ -e $working_directory ]]; then
		echo "Creating a working directory"
		mkdir -p $working_directory
	fi

	#Check that the working directory and the source directory exist
	if ! [[ -d $working_directory ]]; then
		if $first_timestep ; then
			echo "Working directory does not exist or is not a directory (even after attempt to create it)" 1>&2
		else
			echo "Working directory does not exist or is not a directory" 1>&2
		fi
		exit 1
	fi
fi

if ! [[ -d $source_directory ]]; then
	echo "Source directory ($source_directory) does not exist." 1>&2
	exit 1
fi

shopt -s nocasematch
no_conda=${no_conda:-"false"}
if [[ $no_conda == "true" ]] || [[ $no_conda == "t" ]]; then
	no_conda=true
elif [[ $no_conda == "false" ]] || [[ $no_conda == "f" ]]; then
	no_conda=false
else
	echo "Format of no_conda flag (${no_conda}) is unknown, please use True/False or T/F" 1>&2
	exit 1
fi

no_modules=${no_modules:-"false"}
if [[ $no_modules == "true" ]] || [[ $no_modules == "t" ]]; then
	no_modules=true
elif [[ $no_modules == "false" ]] || [[ $no_modules == "f" ]]; then
	no_modules=false
else
	echo "Format of no_modules flag (${no_modules}) is unknown, please use True/False or T/F" 1>&2
	exit 1
fi

no_env_gen=${no_env_gen:-"false"}
if [[ $no_env_gen == "true" ]] || [[ $no_env_gen == "t" ]]; then
        no_env_gen=true
elif [[ $no_env_gen == "false" ]] || [[ $no_env_gen == "f" ]]; then
        no_env_gen=false
else
        echo "Format of no_env_gen flag (${no_env_gen}) is unknown, please use True/False or T/F" 1>&2
        exit 1
fi

if ${no_conda} ; then
no_env_gen=false
fi

shopt -u nocasematch

if ! ${compile_only} ; then
	#Change to the working directory
	cd ${working_directory}

	#Set output_hdstart_filepath to blank if not the first timestep
	if ! ${first_timestep}; then
		output_hdstart_filepath=""
	else
		output_hdstart_filepath="-s ${output_hdstart_filepath}"
	fi
fi

#Check for locks if necesssary and set the compilation_required flag accordingly
if [[ $(hostname -d) == "hpc.dkrz.de" ]] || [[ $(hostname -d) == "atos.local" ]] ; then
	exec 200>"${source_directory}/compilation.lock"
	if $first_timestep ; then
		if flock -x -n 200 ; then
			compilation_required=true
		elif $compile_only ; then
			echo "Can't compile - previous version of code still running" 1>&2
			exit 1
		else
			flock -s 200
			compilation_required=false
		fi
	else
		flock -s 200
		compilation_required=false
	fi
else
	echo "Warning: Flock disabled - Doesn't work on Linux desktop system"
	echo "                        - manually check you aren't trying to"
	echo "                        - recompile during compilation!"
	if $first_timestep ; then
		compilation_required=true
	else
		compilation_required=false
	fi
fi

#Switch off compilation if no_compile option selected
if $no_compile ; then
	compilation_required=false
fi

#Setup conda environment
echo "Setting up environment"
if ! $no_modules ; then
	if [[ $(hostname -d) == "hpc.dkrz.de" ]]; then
		source /sw/rhel6-x64/etc/profile.mistral
		unload_module netcdf_c
	    unload_module imagemagick
		unload_module cdo/1.7.0-magicsxx-gcc48
	    unload_module python
	else
		export MODULEPATH="/sw/common/Modules:/client/Modules"
	fi
fi

if ! $no_modules && ! $no_conda ; then
	if [[ $(hostname -d) == "hpc.dkrz.de" ]]; then
		load_module anaconda3/.bleeding_edge
	elif [[ $(hostname -d) == "atos.local" ]]; then
                load_module python3
	else
		load_module anaconda3
	fi
fi

if ! $no_conda && ! $no_env_gen ; then
	if $compilation_required && conda info -e | grep -q "dyhdenv3"; then
		conda env remove --yes --name dyhdenv3
	fi
	if ! conda info -e | grep -q "dyhdenv3"; then
		${source_directory}/Dynamic_HD_bash_scripts/regenerate_conda_environment.sh $no_modules
	fi
fi
if ! $no_conda ; then
	source activate dyhdenv3
fi

#Load a new version of gcc that doesn't have the polymorphic variable bug
if ! $no_modules ; then
	if [[ $(hostname -d) == "hpc.dkrz.de" ]]; then
		load_module gcc/6.2.0
        elif [[ $(hostname -d) == "atos.local" ]]; then
                load_module gcc/11.2.0-gcc-11.2.0
	else
		load_module gcc/6.3.0
	fi
fi

#Setup correct python path
export PYTHONPATH=${source_directory}/Dynamic_HD_Scripts:${PYTHONPATH}

#Set Open MPI Fortran compiler
export OMPI_FC=/sw/rhel6-x64/gcc/gcc-6.2.0/bin/gfortran

#Call compilation script
${source_directory}/Dynamic_HD_bash_scripts/compile_dynamic_hd_code.sh ${compilation_required} ${compile_only} ${source_directory} ${working_directory} true "compile_only"

if ! ${compile_only} ; then
	#Set enviromental variable
    export USE_MPI_IN_PYTHON=false
	#Run
	echo "Running Dynamic HD Code" 1>&2
	setup_end_time=$(date +%s%N)
	python3 ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/dynamic_hd_and_dynamic_lake_drivers/dynamic_hd_production_run_driver.py ${input_orography_filepath} ${input_ls_mask_filepath} ${present_day_base_orography_filepath} ${glacier_mask_filepath} ${output_hdpara_filepath} ${ancillary_data_directory} ${working_directory} ${output_hdstart_filepath}
    computation_end_time=$(date +%s%N)

	#Delete paragen directory if it exists
	if [[ -d "${working_directory}/paragen" ]]; then
		cd ${working_directory}/paragen
		rm -f paragen.inp soil_partab.txt slope.dat riv_vel.dat riv_n.dat riv_k.dat over_vel.dat over_n.dat over_k.dat
		rm -f hdpara.srv global.inp ddir.inp bas_k.dat
		cd - 2>&1 > /dev/null
		rmdir ${working_directory}/paragen
	fi

	#Delete other files if they exist
	rm -f 30minute_filled_orog_temp.dat 30minute_ls_mask_temp.dat 30minute_river_dirs_temp.dat || true
	rm -f 30minute_ls_mask.dat 30minute_filled_orog.dat 30minute_river_dirs.dat || true

	#Delete other files
	rm -f catchments.log loops.log 30minute_filled_orog_temp.nc 30minute_river_dirs_temp.nc
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
	cleanup_end_time=$(date +%s%N)
	echo "Total top level script total run time: " $(((cleanup_end_time-start_time)/1000000)) " ms"
	echo "Setup time: " $(((setup_end_time-start_time)/1000000)) " ms"
	echo "Computation time: " $(((computation_end_time-setup_end_time)/1000000)) " ms"
	echo "Clean-up time: " $(((cleanup_end_time-computation_end_time)/1000000)) " ms"
fi
