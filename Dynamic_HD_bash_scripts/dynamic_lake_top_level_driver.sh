#!/bin/bash
set -e

echo "Running Version 3.11 of the Dynamic Lake and HD Parameters Generation Code"

#Define module loading function
function load_module
{
module_name=$1
if [[ $(hostname -d) == "atos.local" ]]; then
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
input_hdstart_filepath=${12}
input_lake_volumes_filepath=${13}
output_lakepara_filepath=${14}
output_hdstart_filepath=${15}
output_lakestart_filepath=${16}

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
if [[ $# -ne 16 ]]; then
	echo "Wrong number of positional arguments ($# supplied), script only takes 16"	1>&2
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
input_hdstart_filepath=$(find_abs_path $input_hdstart_filepath)
input_lake_volumes_filepath=$(find_abs_path $input_lake_volumes_filepath)
output_lakepara_filepath=$(find_abs_path $output_lakepara_filepath)
output_hdstart_filepath=$(find_abs_path $output_hdstart_filepath)
output_lakestart_filepath=$(find_abs_path $output_lakestart_filepath)

#Check input files, ancillary data directory and diagnostic output directory exist

if ! [[ -e $input_ls_mask_filepath ]]; then
		echo "Input land sea mask file does not exist" 1>&2
	exit 1
fi

if ! [[ -e $input_orography_filepath ]]; then
		echo "Input orography file does not exist" 1>&2
	exit 1
fi

if ! [[ -e $present_day_base_orography_filepath ]]; then
		echo "Input present day base orography file does not exist" 1>&2
	exit 1
fi

if ! [[ -e $glacier_mask_filepath ]]; then
		echo "Input glacier mask file does not exist" 1>&2
	exit 1
fi

if ! [[ -e $input_hdstart_filepath ]] && ! ${first_timestep}; then
		echo "Input hdstart file does not exist" 1>&2
	exit 1
fi

if ! [[ -e $input_lake_volumes_filepath ]]; then
	echo "Input lake volumes file does not exist" 1>&2
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

if ! [[ -d ${output_lakepara_filepath%/*} ]]; then
	echo "Filepath of output lakepara.nc does not exist" 1>&2
	exit 1
fi

if ! [[ -d ${output_hdstart_filepath%/*} ]]; then
	echo "Filepath of output hdstart.nc does not exist" 1>&2
	exit 1
fi

if ! [[ -d ${output_lakestart_filepath%/*} ]]; then
	echo "Filepath of output lakestart.nc does not exist" 1>&2
  exit 1
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
	echo "Source directory not set in config file or set to a blank string" 1>&2
	exit 1
fi

if ! [[ -f ${extra_fields_filepath} ]]; then
	echo "Extra fields file (${extra_fields_filepath}) doesn't exist" 1>&2
	exit 1
fi

python_config_file="${ancillary_data_directory}/dynamic_lake_production_driver.cfg"
# Check if python config file exists
if ! [[ -f ${python_config_file} ]]; then
	echo "Python config file (${python_config_file}) doesn't exist!"
	exit 1
fi

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

if ! [[ -d $source_directory ]]; then
	echo "Source directory does not exist." 1>&2
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

#Change to the working directory
cd ${working_directory}

#Check for locks if necesssary and set the compilation_required flag accordingly
if [[ $(uname) == "Darwin" ]]; then
	#Need to compile manually to run on macos
	echo "Warning - manual compilation is required on macos" 1>&2
	compilation_required=false
elif ! [[ $(hostname -d) == "atos.local" ]] ; then
	#We are on the linux system... for some reason the flock utility isn't working on
	#linux system at the moment
	echo "Warning - no locking in place; running multiple copies of this script from the"
	echo "same code release will result in unpredictable results" 1>&2
	if $first_timestep ; then
			compilation_required=true
	else
			compilation_required=false
	fi
else
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
fi

#Setup conda environment
echo "Setting up environment"
if ! $no_modules ; then
  if [[ $(hostname -d) == "atos.local" ]]; then
    		source /etc/profile
    		unload_module netcdf_c
      	unload_module imagemagick
    		unload_module cdo/1.7.0-magicsxx-gcc48
      	unload_module python
	else
		export MODULEPATH="/sw/common/Modules:/client/Modules"
	fi
	load_module nco
fi

if ! $no_modules && ! $no_conda ; then
	if [[ $(hostname -d) == "atos.local" ]]; then
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
  if [[ $(hostname -d) == "atos.local" ]]; then
    load_module gcc/11.2.0-gcc-11.2.0
	else
		load_module gcc/6.3.0
	fi
fi

#Setup correct python path
export PYTHONPATH=${source_directory}/Dynamic_HD_Scripts:${PYTHONPATH}

#Call compilation script
${source_directory}/Dynamic_HD_bash_scripts/compile_dynamic_hd_code.sh ${compilation_required} false ${source_directory} ${working_directory} true "compile_only"

if ! ${first_timestep}; then
output_hdstart_argument=""
else
output_hdstart_argument="-s ${output_hdstart_filepath}"
fi

#Run
echo "Running Dynamic HD Code" 1>&2
python ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/dynamic_hd_and_dynamic_lake_drivers/dynamic_lake_production_run_driver.py ${input_orography_filepath} ${input_ls_mask_filepath} ${input_lake_volumes_filepath} ${present_day_base_orography_filepath} ${glacier_mask_filepath} ${working_directory}/hdpara_out_temp.nc ${output_lakepara_filepath} ${ancillary_data_directory} ${working_directory} ${output_lakestart_filepath} ${output_hdstart_argument}

#Add extra fields to lake parameters file; this would be better integrated into python script
cdo merge ${output_lakepara_filepath} ${extra_fields_filepath} lakepara_out_temp.nc
mv lakepara_out_temp.nc ${output_lakepara_filepath}

#Change lake centers FDIR from 5 to -2
ncap2 -s 'where(FDIR == 5.0) FDIR=-2.0' hdpara_out_temp.nc ${output_hdpara_filepath}

#Release trapped water from hdstart file
if ! ${first_timestep}; then
	${source_directory}/Dynamic_HD_bash_scripts/release_trapped_water_from_hdstart_file.sh ${output_hdpara_filepath} ${input_hdstart_filepath} ${output_hdstart_filepath} ${working_directory} ${no_modules}
fi

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
rm -f catchments.log loops_10min.log loops_30min.log 30minute_filled_orog_temp.nc
rm -f 30minute_river_dirs_temp.nc 30minute_ls_mask_temp.nc hdpara_out_temp.nc .nc

#Generate full diagnostic output label
diagnostic_output_label="${diagnostic_output_exp_id_label}_${diagnostic_output_time_label}"

#Move diagnostic output to target location
if [[ $(ls ${working_directory}) ]]; then
	mkdir -p ${diagnostic_output_directory} || true
	for file in ${working_directory}/*.nc ; do
		mv  $file ${diagnostic_output_directory}/$(basename ${file} .nc)_${diagnostic_output_label}.nc
	done
fi
