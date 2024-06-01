#!/bin/bash
set -e
echo "Running Version 1.3 of the ICON HD Parameters Generation Code"

#Define module loading function
function load_module
{
module_name=$1
if [[ $(hostname -d) == "lvt.dkrz.de" ]]; then
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
input_orography_filepath=${1}
input_ls_mask_filepath=${2}
output_hdpara_filepath=${3}
output_catchments_filepath=${4}
output_accumulated_flow_filepath=${5}
config_filepath=${6}
working_directory=${7}
grid_file=${8}
cotat_params_file=${9}
atmos_resolution=${10}
compilation_required=${11:-true}
use_hfrac=${12:-false}
maxl_ls_mask_filepath=${13}
true_sinks_filepath=${14}

#Change first_timestep into a bash command for true or false
shopt -s nocasematch
if [[ $compilation_required == "true" ]] || [[ $compilation_required == "t" ]]; then
	compilation_required=true
elif [[ $compilation_required == "false" ]] || [[ $compilation_required == "f" ]]; then
	compilation_required=false
else
	echo "Format of compilation_required flag is unknown, please use True/False or T/F" 1>&2
	exit 1
fi
shopt -u nocasematch

#Check number of arguments makes sense
if [[ $# -ne 10 ]] && [[ $# -ne 11 ]] && [[ $# -ne 12 ]] && [[ $# -ne 13 ]] && [[ $# -ne 14 ]] ; then
	echo "Wrong number of positional arguments ($# supplied), script only takes between 10 and 14 arguments"	1>&2
	exit 1
fi

if [[ $# -eq 14 ]]; then
	use_truesinks=true
else
	use_truesinks=false
fi

#Check the arguments have the correct file extensions
if ! [[ ${input_orography_filepath##*.} == "nc" ]] || ! [[ ${input_ls_mask_filepath##*.} == "nc" ]] || ! [[ ${grid_file##*.} == "nc" ]] || ! [[ ${cotat_params_file##*.} == "nl" ]]; then
	echo "One or more input files has the wrong file extension" 1>&2
	exit 1
fi

if ${use_hfrac} && ! [[ ${maxl_ls_mask_filepath##*.} == "nc" ]]; then
	echo "One or more input files has the wrong file extension" 1>&2
	exit 1
fi

if ! [[ ${output_hdpara_filepath##*.} == "nc" ]] ; then
	echo "Output hdpara file has the wrong file extension" 1>&2
	exit 1
fi

if  ! [[ ${output_catchments_filepath##*.} == "nc" ]] ; then
	echo "Output catchments file has the wrong file extension" 1>&2
	exit 1
fi

if  ! [[ ${output_accumulated_flow_filepath##*.} == "nc" ]] ; then
	echo "Output accumulated flow file has the wrong file extension" 1>&2
	exit 1
fi

#Convert input filepaths from relative filepaths to absolute filepaths
input_ls_mask_filepath=$(find_abs_path $input_ls_mask_filepath)
input_orography_filepath=$(find_abs_path $input_orography_filepath)
config_filepath=$(find_abs_path $config_filepath)
working_directory=$(find_abs_path $working_directory)
output_hdpara_filepath=$(find_abs_path $output_hdpara_filepath)
output_catchments_filepath=$(find_abs_path $output_catchments_filepath)
output_accumulated_flow_filepath=$(find_abs_path $output_accumulated_flow_filepath)
grid_file=$(find_abs_path $grid_file)
cotat_params_file=$(find_abs_path $cotat_params_file)
if ${use_hfrac}; then
	maxl_ls_mask_filepath=$(find_abs_path $maxl_ls_mask_filepath)
fi

#Check input files, ancillary data directory and diagnostic output directory exist

if ! [[ -e $input_ls_mask_filepath ]] || ! [[ -e $input_orography_filepath ]] || ! [ -e $grid_file ] || ! [ -e $cotat_params_file ]; then
	echo "One or more input files does not exist" 1>&2
	exit 1
fi

if ${use_hfrac} && ! [[ -e $maxl_ls_mask_filepath ]]; then
	echo "One or more input files does not exist" 1>&2
	exit 1
fi

if ! [[ -e $config_filepath ]]; then
	echo "Config file (${config_filepath}) does not exist" 1>&2
	exit 1
fi

if ! [[ -e $cotat_params_file ]]; then
	echo "COTAT parameters file (${cotat_params_file}) does not exist" 1>&2
	exit 1
fi

if ! [[ -d ${output_hdpara_filepath%/*} ]]; then
	echo "Filepath of output hdpara.nc does not exist" 1>&2
	exit 1
fi

if ! [[ -d ${output_catchments_filepath%/*} ]]; then
	echo "Filepath of output catchment file does not exist" 1>&2
	exit 1
fi

if ! [[ -d ${output_accumulated_flow_filepath%/*} ]]; then
	echo "Filepath of output accumulated flow file does not exist" 1>&2
	exit 1
fi

if [[ -e ${output_hdpara_filepath} ]]; then
	echo "Output hdpara.nc already exists" 1>&2
	exit 1
fi

if [[ -e ${output_catchments_filepath} ]]; then
	echo "Output catchment file already exists" 1>&2
	exit 1
fi

if [[ -e ${output_accumulated_flow_filepath} ]]; then
	echo "Output accumulated flow file already exists" 1>&2
	exit 1
fi

if egrep -v -q "^(#.*|.*=.*)$" ${config_filepath}; then
	echo "Config file has wrong format" 1>&2
	exit 1
fi

# Read in source_directory
source ${config_filepath}

# Check we have actually read the variables correctly
if [[ -z ${source_directory} ]]; then
	echo "Source directory not set in config file or set to a blank string" 1>&2
	exit 1
fi

# Prepare a working directory if it doesn't already exist
if ! [[ -e $working_directory ]]; then
	echo "Creating a working directory"
	mkdir -p $working_directory
fi

#Check that the working directory, source directory and external source directory exist
if ! [[ -d $working_directory ]]; then
		echo "Working directory does not exist or is not a directory (even after attempt to create it)" 1>&2
	exit 1
fi

if ! [[ -d $source_directory ]]; then
	echo "Source directory (${source_directory}) does not exist." 1>&2
	exit 1
fi

if [[ -z ${python_config_filepath} ]]; then
	echo "Python config file not set in config file or set to a blank string" 1>&2
	exit 1
fi

python_config_filepath=$(find_abs_path $python_config_filepath)

if ! [[ -e $python_config_filepath ]]; then
	echo "Python config file ($python_config_filepath) does not exist" 1>&2
	exit 1
fi

shopt -s nocasematch
no_env_gen=${no_env_gen:-"false"}
if [[ $no_env_gen == "true" ]] || [[ $no_env_gen == "t" ]]; then
        no_env_gen=true
elif [[ $no_env_gen == "false" ]] || [[ $no_env_gen == "f" ]]; then
        no_env_gen=false
else
        echo "Format of no_env_gen flag (${no_env_gen}) is unknown, please use True/False or T/F" 1>&2
        exit 1
fi

no_conda=${no_conda:-"false"}
if [[ ${no_conda} == "true" ]] || [[ $no_conda == "t" ]]; then
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
shopt -u nocasematch

#Change to the working directory
cd ${working_directory}

# #Check for locks if necesssary and set the compilation_required flag accordingly
# exec 200>"${source_directory}/compilation.lock"
# if flock -x -n 200 ; then
# 	compilation_required=true
# else
# 	flock -s 200
# 	compilation_required=false
# fi

#Setup conda environment
echo "Setting up environment"
if ! $no_modules ; then
  if [[ $(hostname -d) == "lvt.dkrz.de" ]]; then
    		source /etc/profile
    		unload_module netcdf_c
      	unload_module imagemagick
    		unload_module cdo/1.7.0-magicsxx-gcc48
      	unload_module python
	else
			  export MODULEPATH="/sw/common/Modules:/client/Modules"
	fi
fi

export DYLD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DYLD_LIBRARY_PATH

if ! $no_modules && ! $no_conda ; then
	if [[ $(hostname -d) == "lvt.dkrz.de" ]]; then
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
  if [[ $(hostname -d) == "lvt.dkrz.de" ]]; then
    load_module gcc/11.2.0-gcc-11.2.0
	else
		load_module gcc/6.3.0
	fi
fi

if ! $no_modules ; then
	load_module nco
fi

#Setup correct python path
export PYTHONPATH=${source_directory}/Dynamic_HD_Scripts:${source_directory}/lib:${PYTHONPATH}

#Call compilation script
${source_directory}/Dynamic_HD_bash_scripts/compile_dynamic_hd_code.sh ${compilation_required} false ${source_directory} ${working_directory} false "tools_only"

if $use_truesinks; then
	true_sinks_argument="-t ${true_sinks_filepath}"
else
	true_sinks_argument=""
fi

#Clean up any temporary files from previous runs
rm -f downscaled_ls_mask_temp.nc
rm -f downscaled_ls_mask_temp_inverted.nc
rm -f ten_minute_river_direction_temp.nc
rm -f ten_minute_catchments_temp.nc
rm -f ten_minute_accumulated_flow_temp.nc
rm -f icon_intermediate_catchments.nc
rm -f icon_intermediate_rdirs.nc
rm -f icon_intermediate_catchments_loops_log.log
rm -f icon_intermediate_accumulated_flow.nc
rm -f zeros_temp.nc
rm -f icon_final_rdirs.nc
rm -f orography_filled.nc
rm -f grid_in_temp.nc
rm -f mask_in_tmep.nc
rm -f cell_numbers_temp.nc

#Run
echo "Running ICON HD River Direction Generation Code" 1>&2
echo "Generating LatLon to ICON Cross Grid Mapping" 1>&2
ten_minute_orography_file_and_fieldname=$(python ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/utilities/icon_rdirs_creation_config_printer.py ${python_config_filepath} | sed '1d')
ten_minute_orography_filepath=$(cut -d ' ' -f 1 <<< ${ten_minute_orography_file_and_fieldname})
ten_minute_orography_fieldname=$(cut -d ' ' -f 2 <<< ${ten_minute_orography_file_and_fieldname})
cell_numbers_filepath="cell_numbers_temp.nc"
drivers_path=${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/command_line_drivers
python ${drivers_path}/cross_grid_mapper_latlon_to_icon_driver.py ${grid_file} ${ten_minute_orography_filepath} ${cell_numbers_filepath}
echo "Downscaling Landsea Mask" 1>&2
python ${drivers_path}/icon_to_latlon_landsea_downscaler_driver.py ${cell_numbers_filepath} ${input_ls_mask_filepath} downscaled_ls_mask_temp.nc "cell_numbers" "cell_sea_land_mask"
cdo expr,'lsm=(!lsm)' downscaled_ls_mask_temp.nc downscaled_ls_mask_temp_inverted.nc
echo "Generating Combined Hydrosheds and Corrected Data River Directions" 1>&2
ten_minute_river_direction_filepath="ten_minute_river_direction_temp.nc"
ten_minute_catchments_filepath="ten_minute_catchments_temp.nc"
ten_minute_accumulated_flow_filepath="ten_minute_accumulated_flow_temp.nc"
icon_intermediate_catchments_filepath="icon_intermediate_catchments.nc"
icon_intermediate_rdirs_filepath="icon_intermediate_rdirs.nc"
icon_intermediate_accumulated_flow_filepath="icon_intermediate_accumulated_flow.nc"
icon_final_filepath="icon_final_rdirs.nc"
python ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/command_line_drivers/create_icon_coarse_river_directions_driver.py ${true_sinks_argument} downscaled_ls_mask_temp_inverted.nc ${ten_minute_river_direction_filepath} ${ten_minute_catchments_filepath} ${ten_minute_accumulated_flow_filepath} ${python_config_filepath} ${working_directory}
while [[ $(grep -c "[0-9]" "${ten_minute_catchments_filepath%%.nc}_loops.log") -ne 0 ]]; do
	echo "Loop found, edit river directions file ${ten_minute_river_direction_filepath}"
	echo -e "Edited river direction filepath:"
	read -e ten_minute_river_direction_filepath
	while ! [[ -e $ten_minute_river_direction_filepath ]] ; do
		echo "Not a valid file. Re-enter filepath."
		echo -e "Edited river direction filepath:"
		read -e ten_minute_river_direction_filepath
	done
	rm -f ten_minute_catchments_temp.nc
	rm -f ten_minute_accumulated_flow_temp.nc
	rm -f ten_minute_catchments_temp_loops.log
	python ${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/command_line_drivers/create_icon_coarse_river_directions_driver.py -r ${ten_minute_river_direction_filepath} downscaled_ls_mask_temp_inverted.nc dummy.nc ${ten_minute_catchments_filepath} ${ten_minute_accumulated_flow_filepath} ${python_config_filepath} ${working_directory}
done
python ${drivers_path}/cotat_plus_latlon_to_icon_driver.py ${ten_minute_river_direction_filepath}  ${ten_minute_accumulated_flow_filepath} ${grid_file} ${icon_intermediate_rdirs_filepath} "rdirs" "acc" ${cotat_params_file}
python ${drivers_path}/compute_catchments_icon_driver.py --sort-catchments-by-size ${icon_intermediate_rdirs_filepath} ${icon_intermediate_catchments_filepath} ${grid_file} "next_cell_index" ${icon_intermediate_catchments_filepath%%.nc}_loops_log.log
python ${drivers_path}/accumulate_flow_icon_driver.py ${grid_file} ${icon_intermediate_rdirs_filepath} ${icon_intermediate_accumulated_flow_filepath} "next_cell_index"
python ${drivers_path}/latlon_to_icon_loop_breaker_driver.py ${ten_minute_accumulated_flow_filepath}  ${ten_minute_river_direction_filepath}  ${cell_numbers_filepath}  ${grid_file} ${icon_final_filepath} ${icon_intermediate_catchments_filepath} ${icon_intermediate_accumulated_flow_filepath} ${icon_intermediate_rdirs_filepath}  "rdirs" "acc" "cell_numbers" "catchment" "acc" "next_cell_index"  ${icon_intermediate_catchments_filepath%%.nc}_loops_log.log
python ${drivers_path}/compute_catchments_icon_driver.py --sort-catchments-by-size ${icon_final_filepath} ${output_catchments_filepath} ${grid_file} "next_cell_index" ${output_catchments_filepath%%.nc}_loops_log.log
python ${drivers_path}/accumulate_flow_icon_driver.py ${grid_file} ${icon_final_filepath} ${output_accumulated_flow_filepath} "next_cell_index"
  rm -f paragen/area_dlat_dlon.txt
  rm -f paragen/ddir.inp
  rm -f paragen/hd_partab.txt
  rm -f paragen/hd_up
  rm -f paragen/hdpara_icon.nc
  rm -f paragen/mo_read_icon_trafo.mod
  rm -f paragen/paragen.inp
  rm -f paragen/paragen_icon.mod
  rm -f paragen/paragen_icon_driver
  rmdir paragen || true
  mkdir paragen
  cp ${grid_file} grid_in_temp.nc
  cp ${input_ls_mask_filepath} mask_in_temp.nc
  cdo expr,"acc=(next_cell_index == -9999999)" ${icon_intermediate_rdirs_filepath} zeros_temp.nc
  python ${drivers_path}/sink_filling_icon_driver.py ${input_orography_filepath} ${input_ls_mask_filepath} zeros_temp.nc orography_filled.nc ${grid_file} "z" "cell_sea_land_mask" "acc"
  ${source_directory}/Dynamic_HD_bash_scripts/parameter_generation_scripts/generate_icon_hd_file_driver.sh ${working_directory}/paragen ${source_directory}/Dynamic_HD_bash_scripts/parameter_generation_scripts/fortran ${working_directory} grid_in_temp.nc mask_in_temp.nc ${icon_final_filepath}  orography_filled.nc
${source_directory}/Dynamic_HD_bash_scripts/adjust_icon_k_parameters.sh  ${working_directory}/paragen/hdpara_icon.nc ${working_directory}/hdpara_adjusted_temp.nc ${atmos_resolution}
if ! ${use_hfrac}; then
	mv ${working_directory}/hdpara_adjusted_temp.nc ${output_hdpara_filepath}
else
 ${source_directory}/Dynamic_HD_bash_scripts/adjust_icon_hdpara_for_partial_ls.sh ${maxl_ls_mask_filepath}  ${input_ls_mask_filepath} ${working_directory}/hdpara_adjusted_temp.nc  ${output_hdpara_filepath} ${atmos_resolution}
  rm -f hdpara_adjusted_temp.nc
fi
  #Clean up temporary files
  rm -f paragen/area_dlat_dlon.txt
  rm -f paragen/ddir.inp
  rm -f paragen/hd_partab.txt
  rm -f paragen/hd_up
  rm -f paragen/hdpara_icon.nc
  rm -f paragen/mo_read_icon_trafo.mod
  rm -f paragen/paragen.inp
  rm -f paragen/paragen_icon.mod
  rm -f paragen/paragen_icon_driver
  #rmdir paragen
  rm -f downscaled_ls_mask_temp.nc
  rm -f downscaled_ls_mask_temp_inverted.nc
  rm -f grid_in_temp.nc
  rm -f mask_in_temp.nc
  rm -f orography_filled.nc
  rm -f icon_intermediate_rdirs.nc
  rm -f icon_intermediate_catchments_loops_log.log
  rm -f icon_intermediate_catchments.nc
  rm -f icon_intermediate_accumulated_flow.nc
  rm -f zeros_temp.nc
  rm -f icon_final_rdirs.nc
  rm -f cell_numbers_temp.nc
