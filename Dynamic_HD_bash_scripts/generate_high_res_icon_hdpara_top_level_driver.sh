#!/bin/bash
set -e

cd $(dirname ${0})
version=$(git describe --match 'icon_hd_tools_version_*')
export VS=${version}
cd -
echo "Running Version ${version} of the ICON HD Parameters Generation Code"

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

#Define module unloading function (only works on levante)
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
input_true_sinks_filepath=${3}
input_orography_fieldname=${4}
input_lsmask_fieldname=${5}
input_true_sinks_fieldname=${6}
output_hdpara_filepath=${7}
output_catchments_filepath=${8}
output_accumulated_flow_filepath=${9}
config_filepath=${10}
working_directory=${11}
grid_file=${12}
atmos_resolution=${13}
compilation_required=${14}
bifurcate_rivers=${15:-"false"}
river_deltas_filepath=${16}
search_areas_filepath=${17}
use_existing_river_mouth_position_file=${18:-"false"}
input_river_mouth_position_filepath=${19}


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
if [[ $bifurcate_rivers == "true" ]] || [[ $bifurcate_rivers == "t" ]]; then
  bifurcate_rivers=true
elif [[ $bifurcate_rivers == "false" ]] || [[ $bifurcate_rivers == "f" ]]; then
  bifurcate_rivers=false
else
  echo "Format of bifurcate_rivers flag is unknown, please use True/False or T/F" 1>&2
  exit 1
fi
if [[ $use_existing_river_mouth_position_file == "true" ]] || [[ $use_existing_river_mouth_position_file == "t" ]]; then
  use_existing_river_mouth_position_file=true
elif [[ $use_existing_river_mouth_position_file == "false" ]] || [[ $use_existing_river_mouth_position_file == "f" ]]; then
  use_existing_river_mouth_position_file=false
else
  echo "Format of use_existing_river_mouth_position_file flag is unknown, please use True/False or T/F" 1>&2
  exit 1
fi
shopt -u nocasematch

#Check number of arguments makes sense
if [[ $# -ne 14 && $# -ne 17 && $# -ne 19 ]]; then
  echo "Wrong number of positional arguments ($# supplied), script only takes 14, 17 or 19 arguments"  1>&2
  exit 1
fi

#Check the arguments have the correct file extensions
if ! [[ ${input_orography_filepath##*.} == "nc" ]] || ! [[ ${input_ls_mask_filepath##*.} == "nc" ]] || ! [[ ${grid_file##*.} == "nc" ]] || ! [[ ${input_true_sinks_filepath##*.} == "nc" ]]; then
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
input_true_sinks_filepath=$(find_abs_path $input_true_sinks_filepath)
if ${bifurcate_rivers} ; then
  river_deltas_filepath=$(find_abs_path $river_deltas_filepath)
  search_areas_filepath=$(find_abs_path $search_areas_filepath)
  if ${use_existing_river_mouth_position_file} ; then
    input_river_mouth_position_filepath=$(find_abs_path $input_river_mouth_position_filepath)
  fi
fi
config_filepath=$(find_abs_path $config_filepath)
working_directory=$(find_abs_path $working_directory)
output_hdpara_filepath=$(find_abs_path $output_hdpara_filepath)
output_catchments_filepath=$(find_abs_path $output_catchments_filepath)
output_accumulated_flow_filepath=$(find_abs_path $output_accumulated_flow_filepath)
grid_file=$(find_abs_path $grid_file)

#Check input files, ancillary data directory and diagnostic output directory exist

if ! [[ -e $config_filepath ]]; then
  echo "Config file does not exist" 1>&2
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

shopt -s nocasematch
no_mamba=${no_mamba:-"false"}
if [[ ${no_mamba} == "true" ]] || [[ $no_mamba == "t" ]]; then
  no_mamba=true
elif [[ $no_mamba == "false" ]] || [[ $no_mamba == "f" ]]; then
  no_mamba=false
else
  echo "Format of no_mamba flag (${no_mamba}) is unknown, please use True/False or T/F" 1>&2
  exit 1
fi

no_modules=${no_modules:-"false"}
if [[ $no_modules == "true" ]] || [[ $no_modules == "t" ]]; then
  no_modules=true
elif [[ $no_modules == "false" ]] || [[ $no_modules == "f" ]]; then
  no_modules=false
else
  echo "Format of no_modules flag ($no_modules}) is unknown, please use True/False or T/F" 1>&2
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

if ${no_mamba} ; then
no_env_gen=false
fi

shopt -u nocasematch

#Change to the working directory
cd ${working_directory}

# #Check for locks if necesssary and set the compilation_required flag accordingly
# exec 200>"${source_directory}/compilation.lock"
# if flock -x -n 200 ; then
#   compilation_required=true
# else
#   flock -s 200
#   compilation_required=false
# fi

#Setup mamba environment
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

if ! $no_modules && ! $no_mamba ; then
  if [[ $(hostname -d) == "lvt.dkrz.de" ]]; then
    load_module python3
  fi
fi

if ! $no_mamba && ! $no_env_gen ; then
  if $compilation_required && mamba info -e | grep -q "dyhdenv_mamba"; then
    mamba env remove --yes --name dyhdenv_mamba
  fi
  if ! mamba info -e | grep -q "dyhdenv_mamba"; then
    ${source_directory}/Dynamic_HD_bash_scripts/regenerate_mamba_environment.sh $no_modules
  fi
fi
if ! $no_mamba ; then
  #Reactivating mamba environment a second time can cause errors
  #Mamba uses conda environmental variables
  if ! [[ ${CONDA_DEFAULT_ENV} == "dyhdenv_mamba" ]]; then
    source activate dyhdenv_mamba
  fi
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
  load_module julia
fi

#Setup correct python path
export PYTHONPATH=${source_directory}/Dynamic_HD_Scripts:${source_directory}/lib:${PYTHONPATH}

#Setup correct julia path
export JULIA_LOAD_PATH=${source_directory}/src/julia_src/bifurcated_rivermouth_identification_tool:${source_directory}/src/julia_src/cross_grid_true_sink_transfer_tool:${JULIA_LOAD_PATH}

#Call compilation script
${source_directory}/Dynamic_HD_bash_scripts/compile_dynamic_hd_code.sh ${compilation_required} false ${source_directory} ${working_directory} false "tools_only"

#Clean up any temporary files from previous runs
rm -f next_cell_index.nc
rm -f orography_filled.nc
rm -f grid_in_temp.nc
rm -f mask_in_tmep.nc

#Run
next_cell_index_filepath="next_cell_index.nc"
next_cell_index_bifurcated_filepath="bifurcated_next_cell_index.nc"
number_of_outflows_filepath="number_of_outflows.nc"

if $bifurcate_rivers ; then
  next_cell_index_bifurcated_fieldname="bifurcated_next_cell_index"
else
  next_cell_index_bifurcated_filepath=""
  next_cell_index_bifurcated_fieldname=""
fi

drivers_path=${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/command_line_drivers

python ${drivers_path}/sink_filling_icon_driver.py ${input_orography_filepath} ${input_ls_mask_filepath} ${input_true_sinks_filepath} orography_filled.nc ${grid_file} ${input_orography_fieldname} ${input_lsmask_fieldname} ${input_true_sinks_fieldname}

python ${drivers_path}/determine_river_directions_icon_driver.py ${next_cell_index_filepath%%.nc}_temp.nc orography_filled.nc ${input_ls_mask_filepath}  ${input_true_sinks_filepath} ${grid_file} "cell_elevation" ${input_lsmask_fieldname} ${input_true_sinks_fieldname}

if $bifurcate_rivers ; then
  python ${drivers_path}/accumulate_flow_icon_driver.py ${grid_file} ${next_cell_index_filepath%%.nc}_temp.nc accumulated_flow_temp.nc "next_cell_index"
  if ${use_existing_river_mouth_position_file} ; then
    river_mouth_position_filepath=${input_river_mouth_position_filepath}
  else
    #Determine bifurcated river mouth positions
    river_mouth_position_filepath=${working_directory}/rivermouths.txt
    cmd="julia ${source_directory}/src/julia_src/bifurcated_rivermouth_identification_tool/BifurcatedRiverMouthIdentificationDriver.jl -g ${grid_file} -m ${input_ls_mask_filepath} -n cell_sea_land_mask -r ${river_deltas_filepath} -o ${river_mouth_position_filepath} -a accumulated_flow_temp.nc -f acc -s ${search_areas_filepath}"
    echo "Running:"
    echo ${cmd}
    eval ${cmd}
  fi
  python ${drivers_path}/bifurcate_rivers_basic_icon_driver.py --remove-main-channel ${next_cell_index_filepath%%.nc}_temp.nc accumulated_flow_temp.nc ${input_ls_mask_filepath} ${number_of_outflows_filepath} ${next_cell_index_filepath} ${next_cell_index_bifurcated_filepath} ${grid_file} ${river_mouth_position_filepath} "next_cell_index" "acc" ${input_lsmask_fieldname} 10 11 0.1
else
   mv ${next_cell_index_filepath%%.nc}_temp.nc ${next_cell_index_filepath}
fi

python ${drivers_path}/compute_catchments_icon_driver.py --sort-catchments-by-size ${next_cell_index_filepath} ${output_catchments_filepath} ${grid_file} "next_cell_index" ${output_catchments_filepath%%.nc}_loops.log

python ${drivers_path}/accumulate_flow_icon_driver.py ${grid_file} ${next_cell_index_filepath} accumulated_flow.nc "next_cell_index" ${next_cell_index_bifurcated_filepath} ${next_cell_index_bifurcated_fieldname}

mv accumulated_flow.nc ${output_accumulated_flow_filepath}
cp ${grid_file} grid_in_temp.nc
cp ${input_ls_mask_filepath} mask_in_temp.nc
rm -rf paragen
mkdir paragen
${source_directory}/Dynamic_HD_bash_scripts/parameter_generation_scripts/generate_icon_hd_file_driver.sh ${working_directory}/paragen ${source_directory}/Dynamic_HD_bash_scripts/parameter_generation_scripts/fortran ${working_directory} grid_in_temp.nc mask_in_temp.nc ${next_cell_index_filepath} orography_filled.nc ${bifurcate_rivers} ${next_cell_index_bifurcated_filepath} ${number_of_outflows_filepath}
${source_directory}/Dynamic_HD_bash_scripts/adjust_icon_k_parameters.sh  ${working_directory}/paragen/hdpara_icon.nc ${output_hdpara_filepath} ${atmos_resolution}

#Clean up temporary files
unlink paragen/bifurcated_next_cell_index_for_upstream_cell.nc || true
rm -rf paragen
rm -f orography_filled.nc
rm -f grid_in_temp.nc
rm -f mask_in_temp.nc
rm -f next_cell_index.nc
rm -f next_cell_index_temp.nc
rm -f number_of_outflows.nc
rm -f bifurcated_next_cell_index.nc
rm -f accumulated_flow_temp.nc
