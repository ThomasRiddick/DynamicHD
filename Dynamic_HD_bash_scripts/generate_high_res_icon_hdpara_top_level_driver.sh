#!/bin/bash
set -e

cd $(dirname ${0})
version=$(git describe --match 'icon_hd_tools_version_*')
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

rdirs_only=false
case ${1} in
  -r | --rdirs-only)
    rdirs_only=true
    shift
    ;;
  *)
    ;;
esac

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
input_mouth_position_filepath=${16}


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
shopt -u nocasematch

#Check number of arguments makes sense
if [[ $# -ne 14 && $# -ne 16 ]]; then
  echo "Wrong number of positional arguments ($# supplied), script only takes 14 or 16 arguments"  1>&2
  exit 1
fi

if [[ $# -eq 11 ]]; then
  use_truesinks=true
else
  use_truesinks=false
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
input_mouth_position_filepath=$(find_abs_path $input_mouth_position_filepath)
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

if ${no_conda} ; then
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

#Clean up any temporary files from previous runs
rm -f next_cell_index.nc
rm -f orography_filled.nc
rm -f grid_in_temp.nc
rm -f mask_in_tmep.nc

#Run
next_cell_index_filepath="next_cell_index.nc"
next_cell_index_bifurcated_filepath="bifurcated_next_cell_index.nc"
number_of_outflows_filepath="number_of_outflows.nc"
${source_directory}/Dynamic_HD_bash_scripts/generate_fine_icon_rdirs.sh ${source_directory} ${grid_file} ${input_orography_filepath} orography_filled.nc ${input_ls_mask_filepath} ${input_true_sinks_filepath} ${next_cell_index_filepath} ${output_catchments_filepath} accumulated_flow.nc ${input_orography_fieldname} ${input_lsmask_fieldname} ${input_true_sinks_fieldname} ${bifurcate_rivers} ${input_mouth_position_filepath} ${next_cell_index_bifurcated_filepath} ${number_of_outflows_filepath}
mv accumulated_flow.nc ${output_accumulated_flow_filepath}
cp ${grid_file} grid_in_temp.nc
cp ${input_ls_mask_filepath} mask_in_temp.nc
if $rdirs_only ; then
  cp ${next_cell_index_filepath} ${output_hdpara_filepath}.nc
else
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
  ${source_directory}/Dynamic_HD_bash_scripts/parameter_generation_scripts/generate_icon_hd_file_driver.sh ${working_directory}/paragen ${source_directory}/Dynamic_HD_bash_scripts/parameter_generation_scripts/fortran ${working_directory} grid_in_temp.nc mask_in_temp.nc ${next_cell_index_filepath} orography_filled.nc ${bifurcate_rivers} ${next_cell_index_bifurcated_filepath} ${number_of_outflows_filepath}
  ${source_directory}/Dynamic_HD_bash_scripts/adjust_icon_k_parameters.sh  ${working_directory}/paragen/hdpara_icon.nc ${output_hdpara_filepath} ${atmos_resolution}

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
  unlink paragen/bifurcated_next_cell_index_for_upstream_cell.nc || true
  rmdir paragen || true
  rm -f orography_filled.nc
  rm -f grid_in_temp.nc
  rm -f mask_in_temp.nc
  rm -f next_cell_index.nc
  rm -f next_cell_index_temp.nc
  rm -f number_of_outflows.nc
  rm -f bifurcated_next_cell_index.nc
  rm -f accumulated_flow_temp.nc
fi
