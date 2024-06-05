#!/bin/bash
set -e

source_directory=${1}
grid_file=${2}
orography_file=${3}
output_orography_file=${4}
lsmask_file=${5}
true_sinks_file=${6}
next_cell_index_file=${7}
output_catchments_file=${8}
output_accumulated_flow_file=${9}
input_orography_fieldname=${10}
input_lsmask_fieldname=${11}
input_true_sinks_fieldname=${12}
fractional_lsmask_flag=${13:-0}
bifurcate_rivers_flag=${14:false}
input_mouth_position_file=${15}
output_next_cell_index_bifurcated_file=${16}
output_number_of_outflows_file=${17}

if $bifurcate_rivers_flag ; then
  next_cell_index_bifurcated_fieldname="bifurcated_next_cell_index"
else
  output_next_cell_index_bifurcated_file=""
  next_cell_index_bifurcated_fieldname=""
fi

drivers_path=${source_directory}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/command_line_drivers

python ${drivers_path}/sink_filling_icon_driver.py ${orography_file} ${lsmask_file} ${true_sinks_file} ${output_orography_file} ${grid_file} ${input_orography_fieldname} ${input_lsmask_fieldname} ${input_true_sinks_fieldname} 0 0 0.0 1 ${fractional_lsmask_flag}

python ${drivers_path}/determine_river_directions_icon_driver.py ${next_cell_index_file%%.nc}_temp.nc ${output_orography_file} ${lsmask_file}  ${true_sinks_file} ${grid_file} "cell_elevation" ${input_lsmask_fieldname} ${input_true_sinks_fieldname} ${fractional_lsmask_flag} 1 1 0

if $bifurcate_rivers_flag ; then
  python ${drivers_path}/accumulate_flow_icon_driver.py ${grid_file} ${next_cell_index_file%%.nc}_temp.nc ${output_accumulated_flow_file%%.nc}_temp.nc "next_cell_index" "acc"

  python ${drivers_path}/bifurcate_rivers_basic_icon_driver.py ${next_cell_index_file%%.nc}_temp.nc ${output_accumulated_flow_file%%.nc}_temp.nc ${lsmask_file} ${output_number_of_outflows_file} ${next_cell_index_file} ${output_next_cell_index_bifurcated_file} ${grid_file} ${input_mouth_position_file} "next_cell_index" "acc" ${input_lsmask_fieldname} 10 11 0.1
else
   mv ${next_cell_index_file%%.nc}_temp.nc ${next_cell_index_file}
fi

python ${drivers_path}/compute_catchments_icon_driver.py ${next_cell_index_file} ${output_catchments_file} ${grid_file} "next_cell_index" 1 ${output_catchments_file%%.nc}_loops.log 1

python ${drivers_path}/accumulate_flow_icon_driver.py ${grid_file} ${next_cell_index_file} ${output_accumulated_flow_file} "next_cell_index" "acc" ${output_next_cell_index_bifurcated_file} ${next_cell_index_bifurcated_fieldname}
