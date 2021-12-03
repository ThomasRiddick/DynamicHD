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

${source_directory}/Dynamic_HD_Cpp_Code/Release/Fill_Sinks_Icon_SI_Exec ${orography_file} ${lsmask_file} ${true_sinks_file} ${output_orography_file} ${grid_file} ${input_orography_fieldname} ${input_lsmask_fieldname} ${input_true_sinks_fieldname} 0 0 0.0 1 ${fractional_lsmask_flag}

${source_directory}/Dynamic_HD_Cpp_Code/Release/Determine_River_Directions_SI_Exec ${next_cell_index_file%%.nc}_temp.nc ${output_orography_file} ${lsmask_file}  ${true_sinks_file} ${grid_file} "cell_elevation" ${input_lsmask_fieldname} ${input_true_sinks_fieldname} ${fractional_lsmask_flag} 1 1 0

if $bifurcate_rivers_flag ; then
  ${source_directory}/Dynamic_HD_Fortran_Code/Release/Accumulate_Flow_Icon_Simple_Interface_Exec ${grid_file} ${next_cell_index_file%%.nc}_temp.nc ${output_accumulated_flow_file%%.nc}_temp.nc "next_cell_index" "acc"

  ${source_directory}/Dynamic_HD_Cpp_Code/Release/Bifurcate_River_Basic_SI_Exec ${next_cell_index_file%%.nc}_temp.nc ${output_accumulated_flow_file%%.nc}_temp.nc ${lsmask_file} ${output_number_of_outflows_file} ${next_cell_index_file} ${output_next_cell_index_bifurcated_file} ${grid_file} ${input_mouth_position_file} "next_cell_index" "acc" ${input_lsmask_fieldname} 10 11 0.1
else
   mv ${next_cell_index_file%%.nc}_temp.nc ${next_cell_index_file}
fi

${source_directory}/Dynamic_HD_Cpp_Code/Release/Compute_Catchments_SI_Exec ${next_cell_index_file} ${output_catchments_file} ${grid_file} "next_cell_index" 1 ${output_catchments_file%%.nc}_loops.log 1

${source_directory}/Dynamic_HD_Fortran_Code/Release/Accumulate_Flow_Icon_Simple_Interface_Exec ${grid_file} ${next_cell_index_file} ${output_accumulated_flow_file} "next_cell_index" "acc" ${output_next_cell_index_bifurcated_file} ${next_cell_index_bifurcated_fieldname}
