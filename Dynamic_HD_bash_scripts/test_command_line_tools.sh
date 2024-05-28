#!/bin/bash
set -e

function compare_helper {
  compare=$1
  compare_with=$2
  echo -e "Comparing file: \n ${compare}"
  echo -e "with Reference file: \n ${compare_with}"
  echo "Result:"
  if ! [[ ${compare%%.nc} == ${compare} ]]; then
        cdo diffn ${compare} ${compare_with}
  else
        diff ${compare} ${compare_with}
  fi
}

function test_command { 
  cmd=$1 
  compare=$2
  compare_with=$3
  compare_two=$4
  compare_with_two=$5
  compare_three=$6
  compare_with_three=$7
  echo -e "Running command:\n ${cmd}"
  echo "Output:"
  eval '${cmd}'
  compare_helper ${compare} ${compare_with}
  if [[ $# -ge 4 ]]; then
    compare_helper ${compare_two} ${compare_with_two}
  fi
  if [[ $# -ge 6 ]]; then
    compare_helper ${compare_three} ${compare_with_three}
  fi
}

if [[ $# -ne 2 ]]; then
  echo "Wrong number of command line arguments"
fi

#Command line arguments
workdir=$1
repo_base_dir=$2
#end of command line arguments

drivers_dir=${repo_base_dir}/Dynamic_HD_Scripts/Dynamic_HD_Scripts/command_line_drivers
test_data_dir=${workdir}/tests
reference_data_dir=${workdir}/refs
export PYTHONPATH=${repo_base_dir}/lib:${repo_base_dir}/Dynamic_HD_Scripts:${PYTHONPATH}
source activate dyhdenv3
cd ${workdir}

grid_params_in="icon_grid_0030_R02B03_G.nc"
rdirs_file_in="rdirs_test_data.nc"
rdirs_fieldname_in="next_cell_index"
rdirs_for_bifurcation_file_in="rdirs_from_bifurcation_test_data.nc"
rdirs_for_bifurcation_fieldname_in="next_cell_index"
fine_rdirs_file_in="ten_minute_river_direction_temp.nc"
fine_rdirs_fieldname_in="rdirs"
bifurcated_rdirs_file_in="bifurcated_rdirs_test_data.nc"
bifurcated_rdirs_fieldname_in="bifurcated_next_cell_index"
rdirs_file_with_loops_in="rdirs_test_data_with_loops.nc"
rdirs_fieldname_with_loops_in="next_cell_index"
catchment_file_with_loops_in="catchments_tests_data_with_loops.nc"
catchment_fieldname_with_loops_in="catchment"
acc_file_in="acc_test_data.nc"
acc_fieldname_in="acc"
acc_file_with_loops_in="acc_test_data_with_loops.nc"
acc_fieldname_with_loops_in="acc"
fine_acc_file_in="ten_minute_accumulated_flow_temp.nc"
fine_acc_fieldname_in="acc"
fractional_lsmask_file_in="bc_land_frac_1850.nc"
fractional_lsmask_fieldname_in="notsea"
lsmask_file_in="maxlnd_lsm_0030_0036.nc"
lsmask_fieldname_in="cell_sea_land_mask"
river_mouths_positions_file_in="river_mouth_position_test_data.txt"
orography_file_in="filled_orog_test_data.nc"
orography_fieldname_in="cell_elevation"
unfilled_orography_file_in="orog_etop1_remap_to_r2b3_0030_G.nc"
unfilled_orography_fieldname_in="z"
fine_orography_file_in="corrected_orog_intermediary_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson_lsmask_0k_20170517_003802_with_grid.nc"
fine_orography_fieldname_in="field_value"
true_sinks_file_in="true_sinks_version_41_mapped_to_r2b3.nc"
true_sinks_fieldname_in="true_sinks"
subcatchment_list_file_in="subcatchment_list_test_data.txt"
fine_cell_numbers_file_in="latlon_to_icon_map_test_data.nc"
fine_cell_numbers_fieldname_in="cell_numbers"
cotat_plus_parameters_file_in=${repo_base_dir}/Dynamic_HD_Resources/cotat_plus_standard_params.nl
loop_log_file_in="loops_log_test_data_with_loops.txt"

mkdir ${test_data_dir}
mkdir ${test_data_dir}/sink_filling
mkdir ${test_data_dir}/determine_river_directions
mkdir ${test_data_dir}/catchment_computation
mkdir ${test_data_dir}/acc
mkdir ${test_data_dir}/river_bifurcation
mkdir ${test_data_dir}/landsea_mask_downscaler
mkdir ${test_data_dir}/cross_grid_mapper
mkdir ${test_data_dir}/cotat_plus
mkdir ${test_data_dir}/loop_breaker

echo "========================"
echo "FLOW ACCUMULATION DRIVER"
echo "========================"
acc_driver=${drivers_dir}/accumulate_flow_icon_driver.py
acc_output_one_a=${test_data_dir}/acc/acc_test_a.nc
acc_output_one_a_ref=${reference_data_dir}/acc/acc_ref_a.nc
test_command "python ${acc_driver} ${grid_params_in} ${rdirs_file_in} ${acc_output_one_a} ${rdirs_fieldname_in}" ${acc_output_one_a} ${acc_output_one_a_ref}
acc_output_one_b=${test_data_dir}/acc/acc_test_b.nc
acc_output_one_b_ref=${reference_data_dir}/acc/acc_ref_b.nc
test_command "python ${acc_driver} ${grid_params_in} ${rdirs_for_bifurcation_file_in} ${acc_output_one_b} ${rdirs_for_bifurcation_fieldname_in}
                     ${bifurcated_rdirs_file_in} ${bifurcated_rdirs_fieldname_in}" ${acc_output_one_b} ${acc_output_one_b_ref}

echo "========================"
echo "RIVER BIFURCATION DRIVER"
echo "========================"
bifurcate_rivers_driver=${drivers_dir}/bifurcate_rivers_basic_icon_driver.py
bifurcate_rivers_output_one_a=${test_data_dir}/river_bifurcation/num_of_outflow_test_a.nc
bifurcate_rivers_output_one_a_ref=${reference_data_dir}/river_bifurcation/num_of_outflow_ref_a.nc
bifurcate_rivers_output_two_a=${test_data_dir}/river_bifurcation/rdirs_from_bifurcation_test_a.nc
bifurcate_rivers_output_two_a_ref=${reference_data_dir}/river_bifurcation/rdirs_from_bifurcation_ref_a.nc
bifurcate_rivers_output_three_a=${test_data_dir}/river_bifurcation/bifurcated_rdirs_test_a.nc
bifurcate_rivers_output_three_a_ref=${reference_data_dir}/river_bifurcation/bifurcated_rdirs_ref_a.nc
test_command "python ${bifurcate_rivers_driver} ${rdirs_file_in} ${acc_file_in} ${lsmask_file_in} ${bifurcate_rivers_output_one_a}
                                                ${bifurcate_rivers_output_two_a} ${bifurcate_rivers_output_three_a} ${grid_params_in}
                                                ${river_mouths_positions_file_in} ${rdirs_fieldname_in} ${acc_fieldname_in} ${lsmask_fieldname_in}
                                                2 3 0.3" ${bifurcate_rivers_output_one_a} ${bifurcate_rivers_output_one_a_ref} \
                                                ${bifurcate_rivers_output_two_a} ${bifurcate_rivers_output_two_a_ref} \
                                                ${bifurcate_rivers_output_three_a} ${bifurcate_rivers_output_three_a_ref}

echo "============================"
echo "CATCHMENT COMPUTATION DRIVER"
echo "============================"
compute_catchments_driver=${drivers_dir}/compute_catchments_icon_driver.py 
compute_catchments_output_one_a=${test_data_dir}/catchment_computation/catchments_test_a.nc
compute_catchments_output_one_a_ref=${reference_data_dir}/catchment_computation/catchments_ref_a.nc
test_command "python ${compute_catchments_driver} ${rdirs_file_in} ${compute_catchments_output_one_a}
                                                  ${grid_params_in} ${rdirs_fieldname_in}" ${compute_catchments_output_one_a} ${compute_catchments_output_one_a_ref}
echo "+++"
compute_catchments_output_one_b=${test_data_dir}/catchment_computation/catchments_test_b.nc
compute_catchments_output_one_b_ref=${reference_data_dir}/catchment_computation/catchments_ref_b.nc
compute_catchments_output_two_b=${test_data_dir}/catchment_computation/loops_test_b.txt
compute_catchments_output_two_b_ref=${reference_data_dir}/catchment_computation/loops_ref_b.txt
test_command "python ${compute_catchments_driver} ${rdirs_file_in} ${compute_catchments_output_one_b}
                                                  ${grid_params_in} ${rdirs_fieldname_in}
                                                  ${compute_catchments_output_two_b}" ${compute_catchments_output_one_b} ${compute_catchments_output_one_b_ref} \
                                                                                      ${compute_catchments_output_two_b} ${compute_catchments_output_two_b_ref}
echo "+++"
compute_catchments_output_one_c=${test_data_dir}/catchment_computation/catchments_test_c.nc
compute_catchments_output_one_c_ref=${reference_data_dir}/catchment_computation/catchments_ref_c.nc
compute_catchments_output_two_c=${test_data_dir}/catchment_computation/loops_test_c.txt
compute_catchments_output_two_c_ref=${reference_data_dir}/catchment_computation/loops_ref_c.txt
test_command "python ${compute_catchments_driver} ${rdirs_file_in} ${compute_catchments_output_one_c}
                                                  ${grid_params_in} ${rdirs_fieldname_in} ${compute_catchments_output_two_c} --sort-catchments-by-size" \
                                                  ${compute_catchments_output_one_c} ${compute_catchments_output_one_c_ref} \
                                                  ${compute_catchments_output_two_c} ${compute_catchments_output_two_c_ref}
echo "+++"
compute_catchments_output_one_d=${test_data_dir}/catchment_computation/catchments_test_d.nc
compute_catchments_output_one_d_ref=${reference_data_dir}/catchment_computation/catchments_ref_d.nc
test_command "python ${compute_catchments_driver} ${rdirs_file_in} ${compute_catchments_output_one_d}
                                                  ${grid_params_in} ${rdirs_fieldname_in} --subcatchment-list-filepath ${subcatchment_list_file_in}" \
                                                  ${compute_catchments_output_one_d} ${compute_catchments_output_one_d_ref} 

echo "================="
echo "COTAT PLUS DRIVER"
echo "================="
cotat_plus_driver=${drivers_dir}/cotat_plus_latlon_to_icon_driver.py 
cotat_plus_output_one_a=${test_data_dir}/cotat_plus/river_directions_test_a.nc
cotat_plus_output_one_a_ref=${reference_data_dir}/cotat_plus/river_directions_ref_a.nc
test_command "python ${cotat_plus_driver} ${fine_rdirs_file_in} ${fine_acc_file_in} ${grid_params_in}
                                          ${cotat_plus_output_one_a} ${fine_rdirs_fieldname_in} ${fine_acc_fieldname_in}
                                          ${cotat_plus_parameters_file_in}" \
					  ${cotat_plus_output_one_a} ${cotat_plus_output_one_a_ref}
echo "+++"
cotat_plus_output_one_b=${test_data_dir}/cotat_plus/river_directions_test_b.nc
cotat_plus_output_one_b_ref=${reference_data_dir}/cotat_plus/river_directions_ref_b.nc
cotat_plus_output_two_b=${test_data_dir}/cotat_plus/latlon_to_icon_map_test_b.nc
cotat_plus_output_two_b_ref=${reference_data_dir}/cotat_plus/latlon_to_icon_map_ref_b.nc
test_command "python ${cotat_plus_driver} ${fine_rdirs_file_in} ${fine_acc_file_in} ${grid_params_in}
                                          ${cotat_plus_output_one_b} ${fine_rdirs_fieldname_in} ${fine_acc_fieldname_in}
                                          ${cotat_plus_parameters_file_in} ${cotat_plus_output_two_b}" \
					  ${cotat_plus_output_one_b} ${cotat_plus_output_one_b_ref} \
					  ${cotat_plus_output_two_b} ${cotat_plus_output_two_b_ref}
echo "========================"
echo "CROSS GRID MAPPER DRIVER"
echo "========================"
cross_grid_mapper_driver=${drivers_dir}/cross_grid_mapper_latlon_to_icon_driver.py
cross_grid_mapper_output_one_a=${test_data_dir}/cross_grid_mapper/latlon_to_icon_map_test_a.nc
cross_grid_mapper_output_one_a_ref=${reference_data_dir}/cross_grid_mapper/latlon_to_icon_map_ref_a.nc
#Use fine river directions rather than fine orography to avoid issues with exact lat/lon pixel positions
#and floating point errors that prevent bit comparison
test_command "python ${cross_grid_mapper_driver} ${grid_params_in} ${fine_rdirs_file_in} 
						 ${cross_grid_mapper_output_one_a}" \
						 ${cross_grid_mapper_output_one_a} ${cross_grid_mapper_output_one_a_ref} 
echo "================================="
echo "DETERMINE RIVER DIRECTIONS DRIVER"
echo "================================="
determine_rdirs_driver=${drivers_dir}/determine_river_directions_icon_driver.py
determine_rdirs_output_one_a=${test_data_dir}/determine_river_directions/rdirs_test_a.nc
determine_rdirs_output_one_a_ref=${reference_data_dir}/determine_river_directions/rdirs_ref_a.nc
test_command "python ${determine_rdirs_driver} ${determine_rdirs_output_one_a} ${orography_file_in}
                                               ${lsmask_file_in} ${true_sinks_file_in}
                                               ${grid_params_in} ${orography_fieldname_in} ${lsmask_fieldname_in}
                                               ${true_sinks_fieldname_in} --always-flow-to-lowest" \
					       ${determine_rdirs_output_one_a} ${determine_rdirs_output_one_a_ref}
echo "+++"
determine_rdirs_output_one_b=${test_data_dir}/determine_river_directions/rdirs_test_b.nc
determine_rdirs_output_one_b_ref=${reference_data_dir}/determine_river_directions/rdirs_ref_b.nc
test_command "python ${determine_rdirs_driver} ${determine_rdirs_output_one_b} ${orography_file_in}
                                               ${fractional_lsmask_file_in} ${true_sinks_file_in}
                                               ${grid_params_in} ${orography_fieldname_in} ${fractional_lsmask_fieldname_in}
                                               ${true_sinks_fieldname_in} --always-flow-to-lowest --fractional-landsea-mask" \
                                               ${determine_rdirs_output_one_b} ${determine_rdirs_output_one_b_ref}
echo "+++"
determine_rdirs_output_one_c=${test_data_dir}/determine_river_directions/rdirs_test_c.nc
determine_rdirs_output_one_c_ref=${reference_data_dir}/determine_river_directions/rdirs_ref_c.nc
test_command "python ${determine_rdirs_driver} ${determine_rdirs_output_one_c} ${orography_file_in}
                                               ${lsmask_file_in} ${true_sinks_file_in}
                                               ${grid_params_in} ${orography_fieldname_in} ${lsmask_fieldname_in}
                                               ${true_sinks_fieldname_in}" \
                                               ${determine_rdirs_output_one_c} ${determine_rdirs_output_one_c_ref}
echo "+++"
determine_rdirs_output_one_d=${test_data_dir}/determine_river_directions/rdirs_test_d.nc
determine_rdirs_output_one_d_ref=${reference_data_dir}/determine_river_directions/rdirs_ref_d.nc
test_command "python ${determine_rdirs_driver} ${determine_rdirs_output_one_d} ${orography_file_in}
                                               ${lsmask_file_in} ${true_sinks_file_in}
                                               ${grid_params_in} ${orography_fieldname_in} ${lsmask_fieldname_in}
                                               ${true_sinks_fieldname_in} 
					       --always-flow-to-lowest --mark-pits-as-true-sinks" \
                                               ${determine_rdirs_output_one_d} ${determine_rdirs_output_one_d_ref}
echo "============================="
echo "DOWNSCALE LANDSEA MASK DRIVER"
echo "============================="
landsea_downscaler_driver=${drivers_dir}/icon_to_latlon_landsea_downscaler_driver.py
landsea_downscaler_output_one_a=${test_data_dir}/landsea_mask_downscaler/landsea_mask_test_a.nc
landsea_downscaler_output_one_a_ref=${reference_data_dir}/landsea_mask_downscaler/landsea_mask_ref_a.nc
test_command "python ${landsea_downscaler_driver} ${fine_cell_numbers_file_in} ${lsmask_file_in} 
						  ${landsea_downscaler_output_one_a}
                                               	  ${fine_cell_numbers_fieldname_in} ${lsmask_fieldname_in}" \
						  ${landsea_downscaler_output_one_a} \
						  ${landsea_downscaler_output_one_a_ref}

echo "==================="
echo "LOOP BREAKER DRIVER"
echo "==================="
loop_breaker_driver=${drivers_dir}/latlon_to_icon_loop_breaker_driver.py
loop_breaker_output_one_a=${test_data_dir}/loop_breaker/river_directions_test_a.nc
loop_breaker_output_one_a_ref=${reference_data_dir}/loop_breaker/river_directions_ref_a.nc
test_command "python ${loop_breaker_driver}  ${fine_acc_file_in} ${fine_rdirs_file_in} ${fine_cell_numbers_file_in}
                                             ${grid_params_in} ${loop_breaker_output_one_a} ${catchment_file_with_loops_in}
                                             ${acc_file_with_loops_in} ${rdirs_file_with_loops_in} ${fine_rdirs_fieldname_in}
                                             ${fine_acc_fieldname_in} ${fine_cell_numbers_fieldname_in} ${catchment_fieldname_with_loops_in}
                                             ${acc_fieldname_with_loops_in} ${rdirs_fieldname_with_loops_in} ${loop_log_file_in}" \
					     ${loop_breaker_output_one_a} ${loop_breaker_output_one_a_ref}

echo "==================="
echo "SINK FILLING DRIVER"
echo "==================="
sink_filling_driver=${drivers_dir}/sink_filling_icon_driver.py
sink_filling_output_one_a=${test_data_dir}/sink_filling/filled_orog_test_a.nc
sink_filling_output_one_a_ref=${reference_data_dir}/sink_filling/filled_orog_ref_a.nc
test_command "python ${sink_filling_driver} ${unfilled_orography_file_in} ${lsmask_file_in} ${true_sinks_file_in}
                                            ${sink_filling_output_one_a} ${grid_params_in} ${unfilled_orography_fieldname_in}
                                            ${lsmask_fieldname_in} ${true_sinks_fieldname_in}" ${sink_filling_output_one_a} ${sink_filling_output_one_a_ref}
echo "+++"
sink_filling_output_one_b=${test_data_dir}/sink_filling/filled_orog_test_b.nc
sink_filling_output_one_b_ref=${reference_data_dir}/sink_filling/filled_orog_ref_b.nc
test_command "python ${sink_filling_driver} ${unfilled_orography_file_in} ${lsmask_file_in} ${true_sinks_file_in}
                                            ${sink_filling_output_one_b} ${grid_params_in} ${unfilled_orography_fieldname_in}
                                            ${lsmask_fieldname_in} ${true_sinks_fieldname_in} --set-land-sea-as-no-data" \
                                            ${sink_filling_output_one_b} ${sink_filling_output_one_b_ref}
echo "+++"
sink_filling_output_one_c=${test_data_dir}/sink_filling/filled_orog_test_c.nc
sink_filling_output_one_c_ref=${reference_data_dir}/sink_filling/filled_orog_ref_c.nc
test_command "python ${sink_filling_driver} ${unfilled_orography_file_in} ${fractional_lsmask_file_in} ${true_sinks_file_in}
                                            ${sink_filling_output_one_c} ${grid_params_in} ${unfilled_orography_fieldname_in}
                                            ${fractional_lsmask_fieldname_in} ${true_sinks_fieldname_in} --fractional-landsea-mask" \
                                            ${sink_filling_output_one_c} ${sink_filling_output_one_c_ref}
echo "+++"
sink_filling_output_one_d=${test_data_dir}/sink_filling/filled_orog_test_d.nc
sink_filling_output_one_d_ref=${reference_data_dir}/sink_filling/filled_orog_ref_d.nc
test_command "python ${sink_filling_driver} ${unfilled_orography_file_in} ${fractional_lsmask_file_in} ${true_sinks_file_in}
                                            ${sink_filling_output_one_d} ${grid_params_in} ${unfilled_orography_fieldname_in}
                                            ${fractional_lsmask_fieldname_in} ${true_sinks_fieldname_in} --add-slope 1.5" \
                                            ${sink_filling_output_one_d} ${sink_filling_output_one_d_ref}
echo "done"
