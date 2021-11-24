#!/bin/bash
set -e
export LD_LIBRARY_PATH=/Users/thomasriddick/anaconda3/pkgs/netcdf-cxx4-4.3.0-h703b707_9/lib:/Users/thomasriddick/anaconda3/lib:${LD_LIBRARY_PATH}
export DYLD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DYLD_LIBRARY_PATH

icon_data_dir="/Users/thomasriddick/Documents/data/ICONHDdata"
cpp_icon_tool_dir="/Users/thomasriddick/Documents/workspace/Dynamic_HD_Code/Dynamic_HD_Cpp_Code/Release"
#grid_file=${icon_data_dir}/gridfiles/icon_grid_0030_R02B03_G.nc
#grid_file=${icon_data_dir}/gridfiles/icon_grid_0015_R02B09_G.nc
#grid_file=${icon_data_dir}/gridfiles/icon_grid_0013_R02B04_G.nc
grid_file=${icon_data_dir}/gridfiles/icon_grid_0019_R02B05_G.nc
#orography_file=${icon_data_dir}/orographies/orog_etop1_remap_to_r2b3_0030_G.nc
#orography_file=${icon_data_dir}/orographies/icon_extpar_grid_0015_R02B09_G_20180828_added_grid.nc
#orography_file=${icon_data_dir}/orographies/orog_etop1_remap_to_r2b4_0013_G.nc
orography_file=${icon_data_dir}/orographies/orog_top1_remap_to_r2b5_0019_G.nc
#output_orography_file=${icon_data_dir}/orographies/orog_etop1_remap_to_r2b3_0030_G_filled_copy.nc
#output_orography_file=${icon_data_dir}/orographies/icon_extpar_grid_0015_R02B09_G_20180828_filled_v3.nc
#output_orography_file=${icon_data_dir}/orographies/orog_etop1_remap_to_r2b4_0013_G_filled_mask_from_grid_file.nc
output_orography_file=${icon_data_dir}/orographies/orog_top1_remap_to_r2b5_0019_G_filled.nc
#lsmask_file=${icon_data_dir}/lsmasks/lsmHD.maxlnd.r2b3_0030_G.nc
#lsmask_file=${icon_data_dir}/lsmasks/lsm_015_R02B09.nc
#lsmask_file=${icon_data_dir}/lsmasks/maxlnd_lsm_013_0031.nc
#lsmask_file=${icon_data_dir}/lsmasks/lsm_013_R2B4_from_grid_file.nc
lsmask_file=${icon_data_dir}/lsmasks/maxlnd_lsm_019_0032.nc
#true_sinks_file=${icon_data_dir}/truesinks/truesinks.r2b3_0030_G.nc
#true_sinks_file=${icon_data_dir}/truesinks/truesinks_R02B09_G.nc
#true_sinks_file=${icon_data_dir}/truesinks/true_sinks_R02B04_G.nc
true_sinks_file=${icon_data_dir}/truesinks/truesinks_R02B05_0019_G.nc
#next_cell_index_file=${icon_data_dir}/rdirs/r2b3_0030_G_rdirs_copy.nc
#next_cell_index_file=${icon_data_dir}/rdirs/R02B09_0030_G_rdirs_v3.nc
#next_cell_index_file=${icon_data_dir}/rdirs/rdirs_0013_R02B04_G_mask_from_grid_file.nc
next_cell_index_file=${icon_data_dir}/rdirs/rdirs_0019_0032_G.nc
#catchments_file=${icon_data_dir}/catchments/r2b3_0030_G_catchments_copy.nc
#catchments_file=${icon_data_dir}/catchments/R02B09_0030_G_catchments_v3.nc
#catchments_file=${icon_data_dir}/catchments/catchments_0013_R02B04_G_mask_from_grid_file.nc
catchments_file=${icon_data_dir}/catchments/R2B5_0019_0032_G_catchments.nc
input_orography_fieldname="z"
#input_orography_fieldname="topography_c"
#input_orography_fieldname="z"
#input_lsmask_fieldname="cell_sea_land_mask"
input_lsmask_fieldname="cell_sea_land_mask"
#input_lsmask_fieldname="cell_sea_land_mask"
#input_true_sinks_fieldname="cell_sea_land_mask"
input_true_sinks_fieldname="true_sinks"
#input_true_sinks_fieldname="true_sinks"
fractional_lsmask_flag=0

${cpp_icon_tool_dir}/Fill_Sinks_Icon_SI_Exec ${orography_file} ${lsmask_file} ${true_sinks_file} ${output_orography_file} ${grid_file} ${input_orography_fieldname} ${input_lsmask_fieldname} ${input_true_sinks_fieldname} 0 0 0.0 1 ${fractional_lsmask_flag}

${cpp_icon_tool_dir}/Determine_River_Directions_SI_Exec ${next_cell_index_file} ${output_orography_file} ${lsmask_file}  ${true_sinks_file} ${grid_file} "cell_elevation" ${input_lsmask_fieldname} ${input_true_sinks_fieldname} ${fractional_lsmask_flag} 1 1 0

${cpp_icon_tool_dir}/Compute_Catchments_SI_Exec ${next_cell_index_file} ${catchments_file} ${grid_file} "next_cell_index"
