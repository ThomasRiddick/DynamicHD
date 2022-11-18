#!/bin/bash
set -e
echam_restart_file="/Users/thomasriddick/Documents/data/simulation_data/restart_rid003_echam_25991231.nc"
#For now using a lon -179.9167 to 179.9167 ten minute grid file
ten_minute_grid_file=/Users/thomasriddick/Documents/data/HDdata/orographys/generated/updated_orog_0k_ice5g_lake_no_intermediaries_lake_corrections_driver_20220913_120136.nc

ncap2 -s 'defdim("lat",48);lat_index=array(1,1,/$lat/)' lat_field.nc
ncap2 -s 'defdim("lon",96);lon_index=array(1,1,/$lon/)' lon_field.nc
ncap2 -s 'defdim("lon",96);lon_index=array(1,0,/$lon/)' lon_field_ones.nc
ncap2 -s 'defdim("lat",48);lat_index=array(1,0,/$lat/)' lat_field_ones.nc
cdo merge lat_field.nc lon_field_ones.nc lat_field_merged.nc
cdo merge lon_field.nc lat_field_ones.nc lon_field_merged.nc
ncap2 -s 'corresponding_surface_cell_lat_index[lat,lon]=lat_index*lon_index' lat_field_merged.nc lat_field2d_temp.nc
ncap2 -s 'corresponding_surface_cell_lon_index[lat,lon]=lat_index*lon_index' lon_field_merged.nc lon_field2d_temp.nc
cdo select,name="corresponding_surface_cell_lat_index" lat_field2d_temp.nc lat_field2d_temp2.nc
cdo select,name="corresponding_surface_cell_lon_index" lon_field2d_temp.nc lon_field2d_temp2.nc
cdo setgrid,${echam_restart_file}:2 lat_field2d_temp2.nc lat_field2d.nc
cdo setgrid,${echam_restart_file}:2 lon_field2d_temp2.nc lon_field2d.nc
cdo remapnn,${ten_minute_grid_file} lat_field2d.nc lat_field2d_remapped.nc
cdo remapnn,${ten_minute_grid_file} lon_field2d.nc lon_field2d_remapped.nc
cdo merge lat_field2d_remapped.nc lon_field2d_remapped.nc t31_to_ten_min_map_v2.nc
