module IOModule

using NetCDF
using NetCDF: NcFile,NcVar
using GridModule: Grid, LatLonGrid, get_number_of_dimensions
using FieldModule: Field, LatLonDirectionIndicators,round,convert,add_offset,get_data
using HDModule: RiverParameters
using LakeModule: LakeParameters,LatLonLakeParameters, GridSpecificLakeParameters,LakeFields
using MergeTypesModule
using NetCDF
using NetCDF: NcFile,NcVar
import LakeModule: write_lake_numbers_field

function load_field(file_handle::NcFile,grid::Grid,variable_name::AbstractString,
                    field_type::DataType,;timestep::Int64=-1)
  variable::NcVar = file_handle[variable_name]
  values::Array{field_type,get_number_of_dimensions(grid)} = NetCDF.readvar(variable)
  values = permutedims(values, [2,1])
  return Field{field_type}(grid,values)
end

function write_field(grid::LatLonGrid,variable_name::AbstractString,
                     field::Field,;timestep::Int64=-1)
  lat = NcDim("Lat",grid.nlat)
  lon = NcDim("Lon",grid.nlon)
  variable::NcVar = NcVar(variable_name,[ lat; lon ])
  filepath::String = timestep == -1 ? "/Users/thomasriddick/Documents/data/temp/lake_model_out.nc" : "/Users/thomasriddick/Documents/data/temp/lake_model_out_$(timestep).nc"
  NetCDF.create(filepath,variable)
  NetCDF.putvar(variable,get_data(field))
end

function load_river_parameters(hd_para_filepath::AbstractString,grid::Grid)
  println("Loading: " * hd_para_filepath)
  file_handle::NcFile = NetCDF.open(hd_para_filepath)
  try
    if isa(grid,LatLonGrid)
      direction_based_river_directions = load_field(file_handle,grid,"FDIR",Float64)
      flow_directions = LatLonDirectionIndicators(round(Int64,direction_based_river_directions))
    end
    river_reservoir_nums::Field{Int64} =
      round(Int64,load_field(file_handle,grid,"ARF_N",Float64))
    overland_reservoir_nums::Field{Int64} =
      round(Int64,load_field(file_handle,grid,"ALF_N",Float64))
    base_reservoir_nums::Field{Int64} = Field{Int64}(grid,1)
    river_retention_coefficients::Field{Float64} =
      load_field(file_handle,grid,"ARF_K",Float64)
    overland_retention_coefficients::Field{Float64} =
      load_field(file_handle,grid,"ALF_K",Float64)
    base_retention_coefficients::Field{Float64} =
      load_field(file_handle,grid,"AGF_K",Float64)
    landsea_mask::Field{Bool} =
      load_field(file_handle,grid,"FLAG",Bool)
    return  RiverParameters(flow_directions,
                          river_reservoir_nums,
                          overland_reservoir_nums,
                          base_reservoir_nums,
                          river_retention_coefficients,
                          overland_retention_coefficients,
                          base_retention_coefficients,
                          landsea_mask,grid)
  finally
    NetCDF.close(file_handle)
  end
end

function load_lake_parameters(lake_para_filepath::AbstractString,grid::Grid,hd_grid::Grid)
  println("Loading: " * lake_para_filepath)
  file_handle::NcFile = NetCDF.open(lake_para_filepath)
  try
    grid_specific_lake_parameters::GridSpecificLakeParameters =
      load_grid_specific_lake_parameters(file_handle,grid)
    lake_centers::Field{Bool} =
      load_field(file_handle,grid,"lake_centers",Bool)
    connection_volume_thresholds::Field{Float64} =
      load_field(file_handle,grid,"connection_volume_thresholds",Float64)
    flood_volume_thresholds::Field{Float64} =
      load_field(file_handle,grid,"flood_volume_thresholds",Float64)
    flood_local_redirect::Field{Bool} =
      load_field(file_handle,grid,"flood_local_redirect",Bool)
    connect_local_redirect::Field{Bool} =
      load_field(file_handle,grid,"connect_local_redirect",Bool)
    additional_flood_local_redirect::Field{Bool} =
      load_field(file_handle,grid,"additional_flood_local_redirect",Bool)
    additional_connect_local_redirect::Field{Bool} =
      load_field(file_handle,grid,"additional_connect_local_redirect",Bool)
    merge_points::Field{MergeTypes} =
      load_field(file_handle,grid,"merge_points",Int64)
    return LakeParameters(lake_centers,
                          connection_volume_thresholds,
                          flood_volume_thresholds,
                          flood_local_redirect,
                          connect_local_redirect,
                          additional_flood_local_redirect,
                          additional_connect_local_redirect,
                          merge_points,grid,hd_grid,
                          grid_specific_lake_parameters)
  finally
    NetCDF.close(file_handle)
  end
end

function load_grid_specific_lake_parameters(file_handle::NcFile,grid::LatLonGrid)
  flood_next_cell_lat_index::Field{Int64} =
      load_field(file_handle,grid,"flood_next_cell_lat_index",Int64)
  flood_next_cell_lon_index::Field{Int64} =
      load_field(file_handle,grid,"flood_next_cell_lon_index",Int64)
  connect_next_cell_lat_index::Field{Int64} =
      load_field(file_handle,grid,"connect_next_cell_lat_index",Int64)
  connect_next_cell_lon_index::Field{Int64} =
      load_field(file_handle,grid,"connect_next_cell_lon_index",Int64)
  flood_force_merge_lat_index::Field{Int64} =
      load_field(file_handle,grid,"flood_force_merge_lat_index",Int64)
  flood_force_merge_lon_index::Field{Int64} =
      load_field(file_handle,grid,"flood_force_merge_lon_index",Int64)
  connect_force_merge_lat_index::Field{Int64} =
      load_field(file_handle,grid,"connect_force_merge_lat_index",Int64)
  connect_force_merge_lon_index::Field{Int64} =
      load_field(file_handle,grid,"connect_force_merge_lon_index",Int64)
  flood_redirect_lat_index::Field{Int64} =
      load_field(file_handle,grid,"flood_redirect_lat_index",Int64)
  flood_redirect_lon_index::Field{Int64} =
      load_field(file_handle,grid,"flood_redirect_lon_index",Int64)
  connect_redirect_lat_index::Field{Int64} =
      load_field(file_handle,grid,"connect_redirect_lat_index",Int64)
  connect_redirect_lon_index::Field{Int64} =
      load_field(file_handle,grid,"connect_redirect_lon_index",Int64)
  additional_flood_redirect_lat_index::Field{Int64} =
      load_field(file_handle,grid,"additional_flood_redirect_lat_index",Int64)
  additional_flood_redirect_lon_index::Field{Int64} =
      load_field(file_handle,grid,"additional_flood_redirect_lon_index",Int64)
  additional_connect_redirect_lat_index::Field{Int64} =
      load_field(file_handle,grid,"additional_connect_redirect_lat_index",Int64)
  additional_connect_redirect_lon_index::Field{Int64} =
      load_field(file_handle,grid,"additional_connect_redirect_lon_index",Int64)
  add_offset(flood_next_cell_lat_index,1,Int64[-1])
  add_offset(flood_next_cell_lon_index,1,Int64[-1])
  add_offset(connect_next_cell_lat_index,1,Int64[-1])
  add_offset(connect_next_cell_lon_index,1,Int64[-1])
  add_offset(flood_force_merge_lat_index,1,Int64[-1])
  add_offset(flood_force_merge_lon_index,1,Int64[-1])
  add_offset(connect_force_merge_lat_index,1,Int64[-1])
  add_offset(connect_force_merge_lon_index,1,Int64[-1])
  add_offset(flood_redirect_lat_index,1,Int64[-1])
  add_offset(flood_redirect_lon_index,1,Int64[-1])
  add_offset(connect_redirect_lat_index,1,Int64[-1])
  add_offset(connect_redirect_lon_index,1,Int64[-1])
  add_offset(additional_flood_redirect_lat_index,1,Int64[-1])
  add_offset(additional_flood_redirect_lon_index,1,Int64[-1])
  add_offset(additional_connect_redirect_lat_index,1,Int64[-1])
  add_offset(additional_connect_redirect_lon_index,1,Int64[-1])
  return LatLonLakeParameters(flood_next_cell_lat_index,
                              flood_next_cell_lon_index,
                              connect_next_cell_lat_index,
                              connect_next_cell_lon_index,
                              flood_force_merge_lat_index,
                              flood_force_merge_lon_index,
                              connect_force_merge_lat_index,
                              connect_force_merge_lon_index,
                              flood_redirect_lat_index,
                              flood_redirect_lon_index,
                              connect_redirect_lat_index,
                              connect_redirect_lon_index,
                              additional_flood_redirect_lat_index,
                              additional_flood_redirect_lon_index,
                              additional_connect_redirect_lat_index,
                              additional_connect_redirect_lon_index)
end

function write_lake_numbers_field(lake_parameters::LakeParameters,lake_fields::LakeFields;
                                  timestep::Int64=-1)
  variable_name::String = "lake_field"
  write_field(lake_parameters.grid,variable_name,
              lake_fields.lake_numbers,timestep=timestep)
end

end
