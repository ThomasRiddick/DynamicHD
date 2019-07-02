module IOModule

using NetCDF
using NetCDF: NcFile,NcVar
using GridModule: Grid, LatLonGrid, get_number_of_dimensions
using FieldModule: Field, LatLonDirectionIndicators,round,convert,invert,add_offset,get_data
using FieldModule: maximum
using HDModule: RiverParameters,RiverPrognosticFields
using LakeModule: LakeParameters,LatLonLakeParameters, GridSpecificLakeParameters,LakeFields
using MergeTypesModule
using NetCDF
using NetCDF: NcFile,NcVar
import LakeModule: write_lake_numbers_field,write_lake_volumes_field
import HDModule: write_river_initial_values

function load_field(file_handle::NcFile,grid::Grid,variable_name::AbstractString,
                    field_type::DataType,;timestep::Int64=-1)
  variable::NcVar = file_handle[variable_name]
  values::Array{field_type,get_number_of_dimensions(grid)} = NetCDF.readvar(variable)
  if timestep == -1
    values = permutedims(values, [2,1])
  else
    values = permutedims(values, [2,1])[timestep]
  end
  return Field{field_type}(grid,values)
end

function load_array_of_fields(file_handle::NcFile,grid::Grid,variable_base_name::AbstractString,
                              field_type::DataType,number_of_fields::Int64)
  if number_of_fields == 1
    return Field{field_type}[load_field(file_handle,grid,variable_base_name,field_type)]
  else
    array_of_fields::Array{Field{field_type},1} = Array{Field{field_type}}(undef,number_of_fields)
    for i = 1:number_of_fields
      array_of_fields[i] = load_field(file_handle,grid,variable_base_name*string(i),field_type)
    end
    return array_of_fields
  end
end

function prep_array_of_fields(dims::Array{NcDim},
                              variable_base_name::AbstractString,
                              array_of_fields::Array{Field{T},1},
                              fields_to_write::Vector{Pair}) where {T}
  number_of_fields::Int64 = size(array_of_fields,1)
  if number_of_fields == 1
    return prep_field(dims,variable_base_name,array_of_fields[1],fields_to_write)
  else
    for i = 1:number_of_fields
      prep_field(dims,variable_base_name*string(i),
                 array_of_fields[i],fields_to_write)
    end
  end
end

function prep_dims(grid::LatLonGrid)
  lat = NcDim("Lat",grid.nlat)
  lon = NcDim("Lon",grid.nlon)
  dims::Array{NcDim} = NcDim[ lon; lat ]
  return dims
end

function prep_field(dims::Array{NcDim},variable_name::AbstractString,
                    field::Field,fields_to_write::Vector{Pair})
  variable::NcVar = NcVar(variable_name,dims)
  push!(fields_to_write,Pair(variable,permutedims(get_data(field), [2,1])))
end

function write_fields(fields_to_write::Vector{Pair},filepath::AbstractString)
  variables::Array{NcVar} = NcVar[ pair.first for pair in fields_to_write ]
  NetCDF.create(filepath,variables)
  for (variable,data) in fields_to_write
    NetCDF.putvar(variable,data)
  end
end

function write_field(grid::LatLonGrid,variable_name::AbstractString,
                     field::Field,filepath::AbstractString)
  lat = NcDim("Lat",grid.nlat)
  lon = NcDim("Lon",grid.nlon)
  variable::NcVar = NcVar(variable_name,[ lon; lat ])
  NetCDF.create(filepath,variable)
  NetCDF.putvar(variable,permutedims(get_data(field), [2,1]))
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
    landsea_mask = invert(landsea_mask)
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

function load_river_initial_values(hd_start_filepath::AbstractString,
                                   river_parameters::RiverParameters)
  river_prognostic_fields::RiverPrognosticFields =
    RiverPrognosticFields(river_parameters)
  println("Loading: " * hd_start_filepath)
  file_handle::NcFile = NetCDF.open(hd_start_filepath)
  try
    river_prognostic_fields.river_inflow =
      load_field(file_handle,river_parameters.grid,"FINFL",Float64)
    river_prognostic_fields.base_flow_reservoirs =
      load_array_of_fields(file_handle,river_parameters.grid,"FGMEM",Float64,
                            maximum(river_parameters.base_reservoir_nums))
    river_prognostic_fields.overland_flow_reservoirs =
      load_array_of_fields(file_handle,river_parameters.grid,"FLFMEM",Float64,
                            maximum(river_parameters.overland_reservoir_nums))
    river_prognostic_fields.river_flow_reservoirs =
      load_array_of_fields(file_handle,river_parameters.grid,"FRFMEM",Float64,
                            maximum(river_parameters.river_reservoir_nums))
  finally
    NetCDF.close(file_handle)
  end
  return river_prognostic_fields
end

function write_river_initial_values(hd_start_filepath::AbstractString,
                                    river_parameters::RiverParameters,
                                    river_prognostic_fields::RiverPrognosticFields)
  println("Writing: " * hd_start_filepath)
  fields_to_write::Vector{Pair} = Pair[]
  dims::Array{NcDim} = prep_dims(river_parameters.grid)
  prep_field(dims,"FINFL",
             river_prognostic_fields.river_inflow,fields_to_write)
  prep_array_of_fields(dims,"FGMEM",
                       river_prognostic_fields.base_flow_reservoirs,
                       fields_to_write)
  prep_array_of_fields(dims,"FLFMEM",
                       river_prognostic_fields.overland_flow_reservoirs,
                       fields_to_write)
  prep_array_of_fields(dims,"FRFMEM",
                       river_prognostic_fields.river_flow_reservoirs,
                       fields_to_write)
  write_fields(fields_to_write,hd_start_filepath)
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

function load_lake_initial_values(lake_para_filepath::AbstractString,
                                  grid::LatLonGrid,hd_grid::LatLonGrid)
  local initial_water_to_lake_centers::Field{Float64}
  local initial_spillover_to_rivers::Field{Float64}
  println("Loading: " * lake_para_filepath)
  file_handle::NcFile = NetCDF.open(lake_para_filepath)
  try
    initial_water_to_lake_centers =
      load_field(file_handle,grid,"water_redistributed_to_lakes",Float64)
    initial_spillover_to_rivers =
      load_field(file_handle,hd_grid,"water_redistributed_to_rivers",Float64)
  finally
    NetCDF.close(file_handle)
  end
  return initial_water_to_lake_centers,initial_spillover_to_rivers
end

function write_lake_numbers_field(lake_parameters::LakeParameters,lake_fields::LakeFields;
                                  timestep::Int64=-1)
  variable_name::String = "lake_field"
  filepath::String = timestep == -1 ? "/Users/thomasriddick/Documents/data/temp/transient_sim_1/lake_model_results.nc" : "/Users/thomasriddick/Documents/data/temp/transient_sim_1/lake_model_results_$(timestep).nc"
  write_field(lake_parameters.grid,variable_name,
              lake_fields.lake_numbers,filepath)
end

function write_lake_volumes_field(lake_parameters::LakeParameters,lake_volumes::Field{Float64})
  variable_name::String = "lake_field"
  filepath::String = "/Users/thomasriddick/Documents/data/temp/transient_sim_1/lake_model_out.nc"
  write_field(lake_parameters.grid,variable_name,lake_volumes,filepath)
end

function load_drainage_fields(drainages_filename::AbstractString,grid::LatLonGrid;
                              first_timestep::Int64=1,last_timestep::Int64=1)
  println("Loading: " * drainages_filename)
  file_handle::NcFile = NetCDF.open(drainages_filename)
  drainages::Array{Field{Float64},1} = Field{Float64}[]
  try
    for i::Int64 = first_timestep:last_timestep
      drainages[i] =  load_field(file_handle,grid,"drainages",
                                 Float64;timestep=i)
    end
  finally
    NetCDF.close(file_handle)
  end
  return drainages
end

function load_runoff_fields(runoffs_filename::AbstractString,grid::LatLonGrid;
                            first_timestep::Int64=1,last_timestep::Int64=1)
  println("Loading: " * runoffs_filename)
  file_handle::NcFile = NetCDF.open(runoffs_filename)
  runoffs::Array{Field{Float64},1} = Field{Float64}[]
  try
    for i::Int64 = first_timestep:last_timestep
      runoffs[i] = load_field(file_handle,grid,"runoffs",
                              Float64;timestep=i)
    end
  finally
    NetCDF.close(file_handle)
  end
  return runoffs
end

end
