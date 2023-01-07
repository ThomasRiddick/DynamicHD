module IOModule

using NetCDF
using NetCDF: NcFile,NcVar,NC_NETCDF4
using GridModule: Grid, LatLonGrid, UnstructuredGrid, get_number_of_dimensions
using FieldModule: Field, LatLonField,UnstructuredField,LatLonDirectionIndicators
using FieldModule: UnstructuredDirectionIndicators
using FieldModule: round,convert,invert,add_offset,get_data,maximum,equals
using HDModule: RiverParameters,RiverPrognosticFields
using LakeModule: LakeParameters,LatLonLakeParameters, UnstructuredLakeParameters
using LakeModule: GridSpecificLakeParameters,LakeFields
using MergeTypesModule

import LakeModule: write_lake_numbers_field,write_lake_volumes_field
import LakeModule: write_diagnostic_lake_volumes_field
import HDModule: write_river_initial_values,write_river_flow_field

function get_ncells(file_name::AbstractString)
  river_directions::Array{Float64,1} = NetCDF.ncread(file_name,"FDIR")
  return size(river_directions,1)
end

function get_additional_grid_information(file_name::AbstractString)
  clat::Array{Float64,1} = NetCDF.ncread(file_name,"clat")
  clon::Array{Float64,1} = NetCDF.ncread(file_name,"clon")
  clat_bounds::Array{Float64,2} = NetCDF.ncread(file_name,"clat_bnds")
  clon_bounds::Array{Float64,2} = NetCDF.ncread(file_name,"clon_bnds")
  return clat,clon,clat_bounds,clon_bounds
end

function load_field(file_handle::NcFile,grid::Grid,variable_name::AbstractString,
                    field_type::DataType,;timestep::Int64=-1)
  variable::NcVar = file_handle[variable_name]
  values::Array{field_type,get_number_of_dimensions(grid)} = NetCDF.readvar(variable)
  if isa(grid,LatLonGrid)
    if timestep == -1
      values = permutedims(values, [2,1])
    else
      values = permutedims(values, [2,1])[:,:,timestep]
    end
  end
  return Field{field_type}(grid,values)
end

function load_3d_field(file_handle::NcFile,grid::Grid,variable_name::AbstractString,
                       field_type::DataType,layer=Int64,;timestep::Int64=-1)
  variable::NcVar = file_handle[variable_name]
  values::Array{field_type,(timestep == -1) ?
                            get_number_of_dimensions(grid)+1 :
                            get_number_of_dimensions(grid)+2 } = NetCDF.readvar(variable)
  local layer_values::Array{field_type,get_number_of_dimensions(grid)}
  if isa(grid,UnstructuredGrid)
    if timestep == -1
      layer_values = values[:,layer]
    else
      layer_values = values[:,layer,timestep]
    end
  end
  return Field{field_type}(grid,layer_values)
end

function load_array_of_fields(file_handle::NcFile,grid::Grid,variable_base_name::AbstractString,
                              field_type::DataType,number_of_fields::Int64)
  if number_of_fields == 1 && isa(grid,LatLonGrid)
    return Field{field_type}[load_field(file_handle,grid,variable_base_name,field_type)]
  else
    array_of_fields::Array{Field{field_type},1} = Array{Field{field_type}}(undef,number_of_fields)
    for i = 1:number_of_fields
      if isa(grid,LatLonGrid)
        array_of_fields[i] = load_field(file_handle,grid,variable_base_name*string(i),field_type)
      else
        array_of_fields[i] = load_3d_field(file_handle,grid,variable_base_name,field_type,i)
      end
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

function prep_dims(grid::UnstructuredGrid)
  ncells = NcDim("ncells",grid.ncells)
  dims::Array{NcDim} = NcDim[ ncells ]
  return dims
end

function prep_field(dims::Array{NcDim},variable_name::AbstractString,
                    field::Field,fields_to_write::Vector{Pair})
  local variable::NcVar
  if isa(field,LatLonField)
    variable = NcVar(variable_name,dims)
    push!(fields_to_write,Pair(variable,permutedims(get_data(field), [2,1])))
  elseif isa(field,UnstructuredField)
    variable = NcVar(variable_name,dims,atts=Dict("grid_type"=>"unstructured",
                                                  "coordinates"=>"clat clon"))
    push!(fields_to_write,Pair(variable,get_data(field)))
  end
end

function write_fields(grid::Grid,fields_to_write::Vector{Pair},filepath::AbstractString,
                      dims::Array{NcDim})
  variables::Array{NcVar} = NcVar[ pair.first for pair in fields_to_write ]
  if isa(grid,UnstructuredGrid)
    prepare_icon_file(grid,filepath,dims,variables)
  else
    NetCDF.create(filepath,variables,mode=NC_NETCDF4)
  end
  for (variable,data) in fields_to_write
    NetCDF.putvar(variable,data)
  end
end

function prepare_icon_file(grid::UnstructuredGrid,filepath::AbstractString,
                           dims::Array{NcDim},variables::Array{NcVar})
  vertices = NcDim("vertices",3)
  extended_dims::Array{NcDim} = vcat(NcDim[ vertices ], dims)
  clat = NcVar("clat",dims,atts=Dict("long_name"=>"center_latitude",
                                     "units"=>"radians",
                                     "bounds"=>"clat_bnds",
                                     "standard_name"=>"latitude"))
  clon = NcVar("clon",dims,atts=Dict("long_name"=>"center_longitude",
                                     "units"=>"radians",
                                     "bounds"=>"clon_bnds",
                                     "standard_name"=>"longitude"))
  clat_bounds = NcVar("clat_bnds",extended_dims)
  clon_bounds = NcVar("clon_bnds",extended_dims)
  extended_variables::Array{NcVar} = vcat(variables,NcVar[ clat, clon, clat_bounds, clon_bounds])
  NetCDF.create(filepath,extended_variables,mode=NC_NETCDF4)
  NetCDF.putvar(clat,grid.clat)
  NetCDF.putvar(clon,grid.clon)
  NetCDF.putvar(clat_bounds,grid.clat_bounds)
  NetCDF.putvar(clon_bounds,grid.clon_bounds)
end

function write_field(grid::Grid,variable_name::AbstractString,
                     field::Field,filepath::AbstractString)
  dims::Array{NcDim} = prep_dims(grid)
  local variable::NcVar
  if isa(grid,UnstructuredGrid)
    variable = NcVar(variable_name,dims,atts=Dict("grid_type"=>"unstructured",
                                                  "coordinates"=>"clat clon"))
    variables::Array{NcVar} = NcVar[ variable ]
    prepare_icon_file(grid,filepath,dims,variables)
  else
    variable = NcVar(variable_name,dims)
    NetCDF.create(filepath,variable,mode=NC_NETCDF4)
  end
  if isa(grid,LatLonGrid)
    NetCDF.putvar(variable,permutedims(get_data(field), [2,1]))
  else
    NetCDF.putvar(variable,get_data(field))
  end
end

function load_river_parameters(hd_para_filepath::AbstractString,grid::Grid;
                               day_length=86400.0,step_length=86400.0)
  println("Loading: " * hd_para_filepath)
  file_handle::NcFile = NetCDF.open(hd_para_filepath)
  local landsea_mask::Field{Bool}
  try
    if isa(grid,LatLonGrid)
      direction_based_river_directions = load_field(file_handle,grid,"FDIR",Float64)
      flow_directions = LatLonDirectionIndicators(round(Int64,direction_based_river_directions))
      landsea_mask =
        load_field(file_handle,grid,"FLAG",Bool)
    elseif isa(grid,UnstructuredGrid)
      index_based_river_directions = load_field(file_handle,grid,"FDIR",Float64)
      flow_directions = UnstructuredDirectionIndicators(round(Int64,index_based_river_directions))
      landsea_mask_int::Field{Int64} =
        load_field(file_handle,grid,"MASK",Int64)
      landsea_mask = equals(landsea_mask_int,1)
    end
    landsea_mask = invert(landsea_mask)
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
    return  RiverParameters(flow_directions,
                            river_reservoir_nums,
                            overland_reservoir_nums,
                            base_reservoir_nums,
                            river_retention_coefficients,
                            overland_retention_coefficients,
                            base_retention_coefficients,
                            landsea_mask,grid,day_length,
                            step_length)
  finally
    NetCDF.close(file_handle)
  end
end

function load_river_initial_values(hd_start_filepath::AbstractString,grid::Grid,
                                   river_parameters::RiverParameters)
  river_prognostic_fields::RiverPrognosticFields =
    RiverPrognosticFields(river_parameters)
  local reservoir_units_adjustment_factor
  println("Loading: " * hd_start_filepath)
  file_handle::NcFile = NetCDF.open(hd_start_filepath)
  try
    if ! isa(grid,UnstructuredGrid)
      reservoir_units_adjustment = 1.0/river_parameters.step_length
      river_prognostic_fields.river_inflow =
        load_field(file_handle,river_parameters.grid,"FINFL",Float64)
    else
      river_prognostic_fields.river_inflow = Field{Float64}(grid,0.0)
      reservoir_units_adjustment = 1.0
    end
    river_prognostic_fields.base_flow_reservoirs =
      load_array_of_fields(file_handle,river_parameters.grid,"FGMEM",Float64,
                            maximum(river_parameters.base_reservoir_nums))
    river_prognostic_fields.base_flow_reservoirs *= reservoir_units_adjustment
    river_prognostic_fields.overland_flow_reservoirs =
      load_array_of_fields(file_handle,river_parameters.grid,"FLFMEM",Float64,
                            maximum(river_parameters.overland_reservoir_nums))
    river_prognostic_fields.overland_flow_reservoirs *= reservoir_units_adjustment
    river_prognostic_fields.river_flow_reservoirs =
      load_array_of_fields(file_handle,river_parameters.grid,"FRFMEM",Float64,
                            maximum(river_parameters.river_reservoir_nums))
    river_prognostic_fields.river_flow_reservoirs *= reservoir_units_adjustment
  finally
    NetCDF.close(file_handle)
  end
  return river_prognostic_fields
end

function write_river_initial_values(hd_start_filepath::AbstractString,
                                    river_parameters::RiverParameters,
                                    river_prognostic_fields::RiverPrognosticFields)
  println("Writing: " * hd_start_filepath)
  if ! isa(river_parameters.grid,UnstructuredGrid)
    reservoir_units_adjustment = river_parameters.step_length
  else
    reservoir_units_adjustment = 1.0
  end
  river_prognostic_fields.base_flow_reservoirs *= reservoir_units_adjustment
  river_prognostic_fields.overland_flow_reservoirs *= reservoir_units_adjustment
  river_prognostic_fields.river_flow_reservoirs *= reservoir_units_adjustment
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
  write_fields(river_parameters.grid,fields_to_write,hd_start_filepath,dims)
end

function load_lake_parameters(lake_para_filepath::AbstractString,grid::Grid,hd_grid::Grid,
                              surface_model_grid::Grid)
  println("Loading: " * lake_para_filepath)
  file_handle::NcFile = NetCDF.open(lake_para_filepath)
  try
    grid_specific_lake_parameters::GridSpecificLakeParameters =
      load_grid_specific_lake_parameters(file_handle,grid,surface_model_grid)
    lake_centers::Field{Bool} =
      load_field(file_handle,grid,"lake_centers",Bool)
    connection_volume_thresholds::Field{Float64} =
      load_field(file_handle,grid,"connection_volume_thresholds",Float64)
    flood_volume_thresholds::Field{Float64} =
      load_field(file_handle,grid,"flood_volume_thresholds",Float64)
    cell_areas_on_surface_model_grid:Field{Float64} =
      load_field(file_handle,grid,"cell_areas_on_surface_model_grid",Float64)
    return LakeParameters(lake_centers,
                          connection_volume_thresholds,
                          flood_volume_thresholds,
                          cell_areas_on_surface_model_grid,
                          grid,hd_grid,
                          surface_model_grid,
                          grid_specific_lake_parameters)
  finally
    NetCDF.close(file_handle)
  end
end

function write_river_flow_field(river_parameters::RiverParameters,river_flow_field::Field{Float64};timestep::Int64=-1)
  variable_name::String = "river_flow"
  filepath::String = timestep == -1 ? "/Users/thomasriddick/Documents/data/temp/river_model_results.nc" : "/Users/thomasriddick/Documents/data/temp/river_model_results_$(timestep).nc"
  write_field(river_parameters.grid,variable_name,river_flow_field,filepath)
end

function load_grid_specific_lake_parameters(file_handle::NcFile,grid::LatLonGrid,
                                            surface_model_grid::LatLonGrid)
  flood_next_cell_lat_index::Field{Int64} =
      load_field(file_handle,grid,"flood_next_cell_lat_index",Int64)
  flood_next_cell_lon_index::Field{Int64} =
      load_field(file_handle,grid,"flood_next_cell_lon_index",Int64)
  connect_next_cell_lat_index::Field{Int64} =
      load_field(file_handle,grid,"connect_next_cell_lat_index",Int64)
  connect_next_cell_lon_index::Field{Int64} =
      load_field(file_handle,grid,"connect_next_cell_lon_index",Int64)
  corresponding_surface_cell_lat_index::Field{Int64} =
      load_field(file_handle,surface_model_grid,"corresponding_surface_cell_lat_index",Int64)
  corresponding_surface_cell_lon_index::Field{Int64} =
      load_field(file_handle,surface_model_grid,"corresponding_surface_cell_lon_index",Int64)
  add_offset(flood_next_cell_lat_index,1,Int64[-1])
  add_offset(flood_next_cell_lon_index,1,Int64[-1])
  add_offset(connect_next_cell_lat_index,1,Int64[-1])
  add_offset(connect_next_cell_lon_index,1,Int64[-1])
  add_offset(corresponding_surface_cell_lat_index,1,Int64[-1])
  add_offset(corresponding_surface_cell_lon_index,1,Int64[-1])
  return LatLonLakeParameters(flood_next_cell_lat_index,
                              flood_next_cell_lon_index,
                              connect_next_cell_lat_index,
                              connect_next_cell_lon_index,
                              corresponding_surface_cell_lat_index,
                              corresponding_surface_cell_lon_index)
end

function load_grid_specific_lake_parameters(file_handle::NcFile,grid::UnstructuredGrid,
                                            surface_model_grid::UnstructuredGrid)
  flood_next_cell_index::Field{Int64} =
      load_field(file_handle,grid,"flood_next_cell_index",Int64)
  connect_next_cell_index::Field{Int64} =
      load_field(file_handle,grid,"connect_next_cell_index",Int64)
  corresponding_surface_cell_index::Field{Int64} =
      load_field(file_handle,surface_model_grid,"corresponding_surface_cell_index",Int64)
  return UnstructuredLakeParameters(flood_next_cell_index,
                                    connect_next_cell_index,
                                    corresponding_surface_cell_index)
end

function load_lake_initial_values(lake_start_filepath::AbstractString,
                                  grid::Grid,hd_grid::Grid)
  local initial_water_to_lake_centers::Field{Float64}
  local initial_spillover_to_rivers::Field{Float64}
  println("Loading: " * lake_start_filepath)
  file_handle::NcFile = NetCDF.open(lake_start_filepath)
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
  filepath::String = timestep == -1 ? "/Users/thomasriddick/Documents/data/temp/lake_model_results.nc" : "/Users/thomasriddick/Documents/data/temp/lake_model_results_$(timestep).nc"
  write_field(lake_parameters.grid,variable_name,
              lake_fields.lake_numbers,filepath)
end

function write_lake_volumes_field(lake_parameters::LakeParameters,lake_volumes::Field{Float64})
  variable_name::String = "lake_field"
  filepath::String = "/Users/thomasriddick/Documents/data/temp/lake_model_out.nc"
  write_field(lake_parameters.grid,variable_name,lake_volumes,filepath)
end


function write_diagnostic_lake_volumes_field(lake_parameters::LakeParameters,
                                             diagnostic_lake_volumes::Field{Float64};
                                             timestep::Int64=-1)
  variable_name::String = "diagnostic_lake_volumes"
  filepath::String = timestep == -1 ?
                     "/Users/thomasriddick/Documents/data/temp/lake_model_volume_results.nc" :
                     "/Users/thomasriddick/Documents/data/temp/lake_model_volume_results_$(timestep).nc"
  write_field(lake_parameters.grid,variable_name,diagnostic_lake_volumes,filepath)
end

function load_drainage_fields(drainages_filename::AbstractString,grid::Grid;
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

function load_runoff_fields(runoffs_filename::AbstractString,grid::Grid;
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

function load_lake_evaporation_fields(lake_evaporations_filename::AbstractString,
                                      grid::LatLonGrid;first_timestep::Int64=1,
                                      last_timestep::Int64=1)
  println("Loading: " * lake_evaporations_filename)
  file_handle::NcFile = NetCDF.open(lake_evaporations_filename)
  lake_evaporations::Array{Field{Float64},1} = Field{Float64}[]
  try
    for i::Int64 = first_timestep:last_timestep
      lake_evaporations[i] = load_field(file_handle,grid,"evaporation",
                              Float64;timestep=i)
    end
  finally
    NetCDF.close(file_handle)
  end
  return lake_evaporations
end

end
