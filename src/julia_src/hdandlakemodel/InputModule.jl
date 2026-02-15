module InputModule

using NetCDF
using NetCDF: NcFile,NcVar
using GridModule: Grid, LatLonGrid, UnstructuredGrid, get_number_of_dimensions
using FieldModule: Field, LatLonField,UnstructuredField,LatLonDirectionIndicators
using FieldModule: UnstructuredDirectionIndicators
using FieldModule: round,convert,invert,add_offset,maximum,equals
using HDModule: RiverParameters,RiverPrognosticFields
using LakeModule: LakeParameters,LatLonLakeParameters, UnstructuredLakeParameters
using LakeModule: GridSpecificLakeParameters,LakeFields
using LakeModule: create_merge_indices_collections_from_array
using LakeModule: MergeAndRedirectIndicesCollection
using MergeTypesModule

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
                    field_type::DataType;timestep::Int64=-1,convert_to::Union{Nothing,DataType}=nothing)
  variable::NcVar = file_handle[variable_name]
  extra_dims = timestep == -1 ? 0 : 1
  values_orig::Array{field_type,get_number_of_dimensions(grid)+extra_dims} =
    NetCDF.readvar(variable)
  local values::Array{field_type,get_number_of_dimensions(grid)}
  if isa(grid,LatLonGrid)
    if timestep == -1
      values = permutedims(values_orig, [2,1])
    else
      values = permutedims(values_orig[:,:,timestep], [2,1])
    end
  else
    if timestep == -1
      values = values_orig
    else
      values = values_orig[:,timestep]
    end
  end
  if ! isnothing(convert_to)
	converted_values::Array{convert_to,get_number_of_dimensions(grid)} = convert(Array{convert_to},values)
  end
  return Field{isnothing(convert_to) ? field_type : convert_to}(grid,isnothing(convert_to) ? values : converted_values)
end

function load_3d_field(file_handle::NcFile,grid::Grid,variable_name::AbstractString,
                       field_type::DataType,layer::Int64;timestep::Int64=-1)
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

function load_river_parameters(hd_para_filepath::AbstractString,grid::Grid;
                               day_length=86400.0,step_length=86400.0,
                               use_bifurcated_rivers::Bool=false)
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
    if use_bifurcated_rivers
      cells_up::Array{Field{Int64}} =
        load_array_of_fields(file_handle,grid,"CELLS_UP",Int64,12)
      nsplit::Field{Int64} =
        load_field(file_handle,grid,"NSPLIT",Int64)
      return  RiverParameters(flow_directions,
                              cells_up,
                              nsplit,
                              river_reservoir_nums,
                              overland_reservoir_nums,
                              base_reservoir_nums,
                              river_retention_coefficients,
                              overland_retention_coefficients,
                              base_retention_coefficients,
                              landsea_mask,grid,day_length,
                              step_length)
    else
      return  RiverParameters(flow_directions,
                              river_reservoir_nums,
                              overland_reservoir_nums,
                              base_reservoir_nums,
                              river_retention_coefficients,
                              overland_retention_coefficients,
                              base_retention_coefficients,
                              landsea_mask,grid,day_length,
                              step_length)
    end
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
    connect_merge_and_redirect_indices_index::Field{Int64} =
      load_field(file_handle,grid,"connect_merge_and_redirect_indices_index",Int64)
    add_offset(connect_merge_and_redirect_indices_index,1,Int64[])
    flood_merge_and_redirect_indices_index::Field{Int64} =
      load_field(file_handle,grid,"flood_merge_and_redirect_indices_index",Int64)
    add_offset(flood_merge_and_redirect_indices_index,1,Int64[])
    cell_areas_on_surface_model_grid::Field{Float64} =
      load_field(file_handle,surface_model_grid,"cell_areas_on_surface_model_grid",Float64)
    raw_heights::Field{Float64} =
      load_field(file_handle,grid,"raw_heights",Float64)
    corrected_heights::Field{Float64} =
      load_field(file_handle,grid,"corrected_heights",Float64)
    flood_heights::Field{Float64} =
      load_field(file_handle,grid,"flood_heights",Float64)
    connection_heights::Field{Float64} =
      load_field(file_handle,grid,"connection_heights",Float64)
    # variable::NcVar = file_handle["connect_merges_and_redirects"]
    # connect_merges_and_redirect_array::Array{Int64} = NetCDF.readvar(variable)
    # connect_merge_and_redirect_indices_collections::Vector{MergeAndRedirectIndicesCollection} =
    #   create_merge_indices_collections_from_array(connect_merges_and_redirect_array)
    # add_offset(connect_merge_and_redirect_indices_collections,1)
    cell_areas::Field{Float64} =
      load_field(file_handle,grid,"cell_areas",Float64)
    connect_merge_and_redirect_indices_collections::Vector{MergeAndRedirectIndicesCollection} =
      Vector{MergeAndRedirectIndicesCollection}()
    variable = file_handle["flood_merges_and_redirects"]
    flood_merges_and_redirect_array::Array{Int64} = NetCDF.readvar(variable)
    flood_merge_and_redirect_indices_collections::Vector{MergeAndRedirectIndicesCollection} =
      create_merge_indices_collections_from_array(flood_merges_and_redirect_array)
    add_offset(flood_merge_and_redirect_indices_collections,1)
    return LakeParameters(lake_centers,
                          connection_volume_thresholds,
                          flood_volume_thresholds,
                          connect_merge_and_redirect_indices_index,
                          flood_merge_and_redirect_indices_index,
                          connect_merge_and_redirect_indices_collections,
                          flood_merge_and_redirect_indices_collections,
                          cell_areas_on_surface_model_grid,
                          cell_areas,
                          raw_heights,
                          corrected_heights,
                          flood_heights,
                          connection_heights,
                          grid,hd_grid,
                          surface_model_grid,
                          grid_specific_lake_parameters)
  finally
    NetCDF.close(file_handle)
  end
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
      load_field(file_handle,grid,"corresponding_surface_cell_lat_index",Int64)
  corresponding_surface_cell_lon_index::Field{Int64} =
      load_field(file_handle,grid,"corresponding_surface_cell_lon_index",Int64)
  add_offset(flood_next_cell_lat_index,1,Int64[-1])
  add_offset(flood_next_cell_lon_index,1,Int64[-1])
  add_offset(connect_next_cell_lat_index,1,Int64[-1])
  add_offset(connect_next_cell_lon_index,1,Int64[-1])
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

function load_drainage_fields(drainages_filename::AbstractString,grid::Grid;
                              first_timestep::Int64=1,last_timestep::Int64=1)
  println("Loading: " * drainages_filename)
  file_handle::NcFile = NetCDF.open(drainages_filename)
  drainages::Array{Field{Float64},1} = Field{Float64}[]
  try
    for i::Int64 = first_timestep:last_timestep
      push!(drainages,load_field(file_handle,grid,"drainage",
                                 Float32 ;timestep=i,convert_to=Float64))
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
      push!(runoffs,load_field(file_handle,grid,"runoff",
                               Float32 ;timestep=i,convert_to=Float64))
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
      push!(lake_evaporations,load_field(file_handle,grid,"lake_height_change",
                                         Float64;timestep=i))
    end
  finally
    NetCDF.close(file_handle)
  end
  return lake_evaporations
end

end
