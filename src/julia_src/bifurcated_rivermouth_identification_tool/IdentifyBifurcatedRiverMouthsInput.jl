module IdentifyBifurcatedRiverMouthsInput

using TOML
using NetCDF: NcFile,NcVar,NC_NETCDF4
using IdentifyBifurcatedRiverMouths: RiverDelta, Cells
using IdentifyExistingRiverMouths: Area

function load_river_deltas_from_file(filename::String)
  river_deltas_raw::Dict{String,Any} = TOML.parsefile(filename)
  river_deltas::Array{RiverDelta} = RiverDelta[]
  for (name,river_delta_raw) in river_deltas_raw
    push!(river_deltas,RiverDelta(name,river_delta_raw))
  end
end

function load_river_deltas_from_string(river_deltas_input_string::String)
  river_deltas_raw::Dict{String,Any} = TOML.parse(river_deltas_input_string)
  river_deltas::Array{RiverDelta} = RiverDelta[]
  for (name,river_delta_raw) in river_deltas_raw
    push!(river_deltas,RiverDelta(name,river_delta_raw))
  end
  return river_deltas
end

function load_search_areas_from_file(filename::String)
  search_areas_raw::Dict{String,Any} = TOML.parsefile(filename)
  search_areas::Dict{String,Area} = Dict{String,Area}()
  for (name,area) in search_areas_raw
    search_areas[name] = Area(area[0],area[1],area[2],area[3])
  end
  return search_areas
end

function load_icosahedral_field(file_handle::NcFile,var_name::String,ndims::Int64,field_type::DataType)
  variable::NcVar = file_handle[var_name]
  values::Array{field_type,ndims} = NetCDF.readvar(variable)
  return values
end

function load_icosahedral_grid(grid_filepath::String)
  println("Loading: " * grid_filepath)
  file_handle::NcFile = NetCDF.open(grid_filepath)
  cell_indices_int::Array{Int64} = load_icosahedral_field(file_handle,"cell_index",1,Int64)
  cell_neighbors_int::Array{Int64} = load_icosahedral_field(file_handle,"neighbor_cell_index",2,Int64)
  lats::Array{Float64} = load_icosahedral_field(file_handle,"lat_cell_centre",1,Float64)
  lons::Array{Float64} = load_icosahedral_field(file_handle,"lon_cell_centre",1,Float64)
  lat_vertices::Array{Float64} = load_icosahedral_field(file_handle,"clat_vertices",2,Float64)
  lon_vertices::Array{Float64} = load_icosahedral_field(file_handle,"clon_vertices",2,Float64)
  cell_indices::Array{CartesianIndices} =
      CartesianIndex[ CartesianIndex(cell_indices_int[i]) for i=1:length(cell_indices_int)]
  cell_neighbors::Array{Array{CartesianIndices}} =
      CartesianIndex[ CartesianIndex(cell_neighbors_int[i,j]) for i=1:length(cell_neighbors_int),j=1:3 ]
  cell_coords::@NamedTuple{lats::Array{Float64},lons::Array{Float64}}
  cell_vertices::Array{@NamedTuple{lat::Array{Float64},lon::Array{Float64}}}
  return Cells(cell_indices,
               cell_neighbors,
               cell_coords,
               cell_vertices)
end

function load_landsea_mask(lsmask_filepath::String,lsmask_fieldname::String)
  println("Loading: " * lsmask_filepath)
  file_handle::NcFile = NetCDF.open(lsmask_filepath)
  lsmask_int::Array{Int64} = load_icosahedral_field(file_handle,lsmask_fieldname,1,Int64)
  lsmask::Array{Bool} = isone.(lsmask_int)
  return lsmask
end

function load_accumulated_flow(accumulated_flow_filepath::String,
                               accumulated_flow_fieldname::String)
  println("Loading: " * accumulated_flow_filepath)
  file_handle::NcFile = NetCDF.open(accumulated_flow_filepath)
  accumulated_flow::Array{Int64} =
    load_icosahedral_field(file_handle,acumulated_flow_fieldname,1,Int64)
  return accumulated_flow
end

end
