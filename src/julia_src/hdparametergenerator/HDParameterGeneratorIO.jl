module HDParameterGeneratorIO

using NetCDF: NcFile,NcVar,NC_NETCDF4

import HDParameterGenerator: load_input_data,write_hdpara_file
import HDParameterGenerator: Grid,UnstructuredGrid,LatLonGrid

function get_ncells(file_name::AbstractString)
  river_directions::Array{Float64,1} = NetCDF.ncread(file_name,"FDIR")
  return size(river_directions,1)
end

function load_field(file_handle::NcFile,grid::Grid,variable_name::AbstractString,
                    field_type::DataType)
  variable::NcVar = file_handle[variable_name]
  values::Array{field_type,get_number_of_dimensions(grid)} = NetCDF.readvar(variable)
  if isa(grid,LatLonGrid)
    values = permutedims(values, [2,1])
  end
  return values
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

function load_icosahedral_grid(grid_filepath)
  println("Loading: " * grid_filepath)
  file_handle::NcFile = NetCDF.open(grid_filepath)
  ncells::Int64 = file_handle.dim["cell"]
  clat_var::NcVar = file_handle["clat"]
  clat::Array{Float64,1} = readvar(clat_var)
  clon_var::NcVar = file_handle["clon"]
  clon::Array{Float64,1} = readvar(clon_var)
  clat_bounds_var::NcVar = file_handle["clat_bounds"]
  clat_bounds::Array{Float64,2} = readvar(clat_bounds_var)
  clon_bounds_var::NcVar = file_handle["clon_bounds"]
  clon_bounds::Array{Float64,2} = readvar(clon_bounds_var)
  return UnstructuredGrid(ncells,clat,clon,clat_bounds,clon_bounds)
end

function determine_latlon_grid_information(file_handle)
    nlat::Int64 = file_handle.dim["lat"]
    nlon::Int64 = file_handle.dim["lon"]
    lats_var::NcVar = file_handle["lat"]
    lats::Array{Float64,1} = readvar(lats_var)
    lons_var::NcVar = file_handle["lon"]
    lons::Array{Float64,1} = readvar(lons_var)
    return LatLonGrid(nlat,nlon,lats,lons)
end


function load_input_data(input_filepaths::Dict)
  local grid::Grid
  println("Loading: " * input_filepaths["river_directions_filepath"])
  river_directions_file_handle::NcFile = NetCDF.open(input_filepaths["river_directions_filepath"])
  if haskey(input_filepaths,"grid_filepath")
    grid = load_icosahedral_grid(input_filepaths["grid_filepath"])
    river_directions::Array{Int64,ndims} = load_field(river_directions_file_handle,grid,
                                                      input_filepaths["river_directions_fieldname"],
                                                      Int64)
  else
    grid = determine_latlon_grid_information(river_directions_file_handle)
    river_directions::Array{Int64,ndims} =
      round.(Int64,load_field(river_directions_file_handle,grid,
                              input_filepaths["river_directions_fieldname"],Float64)
  end
  ndims::Int64 = size(grid.grid_dimensions)

  println("Loading: " * input_filepaths["orography_variance_filepath"])
  orography_variance_file_handle::NcFile = NetCDF.open(input_filepaths["orography_variance_filepath"]
  orography_variance::Array{Float64,ndims} = load_field(orography_variance_file_handle,grid,
                          input_filepaths["orography_variance__fieldname",Float64)
  println("Loading: " * input_filepaths["innerslope_filepath"])
  innerslope_file_handle::NcFile = NetCDF.open(input_filepaths["innerslope_filepath"]
  innerslope::Array{Float64,ndims} = load_field(innerslope_file_handle,grid,
             input_filepaths["innerslope_fieldname",Float64)
  println("Loading: " * input_filepaths["orography_filepath"])
  orography_file_handle::NcFile = NetCDF.open(input_filepaths["orography_filepath"]
  orography::Array{Float64,ndims} = load_field(orography_file_handle,grid,
                                               input_filepaths["orography_fieldname",
                                               Float64)
  println("Loading: " * input_filepaths["glacier_mask_filepath"])
  glacier_mask_file_handle::NcFile = NetCDF.open(input_filepaths["glacier_mask_filepath"]
  glacier_mask::Array{Bool,ndims} = iszero.(load_field(glacier_mask_file_handle,grid,
                                               input_filepaths["glacier_mask_fieldname",
                                               Int64))
  println("Loading: " * input_filepaths["landsea_mask_filepath"])
  landsea_mask_file_handle::NcFile = NetCDF.open(input_filepaths["landsea_mask_filepath"]
  landsea_mask::Array{Bool,ndims) = iszero.(load_field(landsea_mask_file_handle,grid,
                                            input_filepaths["landsea_mask_fieldname",Float64))
  println("Loading: " * input_filepaths["cell_areas_filepath"])
  cell_areas_file_handle::NcFile = NetCDF.open(input_filepaths["cell_areas_filepath"]
  cell_areas::Array{Float64,ndims} = load_field(cell_areas_file_handle,grid,
                                                input_filepaths["cell_areas_fieldname",
                                                Float64))
  input_data = InputData(landsea_mask,glacier_mask,orography,
                         innerslope,orography_variance,
                         river_directions,cell_areas,grid)
  return input_data::InputData,grid::Grid
end

function write_hdpara_file(output_hdpara_filepath::AbstractString,
                           input_data::InputData,
                           number_of_riverflow_reservoirs{Float64},
                           riverflow_retention_coefficients{Float64},
                           number_of_overlandflow_reservoirs{Float64},
                           overlandflow_retention_coefficients{Float64},
                           number_of_baseflow_reservoirs{Float64},
                           baseflow_retention_coefficients{Float64},
                           grid::Grid)
  dims::Array{NcDim} = prep_dims(grid)
  fields_to_write::Vector{Pair} = Pair[]
  prep_field(dims,"ARF_N",
             number_of_riverflow_reservoirs,fields_to_write)
  prep_field(dims,"ARF_K",
             riverflow_retention_coefficients,fields_to_write)
  prep_field(dims,"ALF_N",
             number_of_overlandflow_reservoirs,fields_to_write)
  prep_field(dims,"ALF_K",
             overlandflow_retention_coefficients,fields_to_write)
  #prep_field(dims,"AGF_N",
  #           number_of_baseflow_reservoirs,fields_to_write)
  prep_field(dims,"AGF_K",
             baseflow_retention_coefficients,fields_to_write)
  prep_field(dims,"Area",
             input_data.cell_areas,fields_to_write)
  if isa(grid,UnstructuredGrid)
    prep_field(dims,"FDIR",
               input_data.grid_specific_input_data.next_cell_index,fields_to_write)
  else
    prep_field(dims,"FDIR",
               input_data.grid_specific_input_data.river_directions,fields_to_write)
  end
  prep_field(dims,"FLAG",
             input_data.landsea_mask,fields_to_write)
  write_fields(grid,fields_to_write,output_hdpara_filepath,dims)
end

end
