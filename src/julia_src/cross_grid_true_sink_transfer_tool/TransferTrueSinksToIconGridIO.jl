module TransferTrueSinksToIconGridIO

using NetCDF
using NetCDF: NcFile,NcVar,NC_NETCDF4

function load_icosahedral_field(file_handle::NcFile,var_name::String,ndims::Int64,field_type::DataType)
  variable::NcVar = file_handle[var_name]
  values::Array{field_type,ndims} = NetCDF.readvar(variable)
  return values
end

function load_icon_grid_parameters(grid_filepath::String)
  println("Loading: " * grid_filepath)
  file_handle::NcFile = NetCDF.open(grid_filepath)
  cell_indices_int::Array{Int64} = load_icosahedral_field(file_handle,"cell_index",1,Int64)
  cell_indices::Array{CartesianIndex} =
      CartesianIndex[CartesianIndex(cell_indices_int[i]) for i=1:length(cell_indices_int)]
  clats::Array{Float64} = load_icosahedral_field(file_handle,"lat_cell_centre",1,Float64)
  clons::Array{Float64} = load_icosahedral_field(file_handle,"lon_cell_centre",1,Float64)
  return cell_indices,clats,clons
end

function load_latlon_true_sinks(truesinks_filepath::String,truesinks_fieldname::String)
  println("Loading: " * truesinks_filepath)
  file_handle::NcFile = NetCDF.open(truesinks_filepath)
  variable::NcVar = file_handle[truesinks_fieldname]
  true_sinks_int::Array{Int64,2} = permutedims(convert.(Int64,NetCDF.readvar(variable)),[2,1])
  true_sinks::Array{Bool} = .!iszero.(true_sinks_int)
  return true_sinks
end

function load_latlon_grid_parameters(truesinks_filepath::String)
  file_handle::NcFile = NetCDF.open(truesinks_filepath)
  variable::NcVar = file_handle["lat"]
  clats = NetCDF.readvar(variable)
  variable = file_handle["lon"]
  clons = NetCDF.readvar(variable)
  return clats,clons
end

function write_icon_grid_true_sinks(icon_grid_true_sinks,
                                    output_icon_grid_truesinks_filepath)
  println("Writing to: " * output_icon_grid_truesinks_filepath)
  dims::Array{NcDim} = NcDim[NcDim("cell",size(icon_grid_true_sinks,1))]
  variable::NcVar = NcVar("true_sinks",dims)
  NetCDF.create(output_icon_grid_truesinks_filepath,
                variable,mode=NC_NETCDF4)
  NetCDF.putvar(variable,Int64.(icon_grid_true_sinks))
end

end
