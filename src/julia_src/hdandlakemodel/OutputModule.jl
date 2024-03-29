module OutputModule
using NetCDF
using NetCDF: NcFile,NcVar,NC_NETCDF4
using GridModule: Grid, LatLonGrid, UnstructuredGrid
using FieldModule: Field, LatLonField,UnstructuredField,get_data
using SharedArrays: sdata, SharedArray

function prep_array_of_fields(dims::Array{NcDim},
                              variable_base_name::AbstractString,
                              array_of_fields::Array{T,1},
                              data_type::DataType,
                              data_to_write::Vector{Pair};
                              no_grid_info::Bool=false) where {T <: Field}
  number_of_fields::Int64 = length(array_of_fields)
  if number_of_fields == 1 && length(dims) == 1
    return prep_field(dims,variable_base_name,array_of_fields[1],data_to_write)
  else
    if isa( array_of_fields[1],UnstructuredField)
      variable = no_grid_info ?
                 NcVar(variable_base_name,dims) :
                 NcVar(variable_base_name,dims,
                       atts=Dict("grid_type"=>"unstructured",
                                 "coordinates"=>"clat clon"))
      local combined_array::Array{data_type,2}
      if isa(get_data(array_of_fields[1]),SharedArray)
        combined_array = reduce(hcat,map(x -> sdata(get_data(x)),array_of_fields))
      else
        combined_array = reduce(hcat,map(x -> get_data(x),array_of_fields))
      end
      push!(data_to_write,Pair(variable,combined_array))
    else
      for i = 1:number_of_fields
        prep_field(dims,variable_base_name*string(i),
                   array_of_fields[i],data_to_write)
      end
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
  ncells::NcDim = NcDim("cell",grid.ncells)
  dims::Array{NcDim} = NcDim[ ncells ]
  return dims
end

function prep_field(dims::Array{NcDim},variable_name::AbstractString,
                    field::Field,data_to_write::Vector{Pair})
  local variable::NcVar
  if isa(field,LatLonField)
    variable = NcVar(variable_name,dims)
    if isa(get_data(field),SharedArray)
      push!(data_to_write,Pair(variable,permutedims(sdata(get_data(field)), [2,1])))
    else
      push!(data_to_write,Pair(variable,permutedims(get_data(field), [2,1])))
    end
  elseif isa(field,UnstructuredField)
    variable = NcVar(variable_name,dims,atts=Dict("grid_type"=>"unstructured",
                                                  "coordinates"=>"clat clon"))
    if isa(get_data(field),SharedArray)
      push!(data_to_write,Pair(variable,sdata(get_data(field))))
    else
      push!(data_to_write,Pair(variable,get_data(field)))
    end
  end
end

function write_fields(grid::Grid,data_to_write::Vector{Pair},filepath::AbstractString,
                      dims::Array{NcDim};no_grid_info::Bool=false)
  variables::Array{NcVar} = NcVar[ pair.first for pair in data_to_write ]
  if isa(grid,UnstructuredGrid)
    prepare_icon_file(grid,filepath,dims,variables,
          no_grid_info=no_grid_info)
  else
    NetCDF.create(filepath,variables,mode=NC_NETCDF4)
  end
  for (variable,data) in data_to_write
    NetCDF.putvar(variable,data)
  end
end

function prepare_icon_file(grid::UnstructuredGrid,filepath::AbstractString,
                           dims::Array{NcDim},variables::Array{NcVar};
                           no_grid_info::Bool=false)
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
  extended_variables::Array{NcVar} = no_grid_info ? variables :
     vcat(variables,NcVar[ clat, clon, clat_bounds, clon_bounds])
  NetCDF.create(filepath,extended_variables,mode=NC_NETCDF4)
  if ! no_grid_info
    NetCDF.putvar(clat,grid.clat)
    NetCDF.putvar(clon,grid.clon)
    NetCDF.putvar(clat_bounds,grid.clat_bounds)
    NetCDF.putvar(clon_bounds,grid.clon_bounds)
  end
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
    NetCDF.putvar(variable,permutedims(sdata(get_data(field)), [2,1]))
  else
    NetCDF.putvar(variable,sdata(get_data(field)))
  end
end

function write_river_flow_field(grid::Grid,
                                river_flow_field::Field{Float64},
                                filepath::AbstractString)
  variable_name::String = "river_flow"
  write_field(grid,variable_name,river_flow_field,filepath)
end

function write_river_initial_values(hd_start_filepath::AbstractString,
                                    grid::Grid,
                                    step_length::Float64,
                                    river_inflow::Field{Float64},
                                    base_flow_reservoirs::Array{Field{Float64},1},
                                    overland_flow_reservoirs::Array{Field{Float64},1},
                                    river_flow_reservoirs::Array{Field{Float64},1})
  println("Writing: " * hd_start_filepath)
  if ! isa(grid,UnstructuredGrid)
    reservoir_units_adjustment = step_length
  else
    reservoir_units_adjustment = 1.0
  end
  base_flow_reservoirs *= reservoir_units_adjustment
  overland_flow_reservoirs *= reservoir_units_adjustment
  river_flow_reservoirs *= reservoir_units_adjustment
  data_to_write::Vector{Pair} = Pair[]
  dims::Array{NcDim,1} = prep_dims(grid)
  if ! isa(grid,UnstructuredGrid)
    prep_field(dims,"FINFL",
               river_inflow,
               Float64,
               data_to_write)
  else
    river_flow_reservoirs[1] += river_inflow
  end
  local dims_fgmem::Array{NcDim,1}
  local dims_flfmem::Array{NcDim,1}
  local dims_frfmem::Array{NcDim,1}
  if  isa(grid,UnstructuredGrid)
    dims_fgmem = vcat(dims,NcDim[NcDim("bresnum",1)])
    dims_flfmem = vcat(dims,NcDim[NcDim("oresnum",1)])
    dims_frfmem = vcat(dims,NcDim[NcDim("rresnum",5)])
  else
    dims_fgmem = dims
    dims_flfmem = dims
    dims_frfmem = dims
  end
  prep_array_of_fields(dims_fgmem,"FGMEM",
                       base_flow_reservoirs,
                       Float64,
                       data_to_write,
                       no_grid_info=true)
  prep_array_of_fields(dims_flfmem,"FLFMEM",
                       overland_flow_reservoirs,
                       Float64,
                       data_to_write,
                       no_grid_info=true)
  prep_array_of_fields(dims_frfmem,"FRFMEM",
                       river_flow_reservoirs,
                       Float64,
                       data_to_write,
                       no_grid_info=true)
  write_fields(grid,data_to_write,hd_start_filepath,dims,no_grid_info=true)
end

function write_lake_numbers_field(grid::Grid,lake_numbers::Field{Float64};
                                  timestep::Int64=-1)
  variable_name::String = "lake_field"
  filepath::String = timestep == -1 ? "/Users/thomasriddick/Documents/data/temp/lake_model_results.nc" : "/Users/thomasriddick/Documents/data/temp/lake_model_results_$(timestep).nc"
  write_field(grid,variable_name,
              lake_numbers,filepath)
end

function write_lake_volumes_field(grid::Grid,lake_volumes::Field{Float64})
  variable_name::String = "lake_field"
  filepath::String = "/Users/thomasriddick/Documents/data/temp/lake_model_out.nc"
  write_field(grid,variable_name,lake_volumes,filepath)
end


function write_diagnostic_lake_volumes_field(grid::Grid,
                                             diagnostic_lake_volumes::Field{Float64};
                                             timestep::Int64=-1)
  variable_name::String = "diagnostic_lake_volumes"
  filepath::String = timestep == -1 ?
                     "/Users/thomasriddick/Documents/data/temp/lake_model_volume_results.nc" :
                     "/Users/thomasriddick/Documents/data/temp/lake_model_volume_results_$(timestep).nc"
  write_field(grid,variable_name,diagnostic_lake_volumes,filepath)
end

end
