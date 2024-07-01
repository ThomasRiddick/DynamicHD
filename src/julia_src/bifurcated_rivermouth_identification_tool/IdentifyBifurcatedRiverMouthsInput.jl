using NetCDF: NcFile,NcVar,NC_NETCDF4

function load_icosahedral_grid(grid_filepath)
  println("Loading: " * grid_filepath)
  file_handle::NcFile = NetCDF.open(grid_filepath)
  cell_indices_int::Array{Int64} =
  cell_neighbors_int::Array{Int64} =

  cell_indices::Array{CartesianIndices}
  cell_neighbors::Array{Array{CartesianIndices}}
  cell_coords::@NamedTuple{lats::Array{Float64},lons::Array{Float64}}
  cell_vertices::Array{@NamedTuple{lat::Array{Float64},lon::Array{Float64}}}
  return Cells(cell_indices,
               cell_neighbors,
               cell_coords,
               cell_vertices)
end
