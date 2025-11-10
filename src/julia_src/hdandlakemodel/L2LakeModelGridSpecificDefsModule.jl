module L2LakeModelGridSpecificDefsModule

using FieldModule: Field

abstract type GridSpecificLakeModelParameters end

struct LatLonLakeModelParameters <: GridSpecificLakeModelParameters
  corresponding_surface_cell_lat_index::Field{Int64}
  corresponding_surface_cell_lon_index::Field{Int64}
end

struct UnstructuredLakeModelParameters <: GridSpecificLakeModelParameters
  corresponding_surface_cell_index::Field{Int64}
end

function get_corresponding_surface_model_grid_cell(::CartesianIndex,
                                                   ::GridSpecificLakeModelParameters)
 error()
end

function get_corresponding_surface_model_grid_cell(coords::CartesianIndex,
                                                   grid_specific_lake_model_parameters::LatLonLakeModelParameters)
  return CartesianIndex(grid_specific_lake_model_parameters.corresponding_surface_cell_lat_index(coords),
                        grid_specific_lake_model_parameters.corresponding_surface_cell_lon_index(coords))
end

function get_corresponding_surface_model_grid_cell(coords::CartesianIndex,
                                                   grid_specific_lake_model_parameters::UnstructuredLakeModelParameters)
  return CartesianIndex(grid_specific_lake_model_parameters.corresponding_surface_cell_index(coords))
end

end
