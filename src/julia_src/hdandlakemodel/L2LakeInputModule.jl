module L2LakeInputModule

using NetCDF
using FieldModule: Field,equals
using GridModule: Grid, LatLonGrid
using InputModule: load_field
using L2LakeModelDefsModule: LakeModelParameters,LakeModelSettings
using L2LakeModelGridSpecificDefsModule: LatLonLakeModelParameters

function load_lake_parameters(lake_para_filepath::AbstractString,
                              cell_areas_on_surface_model_grid::Field{Float64},
                              load_binary_mask::Bool,
                              lake_grid::Grid,hd_grid::Grid,surface_model_grid::Grid)
  println("Loading: " * lake_para_filepath)
  file_handle::NcFile = NetCDF.open(lake_para_filepath)
  try
    grid_specific_lake_model_parameters::LatLonLakeModelParameters =
      load_grid_specific_lake_parameters(file_handle,lake_grid)
    number_of_lakes::Int64 = NetCDF.readvar(file_handle["number_of_lakes"])[1]
    is_lake_int::Field{Int64} =
        load_field(file_handle,lake_grid,"lake_mask",Int64)
    is_lake::Field{Bool} = equals(is_lake_int,1)
    local binary_lake_mask::Field{Bool}
    if load_binary_mask
      binary_lake_mask_int::Field{Int64} =
        load_field(file_handle,surface_model_grid,"binary_lake_mask",Int64)
      binary_lake_mask = equals(binary_lake_mask_int,1)
    else
      binary_lake_mask = Field{Bool}(surface_model_grid,false)
    end
    lake_model_parameters::LakeModelParameters =
      LakeModelParameters(grid_specific_lake_model_parameters,
                          lake_grid,
                          hd_grid,
                          surface_model_grid,
                          cell_areas_on_surface_model_grid,
                          LakeModelSettings(),
                          number_of_lakes,
                          is_lake,
                          binary_lake_mask)
    variable = file_handle["lakes_as_array"]
    lake_parameters_as_array::Array{Int64} = NetCDF.readvar(variable)
    return lake_model_parameters,lake_parameters_as_array
  finally
    NetCDF.close(file_handle)
  end
end

function load_grid_specific_lake_parameters(file_handle::NcFile,lake_grid::LatLonGrid)
  corresponding_surface_cell_lat_index::Field{Int64} =
      load_field(file_handle,lake_grid,"corresponding_surface_cell_lat_index",Int64)
  corresponding_surface_cell_lon_index::Field{Int64} =
      load_field(file_handle,lake_grid,"corresponding_surface_cell_lon_index",Int64)
  return LatLonLakeModelParameters(corresponding_surface_cell_lat_index,
                                   corresponding_surface_cell_lon_index)
end

end
