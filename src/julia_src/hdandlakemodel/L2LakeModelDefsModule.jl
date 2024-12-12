module L2LakeModelDefsModule

using HierarchicalStateMachineModule: State
using FieldModule: Field,set!
using GridModule: Grid,for_all,for_all_fine_cells_in_coarse_cell
using SplittableRootedTree: RootedTreeForest

abstract type Lake <: State end

abstract type GridSpecificLakeModelParameters end

struct LatLonLakeModelParameters <: GridSpecificLakeModelParameters
  corresponding_surface_cell_lat_index::Field{Int64}
  corresponding_surface_cell_lon_index::Field{Int64}
end

struct UnstructuredLakeModelParameters <: GridSpecificLakeModelParameters
  corresponding_surface_cell_index::Field{Int64}
end

struct LakeModelSettings
  lake_retention_constant::Float64
  minimum_lake_volume_threshold::Float64
  function LakeModelSettings()
    lake_retention_constant::Float64 = 0.1
    minimum_lake_volume_threshold::Float64 = 0.0000001
    return new(lake_retention_constant,
               minimum_lake_volume_threshold)
  end
end

struct LakeModelParameters
  grid_specific_lake_model_parameters::GridSpecificLakeModelParameters
  lake_model_grid::Grid
  hd_model_grid::Grid
  surface_model_grid::Grid
  lake_model_settings::LakeModelSettings
  number_of_lakes::Int64
  basins::Vector{Vector{Int64}}
  basin_numbers::Field{Int64}
  cells_with_lakes::Vector{CartesianIndex}
  cell_areas_on_surface_model_grid::Field{Float64}
  lake_centers::Field{Bool}
  number_fine_grid_cells::Field{Int64}
  surface_cell_to_fine_cell_maps::Vector{Vector{CartesianIndex}}
  surface_cell_to_fine_cell_map_numbers::Field{Int64}
  function LakeModelParameters(grid_specific_lake_model_parameters::GridSpecificLakeModelParameters,
                               lake_model_grid::Grid,
                               hd_model_grid::Grid,
                               surface_model_grid::Grid,
                               cell_areas_on_surface_model_grid::Field{Float64},
                               lake_model_settings::LakeModelSettings,
                               lake_centers_list::Vector{CartesianIndex},
                               is_lake::Field{Bool})
    number_fine_grid_cells::Field{Int64} =
      Field{Int64}(lake_model_grid,0)
    surface_cell_to_fine_cell_maps::Vector{Vector{CartesianIndex}} = CartesianIndex[]
    surface_cell_to_fine_cell_map_numbers = Field{Int64}(surface_model_grid,0)
    for_all(lake_model_grid;use_cartesian_index=true) do coords::CartesianIndex
      surface_model_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(coords,grid_specific_lake_model_parameters)
      set!(number_fine_grid_cells,
           surface_model_coords,
           number_fine_grid_cells(surface_model_coords) + 1)
      if is_lake(coords)
        surface_cell_to_fine_cell_map_number::Int64 =
          surface_cell_to_fine_cell_map_numbers(surface_model_coords)
        if  surface_cell_to_fine_cell_map_number == 0
          surface_cell_to_fine_cell_map::Vector{CartesianIndex} = CartesianIndex[coords]
          push!(surface_cell_to_fine_cell_maps,surface_cell_to_fine_cell_map)
          set!(surface_cell_to_fine_cell_map_numbers,surface_model_coords,
               length(surface_cell_to_fine_cell_maps))
        else
          surface_cell_to_fine_cell_map =
            surface_cell_to_fine_cell_maps[surface_cell_to_fine_cell_map_number]
          push!(surface_cell_to_fine_cell_map,coords)
        end
      end
    end
    lake_centers::Field{Bool} = Field{Bool}(lake_model_grid,false)
    for lake_center::CartesianIndex in lake_centers_list
      set!(lake_centers,lake_center,true)
    end
    number_of_lakes::Int64 = length(lake_centers_list)
    basins::Vector{Vector{Int64}} = Vector{Int64}[]
    basin_numbers::Field{Int64} = Field{Int64}(hd_model_grid,0)
    cells_with_lakes::Vector{CartesianIndex} = CartesianIndex[]
    for_all(hd_model_grid;
            use_cartesian_index=true) do coords::CartesianIndex
      contains_lake::Bool = false
      for_all_fine_cells_in_coarse_cell(lake_model_grid,
                                        hd_model_grid,
                                        coords) do fine_coords::CartesianIndex
        if lake_centers(fine_coords)
          contains_lake = true
        end
      end
      if contains_lake
        push!(cells_with_lakes,coords)
      end
    end
    new(grid_specific_lake_model_parameters,
        lake_model_grid,
        hd_model_grid,
        surface_model_grid,
        lake_model_settings,
        number_of_lakes,
        basins,
        basin_numbers,
        cells_with_lakes,
        cell_areas_on_surface_model_grid,
        lake_centers,
        number_fine_grid_cells,
        surface_cell_to_fine_cell_maps,
        surface_cell_to_fine_cell_map_numbers)
  end
end

struct LakeModelPrognostics
  lakes::Vector{Lake}
  lake_numbers::Field{Int64}
  lake_cell_count::Field{Int64}
  effective_volume_per_cell_on_surface_grid::Field{Float64}
  effective_lake_height_on_surface_grid_to_lakes::Field{Float64}
  water_to_lakes::Field{Float64}
  water_to_hd::Field{Float64}
  lake_water_from_ocean::Field{Float64}
  evaporation_from_lakes::Array{Float64,1}
  evaporation_applied::BitArray{1}
  set_forest::RootedTreeForest
  function LakeModelPrognostics(lake_model_parameters::LakeModelParameters)
    lakes::Vector{Lake} = Lake[]
      Field{Bool}(lake_model_parameters.lake_model_grid,false)
    lake_numbers = Field{Int64}(lake_model_parameters.lake_model_grid,0)
    effective_volume_per_cell_on_surface_grid::Field{Float64} =
      Field{Float64}(lake_model_parameters.surface_model_grid,0.0)
    effective_lake_height_on_surface_grid_to_lakes::Field{Float64} =
      Field{Float64}(lake_model_parameters.surface_model_grid,0.0)
    water_to_lakes::Field{Float64} =
      Field{Float64}(lake_model_parameters.hd_model_grid,0.0)
    water_to_hd::Field{Float64} =
      Field{Float64}(lake_model_parameters.hd_model_grid,0.0)
    lake_water_from_ocean::Field{Float64} =
      Field{Float64}(lake_model_parameters.hd_model_grid,0.0)
    lake_cell_count::Field{Int64} =
      Field{Int64}(lake_model_parameters.surface_model_grid,0)
    evaporation_from_lakes::Array{Float64,1} =
      zeros(Float64,lake_model_parameters.number_of_lakes)
    evaporation_applied = falses(lake_model_parameters.number_of_lakes)
    set_forest::RootedTreeForest = RootedTreeForest()
    new(lakes,lake_numbers,lake_cell_count,
        effective_volume_per_cell_on_surface_grid,
        effective_lake_height_on_surface_grid_to_lakes,
        water_to_lakes,water_to_hd,lake_water_from_ocean,
        evaporation_from_lakes,evaporation_applied,
        set_forest)
  end
end

mutable struct LakeModelDiagnostics
  total_lake_volume::Float64
end

LakeModelDiagnostics() = LakeModelDiagnostics(0.0)

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
