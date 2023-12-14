module LakeModule

using HierarchicalStateMachineModule: State, Event
using UserExceptionModule: UserError
using LakeModelParametersModule: LakeModelParameters
using MergeTypesModule
using HDModule: RiverParameters, PrognosticFields, RiverPrognosticFields, RiverDiagnosticFields
using HDModule: RiverDiagnosticOutputFields, PrintResults, PrintSection
using HDModule: print_river_results, get_river_parameters, get_river_fields
using CoordsModule: Coords,LatLonCoords,LatLonSectionCoords,is_lake
using CoordsModule: Generic1DCoords
using GridModule: Grid, LatLonGrid, for_all, for_all_fine_cells_in_coarse_cell, for_all_with_line_breaks
using GridModule: for_section_with_line_breaks,find_coarse_cell_containing_fine_cell
using FieldModule: Field, set!,elementwise_divide, elementwise_multiple
using SplittableRootedTree: RootedTreeForest,make_new_link,split_set,add_set,find_root
using SplittableRootedTree: get_all_node_labels_of_set
import HDModule: water_to_lakes,water_from_lakes
import HierarchicalStateMachineModule: handle_event
import FieldModule: add_offset
import Base.show, Base.iterate, Base.==

const debug = false

abstract type Lake <: State end

struct RunLakes <: Event end

struct AddWater <: Event
 inflow::Float64
end

struct RemoveWater <: Event
  outflow::Float64
end

struct StoreWater <: Event
 inflow::Float64
end

struct AcceptMerge <: Event
  redirect_coords::Coords
  primary_lake_number::Int64
end

struct AcceptSplit <: Event
  primary_lake_number::Int64
end

struct DrainExcessWater <: Event end

struct ReleaseNegativeWater <: Event end

struct WriteLakeNumbers <: Event
  timestep::Int64
end

struct WriteDiagnosticLakeVolumes <: Event
  timestep::Int64
end

struct WriteLakeVolumes <: Event end

struct SetupLakes <: Event
  initial_water_to_lake_centers::Field{Float64}
end

struct DistributeSpillover <: Event
  initial_spillover_to_rivers::Field{Float64}
end

struct CalculateEffectiveLakeVolumePerCell <: Event
  consider_secondary_lake::Bool
  cell_list::Vector{Coords}
  function CalculateEffectiveLakeVolumePerCell(cell_list::Vector{Coords})
    return new(true,cell_list)
  end
  function CalculateEffectiveLakeVolumePerCell()
    return new(false,Coords[])
  end
end

struct CalculateTrueLakeDepths <: Event
end

struct CheckWaterBudget <: Event
  total_initial_water_volume::Float64
end

CheckWaterBudget() = CheckWaterBudget(0.0)

struct SetLakeEvaporation <: Event
  lake_evaporation::Field{Float64}
end

struct SetRealisticLakeEvaporation <: Event
  height_of_water_evaporated::Field{Float64}
end

struct PrintSelectedLakes <: Event
  lakes_to_print::Array{Int64,1}
end

abstract type MergeAndRedirectIndices end

mutable struct LatLonMergeAndRedirectIndices <: MergeAndRedirectIndices
  is_primary_merge::Bool
  merged::Bool
  local_redirect::Bool
  merge_target_lat_index::Int64
  merge_target_lon_index::Int64
  redirect_lat_index::Int64
  redirect_lon_index::Int64
  function LatLonMergeAndRedirectIndices(is_primary_merge::Bool,
                                         local_redirect::Bool,
                                         merge_target_lat_index::Int64,
                                         merge_target_lon_index::Int64,
                                         redirect_lat_index::Int64,
                                         redirect_lon_index::Int64)
    return(new(is_primary_merge,
               false,
               local_redirect,
               merge_target_lat_index,
               merge_target_lon_index,
               redirect_lat_index,
               redirect_lon_index))
  end
end

mutable struct UnstructuredMergeAndRedirectIndices <: MergeAndRedirectIndices
  is_primary_merge::Bool
  merged::Bool
  local_redirect::Bool
  merge_target_cell_index::Int64
  redirect_cell_index::Int64
end

function ==(lhs::MergeAndRedirectIndices,rhs::MergeAndRedirectIndices)
  throw(UserError())
end

function ==(lhs::LatLonMergeAndRedirectIndices,rhs::LatLonMergeAndRedirectIndices)
  return ((lhs.is_primary_merge == rhs.is_primary_merge) &&
          (lhs.local_redirect == rhs.local_redirect) &&
          (lhs.merge_target_lat_index == rhs.merge_target_lat_index) &&
          (lhs.merge_target_lon_index == rhs.merge_target_lon_index) &&
          (lhs.redirect_lat_index == rhs.redirect_lat_index) &&
          (lhs.redirect_lon_index == rhs.redirect_lon_index))
end

function add_offset(merge_indices::LatLonMergeAndRedirectIndices,
                    offset::Int64)
  merge_indices.merge_target_lat_index += offset
  merge_indices.merge_target_lon_index += offset
  merge_indices.redirect_lat_index += offset
  merge_indices.redirect_lon_index += offset
end

function reset(merge_indices::LatLonMergeAndRedirectIndices)
  merge_indices.merged = false
end

function read_merge_and_redirect_indices_from_array(is_primary_merge::Bool,
                                                    array_in::Array{Int64,1})
  return LatLonMergeAndRedirectIndices(is_primary_merge,
                                       (array_in[2]==1),
                                       array_in[3],
                                       array_in[4],
                                       array_in[5],
                                       array_in[6])
end

struct MergeAndRedirectIndicesCollection
  primary_merge::Bool
  secondary_merge::Bool
  primary_merge_and_redirect_indices::Vector{MergeAndRedirectIndices}
  primary_merge_and_redirect_indices_count::Int64
  secondary_merge_and_redirect_indices::Union{MergeAndRedirectIndices,Nothing}
  function MergeAndRedirectIndicesCollection(primary_merge_and_redirect_indices
                                             ::Vector{MergeAndRedirectIndices},
                                             secondary_merge_and_redirect_indices
                                             ::Union{MergeAndRedirectIndices,Nothing})
    primary_merge_and_redirect_indices_count::Int64 =
      length(primary_merge_and_redirect_indices)
    primary_merge::Bool = (primary_merge_and_redirect_indices_count > 0)
    secondary_merge::Bool = secondary_merge_and_redirect_indices !== nothing
    return new(primary_merge,secondary_merge,
               primary_merge_and_redirect_indices,
               primary_merge_and_redirect_indices_count,
               secondary_merge_and_redirect_indices)
  end
end

function iterate(collection::MergeAndRedirectIndicesCollection,
                 state=1)
  if state <= collection.primary_merge_and_redirect_indices_count
    return collection.primary_merge_and_redirect_indices[state], state+1
  elseif state == collection.primary_merge_and_redirect_indices_count + 1
    if collection.secondary_merge_and_redirect_indices !== nothing
      return collection.secondary_merge_and_redirect_indices, state+1
    else
      return nothing
    end
  else
    return nothing
  end
end

function iterate(rcollection::Iterators.Reverse{MergeAndRedirectIndicesCollection},
                 state=rcollection.itr.primary_merge_and_redirect_indices_count + 1)
  if state == rcollection.itr.primary_merge_and_redirect_indices_count + 1
    if rcollection.itr.secondary_merge_and_redirect_indices !== nothing
      return rcollection.itr.secondary_merge_and_redirect_indices, state-1
    else
      state -= 1
    end
  end
  if state < 1
    return nothing
  elseif state <= rcollection.itr.primary_merge_and_redirect_indices_count
    return rcollection.itr.primary_merge_and_redirect_indices[state], state-1
  else
    throw(UserError())
  end
end

function add_offset(collections::Vector{MergeAndRedirectIndicesCollection},
                    offset::Int64)
  for collection in collections
    for merge_indices in collection
      add_offset(merge_indices,offset)
    end
  end
end

function reset(collections::Vector{MergeAndRedirectIndicesCollection})
  for collection in collections
    for merge_indices in collection
      reset(merge_indices)
    end
  end
end

function create_merge_indices_collections_from_array(array_in::Array{Int64,3})
  merge_and_redirect_indices_collections::Vector{MergeAndRedirectIndicesCollection} =
    Vector{MergeAndRedirectIndicesCollection}[]
  permuted_array = permutedims(array_in,(3,2,1))
  for i in axes(permuted_array,1)
    primary_merge_and_redirect_indices::Vector{MergeAndRedirectIndices} =
        Vector{MergeAndRedirectIndices}[]
    local secondary_merge_and_redirect_indices::Union{MergeAndRedirectIndices,Nothing}
    for j in axes(permuted_array,2)
      if permuted_array[i,j,1] == 1
        working_merge_and_redirect_indices::MergeAndRedirectIndices =
          read_merge_and_redirect_indices_from_array((j != 1),permuted_array[i,j,:])
        if j == 1
          secondary_merge_and_redirect_indices = working_merge_and_redirect_indices
        else
          push!(primary_merge_and_redirect_indices,working_merge_and_redirect_indices)
        end
      elseif j == 1
        secondary_merge_and_redirect_indices = nothing
      end
    end
    working_merge_and_redirect_indices_collection =
      MergeAndRedirectIndicesCollection(primary_merge_and_redirect_indices,
                                        secondary_merge_and_redirect_indices)
    push!(merge_and_redirect_indices_collections,
          working_merge_and_redirect_indices_collection)
  end
  return merge_and_redirect_indices_collections
end

abstract type GridSpecificLakeParameters end

struct LakeParameters
  lake_centers::Field{Bool}
  connection_volume_thresholds::Field{Float64}
  flood_volume_thresholds::Field{Float64}
  flood_only::Field{Bool}
  connect_merge_and_redirect_indices_index::Field{Int64}
  flood_merge_and_redirect_indices_index::Field{Int64}
  connect_merge_and_redirect_indices_collections::
    Vector{MergeAndRedirectIndicesCollection}
  flood_merge_and_redirect_indices_collections::
    Vector{MergeAndRedirectIndicesCollection}
  cell_areas_on_surface_model_grid::Field{Float64}
  cell_areas::Field{Float64}
  raw_heights::Field{Float64}
  corrected_heights::Field{Float64}
  flood_heights::Field{Float64}
  connection_heights::Field{Float64}
  basin_numbers::Field{Int64}
  number_fine_grid_cells::Field{Int64}
  basins::Vector{Vector{Coords}}
  surface_cell_to_fine_cell_maps::Vector{Vector{Coords}}
  surface_cell_to_fine_cell_map_numbers::Field{Int64}
  grid::Grid
  hd_grid::Grid
  surface_model_grid::Grid
  grid_specific_lake_parameters::GridSpecificLakeParameters
  lake_model_parameters::LakeModelParameters
  function LakeParameters(lake_centers::Field{Bool},
                          connection_volume_thresholds::Field{Float64},
                          flood_volume_thresholds::Field{Float64},
                          connect_merge_and_redirect_indices_index::Field{Int64},
                          flood_merge_and_redirect_indices_index::Field{Int64},
                          connect_merge_and_redirect_indices_collections
                          ::Vector{MergeAndRedirectIndicesCollection},
                          flood_merge_and_redirect_indices_collections
                          ::Vector{MergeAndRedirectIndicesCollection},
                          cell_areas_on_surface_model_grid::Field{Float64},
                          cell_areas::Field{Float64},
                          raw_heights::Field{Float64},
                          corrected_heights::Field{Float64},
                          flood_heights::Field{Float64},
                          connection_heights::Field{Float64},
                          grid::Grid,
                          hd_grid::Grid,
                          surface_model_grid::Grid,
                          grid_specific_lake_parameters::GridSpecificLakeParameters)
    flood_only::Field{Bool} =  Field{Bool}(grid,false)
    number_fine_grid_cells = Field{Int64}(surface_model_grid,0)
    surface_cell_to_fine_cell_maps::Vector{Vector{Coords}} = Vector{Coords}[]
    surface_cell_to_fine_cell_map_numbers = Field{Int64}(surface_model_grid,0)
    for_all(grid) do coords::Coords
      if connection_volume_thresholds(coords) == -1.0
        set!(flood_only,coords,true)
      end
      surface_model_coords::Coords =
        get_corresponding_surface_model_grid_cell(coords,grid_specific_lake_parameters)
      set!(number_fine_grid_cells,
           surface_model_coords,
           number_fine_grid_cells(surface_model_coords) + 1)
      if flood_volume_thresholds(coords) != -1.0
        local surface_cell_to_fine_cell_map::Vector{Coords}
        surface_cell_to_fine_cell_map_number::Int64 =
          surface_cell_to_fine_cell_map_numbers(surface_model_coords)
        if  surface_cell_to_fine_cell_map_number == 0
          surface_cell_to_fine_cell_map = Coords[coords]
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
    basins::Vector{Vector{Coords}} = Vector{Coords}[]
    basin_numbers::Field{Int64} = Field{Int64}(hd_grid)
    basin_number::Int64 = 1
    for_all(hd_grid) do coords::Coords
      basins_in_coarse_cell::Vector{Coords} = Coords[]
      basins_found::Bool = false
      for_all_fine_cells_in_coarse_cell(grid,hd_grid,coords) do fine_coords::Coords
        if lake_centers(fine_coords)
          push!(basins_in_coarse_cell,fine_coords)
          basins_found = true
        end
      end
      if basins_found
        push!(basins,basins_in_coarse_cell)
        set!(basin_numbers,coords,basin_number)
        basin_number += 1
      end
    end
    return new(lake_centers,
               connection_volume_thresholds,
               flood_volume_thresholds,
               flood_only,
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
               basin_numbers,
               number_fine_grid_cells,
               basins,surface_cell_to_fine_cell_maps,
               surface_cell_to_fine_cell_map_numbers,
               grid,hd_grid,
               surface_model_grid,
               grid_specific_lake_parameters,
               LakeModelParameters())
  end
end

struct LatLonLakeParameters <: GridSpecificLakeParameters
  flood_next_cell_lat_index::Field{Int64}
  flood_next_cell_lon_index::Field{Int64}
  connect_next_cell_lat_index::Field{Int64}
  connect_next_cell_lon_index::Field{Int64}
  corresponding_surface_cell_lat_index::Field{Int64}
  corresponding_surface_cell_lon_index::Field{Int64}
end

struct UnstructuredLakeParameters <: GridSpecificLakeParameters
  flood_next_cell_index::Field{Int64}
  connect_next_cell_index::Field{Int64}
  corresponding_surface_cell_index::Field{Int64}
end

struct LakeFields
  connected_lake_cells::Field{Bool}
  flooded_lake_cells::Field{Bool}
  lake_numbers::Field{Int64}
  buried_lake_numbers::Field{Int64}
  effective_volume_per_cell_on_surface_grid::Field{Float64}
  effective_lake_height_on_surface_grid_to_lakes::Field{Float64}
  water_to_lakes::Field{Float64}
  water_to_hd::Field{Float64}
  lake_water_from_ocean::Field{Float64}
  number_lake_cells::Field{Int64}
  true_lake_depths::Field{Float64}
  cells_with_lakes::Vector{Coords}
  evaporation_from_lakes::Array{Float64,1}
  evaporation_applied::BitArray{1}
  set_forest::RootedTreeForest
  function LakeFields(lake_parameters::LakeParameters)
    connected_lake_cells = Field{Bool}(lake_parameters.grid,false)
    flooded_lake_cells = Field{Bool}(lake_parameters.grid,false)
    lake_numbers = Field{Int64}(lake_parameters.grid,0)
    buried_lake_numbers = Field{Int64}(lake_parameters.grid,0)
    effective_volume_per_cell_on_surface_grid =
      Field{Float64}(lake_parameters.surface_model_grid,0.0)
    effective_lake_height_on_surface_grid_to_lakes =
      Field{Float64}(lake_parameters.surface_model_grid,0.0)
    water_to_lakes = Field{Float64}(lake_parameters.hd_grid,0.0)
    water_to_hd = Field{Float64}(lake_parameters.hd_grid,0.0)
    lake_water_from_ocean = Field{Float64}(lake_parameters.hd_grid,0.0)
    number_lake_cells = Field{Int64}(lake_parameters.surface_model_grid,0)
    true_lake_depths = Field{Float64}(lake_parameters.grid,0.0)
    cells_with_lakes::Vector{Coords} = Coords[]
    for_all(lake_parameters.hd_grid) do coords::Coords
      contains_lake::Bool = false
      for_all_fine_cells_in_coarse_cell(lake_parameters.grid,
                                        lake_parameters.hd_grid,coords) do fine_coords::Coords
        if lake_parameters.lake_centers(fine_coords) contains_lake = true end
      end
      if contains_lake
        push!(cells_with_lakes,coords)
      end
    end
    evaporation_from_lakes::Array{Float64,1} = zeros(Float64,count(lake_parameters.lake_centers))
    evaporation_applied = falses(count(lake_parameters.lake_centers))
    set_forest::RootedTreeForest = RootedTreeForest()
    new(connected_lake_cells,flooded_lake_cells,lake_numbers,buried_lake_numbers,
        effective_volume_per_cell_on_surface_grid,effective_lake_height_on_surface_grid_to_lakes,
        water_to_lakes,water_to_hd,lake_water_from_ocean,number_lake_cells,true_lake_depths,
        cells_with_lakes,evaporation_from_lakes,evaporation_applied,set_forest)
  end
end

struct LakePrognostics
  lakes::Vector{Lake}
  function LakePrognostics(lake_parameters::LakeParameters,
                           lake_fields::LakeFields)
    lakes::Vector{Lake} = Lake[]
    lake_number::Int64 = 1
    for_all(lake_parameters.grid) do coords::Coords
      if lake_parameters.lake_centers(coords)
        coarse_coords = find_coarse_cell_containing_fine_cell(lake_parameters.grid,
                                                              lake_parameters.hd_grid,
                                                              coords)
        push!(lakes,FillingLake(lake_parameters,
                                LakeVariables(coords,lakes,
                                              lake_number,
                                              coarse_coords),
                                lake_fields))
        set!(lake_fields.lake_numbers,coords,lake_number)
        add_set(lake_fields.set_forest,lake_number)
        lake_number += 1
        if lake_parameters.flood_only(coords)
          surface_model_coords::Coords =
            get_corresponding_surface_model_grid_cell(coords,
                                                      lake_parameters.grid_specific_lake_parameters)
          set!(lake_fields.number_lake_cells,surface_model_coords,
               lake_fields.number_lake_cells(surface_model_coords) + 1)
        end
      end
    end
    return new(lakes)
  end
end

mutable struct LakeVariables
  center_cell::Coords
  lake_volume::Float64
  secondary_lake_volume::Float64
  unprocessed_water::Float64
  current_cell_to_fill::Coords
  previous_cell_to_fill::Coords
  other_lakes::Vector{Lake}
  lake_number::Int64
  filled_lake_cells::Vector{Coords}
  center_cell_coarse_coords::Coords
  number_of_flooded_cells::Int64
  secondary_number_of_flooded_cells::Int64
  LakeVariables(center_cell::Coords,
                other_lakes::Vector{Lake},
                lake_number::Int64,
                center_cell_coarse_coords::Coords) =
    new(center_cell,0.0,0.0,0.0,deepcopy(center_cell),deepcopy(center_cell),
        other_lakes,lake_number,Coords[],center_cell_coarse_coords,0,0)
end

get_lake_parameters(obj::T) where {T <: Lake} =
  obj.lake_parameters::LakeParameters

get_lake_variables(obj::T) where {T <: Lake} =
  obj.lake_variables::LakeVariables

get_lake_fields(obj::T) where {T <: Lake} =
  obj.lake_fields::LakeFields

function handle_event(lake::Lake,inflow::Float64)
  throw(UserError())
end

struct FillingLake <: Lake
  lake_parameters::LakeParameters
  lake_variables::LakeVariables
  lake_fields::LakeFields
end

mutable struct OverflowingLakeVariables
  excess_water::Float64
end

OverflowingLakeVariables() = OverflowingLakeVariables(0.0)

struct OverflowingLake <: Lake
  lake_parameters::LakeParameters
  lake_variables::LakeVariables
  lake_fields::LakeFields
  outflow_redirect_coords::Coords
  local_redirect::Bool
  lake_retention_coefficient::Float64
  next_merge_target_coords::Coords
  overflowing_lake_variables::OverflowingLakeVariables
end

struct SubsumedLake <: Lake
  lake_parameters::LakeParameters
  lake_variables::LakeVariables
  lake_fields::LakeFields
  primary_lake_number::Int64
end

mutable struct LakeDiagnosticVariables
  total_lake_volume::Float64
end

LakeDiagnosticVariables() = LakeDiagnosticVariables(0.0)

struct RiverAndLakePrognosticFields <: PrognosticFields
  river_parameters::RiverParameters
  river_fields::RiverPrognosticFields
  river_diagnostic_fields::RiverDiagnosticFields
  river_diagnostic_output_fields::RiverDiagnosticOutputFields
  lake_parameters::LakeParameters
  lake_prognostics::LakePrognostics
  lake_fields::LakeFields
  lake_diagnostics_variables::LakeDiagnosticVariables
  using_lakes::Bool
  function RiverAndLakePrognosticFields(river_parameters::RiverParameters,
                                        river_fields::RiverPrognosticFields,
                                        lake_parameters::LakeParameters,
                                        lake_prognostics::LakePrognostics,
                                        lake_fields::LakeFields)
    for_all(river_parameters.grid) do coords::Coords
      if is_lake(river_parameters.flow_directions(coords))
        set!(river_parameters.cascade_flag,coords,false)
      end
    end
    new(river_parameters,river_fields,RiverDiagnosticFields(river_parameters),
        RiverDiagnosticOutputFields(river_parameters),lake_parameters,
        lake_prognostics,lake_fields,LakeDiagnosticVariables(),true)
  end
end

get_lake_prognostics(river_and_lake_prognostics::RiverAndLakePrognosticFields) =
                     river_and_lake_prognostics.lake_prognostics::LakePrognostics

get_lake_fields(river_and_lake_prognostics::RiverAndLakePrognosticFields) =
                     river_and_lake_prognostics.lake_fields::LakeFields

get_lake_parameters(river_and_lake_prognostics::RiverAndLakePrognosticFields) =
                     river_and_lake_prognostics.lake_parameters::LakeParameters

get_lake_diagnostic_variables(river_and_lake_prognostics::RiverAndLakePrognosticFields) =
  river_and_lake_prognostics.lake_diagnostics_variables::LakeDiagnosticVariables

function water_to_lakes(prognostic_fields::RiverAndLakePrognosticFields)
  lake_fields = get_lake_fields(prognostic_fields)
  return lake_fields.water_to_lakes
end

function water_from_lakes(prognostic_fields::RiverAndLakePrognosticFields,
                          step_length::Float64)
  lake_fields = get_lake_fields(prognostic_fields)
  return lake_fields.water_to_hd/step_length,
         lake_fields.lake_water_from_ocean/step_length
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,setup_lakes::SetupLakes)
  lake_fields::LakeFields = get_lake_fields(prognostic_fields)
  lake_prognostics::LakePrognostics = get_lake_prognostics(prognostic_fields)
  lake_parameters::LakeParameters = get_lake_parameters(prognostic_fields)
  for_all(lake_parameters.grid) do coords::Coords
    if lake_parameters.lake_centers(coords)
      initial_water_to_lake_center::Float64 = setup_lakes.initial_water_to_lake_centers(coords)
      if  initial_water_to_lake_center > 0.0
        add_water::AddWater = AddWater(initial_water_to_lake_center)
        lake_index::Int64 = lake_fields.lake_numbers(coords)
        lake::Lake = lake_prognostics.lakes[lake_index]
        lake_prognostics.lakes[lake_index] = handle_event(lake,add_water)
      end
    end
  end
  return prognostic_fields
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      distribute_spillover::DistributeSpillover)
  lake_fields::LakeFields = get_lake_fields(prognostic_fields)
  lake_parameters::LakeParameters = get_lake_parameters(prognostic_fields)
  for_all(lake_parameters.hd_grid) do coords::Coords
    initial_spillover::Float64 = distribute_spillover.initial_spillover_to_rivers(coords)
    if initial_spillover > 0.0
      set!(lake_fields.water_to_hd,coords,lake_fields.water_to_hd(coords) + initial_spillover)
    end
  end
  return prognostic_fields
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,run_lake::RunLakes)
  river_parameters::RiverParameters = get_river_parameters(prognostic_fields)
  river_fields::RiverPrognosticFields = get_river_fields(prognostic_fields)
  lake_fields::LakeFields = get_lake_fields(prognostic_fields)
  lake_parameters::LakeParameters = get_lake_parameters(prognostic_fields)
  lake_prognostics::LakePrognostics = get_lake_prognostics(prognostic_fields)
  drain_excess_water::DrainExcessWater = DrainExcessWater()
  fill!(lake_fields.water_to_hd,0.0)
  fill!(lake_fields.lake_water_from_ocean,0.0)
  fill!(lake_fields.evaporation_from_lakes,0.0)
  local basins_in_cell::Array{Coords,1}
  new_effective_volume_per_cell_on_surface_grid::Field{Float64} =
    elementwise_multiple(lake_fields.effective_lake_height_on_surface_grid_to_lakes,
                         lake_parameters.cell_areas_on_surface_model_grid)
  evaporation_on_surface_grid::Field{Float64} = lake_fields.effective_volume_per_cell_on_surface_grid -
                                                new_effective_volume_per_cell_on_surface_grid
  fill!(lake_fields.effective_volume_per_cell_on_surface_grid,0.0)
  for_all(lake_parameters.surface_model_grid) do coords::Coords
    number_lake_cells::Int64 = lake_fields.number_lake_cells(coords)
    if evaporation_on_surface_grid(coords) != 0.0 && number_lake_cells > 0
      cell_count_check::Int64 = 0
      map_index::Int64 = lake_parameters.surface_cell_to_fine_cell_map_numbers(coords)
      evaporation_per_lake_cell::Float64 = evaporation_on_surface_grid(coords)/number_lake_cells
      if map_index != 0
        for fine_coords::Coords in lake_parameters.surface_cell_to_fine_cell_maps[map_index]
          target_lake_index::Int64 = lake_fields.lake_numbers(fine_coords)
          if target_lake_index != 0
            lake::Lake = lake_prognostics.lakes[target_lake_index]
            if lake_fields.flooded_lake_cells(fine_coords) ||
               ((lake.lake_variables.current_cell_to_fill == fine_coords) &&
                (lake_parameters.flood_only(fine_coords) ||
                 lake_fields.connected_lake_cells(fine_coords)))
              lake_fields.evaporation_from_lakes[target_lake_index] += evaporation_per_lake_cell
              cell_count_check += 1
            end
          end
        end
      end
      if cell_count_check != number_lake_cells
        error("Inconsitent cell count when assigning evaporation")
      end
    end
  end
  fill!(lake_fields.evaporation_applied,false)
  for coords::Coords in lake_fields.cells_with_lakes
    if lake_fields.water_to_lakes(coords) > 0.0
      basins_in_cell =
        lake_parameters.basins[lake_parameters.basin_numbers(coords)]
      share_to_each_lake::Float64 = lake_fields.water_to_lakes(coords)/length(basins_in_cell)
      add_water::AddWater = AddWater(share_to_each_lake)
      for basin_center::Coords in basins_in_cell
        lake_index::Int64 = lake_fields.lake_numbers(basin_center)
        lake::Lake = lake_prognostics.lakes[lake_index]
        if ! lake_fields.evaporation_applied[lake_index]
          inflow_minus_evaporation::Float64 = share_to_each_lake -
                                              lake_fields.evaporation_from_lakes[lake_index]
          if inflow_minus_evaporation >= 0.0
            add_water_modified::AddWater = AddWater(inflow_minus_evaporation)
            lake_prognostics.lakes[lake_index] = handle_event(lake,add_water_modified)
          else
            remove_water_modified::RemoveWater = RemoveWater(-1.0*inflow_minus_evaporation)
            lake_prognostics.lakes[lake_index] = handle_event(lake,remove_water_modified)
          end
          lake_fields.evaporation_applied[lake_index] = true
        else
          lake_prognostics.lakes[lake_index] = handle_event(lake,add_water)
        end
      end
    elseif lake_fields.water_to_lakes(coords) < 0.0
      basins_in_cell =
        lake_parameters.basins[lake_parameters.basin_numbers(coords)]
      share_to_each_lake = -1.0*lake_fields.water_to_lakes(coords)/length(basins_in_cell)
      remove_water::RemoveWater = RemoveWater(share_to_each_lake)
      for basin_center::Coords in basins_in_cell
        lake_index::Int64 = lake_fields.lake_numbers(basin_center)
        lake::Lake = lake_prognostics.lakes[lake_index]
        lake_prognostics.lakes[lake_index] = handle_event(lake,remove_water)
      end
    end
  end
  for lake::Lake in lake_prognostics.lakes
    lake_variables::LakeVariables = get_lake_variables(lake)
    lake_index::Int64 = lake_variables.lake_number
    if lake_variables.unprocessed_water > 0.0
      lake_prognostics.lakes[lake_index] = handle_event(lake,AddWater(0.0))
    end
  end
  for lake::Lake in lake_prognostics.lakes
    lake_variables::LakeVariables = get_lake_variables(lake)
    lake_index::Int64 = lake_variables.lake_number
    if ! lake_fields.evaporation_applied[lake_index]
      evaporation::Float64 = lake_fields.evaporation_from_lakes[lake_index]
      if evaporation > 0
        lake_prognostics.lakes[lake_index] = handle_event(lake,RemoveWater(evaporation))
      elseif evaporation < 0
        lake_prognostics.lakes[lake_index] = handle_event(lake,AddWater(-1.0*evaporation))
      end
      lake_fields.evaporation_applied[lake_index] = true
    end
  end
  for lake::Lake in lake_prognostics.lakes
    lake_variables::LakeVariables = get_lake_variables(lake)
    lake_index::Int64 = lake_variables.lake_number
    if lake_variables.lake_volume < 0.0
      lake_prognostics.lakes[lake_index] = handle_event(lake,ReleaseNegativeWater())
    end
  end
  for lake::Lake in lake_prognostics.lakes
    lake_variables::LakeVariables = get_lake_variables(lake)
    lake_index::Int64 = lake_variables.lake_number
    if isa(lake,OverflowingLake)
      handle_event(lake,drain_excess_water)
    end
  end
  for lake::Lake in lake_prognostics.lakes
    lake_variables::LakeVariables = get_lake_variables(lake)
    lake_index::Int64 = lake_variables.lake_number
    lake_prognostics.lakes[lake_index] = handle_event(lake,
                                                      CalculateEffectiveLakeVolumePerCell())
  end
  return prognostic_fields
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      calculate_true_lake_depths::CalculateTrueLakeDepths)
  lake_prognostics::LakePrognostics = get_lake_prognostics(prognostic_fields)
  for lake::Lake in lake_prognostics.lakes
    if isa(lake,OverflowingLake) || isa(lake,FillingLake)
      handle_event(lake,calculate_true_lake_depths)
    end
  end
  return prognostic_fields
end

function handle_event(lake::FillingLake,add_water::AddWater)
  inflow::Float64 = add_water.inflow + lake.lake_variables.unprocessed_water
  lake.lake_variables.unprocessed_water = 0.0
  while inflow > 0.0
    inflow,filled::Bool = fill_current_cell(lake,inflow)
    if filled
      merge_indices_index::Int64,is_flood_merge::Bool =
        get_merge_indices_index(lake)
      if merge_indices_index != 0
        merge_indices_collection::MergeAndRedirectIndicesCollection =
          get_merge_indices_collection(lake,merge_indices_index,
                                       is_flood_merge)
        for merge_indices::MergeAndRedirectIndices in merge_indices_collection
          local merge_possible::Bool
          local already_merged::Bool
          local merge_type::MergeTypes
          merge_possible,already_merged,merge_type =
            check_if_merge_is_possible(lake,merge_indices)
          local subsumed_lake::SubsumedLake
          if merge_possible
            if merge_type == secondary_merge
              subsumed_lake = perform_secondary_merge(lake,merge_indices)
              subsumed_lake = handle_event(subsumed_lake,
                                           StoreWater(inflow))
              return subsumed_lake
            else
              perform_primary_merge(lake,merge_indices)
            end
          elseif ! already_merged
            #Note becoming an overflowing lake occur not only when the other basin
            #is not yet full enough but also when the other other basin is overflowing
            #but is filling another basin at the same height (at a tri basin meeting point)
            overflowing_lake::Lake = change_to_overflowing_lake(lake,merge_indices)
            overflowing_lake = handle_event(overflowing_lake,StoreWater(inflow))
            return overflowing_lake
          end
        end
      end
      update_filling_cell(lake)
    end
  end
  return lake
end

function handle_event(lake::FillingLake,remove_water::RemoveWater)
  drained::Bool = true
  if remove_water.outflow <= lake.lake_variables.unprocessed_water
    lake.lake_variables.unprocessed_water -= remove_water.outflow
    return lake
  end
  outflow::Float64 = remove_water.outflow - lake.lake_variables.unprocessed_water
  lake.lake_variables.unprocessed_water = 0.0
  while outflow > 0.0 || drained
    outflow,drained = drain_current_cell(lake,outflow)
    if drained
      rollback_filling_cell(lake)
      merge_indices_index::Int64,is_flood_merge::Bool =
        get_merge_indices_index(lake)
      if merge_indices_index != 0
        merge_indices_collection::MergeAndRedirectIndicesCollection =
          get_merge_indices_collection(lake,merge_indices_index,
                                       is_flood_merge)
        for merge_indices::MergeAndRedirectIndices in
              Iterators.reverse(merge_indices_collection)
          merge_type::MergeTypes = get_merge_type(merge_indices)
          if merge_type == primary_merge
            new_outflow::Float64 = outflow/2.0
            if conditionally_rollback_primary_merge(lake,merge_indices,new_outflow)
              outflow -= new_outflow
            end
          elseif merge_type == secondary_merge
            error("Merge logic failure")
          else
            error("Unrecognised merge type")
          end
        end
      end
    end
  end
  return lake
end

function handle_event(lake::OverflowingLake,add_water::AddWater)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  inflow::Float64 = add_water.inflow +
                    lake_variables.unprocessed_water
  lake_variables.unprocessed_water = 0.0
  if lake.local_redirect
    other_lake_number::Int64 = lake_fields.lake_numbers(lake.outflow_redirect_coords)
    other_lake::Lake =
      lake_variables.other_lakes[other_lake_number]
    if lake_parameters.lake_model_parameters.instant_throughflow
      lake_variables.other_lakes[other_lake_number] = handle_event(other_lake,
                                                             AddWater(inflow))
    else
      lake_variables.other_lakes[other_lake_number] = handle_event(other_lake,
                                                             StoreWater(inflow))
    end
  else
    lake.overflowing_lake_variables.excess_water += inflow
  end
  return lake_variables.other_lakes[lake_variables.lake_number]
end

function handle_event(lake::OverflowingLake,remove_water::RemoveWater)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  if remove_water.outflow <= lake_variables.unprocessed_water
    lake_variables.unprocessed_water -= remove_water.outflow
    return lake
  end
  outflow::Float64 = remove_water.outflow - lake_variables.unprocessed_water
  lake_variables.unprocessed_water = 0.0
  if outflow <= lake.overflowing_lake_variables.excess_water
    lake.overflowing_lake_variables.excess_water -= outflow
    return lake
  end
  outflow -= lake.overflowing_lake_variables.excess_water
  lake.overflowing_lake_variables.excess_water = 0
  if lake_fields.flooded_lake_cells(lake_variables.current_cell_to_fill)
    set!(lake_fields.flooded_lake_cells,lake_variables.current_cell_to_fill,false)
    lake_variables.number_of_flooded_cells -= 1
  else
    set!(lake_fields.connected_lake_cells,lake_variables.current_cell_to_fill,false)
  end
  lake_as_filling_lake::FillingLake = change_to_filling_lake(lake)
  merge_indices_index::Int64,is_flood_merge::Bool = get_merge_indices_index(lake)
  if merge_indices_index != 0
    merge_indices_collection::MergeAndRedirectIndicesCollection =
      get_merge_indices_collection(lake,merge_indices_index,
                                   is_flood_merge)
    for merge_indices::MergeAndRedirectIndices in
          Iterators.reverse(merge_indices_collection)
      merge_type::MergeTypes = get_merge_type(merge_indices)
      if merge_type == primary_merge
        new_outflow::Float64 = outflow/2.0
        if conditionally_rollback_primary_merge(lake,merge_indices,new_outflow)
          outflow -= new_outflow
        end
      elseif merge_type == secondary_merge
        if merge_indices.merged
          error("Merge logic failure")
        end
      else
        error("Unrecognised merge type")
      end
    end
  end
  new_remove_water::RemoveWater = RemoveWater(outflow)
  return handle_event(lake_as_filling_lake,new_remove_water)
end

function handle_event(lake::Lake,store_water::StoreWater)
  lake.lake_variables.unprocessed_water += store_water.inflow
  return lake
end

function handle_event(lake::SubsumedLake,add_water::AddWater)
  lake_variables::LakeVariables = get_lake_variables(lake)
  inflow::Float64 = add_water.inflow + lake_variables.unprocessed_water
  lake_variables.unprocessed_water = 0.0
  lake_variables.other_lakes[lake.primary_lake_number] =
    handle_event(lake_variables.other_lakes[lake.primary_lake_number],
                 AddWater(inflow))
  return lake
end

function handle_event(lake::SubsumedLake,remove_water::RemoveWater)
  lake_variables::LakeVariables = get_lake_variables(lake)
  if remove_water.outflow <= lake_variables.unprocessed_water
    lake_variables.unprocessed_water -= remove_water.outflow
    return lake
  end
  outflow::Float64 = remove_water.outflow - lake_variables.unprocessed_water
  lake_variables.unprocessed_water = 0.0
  lake_variables.other_lakes[lake.primary_lake_number] =
    handle_event(lake_variables.other_lakes[lake.primary_lake_number],
                 RemoveWater(outflow))
  return lake_variables.other_lakes[lake_variables.lake_number]::Lake
end

function handle_event(lake::Lake,release_negative_water::ReleaseNegativeWater)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  set!(lake_fields.lake_water_from_ocean,
       lake_variables.center_cell_coarse_coords,
       lake_fields.lake_water_from_ocean(lake_variables.center_cell_coarse_coords) -
       lake_variables.lake_volume)
  lake_variables.lake_volume = 0.0
  return lake
end

function handle_event(lake::Lake,accept_merge::AcceptMerge)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  if debug
    println("Lake $(lake_variables.lake_number) accepting merge")
  end
  if ! (lake_fields.connected_lake_cells(lake_variables.current_cell_to_fill) ||
        lake_parameters.flood_only(lake_variables.current_cell_to_fill))
    set!(lake_fields.connected_lake_cells,lake_variables.current_cell_to_fill,true)
  elseif ! lake_fields.flooded_lake_cells(lake_variables.current_cell_to_fill)
    set!(lake_fields.flooded_lake_cells,lake_variables.current_cell_to_fill,true)
    lake_variables.number_of_flooded_cells += 1
  end
  subsumed_lake::SubsumedLake = SubsumedLake(lake_parameters,
                                             lake_variables,
                                             lake_fields,
                                             lake_fields.
                                             lake_numbers(accept_merge.
                                                          redirect_coords))
  make_new_link(lake_fields.set_forest,
                accept_merge.primary_lake_number,
                lake_variables.lake_number)
  if isa(lake,OverflowingLake)
    subsumed_lake = handle_event(subsumed_lake,StoreWater(lake.
                                                          overflowing_lake_variables.
                                                          excess_water))
  end
  return subsumed_lake
end

function handle_event(lake::SubsumedLake,accept_split::AcceptSplit)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  if debug
    println("Lake $(lake_variables.lake_number) accepting split")
  end
  if lake_fields.flooded_lake_cells(lake_variables.current_cell_to_fill)
      set!(lake_fields.flooded_lake_cells,lake_variables.current_cell_to_fill,false)
      lake_variables.number_of_flooded_cells -= 1
  else
    set!(lake_fields.connected_lake_cells,lake_variables.current_cell_to_fill,false)
  end
  filling_lake::FillingLake = FillingLake(lake_parameters,
                                          lake_variables,
                                          lake.lake_fields)
  split_set(lake_fields.set_forest,
            accept_split.primary_lake_number,
            lake_variables.lake_number)
  return filling_lake
end

function change_to_overflowing_lake(lake::FillingLake,merge_indices::MergeAndRedirectIndices)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  if debug
    println("Lake $(lake_variables.lake_number) changing from filling to overflowing lake ")
  end
  outflow_redirect_coords,local_redirect =
    get_outflow_redirect_coords(merge_indices)
  next_merge_target_coords::Coords = get_merge_target_coords(merge_indices)
  if ! (lake_fields.connected_lake_cells(lake_variables.current_cell_to_fill) ||
        lake_parameters.flood_only(lake_variables.current_cell_to_fill))
    set!(lake_fields.connected_lake_cells,lake_variables.current_cell_to_fill,true)
  elseif ! lake_fields.flooded_lake_cells(lake_variables.current_cell_to_fill)
    set!(lake_fields.flooded_lake_cells,lake_variables.current_cell_to_fill,true)
    lake_variables.number_of_flooded_cells += 1
  end
  return OverflowingLake(lake.lake_parameters,
                         lake_variables,
                         lake_fields,
                         outflow_redirect_coords,
                         local_redirect,
                         lake.lake_parameters.lake_model_parameters.lake_retention_constant,
                         next_merge_target_coords,
                         OverflowingLakeVariables())
end

function handle_event(lake::OverflowingLake,drain_excess_water::DrainExcessWater)
  lake_fields::LakeFields = get_lake_fields(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  if lake.overflowing_lake_variables.excess_water > 0.0
    lake.overflowing_lake_variables.excess_water +=
      lake_variables.unprocessed_water
    lake_variables.unprocessed_water = 0.0
    flow = (lake.overflowing_lake_variables.excess_water+
            lake_variables.lake_volume+
            lake_variables.secondary_lake_volume)/
           (lake.lake_retention_coefficient + 1.0)
    flow = min(flow,lake.overflowing_lake_variables.excess_water)
    set!(lake_fields.water_to_hd,lake.outflow_redirect_coords,
         lake_fields.water_to_hd(lake.outflow_redirect_coords)+flow)
    lake.overflowing_lake_variables.excess_water -= flow
  end
  return lake
end

function handle_event(lake::Union{OverflowingLake,FillingLake},
    calculate_effective_lake_volume_per_cell::CalculateEffectiveLakeVolumePerCell)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  total_number_of_flooded_cells::Int64 =
    lake_variables.number_of_flooded_cells+lake_variables.secondary_number_of_flooded_cells
  effective_volume_per_cell::Float64 =
    (lake_variables.lake_volume + lake_variables.secondary_lake_volume) /
    ( total_number_of_flooded_cells + ((isa(lake,FillingLake) ||
                                       (total_number_of_flooded_cells == 0)) ? 1 : 0))
  working_cell_list::Vector{Coords} =
    calculate_effective_lake_volume_per_cell.consider_secondary_lake ?
    calculate_effective_lake_volume_per_cell.cell_list :
    lake_variables.filled_lake_cells
  if (! calculate_effective_lake_volume_per_cell.consider_secondary_lake)
    surface_model_coords::Coords =
      get_corresponding_surface_model_grid_cell(lake_variables.current_cell_to_fill,
                                                lake_parameters.grid_specific_lake_parameters)
    set!(lake_fields.effective_volume_per_cell_on_surface_grid,surface_model_coords,
         lake_fields.effective_volume_per_cell_on_surface_grid(surface_model_coords)+
         effective_volume_per_cell)
  end
  for coords::Coords in working_cell_list
    surface_model_coords = get_corresponding_surface_model_grid_cell(coords,
                              lake_parameters.grid_specific_lake_parameters)
    set!(lake_fields.effective_volume_per_cell_on_surface_grid,surface_model_coords,
         lake_fields.effective_volume_per_cell_on_surface_grid(surface_model_coords)+
         effective_volume_per_cell)
  end

  return lake
end

function handle_event(lake::SubsumedLake,
    calculate_effective_lake_volume_per_cell::CalculateEffectiveLakeVolumePerCell)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_variables.other_lakes[lake.primary_lake_number] =
    handle_event(lake_variables.other_lakes[lake.primary_lake_number],
                 CalculateEffectiveLakeVolumePerCell(lake_variables.filled_lake_cells))
  return lake
end

function handle_event(lake::FillingLake,
                      calculate_true_lake_depths::CalculateTrueLakeDepths)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  current_cell_to_fill_height::Float64 =
    (lake_fields.connected_lake_cells(lake_variables.current_cell_to_fill) ||
     lake_parameters.flood_only(lake_variables.current_cell_to_fill) ?
     lake_parameters.raw_heights(lake_variables.current_cell_to_fill) :
     lake_parameters.corrected_heights(lake_variables.current_cell_to_fill))
  local highest_threshold_reached::Float64
  if lake_variables.previous_cell_to_fill != lake_variables.current_cell_to_fill
    highest_threshold_reached =
      (lake_fields.flooded_lake_cells(lake_variables.previous_cell_to_fill) ||
      lake_parameters.flood_only(lake_variables.previous_cell_to_fill) ?
      lake_parameters.flood_volume_thresholds(lake_variables.previous_cell_to_fill)
      : lake_parameters.connect_volume_thresholds(lake_variables.previous_cell_to_fill))
  else
    highest_threshold_reached = 0.0
  end
  volume_above_current_filling_cell::Float64 =
    lake_variables.lake_volume +
    lake_variables.unprocessed_water -
    highest_threshold_reached
  working_cell_list::Vector{Coords} = Coords[]
  for coords in lake_variables.filled_lake_cells
    if coords != lake_variables.current_cell_to_fill
      push!(working_cell_list,coords)
    end
  end
  push!(working_cell_list,lake_variables.current_cell_to_fill)
  for secondary_lake_number::Int64 in get_all_node_labels_of_set(lake_fields.set_forest,
                                                                 lake_variables.lake_number)
    if ! (secondary_lake_number == lake_variables.lake_number)
      secondary_lake::Lake = lake_variables.other_lakes[secondary_lake_number]
      for coords in secondary_lake.lake_variables.filled_lake_cells
        if lake_fields.lake_numbers(coords) == secondary_lake_number
          push!(working_cell_list,coords)
        end
      end
      if lake_fields.lake_numbers(secondary_lake.lake_variables.current_cell_to_fill) ==
         secondary_lake_number
        push!(working_cell_list,secondary_lake.lake_variables.current_cell_to_fill)
      end
    end
  end
  total_lake_area::Float64 = 0.0
  for coords::Coords in working_cell_list
    total_lake_area +=
      (lake_fields.flooded_lake_cells(coords) ||
       lake_parameters.flood_only(coords) ||
       (coords == lake_variables.current_cell_to_fill &&
        lake_fields.connected_lake_cells(coords))) ?
       lake_parameters.cell_areas(coords) : 0.0
  end
  depth_above_current_filling_cell::Float64 =
    volume_above_current_filling_cell / total_lake_area
  for coords::Coords in working_cell_list
    if lake_fields.flooded_lake_cells(coords) ||
       lake_parameters.flood_only(coords) ||
       (coords == lake_variables.current_cell_to_fill &&
        lake_fields.connected_lake_cells(coords))
      set!(lake_fields.true_lake_depths,coords,
           depth_above_current_filling_cell + current_cell_to_fill_height -
           lake_parameters.raw_heights(coords))
    end
  end
  return lake
end

function handle_event(lake::OverflowingLake,
                      calculate_true_lake_depths::CalculateTrueLakeDepths)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  sill_height::Float64 =
    (lake_fields.flooded_lake_cells(lake_variables.current_cell_to_fill) ||
     lake_parameters.flood_only(lake_variables.current_cell_to_fill) ?
     lake_parameters.flood_heights(lake_variables.current_cell_to_fill) :
     lake_parameters.connection_heights(lake_variables.current_cell_to_fill))
  volume_above_sill::Float64 = lake.overflowing_lake_variables.excess_water +
                               lake_variables.unprocessed_water
  working_cell_list::Vector{Coords} =
    lake_variables.filled_lake_cells
  push!(working_cell_list,lake_variables.current_cell_to_fill)
  for secondary_lake_number::Int64 in get_all_node_labels_of_set(lake_fields.set_forest,
                                                                 lake_variables.lake_number)
    if ! (secondary_lake_number == lake_variables.lake_number)
      secondary_lake::Lake = lake_variables.other_lakes[secondary_lake_number]
      append!(working_cell_list,
              secondary_lake.lake_variables.filled_lake_cells)
      push!(working_cell_list,secondary_lake.lake_variables.current_cell_to_fill)
    end
  end
  total_lake_area::Float64 =
    (lake_fields.flooded_lake_cells(lake_variables.current_cell_to_fill) ||
     lake_parameters.flood_only(lake_variables.current_cell_to_fill) ?
     lake_parameters.cell_areas(lake_variables.current_cell_to_fill) : 0.0)
  for coords::Coords in working_cell_list
    total_lake_area +=
      (lake_fields.flooded_lake_cells(coords) ||
       lake_parameters.flood_only(coords) ?
       lake_parameters.cell_areas(coords) : 0.0)
  end
  depth_above_sill::Float64 =
    volume_above_sill / total_lake_area
  for coords::Coords in working_cell_list
    if lake_fields.flooded_lake_cells(coords) ||
       lake_parameters.flood_only(coords)
      set!(lake_fields.true_lake_depths,coords,
           depth_above_sill + sill_height -
           lake_parameters.raw_heights(coords))
    end
  end
  return lake
end

function fill_current_cell(lake::FillingLake,inflow::Float64)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  current_cell_to_fill::Coords = lake_variables.current_cell_to_fill
  new_lake_volume::Float64 = inflow + lake_variables.lake_volume
  maximum_new_lake_volume::Float64 =
    (lake_fields.connected_lake_cells(current_cell_to_fill) ||
    lake_parameters.flood_only(current_cell_to_fill)) ?
    lake_parameters.flood_volume_thresholds(current_cell_to_fill) :
    lake_parameters.connection_volume_thresholds(current_cell_to_fill)
  if new_lake_volume <= maximum_new_lake_volume
    lake_variables.lake_volume = new_lake_volume
    return 0.0,false
  else
    inflow = new_lake_volume - maximum_new_lake_volume
    lake_variables.lake_volume = maximum_new_lake_volume
    return inflow,true
  end
end

function drain_current_cell(lake::FillingLake,outflow::Float64)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  if size(lake_variables.filled_lake_cells,1) == 0
    lake_variables.lake_volume -= outflow
    return 0.0,false
  end
  current_cell_to_fill::Coords = lake_variables.current_cell_to_fill
  previous_cell_to_fill::Coords = lake_variables.previous_cell_to_fill
  new_lake_volume::Float64 = lake_variables.lake_volume - outflow
  minimum_new_lake_volume::Float64 =
    lake_fields.flooded_lake_cells(previous_cell_to_fill)  ?
    lake_parameters.flood_volume_thresholds(previous_cell_to_fill) :
    lake_parameters.connection_volume_thresholds(previous_cell_to_fill)
  if new_lake_volume <= 0.0 && minimum_new_lake_volume <= 0.0
    lake_variables.lake_volume = new_lake_volume
    return 0.0,true
  elseif (new_lake_volume <=
          lake_parameters.lake_model_parameters.minimum_lake_volume_threshold) &&
         (minimum_new_lake_volume <= 0.0)
    lake_variables.lake_volume = 0.0
    set!(lake_fields.lake_water_from_ocean,
         lake_variables.center_cell_coarse_coords,
         lake_fields.lake_water_from_ocean(lake_variables.center_cell_coarse_coords) -
         new_lake_volume)
    return 0.0,true
  elseif new_lake_volume >= minimum_new_lake_volume
    lake_variables.lake_volume = new_lake_volume
    return 0.0,false
  else
    outflow = minimum_new_lake_volume - new_lake_volume
    lake_variables.lake_volume = minimum_new_lake_volume
    return outflow,true
  end
end

function get_merge_type(merge_indices::MergeAndRedirectIndices)
  if merge_indices.is_primary_merge
    return primary_merge
  else
    return secondary_merge
  end
end

function get_merge_indices_index(lake::Lake)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  current_cell::Coords = lake_variables.current_cell_to_fill
  if lake_fields.connected_lake_cells(current_cell) ||
     lake_parameters.flood_only(current_cell)
    return lake_parameters.flood_merge_and_redirect_indices_index(current_cell),true
  else
    return lake_parameters.connect_merge_and_redirect_indices_index(current_cell),false
  end
end

function get_merge_indices_collection(lake::Lake,merge_indices_index::Int64,
                                      is_flood_merge::Bool)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  if is_flood_merge
    return lake_parameters.
           flood_merge_and_redirect_indices_collections[merge_indices_index]
  else
    return lake_parameters.
           connect_merge_and_redirect_indices_collections[merge_indices_index]
  end
end

function check_if_merge_is_possible(lake::Lake,merge_indices::MergeAndRedirectIndices)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  merge_type::MergeTypes = get_merge_type(merge_indices)
  if merge_indices.merged
    return false,true,merge_type
  end
  target_cell::Coords = get_merge_target_coords(merge_indices)
  other_lake_number::Int64 = lake_fields.lake_numbers(target_cell)
  if other_lake_number == 0
    return false,false,merge_type
  end
  other_lake::Lake = lake_variables.other_lakes[other_lake_number]
  other_lake = find_true_primary_lake(other_lake)
  if (get_lake_variables(other_lake)::LakeVariables).lake_number == lake_variables.lake_number
    return false,true,merge_type
  elseif isa(other_lake,OverflowingLake)
    other_lake_target_lake_number::Int64 = lake_fields.lake_numbers(other_lake.next_merge_target_coords)
    if other_lake_target_lake_number == 0
      return false,false,merge_type
    end
    other_lake_target_lake::Lake = lake_variables.other_lakes[other_lake_target_lake_number]
    other_lake_target_lake = find_true_primary_lake(other_lake_target_lake)
    if (get_lake_variables(other_lake_target_lake)::LakeVariables).lake_number == lake_variables.lake_number
      return true,false,merge_type
    else
      return false,false,merge_type
    end
  else
    return false,false,merge_type
  end
end

function perform_primary_merge(lake::FillingLake,merge_indices::MergeAndRedirectIndices)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  if debug
    println("Lake $(lake_variables.lake_number) performing primary merge")
  end
  target_cell::Coords = get_merge_target_coords(merge_indices)
  merge_indices.merged = true
  other_lake_number::Integer = lake_fields.lake_numbers(target_cell)
  other_lake::Lake = lake_variables.other_lakes[other_lake_number]
  other_lake = find_true_primary_lake(other_lake)
  lake_variables.secondary_lake_volume += (other_lake.lake_variables.lake_volume +
    other_lake.lake_variables.secondary_lake_volume)
  lake_variables.secondary_number_of_flooded_cells += (other_lake.lake_variables.number_of_flooded_cells +
    other_lake.lake_variables.secondary_number_of_flooded_cells)
  other_lake_number = other_lake.lake_variables.lake_number
  accept_merge::AcceptMerge = AcceptMerge(lake_variables.center_cell,
                                          lake_variables.lake_number)
  lake_variables.other_lakes[other_lake_number] =
    handle_event(other_lake,accept_merge)
end

function perform_secondary_merge(lake::FillingLake,merge_indices::MergeAndRedirectIndices)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  if debug
    println("Lake $(lake_variables.lake_number) performing secondary merge")
  end
  target_cell::Coords = get_merge_target_coords(merge_indices)
  local surface_model_coords::Coords
  if ! (lake_fields.connected_lake_cells(lake_variables.current_cell_to_fill) ||
        lake_parameters.flood_only(lake_variables.current_cell_to_fill))
    set!(lake_fields.connected_lake_cells,
         lake_variables.current_cell_to_fill,true)
  elseif ! (lake_fields.flooded_lake_cells(lake_variables.current_cell_to_fill))
    set!(lake_fields.flooded_lake_cells,
         lake_variables.current_cell_to_fill,true)
    lake_variables.number_of_flooded_cells += 1
  end
  other_lake_number::Integer = lake_fields.lake_numbers(target_cell)
  other_lake::Lake = lake_variables.other_lakes[other_lake_number]
  other_lake = find_true_primary_lake(other_lake)
  other_lake.lake_variables.secondary_lake_volume += (lake_variables.lake_volume +
    lake_variables.secondary_lake_volume)
  other_lake.lake_variables.secondary_number_of_flooded_cells +=
    (lake_variables.number_of_flooded_cells +
     lake_variables.secondary_number_of_flooded_cells)
  other_lake_number = other_lake.lake_variables.lake_number
  other_lake_current_cell_to_fill::Coords = other_lake.lake_variables.current_cell_to_fill
  if ! (lake_fields.flooded_lake_cells(other_lake_current_cell_to_fill) ||
        lake_parameters.flood_only(other_lake_current_cell_to_fill))
    set!(lake_fields.connected_lake_cells,other_lake_current_cell_to_fill,false)
  else
    set!(lake_fields.flooded_lake_cells,other_lake_current_cell_to_fill,false)
    other_lake.lake_variables.number_of_flooded_cells -= 1
  end
  other_lake_as_filling_lake::FillingLake = change_to_filling_lake(other_lake)
  lake_variables.other_lakes[other_lake_number] = other_lake_as_filling_lake
  accept_merge::AcceptMerge =
    AcceptMerge(get_lake_variables(other_lake_as_filling_lake).center_cell,
                other_lake_number)
  return handle_event(lake,accept_merge)
end

function conditionally_rollback_primary_merge(lake::Lake,merge_indices::MergeAndRedirectIndices,
                                              other_lake_outflow::Float64)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  target_cell::Coords = get_merge_target_coords(merge_indices)
  other_lake_number::Integer = lake_fields.lake_numbers(target_cell)
  other_lake::Lake = lake_variables.other_lakes[other_lake_number]
  if ! isa(other_lake,SubsumedLake)
    return false
  end
  other_lake = find_true_rolledback_primary_lake(other_lake)
  if find_true_primary_lake(other_lake).lake_variables.lake_number != lake_variables.lake_number
    return false
  end
  if debug
    println("Lake $(lake_variables.lake_number) rolling back primary merge")
  end
  merge_indices.merged = false
  lake_variables.secondary_lake_volume -=
    (other_lake.lake_variables.lake_volume +
     other_lake.lake_variables.secondary_lake_volume)
  lake_variables.secondary_number_of_flooded_cells -=
    (other_lake.lake_variables.number_of_flooded_cells +
     other_lake.lake_variables.secondary_number_of_flooded_cells)
  other_lake_number = other_lake.lake_variables.lake_number
  accept_split::AcceptSplit = AcceptSplit(lake_variables.lake_number)
  other_lake = handle_event(other_lake,accept_split)
  lake_variables.other_lakes[other_lake_number] = other_lake
  merge_indices_index::Int64,is_flood_merge::Bool = get_merge_indices_index(other_lake)
  if merge_indices_index != 0
    merge_indices_collection::MergeAndRedirectIndicesCollection =
      get_merge_indices_collection(other_lake,merge_indices_index,
                                   is_flood_merge)
    for merge_indices::MergeAndRedirectIndices in
          Iterators.reverse(merge_indices_collection)
      merge_type::MergeTypes = get_merge_type(merge_indices)
      if merge_type == primary_merge
        conditionally_rollback_primary_merge(other_lake,merge_indices,0.0)
      elseif merge_type == secondary_merge
        if merge_indices.merged
          error("Merge logic failure")
        end
      else
          error("Unrecognised merge type")
      end
    end
  end
  remove_water::RemoveWater = RemoveWater(other_lake_outflow)
  lake_variables.other_lakes[other_lake_number] =
    handle_event(other_lake,remove_water)
  return true
end

function change_to_filling_lake(lake::OverflowingLake)
  lake.lake_variables.unprocessed_water +=
    lake.overflowing_lake_variables.excess_water
  filling_lake::FillingLake = FillingLake(lake.lake_parameters,
                                          lake.lake_variables,
                                          lake.lake_fields)
  return filling_lake
end

function change_to_filling_lake(lake::SubsumedLake)
  filling_lake::FillingLake = FillingLake(lake.lake_parameters,
                                          lake.lake_variables,
                                          lake.lake_fields)
end

function update_filling_cell(lake::FillingLake)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  coords::Coords = lake_variables.current_cell_to_fill
  local new_coords::Coords
  if lake_fields.connected_lake_cells(coords) || lake_parameters.flood_only(coords)
    new_coords = get_flood_next_cell_coords(lake_parameters,coords)
  else
    new_coords = get_connect_next_cell_coords(lake_parameters,coords)
  end
  old_lake_number::Int64 = lake_fields.lake_numbers(new_coords)
  if old_lake_number != 0 &&
     old_lake_number != lake_variables.lake_number
     set!(lake_fields.buried_lake_numbers,new_coords,old_lake_number)
  end
  set!(lake_fields.lake_numbers,new_coords,lake_variables.lake_number)
  if ! (lake_fields.connected_lake_cells(coords) ||
        lake_parameters.flood_only(coords))
    set!(lake_fields.connected_lake_cells,coords,true)
  elseif ! lake_fields.flooded_lake_cells(coords)
    set!(lake_fields.flooded_lake_cells,coords,true)
    lake_variables.number_of_flooded_cells += 1
  end
  if (lake_fields.connected_lake_cells(new_coords) ||
      lake_parameters.flood_only(new_coords)) &&
     ! lake_fields.flooded_lake_cells(new_coords)
    surface_model_new_coords::Coords =
      get_corresponding_surface_model_grid_cell(new_coords,
                                                lake_parameters.grid_specific_lake_parameters)
    set!(lake_fields.number_lake_cells,surface_model_new_coords,
         lake_fields.number_lake_cells(surface_model_new_coords) + 1)
  end
  push!(lake_variables.filled_lake_cells,coords)
  lake_variables.previous_cell_to_fill = coords
  lake_variables.current_cell_to_fill = new_coords
end

function rollback_filling_cell(lake::FillingLake)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  coords::Coords = lake_variables.current_cell_to_fill
  new_coords::Coords = lake_variables.previous_cell_to_fill
  if lake_parameters.flood_only(coords) ||
     ! lake_fields.connected_lake_cells(coords)
    set!(lake_fields.lake_numbers,coords,0)
  elseif lake_fields.buried_lake_numbers(coords) != 0
    set!(lake_fields.lake_numbers,coords,lake_fields.buried_lake_numbers(coords))
    set!(lake_fields.buried_lake_numbers,coords,0)
  end
  if ! (lake_fields.flooded_lake_cells(new_coords) ||
        lake_parameters.flood_only(new_coords))
    set!(lake_fields.connected_lake_cells,new_coords,false)
  else
    set!(lake_fields.flooded_lake_cells,new_coords,false)
    lake_variables.number_of_flooded_cells -= 1
  end
  if lake_fields.connected_lake_cells(coords) ||
     lake_parameters.flood_only(coords)
    surface_model_coords::Coords =
      get_corresponding_surface_model_grid_cell(coords,
                                                lake_parameters.grid_specific_lake_parameters)
    set!(lake_fields.number_lake_cells,surface_model_coords,
         lake_fields.number_lake_cells(surface_model_coords) - 1)
  end
  lake_variables.current_cell_to_fill = new_coords
  pop!(lake_variables.filled_lake_cells)
  if size(lake_variables.filled_lake_cells,1) != 0
    lake_variables.previous_cell_to_fill = lake_variables.filled_lake_cells[end]
  else
    lake_variables.previous_cell_to_fill = new_coords
  end
end


function find_true_primary_lake(lake::Lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  if isa(lake,SubsumedLake)
    true_primary_lake_number::Int64 =
      find_root(lake_fields.set_forest,lake.primary_lake_number)
    return lake_variables.other_lakes[true_primary_lake_number]
  else
    return lake
  end
end

function find_true_rolledback_primary_lake(lake::SubsumedLake)
  next_lake::Lake = lake.lake_variables.other_lakes[lake.primary_lake_number]
  if isa(next_lake,SubsumedLake)
    return find_true_rolledback_primary_lake(next_lake)
  else
    return lake
  end
end

function get_flood_next_cell_coords(lake_parameters::LakeParameters,initial_coords::Coords)
  throw(UserError())
end

function get_connect_next_cell_coords(lake_parameters::LakeParameters,initial_coords::Coords)
  throw(UserError())
end

function get_merge_target_coords(merge_indices::MergeAndRedirectIndices)
  throw(UserError())
end

function get_outflow_redirect_coords(merge_indices::MergeAndRedirectIndices)
  throw(UserError())
end

function get_corresponding_surface_model_grid_cell(coords::Coords,
                                                   grid_specific_lake_parameters::GridSpecificLakeParameters)
 throw(UsersError())
end

function get_flood_next_cell_coords(lake_parameters::LakeParameters,initial_coords::LatLonCoords)
  return LatLonCoords(
    lake_parameters.grid_specific_lake_parameters.flood_next_cell_lat_index(initial_coords),
    lake_parameters.grid_specific_lake_parameters.flood_next_cell_lon_index(initial_coords))
end

function get_flood_next_cell_coords(lake_parameters::LakeParameters,initial_coords::Generic1DCoords)
  return Generic1DCoords(
    lake_parameters.grid_specific_lake_parameters.flood_next_cell_index(initial_coords))
end

function get_connect_next_cell_coords(lake_parameters::LakeParameters,initial_coords::LatLonCoords)
  return LatLonCoords(
    lake_parameters.grid_specific_lake_parameters.connect_next_cell_lat_index(initial_coords),
    lake_parameters.grid_specific_lake_parameters.connect_next_cell_lon_index(initial_coords))
end

function get_connect_next_cell_coords(lake_parameters::LakeParameters,initial_coords::Generic1DCoords)
  return Generic1DCoords(
    lake_parameters.grid_specific_lake_parameters.connect_next_cell_index(initial_coords))
end

function get_merge_target_coords(merge_indices::LatLonMergeAndRedirectIndices)
  return LatLonCoords(merge_indices.merge_target_lat_index,
                      merge_indices.merge_target_lon_index)
end

function get_merge_target_coords(merge_indices::UnstructuredMergeAndRedirectIndices)
  return Generic1DCoords(merge_indices.merge_target_cell_index)
end

function get_outflow_redirect_coords(merge_indices::LatLonMergeAndRedirectIndices)
  return LatLonCoords(merge_indices.redirect_lat_index,
                      merge_indices.redirect_lon_index)::LatLonCoords,
         merge_indices.local_redirect::Bool
end

function get_outflow_redirect_coords(merge_indices::UnstructuredMergeAndRedirectIndices)
  return Generic1DCoords(merge_indices.redirect_cell_index)::Generic1DCoords,
         merge_indices.local_redirect::Bool
end

function get_corresponding_surface_model_grid_cell(coords::LatLonCoords,
                                                   grid_specific_lake_parameters::LatLonLakeParameters)
  return LatLonCoords(grid_specific_lake_parameters.corresponding_surface_cell_lat_index(coords),
                      grid_specific_lake_parameters.corresponding_surface_cell_lon_index(coords))
end

function get_corresponding_surface_model_grid_cell(coords::Generic1DCoords,
                                                   grid_specific_lake_parameters::UnstructuredLakeParameters)
  return Generic1DCoords(grid_specific_lake_parameters.corresponding_surface_cell_index(coords))
end

function set_effective_lake_height_on_surface_grid_to_lakes(lake_parameters::LakeParameters,
                                                            lake_fields::LakeFields,
                                                            effective_lake_height_on_surface_grid::Field{Float64})
  for_all(lake_parameters.surface_model_grid) do coords::Coords
    set!(lake_fields.effective_lake_height_on_surface_grid_to_lakes,coords,
         effective_lake_height_on_surface_grid(coords))
  end
end

function calculate_lake_fraction_on_surface_grid(lake_parameters::LakeParameters,
                                                 lake_fields::LakeFields)
  return elementwise_divide(lake_fields.number_lake_cells,
                            lake_parameters.number_fine_grid_cells)::Field{Float64}
end

function calculate_effective_lake_height_on_surface_grid(lake_parameters::LakeParameters,
                                                         lake_fields::LakeFields)
  return elementwise_divide(lake_fields.effective_volume_per_cell_on_surface_grid,
                            lake_parameters.cell_areas_on_surface_model_grid)::Field{Float64}
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      print_results::PrintResults)
  lake_parameters::LakeParameters = get_lake_parameters(prognostic_fields)
  lake_prognostics::LakePrognostics = get_lake_prognostics(prognostic_fields)
  lake_fields::LakeFields = get_lake_fields(prognostic_fields)
  println("Timestep: $(print_results.timestep)")
  print_river_results(prognostic_fields)
  println("")
  println("Water to HD")
  println(lake_fields.water_to_hd)
  for lake in lake_prognostics.lakes
    println("Lake Center: $(lake.lake_variables.center_cell) "*
            "Lake Volume: $(lake.lake_variables.lake_volume)")
  end
  println("")
  println("Diagnostic Lake Volumes:")
  print_lake_types(lake_parameters.grid,lake_prognostics,lake_fields)
  println("")
  diagnostic_lake_volumes::Field{Float64} = calculate_diagnostic_lake_volumes_field(lake_parameters,
                                                                                    lake_fields,
                                                                                    lake_prognostics)
  println(diagnostic_lake_volumes)
  return prognostic_fields
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      print_selected_lakes::PrintSelectedLakes)
  lake_prognostics::LakePrognostics = get_lake_prognostics(prognostic_fields)
  for lake_number in print_selected_lakes.lakes_to_print

    if length(lake_prognostics.lakes) >= lake_number
      lake::Lake = lake_prognostics.lakes[lake_number]
      println(lake)
    end
  end
  return prognostic_fields
end

function print_lake_types(grid::LatLonGrid,lake_prognostics::LakePrognostics,
                          lake_fields::LakeFields)
  for_all_with_line_breaks(grid) do coords::Coords
    print_lake_type(coords,lake_prognostics,lake_fields)
  end
end

function print_lake_type(coords::Coords,lake_prognostics::LakePrognostics,
                         lake_fields::LakeFields)
  lake_number::Int64 = lake_fields.lake_numbers(coords)
  if lake_number == 0
    print("- ")
  else
    lake::Lake = lake_prognostics.lakes[lake_number]
    if isa(lake,FillingLake)
      print("F ")
    elseif isa(lake,OverflowingLake)
      print("O ")
    elseif isa(lake,SubsumedLake)
      print("S ")
    else
      print("U ")
    end
  end
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      print_results::PrintSection)
  lake_parameters::LakeParameters = get_lake_parameters(prognostic_fields)
  lake_prognostics::LakePrognostics = get_lake_prognostics(prognostic_fields)
  lake_fields::LakeFields = get_lake_fields(prognostic_fields)
  println("")
  print_lake_types_section(lake_parameters.grid,lake_prognostics,lake_fields)
  return prognostic_fields
end

function print_lake_types_section(grid::LatLonGrid,lake_prognostics::LakePrognostics,
                                  lake_fields::LakeFields)
  section_coords::LatLonSectionCoords = LatLonSectionCoords(225,295,530,580)
  for_section_with_line_breaks(grid,section_coords) do coords::Coords
    print_lake_type(coords,lake_prognostics,lake_fields)
  end
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      write_lake_numbers::WriteLakeNumbers)
  lake_parameters::LakeParameters = get_lake_parameters(prognostic_fields)
  lake_fields::LakeFields = get_lake_fields(prognostic_fields)
  write_lake_numbers_field(lake_parameters,lake_fields,timestep=write_lake_numbers.timestep)
  return prognostic_fields
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      write_lake_volumes::WriteLakeVolumes)
  lake_parameters::LakeParameters = get_lake_parameters(prognostic_fields)
  lake_prognostics::LakePrognostics = get_lake_prognostics(prognostic_fields)
  lake_volumes::Field{Float64} = Field{Float64}(lake_parameters.grid,0.0)
  for lake::Lake in lake_prognostics.lakes
    lake_variables::LakeVariables = get_lake_variables(lake)
    lake_center_cell::Coords = lake_variables.center_cell
    set!(lake_volumes,lake_center_cell,lake_variables.lake_volume +
                                      lake_variables.unprocessed_water)
  end
  write_lake_volumes_field(lake_parameters,lake_volumes)
  return prognostic_fields
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      write_diagnostic_lake_volumes::WriteDiagnosticLakeVolumes)
  lake_parameters::LakeParameters = get_lake_parameters(prognostic_fields)
  lake_fields::LakeFields = get_lake_fields(prognostic_fields)
  lake_prognostics::LakePrognostics = get_lake_prognostics(prognostic_fields)
  diagnostic_lake_volumes::Field{Float64} = calculate_diagnostic_lake_volumes_field(lake_parameters,
                                                                                    lake_fields,
                                                                                    lake_prognostics)
  write_diagnostic_lake_volumes_field(lake_parameters,diagnostic_lake_volumes,
                                      timestep=write_diagnostic_lake_volumes.timestep)
  return prognostic_fields
end

function calculate_diagnostic_lake_volumes_field(lake_parameters::LakeParameters,
                                                 lake_fields::LakeFields,
                                                 lake_prognostics::LakePrognostics)
  lake_volumes_by_lake_number::Vector{Float64} = zeros(Float64,size(lake_prognostics.lakes,1))
  diagnostic_lake_volumes::Field{Float64} = Field{Float64}(lake_parameters.grid,0.0)
  for lake::Lake in lake_prognostics.lakes
    lake_variables::LakeVariables = get_lake_variables(lake)
    original_lake_index::Int64 = lake_variables.lake_number
    if isa(lake,SubsumedLake)
      lake = find_true_primary_lake(lake)
    end
    lake_volumes_by_lake_number[original_lake_index] =
      lake.lake_variables.lake_volume + lake.lake_variables.secondary_lake_volume +
      lake.lake_variables.unprocessed_water
    if isa(lake,OverflowingLake)
      lake_volumes_by_lake_number[original_lake_index] +=
        lake.overflowing_lake_variables.excess_water
    end
  end
  for_all(lake_parameters.grid) do coords::Coords
    lake_number = lake_fields.lake_numbers(coords)
    if (lake_number > 0)
      set!(diagnostic_lake_volumes,coords,lake_volumes_by_lake_number[lake_number])
    end
  end
  return diagnostic_lake_volumes
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      set_lake_evaporation::SetLakeEvaporation)
  lake_parameters::LakeParameters = get_lake_parameters(prognostic_fields)
  lake_fields::LakeFields = get_lake_fields(prognostic_fields)
  old_effective_lake_height_on_surface_grid::Field{Float64} =
    calculate_effective_lake_height_on_surface_grid(lake_parameters,lake_fields)
  effective_lake_height_on_surface_grid::Field{Float64} =
    old_effective_lake_height_on_surface_grid -
    elementwise_divide(set_lake_evaporation.lake_evaporation,
                       lake_parameters.cell_areas_on_surface_model_grid)
  set_effective_lake_height_on_surface_grid_to_lakes(lake_parameters,lake_fields,
                                                     effective_lake_height_on_surface_grid)
  return prognostic_fields
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      set_lake_evaporation::SetRealisticLakeEvaporation)
  lake_parameters::LakeParameters = get_lake_parameters(prognostic_fields)
  lake_fields::LakeFields = get_lake_fields(prognostic_fields)
  old_effective_lake_height_on_surface_grid::Field{Float64} =
    calculate_effective_lake_height_on_surface_grid(lake_parameters,lake_fields)
  effective_lake_height_on_surface_grid::Field{Float64} = old_effective_lake_height_on_surface_grid
  for_all(lake_parameters.surface_model_grid) do coords::Coords
    if old_effective_lake_height_on_surface_grid(coords) > 0.0
      working_effective_lake_height_on_surface_grid::Float64 =
        old_effective_lake_height_on_surface_grid(coords) -
        set_lake_evaporation.height_of_water_evaporated(coords)
      if working_effective_lake_height_on_surface_grid > 0.0
        set!(effective_lake_height_on_surface_grid,coords,
             working_effective_lake_height_on_surface_grid)
      else
        set!(effective_lake_height_on_surface_grid,coords,0.0)
      end
    end
  end
  set_effective_lake_height_on_surface_grid_to_lakes(lake_parameters,lake_fields,
                                                     effective_lake_height_on_surface_grid)
  return prognostic_fields
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      check_water_budget::CheckWaterBudget)
  lake_prognostics::LakePrognostics = get_lake_prognostics(prognostic_fields)
  lake_fields::LakeFields = get_lake_fields(prognostic_fields)
  lake_diagnostic_variables = get_lake_diagnostic_variables(prognostic_fields)
  new_total_lake_volume::Float64 = 0.0
  for lake::Lake in lake_prognostics.lakes
    lake_variables::LakeVariables = get_lake_variables(lake)
    new_total_lake_volume += lake_variables.unprocessed_water + lake_variables.lake_volume
    if isa(lake,OverflowingLake)
      new_total_lake_volume += lake.overflowing_lake_variables.excess_water
    end
  end
  change_in_total_lake_volume::Float64 = new_total_lake_volume -
                                         lake_diagnostic_variables.total_lake_volume -
                                         check_water_budget.
                                         total_initial_water_volume
  total_water_to_lakes::Float64 = sum(lake_fields.water_to_lakes)
  total_inflow_minus_outflow::Float64 = total_water_to_lakes +
                                        sum(lake_fields.lake_water_from_ocean) -
                                        sum(lake_fields.water_to_hd) -
                                        sum(lake_fields.evaporation_from_lakes)
  difference::Float64 = change_in_total_lake_volume - total_inflow_minus_outflow
  tolerance::Float64 = 10e-15*max(new_total_lake_volume, total_water_to_lakes)
  if ! isapprox(difference,0,atol=tolerance)
    println("*** Lake Water Budget ***")
    println("Total lake volume: $(new_total_lake_volume)")
    println("Total inflow - outflow: $(total_inflow_minus_outflow)")
    println("Change in lake volume: $(change_in_total_lake_volume)")
    println("Difference: $(difference)")
    println("Total water to lakes: $total_water_to_lakes")
    println("Total water from ocean: "*string(sum(lake_fields.lake_water_from_ocean)))
    println("Total water to HD model from lakes: "*string(sum(lake_fields.water_to_hd)))
    println("Old total lake volume: "*string(lake_diagnostic_variables.total_lake_volume))
  end
  lake_diagnostic_variables.total_lake_volume = new_total_lake_volume
  return prognostic_fields
end

function show(io::IO,lake::Lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  println(io,"-----------------------------------------------")
  println(io,"Center cell: "*string(lake_variables.center_cell))
  println(io,"Lake volume: "*string(lake_variables.lake_volume))
  println(io,"Secondary lake volume: "*string(lake_variables.secondary_lake_volume))
  println(io,"Unprocessed water: "*string(lake_variables.unprocessed_water))
  println(io,"Current cell to fill: "*string(lake_variables.current_cell_to_fill))
  println(io,"Previous cell to fill: "*string(lake_variables.previous_cell_to_fill))
  println(io,"Lake number: "*string(lake_variables.lake_number))
  println(io,"Center cell coarse coords: "*string(lake_variables.center_cell_coarse_coords))
  println(io,"Number of flooded cells: "*string(lake_variables.number_of_flooded_cells))
  println(io,"Secondary number of flooded cells: "*string(lake_variables.secondary_number_of_flooded_cells))
  if isa(lake,OverflowingLake)
    println(io,"O")
    println(io,"Excess water: "*string(lake.overflowing_lake_variables.excess_water))
  elseif isa(lake,FillingLake)
    println(io,"F")
  elseif isa(lake,SubsumedLake)
    println(io,"S")
    println(io,"Primary lake number: "*string(lake.primary_lake_number))
  end
end

end
