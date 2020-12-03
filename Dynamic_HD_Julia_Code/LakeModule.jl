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
using FieldModule: Field, set!
import HDModule: water_to_lakes,water_from_lakes
import HierarchicalStateMachineModule: handle_event

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
end

struct AcceptSplit <: Event end

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

struct CheckWaterBudget <: Event end

abstract type GridSpecificLakeParameters end

struct LakeParameters
  lake_centers::Field{Bool}
  connection_volume_thresholds::Field{Float64}
  flood_volume_thresholds::Field{Float64}
  flood_only::Field{Bool}
  flood_local_redirect::Field{Bool}
  connect_local_redirect::Field{Bool}
  additional_flood_local_redirect::Field{Bool}
  additional_connect_local_redirect::Field{Bool}
  merge_points::Field{MergeTypes}
  basin_numbers::Field{Int64}
  basins::Vector{Vector{Coords}}
  grid::Grid
  hd_grid::Grid
  grid_specific_lake_parameters::GridSpecificLakeParameters
  lake_model_parameters::LakeModelParameters
  function LakeParameters(lake_centers::Field{Bool},
                          connection_volume_thresholds::Field{Float64},
                          flood_volume_thresholds::Field{Float64},
                          flood_local_redirect::Field{Bool},
                          connect_local_redirect::Field{Bool},
                          additional_flood_local_redirect::Field{Bool},
                          additional_connect_local_redirect::Field{Bool},
                          merge_points::Field{MergeTypes},
                          grid::Grid,
                          hd_grid::Grid,
                          grid_specific_lake_parameters::GridSpecificLakeParameters)
    flood_only::Field{Bool} =  Field{Bool}(grid,false)
    for_all(grid) do coords::Coords
      if connection_volume_thresholds(coords) == -1.0
        set!(flood_only,coords,true)
      end
    end
    basins::Vector{Vector{Coords}} = Vector{Vector{Coords}}[]
    basin_numbers::Field{Int64} = Field{Int64}(hd_grid)
    basin_number::Int64 = 1
    for_all(hd_grid) do coords::Coords
      basins_in_coarse_cell::Vector{Coords} = Vector{Coords}[]
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
               flood_local_redirect,
               connect_local_redirect,
               additional_flood_local_redirect,
               additional_connect_local_redirect,
               merge_points,
               basin_numbers,
               basins,grid,hd_grid,
               grid_specific_lake_parameters,
               LakeModelParameters())
  end
end

struct LatLonLakeParameters <: GridSpecificLakeParameters
  flood_next_cell_lat_index::Field{Int64}
  flood_next_cell_lon_index::Field{Int64}
  connect_next_cell_lat_index::Field{Int64}
  connect_next_cell_lon_index::Field{Int64}
  flood_force_merge_lat_index::Field{Int64}
  flood_force_merge_lon_index::Field{Int64}
  connect_force_merge_lat_index::Field{Int64}
  connect_force_merge_lon_index::Field{Int64}
  flood_redirect_lat_index::Field{Int64}
  flood_redirect_lon_index::Field{Int64}
  connect_redirect_lat_index::Field{Int64}
  connect_redirect_lon_index::Field{Int64}
  additional_flood_redirect_lat_index::Field{Int64}
  additional_flood_redirect_lon_index::Field{Int64}
  additional_connect_redirect_lat_index::Field{Int64}
  additional_connect_redirect_lon_index::Field{Int64}
end

struct UnstructuredLakeParameters <: GridSpecificLakeParameters
  flood_next_cell_index::Field{Int64}
  connect_next_cell_index::Field{Int64}
  flood_force_merge_index::Field{Int64}
  connect_force_merge_index::Field{Int64}
  flood_redirect_index::Field{Int64}
  connect_redirect_index::Field{Int64}
  additional_flood_redirect_index::Field{Int64}
  additional_connect_redirect_index::Field{Int64}
end

struct LakeFields
  completed_lake_cells::Field{Bool}
  lake_numbers::Field{Int64}
  water_to_lakes::Field{Float64}
  water_to_hd::Field{Float64}
  lake_water_from_ocean::Field{Float64}
  cells_with_lakes::Vector{Coords}
  function LakeFields(lake_parameters::LakeParameters)
    completed_lake_cells = Field{Bool}(lake_parameters.grid,false)
    lake_numbers = Field{Int64}(lake_parameters.grid,0)
    water_to_lakes = Field{Float64}(lake_parameters.hd_grid,0.0)
    water_to_hd = Field{Float64}(lake_parameters.hd_grid,0.0)
    lake_water_from_ocean = Field{Float64}(lake_parameters.hd_grid,0.0)
    cells_with_lakes::Vector{Coords} = Vector{Coords}[]
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
    new(completed_lake_cells,lake_numbers,water_to_lakes,water_to_hd,lake_water_from_ocean,
        cells_with_lakes)
  end
end

struct LakePrognostics
  lakes::Vector{Lake}
  function LakePrognostics(lake_parameters::LakeParameters,
                           lake_fields::LakeFields)
    lakes::Vector{Lake} = Vector{Lake}[]
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
        lake_number += 1
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
  LakeVariables(center_cell::Coords,
                other_lakes::Vector{Lake},
                lake_number::Int64,
                center_cell_coarse_coords::Coords) =
    new(center_cell,0.0,0.0,0.0,deepcopy(center_cell),deepcopy(center_cell),
        other_lakes,lake_number,Vector{Coords}[],center_cell_coarse_coords)
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

mutable struct FillingLakeVariables
  primary_merge_completed::Bool
  use_additional_fields::Bool
end

struct FillingLake <: Lake
  lake_parameters::LakeParameters
  lake_variables::LakeVariables
  lake_fields::LakeFields
  filling_lake_variables::FillingLakeVariables
end

FillingLake(lake_parameters::LakeParameters,
            lake_variables::LakeVariables,
            lake_fields::LakeFields) =
  FillingLake(lake_parameters,
              lake_variables,
              lake_fields,
              FillingLakeVariables(false,false))

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

function water_to_lakes(prognostic_fields::RiverAndLakePrognosticFields,coords::Coords,
                        inflow::Float64,step_length::Float64)
  lake_fields = get_lake_fields(prognostic_fields)
  set!(lake_fields.water_to_lakes,coords,inflow*step_length)
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
  local basins_in_cell::Array{Coords,1}
  for coords::Coords in lake_fields.cells_with_lakes
    if lake_fields.water_to_lakes(coords) > 0.0
      basins_in_cell =
        lake_parameters.basins[lake_parameters.basin_numbers(coords)]
      share_to_each_lake = lake_fields.water_to_lakes(coords)/length(basins_in_cell)
      add_water::AddWater = AddWater(share_to_each_lake)
      for basin_center::Coords in basins_in_cell
        lake_index::Int64 = lake_fields.lake_numbers(basin_center)
        lake::Lake = lake_prognostics.lakes[lake_index]
        lake_prognostics.lakes[lake_index] = handle_event(lake,add_water)
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
    if lake_variables.lake_volume < 0.0
      lake_prognostics.lakes[lake_index] = handle_event(lake,ReleaseNegativeWater())
    end
    if isa(lake,OverflowingLake)
      handle_event(lake,drain_excess_water)
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
      merge_type::SimpleMergeTypes = get_merge_type(lake)
      if merge_type != no_merge
        while true
          if ! (merge_type == primary_merge &&
                lake.filling_lake_variables.primary_merge_completed)
            if (merge_type == double_merge &&
                lake.filling_lake_variables.primary_merge_completed)
              merge_type = secondary_merge
              lake.filling_lake_variables.use_additional_fields = true
            end
            local merge_possible::Bool
            local already_merged::Bool
            merge_possible,already_merged = check_if_merge_is_possible(lake,merge_type)
            local subsumed_lake::SubsumedLake
            if merge_possible
              if merge_type == secondary_merge
                subsumed_lake = perform_secondary_merge(lake)
                subsumed_lake = handle_event(subsumed_lake,
                                             StoreWater(inflow))
                return subsumed_lake
              else
                perform_primary_merge(lake)
              end
            elseif ! already_merged
              overflowing_lake::Lake = change_to_overflowing_lake(lake,merge_type)
              overflowing_lake = handle_event(overflowing_lake,StoreWater(inflow))
              return overflowing_lake
            elseif merge_type == secondary_merge
              target_cell::Coords = get_secondary_merge_coords(lake)
              other_lake_number::Integer = lake.lake_fields.lake_numbers(target_cell)
              other_lake::SubsumedLake = lake.lake_variables.other_lakes[other_lake_number]
              other_lake_as_filling_lake::FillingLake = change_to_filling_lake(other_lake)
              lake.lake_variables.other_lakes[other_lake_number] =
                change_to_overflowing_lake(other_lake_as_filling_lake,primary_merge)
              subsumed_lake = perform_secondary_merge(lake)
              subsumed_lake = handle_event(subsumed_lake,
                                             StoreWater(inflow))
              return subsumed_lake
            end
          end
          if merge_type != double_merge
            break
          end
        end
      end
      update_filling_cell(lake)
    end
  end
  return lake
end

function handle_event(lake::FillingLake,remove_water::RemoveWater)
  if remove_water.outflow <= lake.lake_variables.unprocessed_water
    lake.lake_variables.unprocessed_water -= remove_water.outflow
    return lake
  end
  outflow::Float64 = remove_water.outflow - lake.lake_variables.unprocessed_water
  lake.lake_variables.unprocessed_water = 0.0
  while outflow > 0.0
    outflow,drained::Bool = drain_current_cell(lake,outflow)
    if drained
      rollback_filling_cell(lake)
      merge_type::SimpleMergeTypes = get_merge_type(lake)
      if merge_type == primary_merge || merge_type == double_merge
        new_outflow::Float64 = outflow/2.0
        rollback_primary_merge(lake,new_outflow)
        outflow -= new_outflow
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
  if lake_parameters.flood_only(lake_variables.current_cell_to_fill) ||
     lake_variables.lake_volume <=
     lake_parameters.connection_volume_thresholds(lake_variables.current_cell_to_fill)
    set!(lake_fields.completed_lake_cells,lake_variables.current_cell_to_fill,false)
  end
  lake_as_filling_lake::FillingLake = change_to_filling_lake(lake)
  merge_type::SimpleMergeTypes = get_merge_type(lake)
  if merge_type == double_merge
    if lake_as_filling_lake.filling_lake_variables.primary_merge_completed
        new_outflow::Float64 = outflow/2.0
        rollback_primary_merge(lake,new_outflow)
        outflow -= new_outflow
    end
  end
  lake_as_filling_lake.filling_lake_variables.primary_merge_completed = false
  lake_as_filling_lake.filling_lake_variables.use_additional_fields = false
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
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  set!(lake_fields.completed_lake_cells,lake_variables.current_cell_to_fill,true)
  subsumed_lake::SubsumedLake = SubsumedLake(get_lake_parameters(lake),
                                             lake_variables,
                                             lake_fields,
                                             lake_fields.
                                             lake_numbers(accept_merge.
                                                          redirect_coords))
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
  if lake_parameters.flood_only(lake_variables.current_cell_to_fill) ||
     lake_variables.lake_volume <=
     lake_parameters.connection_volume_thresholds(lake_variables.current_cell_to_fill)
    set!(lake_fields.completed_lake_cells,lake_variables.current_cell_to_fill,false)
  end
  filling_lake::FillingLake = FillingLake(lake_parameters,
                                          lake_variables,
                                          lake.lake_fields,
                                          FillingLakeVariables(false,
                                                               false))
  return filling_lake
end

function change_to_overflowing_lake(lake::FillingLake,merge_type::SimpleMergeTypes)
  lake_fields::LakeFields = get_lake_fields(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  set!(lake_fields.completed_lake_cells,lake_variables.current_cell_to_fill,true)
  outflow_redirect_coords,local_redirect =
    get_outflow_redirect_coords(lake)
  local next_merge_target_coords::Coords
  if merge_type == secondary_merge
    next_merge_target_coords = get_secondary_merge_coords(lake)
  else
    next_merge_target_coords = get_primary_merge_coords(lake)
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

function fill_current_cell(lake::FillingLake,inflow::Float64)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  current_cell_to_fill::Coords = lake_variables.current_cell_to_fill
  new_lake_volume::Float64 = inflow + lake_variables.lake_volume
  maximum_new_lake_volume::Float64 =
    (lake_fields.completed_lake_cells(current_cell_to_fill) ||
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
    (lake_fields.completed_lake_cells(previous_cell_to_fill) ||
    lake_parameters.flood_only(previous_cell_to_fill)) ?
    lake_parameters.flood_volume_thresholds(previous_cell_to_fill) :
    lake_parameters.connection_volume_thresholds(previous_cell_to_fill)
  if new_lake_volume >= minimum_new_lake_volume
    lake_variables.lake_volume = new_lake_volume
    return 0.0,false
  else
    outflow = minimum_new_lake_volume - new_lake_volume
    lake_variables.lake_volume = minimum_new_lake_volume
    return outflow,true
  end
end

function get_merge_type(lake::Lake)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  current_cell::Coords = lake_variables.current_cell_to_fill
  completed_lake_cell::Bool = lake_fields.completed_lake_cells(current_cell)
  extended_merge_type::MergeTypes = lake_parameters.merge_points(current_cell)
  if lake_fields.completed_lake_cells(current_cell) ||
     lake_parameters.flood_only(current_cell)
    return convert_to_simple_merge_type_flood[Int(extended_merge_type)+1]
  else
    return convert_to_simple_merge_type_connect[Int(extended_merge_type)+1]
  end
end

function check_if_merge_is_possible(lake::Lake,merge_type::SimpleMergeTypes)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  local target_cell::Coords
  if merge_type == secondary_merge
    target_cell = get_secondary_merge_coords(lake)
  else
    target_cell = get_primary_merge_coords(lake)
  end
  if ! lake_fields.completed_lake_cells(target_cell)
    return false,false
  end
  other_lake::Lake = lake_variables.other_lakes[lake_fields.lake_numbers(target_cell)]
  other_lake = find_true_primary_lake(other_lake)
  if (get_lake_variables(other_lake)::LakeVariables).lake_number == lake_variables.lake_number
    return false,true
  elseif isa(other_lake,OverflowingLake)
    other_lake_target_lake_number::Int64 = lake_fields.lake_numbers(other_lake.next_merge_target_coords)
    while true
      if other_lake_target_lake_number == 0
        return false,false
      end
      other_lake_target_lake::Lake = lake_variables.other_lakes[other_lake_target_lake_number]
      if (get_lake_variables(other_lake_target_lake)::LakeVariables).lake_number == lake_variables.lake_number
        return true,false
      else
        if isa(other_lake_target_lake,OverflowingLake)
          other_lake_target_lake_number =
            lake_fields.lake_numbers(other_lake_target_lake.next_merge_target_coords)
        elseif isa(other_lake_target_lake,SubsumedLake)
          other_lake_target_lake_number =
            (get_lake_variables(find_true_primary_lake(other_lake_target_lake)::Lake)
                                ::LakeVariables).lake_number
        else
          return false,false
        end
      end
    end
  else
    return false,false
  end
end

function perform_primary_merge(lake::FillingLake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  target_cell::Coords = get_primary_merge_coords(lake)
  other_lake_number::Integer = lake_fields.lake_numbers(target_cell)
  other_lake::Lake = lake_variables.other_lakes[other_lake_number]
  other_lake = find_true_primary_lake(other_lake)
  lake_variables.secondary_lake_volume += (other_lake.lake_variables.lake_volume +
    other_lake.lake_variables.secondary_lake_volume)
  other_lake_number = other_lake.lake_variables.lake_number
  lake.filling_lake_variables.primary_merge_completed = true
  accept_merge::AcceptMerge = AcceptMerge(lake_variables.center_cell)
  lake_variables.other_lakes[other_lake_number] =
    handle_event(other_lake,accept_merge)
end

function perform_secondary_merge(lake::FillingLake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  set!(lake_fields.completed_lake_cells,
       lake_variables.current_cell_to_fill,true)
  target_cell::Coords = get_secondary_merge_coords(lake)
  other_lake_number::Integer = lake_fields.lake_numbers(target_cell)
  other_lake::Lake = lake_variables.other_lakes[other_lake_number]
  other_lake = find_true_primary_lake(other_lake)
  other_lake.lake_variables.secondary_lake_volume += (lake_variables.lake_volume +
    lake_variables.secondary_lake_volume)
  other_lake_number = other_lake.lake_variables.lake_number
  other_lake_as_filling_lake::FillingLake = change_to_filling_lake(other_lake)
  lake_variables.other_lakes[other_lake_number] = other_lake_as_filling_lake
  accept_merge::AcceptMerge =
    AcceptMerge(get_lake_variables(other_lake_as_filling_lake).center_cell)
  return handle_event(lake,accept_merge)
end

function rollback_primary_merge(lake::FillingLake,other_lake_outflow::Float64)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  target_cell::Coords = get_primary_merge_coords(lake)
  other_lake_number::Integer = lake_fields.lake_numbers(target_cell)
  other_lake::Lake = lake_variables.other_lakes[other_lake_number]
  other_lake = find_true_rolledback_primary_lake(other_lake)
  lake_variables.secondary_lake_volume -=
    (other_lake.lake_variables.lake_volume +
     other_lake.lake_variables.secondary_lake_volume)
  other_lake_number = other_lake.lake_variables.lake_number
  accept_split::AcceptSplit = AcceptSplit()
  lake_variables.other_lakes[other_lake_number] =
    handle_event(other_lake,accept_split)
  other_lake = lake_variables.other_lakes[other_lake_number]
  remove_water::RemoveWater = RemoveWater(other_lake_outflow)
  lake_variables.other_lakes[other_lake_number] =
    handle_event(other_lake,remove_water)
end

function change_to_filling_lake(lake::OverflowingLake)
  lake.lake_variables.unprocessed_water +=
    lake.overflowing_lake_variables.excess_water
  use_additional_fields = get_merge_type(lake) == double_merge
  filling_lake::FillingLake = FillingLake(lake.lake_parameters,
                                          lake.lake_variables,
                                          lake.lake_fields,
                                          FillingLakeVariables(true,
                                                               use_additional_fields))
  return filling_lake
end

function change_to_filling_lake(lake::SubsumedLake)
  filling_lake::FillingLake = FillingLake(lake.lake_parameters,
                                          lake.lake_variables,
                                          lake.lake_fields,
                                          FillingLakeVariables(false,
                                                               false))
end

function update_filling_cell(lake::FillingLake)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  lake.filling_lake_variables.primary_merge_completed = false
  coords::Coords = lake_variables.current_cell_to_fill
  local new_coords::Coords
  if lake_fields.completed_lake_cells(coords) || lake_parameters.flood_only(coords)
    new_coords = get_flood_next_cell_coords(lake_parameters,coords)
  else
    new_coords = get_connect_next_cell_coords(lake_parameters,coords)
  end
  set!(lake_fields.lake_numbers,new_coords,lake_variables.lake_number)
  set!(lake_fields.completed_lake_cells,coords,true)
  push!(lake_variables.filled_lake_cells,coords)
  lake_variables.previous_cell_to_fill = coords
  lake_variables.current_cell_to_fill = new_coords
end

function rollback_filling_cell(lake::FillingLake)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  lake.filling_lake_variables.primary_merge_completed = false
  coords::Coords = lake_variables.current_cell_to_fill
  new_coords::Coords = lake_variables.previous_cell_to_fill
  if lake_parameters.flood_only(coords) ||
     lake_variables.lake_volume <=
     lake_parameters.connection_volume_thresholds(coords)
    set!(lake_fields.lake_numbers,coords,0)
  end
  if lake_parameters.flood_only(new_coords) ||
     lake_variables.lake_volume <=
     lake_parameters.connection_volume_thresholds(new_coords)
    set!(lake_fields.completed_lake_cells,new_coords,false)
  end
  lake_variables.current_cell_to_fill = new_coords
  pop!(lake_variables.filled_lake_cells)
  if size(lake_variables.filled_lake_cells,1) != 0
    lake_variables.previous_cell_to_fill = lake_variables.filled_lake_cells[end]
  else
    lake_variables.previous_cell_to_fill = new_coords
  end
end

function get_primary_merge_coords(lake::Lake)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  current_cell::Coords = lake_variables.current_cell_to_fill
  return get_primary_merge_coords(lake_parameters,
                                  current_cell,
                                  lake_fields.completed_lake_cells(current_cell) ||
                                  lake_parameters.flood_only(current_cell))
end

function get_secondary_merge_coords(lake::Lake)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  current_cell::Coords = lake_variables.current_cell_to_fill
  return get_secondary_merge_coords(lake_parameters,
                                    current_cell,
                                    lake_fields.completed_lake_cells(current_cell) ||
                                    lake_parameters.flood_only(current_cell))
end

function get_outflow_redirect_coords(lake::FillingLake)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  current_cell::Coords = lake_variables.current_cell_to_fill
  completed_lake_cell::Bool = lake_fields.completed_lake_cells(current_cell) ||
                              lake_parameters.flood_only(current_cell)
  use_additional_fields = lake.filling_lake_variables.use_additional_fields
  local local_redirect::Bool
  if use_additional_fields
    local_redirect = completed_lake_cell ?
      lake_parameters.additional_flood_local_redirect(current_cell) :
      lake_parameters.additional_connect_local_redirect(current_cell)
  else
    local_redirect = completed_lake_cell ?
      lake_parameters.flood_local_redirect(current_cell) :
      lake_parameters.connect_local_redirect(current_cell)
  end
  return get_outflow_redirect_coords(lake_parameters,
                                     current_cell,
                                     completed_lake_cell,
                                     use_additional_fields),local_redirect
end

function find_true_primary_lake(lake::Lake)
  if isa(lake,SubsumedLake)
    return find_true_primary_lake(lake.lake_variables.other_lakes[lake.primary_lake_number])
  else
    return lake
  end
end

function find_true_rolledback_primary_lake(lake::Lake)
  next_lake::Lake = lake.lake_variables.other_lakes[lake.primary_lake_number]
  if isa(next_lake,SubsumedLake)
    return find_true_primary_lake(next_lake)
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

function get_primary_merge_coords(lake_parameters::LakeParameters,initial_coords::Coords,
                                  completed_lake_cell::Bool)
  throw(UserError())
end

function get_secondary_merge_coords(lake_parameters::LakeParameters,initial_coords::Coords,
                                    completed_lake_cell::Bool)
  throw(UserError())
end

function get_outflow_redirect_coords(lake_parameters::LakeParameters,
                                     initial_coords::Coords,
                                     use_flood_redirect::Bool,
                                     use_additional_fields::Bool)
  throw(UserError())
end

function get_primary_merge_coords(lake_parameters::LakeParameters,initial_coords::LatLonCoords,
                                  completed_lake_cell::Bool)
  if completed_lake_cell
    return LatLonCoords(
           lake_parameters.grid_specific_lake_parameters.flood_force_merge_lat_index(initial_coords),
           lake_parameters.grid_specific_lake_parameters.flood_force_merge_lon_index(initial_coords))
  else
    return LatLonCoords(
           lake_parameters.grid_specific_lake_parameters.connect_force_merge_lat_index(initial_coords),
           lake_parameters.grid_specific_lake_parameters.connect_force_merge_lon_index(initial_coords))
  end
end

function get_primary_merge_coords(lake_parameters::LakeParameters,initial_coords::Generic1DCoords,
                                  completed_lake_cell::Bool)
  if completed_lake_cell
    return Generic1DCoords(
           lake_parameters.grid_specific_lake_parameters.flood_force_merge_index(initial_coords))
  else
    return Generic1DCoords(
           lake_parameters.grid_specific_lake_parameters.connect_force_merge_index(initial_coords))
  end
end

function get_secondary_merge_coords(lake_parameters::LakeParameters,initial_coords::LatLonCoords,
                                    completed_lake_cell::Bool)
  if completed_lake_cell
    return LatLonCoords(
           lake_parameters.grid_specific_lake_parameters.flood_next_cell_lat_index(initial_coords),
           lake_parameters.grid_specific_lake_parameters.flood_next_cell_lon_index(initial_coords))
  else
    return LatLonCoords(
           lake_parameters.grid_specific_lake_parameters.connect_next_cell_lat_index(initial_coords),
           lake_parameters.grid_specific_lake_parameters.connect_next_cell_lon_index(initial_coords))
  end
end

function get_secondary_merge_coords(lake_parameters::LakeParameters,initial_coords::Generic1DCoords,
                                    completed_lake_cell::Bool)
  if completed_lake_cell
    return Generic1DCoords(
           lake_parameters.grid_specific_lake_parameters.flood_next_cell_index(initial_coords))
  else
    return Generic1DCoords(
           lake_parameters.grid_specific_lake_parameters.connect_next_cell_index(initial_coords))
  end
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

function get_outflow_redirect_coords(lake_parameters::LakeParameters,
                                     initial_coords::LatLonCoords,
                                     use_flood_redirect::Bool,
                                     use_additional_fields::Bool)
  local lat::Int64
  local lon::Int64
  if use_additional_fields
    lat = use_flood_redirect ?
      lake_parameters.grid_specific_lake_parameters.additional_flood_redirect_lat_index(initial_coords) :
      lake_parameters.grid_specific_lake_parameters.additional_connect_redirect_lat_index(initial_coords)

    lon = use_flood_redirect ?
      lake_parameters.grid_specific_lake_parameters.additional_flood_redirect_lon_index(initial_coords) :
      lake_parameters.grid_specific_lake_parameters.additional_connect_redirect_lon_index(initial_coords)
  else
    lat = use_flood_redirect ?
      lake_parameters.grid_specific_lake_parameters.flood_redirect_lat_index(initial_coords) :
      lake_parameters.grid_specific_lake_parameters.connect_redirect_lat_index(initial_coords)

    lon = use_flood_redirect ?
      lake_parameters.grid_specific_lake_parameters.flood_redirect_lon_index(initial_coords) :
      lake_parameters.grid_specific_lake_parameters.connect_redirect_lon_index(initial_coords)
  end
  return LatLonCoords(lat,lon)
end

function get_outflow_redirect_coords(lake_parameters::LakeParameters,
                                     initial_coords::Generic1DCoords,
                                     use_flood_redirect::Bool,
                                     use_additional_fields::Bool)
  local index::Int64
  if use_additional_fields
    index = use_flood_redirect ?
      lake_parameters.grid_specific_lake_parameters.additional_flood_redirect_index(initial_coords) :
      lake_parameters.grid_specific_lake_parameters.additional_connect_redirect_index(initial_coords)
  else
    index = use_flood_redirect ?
      lake_parameters.grid_specific_lake_parameters.flood_redirect_index(initial_coords) :
      lake_parameters.grid_specific_lake_parameters.connect_redirect_index(initial_coords)
  end
  return Generic1DCoords(index)
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
  print_lake_types(lake_parameters.grid,lake_prognostics,lake_fields)
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

function write_lake_numbers_field(lake_parameters::LakeParameters,lake_fields::LakeFields;
                                  timestep::Int64=-1)
  throw(UserError())
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
      lake.lake_variables.lake_volume + lake.lake_variables.secondary_lake_volume
  end
  for_all(lake_parameters.grid) do coords::Coords
    lake_number = lake_fields.lake_numbers(coords)
    if (lake_number > 0)
      set!(diagnostic_lake_volumes,coords,lake_volumes_by_lake_number[lake_number])
    end
  end
  return diagnostic_lake_volumes
end

function write_lake_volumes_field(lake_parameters::LakeParameters,lake_volumes::Field{Float64})
  throw(UserError())
end

function write_diagnostic_lake_volumes_field(lake_parameters::LakeParameters,
                                             diagnostic_lake_volumes::Field{Float64};
                                             timestep::Int64=-1)
  throw(UserError())
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
                                         lake_diagnostic_variables.total_lake_volume
  total_water_to_lakes::Float64 = sum(lake_fields.water_to_lakes)
  total_inflow_minus_outflow::Float64 = total_water_to_lakes +
                                        sum(lake_fields.lake_water_from_ocean) -
                                        sum(lake_fields.water_to_hd)
  difference::Float64 = change_in_total_lake_volume - total_inflow_minus_outflow
  tolerance::Float64 = 10e-16*max(new_total_lake_volume, total_water_to_lakes)
  if ! isapprox(difference,0,atol=tolerance)
    println("*** Lake Water Budget ***")
    println("Total lake volume: $(new_total_lake_volume)")
    println("Total inflow - outflow: $(total_inflow_minus_outflow)")
    println("Change in lake volume: $(change_in_total_lake_volume)")
    println("Difference: $(difference)")
  end
  lake_diagnostic_variables.total_lake_volume = new_total_lake_volume
  return prognostic_fields
end

end
