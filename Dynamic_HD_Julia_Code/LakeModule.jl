module LakeModule

using HierarchicalStateMachineModule: State, Event
using UserExceptionModule: UserError
using LakeModelParametersModule: LakeModelParameters
using MergeTypesModule
using HDModule: RiverParameters, PrognosticFields, RiverPrognosticFields, RiverDiagnosticFields, PrintResults
using HDModule: print_river_results, get_river_parameters, get_river_fields
using CoordsModule: Coords,LatLonCoords,is_lake
using GridModule: Grid, LatLonGrid, for_all, for_all_fine_cells_in_coarse_cell, for_all_with_line_breaks
using FieldModule: Field, set!
import HDModule: water_to_lakes,water_from_lakes
import HierarchicalStateMachineModule: handle_event

abstract type Lake <: State end

struct RunLakes <: Event end


struct AddWater <: Event
 inflow::Float64
end

struct StoreWater <: Event
 inflow::Float64
end

struct AcceptMerge <: Event
  redirect_coords::Coords
end

struct DrainExcessWater <: Event end

abstract type GridSpecificLakeParameters end

struct LakeParameters
  lake_centers::Field{Bool}
  connection_volume_thresholds::Field{Float64}
  flood_volume_thresholds::Field{Float64}
  flood_only::Field{Bool}
  flood_local_redirect::Field{Bool}
  connect_local_redirect::Field{Bool}
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
end

struct LakeFields
  completed_lake_cells::Field{Bool}
  lake_numbers::Field{Int64}
  water_to_lakes::Field{Float64}
  water_to_hd::Field{Float64}
  function LakeFields(river_parameters::RiverParameters,
                      lake_parameters::LakeParameters)
    completed_lake_cells = Field{Bool}(lake_parameters.grid,false)
    lake_numbers = Field{Int64}(lake_parameters.grid,0)
    water_to_lakes = Field{Float64}(river_parameters.grid,0.0)
    water_to_hd = Field{Float64}(river_parameters.grid,0.0)
    new(completed_lake_cells,lake_numbers,water_to_lakes,water_to_hd)
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
        push!(lakes,FillingLake(lake_parameters,
                                LakeVariables(coords,lakes,
                                              lake_number),
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
  unprocessed_water::Float64
  current_cell_to_fill::Coords
  other_lakes::Vector{Lake}
  lake_number::Int64
  LakeVariables(center_cell::Coords,
                other_lakes::Vector{Lake},
                lake_number::Int64) =
    new(center_cell,0.0,0.0,deepcopy(center_cell),other_lakes,
        lake_number)
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
              FillingLakeVariables(false))

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
  overflowing_lake_variables::OverflowingLakeVariables
end

struct SubsumedLake <: Lake
  lake_parameters::LakeParameters
  lake_variables::LakeVariables
  lake_fields::LakeFields
  primary_lake_number::Int64
end


struct RiverAndLakePrognosticFields <: PrognosticFields
  river_parameters::RiverParameters
  river_fields::RiverPrognosticFields
  river_diagnostic_fields::RiverDiagnosticFields
  lake_parameters::LakeParameters
  lake_prognostics::LakePrognostics
  lake_fields::LakeFields
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
        lake_parameters,lake_prognostics,lake_fields,true)
  end
end

get_lake_prognostics(river_and_lake_prognostics::RiverAndLakePrognosticFields) =
                     river_and_lake_prognostics.lake_prognostics::LakePrognostics

get_lake_fields(river_and_lake_prognostics::RiverAndLakePrognosticFields) =
                     river_and_lake_prognostics.lake_fields::LakeFields

get_lake_parameters(river_and_lake_prognostics::RiverAndLakePrognosticFields) =
                     river_and_lake_prognostics.lake_parameters::LakeParameters

function water_to_lakes(prognostic_fields::RiverAndLakePrognosticFields,coords::Coords,
                        inflow::Float64)
  lake_fields = get_lake_fields(prognostic_fields)
  set!(lake_fields.water_to_lakes,coords,inflow)
end

function water_from_lakes(prognostic_fields::RiverAndLakePrognosticFields)
  lake_fields = get_lake_fields(prognostic_fields)
  return lake_fields.water_to_hd
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,run_lake::RunLakes)
  river_parameters = get_river_parameters(prognostic_fields)
  river_fields = get_river_fields(prognostic_fields)
  lake_fields = get_lake_fields(prognostic_fields)
  lake_parameters = get_lake_parameters(prognostic_fields)
  lake_prognostics = get_lake_prognostics(prognostic_fields)
  drain_excess_water::DrainExcessWater = DrainExcessWater()
  fill!(lake_fields.water_to_hd,0.0)
  for_all(river_parameters.grid) do coords::Coords
    if lake_fields.water_to_lakes(coords) > 0.0
      basins_in_cell::Array{Coords,1} =
        lake_parameters.basins[lake_parameters.basin_numbers(coords)]
      share_to_each_lake = lake_fields.water_to_lakes(coords)/length(basins_in_cell)
      add_water::AddWater = AddWater(share_to_each_lake)
      for basin_center::Coords in basins_in_cell
        lake_index::Int64 = lake_fields.lake_numbers(basin_center)
        lake::Lake = lake_prognostics.lakes[lake_index]
        lake_prognostics.lakes[lake_index] = handle_event(lake,add_water)
      end
    end
  end
  for lake::Lake in lake_prognostics.lakes
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
        if ! (merge_type == primary_merge &&
              lake.filling_lake_variables.primary_merge_completed)
          if (check_if_merge_is_possible(lake,merge_type))
            if merge_type == secondary_merge
              subsumed_lake::Lake = perform_secondary_merge(lake)
              subsumed_lake = handle_event(subsumed_lake,
                                           StoreWater(inflow))
              return subsumed_lake
            else
              perform_primary_merge(lake)
            end
          else
            overflowing_lake::Lake = change_to_overflowing_lake(lake)
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

function handle_event(lake::OverflowingLake,add_water::AddWater)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  inflow::Float64 = add_water.inflow +
                    lake_variables.unprocessed_water
  lake_variables.unprocessed_water = 0.0
  if lake.local_redirect
    if lake_parameters.lake_model_parameters.instant_throughflow
      lake_number::Int64 = lake_fields.lake_numbers(lake.outflow_redirect_coords)
      other_lake::Lake =
        lake_variables.other_lakes[lake_number]
      lake_variables.other_lakes[lake_number] = handle_event(other_lake,
                                                             AddWater(inflow))
    else
      lake_fields.water_to_lakes[lake.outflow_redirect_coords] += inflow
    end
  else
    lake.overflowing_lake_variables.excess_water += inflow
  end
  return lake
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

function handle_event(lake::Lake,accept_merge::AcceptMerge)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
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

function change_to_overflowing_lake(lake::FillingLake)
  outflow_redirect_coords,local_redirect =
    get_outflow_redirect_coords(lake)
  return OverflowingLake(lake.lake_parameters,
                         lake.lake_variables,
                         lake.lake_fields,
                         outflow_redirect_coords,
                         local_redirect,
                         lake.lake_parameters.lake_model_parameters.lake_retention_constant,
                         OverflowingLakeVariables())
end

function handle_event(lake::OverflowingLake,drain_excess_water::DrainExcessWater)
  lake_fields::LakeFields = get_lake_fields(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  if lake.overflowing_lake_variables.excess_water > 0.0
    lake.overflowing_lake_variables.excess_water +=
      lake_variables.unprocessed_water
    lake_variables.unprocessed_water = 0.0
    flow = lake.overflowing_lake_variables.excess_water*lake.lake_retention_coefficient
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
  if merge_type == primary_merge
    target_cell = get_primary_merge_coords(lake)
  else
    target_cell = get_secondary_merge_coords(lake)
  end
  if ! lake_fields.completed_lake_cells(target_cell)
    return false
  end
  other_lake::Lake = lake_variables.other_lakes[lake_fields.lake_numbers(target_cell)]
  other_lake = find_true_primary_lake(other_lake)
  if isa(other_lake,OverflowingLake)
    return true
  else
    return false
  end
end

function perform_primary_merge(lake::FillingLake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  target_cell::Coords = get_primary_merge_coords(lake)
  other_lake::Lake = lake_variables.other_lakes[lake_fields.lake_numbers(target_cell)]
  other_lake = find_true_primary_lake(other_lake)
  lake.filling_lake_variables.primary_merge_completed = true
  accept_merge::AcceptMerge = AcceptMerge(lake_variables.center_cell)
  lake_variables.other_lakes[lake_fields.lake_numbers(target_cell)] =
    handle_event(other_lake,accept_merge)
end

function perform_secondary_merge(lake::FillingLake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  set!(lake_fields.completed_lake_cells,
       lake_variables.current_cell_to_fill,true)
  target_cell::Coords = get_secondary_merge_coords(lake)
  other_lake_number = lake_fields.lake_numbers(target_cell)
  other_lake::Lake = lake_variables.other_lakes[other_lake_number]
  other_lake = find_true_primary_lake(other_lake)
  other_lake_as_filling_lake::FillingLake = change_to_filling_lake(other_lake)
  lake_variables.other_lakes[other_lake_number] = other_lake_as_filling_lake
  accept_merge::AcceptMerge =
    AcceptMerge(get_lake_variables(other_lake_as_filling_lake).center_cell)
  return handle_event(lake,accept_merge)
end

function change_to_filling_lake(lake::OverflowingLake)
  lake.lake_variables.unprocessed_water +=
    lake.overflowing_lake_variables.excess_water
  filling_lake::FillingLake = FillingLake(lake.lake_parameters,
                                          lake.lake_variables,
                                          lake.lake_fields,
                                          FillingLakeVariables(true))
  return filling_lake
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
  lake_variables.current_cell_to_fill = new_coords
end

function get_primary_merge_coords(lake::Lake)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  current_cell::Coords = lake_variables.current_cell_to_fill
  get_primary_merge_coords(lake_parameters,
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

function get_outflow_redirect_coords(lake::Lake)
  lake_parameters::LakeParameters = get_lake_parameters(lake)
  lake_variables::LakeVariables = get_lake_variables(lake)
  lake_fields::LakeFields = get_lake_fields(lake)
  current_cell::Coords = lake_variables.current_cell_to_fill
  completed_lake_cell::Bool = lake_fields.completed_lake_cells(current_cell) ||
                              lake_parameters.flood_only(current_cell)
  local_redirect::Bool = completed_lake_cell ?
    lake_parameters.flood_local_redirect(current_cell) :
    lake_parameters.connect_local_redirect(current_cell)
  return get_outflow_redirect_coords(lake_parameters,
                                     current_cell,
                                     completed_lake_cell),local_redirect
end

function find_true_primary_lake(lake::Lake)
  if isa(lake,SubsumedLake)
    return find_true_primary_lake(lake.lake_variables.other_lakes[lake.primary_lake_number])
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
                                     use_flood_redirect::Bool)
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

function get_flood_next_cell_coords(lake_parameters::LakeParameters,initial_coords::LatLonCoords)
  return LatLonCoords(
    lake_parameters.grid_specific_lake_parameters.flood_next_cell_lat_index(initial_coords),
    lake_parameters.grid_specific_lake_parameters.flood_next_cell_lon_index(initial_coords))
end

function get_connect_next_cell_coords(lake_parameters::LakeParameters,initial_coords::LatLonCoords)
  return LatLonCoords(
    lake_parameters.grid_specific_lake_parameters.connect_next_cell_lat_index(initial_coords),
    lake_parameters.grid_specific_lake_parameters.connect_next_cell_lon_index(initial_coords))
end

function get_outflow_redirect_coords(lake_parameters::LakeParameters,
                                     initial_coords::LatLonCoords,
                                     use_flood_redirect::Bool)
  lat::Int64 = use_flood_redirect ?
    lake_parameters.grid_specific_lake_parameters.flood_redirect_lat_index(initial_coords) :
    lake_parameters.grid_specific_lake_parameters.connect_redirect_lat_index(initial_coords)

  lon::Int64 = use_flood_redirect ?
    lake_parameters.grid_specific_lake_parameters.flood_redirect_lon_index(initial_coords) :
    lake_parameters.grid_specific_lake_parameters.connect_redirect_lon_index(initial_coords)
  return LatLonCoords(lat,lon)
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
end

end