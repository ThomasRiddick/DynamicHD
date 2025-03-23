module L2LakeModule

using L2LakeModelDefsModule: Lake,LakeModelParameters,LakeModelPrognostics
using L2LakeModelDefsModule: GridSpecificLakeModelParameters
using L2LakeModelGridSpecificDefsModule: LatLonLakeModelParameters
using L2LakeModelGridSpecificDefsModule: UnstructuredLakeModelParameters
using L2LakeModelDefsModule: get_corresponding_surface_model_grid_cell
using L2CalculateLakeFractionsModule: add_pixel_by_coords,remove_pixel_by_coords
using SplittableRootedTree: find_root,for_elements_in_set,get_label
using SplittableRootedTree: make_new_link,split_set
using FieldModule: set!
using HierarchicalStateMachineModule: Event
using GridModule: Grid,find_coarse_cell_containing_fine_cell
import HierarchicalStateMachineModule: handle_event
import Base.show

const debug = true

@enum HeightType connect_height flood_height

struct AddWater <: Event
 inflow::Float64
 store_water::Bool
end

struct RemoveWater <: Event
  outflow::Float64
  store_water::Bool
end

struct ProcessWater <: Event end

struct DrainExcessWater <: Event end

struct ReleaseNegativeWater <: Event end

struct CalculateEffectiveLakeVolumePerCell <: Event end

struct Cell
  coords::CartesianIndex
  height_type::HeightType
  fill_threshold::Float64
  height::Float64
end

struct Redirect
  use_local_redirect::Bool
  local_redirect_target_lake_number::Int64
  non_local_redirect_target::CartesianIndex
end

struct LakeParameters
  center_coords::CartesianIndex
  center_cell_coarse_coords::CartesianIndex
  lake_number::Int64
  is_primary::Bool
  is_leaf::Bool
  primary_lake::Int64
  secondary_lakes::Vector{Int64}
  filling_order::Vector{Cell}
  outflow_points::Dict{Int64,Redirect}
  function LakeParameters(lake_number::Int64,
                          primary_lake::Int64,
                          secondary_lakes::Vector{Int64},
                          center_coords::CartesianIndex,
                          filling_order::Vector{Cell},
                          outflow_points::Dict{Int64,Redirect},
                          fine_grid::Grid,coarse_grid::Grid)
  center_cell_coarse_coords::CartesianIndex =
    find_coarse_cell_containing_fine_cell(fine_grid,coarse_grid,
                                          center_coords)
  is_primary::Bool = primary_lake == -1
  if is_primary && length(outflow_points) > 1
    error("Primary lake has more than one outflow point")
  end
  is_leaf::Bool = length(secondary_lakes) == 0
  new(center_coords,center_cell_coarse_coords,
      lake_number,is_primary,is_leaf,
      primary_lake,secondary_lakes,
      filling_order,
      outflow_points)
  end
end

mutable struct LakeVariables
  unprocessed_water::Float64
  active_lake::Bool
  function LakeVariables(active_lake::Bool)
    new(0,active_lake)
  end
end

mutable struct FillingLake <: Lake
  parameters::LakeParameters
  variables::LakeVariables
  lake_model_parameters::LakeModelParameters
  lake_model_prognostics::LakeModelPrognostics
  current_cell_to_fill::CartesianIndex
  current_height_type::HeightType
  current_filling_cell_index::Int64
  next_cell_volume_threshold::Float64
  previous_cell_volume_threshold::Float64
  lake_volume::Float64
  function FillingLake(lake_parameters::LakeParameters,
                       lake_variables::LakeVariables,
                       lake_model_parameters::LakeModelParameters,
                       lake_model_prognostics::LakeModelPrognostics,
                       initialise_filled::Bool)
    if initialise_filled
      current_cell_to_fill = lake_parameters.filling_order[end].coords
      current_height_type = lake_parameters.filling_order[end].height_type
      current_filling_cell_index = length(lake_parameters.filling_order)
      next_cell_volume_threshold =
        lake_parameters.filling_order[end].fill_threshold
      if length(lake_parameters.filling_order) > 1
        previous_cell_volume_threshold =
          lake_parameters.filling_order[end-1].fill_threshold
      else
        previous_cell_volume_threshold = 0.0
      end
      lake_volume = lake_parameters.filling_order[end].fill_threshold
    else
      current_cell_to_fill =
        lake_parameters.filling_order[1].coords
      current_height_type =
        lake_parameters.filling_order[1].height_type
      current_filling_cell_index = 1
      next_cell_volume_threshold =
        lake_parameters.filling_order[1].fill_threshold
      previous_cell_volume_threshold = 0.0
      lake_volume = 0.0
    end
    new(lake_parameters,lake_variables,lake_model_parameters,lake_model_prognostics,
        current_cell_to_fill,current_height_type,current_filling_cell_index,
        next_cell_volume_threshold,previous_cell_volume_threshold,lake_volume)
  end
end

mutable struct OverflowingLake <: Lake
  parameters::LakeParameters
  variables::LakeVariables
  lake_model_parameters::LakeModelParameters
  lake_model_prognostics::LakeModelPrognostics
  current_redirect::Redirect
  excess_water::Float64
end

mutable struct SubsumedLake <: Lake
  parameters::LakeParameters
  variables::LakeVariables
  lake_model_parameters::LakeModelParameters
  lake_model_prognostics::LakeModelPrognostics
  redirect_target::Int64
  function SubsumedLake(parameters::LakeParameters,
                        variables::LakeVariables,
                        lake_model_parameters::LakeModelParameters,
                        lake_model_prognostics::LakeModelPrognostics)
    new(parameters,variables,lake_model_parameters,
        lake_model_prognostics,-1)
  end
end

get_lake_data(obj::T) where {T <: Lake} =
  obj.parameters::LakeParameters,obj.variables::LakeVariables,
    obj.lake_model_parameters::LakeModelParameters,
    obj.lake_model_prognostics::LakeModelPrognostics

function handle_event(lake::FillingLake,add_water::AddWater)
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
    lake_model_parameters::LakeModelParameters,
    lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  if ! lake_variables.active_lake
    #Take the first sub-basin of this lake
    sublake::Lake = lake_model_prognostics.lakes[lake_parameters.secondary_lakes[1]]
    lake_model_prognostics.lakes[lake_parameters.secondary_lakes[1]] =
      handle_event(sublake,AddWater(add_water.inflow,false))
    return lake_model_prognostics.lakes[lake_parameters.lake_number]
  end
  if add_water.store_water
    lake_variables.unprocessed_water += add_water.inflow
    return lake
  end
  inflow::Float64 = add_water.inflow
  while inflow > 0.0
    local surface_model_coords::CartesianIndex
    if lake.lake_volume >= 0 && lake.current_filling_cell_index == 1 &&
       lake_model_prognostics.lake_numbers(lake.current_cell_to_fill) == 0 &&
       ! lake_parameters.is_leaf && lake.current_height_type == flood_height
      surface_model_coords =
        get_corresponding_surface_model_grid_cell(lake.current_cell_to_fill,
                                                  lake_model_parameters.
                                                  grid_specific_lake_model_parameters)
      set!(lake_model_prognostics.lake_cell_count,surface_model_coords,
           lake_model_prognostics.lake_cell_count(surface_model_coords) + 1)

      add_pixel_by_coords(lake.current_cell_to_fill,
                          lake_model_prognostics.adjusted_lake_cell_count,
                          lake_model_prognostics.lake_fraction_prognostics)
      set!(lake_model_prognostics.lake_numbers,lake.current_cell_to_fill,
             lake.parameters.lake_number)
    end
    new_lake_volume::Float64 = inflow + lake.lake_volume
    if new_lake_volume <= lake.next_cell_volume_threshold
      lake.lake_volume = new_lake_volume
      inflow = 0.0
    else
      inflow = new_lake_volume - lake.next_cell_volume_threshold
      lake.lake_volume = lake.next_cell_volume_threshold
      if lake.current_filling_cell_index == length(lake.parameters.filling_order)
        if check_if_merge_is_possible(lake_parameters,lake_model_prognostics)
          subsumed_lake::SubsumedLake = merge_lakes(lake,inflow,lake_parameters,
                                                    lake_model_prognostics)
          return subsumed_lake
        else
          overflowing_lake::OverflowingLake = change_to_overflowing_lake(lake,inflow)
          return overflowing_lake
        end
      end
      lake.current_filling_cell_index += 1
      lake.previous_cell_volume_threshold = lake.next_cell_volume_threshold
      lake.next_cell_volume_threshold =
        lake.parameters.filling_order[lake.current_filling_cell_index].fill_threshold
      lake.current_cell_to_fill =
        lake.parameters.filling_order[lake.current_filling_cell_index].coords
      lake.current_height_type =
        lake.parameters.filling_order[lake.current_filling_cell_index].height_type
      if lake.current_height_type == flood_height &&
         lake_model_prognostics.lake_numbers(lake.current_cell_to_fill) == 0
        surface_model_coords =
          get_corresponding_surface_model_grid_cell(lake.current_cell_to_fill,
                                                    lake_model_parameters.
                                                    grid_specific_lake_model_parameters)
        set!(lake_model_prognostics.lake_cell_count,surface_model_coords,
           lake_model_prognostics.lake_cell_count(surface_model_coords) + 1)
        add_pixel_by_coords(lake.current_cell_to_fill,
                            lake_model_prognostics.adjusted_lake_cell_count,
                            lake_model_prognostics.lake_fraction_prognostics)
        set!(lake_model_prognostics.lake_numbers,lake.current_cell_to_fill,
             lake.parameters.lake_number)
      end
    end
  end
  return lake
end

function handle_event(lake::OverflowingLake,add_water::AddWater)
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
    _,lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  if ! lake_variables.active_lake
    error("Water added to inactive lake")
  end
  if add_water.store_water
    lake_variables.unprocessed_water += add_water.inflow
    return lake
  end
  inflow::Float64 = add_water.inflow
  if lake.current_redirect.use_local_redirect
    other_lake_number::Int64 = lake.current_redirect.local_redirect_target_lake_number
    other_lake::Lake =
      lake_model_prognostics.lakes[other_lake_number]
    lake_model_prognostics.lakes[other_lake_number] =
      handle_event(other_lake,AddWater(inflow,false))
  else
    lake.excess_water += inflow
  end
  #Don't just return lake - instead use pointer to this lake in lakes
  #array in case adding water changed this lakes type
  return lake_model_prognostics.lakes[lake_parameters.lake_number]
end

function handle_event(lake::SubsumedLake,add_water::AddWater)
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
    _,lake_model_prognostics::LakeModelPrognostics =
    get_lake_data(lake)
  if ! lake_variables.active_lake
    error("Water added to inactive lake")
  end
  if lake.redirect_target == -1
    lake.redirect_target = find_root(lake.lake_model_prognostics.set_forest,
                                     lake_parameters.lake_number)
  end
  lake_model_prognostics.lakes[lake.redirect_target] =
    handle_event(lake_model_prognostics.lakes[lake.redirect_target],
                 add_water)
  return lake
end

function handle_event(lake::FillingLake,remove_water::RemoveWater)
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
    lake_model_parameters::LakeModelParameters,
    lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  if ! lake_variables.active_lake
    error("Water removed from inactive lake")
  end
  if remove_water.store_water
    lake_variables.unprocessed_water -= remove_water.outflow
    return lake
  end
  outflow::Float64 = remove_water.outflow
  local surface_model_coords::CartesianIndex
  repeat_loop::Bool = false
  while outflow > 0.0 || repeat_loop
    repeat_loop = false
    new_lake_volume::Float64 = lake.lake_volume - outflow
    if new_lake_volume > 0 && new_lake_volume <=
       lake_model_parameters.lake_model_settings.minimum_lake_volume_threshold &&
       lake_parameters.is_leaf
      set!(lake_model_prognostics.lake_water_from_ocean,
           lake_parameters.center_cell_coarse_coords,
           lake_model_prognostics.lake_water_from_ocean(lake_parameters.center_cell_coarse_coords)
           - new_lake_volume)
      new_lake_volume = 0
    end
    minimum_new_lake_volume::Float64 = lake.previous_cell_volume_threshold
    if new_lake_volume <= 0.0 && lake.current_filling_cell_index == 1
      outflow = 0.0
      if lake_parameters.is_leaf
        lake.lake_volume = new_lake_volume
      else
        lake.lake_volume = 0.0
        split_lake(lake,-new_lake_volume,lake_parameters,lake_variables,
                   lake_model_prognostics)
      end
      if lake_model_prognostics.lake_numbers(lake.current_cell_to_fill) == lake.parameters.lake_number &&
          ! lake_parameters.is_leaf && lake.current_height_type == flood_height
        surface_model_coords =
          get_corresponding_surface_model_grid_cell(lake.current_cell_to_fill,
                                                    lake_model_parameters.
                                                    grid_specific_lake_model_parameters)
        set!(lake_model_prognostics.lake_cell_count,surface_model_coords,
             lake_model_prognostics.lake_cell_count(surface_model_coords) - 1)
        remove_pixel_by_coords(lake.current_cell_to_fill,
                               lake_model_prognostics.adjusted_lake_cell_count,
                               lake_model_prognostics.lake_fraction_prognostics)
        set!(lake_model_prognostics.lake_numbers,lake.current_cell_to_fill,0)
      end
    elseif new_lake_volume >= minimum_new_lake_volume &&
           new_lake_volume > 0
      outflow = 0.0
      lake.lake_volume = new_lake_volume
    else
      outflow = minimum_new_lake_volume - new_lake_volume
      lake.lake_volume = minimum_new_lake_volume
      if lake.current_height_type == flood_height &&
         lake_model_prognostics.lake_numbers(lake.current_cell_to_fill) == lake.parameters.lake_number
        surface_model_coords =
          get_corresponding_surface_model_grid_cell(lake.current_cell_to_fill,
                                                    lake_model_parameters.
                                                    grid_specific_lake_model_parameters)
        set!(lake_model_prognostics.lake_cell_count,surface_model_coords,
             lake_model_prognostics.lake_cell_count(surface_model_coords) - 1)
        remove_pixel_by_coords(lake.current_cell_to_fill,
                               lake_model_prognostics.adjusted_lake_cell_count,
                               lake_model_prognostics.lake_fraction_prognostics)
        set!(lake_model_prognostics.lake_numbers,lake.current_cell_to_fill,0)
      end
      lake.current_filling_cell_index -= 1
      lake.next_cell_volume_threshold = lake.previous_cell_volume_threshold
      if lake.current_filling_cell_index > 1
        lake.previous_cell_volume_threshold =
          lake.parameters.filling_order[lake.current_filling_cell_index-1].fill_threshold
      else
        lake.previous_cell_volume_threshold = 0.0
      end
      lake.current_cell_to_fill = lake.parameters.filling_order[lake.current_filling_cell_index].coords
      lake.current_height_type =
        lake.parameters.filling_order[lake.current_filling_cell_index].height_type
      if lake.current_filling_cell_index > 1
        repeat_loop = true
      end
    end
  end
  return lake
end

function handle_event(lake::OverflowingLake,remove_water::RemoveWater)
  _,lake_variables::LakeVariables,_,_= get_lake_data(lake)
  if ! lake_variables.active_lake
    error("Water removed from inactive lake")
  end
  if remove_water.store_water
    lake_variables.unprocessed_water -= remove_water.outflow
    return lake
  end
  outflow::Float64 = remove_water.outflow
  if outflow <= lake.excess_water
    lake.excess_water -= outflow
    return lake
  end
  outflow -= lake.excess_water
  lake.excess_water = 0.0
  lake_as_filling_lake::FillingLake = change_to_filling_lake(lake)
  return handle_event(lake_as_filling_lake,RemoveWater(outflow,false))
end

function handle_event(lake::SubsumedLake,remove_water::RemoveWater)
  #Redirect to primary
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
    _,lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  if ! lake_variables.active_lake
    error("Water removed from inactive lake")
  end
  if lake.redirect_target == -1
    lake.redirect_target = find_root(lake_model_prognostics.set_forest,
                                     lake_parameters.lake_number)
  end
  #Copy this variables from the lake object as the lake object will
  #have its redirect set to -1 if the redirect target splits
  redirect_target::Int64 = lake.redirect_target
  lake_model_prognostics.lakes[redirect_target] =
    handle_event(lake_model_prognostics.lakes[redirect_target],
                 remove_water)
  #Return this lake via a pointer in case it has changed type
  return lake_model_prognostics.lakes[lake_parameters.lake_number]::Lake
end

function handle_event(lake::Lake,::ProcessWater)
  _,lake_variables::LakeVariables,_,_ = get_lake_data(lake)
  unprocessed_water::Float64 = lake_variables.unprocessed_water
  lake_variables.unprocessed_water = 0.0
  if unprocessed_water > 0.0
      return handle_event(lake,AddWater(unprocessed_water,false))
  elseif unprocessed_water < 0.0
      return handle_event(lake,RemoveWater(-unprocessed_water,false))
  end
end

function check_if_merge_is_possible(lake_parameters::LakeParameters,
                                    lake_model_prognostics::LakeModelPrognostics)
  if ! lake_parameters.is_primary
    all_secondary_lakes_filled::Bool = true
    for secondary_lake in lake_model_prognostics.lakes[lake_parameters.primary_lake].
                          parameters.secondary_lakes
      if secondary_lake != lake_parameters.lake_number
        all_secondary_lakes_filled &= isa(lake_model_prognostics.lakes[secondary_lake],OverflowingLake)
      end
    end
    return all_secondary_lakes_filled
  else
    return false
  end
end

function merge_lakes(::FillingLake,inflow::Float64,
                     lake_parameters::LakeParameters,
                     lake_model_prognostics::LakeModelPrognostics)
  total_excess_water::Float64 = 0.0
  primary_lake::FillingLake = lake_model_prognostics.lakes[lake_parameters.primary_lake]
  primary_lake.variables.active_lake = true
  for secondary_lake in primary_lake.parameters.secondary_lakes
    other_lake::Lake = lake_model_prognostics.lakes[secondary_lake]
    if (isa(other_lake,FillingLake) &&
        lake_parameters.lake_number != secondary_lake) ||
        isa(other_lake,SubsumedLake)
      error("wrong lake type when merging")
    end
    subsumed_lake::SubsumedLake,excess_water::Float64 = change_to_subsumed_lake(other_lake)
    lake_model_prognostics.lakes[secondary_lake] = subsumed_lake
    total_excess_water += excess_water
  end
  lake_model_prognostics.lakes[primary_lake.parameters.lake_number] =
    handle_event(lake_model_prognostics.lakes[primary_lake.parameters.lake_number],
                 AddWater(inflow+total_excess_water,false))
  return lake_model_prognostics.lakes[lake_parameters.lake_number]
end

function split_lake(::FillingLake,water_deficit::Float64,
                    lake_parameters::LakeParameters,
                    lake_variables::LakeVariables,
                    lake_model_prognostics::LakeModelPrognostics)
  lake_variables.active_lake = false
  water_deficit_per_lake::Float64 = water_deficit/length(lake_parameters.secondary_lakes)
  for secondary_lake::Int64 in lake_parameters.secondary_lakes
    filling_lake::FillingLake =
      change_to_filling_lake(lake_model_prognostics.lakes[secondary_lake],
                             lake_parameters.lake_number)
    lake_model_prognostics.lakes[secondary_lake] = filling_lake
    lake_model_prognostics.lakes[secondary_lake] =
      handle_event(filling_lake,RemoveWater(water_deficit_per_lake,false))
  end
end

function change_to_overflowing_lake(lake::FillingLake,inflow::Float64)
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
    lake_model_parameters::LakeModelParameters,
    lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  if debug
    println("Lake $(lake_parameters.lake_number) changing from filling to overflowing lake ")
  end
  local redirect::Redirect
  local redirect_target_found::Bool
  if ! lake_parameters.is_primary
    for secondary_lake in
        lake_model_prognostics.lakes[lake_parameters.primary_lake].
        parameters.secondary_lakes
      redirect_target_found = false
      if isa(lake_model_prognostics.lakes[secondary_lake],FillingLake) &&
         secondary_lake != lake.parameters.lake_number
        redirect =
          lake_parameters.outflow_points[secondary_lake]
        redirect_target_found = true
        break
      end
    end
    if ! redirect_target_found
      error("Non primary lake has no valid outflows")
    end
  else
    if haskey(lake_parameters.outflow_points,-1)
      redirect = lake_parameters.outflow_points[-1]
    else
      for key in keys(lake_parameters.outflow_points)
        redirect = lake_parameters.outflow_points[key]
      end
    end
  end
  return OverflowingLake(lake_parameters,
                         lake_variables,
                         lake_model_parameters,
                         lake_model_prognostics,
                         redirect,
                         inflow)
end

function change_to_subsumed_lake(lake::Union{FillingLake,OverflowingLake})
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
    lake_model_parameters::LakeModelParameters,
    lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  if debug
    println("Lake $(lake_parameters.lake_number) accepting merge")
  end
  subsumed_lake::SubsumedLake = SubsumedLake(lake_parameters,
                                             lake_variables,
                                             lake_model_parameters,
                                             lake_model_prognostics)
  make_new_link(lake_model_prognostics.set_forest,
                lake_parameters.primary_lake,
                lake_parameters.lake_number)
  if isa(lake,OverflowingLake)
    return subsumed_lake,lake.excess_water
  else
    return subsumed_lake,0.0
  end
end

function change_to_filling_lake(lake::OverflowingLake)
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
    lake_model_parameters::LakeModelParameters,
    lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  if lake.excess_water != 0.0
    error("Can't change lake with excess water back to filling lake")
  end
  if debug
    println("Lake $(lake_parameters.lake_number) changing from overflowing to filling lake ")
  end
  filling_lake::FillingLake = FillingLake(lake_parameters,
                                          lake_variables,
                                          lake_model_parameters,
                                          lake_model_prognostics,
                                          true)
  return filling_lake
end

function change_to_filling_lake(lake::SubsumedLake,split_from_lake_number::Int64)
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
  lake_model_parameters::LakeModelParameters,
  lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  split_set(lake_model_prognostics.set_forest,
            split_from_lake_number,
            lake_parameters.lake_number)
  for_elements_in_set(lake_model_prognostics.set_forest,
                      lake_parameters.lake_number,
                      x ->
                      lake_model_prognostics.lakes[get_label(x)].redirect_target = -1)
  filling_lake::FillingLake = FillingLake(lake_parameters,
                                          lake_variables,
                                          lake_model_parameters,
                                          lake_model_prognostics,
                                          true)
  return filling_lake
end

function handle_event(lake::OverflowingLake,::DrainExcessWater)
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
  lake_model_parameters::LakeModelParameters,
  lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  if lake.excess_water > 0.0
    if lake.current_redirect.use_local_redirect
      other_lake_number::Int64 = lake.current_redirect.local_redirect_target_lake_number
      other_lake::Lake =
        lake_model_prognostics.lakes[other_lake_number]
      excess_water::Float64 = lake.excess_water
      lake.excess_water = 0.0
      lake_model_prognostics.lakes[other_lake_number] =
        handle_event(other_lake,AddWater(excess_water,false))
    else
      total_lake_volume::Float64 = 0.0
      for_elements_in_set(lake_model_prognostics.set_forest,
                          find_root(lake_model_prognostics.set_forest,
                                    lake_parameters.lake_number),
                          x -> total_lake_volume +=
                          get_lake_volume(lake_model_prognostics.lakes[get_label(x)]))
      flow = (total_lake_volume)/
             (lake_model_parameters.lake_model_settings.lake_retention_constant + 1.0)
      flow = min(flow,lake.excess_water)
      set!(lake_model_prognostics.water_to_hd,
           lake.current_redirect.non_local_redirect_target,
           lake_model_prognostics.water_to_hd(lake.current_redirect.
                                              non_local_redirect_target)+flow)
      lake.excess_water -= flow
    end
  end
  return lake
end

function handle_event(lake::FillingLake,::ReleaseNegativeWater)
  lake_parameters::LakeParameters,_,_,
    lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  set!(lake_model_prognostics.lake_water_from_ocean,
       lake_parameters.center_cell_coarse_coords,
       lake_model_prognostics.lake_water_from_ocean(lake_parameters.center_cell_coarse_coords) -
       lake.lake_volume)
  lake.lake_volume = 0.0
  return lake
end

get_lake_volume(lake::FillingLake) = lake.lake_volume + lake.variables.unprocessed_water

get_lake_volume(lake::OverflowingLake) = lake.parameters.filling_order[end].fill_threshold +
                                         lake.excess_water +
                                         lake.variables.unprocessed_water

get_lake_volume(lake::SubsumedLake) = lake.parameters.filling_order[end].fill_threshold +
                                      lake.variables.unprocessed_water

get_lake_filled_cells(lake::FillingLake) =
  map(f->f.coords,lake.parameters.
                  filling_order[1:lake.current_filling_cell_index])

get_lake_filled_cells(lake::Union{OverflowingLake,SubsumedLake}) =
  map(f->f.coords,lake.parameters.filling_order)

function find_top_level_primary_lake_number(lake::Lake)
  if lake.parameters.is_primary
    return lake.parameters.lake_number
  else
    primary_lake::Lake = lake.lake_model_prognostics.lakes[lake.parameters.primary_lake]
    return find_top_level_primary_lake_number(primary_lake)
  end
end

function handle_event(lake::Union{OverflowingLake,FillingLake},
    calculate_effective_lake_volume_per_cell::CalculateEffectiveLakeVolumePerCell)
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
  lake_model_parameters::LakeModelParameters,
  lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  total_number_of_flooded_cells::Int64 = 0
  total_lake_volume::Float64 = 0.0
  working_cell_list::Vector{CartesianIndex} = CartesianIndex[]
  for_elements_in_set(lake_model_prognostics.set_forest,
                      find_root(lake_model_prognostics.set_forest,
                                lake.parameters.lake_number),
                      function (x)
                        other_lake::Lake = lake_model_prognostics.lakes[get_label(x)]
                        total_lake_volume += get_lake_volume(other_lake)
                        other_lake_working_cells::Vector{CartesianIndex} =
                          get_lake_filled_cells(other_lake)
                        total_number_of_flooded_cells +=
                          length(other_lake_working_cells)
                        append!(working_cell_list,
                                other_lake_working_cells)
                      end)
  effective_volume_per_cell::Float64 =
    total_lake_volume / total_number_of_flooded_cells
  for coords::CartesianIndex in working_cell_list
    surface_model_coords = get_corresponding_surface_model_grid_cell(coords,
                              lake_model_parameters.grid_specific_lake_model_parameters)
    set!(lake_model_prognostics.
         effective_volume_per_cell_on_surface_grid,surface_model_coords,
         lake_model_prognostics.
         effective_volume_per_cell_on_surface_grid(surface_model_coords) +
         effective_volume_per_cell)
  end
  return lake
end

function handle_event(lake::SubsumedLake,
    calculate_effective_lake_volume_per_cell::CalculateEffectiveLakeVolumePerCell)
  return lake
end

function show(io::IO,lake::Lake)
  lake_parameters::LakeParameters,lake_variables::LakeVariables,_,_ = get_lake_data(lake)
  println(io,"-----------------------------------------------")
  println(io,"Lake number: "*string(lake_parameters.lake_number))
  println(io,"Center cell: "*string(lake_parameters.center_coords))
  println(io,"Unprocessed water: "*string(lake_variables.unprocessed_water))
  println(io,"Active Lake: "*string(lake_variables.active_lake))
  println(io,"Center cell coarse coords: "*string(lake_parameters.center_cell_coarse_coords))
  println(io,"Primary lake: "*string(lake_parameters.primary_lake))
  if isa(lake,OverflowingLake)
    println(io,"O")
    println(io,"Excess water: "*string(lake.excess_water))
    println(io,"Current Redirect: "*string(lake.current_redirect))
  elseif isa(lake,FillingLake)
    println(io,"F")
    println(io,"Lake volume: "*string(lake.lake_volume))
    println(io,"Current cell to fill: "*string(lake.current_cell_to_fill))
    println(io,"Current height type: "*string(lake.current_height_type))
    println(io,"Current filling cell index: "*string(lake.current_filling_cell_index))
    println(io,"Next cell volume threshold: "*string(lake.next_cell_volume_threshold))
    println(io,"Previous cell volume threshold: "*string(lake.previous_cell_volume_threshold))
  elseif isa(lake,SubsumedLake)
    println(io,"S")
    println(io,"Redirect target : "*string(lake.redirect_target))
  end
end

end
