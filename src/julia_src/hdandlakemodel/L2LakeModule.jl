module L2LakeModule
#First lay foundations of new module... then merge in material from only module
#Use tree to accelerate forwarding of redirects etc

@enum HeightType connect_height flood_height

const debug = false

struct AddWater <: Event
 inflow::Float64
end

struct RemoveWater <: Event
  outflow::Float64
end

struct DrainExcessWater <: Event end

struct ReleaseNegativeWater <: Event end

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
  center_cell::CartesianIndex
  center_cell_coarse_coords::CartesianIndex
  lake_number::Int64
  is_primary::Bool
  is_leaf::Bool
  primary_lake::Int64
  secondary_lakes::Vector{Int64}
  filling_order::Vector{Cells}
  outflow_points::Dict{Int64,Redirect}
  function LakeParameters(lake_number::Int64,
                          primary_lake::Int64,
                          secondary_lakes::Int64,
                          center_coords::CartesianIndex,
                          filling_order::Vector{Cells},
                          outflow_points::Dict{Int64,Redirect},
                          fine_grid::Grid,coarse_Grid::Grid)
  center_cell_coarse_coords::CartesianIndex =
    find_coarse_cell_containing_fine_cell(fine_grid,coarse_grid,
                                          center_coords)
  is_primary::Bool = primary_lake == -1
  is_leaf::Bool = length(secondary_lakes) == 0
  new(center_cell,center_cell_coarse_coords,
      lake_number,is_primary,is_leaf,
      primary_lake,secondary_lakes,
      filling_order,
      outflow_points)
  end
end

struct LakeVariables
  unprocessed_water::Float64
  active_lake::Bool
  function LakeVariables(active_lake::Bool)
    new(0,active_lake)
  end
end

abstract type Lake <: State end

get_lake_data(obj::T) where {T <: Lake} =
  obj.lake_parameters::LakeParameters,obj.lake_variables::LakeVariables,
  obj.lake_model_parameters::LakeModelParameters,obj.lake_model_prognostics

struct FillingLake <: Lake
  parameters::LakeParameters
  variables::LakeVariables
  lake_model_parameters::LakeModelParameters
  lake_model_prognostics::LakeModelPrognostics
  current_cell_to_fill::Int64
  current_height_type::HeightType
  current_filling_cell_index::Int64
  next_cell_volume_threshold::Int64
  previous_cell_volume_threshold::Int64
  lake_volume::Float64
  function FillingLake(lake_parameters::LakeParameters,
                       lake_variables::LakeVariables,
                       lake_model_parameters::LakeModelParameters,
                       lake_model_prognostics::LakeModelPrognostics,
                       initialise_filled::Bool)
    if initialise_filled
      current_cell_to_fill = filling_order[-1].coords
      current_height_type = filling_order[-1].height_type
      current_filling_cell_index = length(filling_order)
      next_cell_volume_threshold =
        filling_order[-1].fill_threshold
      previous_cell_volume_threshold =
        filling_order[-2].fill_threshold
      lake_volume = filling_order[-1].fill_threshold
    else
      current_cell_to_fill =
        filling_order[1].coords
      current_height_type =
        filling_order[1].height_type
      current_filling_cell_index = 1
      next_cell_volume_threshold =
        filling_order[1].fill_threshold
      previous_cell_volume_threshold = 0.0
      lake_volume = 0.0
    end
    new(lake_parameters,lake_variables,lake_model_parameters,lake_model_prognostics,
        current_cell_to_fill,current_height_type,current_filling_cell_index,
        next_cell_volume_threshold,previous_cell_volume_threshold,lake_volume)
  end
end

struct OverflowingLake <: Lake
  parameters::LakeParameters
  variables::LakeVariables
  lake_model_parameters::LakeModelParameters
  lake_model_prognostics::LakeModelPrognostics
  current_redirect::Redirect
  execess_water::Float64
end

struct SubsumedLake <: Lake
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

function handle_event(lake::FillingLake,add_water::AddWater)
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
    lake_model_parameters::LakeModelParameters,
    lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  inflow::Float64 = add_water.inflow + lake_variables.unprocessed_water
  lake_variables.unprocessed_water = 0.0
  while inflow > 0.0
    if lake.lake_volume <= 0 && lake.current_filling_cell_index == 1
      surface_model_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(lake.current_cell_to_fill,
                                                  lake_model_parameters.grid_specific_lake_model_parameters)
      set!(lake_model_prognostics.lake_cell_count,surface_model_coords,
           lake_model_prognostics.lake_cell_count(surface_model_coords) + 1)
      lake_model_prognostics.flooded_lake_cells(lake.current_cell_to_fill) = true
    end
    new_lake_volume::Float64 = inflow + lake.lake_volume
    if new_lake_volume <= lake.next_cell_volume_threshold
      lake.lake_volume = lake.next_cell_volume_threshold
      inflow = 0.0
    else
      inflow = new_lake_volume - lake.next_cell_volume_threshold
      lake.lake_volume = lake.next_cell_volume_threshold
      if lake.current_filling_cell_index == length(lake.lake_parameters.filling_order)
        if check_if_merge_is_possible(lake_parameters,lake_model_prognostics)
          subsumed_lake::SubsumedLake = merge_lakes(lake,inflow,lake_parameters,
                                                    lake_model_prognostics)
          return subsumed_lake
        else
          overflowing_lake::OverflowingLake = change_to_overflowing_lake(lake,inflow)
          return overflowing_lake
      end
      lake.current_filling_cell_index += 1
      lake.previous_cell_volume_threshold = lake.next_cell_volume_threshold
      lake.next_cell_volume_threshold = \
        lake.lake_parameters.filling_order[lake.current_filling_cell_index].fill_threshold
      lake.previous_cell_to_fill = lake.current_cell_to_fill
      lake.current_cell_to_fill =
        lake.lake_parameters.filling_order[lake.current_filling_cell_index].coords
      if lake_model_prognostics.lake_numbers[lake.current_cell_to_fill] == 0
        lake_model_prognostics.lake_numbers[lake.current_cell_to_fill] =
          lake.parameters.lake_number
      end
      if lake.current_height_type == flood_height
        surface_model_coords::CartesianIndex =
          get_corresponding_surface_model_grid_cell(lake.current_cell_to_fill,
                                                    lake_model_parameter.grid_specific_lake_model_parameters)
        set!(lake_model_prognostics.lake_cell_count,surface_model_coords,
             lake_model_prognostics.lake_cell_count(surface_model_coords) + 1)
        lake_model_prognostics.flooded_lake_cells(lake.current_cell_to_fill) = true
      end
    end
  end
  return lake
end

function handle_event(lake::OverflowingLake,add_water::AddWater)
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
    ::LakeModelParameters,lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  inflow::Float64 = add_water.inflow +
                    lake_variables.unprocessed_water
  lake_variables.unprocessed_water = 0.0
  if lake.current_redirect.use_local_redirect
    other_lake_number::Int64 = lake.current_redirect.local_redirect_target_lake_number
    other_lake::Lake =
      lake_model_prognostics[other_lake_number]
    lake_model_prognostics[other_lake_number] = handle_event(other_lake,
                                                                  AddWater(inflow))
  else
    lake.excess_water += inflow
  end
  #Don't just return lake - instead use pointer to this lake in lakes
  #array in case adding water changed this lakes type
  return lake_model_prognostics[lake_parameters.lake_number]
end

function handle_event(lake::SubsumedLake,add_water::AddWater)
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
    ::LakeModelParameters,lake_model_prognostics::LakeModelPrognostics =
    get_lake_data(lake)
  inflow::Float64 = add_water.inflow + lake_variables.unprocessed_water
  lake_variables.unprocessed_water = 0.0
  if lake.redirect_target == -1
    lake.redirect_target = find_root(lake.lake_model_prognostics.set_forest,
                                     lake_parameters.lake_number)
  end
  lake_model_prognostics[lake.redirect_target] =
    handle_event(lake_model_prognostics[lake.redirect_target],
                 AddWater(inflow))
  return lake
end

function handle_event(lake::FillingLake,remove_water::RemoveWater)
  #if volume is zero and not leaf then remove from secondary lakes and
  #make them back into filling lakes
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
    lake_model_parameters::LakeModelParameters,
    lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  if remove_water.outflow <= lake.lake_variables.unprocessed_water
    lake.lake_variables.unprocessed_water -= remove_water.outflow
    return lake
  end
  outflow::Float64 = remove_water.outflow - lake.lake_variables.unprocessed_water
  lake.lake_variables.unprocessed_water = 0.0
  while outflow > 0.0
    new_lake_volume::Float64 = lake.lake_volume - outflow
    minimum_new_lake_volume::Float64 = lake.previous_cell_volume_threshold
    if new_lake_volume <= 0.0 && lake.current_filling_cell_index == 1
      if lake.is_leaf
        lake.lake_volume = new_lake_volume
      else
        lake.lake_volume = 0.0
        split_lake(lake,-new_lake_volume,lake_parameters,lake_variables)
      end
      if lake.current_height_type == flood_height
        surface_model_coords::CartesianIndex =
          get_corresponding_surface_model_grid_cell(lake.current_cell_to_fill,
                                                    lake_model_parameters.grid_specific_lake_model_parameters)
        set!(lake_model_prognostics.lake_cell_count,surface_model_coords,
             lake_model_prognostics.lake_cell_count(surface_model_coords) - 1)
        lake_model_prognostics.flooded_lake_cells(lake.current_cell_to_fill) = false
      end
    elseif new_lake_volume <=
        lake_model_parameters.lake_model_settings.minimum_lake_volume_threshold &&
       lake.current_filling_cell_index == 1 && lake.lake_parameters.is_leaf
      lake.lake_volume = 0.0
      set!(lake_model_prognostics.lake_water_from_ocean,
           lake_parameters.center_cell_coarse_coords,
           lake_model_prognostics.lake_water_from_ocean(lake_parameters.center_cell_coarse_coords) -
           new_lake_volume)
    elseif new_lake_volume >= minimum_new_lake_volume
      lake.lake_volume = new_lake_volume
    else
      outflow = minimum_new_lake_volume - new_lake_volume
      lake.lake_volume = minimum_new_lake_volume
      if lake.current_height_type == flood_height
        surface_model_coords::CartesianIndex =
          get_corresponding_surface_model_grid_cell(lake.current_cell_to_fill,
                                                    lake_model_parameters.grid_specific_lake_model_parameters)
        set!(lake_model_prognostics.lake_cell_count,surface_model_coords,
             lake_model_prognostics.lake_cell_count(surface_model_coords) - 1)
        lake_model_prognostics.flooded_lake_cells(lake.current_cell_to_fill) = false
      end
      if lake_model_prognostics.lake_numbers[lake.current_cell_to_fill] == lake.parameters.lake_number
        lake_model_prognostics.lake_numbers[lake.current_cell_to_fill] = 0
      end
      lake.current_filling_cell_index -= 1
      lake.next_cell_volume_threshold = lake.previous_cell_volume_threshold
      lake.previous_cell_volume_threshold = \
        lake.lake_parameters.filling_order[lake.current_filling_cell_index].fill_threshold
      lake.current_cell_to_fill = lake.previous_cell_to_fill
      lake.previous_cell_to_fill =
        lake.lake_parameters.filling_order[lake.current_filling_cell_index].coords
    end
  return lake
end

function handle_event(lake::OverflowingLake,remove_water::RemoveWater)
  ::LakeParameters,lake_variables::LakeVariables,
  ::LakeModelParameters,::LakeModelPrognostics = get_lake_data(lake)
  if remove_water.outflow <= lake_variables.unprocessed_water
    lake_variables.unprocessed_water -= remove_water.outflow
    return lake
  end
  outflow::Float64 = remove_water.outflow - lake_variables.unprocessed_water
  lake_variables.unprocessed_water = 0.0
  if outflow <= lake.excess_water
    lake.excess_water -= outflow
    return lake
  end
  outflow -= lake.excess_water
  lake.excess_water = 0.0
  lake_as_filling_lake::FillingLake = change_to_filling_lake(lake)
  return handle_event(lake_as_filling_lake,RemoveWater(outflow))
end

function handle_event(lake::SubsumedLake,remove_water::RemoveWater)
  #Redirect to primary
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
  ::LakeModelParameters,lake_model_prognostics::LakeModelPrognostics =
    get_lake_data(lake)
  if remove_water.outflow <= lake_variables.unprocessed_water
    lake_variables.unprocessed_water -= remove_water.outflow
    return lake
  end
  outflow::Float64 = remove_water.outflow - lake_variables.unprocessed_water
  lake_variables.unprocessed_water = 0.0
  if lake.redirect_target == -1
    lake.redirect_target = find_root(lake_model_prognostics.set_forest,
                                     lake_parameters.lake_number)
  end
  lake_model_prognostics.lakes[lake.redirect_target] =
    handle_event(lake_model_prognostics.lakes[lake.redirect_target],
                 RemoveWater(outflow))
  #Return this lake via a pointer in case it has changed type
  return lake_model_prognostics.lakes[lake_parameters.lake_number]::Lake
end

function check_if_merge_possible(lake_parameters::LakeParameters,
                                 lake_model_prognostics::LakeModelPrognostics)
  if ! lake_parameters.is_primary
    all_secondary_lakes_filled::Bool = false
    for secondary_lake in lakes[lake.lake_parameters.primary_lake].lake_parameters.secondary_lakes
      all_secondary_lakes_filled |= isa(lake_model_prognostics.lakes[secondary_lake.lake_number],OverflowingLake)
      #Check this is type stable
      println(typeof(all_secondary_lake_filled))
    end
    return all_secondary_lakes_filled
  else
    return false
  end
end

function merge_lakes(::FillingLake,inflow::Float64,
                     lake_parameters::LakeParameters,
                     lake_model_prognostics::LakeModelPrognostics)
  total_excess_water:Float64 = 0.0
  primary_lake::FillingLake = lakes[lake_parameters.primary_lake]
  primary_lake.lake_variables.active_lake = true
  for secondary_lake in primary_lake.lake_parameters.secondary_lakes
    overflowing_lake::OverflowingLake = lake_model_prognostics.lakes[secondary_lake.lake_number]
    subsumed_lake::SubsumedLake,excess_water::Float64 = change_to_subsumed_lake(overflowing_lake)
    lake_model_prognostics.lakes[secondary_lake.lake_number] = subsumed_lake
    total_excess_water += excess_water
  end
  handle_event(lake_model_prognostics.lakes[primary_lake.lake_parameters.lake_number],
               AddWater(inflow+total_excess_water))
  return lake_model_prognostics.lakes[lake_parameters.lake_number]
end

function split_lake(lake::FillingLake,water_deficit::Float64,
                    lake_parameters::LakeParameters,
                    lake_variables::LakeVariables)
  lake_variables.active_lake = false
  water_deficit_per_lake::Float64 = water_deficit/length(lake_parameters.secondary_lakes)
  for secondary_lake in lake.lake_parameters.secondary_lakes
    filling_lake::FillingLake = \
      change_to_filling_lake(lake_model_prognostics.lakes[secondary_lake],
                             lake_parameters.lake_number)
    lake_model_prognostics.lakes[secondary_lake.lake_number] = filling_lake
    handle_event(filling_lake,RemoveWater(water_deficit_per_lake))
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
  if ! lake_parameters.is_primary
    for secondary_lake in
        lake_model_prognostics.lakes[lake_parameters.primary_lake].lake_parameters.secondary_lakes
      redirect_target_found::Bool = false
      if isa(lake_model_prognostics.lakes[secondary_lake],FillingLake) &&
         secondary_lake != lake.lake_parameters.lake_number
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
    redirect = lake_parameters.outflow_points[-1]
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

function handle_event(lake::OverflowingLake,drain_excess_water::DrainExcessWater)
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
  lake_model_parameters::LakeModelParameters,
  lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  if lake.excess_water > 0.0
    lake.excess_water +=
      lake_variables.unprocessed_water
    lake_variables.unprocessed_water = 0.0
    total_lake_volume::Float64 = 0.0
    for_elements_in_set(lake_model_prognostics.set_forest,
                        find_root(lake_model_prognostics.set_forest,
                                  lake.lake_parameters.lake_number)
                        x -> total_lake_volume =
                        get_lake_volume(lake_model_prognostics.lakes[get_label(x)]))
    flow = (total_lake_volume)/
           (lake_model_parameters.lake_model_settings.lake_retention_coefficient + 1.0)
    flow = min(flow,lake.excess_water)
    set!(lake_model_prognostics.water_to_hd,lake.outflow_redirect_coords,
         lake_model_prognosticss.water_to_hd(lake.outflow_redirect_coords)+flow)
    lake.excess_water -= flow
  end
  return lake
end

function handle_event(lake::FillingLakeLake,::ReleaseNegativeWater)
  lake_parameters::LakeParameters,::LakeVariables,
    ::LakeModelParameters,lake_model_prognostics::LakeModelPrognostics = get_lake_data(lake)
  set!(lake_model_prognostics.lake_water_from_ocean,
       lake_parameters.center_cell_coarse_coords,
       lake_model_prognostics.lake_water_from_ocean(lake_parameters.center_cell_coarse_coords) -
       lake.lake_volume)
  lake.lake_volume = 0.0
  return lake
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

get_lake_volume(lake::FillingLake) = lake.lake_volume + lake.lake_variables.unprocessed_water

get_lake_volume(lake::OverflowingLake) = lake.parameters.filling_order[-1].fill_threshold +
                                         lake.excess_water,
                                         lake_variables.unprocessed_water

get_lake_volume(lake::SubsumedLake) = lake.parameters.filling_order[-1].fill_threshold +
                                      lake_variables.unprocessed_water

function show(io::IO,lake::Lake)
  lake_parameters::LakeParameters,lake_variables::LakeVariables,
    ::LakeModelParameters,::LakeModelPrognostics = get_lake_data(lake)
  println(io,"-----------------------------------------------")
  println(io,"Lake number: "*string(lake_parameters.lake_number))
  println(io,"Center cell: "*string(lake_parameters.center_cell))
  println(io,"Unprocessed water: "*string(lake_variables.unprocessed_water))
  println(io,"Center cell coarse coords: "*string(lake_parameters.center_cell_coarse_coords))
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
    println(io,"Prevoius cell volume threshold: "*string(lake.previous_cell_volume_threshold))
  elseif isa(lake,SubsumedLake)
    println(io,"S")
    println(io,"Redirect target : "*string(lake.redirect_target))
  end
end

end
