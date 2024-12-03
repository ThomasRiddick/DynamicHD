module L2LakeModelModule

struct RunLakes <: Event end

struct DistributeSpillover <: Event
  initial_spillover_to_rivers::Field{Float64}
end

struct SetupLakes <: Event
  initial_water_to_lake_centers::Field{Float64}
end

struct WriteLakeNumbers <: Event
  timestep::Int64
end

struct WriteDiagnosticLakeVolumes <: Event
  timestep::Int64
end

struct WriteLakeVolumes <: Event end

struct CalculateEffectiveLakeVolumePerCell <: Event
  consider_secondary_lake::Bool
  cell_list::Vector{CartesianIndex}
  function CalculateEffectiveLakeVolumePerCell(cell_list::Vector{CartesianIndex})
    return new(true,cell_list)
  end
  function CalculateEffectiveLakeVolumePerCell()
    return new(false,CartesianIndex[])
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

struct LakeModelParameters
  grid_specific_lake_model_parameters::GridSpecificLakeModelParameters
  lake_model_grid::Grid
  surface_model_grid::Grid
  lake_model_settings::LakeModelSettings
  basins::Vector{Vector{CartesianIndex}}
  basin_numbers::Field{Int64}
  cell_areas_on_surface_model_grid::Field{Float64}
  lake_centers::Field{Bool}
  number_fine_grid_cells::Field{Int64}
  surface_cell_to_fine_cell_map::Vector{CartesianIndex}
  surface_cell_to_fine_cell_map_numbers::Field{Int64}
  function LakeModelParameters(grid_specific_lake_model_parameters::GridSpecificLakeModelParameters,
                               lake_model_grid::Grid,
                               hd_model_grid::Grid,
                               surface_model_grid::Grid,
                               lake_model_settings::LakeModelSettings,
                               lake_centers::Field{Bool},
                               is_lake::Field{Bool})
    number_fine_grid_cells::Field{Int64} =
      Field{Int64}(lake_parameters.lake_model_grid,0)
    for_all(lake_model_grid;use_cartestian_index=true) do coords::CartesianIndex
      surface_model_coords::CartesianIndex =
        get_corresponding_surface_model_grid_cell(coords,grid_specific_lake_parameters)
      set!(number_fine_grid_cells,
           surface_model_coords,
           number_fine_grid_cells(surface_model_coords) + 1)
      if is_lake(coords)
        surface_cell_to_fine_cell_map::Vector{CartesianIndex} = CartesianIndex[]
        surface_cell_to_fine_cell_map_number::Int64 =
          surface_cell_to_fine_cell_map_numbers(surface_model_coords)
        if  surface_cell_to_fine_cell_map_number == 0
          surface_cell_to_fine_cell_map = CartesianIndex(coords)
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
    basins::Vector{Vector{CartesianIndex}} = Vector{CartesianIndex}[]
    basin_numbers::Field{Int64} = Field{Int64}(hd_model_grid)
    basin_number::Int64 = 1
    for_all(hd_model_grid;
            use_cartestian_index=true) do coords::CartesianIndex
      basins_in_coarse_cell::Vector{CartesianIndex} = CartesianIndex[]
      basins_found::Bool = false
      for_all_fine_cells_in_coarse_cell(grid,hd_model_grid,
                                        coords) do fine_coords::CartesianIndex
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
    new(grid_specific_lake_model_parameters,
        surface_model_grid,
        lake_model_grid,
        lake_model_settings,
        basins,
        basin_numbers,
        cell_areas_on_surface_model_grid,
        lake_centers,
        number_fine_grid_cells,
        surface_cell_to_fine_cell_map,
        surface_cell_to_fine_cell_map_numbers)
  end
end

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

struct LakeModelPrognostics
  lakes::Vector{Lake}
  lake_numbers::Field{Int64}
  lake_cell_count::Field{Int64}
  #Includes the currently filling cell if that is a flood height
  flooded_lake_cells::Field{Bool}
  effective_volume_per_cell_on_surface_grid::Field{Float64}
  effective_lake_height_on_surface_grid_to_lakes::Field{Float64}
  water_to_lakes::Field{Float64}
  water_to_hd::Field{Float64}
  lake_water_from_ocean::Field{Float64}
  cells_with_lakes::Vector{CartesianIndex}
  evaporation_from_lakes::Array{Float64,1}
  evaporation_applied::BitArray{1}
  set_forest::RootedTreeForest
  function LakeModelPrognostics(lake_model_parameters::LakeModelParameters)
    lakes::Vector{Lake} = Lake[]
    flooded_lake_cells::Field{Bool} =
      Field{Bool}(lake_model_parameters.lake_model_grid,false)
    lake_numbers = Field{Int64}(lake_model_parameters.lake_model_grid,0)
    effective_volume_per_cell_on_surface_grid::Field{Float64} =
      Field{Float64}(lake_parameters.surface_model_grid,0.0)
    effective_lake_height_on_surface_grid_to_lakes::Field{Float64} =
      Field{Float64}(lake_parameters.surface_model_grid,0.0)
    water_to_lakes::Field{Float64} =
      Field{Float64}(lake_model_parameters.hd_model_grid,0.0)
    water_to_hd::Field{Float64} =
      Field{Float64}(lake_model_parameters.hd_model_grid,0.0)
    lake_water_from_ocean::Field{Float64} =
      Field{Float64}(lake_model_parameters.hd_model_grid,0.0)
    lake_cell_count::Field{Int64} =
      Field{Int64}(lake_parameters.surface_model_grid,0)
    cells_with_lakes::Vector{CartesianIndex} = CartesianIndex[]
    for_all(lake_model_parameters.hd_model_grid;
            use_cartestian_index=true) do coords::CartesianIndex
      contains_lake::Bool = false
      for_all_fine_cells_in_coarse_cell(lake_model_parameters.lake_model_grid,
                                        lake_model_parameters.hd_model_grid,
                                        coords) do fine_coords::CartesianIndex
        if lake_parameters.lake_centers(fine_coords)
          contains_lake = true
        end
      end
      if contains_lake
        push!(cells_with_lakes,coords)
      end
    end
    evaporation_from_lakes::Array{Float64,1} =
      zeros(Float64,count(lake_parameters.lake_centers))
    evaporation_applied = falses(count(lake_parameters.lake_centers))
    set_forest::RootedTreeForest = RootedTreeForest()
    new(lakes,lake_numbers,lake_cell_count,flooded_lake_cells,
        effective_volume_per_cell_on_surface_grid,
        effective_lake_height_on_surface_grid_to_lakes,
        water_to_lakes,water_to_hd,lake_water_from_ocean,
        cells_with_lakes,evaporation_from_lakes,evaporation_applied,
        set_forest)
  end
end

function create_lakes(lake_model_parameters::LakeModelParameters,
                      lake_model_prognostics::LakeModelPrognostics,
                      lake_parameters_as_array::Array{Float64})
  lake_parameters_array::Array{LakeParameters} =
    get_lake_parameters_from_array(lake_parameters_as_array;
                                   single_index=
                                   isa(lake_model_parameters.lake_model_grid,
                                       UnstructuredGrid))
  for i in eachindex(lake_parameters_array)
    lake_parameters::LakeParameters = lake_parameters_array[i]
    if i != lake_parameters.lake_number
      error("Lake number doesn't match position when creating lakes")
    end
    push!(lake_model_prognostics.lakes,
          FillingLake(lake_parameters,
                      LakeVariables(lake_parameters.is_leaf),
                      lake_model_parameters,
                      lake_model_prognostics,
                      false))
  end
end

struct LakeModelDiagnostics
  total_lake_volume::Float64
end

struct RiverAndLakePrognosticFields <: PrognosticFields
  river_parameters::RiverParameters
  river_fields::RiverPrognosticFields
  river_diagnostic_fields::RiverDiagnosticFields
  river_diagnostic_output_fields::RiverDiagnosticOutputFields
  lake_model_parameters::LakeModelParameters
  lake_model_prognostics::LakeModelPrognostics
  using_lakes::Bool
  function RiverAndLakePrognosticFields(river_parameters::RiverParameters,
                                        river_fields::RiverPrognosticFields,
                                        lake_model_parameters::LakeParameters,
                                        lake_model_prognostics::LakePrognostics)
    for_all(river_parameters.grid;
            use_cartestian_index=true) do coords::CartesianIndex
      if is_lake(river_parameters.flow_directions(coords))
        set!(river_parameters.cascade_flag,coords,false)
      end
    end
    new(river_parameters,river_fields,RiverDiagnosticFields(river_parameters),
        RiverDiagnosticOutputFields(river_parameters),lake_model_parameters,
        lake_model_prognostics,true)
  end
end

get_lake_model_prognostics(river_and_lake_prognostics::RiverAndLakePrognosticFields) =
                           river_and_lake_prognostics.lake_model_prognostics::LakeModelPrognostics

get_lake_model_parameters(river_and_lake_prognostics::RiverAndLakePrognosticFields) =
                          river_and_lake_prognostics.lake_model_parameters::LakeModelParameters

function water_to_lakes(prognostic_fields::RiverAndLakePrognosticFields)
  lake_prognostics::LakePrognostics = get_lake_prognostics(prognostic_fields)
  return lake_prognostics.water_to_lakes
end

function water_from_lakes(prognostic_fields::RiverAndLakePrognosticFields,
                          step_length::Float64)
  lake_prognostics::LakePrognostics = get_lake_prognostics(prognostic_fields)
  return lake_prognostics.water_to_hd/step_length,
         lake_prognostics.lake_water_from_ocean/step_length
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,setup_lakes::SetupLakes)
  lake_model_prognostics::LakeModelPrognostics = get_lake_model_prognostics(prognostic_fields)
  lake_model_parameters::LakeModelParameters = get_lake_model_parameters(prognostic_fields)
  for_all(lake_model_parameter.lake_model_grid;
          use_cartestian_index=true) do coords::CartesianIndex
    if lake_model_parameters.lake_centers(coords)
      initial_water_to_lake_center::Float64 =
        setup_lakes.initial_water_to_lake_centers(coords)
      if  initial_water_to_lake_center > 0.0
        add_water::AddWater = AddWater(initial_water_to_lake_center)
        lake_index::Int64 = lake_fields.lake_numbers(coords)
        lake::Lake = lake_prognostics.lakes[lake_index]
        lake_model_prognostics.lakes[lake_index] = handle_event(lake,add_water)
      end
    end
  end
  return prognostic_fields
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      distribute_spillover::DistributeSpillover)
  lake_model_prognostics::LakeModelPrognostics = get_lake_model_prognostics(prognostic_fields)
  lake_model_parameters::LakeParameters = get_lake_model_parameters(prognostic_fields)
  for_all(lake_model_parameters.hd_model_grid;
          use_cartestian_index=true) do coords::CartesianIndex
    initial_spillover::Float64 = distribute_spillover.initial_spillover_to_rivers(coords)
    if initial_spillover > 0.0
      set!(lake_model_prognostics.water_to_hd,coords,
           lake_model_prognosticss.water_to_hd(coords) + initial_spillover)
    end
  end
  return prognostic_fields
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,::RunLakes)
  lake_model_prognostics::LakeModelPrognostics = get_lake_model_prognostics(prognostic_fields)
  lake_model_parameters::LakeModelParameters = get_lake_model_parameters(prognostic_fields)
  drain_excess_water::DrainExcessWater = DrainExcessWater()
  fill!(lake_model_prognostics.water_to_hd,0.0)
  fill!(lake_model_prognostics.lake_water_from_ocean,0.0)
  fill!(lake_model_prognostics.evaporation_from_lakes,0.0)
  new_effective_volume_per_cell_on_surface_grid::Field{Float64} =
    elementwise_multiple(lake_model_prognostics.effective_lake_height_on_surface_grid_to_lakes,
                         lake_model_parameters.cell_areas_on_surface_model_grid)
  evaporation_on_surface_grid::Field{Float64} = lake_model_prognostics.effective_volume_per_cell_on_surface_grid -
                                                new_effective_volume_per_cell_on_surface_grid
  fill!(lake_model_prognostics.effective_volume_per_cell_on_surface_grid,0.0)
  for_all(lake_model_parameters.surface_model_grid;use_cartestian_index=true) do coords::CartesianIndex
    lake_cell_count::Int64 = lake_model_prognostics.lake_cell_count(coords)
    if evaporation_on_surface_grid(coords) != 0.0 && lake_cell_count > 0
      cell_count_check::Int64 = 0
      map_index::Int64 = lake_model_parameters.surface_cell_to_fine_cell_map_numbers(coords)
      evaporation_per_lake_cell::Float64 = evaporation_on_surface_grid(coords)/lake_cell_count
      if map_index != 0
        for fine_coords::CartesianIndex in lake_model_parameters.surface_cell_to_fine_cell_maps[map_index]
          target_lake_index::Int64 = lake_model_prognostics.lake_numbers(fine_coords)
          if target_lake_index != 0
            lake::Lake = lake_model_prognostics.lakes[target_lake_index]
            if lake_model_prognostics.flooded_lake_cells(fine_coords)
              lake_model_prognostics.evaporation_from_lakes[target_lake_index] += evaporation_per_lake_cell
              cell_count_check += 1
            end
          end
        end
      end
      if cell_count_check != lake_cell_count
        error("Inconsistent cell count when assigning evaporation")
      end
    end
  end
  fill!(lake_model_prognostics.evaporation_applied,false)
  for coords::CartesianIndex in lake_model_prognostics.cells_with_lakes
    if lake_model_prognostics.water_to_lakes(coords) > 0.0
      lakes_in_cell =
        lake_model_parameters.basins[lake_model_parameters.basin_numbers(coords)]
      filter!(lake_index->lake_model_prognostics.lakes[lake_index].lake_variables.active_lake,
              lakes_in_cell)
      share_to_each_lake::Float64 = lake_model_prognostics.water_to_lakes(coords)/length(lakes_in_cell)
      add_water::AddWater = AddWater(share_to_each_lake)
      for lake_index::Int64 in lakes_in_cell
        lake::Lake = lake_model_prognostics.lakes[lake_index]
        if ! lake_model_prognostics.evaporation_applied[lake_index]
          inflow_minus_evaporation::Float64 = share_to_each_lake -
                                              lake_model_prognostics.evaporation_from_lakes[lake_index]
          if inflow_minus_evaporation >= 0.0
            add_water_modified::AddWater = AddWater(inflow_minus_evaporation)
            lake_model_prognostics.lakes[lake_index] = handle_event(lake,add_water_modified)
          else
            remove_water_modified::RemoveWater = RemoveWater(-1.0*inflow_minus_evaporation)
            lake_model_prognostics.lakes[lake_index] = handle_event(lake,remove_water_modified)
          end
          lake_model_prognostics.evaporation_applied[lake_index] = true
        else
          lake_model_prognostics.lakes[lake_index] = handle_event(lake,add_water)
        end
      end
    elseif lake_model_prognostics.water_to_lakes(coords) < 0.0
      lakes_in_cell =
        lake_model_parameters.basins[lake_model_parameters.basin_numbers(coords)]
      share_to_each_lake = -1.0*lake_model_prognostics.water_to_lakes(coords)/length(lakes_in_cell)
      remove_water::RemoveWater = RemoveWater(share_to_each_lake)
      for lake_index::Int64 in lakes_in_cell
        lake::Lake = lake_model_prognostics.lakes[lake_index]
        lake_model_prognostics.lakes[lake_index] = handle_event(lake,remove_water)
      end
    end
  end
  for lake::Lake in lake_model_prognostics.lakes
    lake_variables::LakeVariables = get_lake_variables(lake)
    lake_index::Int64 = lake_parameters.lake_number
    if lake_variables.unprocessed_water > 0.0
      lake_model_prognostics.lakes[lake_index] = handle_event(lake,AddWater(0.0))
    end
  end
  for lake::Lake in lake_model_prognostics.lakes
    lake_index::Int64 = lake_parameters.lake_number
    if ! lake_model_prognostics.evaporation_applied[lake_index]
      evaporation::Float64 = lake_model_prognostics.evaporation_from_lakes[lake_index]
      if evaporation > 0
        lake_model_prognostics.lakes[lake_index] = handle_event(lake,RemoveWater(evaporation))
      elseif evaporation < 0
        lake_model_prognostics.lakes[lake_index] = handle_event(lake,AddWater(-1.0*evaporation))
      end
      lake_model_prognostics.evaporation_applied[lake_index] = true
    end
  end
  for lake::Lake in lake_model_prognostics.lakes
    if isa(lake,FillingLake)
      lake_index::Int64 = lake.lake_parameters.lake_number
      if lake.lake_volume < 0.0
        lake_model_prognostics.lakes[lake_index] = handle_event(lake,ReleaseNegativeWater())
      end
    end
  end
  for lake::Lake in lake_model_prognostics.lakes
    if isa(lake,OverflowingLake)
      handle_event(lake,drain_excess_water)
    end
  end
  for lake::Lake in lake_model_prognostics.lakes
    lake_model_prognostics.lakes[lake.parameters.lake_number] =
      handle_event(lake,CalculateEffectiveLakeVolumePerCell())
  end
  return prognostic_fields
end

function set_effective_lake_height_on_surface_grid_to_lakes(lake_model_parameters::LakeModelParameters,
                                                            lake_model_prognostics::LakeModelPrognostics,
                                                            effective_lake_height_on_surface_grid::Field{Float64})
  for_all(lake_model_parameters.surface_model_grid;use_cartestian_index=true) do coords::CartesianIndex
    set!(lake_model_prognostics.effective_lake_height_on_surface_grid_to_lakes,coords,
         effective_lake_height_on_surface_grid(coords))
  end
end

function calculate_lake_fraction_on_surface_grid(lake_model_parameters::LakeModelParameters,
                                                 lake_model_prognostics::LakeModelPrognostics)
  return elementwise_divide(lake_model_prognostics.number_lake_cells,
                            lake_model_parameters.number_fine_grid_cells)::Field{Float64}
end

function calculate_effective_lake_height_on_surface_grid(lake_model_parameters::LakeModelParameters,
                                                         lake_model_prognostics::LakeModelPrognostics)
  return elementwise_divide(lake_model_prognostics.effective_volume_per_cell_on_surface_grid,
                            lake_model_parameters.cell_areas_on_surface_model_grid)::Field{Float64}
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      print_results::PrintResults)
  lake_model_prognostics::LakeModelPrognostics = get_lake_model_prognostics(prognostic_fields)
  lake_model_parameters::LakeModelParameters = get_lake_model_parameters(prognostic_fields)
  println("Timestep: $(print_results.timestep)")
  print_river_results(prognostic_fields)
  println("")
  println("Water to HD")
  println(lake_model_prognostics.water_to_hd)
  for lake in lake_model_prognostics.lakes
    println("Lake Center: $(lake.lake_parameters.center_cell) "*
            "Lake Volume: $(get_lake_volume(lake))")
  end
  println("")
  println("Diagnostic Lake Volumes:")
  print_lake_types(lake_model_parameters.grid,lake_model_prognostics)
  println("")
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  println(diagnostic_lake_volumes)
  return prognostic_fields
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      print_selected_lakes::PrintSelectedLakes)
  lake_model_prognostics::LakePrognostics = get_lake_model_prognostics(prognostic_fields)
  for lake_number in print_selected_lakes.lakes_to_print
    if length(lake_model_prognostics.lakes) >= lake_number
      lake::Lake = lake_model_prognostics.lakes[lake_number]
      println(lake)
    end
  end
  return prognostic_fields
end

function print_lake_types(grid::LatLonGrid,
                          lake_model_prognostics::LakeModelPrognostics)
  for_all_with_line_breaks(grid) do coords::Coords
    print_lake_type(coords,lake_model_prognostics)
  end
end

function print_lake_type(coords::Coords,
                         lake_model_prognostics::LakeModelPrognostics)
  lake_number::Int64 = lake_model_prognostics.lake_numbers(coords)
  if lake_number == 0
    print("- ")
  else
    lake::Lake = lake_prognostics.lakes[lake_number]
    if lake.lake_variables.active_lake
      if isa(lake,FillingLake)
        print("F ")
      elseif isa(lake,OverflowingLake)
        print("O ")
      elseif isa(lake,SubsumedLake)
        print("S ")
      else
        print("U ")
      end
    else
      print("- ")
    end
  end
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      ::PrintSection)
  lake_model_parameters::LakeParameters = get_lake_parameters(prognostic_fields)
  lake_model_prognostics::LakePrognostics = get_lake_model_prognostics(prognostic_fields)
  println("")
  print_lake_types_section(lake_model_parameters.grid,lake_model_prognostics)
  return prognostic_fields
end

function print_lake_types_section(grid::LatLonGrid,
                                  lake_model_prognostics::LakeModelPrognostics)
  section_coords::LatLonSectionCoords = LatLonSectionCoords(225,295,530,580)
  for_section_with_line_breaks(grid,section_coords) do coords::Coords
    print_lake_type(coords,lake_model_prognostics)
  end
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      write_lake_numbers::WriteLakeNumbers)
  lake_model_parameters::LakeParameters = get_lake_parameters(prognostic_fields)
  lake_model_prognostics::LakePrognostics = get_lake_model_prognostics(prognostic_fields)
  write_lake_numbers_field(lake_model_parameters.lake_model_grid,
                           lake_model_prognostics.lake_numbers,
                           timestep=write_lake_numbers.timestep)
  return prognostic_fields
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      ::WriteLakeVolumes)
  lake_model_parameters::LakeModelParameters = get_lake_model_parameters(prognostic_fields)
  lake_model_prognostics::LakeModelPrognostics = get_lake_model_prognostics(prognostic_fields)
  lake_volumes::Field{Float64} =
    Field{Float64}(lake_model_parameters.lake_model_grid,0.0)
  for lake::Lake in lake_model_prognostics.lakes
    lake_center_cell::CartesianIndex = lake.lake_parameters.center_cell
    set!(lake_volumes,lake_center_cell,get_lake_volume(lake))
  end
  write_lake_volumes_field(lake_model_parameters.lake_model_grid,lake_volumes)
  return prognostic_fields
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      write_diagnostic_lake_volumes::WriteDiagnosticLakeVolumes)
  lake_model_parameters::LakeModelParameters = get_lake_model_parameters(prognostic_fields)
  lake_model_prognostics::LakeModelPrognostics = get_lake_model_prognostics(prognostic_fields)
  diagnostic_lake_volumes::Field{Float64} =
    calculate_diagnostic_lake_volumes_field(lake_model_parameters,
                                            lake_model_prognostics)
  write_diagnostic_lake_volumes_field(lake_model_parameters.lake_model_grid,
                                      diagnostic_lake_volumes,
                                      timestep=write_diagnostic_lake_volumes.timestep)
  return prognostic_fields
end

function calculate_diagnostic_lake_volumes_field(lake_model_parameters::LakeParameters,
                                                 lake_model_prognostics::LakePrognostics)
  lake_volumes_by_lake_number::Vector{Float64} = zeros(Float64,length(lake_prognostics.lakes))
  diagnostic_lake_volumes::Field{Float64} =
    Field{Float64}(lake_model_parameters.lake_model_grid,0.0)
  for i::CartesianIndex in eachindex(lake_prognostics.lakes)
    lake::Lake = lake_model_prognostics.lakes[i]
    if lake.lake_variables.active_lake
      total_lake_volume::Float64 = 0.0
      for_elements_in_set(lake_model_prognostics.set_forest,
                          find_root(lake_model_prognostics.set_forest,
                                    lake.lake_parameters.lake_number)
                          x -> total_lake_volume =
                          get_lake_volume(lake_model_prognostics.lakes[get_label(x)]))
      lake_volumes_by_lake_number[i] = total_lake_volume
    end
  end
  for_all(lake_model_parameters.lake_model_grid;
          use_cartestian_index=true) do coords::CartesianIndex
    lake_number = lake_fields.lake_numbers(coords)
    if (lake_number > 0)
      set!(diagnostic_lake_volumes,coords,lake_volumes_by_lake_number[lake_number])
    end
  end
  return diagnostic_lake_volumes
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      set_lake_evaporation::SetLakeEvaporation)
  lake_model_parameters::LakeModelParameters = get_lake_model_parameters(prognostic_fields)
  lake_model_prognostics::LakeModelPrognostics = get_lake_model_prognostics(prognostic_fields)
  old_effective_lake_height_on_surface_grid::Field{Float64} =
    calculate_effective_lake_height_on_surface_grid(lake_model_parameters,lake_model_prognostics)
  effective_lake_height_on_surface_grid::Field{Float64} =
    old_effective_lake_height_on_surface_grid -
    elementwise_divide(set_lake_evaporation.lake_evaporation,
                       lake_model_parameters.cell_areas_on_surface_model_grid)
  set_effective_lake_height_on_surface_grid_to_lakes(lake_model_parameters,lake_model_prognostics,
                                                     effective_lake_height_on_surface_grid)
  return prognostic_fields
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      set_lake_evaporation::SetRealisticLakeEvaporation)
  lake_model_parameters::LakeModelParameters = get_lake_model_parameters(prognostic_fields)
  lake_model_prognostics::LakeModelPrognostics = get_lake_model_prognostics(prognostic_fields)
  old_effective_lake_height_on_surface_grid::Field{Float64} =
    calculate_effective_lake_height_on_surface_grid(lake_model_parameters,lake_fields)
  effective_lake_height_on_surface_grid::Field{Float64} = old_effective_lake_height_on_surface_grid
  for_all(lake_model_parameters.surface_model_grid,
          use_cartestian_index=true) do coords::CartesianIndex
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
  set_effective_lake_height_on_surface_grid_to_lakes(lake_model_parameters,lake_model_prognostics,
                                                     effective_lake_height_on_surface_grid)
  return prognostic_fields
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,
                      check_water_budget::CheckWaterBudget)
  lake_model_prognostics::LakeModelPrognostics = get_lake_model_prognostics(prognostic_fields)
  new_total_lake_volume::Float64 = 0.0
  for lake::Lake in lake_prognostics.lakes
    new_total_lake_volume += get_lake_volume(lake)
  end
  change_in_total_lake_volume::Float64 = new_total_lake_volume -
                                         lake_diagnostics_variables.total_lake_volume -
                                         check_water_budget.
                                         total_initial_water_volume
  total_water_to_lakes::Float64 = sum(lake_model_prognostics.water_to_lakes)
  total_inflow_minus_outflow::Float64 = total_water_to_lakes +
                                        sum(lake_model_prognostics.lake_water_from_ocean) -
                                        sum(lake_model_prognostics.water_to_hd) -
                                        sum(lake_model_prognostics.evaporation_from_lakes)
  difference::Float64 = change_in_total_lake_volume - total_inflow_minus_outflow
  tolerance::Float64 = 10e-15*max(new_total_lake_volume, total_water_to_lakes)
  if ! isapprox(difference,0,atol=tolerance)
    println("*** Lake Water Budget ***")
    println("Total lake volume: $(new_total_lake_volume)")
    println("Total inflow - outflow: $(total_inflow_minus_outflow)")
    println("Change in lake volume: $(change_in_total_lake_volume)")
    println("Difference: $(difference)")
    println("Total water to lakes: $total_water_to_lakes")
    println("Total water from ocean: "*string(sum(lake_model_prognostics.lake_water_from_ocean)))
    println("Total water to HD model from lakes: "*string(sum(lake_model_prognostics.water_to_hd)))
    println("Old total lake volume: "*string(lake_diagnostic_variables.total_lake_volume))
  end
  lake_diagnostic_variables.total_lake_volume = new_total_lake_volume
  return prognostic_fields
end

end
