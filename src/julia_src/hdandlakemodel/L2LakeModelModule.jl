module L2LakeModelModule

using HierarchicalStateMachineModule: Event
using L2LakeModelDefsModule: Lake, LakeModelParameters, LakeModelPrognostics
using L2LakeModelDefsModule: LakeModelDiagnostics
using L2LakeModule: LakeParameters, LakeVariables, FillingLake, OverflowingLake
using L2LakeModule: DrainExcessWater, AddWater, RemoveWater, ProcessWater
using L2LakeModule: ReleaseNegativeWater
using L2LakeModule: handle_event, get_lake_volume
using L2LakeModule: get_corresponding_surface_model_grid_cell
using L2LakeModule: flood_height
using L2LakeArrayDecoderModule: get_lake_parameters_from_array
using CoordsModule: Coords,is_lake
using GridModule: LatLonGrid,UnstructuredGrid,for_all
using GridModule: for_all_fine_cells_in_coarse_cell
using HDModule: RiverParameters, PrognosticFields, RiverPrognosticFields, RiverDiagnosticFields
using HDModule: RiverDiagnosticOutputFields, PrintResults, PrintSection
using FieldModule: Field,set!,elementwise_divide,elementwise_multiple
using SplittableRootedTree: add_set,find_root,for_elements_in_set,get_label
import HierarchicalStateMachineModule: handle_event
import L2LakeModule: handle_event
import HDModule: water_to_lakes,water_from_lakes

struct SetupLakes <: Event
  initial_water_to_lake_centers::Field{Float64}
end

struct DistributeSpillover <: Event
  initial_spillover_to_rivers::Field{Float64}
end

struct RunLakes <: Event end

struct PrintSelectedLakes <: Event
  lakes_to_print::Array{Int64,1}
end

struct WriteLakeNumbers <: Event
  timestep::Int64
end

struct WriteLakeVolumes <: Event end

struct WriteDiagnosticLakeVolumes <: Event
  timestep::Int64
end

struct SetLakeEvaporation <: Event
  lake_evaporation::Field{Float64}
end

struct SetRealisticLakeEvaporation <: Event
  height_of_water_evaporated::Field{Float64}
end

struct CheckWaterBudget <: Event
  total_initial_water_volume::Float64
end

CheckWaterBudget() = CheckWaterBudget(0.0)

struct RiverAndLakePrognosticFields <: PrognosticFields
  river_parameters::RiverParameters
  river_fields::RiverPrognosticFields
  river_diagnostic_fields::RiverDiagnosticFields
  river_diagnostic_output_fields::RiverDiagnosticOutputFields
  lake_model_parameters::LakeModelParameters
  lake_model_prognostics::LakeModelPrognostics
  lake_model_diagnostics::LakeModelDiagnostics
  using_lakes::Bool
  function RiverAndLakePrognosticFields(river_parameters::RiverParameters,
                                        river_fields::RiverPrognosticFields,
                                        lake_model_parameters::LakeModelParameters,
                                        lake_model_prognostics::LakeModelPrognostics)
    #This only works with Coords and not if we use a CartesianIndex
    for_all(river_parameters.grid) do coords::Coords
      if is_lake(river_parameters.flow_directions(coords))
        set!(river_parameters.cascade_flag,coords,false)
      end
    end
    new(river_parameters,river_fields,RiverDiagnosticFields(river_parameters),
        RiverDiagnosticOutputFields(river_parameters),lake_model_parameters,
        lake_model_prognostics,LakeModelDiagnostics(),true)
  end
end

function create_lakes(lake_model_parameters::LakeModelParameters,
                      lake_model_prognostics::LakeModelPrognostics,
                      lake_parameters_as_array::Array{Float64})
  lake_parameters_array::Array{LakeParameters} =
    get_lake_parameters_from_array(lake_parameters_as_array,
                                   lake_model_parameters.lake_model_grid,
                                   lake_model_parameters.hd_model_grid;
                                   single_index=
                                   isa(lake_model_parameters.lake_model_grid,
                                       UnstructuredGrid))
  for i in eachindex(lake_parameters_array)
    lake_parameters::LakeParameters = lake_parameters_array[i]
    if i != lake_parameters.lake_number
      error("Lake number doesn't match position when creating lakes")
    end
    add_set(lake_model_prognostics.set_forest,lake_parameters.lake_number)
    lake::FillingLake = FillingLake(lake_parameters,
                                    LakeVariables(lake_parameters.is_leaf),
                                    lake_model_parameters,
                                    lake_model_prognostics,
                                    false)
    push!(lake_model_prognostics.lakes,lake)
    if lake_parameters.is_leaf && (lake.current_height_type == flood_height
                                   || length(lake_parameters.filling_order) == 1)
      surface_model_coords =
        get_corresponding_surface_model_grid_cell(lake.current_cell_to_fill,
                                                  lake_model_parameters.grid_specific_lake_model_parameters)
      set!(lake_model_prognostics.lake_cell_count,surface_model_coords,
           lake_model_prognostics.lake_cell_count(surface_model_coords) + 1)
      set!(lake_model_prognostics.lake_numbers,lake.current_cell_to_fill,
           lake.parameters.lake_number)
    end
  end
  for lake::Lake in lake_model_prognostics.lakes
      set!(lake_model_parameters.lake_centers,lake.parameters.center_coords,true)
  end
  for_all(lake_model_parameters.hd_model_grid;
          use_cartesian_index=true) do coords::CartesianIndex
    contains_lake::Bool = false
    for_all_fine_cells_in_coarse_cell(lake_model_parameters.lake_model_grid,
                                      lake_model_parameters.hd_model_grid,
                                      coords) do fine_coords::CartesianIndex
      if lake_model_parameters.lake_centers(fine_coords)
        contains_lake = true
      end
    end
    if contains_lake
      push!(lake_model_parameters.cells_with_lakes,coords)
    end
  end
  basin_number::Int64 = 1
  for_all(lake_model_parameters.hd_model_grid;
          use_cartesian_index=true) do coords::CartesianIndex
    basins_in_coarse_cell::Vector{Int64} = CartesianIndex[]
    basins_found::Bool = false
    for_all_fine_cells_in_coarse_cell(lake_model_parameters.lake_model_grid,
                                      lake_model_parameters.hd_model_grid,
                                      coords) do fine_coords::CartesianIndex
      if lake_model_parameters.lake_centers(fine_coords)
        for lake::Lake in lake_model_prognostics.lakes
          if lake.parameters.center_coords == fine_coords
            push!(basins_in_coarse_cell,lake.parameters.lake_number)
            basins_found = true
          end
        end
      end
    end
    if basins_found
      push!(lake_model_parameters.basins,basins_in_coarse_cell)
      set!(lake_model_parameters.basin_numbers,coords,basin_number)
      basin_number += 1
    end
  end
end

get_lake_model_prognostics(river_and_lake_prognostics::RiverAndLakePrognosticFields) =
                           river_and_lake_prognostics.lake_model_prognostics::LakeModelPrognostics

get_lake_model_parameters(river_and_lake_prognostics::RiverAndLakePrognosticFields) =
                          river_and_lake_prognostics.lake_model_parameters::LakeModelParameters

get_lake_model_diagnostics(river_and_lake_prognostics::RiverAndLakePrognosticFields) =
  river_and_lake_prognostics.lake_model_diagnostics::LakeModelDiagnostics

function water_to_lakes(prognostic_fields::RiverAndLakePrognosticFields)
  lake_prognostics::LakeModelPrognostics = get_lake_model_prognostics(prognostic_fields)
  return lake_prognostics.water_to_lakes
end

function water_from_lakes(prognostic_fields::RiverAndLakePrognosticFields,
                          step_length::Float64)
  lake_prognostics::LakeModelPrognostics = get_lake_model_prognostics(prognostic_fields)
  return lake_prognostics.water_to_hd/step_length,
         lake_prognostics.lake_water_from_ocean/step_length
end

function handle_event(prognostic_fields::RiverAndLakePrognosticFields,setup_lakes::SetupLakes)
  lake_model_parameters::LakeModelParameters = get_lake_model_parameters(prognostic_fields)
  lake_model_prognostics::LakeModelPrognostics = get_lake_model_prognostics(prognostic_fields)
  for_all(lake_model_parameters.lake_model_grid;
          use_cartesian_index=true) do coords::CartesianIndex
    if lake_model_parameters.lake_centers(coords)
      initial_water_to_lake_center::Float64 =
        setup_lakes.initial_water_to_lake_centers(coords)
      if  initial_water_to_lake_center > 0.0
        add_water::AddWater = AddWater(initial_water_to_lake_center,false)
        lake_index::Int64 = lake_model_prognostics.lake_numbers(coords)
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
          use_cartesian_index=true) do coords::CartesianIndex
    initial_spillover::Float64 = distribute_spillover.initial_spillover_to_rivers(coords)
    if initial_spillover > 0.0
      set!(lake_model_prognostics.water_to_hd,coords,
           lake_model_prognostic.water_to_hd(coords) + initial_spillover)
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
  for_all(lake_model_parameters.surface_model_grid;use_cartesian_index=true) do coords::CartesianIndex
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
            lake_model_prognostics.evaporation_from_lakes[target_lake_index] += evaporation_per_lake_cell
            cell_count_check += 1
          end
        end
      end
      if cell_count_check != lake_cell_count
        error("Inconsistent cell count when assigning evaporation")
      end
    end
  end
  fill!(lake_model_prognostics.evaporation_applied,false)
  for coords::CartesianIndex in lake_model_parameters.cells_with_lakes
    if lake_model_prognostics.water_to_lakes(coords) > 0.0
      lakes_in_cell =
        lake_model_parameters.basins[lake_model_parameters.basin_numbers(coords)]
      filter!(lake_index::Int64->
              lake_model_prognostics.lakes[lake_index].variables.active_lake,
              lakes_in_cell)
      share_to_each_lake::Float64 = lake_model_prognostics.water_to_lakes(coords)/length(lakes_in_cell)
      add_water::AddWater = AddWater(share_to_each_lake,true)
      for lake_index::Int64 in lakes_in_cell
        lake::Lake = lake_model_prognostics.lakes[lake_index]
        if ! lake_model_prognostics.evaporation_applied[lake_index]
          inflow_minus_evaporation::Float64 = share_to_each_lake -
                                              lake_model_prognostics.evaporation_from_lakes[lake_index]
          if inflow_minus_evaporation >= 0.0
            add_water_modified::AddWater = AddWater(inflow_minus_evaporation,true)
            lake_model_prognostics.lakes[lake_index] = handle_event(lake,add_water_modified)
          else
            remove_water_modified::RemoveWater =
              RemoveWater(-1.0*inflow_minus_evaporation,true)
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
      filter!(lake_index::Int64->
              lake_model_prognostics.lakes[lake_index].variables.active_lake,
              lakes_in_cell)
      share_to_each_lake = -1.0*lake_model_prognostics.water_to_lakes(coords)/length(lakes_in_cell)
      remove_water::RemoveWater = RemoveWater(share_to_each_lake,true)
      for lake_index::Int64 in lakes_in_cell
        lake::Lake = lake_model_prognostics.lakes[lake_index]
        lake_model_prognostics.lakes[lake_index] = handle_event(lake,remove_water)
      end
    end
  end
  for lake::Lake in lake_model_prognostics.lakes
    lake_index::Int64 = lake.parameters.lake_number
    if ! lake_model_prognostics.evaporation_applied[lake_index]
      evaporation::Float64 = lake_model_prognostics.evaporation_from_lakes[lake_index]
      if evaporation > 0
        lake_model_prognostics.lakes[lake_index] =
          handle_event(lake,RemoveWater(evaporation,true))
      elseif evaporation < 0
        lake_model_prognostics.lakes[lake_index] =
          handle_event(lake,AddWater(-1.0*evaporation,true))
      end
      lake_model_prognostics.evaporation_applied[lake_index] = true
    end
  end
  for lake::Lake in lake_model_prognostics.lakes
    lake_index::Int64 = lake.parameters.lake_number
    if lake.variables.unprocessed_water != 0.0
      lake_model_prognostics.lakes[lake_index] = handle_event(lake,ProcessWater())
    end
  end
  for lake::Lake in lake_model_prognostics.lakes
    if isa(lake,FillingLake)
      lake_index::Int64 = lake.parameters.lake_number
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
  # for lake::Lake in lake_model_prognostics.lakes
  #   lake_model_prognostics.lakes[lake.parameters.lake_number] =
  #     handle_event(lake,CalculateEffectiveLakeVolumePerCell())
  # end
  return prognostic_fields
end

function set_effective_lake_height_on_surface_grid_to_lakes(lake_model_parameters::LakeModelParameters,
                                                            lake_model_prognostics::LakeModelPrognostics,
                                                            effective_lake_height_on_surface_grid::Field{Float64})
  for_all(lake_model_parameters.surface_model_grid;use_cartesian_index=true) do coords::CartesianIndex
    set!(lake_model_prognostics.effective_lake_height_on_surface_grid_to_lakes,coords,
         effective_lake_height_on_surface_grid(coords))
  end
end

function calculate_lake_fraction_on_surface_grid(lake_model_parameters::LakeModelParameters,
                                                 lake_model_prognostics::LakeModelPrognostics)
  return elementwise_divide(lake_model_prognostics.lake_cell_count,
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
    println("Lake Center: $(lake.parameters.center_cell) "*
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
    if lake.variables.active_lake
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
    lake_center_cell::CartesianIndex = lake.parameters.center_cell
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

function calculate_diagnostic_lake_volumes_field(lake_model_parameters::LakeModelParameters,
                                                 lake_model_prognostics::LakeModelPrognostics)
  lake_volumes_by_lake_number::Vector{Float64} =
    zeros(Float64,length(lake_model_prognostics.lakes))
  diagnostic_lake_volumes::Field{Float64} =
    Field{Float64}(lake_model_parameters.lake_model_grid,0.0)
  for i::Int64 in eachindex(lake_model_prognostics.lakes)
    lake::Lake = lake_model_prognostics.lakes[i]
    if lake.variables.active_lake
      total_lake_volume::Float64 = 0.0
      for_elements_in_set(lake_model_prognostics.set_forest,
                          find_root(lake_model_prognostics.set_forest,
                                    lake.parameters.lake_number),
                          x -> total_lake_volume +=
                          get_lake_volume(lake_model_prognostics.lakes[get_label(x)]))
      lake_volumes_by_lake_number[i] = total_lake_volume
    end
  end
  for_all(lake_model_parameters.lake_model_grid;
          use_cartesian_index=true) do coords::CartesianIndex
    lake_number = lake_model_prognostics.lake_numbers(coords)
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
          use_cartesian_index=true) do coords::CartesianIndex
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
  lake_model_diagnostics::LakeModelDiagnostics = get_lake_model_diagnostics(prognostic_fields)
  new_total_lake_volume::Float64 = 0.0
  for lake::Lake in lake_model_prognostics.lakes
    new_total_lake_volume += get_lake_volume(lake)
  end
  change_in_total_lake_volume::Float64 = new_total_lake_volume -
                                         lake_model_diagnostics.total_lake_volume -
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
    println("Old total lake volume: "*string(lake_model_diagnostics.total_lake_volume))
  end
  lake_model_diagnostics.total_lake_volume = new_total_lake_volume
  return prognostic_fields
end

end
