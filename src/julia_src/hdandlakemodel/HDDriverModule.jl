module HDDriverModule

using Profile
using HierarchicalStateMachineModule: HierarchicalStateMachine
using HDModule: RiverParameters,RiverPrognosticFields,RiverPrognosticFieldsOnly,RunHD,handle_event
using HDModule: SetDrainage,SetRunoff,PrintResults,PrognosticFields,WriteRiverInitialValues
using HDModule: WriteRiverFlow,AccumulateRiverFlow,ResetCumulativeRiverFlow,WriteMeanRiverFlow
using HDModule: PrintGlobalValues, water_to_lakes,water_from_lakes,get_river_parameters
using FieldModule: Field,elementwise_multiple
using LakeModule: LakeParameters,LakePrognostics,LakeFields,RiverAndLakePrognosticFields,RunLakes
using LakeModule: PrintSection,WriteLakeNumbers,WriteLakeVolumes,SetupLakes,DistributeSpillover
using LakeModule: WriteDiagnosticLakeVolumes,CheckWaterBudget,SetLakeEvaporation,Lake
using LakeModule: SetRealisticLakeEvaporation,PrintSelectedLakes,CalculateTrueLakeDepths
using LakeModule: water_to_lakes,handle_event,water_from_lakes
using LakeModule: calculate_lake_fraction_on_surface_grid
using GridModule: get_number_of_cells

FloatFieldOrNothing = Union{Field{Float64},Nothing}
VectorOfVectorOfFloatsOrNothing = Union{Vector{Vector{Float64}},Nothing}

function drive_hd_model_with_or_without_lakes(prognostic_fields::PrognosticFields,
                                              drainages::Array{Field{Float64},1},
                                              runoffs::Array{Field{Float64},1},
                                              lake_evaporations::Array{Field{Float64},1},
                                              timesteps::Int64,run_lakes_flag::Bool,
                                              process_initial_lake_water::Bool=false,
                                              initial_water_to_lake_centers::FloatFieldOrNothing=
                                              nothing,
                                              initial_spillover_to_rivers::FloatFieldOrNothing=
                                              nothing;
                                              print_timestep_results::Bool=true,
                                              output_timestep::Int64=100,
                                              write_output::Bool=true,
                                              use_realistic_surface_coupling::Bool=false,
                                              return_lake_volumes::Bool=false,
                                              diagnostic_lake_volumes::VectorOfVectorOfFloatsOrNothing=
                                              nothing,
                                              input_data_is_monthly_mean::Bool=true)
  ncells::Int64 = get_number_of_cells(get_river_parameters(prognostic_fields).grid)
  hsm::HierarchicalStateMachine = HierarchicalStateMachine(prognostic_fields)
  runHD::RunHD = RunHD()
  if run_lakes_flag
    run_lakes::RunLakes = RunLakes()
    if process_initial_lake_water
      setup_lakes::SetupLakes = SetupLakes(initial_water_to_lake_centers)
      distribute_spillover::DistributeSpillover =
        DistributeSpillover(initial_spillover_to_rivers)
      handle_event(hsm,setup_lakes)
      # Run lakes once to cascade water from any overflowing lakes
      # and process through flow if not using instant through flow
      handle_event(hsm,run_lakes)
      # This must go after the initial lake run as the lake run
      # will reset the water to hd.
      handle_event(hsm,distribute_spillover)
      handle_event(hsm,CheckWaterBudget(sum(initial_water_to_lake_centers)+
                                        sum(initial_spillover_to_rivers)))
    end
  end
  for i in 1:timesteps
    if ! input_data_is_monthly_mean
      set_drainage = SetDrainage(drainages[i])
    elseif size(drainages) == (1,)
      set_drainage = SetDrainage(deepcopy(drainages[1]))
    else
      set_drainage = SetDrainage(deepcopy(drainages[convert(Int64,ceil(convert(Float64,i)/30.0))]))
    end
    handle_event(hsm,set_drainage)
    if ! input_data_is_monthly_mean
      set_runoff = SetRunoff(runoffs[i])
    elseif size(runoffs) == (1,)
      set_runoff = SetRunoff(deepcopy(runoffs[1]))
    else
      set_runoff = SetRunoff(deepcopy(runoffs[convert(Int64,ceil(convert(Float64,i)/30.0))]))
    end
    handle_event(hsm,set_runoff)
    if use_realistic_surface_coupling
      lake_fraction_on_surface_grid::Field{Float64} =
        calculate_lake_fraction_on_surface_grid(prognostic_fields.lake_parameters,
                                                prognostic_fields.lake_fields)
      local lake_evaporation::Field{Float64}
      if ! input_data_is_monthly_mean
        lake_evaporation = lake_evaporations[i]
      elseif size(lake_evaporations) == (1,)
        lake_evaporation = lake_evaporations[1]
      else
        lake_evaporation = lake_evaporations[convert(Int64,ceil(convert(Float64,i)/30.0))]
      end
      if ! input_data_is_monthly_mean
       set_lake_evaporation =
          SetRealisticLakeEvaporation(lake_evaporation)
      else
        fraction_adjusted_lake_evaporation::Field{Float64} =
          elementwise_multiple(lake_fraction_on_surface_grid,lake_evaporation)
        set_lake_evaporation =
          SetRealisticLakeEvaporation(fraction_adjusted_lake_evaporation)
      end
    else
      if ! input_data_is_monthly_mean
        lake_evaporation = SetLakeEvaporation(lake_evaporations[i])
      elseif size(lake_evaporations) == (1,)
        set_lake_evaporation =
          SetLakeEvaporation(deepcopy(lake_evaporations[1]))
      else
        set_lake_evaporation =
          SetLakeEvaporation(deepcopy(lake_evaporations[convert(Int64,ceil(convert(Float64,i)/30.0))]))
      end
    end
    handle_event(hsm,runHD)
    if run_lakes_flag
      handle_event(hsm,set_lake_evaporation)
      handle_event(hsm,run_lakes)
      handle_event(hsm,CheckWaterBudget())
      if return_lake_volumes
        lake_volumes_for_timestep::Array{Float64} = Float64[]
        for lake::Lake in prognostic_fields.lake_prognostics.lakes
          append!(lake_volumes_for_timestep,lake.lake_variables.lake_volume)
        end
        push!(diagnostic_lake_volumes,lake_volumes_for_timestep)
      end
    end
    if print_timestep_results
      if i%output_timestep == 0 || i == 1
        if ncells < 1000
          print_results::PrintResults = PrintResults(i)
          handle_event(hsm,print_results)
        else
          print_section::PrintSection = PrintSection()
          handle_event(hsm,print_section)
        end
      end
    end
    if write_output
      accumulate_river_flow::AccumulateRiverFlow = AccumulateRiverFlow()
      handle_event(hsm,accumulate_river_flow)
      if i%output_timestep == 0 || i == 1
        print_global_values::PrintGlobalValues = PrintGlobalValues()
        handle_event(hsm,print_global_values::PrintGlobalValues)
        #write_mean_river_flow::WriteMeanRiverFlow = WriteMeanRiverFlow(i,output_timestep)
        #handle_event(hsm,write_mean_river_flow)
        write_river_flow::WriteRiverFlow = WriteRiverFlow(i)
        handle_event(hsm,write_river_flow)
        reset_cumulative_river_flow::ResetCumulativeRiverFlow = ResetCumulativeRiverFlow()
        handle_event(hsm,reset_cumulative_river_flow)
        if run_lakes_flag
           #write_lake_numbers::WriteLakeNumbers = WriteLakeNumbers(i)
           #handle_event(hsm,write_lake_numbers)
           write_diagnostic_lake_volumes::WriteDiagnosticLakeVolumes =
              WriteDiagnosticLakeVolumes(i)
           handle_event(hsm,write_diagnostic_lake_volumes)
         end
      end
    end
  end
  if run_lakes_flag
    calculate_true_lake_depths::CalculateTrueLakeDepths = CalculateTrueLakeDepths()
    handle_event(hsm,calculate_true_lake_depths)
  end
  if write_output
    write_river_initial_values::WriteRiverInitialValues = WriteRiverInitialValues()
    handle_event(hsm,write_river_initial_values)
    if run_lakes_flag
      write_lake_volumes::WriteLakeVolumes = WriteLakeVolumes()
      handle_event(hsm,write_lake_volumes)
    end
  end
end

function drive_hd_model(river_parameters::RiverParameters,river_fields::RiverPrognosticFields,
                        drainages::Array{Field{Float64},1},runoffs::Array{Field{Float64},1},
                        lake_evaporations::Array{Field{Float64},1},
                        timesteps::Int64;print_timestep_results::Bool=false,
                        output_timestep::Int64=100,
                        write_output::Bool=true,
                        return_output::Bool=false,
                        use_realistic_surface_coupling::Bool=false,
                        input_data_is_monthly_mean::Bool=true)
  prognostic_fields::PrognosticFields = RiverPrognosticFieldsOnly(river_parameters,river_fields)
  drive_hd_model_with_or_without_lakes(prognostic_fields,drainages,runoffs,
                                       lake_evaporations,timesteps,false,
                                       print_timestep_results=print_timestep_results,
                                       output_timestep=output_timestep,
                                       write_output=write_output,
                                       use_realistic_surface_coupling=
                                       use_realistic_surface_coupling,
                                       input_data_is_monthly_mean=
                                       input_data_is_monthly_mean)
  if return_output
    return river_fields.water_to_ocean,river_fields.river_inflow
  end
end

function drive_hd_model(river_parameters::RiverParameters,drainages::Array{Field{Float64},1},
                        runoffs::Array{Field{Float64},1},
                        lake_evaporations::Array{Field{Float64},1},timesteps::Int64;
                        print_timestep_results::Bool=false,
                        output_timestep::Int64=100,
                        write_output::Bool=true,
                        return_output::Bool=false,
                        use_realistic_surface_coupling::Bool=false,
                        input_data_is_monthly_mean::Bool=true)
  river_fields::RiverPrognosticFields = RiverPrognosticFields(river_parameters)
  drive_hd_model(river_parameters,river_fields,drainages,runoffs,lake_evaporations,
                 timesteps,print_timestep_results=print_timestep_results,
                 output_timestep=output_timestep,
                 write_output=write_output,
                 return_output=false,
                 use_realistic_surface_coupling=
                 use_realistic_surface_coupling,
                 input_data_is_monthly_mean=
                 input_data_is_monthly_mean)
  if return_output
    return river_fields.water_to_ocean,river_fields.river_inflow
  end
end

function drive_hd_and_lake_model(river_parameters::RiverParameters,river_fields::RiverPrognosticFields,
                                 lake_parameters::LakeParameters,lake_prognostics::LakePrognostics,
                                 lake_fields::LakeFields,drainages::Array{Field{Float64},1},
                                 runoffs::Array{Field{Float64},1},
                                 lake_evaporations::Array{Field{Float64},1},timesteps::Int64,
                                 process_initial_lake_water::Bool=false,
                                 initial_water_to_lake_centers::FloatFieldOrNothing=nothing,
                                 initial_spillover_to_rivers::FloatFieldOrNothing=nothing;
                                 print_timestep_results::Bool=false,
                                 output_timestep::Int64=100,
                                 write_output::Bool=true,
                                 return_output::Bool=true,
                                 use_realistic_surface_coupling::Bool=false,
                                 return_lake_volumes::Bool=false,
                                 diagnostic_lake_volumes::VectorOfVectorOfFloatsOrNothing=
                                 nothing,
                                 input_data_is_monthly_mean::Bool=true)
  prognostic_fields::PrognosticFields = RiverAndLakePrognosticFields(river_parameters,river_fields,
                                                                     lake_parameters,lake_prognostics,
                                                                     lake_fields)
  drive_hd_model_with_or_without_lakes(prognostic_fields,drainages,runoffs,
                                       lake_evaporations,timesteps,
                                       true,process_initial_lake_water,
                                       initial_water_to_lake_centers,
                                       initial_spillover_to_rivers,
                                       print_timestep_results=print_timestep_results,
                                       output_timestep=output_timestep,
                                       write_output=write_output,
                                       use_realistic_surface_coupling=
                                       use_realistic_surface_coupling,
                                       return_lake_volumes=
                                       return_lake_volumes,
                                       diagnostic_lake_volumes=
                                       diagnostic_lake_volumes,
                                       input_data_is_monthly_mean=
                                       input_data_is_monthly_mean)
  if return_output
    return river_fields,lake_prognostics,lake_fields
  end
end

function drive_hd_and_lake_model(river_parameters::RiverParameters,
                                 river_fields::RiverPrognosticFields,
                                 lake_parameters::LakeParameters,
                                 drainages::Array{Field{Float64},1},
                                 runoffs::Array{Field{Float64},1},
                                 lake_evaporations::Array{Field{Float64},1},
                                 timesteps::Int64,
                                 process_initial_lake_water::Bool=false,
                                 initial_water_to_lake_centers::FloatFieldOrNothing=nothing,
                                 initial_spillover_to_rivers::FloatFieldOrNothing=nothing;
                                 print_timestep_results::Bool=false,
                                 output_timestep::Int64=100,
                                 write_output::Bool=true,
                                 return_output::Bool=false,
                                 use_realistic_surface_coupling::Bool=false,
                                 return_lake_volumes::Bool=false,
                                 diagnostic_lake_volumes::VectorOfVectorOfFloatsOrNothing=
                                 nothing,
                                 input_data_is_monthly_mean::Bool=true)
  lake_fields::LakeFields = LakeFields(lake_parameters)
  lake_prognostics::LakePrognostics = LakePrognostics(lake_parameters,lake_fields)
  drive_hd_and_lake_model(river_parameters,river_fields,lake_parameters,lake_prognostics,
                          lake_fields,drainages,runoffs,lake_evaporations,timesteps,
                          process_initial_lake_water,initial_water_to_lake_centers,
                          initial_spillover_to_rivers,
                          print_timestep_results=print_timestep_results,
                          output_timestep=output_timestep,
                          write_output=write_output,
                          return_output=false,
                          use_realistic_surface_coupling=
                          use_realistic_surface_coupling,
                          return_lake_volumes=
                          return_lake_volumes,
                          diagnostic_lake_volumes=
                          diagnostic_lake_volumes,
                          input_data_is_monthly_mean=
                          input_data_is_monthly_mean)
  if return_output
    return river_fields,lake_prognostics,lake_fields
  end
end

function drive_hd_and_lake_model(river_parameters::RiverParameters,lake_parameters::LakeParameters,
                                 lake_prognostics::LakePrognostics,lake_fields::LakeFields,
                                 drainages::Array{Field{Float64},1},
                                 runoffs::Array{Field{Float64},1},
                                 lake_evaporations::Array{Field{Float64},1},timesteps::Int64,
                                 process_initial_lake_water::Bool=false,
                                 initial_water_to_lake_centers::FloatFieldOrNothing=nothing,
                                 initial_spillover_to_rivers::FloatFieldOrNothing=nothing;
                                 print_timestep_results::Bool=false,
                                 output_timestep::Int64=100,
                                 write_output::Bool=true,
                                 return_output::Bool=false,
                                 use_realistic_surface_coupling::Bool=false,
                                 return_lake_volumes::Bool=false,
                                 diagnostic_lake_volumes::VectorOfVectorOfFloatsOrNothing=
                                 nothing,
                                 input_data_is_monthly_mean::Bool=true)
  river_fields::RiverPrognosticFields = RiverPrognosticFields(river_parameters)
  drive_hd_and_lake_model(river_parameters,river_fields,lake_parameters,lake_prognostics,
                          lake_fields,drainages,runoffs,lake_evaporations,timesteps,
                          process_initial_lake_water,initial_water_to_lake_centers,
                          initial_spillover_to_rivers,
                          print_timestep_results=print_timestep_results,
                          output_timestep=output_timestep,
                          write_output=write_output,
                          return_output=false,
                          use_realistic_surface_coupling=
                          use_realistic_surface_coupling,
                          return_lake_volumes=
                          return_lake_volumes,
                          diagnostic_lake_volumes=
                          diagnostic_lake_volumes,
                          input_data_is_monthly_mean=
                          input_data_is_monthly_mean)
  if return_output
    return river_fields,lake_prognostics,lake_fields
  end
end

function drive_hd_and_lake_model(river_parameters::RiverParameters,lake_parameters::LakeParameters,
                                 drainages::Array{Field{Float64},1},runoffs::Array{Field{Float64},1},
                                 lake_evaporations::Array{Field{Float64},1},timesteps::Int64,
                                 process_initial_lake_water::Bool=false,
                                 initial_water_to_lake_centers::FloatFieldOrNothing=nothing,
                                 initial_spillover_to_rivers::FloatFieldOrNothing=nothing;
                                 print_timestep_results::Bool=false,
                                 output_timestep::Int64=100,
                                 write_output::Bool=true,
                                 return_output::Bool=false,
                                 use_realistic_surface_coupling::Bool=false,
                                 return_lake_volumes::Bool=false,
                                 diagnostic_lake_volumes::VectorOfVectorOfFloatsOrNothing=
                                 nothing,
                                 input_data_is_monthly_mean::Bool=true)
  river_fields::RiverPrognosticFields = RiverPrognosticFields(river_parameters)
  lake_fields::LakeFields = LakeFields(lake_parameters)
  lake_prognostics::LakePrognostics = LakePrognostics(lake_parameters,lake_fields)
  drive_hd_and_lake_model(river_parameters,river_fields,lake_parameters,lake_prognostics,
                          lake_fields,drainages,runoffs,lake_evaporations,timesteps,
                          process_initial_lake_water,initial_water_to_lake_centers,
                          initial_spillover_to_rivers,
                          print_timestep_results=print_timestep_results,
                          output_timestep=output_timestep,
                          write_output=write_output,
                          return_output=false,
                          use_realistic_surface_coupling=
                          use_realistic_surface_coupling,
                          return_lake_volumes=
                          return_lake_volumes,
                          diagnostic_lake_volumes=
                          diagnostic_lake_volumes,
                          input_data_is_monthly_mean=
                          input_data_is_monthly_mean)
  if return_output
    return river_fields,lake_prognostics,lake_fields
  end
end

end
