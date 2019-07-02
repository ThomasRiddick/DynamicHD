module HDDriverModule

using Profile
using HierarchicalStateMachineModule: HierarchicalStateMachine
using HDModule: RiverParameters,RiverPrognosticFields,RiverPrognosticFieldsOnly,RunHD,handle_event
using HDModule: SetDrainage,SetRunoff,PrintResults,PrognosticFields,WriteRiverInitialValues
using HDModule: water_to_lakes,water_from_lakes
using FieldModule: Field
using LakeModule: LakeParameters,LakePrognostics,LakeFields,RiverAndLakePrognosticFields,RunLakes
using LakeModule: PrintSection,WriteLakeNumbers,WriteLakeVolumes,SetupLakes,DistributeSpillover
using LakeModule: water_to_lakes,handle_event,water_from_lakes

FloatFieldOrNothing = Union{Field{Float64},Nothing}

function drive_hd_model_with_or_without_lakes(prognostic_fields::PrognosticFields,
                                              drainages::Array{Field{Float64},1},
                                              runoffs::Array{Field{Float64},1},
                                              timesteps::Int64,run_lakes_flag::Bool,
                                              process_initial_lake_water::Bool=false,
                                              initial_water_to_lake_centers::FloatFieldOrNothing=
                                              nothing,
                                              initial_spillover_to_rivers::FloatFieldOrNothing=
                                              nothing;
                                              print_timestep_results::Bool=false)
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
    end
  end
  for i in 1:timesteps
    set_drainage = SetDrainage(deepcopy(drainages[convert(Int64,ceil(convert(Float64,i)/30.0))]))
    handle_event(hsm,set_drainage)
    set_runoff = SetRunoff(deepcopy(runoffs[convert(Int64,ceil(convert(Float64,i)/30.0))]))
    handle_event(hsm,set_runoff)
    handle_event(hsm,runHD)
    if run_lakes_flag
      handle_event(hsm,run_lakes)
    end
    if print_timestep_results
      if false #i%100 == 0
        # print_results::PrintResults = PrintResults(i)
        # handle_event(hsm,print_results)
        print_section::PrintSection = PrintSection()
        handle_event(hsm,print_section)
      end
    end
    if run_lakes_flag
      if i%365 == 0
        write_lake_numbers::WriteLakeNumbers = WriteLakeNumbers(i)
        handle_event(hsm,write_lake_numbers)
      end
    end
  end
  if run_lakes_flag
    write_lake_volumes::WriteLakeVolumes = WriteLakeVolumes()
    handle_event(hsm,write_lake_volumes)
  end
  write_river_initial_values::WriteRiverInitialValues = WriteRiverInitialValues()
  handle_event(hsm,write_river_initial_values)
end

function drive_hd_model(river_parameters::RiverParameters,river_fields::RiverPrognosticFields,
                        drainages::Array{Field{Float64},1},runoffs::Array{Field{Float64},1},
                        timesteps::Int64;print_timestep_results::Bool=false)
  prognostic_fields::PrognosticFields = RiverPrognosticFieldsOnly(river_parameters,river_fields)
  drive_hd_model_with_or_without_lakes(prognostic_fields,drainages,runoffs,timesteps,false,
                                       print_timestep_results=print_timestep_results)

end

function drive_hd_model(river_parameters::RiverParameters,drainages::Array{Field{Float64},1},
                        runoffs::Array{Field{Float64},1},timesteps::Int64;
                        print_timestep_results::Bool=false)
  river_fields::RiverPrognosticFields = RiverPrognosticFields(river_parameters)
  drive_hd_model(river_parameters,river_fields,drainages,runoffs,timesteps,
                 print_timestep_results=print_timestep_results)
end

function drive_hd_and_lake_model(river_parameters::RiverParameters,river_fields::RiverPrognosticFields,
                                 lake_parameters::LakeParameters,lake_prognostics::LakePrognostics,
                                 lake_fields::LakeFields,drainages::Array{Field{Float64},1},
                                 runoffs::Array{Field{Float64},1},timesteps::Int64,
                                 process_initial_lake_water::Bool=false,
                                 initial_water_to_lake_centers::FloatFieldOrNothing=nothing,
                                 initial_spillover_to_rivers::FloatFieldOrNothing=nothing;
                                 print_timestep_results::Bool=false)
  prognostic_fields::PrognosticFields = RiverAndLakePrognosticFields(river_parameters,river_fields,
                                                                     lake_parameters,lake_prognostics,
                                                                     lake_fields)
  drive_hd_model_with_or_without_lakes(prognostic_fields,drainages,runoffs,timesteps,
                                       true,process_initial_lake_water,
                                       initial_water_to_lake_centers,
                                       initial_spillover_to_rivers,
                                       print_timestep_results=print_timestep_results)
end

function drive_hd_and_lake_model(river_parameters::RiverParameters,
                                 river_fields::RiverPrognosticFields,
                                 lake_parameters::LakeParameters,
                                 drainages::Array{Field{Float64},1},
                                 runoffs::Array{Field{Float64},1},timesteps::Int64,
                                 process_initial_lake_water::Bool=false,
                                 initial_water_to_lake_centers::FloatFieldOrNothing=nothing,
                                 initial_spillover_to_rivers::FloatFieldOrNothing=nothing;
                                 print_timestep_results::Bool=false)
  lake_fields::LakeFields = LakeFields(lake_parameters)
  lake_prognostics::LakePrognostics = LakePrognostics(lake_parameters,lake_fields)
  drive_hd_and_lake_model(river_parameters,river_fields,lake_parameters,lake_prognostics,
                          lake_fields,drainages,runoffs,timesteps,
                          process_initial_lake_water,initial_water_to_lake_centers,
                          initial_spillover_to_rivers,
                          print_timestep_results=print_timestep_results)
end

function drive_hd_and_lake_model(river_parameters::RiverParameters,lake_parameters::LakeParameters,
                                 lake_prognostics::LakePrognostics,lake_fields::LakeFields,
                                 drainages::Array{Field{Float64},1},
                                 runoffs::Array{Field{Float64},1},timesteps::Int64,
                                 process_initial_lake_water::Bool=false,
                                 initial_water_to_lake_centers::FloatFieldOrNothing=nothing,
                                 initial_spillover_to_rivers::FloatFieldOrNothing=nothing;
                                 print_timestep_results::Bool=false)
  river_fields::RiverPrognosticFields = RiverPrognosticFields(river_parameters)
  drive_hd_and_lake_model(river_parameters,river_fields,lake_parameters,lake_prognostics,
                          lake_fields,drainages,runoffs,timesteps,
                          process_initial_lake_water,initial_water_to_lake_centers,
                          initial_spillover_to_rivers,
                          print_timestep_results=print_timestep_results)
end

function drive_hd_and_lake_model(river_parameters::RiverParameters,lake_parameters::LakeParameters,
                                 drainages::Array{Field{Float64},1},runoffs::Array{Field{Float64},1},
                                 timesteps::Int64,
                                 process_initial_lake_water::Bool=false,
                                 initial_water_to_lake_centers::FloatFieldOrNothing=nothing,
                                 initial_spillover_to_rivers::FloatFieldOrNothing=nothing;
                                 print_timestep_results::Bool=false)
  river_fields::RiverPrognosticFields = RiverPrognosticFields(river_parameters)
  lake_fields::LakeFields = LakeFields(lake_parameters)
  lake_prognostics::LakePrognostics = LakePrognostics(lake_parameters,lake_fields)
  drive_hd_and_lake_model(river_parameters,river_fields,lake_parameters,lake_prognostics,
                          lake_fields,drainages,runoffs,timesteps,
                          process_initial_lake_water,initial_water_to_lake_centers,
                          initial_spillover_to_rivers,
                          print_timestep_results=print_timestep_results)
end

end
