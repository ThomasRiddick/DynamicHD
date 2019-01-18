module HDDriverModule

using Profile
using HierarchicalStateMachineModule: HierarchicalStateMachine
using HDModule: RiverParameters,RiverPrognosticFields,RiverPrognosticFieldsOnly,RunHD,handle_event,SetDrainage,SetRunoff,
                PrintResults,PrognosticFields,water_to_lakes,water_from_lakes
using FieldModule: Field
using LakeModule: LakeParameters,LakePrognostics,LakeFields,RiverAndLakePrognosticFields,RunLakes
using LakeModule: water_to_lakes,handle_event,water_from_lakes

function drive_hd_model_with_or_without_lakes(prognostic_fields::PrognosticFields,
                                              drainages::Array{Field{Float64},1},
                                              runoffs::Array{Field{Float64},1},
                                              timesteps::Int64,run_lakes_flag::Bool;
                                              print_timestep_results::Bool=false)
  hsm::HierarchicalStateMachine = HierarchicalStateMachine(prognostic_fields)
  runHD::RunHD = RunHD()
  if run_lakes_flag
    run_lakes::RunLakes = RunLakes()
  end
  for i in 1:timesteps
    set_drainage = SetDrainage(drainages[i])
    handle_event(hsm,set_drainage)
    set_runoff = SetRunoff(runoffs[i])
    handle_event(hsm,set_runoff)
    handle_event(hsm,runHD)
    if run_lakes_flag
      handle_event(hsm,run_lakes)
    end
    if print_timestep_results
      if i%100 == 0
        print_results::PrintResults = PrintResults(i)
        handle_event(hsm,print_results)
      end
    end
  end
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
                                 runoffs::Array{Field{Float64},1},timesteps::Int64;
                                 print_timestep_results::Bool=false)
  prognostic_fields::PrognosticFields = RiverAndLakePrognosticFields(river_parameters,river_fields,
                                                                     lake_parameters,lake_prognostics,
                                                                     lake_fields)
  drive_hd_model_with_or_without_lakes(prognostic_fields,drainages,runoffs,timesteps,
                                       true,print_timestep_results=print_timestep_results)
end

function drive_hd_and_lake_model(river_parameters::RiverParameters,lake_parameters::LakeParameters,
                                 drainages::Array{Field{Float64},1},runoffs::Array{Field{Float64},1},
                                 timesteps::Int64;print_timestep_results::Bool=false)
  river_fields::RiverPrognosticFields = RiverPrognosticFields(river_parameters)
  lake_fields::LakeFields = LakeFields(river_parameters,lake_parameters)
  lake_prognostics::LakePrognostics = LakePrognostics(lake_parameters,lake_fields)
  drive_hd_and_lake_model(river_parameters,river_fields,lake_parameters,lake_prognostics,
                          lake_fields,drainages,runoffs,timesteps,
                          print_timestep_results=print_timestep_results)
end

end
