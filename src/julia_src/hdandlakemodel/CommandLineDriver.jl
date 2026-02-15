push!(LOAD_PATH, "/Users/thomasriddick/Documents/workspace/Dynamic_HD_Code/src/julia_src/hdandlakemodel")
using ArgParse
using HDDriverModule: drive_hd_model,drive_hd_and_lake_model
using InputModule: load_river_parameters, load_lake_parameters
using InputModule: load_drainage_fields, load_runoff_fields
using InputModule: load_lake_evaporation_fields
using InputModule: load_lake_initial_values,load_river_initial_values
using GridModule: Grid, LatLonGrid
using FieldModule: Field,LatLonField,repeat,divide
using Profile
using ProfileView
using Serialization
using InteractiveUtils

function pass_arguments()
  settings = ArgParseSettings()
  @add_arg_table settings begin
  "--hd-para-file", "-p"
    help = "Filepath to HD parameters file"
    required = true
  "--hd-init-file", "-i"
    help = "Filepath to HD initialisation file (optional)"
  "--lake-para-file", "-l"
    help = "Filepath to lake model parameters file (optional)"
  "--lake-init-file", "-n"
    help = "Filepath to lake model initialisation file (optional)"
  "--drainage-file", "-d"
    help = "Filepath to file containing drainage values"
  "--runoff-file", "-r"
    help = "Filepath to file containing runoff values"
  "--lake-evaporation-file", "-e"
    help = "Filepath to file containing values for evaporation from lakes"
  "--use-input-data-for-individual-timesteps", "-s"
    help = "Data read from input files is for individual timesteps"
    action = :store_true
  "--timesteps", "-t"
    help = "Number of timesteps to run"
    arg_type = Int
    default=100
  end
  return parse_args(settings)
end

function main()
  args = pass_arguments()
  timesteps = args["timesteps"]
  grid = LatLonGrid(360,720,true)
  lake_grid = LatLonGrid(1080,2160,true)
  surface_model_grid = LatLonGrid(48,96,true)
  river_parameters = load_river_parameters(args["hd-para-file"],grid)
  local drainages::Array{Field{Float64},1}
  local runoffs::Array{Field{Float64},1}
  local lake_evaporations::Array{Field{Float64},1}
  if args["drainage-file"] != nothing
    if args["use-input-data-for-individual-timesteps"]
      drainages = load_drainage_fields(args["drainage-file"],grid,
                                       last_timestep=9)
    else
      drainages = load_drainage_fields(args["drainage-file"],grid,
                                       last_timestep=12)
      drainages = [divide(x,30.0) for x in drainages]
    end
  else
    drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,269747790.0*2.25/100.0)
    drainages = repeat(drainage,12*51,false)
  end
  if args["runoff-file"] != nothing
    if args["use-input-data-for-individual-timesteps"]
      runoffs = load_runoff_fields(args["runoff-file"],grid,
                                   last_timestep=9)
    else
      runoffs = load_runoff_fields(args["runoff-file"],grid,
                                   last_timestep=12)
      runoffs = [divide(x,30.0) for x in runoffs]
    end
  else
    runoffs = deepcopy(drainages)
  end
  if args["lake-evaporation-file"] != nothing
    if args["use-input-data-for-individual-timesteps"]
      lake_evaporations = load_lake_evaporation_fields(args["lake-evaporation-file"],surface_model_grid,
                                                       last_timestep=9)
    else
      lake_evaporations = load_lake_evaporation_fields(args["lake-evaporation-file"],surface_model_grid,
                                                       last_timestep=12)
      lake_evaporations = [divide(x,30.0) for x in lake_evaporations]
    end
  else
    lake_evaporation::Field{Float64} = LatLonField{Float64}(surface_model_grid,0.0)
    lake_evaporations = repeat(lake_evaporation,12*51,false)
  end
  input_data_is_monthly_mean::Bool = ! args["use-input-data-for-individual-timesteps"]
  if args["lake-para-file"] != nothing
    lake_parameters = load_lake_parameters(args["lake-para-file"],lake_grid,grid,surface_model_grid)
    drainages_copy = deepcopy(drainages)
    runoffs_copy = deepcopy(runoffs)
    lake_evaporations_copy = deepcopy(lake_evaporations)
    if args["hd-init-file"] != nothing
      river_fields = load_river_initial_values(args["hd-init-file"],grid,river_parameters)
    end
    if args["lake-init-file"] != nothing
      initial_water_to_lake_centers::LatLonField{Float64},
      initial_spillover_to_rivers::LatLonField{Float64} =
        load_lake_initial_values(args["lake-init-file"],lake_grid,grid)
      if args["hd-init-file"] != nothing
        drive_hd_and_lake_model(river_parameters,river_fields,
                                lake_parameters,drainages,runoffs,lake_evaporations,
                                timesteps,true,initial_water_to_lake_centers,
                                initial_spillover_to_rivers;output_timestep=10,
                                print_timestep_results=false,
                                use_realistic_surface_coupling=true,
                                input_data_is_monthly_mean=
                                input_data_is_monthly_mean)
        # Profile.clear()
        # Profile.init(delay=0.01)
        # @time drive_hd_and_lake_model(river_parameters,river_fields,
        #                               lake_parameters,drainages_copy,runoffs_copy,
        #                               lake_evaporations_copy,timesteps,
        #                               true,initial_water_to_lake_centers,
        #                               initial_spillover_to_rivers;print_timestep_results=true)
        # Profile.print()
        # r = Profile.retrieve();
        # f = open("/Users/thomasriddick/Downloads/profile.bin", "w")
        # Serialization.serialize(f, r)
        # close(f)
      else
        drive_hd_and_lake_model(river_parameters,lake_parameters,drainages,runoffs,
                                lake_evaporations,timesteps,true,initial_water_to_lake_centers,
                                initial_spillover_to_rivers;output_timestep=10,
                                print_timestep_results=true,
                                use_realistic_surface_coupling=true,
                                input_data_is_monthly_mean=
                                input_data_is_monthly_mean)
      end
    else
      if args["hd-init-file"] != nothing
        drive_hd_and_lake_model(river_parameters,river_fields,
                                lake_parameters,drainages,runoffs,
                                lake_evaporations,timesteps;output_timestep=10,
                                print_timestep_results=true,
                                use_realistic_surface_coupling=true,
                                input_data_is_monthly_mean=
                                input_data_is_monthly_mean)
      else
        drive_hd_and_lake_model(river_parameters,lake_parameters,
                                drainages,runoffs,lake_evaporations,
                                timesteps;output_timestep=10,print_timestep_results=true,
                                use_realistic_surface_coupling=true,
                                input_data_is_monthly_mean=
                                input_data_is_monthly_mean)
        # @time drive_hd_and_lake_model(river_parameters,lake_parameters,
        #                               drainages_copy,runoffs_copy,
        #                               lake_evaporations_copy,timesteps;
        #                               print_timestep_results=true)
      end
    end

  else
    if args["hd-init-file"] != nothing
      river_fields = load_river_initial_values(args["hd-init-file"],grid,river_parameters)
      drive_hd_model(river_parameters,river_fields,
                     drainages,runoffs,lake_evaporations,
                     timesteps;output_timestep=10,print_timestep_results=false)
      @time drive_hd_model(river_parameters,river_fields,
                           drainages_copy,runoffs_copy,lake_evaporations_copy,
                           timesteps;output_timestep=10,print_timestep_results=false,
                           input_data_is_monthly_mean=
                           input_data_is_monthly_mean)
    else
      drainages_copy = deepcopy(drainages)
      runoffs_copy = deepcopy(runoffs)
      lake_evaporations_copy = deepcopy(lake_evaporations)
      drive_hd_model(river_parameters,drainages,runoffs,
                     lake_evaporations,timesteps;output_timestep=10,
                     print_timestep_results=true)
      Profile.clear()
      Profile.init(delay=0.0001)
      @time drive_hd_model(river_parameters,drainages_copy,
                           runoffs_copy,lake_evaporations_copy,
                           timesteps;output_timestep=10,print_timestep_results=false,
                           input_data_is_monthly_mean=
                           input_data_is_monthly_mean)
      Profile.print()
      r = Profile.retrieve();
      f = open("/Users/thomasriddick/Downloads/profile.bin", "w")
      Serialization.serialize(f, r)
      close(f)
    end
  end
end

# empty!(ARGS)
#push!(ARGS,"-p/Users/thomasriddick/Documents/data/HDdata/hdfiles/hdpara_file_from_current_model.nc")
# push!(ARGS,"-p/Users/thomasriddick/Documents/data/HDdata/hdfiles/generated/hd_file_prepare_basins_from_glac1D_20190925_225029_1400.nc")
# push!(ARGS,"-l/Users/thomasriddick/Documents/data/HDdata/lakeparafiles/lakeparas_prepare_basins_from_glac1D_20190925_225029_1400.nc")
#push!(ARGS,"-n/Users/thomasriddick/Documents/data/temp/transient_sim_1/results_for_1400/lake_model_start_1400.nc")
#push!(ARGS,"-i/Users/thomasriddick/Documents/data/temp/transient_sim_1/results_for_1400/hdstart_1400.nc")
# push!(ARGS,"-t6000")
push!(ARGS,"-p/Users/thomasriddick/Documents/data/temp/10070/hdpara_10070k.nc")
push!(ARGS,"-l/Users/thomasriddick/Documents/data/temp/10070/lakepara_10070k.nc")
push!(ARGS,"-n/Users/thomasriddick/Documents/data/temp/10070/lakestart_10070k.nc")
push!(ARGS,"-i/Users/thomasriddick/Documents/data/temp/10070/restart_rid004_hd_29291231.nc")
push!(ARGS,"-s")
push!(ARGS,"-d/Users/thomasriddick/Documents/data/temp/10070/lake_and_hd_model_debug_rf_and_drmerged.nc")
push!(ARGS,"-r/Users/thomasriddick/Documents/data/temp/10070/lake_and_hd_model_debug_rf_and_drmerged.nc")
push!(ARGS,"-e/Users/thomasriddick/Documents/data/temp/10070/lake_and_hd_model_debug_lhmerged.nc")
#push!(ARGS,"-n/Users/thomasriddick/Documents/data/temp/transient_sim_1/results_for_1400/lake_model_start_1400.nc")
#push!(ARGS,"-i/Users/thomasriddick/Documents/data/temp/transient_sim_1/results_for_1400/hdstart_1400.nc")
push!(ARGS,"-t9")
main()
