push!(LOAD_PATH, "/Users/thomasriddick/Documents/workspace/Dynamic_HD_Code/Dynamic_HD_Julia_Code/")
using ArgParse
using HDDriverModule: drive_hd_model,drive_hd_and_lake_model
using IOModule: load_river_parameters, load_lake_parameters
using IOModule: load_drainage_fields, load_runoff_fields
using IOModule: load_lake_initial_values,load_river_initial_values
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
  river_parameters = load_river_parameters(args["hd-para-file"],grid)
  local drainages::Array{Field{Float64},1}
  local runoffs::Array{Field{Float64},1}
  if args["drainage-file"] != nothing
    drainages = load_drainage_fields(args["drainage-file"],grid,
                                     last_timestep=12)
    drainages = [divide(x,30.0) for x in drainages]
  else
    drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,269747790.0*2.25/100.0)
    drainages = repeat(drainage,12*51)
  end
  if args["runoff-file"] != nothing
    runoffs = load_runoff_fields(args["runoff-file"],grid,
                                 last_timestep=12)
    runoffs = [divide(x,30.0) for x in runoffs]
  else
    runoffs = deepcopy(drainages)
  end
  if args["lake-para-file"] != nothing
    lake_parameters = load_lake_parameters(args["lake-para-file"],lake_grid,grid)
    drainages_copy = deepcopy(drainages)
    runoffs_copy = deepcopy(runoffs)
    if args["hd-init-file"] != nothing
      river_fields = load_river_initial_values(args["hd-init-file"],river_parameters)
    end
    if args["lake-init-file"] != nothing
      initial_water_to_lake_centers::LatLonField{Float64},
      initial_spillover_to_rivers::LatLonField{Float64} =
        load_lake_initial_values(args["lake-init-file"],lake_grid,grid)
      if args["hd-init-file"] != nothing
        drive_hd_and_lake_model(river_parameters,river_fields,
                                lake_parameters,drainages,runoffs,
                                timesteps,true,initial_water_to_lake_centers,
                                initial_spillover_to_rivers;print_timestep_results=true)
        # Profile.clear()
        # Profile.init(delay=0.01)
        # @time drive_hd_and_lake_model(river_parameters,river_fields,
        #                               lake_parameters,drainages_copy,runoffs_copy,
        #                               timesteps,true,initial_water_to_lake_centers,
        #                               initial_spillover_to_rivers;print_timestep_results=true)
        # Profile.print()
        # r = Profile.retrieve();
        # f = open("/Users/thomasriddick/Downloads/profile.bin", "w")
        # Serialization.serialize(f, r)
        # close(f)
      else
        drive_hd_and_lake_model(river_parameters,lake_parameters,drainages,runoffs,
                                timesteps,true,initial_water_to_lake_centers,
                                initial_spillover_to_rivers;print_timestep_results=true)
      end
    else
      if args["hd-init-file"] != nothing
        drive_hd_and_lake_model(river_parameters,river_fields,
                                lake_parameters,drainages,runoffs,
                                timesteps;print_timestep_results=true)
      else
        drive_hd_and_lake_model(river_parameters,lake_parameters,
                                drainages,runoffs,
                                timesteps;print_timestep_results=true)
        # @time drive_hd_and_lake_model(river_parameters,lake_parameters,
        #                               drainages_copy,runoffs_copy,
        #                               timesteps;print_timestep_results=true)
      end
    end

  else
    if args["hd-init-file"] != nothing
      river_fields = load_river_initial_values(args["hd-init-file"],river_parameters)
      drive_hd_model(river_parameters,river_fields,
                     drainages,runoffs,timesteps;
                     print_timestep_results=false)
      @time drive_hd_model(river_parameters,river_fields,
                           drainages_copy,runoffs_copy,timesteps;
                           print_timestep_results=false)
    else
      drainages_copy = deepcopy(drainages)
      runoffs_copy = deepcopy(runoffs)
      drive_hd_model(river_parameters,drainages,
                     runoffs,timesteps;print_timestep_results=true)
      Profile.clear()
      Profile.init(delay=0.0001)
      @time drive_hd_model(river_parameters,drainages_copy,
                           runoffs_copy,timesteps;print_timestep_results=false)
      Profile.print()
      r = Profile.retrieve();
      f = open("/Users/thomasriddick/Downloads/profile.bin", "w")
      Serialization.serialize(f, r)
      close(f)
    end
  end
end

#empty!(ARGS)
#push!(ARGS,"-p/Users/thomasriddick/Documents/data/HDdata/hdfiles/hdpara_file_from_current_model.nc")
# push!(ARGS,"-p/Users/thomasriddick/Documents/data/HDdata/hdfiles/generated/hd_file_prepare_basins_from_glac1D_20190720_142235_1150.nc")
# push!(ARGS,"-l/Users/thomasriddick/Documents/data/HDdata/lakeparafiles/lakeparas_prepare_basins_from_glac1D_20190720_142235_1150.nc")
# push!(ARGS,"-n/Users/thomasriddick/Documents/data/temp/transient_sim_1/lake_model_start_1200.nc")
# push!(ARGS,"-i/Users/thomasriddick/Documents/data/temp/transient_sim_1/initial_hdstart.nc")
# push!(ARGS,"-t600")
main()
