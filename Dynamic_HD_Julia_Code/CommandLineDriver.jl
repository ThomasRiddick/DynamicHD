using ArgParse
using HDDriverModule: drive_hd_model,drive_hd_and_lake_model
using IOModule: load_river_parameters, load_lake_parameters
using GridModule: Grid, LatLonGrid
using FieldModule: Field,LatLonField,repeat
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
  drainage::Field{Float64} = LatLonField{Float64}(river_parameters.grid,1.5)
  drainages::Array{Field{Float64},1} = repeat(drainage,500)
  runoffs::Array{Field{Float64},1} = deepcopy(drainages)
  if args["lake-para-file"] != nothing
    lake_parameters = load_lake_parameters(args["lake-para-file"],lake_grid,grid)
    if args["lake-init-file"] != nothing
      initial_water_to_lake_centers::LatLonField{Float64},
      initial_spillover_to_rivers::LatLonField{Float64} =
        load_lake_initial_values(args["lake-init-file"],lake_grid,grid)
      drive_hd_and_lake_model(river_parameters,lake_parameters,drainages,runoffs,
                              timesteps,true,initial_water_to_lake_centers,
                              initial_spillover_to_rivers;print_timestep_results=true)
    else
      drive_hd_and_lake_model(river_parameters,lake_parameters,
                              drainages,runoffs,
                              timesteps;print_timestep_results=true)
    end
    drainages = repeat(drainage,500)
    runoffs = deepcopy(drainages)
    Profile.clear()
    Profile.init(delay=0.01)
    # @profile drive_hd_and_lake_model(river_parameters,lake_parameters,
    #                               drainages,runoffs,
    #                               timesteps;print_timestep_results=false)
    Profile.print()
    r = Profile.retrieve();
    f = open("/Users/thomasriddick/Downloads/profile.bin", "w")
    Serialization.serialize(f, r)
    close(f)
  else
    if args["hd-init-file"] != nothing
      # drive_hd_model(river_parameters,river_fields,
      #                drainages,runoffs,timesteps;
      #                print_timestep_results=false)
    else
      drive_hd_model(river_parameters,drainages,
                     runoffs,timesteps;print_timestep_results=true)
      drainages = repeat(drainage,500)
      runoffs = deepcopy(drainages)
      Profile.clear()
      Profile.init(delay=0.0001)
      # @time drive_hd_model(river_parameters,drainages,
      #                      runoffs,timesteps;print_timestep_results=false)

      Profile.print()
      r = Profile.retrieve();
      f = open("/Users/thomasriddick/Downloads/profile.bin", "w")
      Serialization.serialize(f, r)
      close(f)
    end
  end
end

empty!(ARGS)
# push!(ARGS,"-p/Users/thomasriddick/Documents/data/HDdata/hdfiles/hdpara_file_from_current_model.nc")
push!(ARGS,"-p/Users/thomasriddick/Documents/data/HDdata/hdfiles/generated/hd_file_prepare_river_directions_with_depressions_20190503_170551.nc")
push!(ARGS,"-l/Users/thomasriddick/Documents/data/HDdata/lakeparafiles/lakeparasevaluate_glac1D_ts1900_basins_20190509_101249.nc")
push!(ARGS,"-t500")
main()
