push!(LOAD_PATH, "/Users/thomasriddick/Documents/workspace/Dynamic_HD_Code/src/julia_src/hdandlakemodel")
using Profile
using ArgParse
using HDDriverModule: drive_hd_model,drive_hd_and_lake_model
using FieldModule: UnstructuredField,Field,repeat
using IOModule: load_river_parameters, load_river_initial_values,get_ncells
using IOModule: load_lake_initial_values,load_lake_parameters
using IOModule: get_additional_grid_information
using GridModule: UnstructuredGrid

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
  "--timesteps", "-t"
    help = "Number of timesteps to run"
    arg_type = Int
    default=100
  "--step-length","-s"
    help = "Length of each timestep in seconds"
    arg_type = Float64
    default = 86400.0
  end
  return parse_args(settings)
end

function main()
  args = pass_arguments()
  timesteps = args["timesteps"]
  step_length = args["step-length"]
  ncells = get_ncells(args["hd-para-file"])
  local clat::Array{Float64,1}
  local clon::Array{Float64,1}
  local clat_bounds::Array{Float64,2}
  local clon_bounds::Array{Float64,2}
  clat,clon,clat_bounds,clon_bounds = get_additional_grid_information(args["hd-para-file"])
  grid = UnstructuredGrid(ncells,clat,clon,clat_bounds,clon_bounds)
  lake_grid = UnstructuredGrid(ncells,clat,clon,clat_bounds,clon_bounds)
  surface_model_grid = UnstructuredGrid(ncells,clat,clon,clat_bounds,clon_bounds)
  river_parameters = load_river_parameters(args["hd-para-file"],grid;
                                           day_length=86400.0,step_length=step_length)
  local drainages::Array{Field{Float64},1}
  local runoffs::Array{Field{Float64},1}
  local lake_evaporations::Array{Field{Float64},1}
  if args["drainage-file"] != nothing
    drainages = load_drainage_fields(args["drainage-file"],grid,
                                     last_timestep=12)
    drainages = [divide(x,30.0) for x in drainages]
  else
    drainage::Field{Float64} = UnstructuredField{Float64}(river_parameters.grid,
                                                          100*0.0000000227*step_length*2.6*10000000000)
    #drainages = repeat(drainage,Int(round(timesteps/30)+1))
    drainages = [drainage]
  end
  if args["runoff-file"] != nothing
    runoffs = load_runoff_fields(args["runoff-file"],grid,
                                 last_timestep=12)
    runoffs = [divide(x,30.0) for x in runoffs]
  else
    runoffs = deepcopy(drainages)
  end
  if args["lake-evaporation-file"] != nothing
    lake_evaporations = load_lake_evaporation_fields(args["lake-evaporation-file"],grid,
                                                     last_timestep=12)
    lake_evaporations = [divide(x,30.0) for x in lake_evaporations]
  else
    lake_evaporation_none ::Field{Float64} = UnstructuredField{Float64}(river_parameters.grid,0.0)
    lake_evaporation_high ::Field{Float64} =
      UnstructuredField{Float64}(river_parameters.grid,
                                 100000*0.0000000227*step_length*2.6*10000000000)
    lake_evaporations_none = repeat(lake_evaporation_none,Int(round(timesteps/60)+1))
    lake_evaporations_high = repeat(lake_evaporation_high,Int(round(timesteps/60)+1))
    lake_evaporations = vcat(lake_evaporations_none,lake_evaporations_high)
  end
  if args["lake-para-file"] != nothing
    lake_parameters = load_lake_parameters(args["lake-para-file"],lake_grid,grid,surface_model_grid)
    drainages_copy = deepcopy(drainages)
    runoffs_copy = deepcopy(runoffs)
    if args["hd-init-file"] != nothing
      river_fields = load_river_initial_values(args["hd-init-file"],grid,river_parameters)
    end
    if args["lake-init-file"] != nothing
      initial_water_to_lake_centers::UnstructuredField{Float64},
      initial_spillover_to_rivers::UnstructuredField{Float64} =
        load_lake_initial_values(args["lake-init-file"],lake_grid,grid)
      if args["hd-init-file"] != nothing
        drive_hd_and_lake_model(river_parameters,river_fields,
                                lake_parameters,drainages,runoffs,
                                lake_evaporations,timesteps,true,
                                initial_water_to_lake_centers,
                                initial_spillover_to_rivers;print_timestep_results=false)
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
        drive_hd_and_lake_model(river_parameters,lake_parameters,
                                drainages,runoffs,
                                lake_evaporations,timesteps,true,
                                initial_water_to_lake_centers,
                                initial_spillover_to_rivers;
                                print_timestep_results=true)
      end
    end
  else
    if args["hd-init-file"] != nothing
      river_fields = load_river_initial_values(args["hd-init-file"],grid,river_parameters)
      drive_hd_model(river_parameters,river_fields,
                     drainages,runoffs,timesteps;
                     print_timestep_results=false,
                     output_timestep=160)
      # @time drive_hd_model(river_parameters,river_fields,
      #                      drainages_copy,runoffs_copy,timesteps;
      #                      print_timestep_results=false)
    else
      # drainages_copy = deepcopy(drainages)
      # runoffs_copy = deepcopy(runoffs)
      drive_hd_model(river_parameters,drainages,
                     runoffs,timesteps;print_timestep_results=false)
      # Profile.clear()
      # Profile.init(delay=0.0001)
      # @time drive_hd_model(river_parameters,drainages_copy,
      #                      runoffs_copy,timesteps;print_timestep_results=false)
      # Profile.print()
      # r = Profile.retrieve();
      # f = open("/Users/thomasriddick/Downloads/profile.bin", "w")
      # Serialization.serialize(f, r)
      # close(f)
    end
  end
end

empty!(ARGS)
# push!(ARGS,"-p/Users/thomasriddick/Documents/data/ICONHDdata/hdparafiles/hd_para_icon_r2b9_30_20191452_retuned_v3.nc")
# push!(ARGS,"-i/Users/thomasriddick/Documents/data/ICONHDdata/hdstartfiles/hdrestart_R02B09_015_G_241019_1337_v2.nc")
# push!(ARGS,"-t1920")
# push!(ARGS,"-s45")
push!(ARGS,"-p/Users/thomasriddick/Documents/data/temp/icon_lake_model_test/hdpara_icon.nc")
push!(ARGS,"-i/Users/thomasriddick/Documents/data/temp/icon_lake_model_test/hdrestart_R02B04_013_G_231019_1242_v2.nc")
push!(ARGS,"-l/Users/thomasriddick/Documents/data/temp/icon_lake_model_test/lakeparams.nc")
push!(ARGS,"-n/Users/thomasriddick/Documents/data/temp/icon_lake_model_test/lakestart.nc")
push!(ARGS,"-t3600")
push!(ARGS,"-s86400")
main()
