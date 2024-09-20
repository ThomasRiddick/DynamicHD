using ArgParse
using IdentifyBifurcatedRiverMouthsInput: load_icosahedral_grid,load_landsea_mask
using IdentifyBifurcatedRiverMouthsInput: load_river_deltas_from_file
using IdentifyBifurcatedRiverMouthsOutput: write_river_mouth_indices
using IdentifyBifurcatedRiverMouths: identify_bifurcated_river_mouths

function pass_arguments()
  settings = ArgParseSettings()
  @add_arg_table settings begin
  "--grid-filepath", "-g"
    help = "Filepath to ICON grid file"
    required = true
  "--lsmask-filepath", "-m"
    help = "Filepath to landsea mask file"
    required = true
  "--lsmask-fieldname", "-n"
    help = "Fieldname of landsea mask field"
    required = true
  "--river-deltas-filepath","-r"
    help = "Filepath to river deltas file"
    required = true
  "--output-river-mouths-filepath","-o"
    help = "Target filepath for output river mouth indices file"
    required = true
  end
  return parse_args(settings)
end

function bifurcated_river_mouth_identification_driver(grid_filepath::String,
                                                      lsmask_filepath::String,
                                                      lsmask_fieldname::String,
                                                      river_deltas_filepath::String,
                                                      output_river_mouths_filepath::String)
  cells::Cells = load_icosahedral_grid(grid_filepath)
  lsmask::Array{Bool} = load_landsea_mask(lsmask_filepath,lsmask_fieldname)
  river_deltas::Array{RiverDelta} = load_river_deltas_from_file(river_deltas_filepath)
  river_mouth_indices::Dict{Array{CartesianIndex}} =
    identify_bifurcated_river_mouths(river_deltas::Array{RiverDelta},
                                     cells::Cells,
                                     lsmask::Array{Bool})
  write_river_mouth_indices(river_mouth_indices,output_river_mouths_filepath)
end

args = pass_arguments()
bifurcated_river_mouth_identification_driver(args["grid-filepath"],
                                             args["lsmask-filepath"],
                                             args["lsmask-fieldname"],
                                             args["river-deltas-filepath"],
                                             args["output-river-mouths-filepath"])

