using ArgParse
using IdentifyBifurcatedRiverMouthsInput: load_icosahedral_grid,load_landsea_mask
using IdentifyBifurcatedRiverMouthsInput: load_river_deltas_from_file
using IdentifyBifurcatedRiverMouthsInput: load_accumulated_flow
using IdentifyBifurcatedRiverMouthsInput: load_search_areas_from_file
using IdentifyBifurcatedRiverMouthsOutput: write_river_mouth_indices
using IdentifyBifurcatedRiverMouths: identify_bifurcated_river_mouths,Cells
using IdentifyBifurcatedRiverMouths: RiverDelta
using IdentifyExistingRiverMouths: Area, identify_existing_river_mouths
using SharedArrays

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
  "--accumulated-flow-filepath","-a"
    help = "Accumulated flow filepath"
  "--accumulated-flow-fieldname","-f"
    help = "Accumulated flow fieldname"
  "--search-areas-filepath","-s"
    help = "Search areas filepath"
  end
  args = parse_args(settings)
  if args["accumulated-flow-filepath"] != nothing &&
     args["accumulated-flow-fieldname"] != nothing &&
     args["search-areas-filepath"] != nothing
    args["find-existing-river-mouths"] = true
  elseif args["accumulated-flow-filepath"] != nothing ||
         args["accumulated-flow-fieldname"] != nothing ||
         args["search-areas-filepath"] != nothing
    args["find-existing-river-mouths"] = false
    error("Must specify all of options for searching for existing filename or none")
  else
    args["find-existing-river-mouths"] = false
  end
  return args
end

function bifurcated_river_mouth_identification_driver(args::Dict{String})
  cells::Cells = load_icosahedral_grid(args["grid-filepath"])
  lsmask::Array{Bool} = load_landsea_mask(args["lsmask-filepath"],args["lsmask-fieldname"])
  river_deltas::Array{RiverDelta} = load_river_deltas_from_file(args["river-deltas-filepath"])
  local existing_river_mouths::Dict{String,CartesianIndex}
  if args["find-existing-river-mouths"]
    accumulated_flow::Array{Int64} = load_accumulated_flow(args["accumulated-flow-filepath"],
                                                           args["accumulated-flow-fieldname"])
    search_areas::Dict{String,Area} = load_search_areas_from_file(args["search-areas-filepath"])
    existing_river_mouths = identify_existing_river_mouths(cells,
                                                           accumulated_flow,
                                                           search_areas)
  else
    existing_river_mouths = Dict{String,CartesianIndex}()
  end
  river_mouth_indices::Dict{String,Array{CartesianIndex}} =
    identify_bifurcated_river_mouths(river_deltas,
                                     cells,
                                     SharedArray(lsmask))
  write_river_mouth_indices(river_mouth_indices,existing_river_mouths,
                            args["output-river-mouths-filepath"])
end

args = pass_arguments()
bifurcated_river_mouth_identification_driver(args)
