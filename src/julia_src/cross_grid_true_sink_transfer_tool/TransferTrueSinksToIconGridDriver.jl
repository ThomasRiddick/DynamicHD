using ArgParse
using TransferTrueSinksToIconGridIO: load_icon_grid_parameters,load_latlon_true_sinks
using TransferTrueSinksToIconGridIO: load_latlon_grid_parameters,write_icon_grid_true_sinks
using TransferTrueSinksToIconGrid: transfer_true_sinks_from_latlon_to_icon_grid

function pass_arguments()
  settings = ArgParseSettings()
  @add_arg_table settings begin
  "--grid-filepath", "-g"
    help = "Filepath to ICON grid file"
    required = true
  "--input-true-sinks-filepath", "-i"
    help = "Filepath to input lat-lon true sinks file"
    required = true
  "--input-true-sinks-fieldname", "-f"
    help = "Fieldname of input lat-lon true sinks field"
    required = true
  "--output-true-sinks-filepath","-o"
    help = "Target filepath for output icon grid true sinks field"
    required = true
  end
  args = parse_args(settings)
  return args
end

function transfer_true_sinks_to_icon_grid_driver(args::Dict{String})
  icon_cell_indices,icon_grid_clats,icon_grid_clons =
    load_icon_grid_parameters(args["grid-filepath"])
  latlon_true_sinks_field =
    load_latlon_true_sinks(args["input-true-sinks-filepath"],
                           args["input-true-sinks-fieldname"])
  latlon_grid_clats,latlon_grid_clons =
    load_latlon_grid_parameters(args["input-true-sinks-filepath"])
  icon_grid_true_sinks::Array{Bool} =
    transfer_true_sinks_from_latlon_to_icon_grid(latlon_true_sinks_field,
                                                 latlon_grid_clats,
                                                 latlon_grid_clons,
                                                 icon_cell_indices,
                                                 icon_grid_clats,
                                                 icon_grid_clons)
  write_icon_grid_true_sinks(icon_grid_true_sinks,
                             args["output-true-sinks-filepath"])
end

args = pass_arguments()
transfer_true_sinks_to_icon_grid_driver(args)
