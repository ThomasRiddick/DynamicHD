push!(LOAD_PATH, "/Users/thomasriddick/Documents/workspace/Dynamic_HD_Code/"
                  "src/julia_src/hdparametergenerator/")
using ArgParse

function pass_arguments()
  settings = ArgParseSettings()
  @add_arg_table settings begin
  "--river-directions-filepath", "-r"
    help = "Filepath to river directions file"
    required = true
  "--grid-filepath", "-g"
    help = "Filepath to ICON grid file"
    required = false
  "--orography-variance-filepath", "-v"
    help = "Filepath to orography variance file"
    required = true
  "--innerslope-filepath", "-i"
    help = "Filepath to innerslope file"
    required = true
  "--orography-filepath", "-o"
    help = "Filepath to orography file"
    required = true
  "--glacier_mask_filepath", "-m"
    help = "Glacier mask filepath"
    required = true
  "--landsea-mask-filepath", "-l"
    help = "Filepath to landsea mask file"
    required = true
  "--cell-areas-filepath", "-a"
    help = "Filepath to cell areas file"
    required = true
  "--configuration-filepath", "-c"
    help = "Filepath to configuration file"
    required = true
  "--output-hdpara-filepath", "-h"
    help = "Target filepath for output HD parameters file"
    required = true
  end
  return parse_args(settings)
end

function parameter_generation_driver(input_filepaths::Dict,
                                     output_hdpara_filepath::AbstractString)
  configuration::Configuration = load_configuration(input_filepaths)
  input_data::InputData,grid::Grid = load_input_data(input_filepaths)
  number_of_riverflow_reservoirs::Array,
  riverflow_retention_coefficients::Array,
  number_of_overlandflow_reservoirs::Array,
  overlandflow_retention_coefficients::Array,
  number_of_baseflow_reservoirs::Array,
  baseflow_retention_coefficients::Array = generate_parameters(configuration,input_data,grid)
  write_hdpara_file(output_hdpara_filepath,input_data,
                    number_of_riverflow_reservoirs,
                    riverflow_retention_coefficients,
                    number_of_overlandflow_reservoirs,
                    overlandflow_retention_coefficients,
                    number_of_baseflow_reservoirs,
                    baseflow_retention_coefficients)
end

function main()
  args = pass_arguments()
  parameter_generation_driver(args,args["output_hdpara_filepath"])
end

push!(ARGS,"-g/Users/thomasriddick/Documents/data/temp/paragen_test_data/grid")
push!(ARGS,"-r/Users/thomasriddick/Documents/data/temp/paragen_test_data/rdirs")
push!(ARGS,"-v/Users/thomasriddick/Documents/data/temp/paragen_test_data/variance")
push!(ARGS,"-i/Users/thomasriddick/Documents/data/temp/paragen_test_data/innerslope")
push!(ARGS,"-o/Users/thomasriddick/Documents/data/temp/paragen_test_data/orography")
push!(ARGS,"-m/Users/thomasriddick/Documents/data/temp/paragen_test_data/glacier")
push!(ARGS,"-l/Users/thomasriddick/Documents/data/temp/paragen_test_data/landsea")
push!(ARGS,"-a/Users/thomasriddick/Documents/data/temp/paragen_test_data/areas")
push!(ARGS,"-c/Users/thomasriddick/Documents/data/temp/paragen_test_data/config")
push!(ARGS,"-h/Users/thomasriddick/Documents/data/temp/paragen_test_data/hdpara")
main()
