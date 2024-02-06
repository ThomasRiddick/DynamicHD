import xarray as xr
from Dynamic_HD_Scripts.utilities.check_driver_inputs \
    import check_input_files, check_output_files

class AccumulateFlowIconDriver:

    def __init__(self,args):
        self.args = args

    def run(self):
        print("*** ICON Flow Accumulation Tool ***")
        print("Settings:")
        for key,value in self.args.items():
            print("{}: {}".format(key,value))
        check_input_files([self.args["next_cell_index_filepath"],
                           self.args["grid_params_filepath"]])
        check_output_files([self.args["output_cumulative_flow_filepath"]])
        if input_bifurcated_next_cell_index_filepath is not None:
            check_input_files([self.args["input_bifurcated_next_cell_index_filepath"]])
        next_cell_index_in_ds = open_dataset(self.args["next_cell_index_filepath"])
        next_cell_index_in = \
            next_cell_index_in_ds[self.args["next_cell_index_fieldname"]].values
        if input_bifurcated_next_cell_index_filepath is not None:
            next_cell_index_in_ds = \
                open_dataset(self.args["input_bifurcated_next_cell_index_filepath"])
            next_cell_index_in = \
                next_cell_index_in_ds[self.args["next_cell_index_fieldname"]].values
        grid_params_ds = open_dataset(self.args["grid_params_filepath"])
        neighboring_cell_indices_in = grid_params_ds["neighbor_cell_index"]
        cumulative_flow_out = np.zeros(next_cell_index_in.shape,
                                       dtype=np.int64)

        # RUN FORTRAN CODE HERE
        # if (num_args == 7) then
        #   call accumulate_flow_icon_single_index(cell_neighbors, &
        #                                          rdirs, cumulative_flow, &
        #                                          bifurcated_rdirs)
        # else
        #   call accumulate_flow_icon_single_index(cell_neighbors, &
        #                                          rdirs, cumulative_flow)
        # end if

        cumulative_flow_out_ds = next_cell_index_in_ds.Copy(deep=True,
                                                            data={"acc":
                                                            cumulative_flow_out})
        cumulative_flow_out_ds.to_netcdf(self.args["output_cumulative_flow_filepath"])

class Arguments:
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser(prog='ICON Flow Accumulation Tool',
                                     description='Add bifurcated river mouths to existing'
                                                 'river directions on icosahedral grids ',
                                     epilog='')
    parser.add_argument("grid_params_filepath",
                        metavar='Grid Parameters Filepath',
                        type=str,
                        help="Filepath of the input atmospheric grid parameters"
                             " netCDF file")
    parser.add_argument("next_cell_index_filepath",metavar='Next Cell Index Filepath',
                        type=str,
                        help="Filepath to input next cell index (river directions) netCDF file")
    parser.add_argument("output_cumulative_flow_filepath",
                        metavar='Target Output Accumulated Flow Filepath',
                        type=str,
                        help="Target filepath for output accumulated flow "
                             "netCDF file")
    parser.add_argument("next_cell_index_fieldname",
                        metavar='Next Cell Index Fieldname',
                        type=str,
                        help="Fieldname of the input next cell index"
                             "(river directions) field")
    parser.add_argument("next_cell_index_filepath",metavar='Bifurcated Next Cell Index Filepath',
                        type=str,
                        default=None,
                        help="Filepath to bifurcated input next cell index "
                             "(river directions) netCDF file")
    parser.add_argument("next_cell_index_fieldname",
                        metavar='Bifurcated Next Cell Index Fieldname',
                        type=str,
                        default=None,
                        help="Fieldname of the bifurcated input next cell index"
                             "(river directions) field")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_flow_accumulation_icon_driver(args):
    driver = AccumulateFlowIconDriver(**vars(args))
    driver.run()
