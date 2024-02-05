import xarray as xr
from Dynamic_HD_Scripts.utilities.check_driver_inputs \
    import check_input_files, check_output_files

class ComputeCatchmentsIconDriver:

    def __init__(self,args):
        self.args = args

    def run(self):
        print("*** ICON Catchment Computation Tool ***")
        print("Settings:")
        for key,value in self.args.items():
            print("{}: {}".format(key,value))
        check_input_files([self.args["next_cell_index_filepath"],
                           self.args["grid_params_filepath"]])
        check_output_files([self.args["output_catchment_number_filepath"],
                            self.args["loop_log_file_path"]])
        if self.args["subcatchment-list-filepath"] is not None:
            generate_selected_subcatchments_only = True
            print("Note: Loops not searched for when creating catchments"
                  " for selected cells only. Loop log file path argument"
                  " ignored!")
            check_input_file(self.args["subcatchment_list_filepath"])
        else:
            generate_selected_subcatchments_only = False
        next_cell_index_in_ds = open_dataset(self.args["next_cell_index_filepath"])
        next_cell_index_in = \
            next_cell_index_in_ds[self.args["next_cell_index_fieldname"]].values
        grid_params_ds = open_dataset(self.args["grid_params_filepath"])
        neighboring_cell_indices_in = grid_params_ds["neighbor_cell_index"]
        catchment_numbers_out = np.zeros(next_cell_index_in.shape,
                                         dtype=np.int64)
        compute_catchments_icon_cpp(next_cell_index_in,
                                    catchment_numbers_out,
                                    neighboring_cell_indices_in,
                                    self.args["sort_catchments_by_size"],
                                    self.args["loop_log_filepath"]
                                    generate_selected_subcatchments_only,
                                    self.args["subcatchment_list_filepath"])
        catchment_numbers_out_ds = \
            next_cell_index_in_ds.Copy(deep=True,
                                       data={"catchment":
                                             catchment_numbers_out})
        catchment_numbers_out_ds.\
            to_netcdf(self.args["output_catchment_number_filepath"])

class Arguments:
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser(prog='Icosahedral Grid Catchment Computation Tool',
                                     description='Generate the catchments of a set of '
                                                 'river direction on the ICON icosahedral '
                                                 'grid',
                                     epilog='')
    parser.add_argument("next_cell_index_filepath",metavar='Next Cell Index Filepath',
                        type=str,
                        help="Filepath to input next cell index (river directions) netCDF file")
    parser.add_argument("output_catchment_number_filepath",
                        metavar='Output Catchment Number Filepath',
                        type=str,
                        help="Target filepath for output catchment number "
                             "netCDF file")
    parser.add_argument("grid_params_filepath",
                        metavar='Grid Parameters Filepath',
                        type=str,
                        help="Filepath of the input atmospheric grid parameters"
                             " netCDF file")
    parser.add_argument("next_cell_index_fieldname",
                        metavar='Next Cell Index Fieldname',
                        type=str,
                        help="Fieldname of the input next cell index"
                             "(river directions) input field")
    parser.add_argument("use_secondary_neighbors_flag",
                        metavar="Use Secondary Neighbors Flag",
                        type=bool,
                        default=True,
                        help="Use the 8 or 9 additional neighbors which "
                             "share vertices with this cell but not edges")
    parser.add_argument("loop_log_file_path",
                        metavar="Loop Log File Path",
                        type=str,
                        default="",
                        help="Filepath to the target loop log text file")
    parser.add_argument("sort_catchments_by_size_flag",
                        metavar="Sort Catchments By Size Flag",
                        type=bool,
                        default=False,
                        help="Sort and renumber the catchments by size once they have"
                             "been generated")
    parser.add_argument("-s","--subcatchment-list-filepath",
                        type=str,
                        default="",
                        help="Instead of generating full catchment set"
                             "generate the subset of catchments for the points "
                             "(no necessarily river mouths) listed in "
                             "this file")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_catchment_computation_icon_driver(args):
    driver = ComputeCatchmentsIconDriver(**vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_catchment_computation_icon_driver(args)


