import xarray as xr
import argparse
import numpy as np
from compute_catchments_icon_wrapper import compute_catchments_icon_cpp
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
        check_output_files([self.args["output_catchment_number_filepath"]])
        if self.args["loop_log_filepath"] is not None:
            check_output_files([self.args["loop_log_filepath"]])
            loop_log_filepath = self.args["loop_log_filepath"]
        else:
            loop_log_filepath = ""
        if self.args["subcatchment_list_filepath"] is not None:
            generate_selected_subcatchments_only = True
            print("Note: Loops not searched for when creating catchments"
                  " for selected cells only. Loop log file path argument"
                  " ignored!")
            check_input_files([self.args["subcatchment_list_filepath"]])
            subcatchment_list_filepath = self.args["subcatchment_list_filepath"]
        else:
            generate_selected_subcatchments_only = False
            subcatchment_list_filepath = ""
        next_cell_index_in_ds = xr.open_dataset(self.args["next_cell_index_filepath"])
        next_cell_index_in = \
            next_cell_index_in_ds[self.args["next_cell_index_fieldname"]].values
        grid_params_ds = xr.open_dataset(self.args["grid_params_filepath"])
        neighboring_cell_indices_in = grid_params_ds["neighbor_cell_index"].values
        catchment_numbers_out = np.zeros(next_cell_index_in.shape,
                                         dtype=np.int32)
        compute_catchments_icon_cpp(next_cell_index_in.astype(np.int32),
                                    catchment_numbers_out,
                                    neighboring_cell_indices_in.\
                                        astype(np.int32).swapaxes(0,1).flatten(),
                                    self.args["sort_catchments_by_size"],
                                    loop_log_filepath,
                                    generate_selected_subcatchments_only,
                                    subcatchment_list_filepath)

        catchment_numbers_out_ds = \
            xr.Dataset(data_vars={"catchment":(["cell"],catchment_numbers_out)},
                       coords={"clat":next_cell_index_in_ds["clat"],
                               "clon":next_cell_index_in_ds["clon"],
                               "clon_bnds":(["cell","nv"],next_cell_index_in_ds["clon_bnds"].values),
                               "clat_bnds":(["cell","nv"],next_cell_index_in_ds["clat_bnds"].values)},
                       attrs={"number_of_grid_used":next_cell_index_in_ds.attrs["number_of_grid_used"],
                              "ICON_grid_file_uri":grid_params_ds.attrs["ICON_grid_file_uri"],
                              "uuidOfHGrid":next_cell_index_in_ds.attrs["uuidOfHGrid"]})
        catchment_numbers_out_ds["catchment"].attrs["CDI_grid_type"] = "unstructured"
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
    parser.add_argument("next_cell_index_filepath",metavar='Next_Cell_Index_Filepath',
                        type=str,
                        help="Filepath to input next cell index (river directions) netCDF file")
    parser.add_argument("output_catchment_number_filepath",
                        metavar='Output_Catchment_Number_Filepath',
                        type=str,
                        help="Target filepath for output catchment number "
                             "netCDF file")
    parser.add_argument("grid_params_filepath",
                        metavar='Grid_Parameters_Filepath',
                        type=str,
                        help="Filepath of the input atmospheric grid parameters"
                             " netCDF file")
    parser.add_argument("next_cell_index_fieldname",
                        metavar='Next_Cell_Index_Fieldname',
                        type=str,
                        help="Fieldname of the input next cell index"
                             "(river directions) input field")
    parser.add_argument("loop_log_filepath",
                        metavar="Loop_Log_File_Path",
                        type=str,
                        nargs="?",
                        default=None,
                        help="Filepath to the target loop log text file")
    parser.add_argument("-r","--sort-catchments-by-size",
                        action="store_true",
                        default=False,
                        help="Sort and renumber the catchments by size once they have "
                             "been generated")
    parser.add_argument("-s","--subcatchment-list-filepath",
                        type=str,
                        nargs="?",
                        default=None,
                        help="Instead of generating full catchment set"
                             "generate the subset of catchments for the points "
                             "(no necessarily river mouths) listed in "
                             "this file")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_catchment_computation_icon_driver(args):
    driver = ComputeCatchmentsIconDriver(vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_catchment_computation_icon_driver(args)


