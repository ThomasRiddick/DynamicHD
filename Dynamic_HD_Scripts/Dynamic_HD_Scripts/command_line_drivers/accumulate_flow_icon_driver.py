import xarray as xr
import argparse
import numpy as np
from Dynamic_HD_Scripts.utilities.check_driver_inputs \
    import check_input_files, check_output_files
from Dynamic_HD_Scripts.tools.flow_to_grid_cell \
    import accumulate_flow_icon_single_index

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
        if self.args["bifurcated_next_cell_index_filepath"] is not None:
            check_input_files([self.args["bifurcated_next_cell_index_filepath"]])
        next_cell_index_in_ds = xr.open_dataset(self.args["next_cell_index_filepath"])
        next_cell_index_in = \
            next_cell_index_in_ds[self.args["next_cell_index_fieldname"]].values
        if self.args["bifurcated_next_cell_index_filepath"] is not None:
            bifurcated_next_cell_index_in_ds = \
                xr.open_dataset(self.args["bifurcated_next_cell_index_filepath"])
            bifurcated_next_cell_index_in = \
                bifurcated_next_cell_index_in_ds[
                self.args["bifurcated_next_cell_index_fieldname"]].values
        else:
            bifurcated_next_cell_index_in = None
        grid_params_ds = xr.open_dataset(self.args["grid_params_filepath"])
        neighboring_cell_indices_in = grid_params_ds["neighbor_cell_index"].values
        if bifurcated_next_cell_index_in is not None:
            cumulative_flow_out = \
                accumulate_flow_icon_single_index(cell_neighbors=
                                                  neighboring_cell_indices_in.swapaxes(0,1),
                                                  input_river_directions=
                                                  next_cell_index_in,
                                                  input_bifurcated_river_directions=
                                                  bifurcated_next_cell_index_in.swapaxes(0,1))
        else:
            cumulative_flow_out = \
                accumulate_flow_icon_single_index(cell_neighbors=
                                                  neighboring_cell_indices_in.swapaxes(0,1),
                                                  input_river_directions=
                                                  next_cell_index_in)
        cumulative_flow_out_ds = \
            xr.Dataset(data_vars={"acc":(["cell"],cumulative_flow_out)},
                       coords={"clat":next_cell_index_in_ds["clat"],
                               "clon":next_cell_index_in_ds["clon"],
                               "clon_bnds":(["cell","nv"],next_cell_index_in_ds["clon_bnds"].values),
                               "clat_bnds":(["cell","nv"],next_cell_index_in_ds["clat_bnds"].values)},
                       attrs={"number_of_grid_used":next_cell_index_in_ds.attrs["number_of_grid_used"],
                              "grid_file_uri":next_cell_index_in_ds.attrs["grid_file_uri"],
                              "uuidOfHGrid":next_cell_index_in_ds.attrs["uuidOfHGrid"]})
        cumulative_flow_out_ds["acc"].attrs["CDI_grid_type"] = "unstructured"
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
                        metavar='Grid_Parameters_Filepath',
                        type=str,
                        help="Filepath of the input atmospheric grid parameters"
                             " netCDF file")
    parser.add_argument("next_cell_index_filepath",metavar='Next_Cell_Index_Filepath',
                        type=str,
                        help="Filepath to input next cell index (river directions) netCDF file")
    parser.add_argument("output_cumulative_flow_filepath",
                        metavar='Target_Output_Accumulated_Flow_Filepath',
                        type=str,
                        help="Target filepath for output accumulated flow "
                             "netCDF file")
    parser.add_argument("next_cell_index_fieldname",
                        metavar='Next_Cell_Index_Fieldname',
                        type=str,
                        help="Fieldname of the input next cell index"
                             "(river directions) field")
    parser.add_argument("bifurcated_next_cell_index_filepath",
                        metavar='Bifurcated_Next_Cell_Index_Filepath',
                        type=str,
                        default=None,
                        nargs="?",
                        help="Filepath to bifurcated input next cell index "
                             "(river directions) netCDF file (optional)")
    parser.add_argument("bifurcated_next_cell_index_fieldname",
                        metavar='Bifurcated_Next_Cell_Index_Fieldname',
                        type=str,
                        default=None,
                        nargs="?",
                        help="Fieldname of the bifurcated input next cell index"
                             "(river directions) field (optional)")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_flow_accumulation_icon_driver(args):
    driver = AccumulateFlowIconDriver(vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_flow_accumulation_icon_driver(args)
