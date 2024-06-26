import xarray as xr
import numpy as np
import argparse
from Dynamic_HD_Scripts.utilities.check_driver_inputs \
    import check_input_files, check_output_files
import determine_river_directions_icon_wrapper

class DetermineRiverDirectionsIconDriver:

    def __init__(self,args):
        self.args = args

    def run(self):
        print("*** ICON river direction determination tool ***")
        print("Settings:")
        for key,value in self.args.items():
            print("{}: {}".format(key,value))
        check_input_files([self.args["orography_filepath"],
                           self.args["landsea_filepath"],
                           self.args["true_sinks_filepath"],
                           self.args["grid_params_filepath"]])
        check_output_files([self.args["next_cell_index_out_filepath"]])
        orography_in_ds = xr.open_dataset(self.args["orography_filepath"])
        orography_in = \
            orography_in_ds[self.args["orography_fieldname"]].values
        landsea_in_ds = xr.open_dataset(self.args["landsea_filepath"])
        landsea_in = \
            landsea_in_ds[self.args["landsea_fieldname"]].values
        if self.args["fractional_landsea_mask"]:
            landsea_in_int = np.zeros((1,),dtype=np.int32)
            landsea_in_double = landsea_in
        else:
            landsea_in_int = landsea_in
            landsea_in_double = np.zeros((1,),dtype=np.float64)
        true_sinks_in_ds = xr.open_dataset(self.args["true_sinks_filepath"])
        true_sinks_in_int = \
            true_sinks_in_ds[self.args["true_sinks_fieldname"]].values
        grid_params_ds = xr.open_dataset(self.args["grid_params_filepath"])
        neighboring_cell_indices_in = grid_params_ds["neighbor_cell_index"].values
        next_cell_index_out = np.zeros(orography_in.shape,
                                       dtype=np.int32)
        determine_river_directions_icon_wrapper.\
            determine_river_directions_icon_cpp(orography_in.astype(np.float64),
                                                landsea_in_int.astype(np.int32),
                                                landsea_in_double.astype(np.float64),
                                                true_sinks_in_int.astype(np.int32),
                                                neighboring_cell_indices_in.\
                                                astype(np.int32).swapaxes(0,1).flatten(),
                                                next_cell_index_out,
                                                self.args["fractional_landsea_mask"],
                                                self.args["always_flow_to_lowest"],
                                                self.args["mark_pits_as_true_sinks"])
        next_cell_index_out_ds = \
            xr.Dataset(data_vars={"next_cell_index":(["cell"],next_cell_index_out)},
                       coords={"clat":orography_in_ds["clat"],
                               "clon":orography_in_ds["clon"],
                               "clon_bnds":(["cell","nv"],orography_in_ds["clon_bnds"].values),
                               "clat_bnds":(["cell","nv"],orography_in_ds["clat_bnds"].values)},
                       attrs={"number_of_grid_used":orography_in_ds.attrs["number_of_grid_used"],
                              "grid_file_uri":orography_in_ds.attrs["grid_file_uri"],
                              "uuidOfHGrid":orography_in_ds.attrs["uuidOfHGrid"]})
        next_cell_index_out_ds["next_cell_index"].attrs["CDI_grid_type"] = "unstructured"
        next_cell_index_out_ds.\
            to_netcdf(self.args["next_cell_index_out_filepath"])

class Arguments:
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser(prog='ICON River Direction Determination Tool',
                                     description='Determine river directions on a ICON '
                                                 'icosahedral grid using a down slope '
                                                 'routing. Includes the resolution of '
                                                 'flat areas and the possibility of '
                                                 'marking depressions as true sink '
                                                 'points (terminal lakes of endorheic '
                                                 'basins).',
                                     epilog='')
    parser.add_argument("next_cell_index_out_filepath",
                        metavar='Output_Next_Cell_Index_Filepath',
                        type=str,
                        help="Full path to target output file for the next "
                             "cell index values; these are the ICON equivalent "
                             "of river directions.")
    parser.add_argument("orography_filepath",metavar='Orography_Filepath',
                        type=str,
                        help="Full path to the input orography file")
    parser.add_argument("landsea_filepath",metavar='Landsea_Mask_Filepath',
                        type=str,
                        help="Full path to input landsea mask file")
    parser.add_argument("true_sinks_filepath",metavar='True_Sinks_Filepath',
                        type=str,
                        help="Full path to input true sinks file")
    parser.add_argument("grid_params_filepath",metavar='Grid_Parameters_Filepath',
                        type=str,
                        help="Full path to the grid description file for the ICON "
                             "grid being used")
    parser.add_argument("orography_fieldname",metavar='Orography_Fieldname',
                        type=str,
                        help="Name of orography field within the orography file")
    parser.add_argument("landsea_fieldname",metavar='Landsea_Fieldname',
                        type=str,
                        help="Name of the landsea mask field within the landsea "
                             "mask file")
    parser.add_argument("true_sinks_fieldname",metavar='True_Sinks_Fieldname',
                        type=str,
                        help="Name of the true sinks field within the true sinks "
                             "file")
    parser.add_argument("-f","--fractional-landsea-mask",
                        action="store_true",
                        default=False,
                        help="Land sea mask expresses fraction of land "
                             "as a floating point number")
    parser.add_argument("-a","--always-flow-to-lowest",
                        action="store_true",
                        default=False,
                        help="Alway mark flow direction towards the lowest "
                             "neighbor even when there is another neighbor "
                             "that is an ocean point (but not the lowest neighbor")
    parser.add_argument("-m","--mark-pits-as-true-sinks",
                        action="store_true",
                        default=False,
                        help="Mark any depression found as a true sink point")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_determine_river_directions_icon_driver(args):
    driver = DetermineRiverDirectionsIconDriver(vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_determine_river_directions_icon_driver(args)
