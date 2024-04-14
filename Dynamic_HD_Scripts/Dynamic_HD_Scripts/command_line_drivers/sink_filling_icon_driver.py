import xarray as xr
import argparse
import numpy as np
from sink_filling_icon_wrapper import sink_filling_icon_cpp
from Dynamic_HD_Scripts.utilities.check_driver_inputs \
    import check_input_files, check_output_files

class SinkFillingIconDriver:

    def __init__(self,args):
        self.args = args

    def run(self):
        print("*** ICON Sink Filling Tool ***")
        print("Settings:")
        for key,value in self.args.items():
            print("{}: {}".format(key,value))
        check_input_files([self.args["input_orography_filepath"],
                           self.args["landsea_filepath"],
                           self.args["true_sinks_filepath"],
                           self.args["grid_params_filepath"]])
        check_output_files([self.args["output_orography_filepath"]])
        orography_in_ds = xr.open_dataset(self.args["input_orography_filepath"])
        orography_inout = \
            orography_in_ds[self.args["input_orography_fieldname"]].values
        landsea_in_ds = xr.open_dataset(self.args["landsea_filepath"])
        landsea_in = \
            landsea_in_ds[self.args["landsea_fieldname"]].values
        if landsea_in.dtype == np.int64 or landsea_in.dtype == np.int32:
            landsea_in_int = landsea_in
            landsea_in_double = np.zeros((1,),dtype=np.float64)
        elif landsea_in.dtype == np.float64 or landsea_in.dtype == np.float32:
            landsea_in_int = np.zeros((1,),dtype=np.int32)
            landsea_in_double = landsea_in
        else:
            raise RuntimeError("Landsea mask type not recognised")
        true_sinks_in_ds = xr.open_dataset(self.args["true_sinks_filepath"])
        true_sinks_in_int = \
            true_sinks_in_ds[self.args["true_sinks_fieldname"]].values
        grid_params_ds = xr.open_dataset(self.args["grid_params_filepath"])
        neighboring_cell_indices_in = grid_params_ds["neighbor_cell_index"].values
        sink_filling_icon_cpp(neighboring_cell_indices_in.\
                              astype(np.int32).swapaxes(0,1).flatten(),
                              orography_inout.astype(np.float64),
                              landsea_in_int.astype(np.int32),
                              landsea_in_double.astype(np.float64),
                              true_sinks_in_int.astype(np.int32),
                              self.args["fractional_landsea_mask_flag"],
                              self.args["set_land_sea_as_no_data_flag"],
                              self.args["add_slope_flag"],
                              self.args["epsilon"])

        output_orography_ds = \
            xr.Dataset(data_vars={"cell_elevation":(["cell"],orography_inout)},
                       coords={"clat":orography_in_ds["clat"],
                               "clon":orography_in_ds["clon"],
                               "clon_bnds":(["cell","nv"],orography_in_ds["clon_bnds"].values),
                               "clat_bnds":(["cell","nv"],orography_in_ds["clat_bnds"].values)},
                       attrs={"number_of_grid_used":orography_in_ds.attrs["number_of_grid_used"],
                              "grid_file_uri":orography_in_ds.attrs["grid_file_uri"],
                              "uuidOfHGrid":orography_in_ds.attrs["uuidOfHGrid"]})
        output_orography_ds["cell_elevation"].attrs["CDI_grid_type"] = "unstructured"
        output_orography_ds.\
            to_netcdf(self.args["output_orography_filepath"])

class Arguments:
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser(prog='ICON Sink Filling Tool',
                                     description="Fill the sinks in an orography "
                                                 "using an accelerated priority "
                                                 "flood technique on a ICON "
                                                 "icosohedral grid",
                                     epilog='')
    parser.add_argument("input_orography_filepath",metavar='Input_Orography_Filepath',
                        type=str,
                        help="Full path to the input orography file")
    parser.add_argument("landsea_filepath",metavar='Landsea_Mask_Filepath',
                        type=str,
                        help="Full path to input landsea mask file")
    parser.add_argument("true_sinks_filepath",metavar='True_Sinks_Filepath',
                        type=str,
                        help="Full path to input true sinks file")
    parser.add_argument("output_orography_filepath",metavar='Output_Orography_Filepath',
                        type=str,
                        help="Full path to target output orography file")
    parser.add_argument("grid_params_filepath",metavar='Grid_Parameters_Filepath',
                        type=str,
                        help="Full path to the grid description file for the ICON "
                             "grid being used")
    parser.add_argument("input_orography_fieldname",metavar='Input_Orography_Fieldname',
                        type=str,
                        help="Name of input orography field within the specified file")
    parser.add_argument("landsea_fieldname",metavar='Landsea_Fieldname',
                        type=str,
                        help="Name of the landsea mask field within the specified file")
    parser.add_argument("true_sinks_fieldname",metavar='True_Sinks_Fieldname',
                        type=str,
                        help="Name of the true sinks field within the specified file")
    parser.add_argument("epsilon",metavar='Epsilon',
                        type=float,
                        nargs="?",
                        default=0.0,
                        help="Additional height added to each progressive cell when"
                             "adding a slight slope")
    parser.add_argument("-n","--set-land-sea-as-no-data-flag",
                        action="store_true",
                        default=False,
                        help="Flag to turn on and off setting "
                             "all landsea points to no data")
    parser.add_argument("-s","--add-slope-flag",
                        action="store_true",
                        default=False,
                        help="Land sea mask expresses fraction of land "
                             "as a floating point number which requires "
                             "conversion to a binary mask(default false)")
    parser.add_argument("-f","--fractional-landsea-mask-flag",
                        action="store_true",
                        default=False,
                        help="Land sea mask expresses fraction of land "
                             "as a floating point number which requires "
                             "conversion to a binary mask(default false)")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_sink_filling_icon_driver(args):
    driver = SinkFillingIconDriver(vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_sink_filling_icon_driver(args)
