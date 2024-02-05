import xarray as xr
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
        orography_in_ds = open_dataset(self.args["input_orography_filepath"])
        orography_inout = \
            next_cell_index_in_ds[self.args["input_orography_fieldname"]].values
        landsea_in_ds = open_dataset(self.args["landsea_filepath"])
        landsea_in = \
            landsea_in_ds[self.args["landsea_fieldname"]].values
        if landsea_in.dtype == np.int64 or landsea.dtype == np.int32
            landsea_in_int = landsea_in
            landsea_in_double = None
        else if landsea_in.dtype == np.float64 or landsea.dtype == np.float32:
            landsea_in_int = None
            landsea_in_double = landsea_in
        else raise RuntimeError("Landsea mask type not recognised")
        true_sinks_in_ds = open_dataset(self.args["true_sinks_filepath"])
        true_sinks_in_int = \
            true_sinks_in_ds[self.args["true_sinks_fieldname"]].values
        grid_params_ds = open_dataset(self.args["grid_params_filepath"])
        neighboring_cell_indices_in = grid_params_ds["neighbor_cell_index"]
        sink_filling_icon_cpp(neighboring_cell_indices_in,
                              orography_inout,
                              landsea_in_int,
                              landsea_in_double,
                              true_sinks_in_int,
                              self.arg["fractional_landsea_mask_flag"],
                              self.arg["set_ls_as_no_data_flag"],
                              self.arg["add_slope_flag"],
                              self.arg["epsilon"])
        output_orography_ds = \
            orography_in_ds.Copy(deep=True,
                                 data={"cell_elevation":
                                       orography_inout})
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
    parser.add_argument("input_orography_filepath",metavar='Input Orography Filepath',
                        type=str,
                        help="Full path to the input orography file")
    parser.add_argument("landsea_filepath",metavar='Landsea Mask Filepath',
                        type=str,
                        help="Full path to input landsea mask file")
    parser.add_argument("true_sinks_filepath",metavar='True Sinks Filepath',
                        type=str,
                        help="Full path to input true sinks file")
    parser.add_argument("output_orography_filepath",metavar='Output Orography Filepath',
                        type=str,
                        help="Full path to target output orography file")
    parser.add_argument("grid_params_filepath",metavar='Grid Parameters Filepath',
                        type=str,
                        help="Full path to the grid description file for the ICON "
                             "grid being used")
    parser.add_argument("input_orography_fieldname",metavar='Input Orography Fieldname',
                        type=str,
                        help="Name of input orography field within the specified file")
    parser.add_argument("landsea_fieldname",metavar='Landsea Fieldname',
                        type=str,
                        help="Name of the landsea mask field within the specified file")
    parser.add_argument("true_sinks_fieldname",metavar='True Sinks Fieldname',
                        type=str,
                        help="Name of the true sinks field within the specified file")
    parser.add_argument("set_land_sea_as_no_data_flag",metavar='Set Landsea as No Data Flag',
                        type=bool,
                        help="Flag to turn on and off setting "
                             "all landsea points to no data")
    parser.add_argument("add_slope_flag",metavar='Add Slope Flag',
                        type=bool,
                        help="Land sea mask expresses fraction of land "
                             "as a floating point number which requires "
                             "conversion to a binary mask(default false)")
    parser.add_argument("epsilon",metavar='Epsilon',
                        type=float,
                        help="Additional height added to each progressive cell when"
                             "adding a slight slope")
    parser.add_argument("use_secondary_neighbors_flag",
                        metavar="Use Secondary Neighbors Flag",
                        type=bool,
                        default=True,
                        help="Use the 8 or 9 additional neighbors which "
                             "share vertices with this cell but not edges")
    parser.add_argument("fractional_landsea_mask_flag",
                        metavar='Fractional Landsea Mask Flag',
                        type=bool,
                        default=False,
                        help="Land sea mask expresses fraction of land "
                             "as a floating point number which requires "
                             "conversion to a binary mask(default false)")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_sink_filling_icon_driver(args):
    driver = SinkFillingIconDriver(**vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_sink_filling_icon_driver(args)
