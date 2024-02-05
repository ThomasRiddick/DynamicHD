import xarray as xr
from Dynamic_HD_Scripts.utilities.check_driver_inputs \
    import check_input_files, check_output_files

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
        check_output_files(self.args["next_cell_index_out_filepath"],
                           self.args["loop_log_file_path"])
        orography_in_ds = open_dataset(self.args["orography_filepath"])
        orography_in = \
            next_cell_index_in_ds[self.args["orography_fieldname"]].values
        landsea_in_ds = open_dataset(self.args["landsea_filepath"])
        landsea_in = \
            landsea_in_ds[self.args["landsea_fieldname"]].values
        if landsea_in.dtype == np.int64 or landsea.dtype == np.int32:
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
        next_cell_index_out = np.zeros(orography_in.shape,
                                       dtype=np.int64)
        determine_river_directions_icon_cpp(orography_in,
                                            landsea_in_int,
                                            landsea_in_double,
                                            true_sinks_in_int,
                                            neighboring_cell_indices_in,
                                            next_cell_index_out,
                                            self.args["fractional_landsea_mask_in"],
                                            self.args["always_flow_to_sea_in"],
                                            self.args["mark_pits_as_true_sinks_in"])
        next_cell_index_out_ds = \
            orography_in_ds.Copy(deep=True,
                                 data={"next_cell_index":
                                       next_cell_index_out})
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
                        metavar='Output Next Cell Index Filepath',
                        type=str,
                        help="Full path to target output file for the next "
                             "cell index values; these are the ICON equivalent "
                             "of river directions.")
    parser.add_argument("orography_filepath",metavar='Orography Filepath',
                        type=str,
                        help="Full path to the input orography file")
    parser.add_argument("landsea_filepath",metavar='Landsea Mask Filepath',
                        type=str,
                        help="Full path to input landsea mask file")
    parser.add_argument("true_sinks_filepath",metavar='True Sinks Filepath',
                        type=str,
                        help="Full path to input true sinks file")
    parser.add_argument("grid_params_filepath",metavar='Grid Parameters Filepath',
                        type=str,
                        help="Full path to the grid description file for the ICON "
                             "grid being used")
    parser.add_argument("orography_fieldname",metavar='Orography Fieldname',
                        type=str,
                        help="Name of orography field within the orography file")
    parser.add_argument("landsea_fieldname",metavar='Landsea Fieldname',
                        type=str,
                        help="Name of the landsea mask field within the landsea "
                             "mask file")
    parser.add_argument("true_sinks_fieldname",metavar='True Sinks Fieldname',
                        type=str,
                        help="Name of the true sinks field within the true sinks "
                             "file")
    parser.add_argument("fractional_landsea_mask_flag",metavar='Fractional Landsea Mask Flag',
                        type=bool,
                        default=False,
                        help="Land sea mask expresses fraction of land "
                             "as a floating point number (default false)")
    parser.add_argument("always_flow_to_sea_flag",metavar='Always Flow to Sea Flag',
                        type=bool,
                        default=True,
                        help="Alway mark flow direction towards a neighboring "
                             "ocean point even when a neighboring land point "
                             "has a lower elevation (default true)")
    parser.add_argument("mark_pits_as_true_sinks_flag",metavar='Mark Pits As True Sinks Flag',
                        type=bool,
                        default=False,
                        help="Mark any depression found as a true sink point "
                             "(default true)")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_determine_river_directions_icon_driver(args):
    driver = DetermineRiverDirectionsIconDriver(**vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_determine_river_directions_icon_driver(args)
