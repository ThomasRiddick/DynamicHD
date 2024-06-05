import xarray as xr
import argparse
from Dynamic_HD_Scripts.utilities.check_driver_inputs \
    import check_input_files, check_output_files
from Dynamic_HD_Scripts.tools.landsea_downscaler_driver \
    import landsea_downscaler_icosohedral_cell_latlon_pixel

class IconToLatLonLandSeaDownscalerDriver:

    def __init__(self,args):
        self.args = args

    def run(self):
        print("*** ICON To LatLon Landsea Downscaling Tool ***")
        print("Settings:")
        for key,value in self.args.items():
            print("{}: {}".format(key,value))
        check_input_files([self.args["input_coarse_to_fine_cell_numbers_mapping_filepath"],
                           self.args["input_coarse_landsea_mask_filepath"]])
        check_output_files([self.args["output_fine_landsea_mask_filepath"]])
        coarse_to_fine_cell_numbers_mapping_in_ds = \
            xr.open_dataset(self.args["input_coarse_to_fine_cell_numbers_mapping_filepath"])
        coarse_to_fine_cell_numbers_mapping_in = \
            coarse_to_fine_cell_numbers_mapping_in_ds[self.args[
            "input_coarse_to_fine_cell_numbers_mapping_fieldname"]].values
        coarse_landsea_mask_in_ds = \
            xr.open_dataset(self.args["input_coarse_landsea_mask_filepath"])
        coarse_landsea_mask_in = \
            coarse_landsea_mask_in_ds[self.args["input_coarse_landsea_mask_fieldname"]].values
        fine_landsea_mask_out = \
            landsea_downscaler_icosohedral_cell_latlon_pixel(
                input_coarse_landsea_mask=coarse_landsea_mask_in,
                coarse_to_fine_cell_numbers_mapping=coarse_to_fine_cell_numbers_mapping_in)
        fine_landsea_mask_out_ds = \
            xr.Dataset(data_vars={"lsm":(["lat","lon"],fine_landsea_mask_out)},
                       coords={"lat":coarse_to_fine_cell_numbers_mapping_in_ds["lat"],
                               "lon":coarse_to_fine_cell_numbers_mapping_in_ds["lon"]})
        fine_landsea_mask_out_ds.to_netcdf(self.args["output_fine_landsea_mask_filepath"])

class Arguments:
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser(prog='ICON To LatLon Landsea Downscaling Tool',
                                     description="Downscale landsea masks from a coarse "
                                                 "ICON grid to a fine latlon grid",
                                     epilog='')
    parser.add_argument("input_coarse_to_fine_cell_numbers_mapping_filepath",
                        metavar='Coarse_To_Fine_Cell_Numbers_Mapping_Filepath',
                        type=str,
                        help="Filepath to input netCDF file containing a mapping from "
                             "the coarse ICON grid to fine latlon grid")
    parser.add_argument("input_coarse_landsea_mask_filepath",
                        metavar='Coarse_Landsea_Mask_Filepath',
                        type=str,
                        help="Filepath to input coarse landsea mask netCDF file")
    parser.add_argument("output_fine_landsea_mask_filepath",
                        metavar='Target_Output_Fine_Landsea_Mask_Filepath',
                        type=str,
                        help="Target filepath for output fine landsea mask netCDF file")
    parser.add_argument("input_coarse_to_fine_cell_numbers_mapping_fieldname",
                        metavar="Coarse_To_Fine_Cell_Numbers_Mapping_Fieldname",
                        type=str,
                        help="Fieldname of input field containing a mapping from "
                             "the coarse ICON grid to fine latlon grid")
    parser.add_argument("input_coarse_landsea_mask_fieldname",
                        metavar='Coarse_Landsea_Mask_Fieldname',
                        type=str,
                        help="Fieldname of input coarse landsea mask field")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_icon_to_latlon_landsea_downscaler_driver(args):
    driver = IconToLatLonLandSeaDownscalerDriver(vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_icon_to_latlon_landsea_downscaler_driver(args)



