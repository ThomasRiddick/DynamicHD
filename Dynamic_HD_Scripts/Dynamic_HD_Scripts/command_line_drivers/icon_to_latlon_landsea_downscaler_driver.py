import xarray as xr
from Dynamic_HD_Scripts.utilities.check_driver_inputs \
    import check_input_files, check_output_files

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
        input_coarse_to_fine_cell_numbers_mapping_in_ds =
            open_dataset(self.args["input_coarse_to_fine_cell_numbers_mapping_filepath"])
        input_coarse_to_fine_cell_numbers_mapping_in = \
            input_coarse_to_fine_cell_numbers_mapping_in_ds[self.args[
            "input_coarse_to_fine_cell_numbers_mapping_fieldname"]].values
        coarse_landsea_mask_in_ds =
            open_dataset(self.args["input_coarse_landsea_mask_filepath"])
        coarse_landsea_mask_in =
            coarse_landsea_mask_in_ds[self.args["input_coarse_landsea_mask_fieldname"]].values
        fine_landsea_mask_out = np.zeros(input_coarse_to_fine_cell_numbers_mapping_in.shape,
                                         dtype=np.int64)
        # allocate(coarse_to_fine_cell_numbers_mapping(nlat,nlon))
        # do j = 1,nlon
        #   do i = 1,nlat
        #     coarse_to_fine_cell_numbers_mapping(i,j) = coarse_to_fine_cell_numbers_mapping_raw(j,i)
        #   end do
        # end do

        # fine_landsea_mask => &
        # downscale_coarse_icon_landsea_mask_to_fine_latlon_grid((coarse_landsea_mask == 1), &
        #                                                                             coarse_to_fine_cell_numbers_mapping)
        # allocate(fine_landsea_mask_integer(nlat,nlon))
        # where (fine_landsea_mask)
        #   fine_landsea_mask_integer = 1
        # else where
        #   fine_landsea_mask_integer = 0
        # end where

        # allocate(fine_landsea_mask_integer_processed(nlon,nlat))
        # do j = 1,nlon
        #   do i = 1,nlat
        #     fine_landsea_mask_integer_processed(j,i) = fine_landsea_mask_integer(i,j)
        #   end do
        # end do

        fine_landsea_mask_out_ds = \
            input_coarse_to_fine_cell_numbers_mapping_in_ds.Copy(deep=True,
                                                                 data={"lsm":
                                                                       fine_landsea_mask_out})
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
                        metavar='Coarse to Fine Cell Numbers Mapping Filepath',
                        type=str,
                        help="Filepath to input netCDF file containing a mapping from "
                             "the coarse ICON grid to fine latlon grid")
    parser.add_argument("input_coarse_landsea_mask_filepath",
                        metavar='Coarse Landsea Mask Filepath',
                        type=str,
                        help="Filepath to input coarse landsea mask netCDF file")
    parser.add_argument("output_fine_landsea_mask_filepath",
                        metavar='Target Output Fine Landsea Mask Filepath',
                        type=str,
                        help="Target filepath for output fine landsea mask netCDF file")
    parser.add_argument("input_coarse_to_fine_cell_numbers_mapping_fieldname",
                        metavar="Coarse to Fine Cell Numbers Mapping Fieldname",
                        type=str,
                        help="Fieldname of input field containing a mapping from "
                             "the coarse ICON grid to fine latlon grid")
    parser.add_argument("input_coarse_landsea_mask_fieldname",
                        metavar='Coarse Landsea Mask Fieldname',
                        type=str,
                        help="Fieldname of input coarse landsea mask field")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_icon_to_latlon_landsea_downscaler_driver(args):
    driver = IconToLatLonLandSeaDownscalerDriver(**vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_icon_to_latlon_landsea_downscaler_driver(args)



