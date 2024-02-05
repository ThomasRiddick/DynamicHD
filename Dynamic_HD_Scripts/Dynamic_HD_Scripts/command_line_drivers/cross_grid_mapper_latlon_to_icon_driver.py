import xarray as xr
from Dynamic_HD_Scripts.utilities.check_driver_inputs \
    import check_input_files, check_output_files

class CrossGridMapperLatLonToICONDriver:

    def __init__(self,args):
        self.args = args

    def run(self):
        print("*** LatLon to ICON Cross Grid Mapper ***")
        print("Settings:")
        for key,value in self.args.items():
            print("{}: {}".format(key,value))
        check_input_files([self.args["coarse_grid_params_filepath"],
                           self.args["input_fine_orography_filepath"]])
        check_output_files([self.args["output_cell_numbers_filepath"]])
        fine_orography_ds  = open_dataset(self.args["input_fine_orography_filepath"])
        fine_orography = \
            fine_orography_ds[self.args["input_fine_orography_fieldname"]].values
        coarse_grid_params_ds = open_dataset(self.args["coarse_grid_params_filepath"])

        # do j = 1,3
        #   do i=1,ncells_coarse
        #     cell_vertices_lats(i,j) = cell_vertices_lats_raw(j,i)*(180.0/PI)
        #     if(cell_vertices_lats(i,j) > 90.0 - ABS_TOL_FOR_DEG) cell_vertices_lats(i,j) = 90.0
        #     if(cell_vertices_lats(i,j) < -90.0 + ABS_TOL_FOR_DEG) cell_vertices_lats(i,j) = -90.0
        #     cell_vertices_lons(i,j) = cell_vertices_lons_raw(j,i)*(180.0/PI)
        #   end do
        # end do
        # call cross_grid_mapper_latlon_to_icon(pixel_center_lats,&
        #                                       pixel_center_lons,&
        #                                       cell_vertices_lats, &
        #                                       cell_vertices_lons, &
        #                                       cell_neighbors, &
        #                                       output_cell_numbers, &
        #                                       .true.)
        # output_cell_numbers_processed = transpose(output_cell_numbers)
        output_cell_numbers_ds = fine_orography_ds.Copy(deep=True,
                                                        data={"cell_index":
                                                              output_cell_numbers})
        output_cell_numbers_ds.to_netcdf(self.args["output_cell_numbers_filepath"])

class Arguments:
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser(prog='LatLon to ICON Cross Grid Mapping Tool',
                                     description='',
                                     epilog="Generate a mapping from a "
                                            "latitude-longitude grid "
                                            "to an ICON icosohedral grid")
    parser.add_argument("coarse_grid_params_filepath",
                        metavar='Coarse Grid Parameters Filepath',
                        type=str,
                        help="Filepath of the input coarse atmospheric grid parameters"
                             " netCDF file")
    parser.add_argument("input_fine_orography_filepath",
                        metavar='Input Fine Orography Filepath',
                        type=str,
                        help="Filepath of the input fine orography"
                             " netCDF file - this is only used to"
                             " determine the parameters of the"
                             " latitude-longitude grid itself"
                             " and not for the actual height data")
    parser.add_argument("input_fine_orography_fieldname",
                        metavar='Input Fine Orography Fieldname',
                        type=str,
                        help="Filepath of the input fine orography field")
    parser.add_argument("output_cell_numbers_filepath",
                        metavar='Output Cell Numbers Filepath',
                        type=str,
                        help="Target output filepath for cell numbers"
                             " netCDF file")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_cross_grid_mapper_latlon_to_icon_driver(args):
    driver = CrossGridMapperLatLonToICONDriver(**vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_cross_grid_mapper_latlon_to_icon_driver(args)
