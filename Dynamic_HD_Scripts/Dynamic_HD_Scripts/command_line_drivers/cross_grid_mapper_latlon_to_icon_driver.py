import xarray as xr
import argparse
from Dynamic_HD_Scripts.utilities.check_driver_inputs \
    import check_input_files, check_output_files
from Dynamic_HD_Scripts.tools.cross_grid_mapper_driver \
    import cross_grid_mapper_latlon_to_icon

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
        fine_orography_ds  = xr.open_dataset(self.args["input_fine_orography_filepath"])
        pixel_center_lats = fine_orography_ds["lat"]
        pixel_center_lons = fine_orography_ds["lon"]
        coarse_grid_params_ds = xr.open_dataset(self.args["coarse_grid_params_filepath"])
        neighboring_cell_indices_in = coarse_grid_params_ds["neighbor_cell_index"].values
        cell_vertices_lats = coarse_grid_params_ds["clat_vertices"]
        cell_vertices_lons = coarse_grid_params_ds["clon_vertices"]
        output_cell_numbers = \
            cross_grid_mapper_latlon_to_icon(cell_neighbors=
                                             neighboring_cell_indices_in.swapaxes(0,1),
                                             pixel_center_lats=pixel_center_lats,
                                             pixel_center_lons=pixel_center_lons,
                                             cell_vertices_lats=cell_vertices_lats,
                                             cell_vertices_lons=cell_vertices_lons)
        output_cell_numbers_ds = \
            xr.Dataset(data_vars={"cell_numbers":(["lat","lon"],output_cell_numbers)},
                       coords={"lat":fine_orography_ds["lat"],
                               "lon":fine_orography_ds["lon"]})
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
                        metavar='Coarse_Grid_Parameters_Filepath',
                        type=str,
                        help="Filepath of the input coarse atmospheric grid parameters"
                             " netCDF file")
    parser.add_argument("input_fine_orography_filepath",
                        metavar='Input_Fine_Orography_Filepath',
                        type=str,
                        help="Filepath of the input fine orography"
                             " netCDF file - this is only used to"
                             " determine the parameters of the"
                             " latitude-longitude grid itself"
                             " and not for the actual height data")
    parser.add_argument("output_cell_numbers_filepath",
                        metavar='Output_Cell_Numbers_Filepath',
                        type=str,
                        help="Target output filepath for cell numbers"
                             " netCDF file")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_cross_grid_mapper_latlon_to_icon_driver(args):
    driver = CrossGridMapperLatLonToICONDriver(vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_cross_grid_mapper_latlon_to_icon_driver(args)
