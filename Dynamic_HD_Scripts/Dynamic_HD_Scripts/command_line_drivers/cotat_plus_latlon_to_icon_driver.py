import xarray as xr
import argparse
import numpy as np
from Dynamic_HD_Scripts.utilities.check_driver_inputs \
    import check_input_files, check_output_files
from Dynamic_HD_Scripts.tools.cotat_plus_driver \
    import cotat_plus_icon_icosohedral_cell_latlon_pixel

class CotatPlusLatLonToIconDriver:

    def __init__(self,args):
        self.args = args

    def run(self):
        print("*** Cotat Plus LatLon to ICON Upscaling Tool ***")
        print("Settings:")
        for key,value in self.args.items():
            print("{}: {}".format(key,value))
        check_input_files([self.args["input_fine_rdirs_filepath"],
                           self.args["input_fine_total_cumulative_flow_filepath"],
                           self.args["coarse_grid_params_filepath"],
                           self.args["cotat_parameters_filepath"]])
        check_output_files([self.args["output_rdirs_filepath"]])
        fine_rdirs_in_ds = xr.open_dataset(self.args["input_fine_rdirs_filepath"])
        fine_rdirs_in = \
            fine_rdirs_in_ds[self.args["input_fine_rdirs_fieldname"]].values
        pixel_center_lats = fine_rdirs_in_ds["lat"]
        pixel_center_lons = fine_rdirs_in_ds["lon"]
        fine_total_cumulative_flow_ds = \
            xr.open_dataset(self.args["input_fine_total_cumulative_flow_filepath"])
        fine_total_cumulative_flow_in = \
            fine_total_cumulative_flow_ds[self.args["input_fine_total_cumulative_flow_fieldname"]].values
        coarse_grid_params_ds = xr.open_dataset(self.args["coarse_grid_params_filepath"])
        neighboring_cell_indices_in = coarse_grid_params_ds["neighbor_cell_index"].values
        cell_vertices_lats = coarse_grid_params_ds["clat_vertices"]
        cell_vertices_lons = coarse_grid_params_ds["clon_vertices"]
        coarse_next_cell_index_out = np.zeros((coarse_grid_params_ds["cell"].values.shape[0],),
                                               dtype=np.int64)
        next_cell_index_out,cell_numbers_out = \
            cotat_plus_icon_icosohedral_cell_latlon_pixel(input_fine_river_directions=
                                                          fine_rdirs_in,
                                                          input_fine_total_cumulative_flow=
                                                          fine_total_cumulative_flow_in,
                                                          cell_neighbors=
                                                          neighboring_cell_indices_in.swapaxes(0,1),
                                                          pixel_center_lats=
                                                          pixel_center_lats,
                                                          pixel_center_lons=
                                                          pixel_center_lons,
                                                          cell_vertices_lats=
                                                          cell_vertices_lats,
                                                          cell_vertices_lons=
                                                          cell_vertices_lons,
                                                          cotat_plus_parameters_filepath=
                                                          self.args["cotat_parameters_filepath"])
        next_cell_index_out_ds = \
            xr.Dataset(data_vars={"next_cell_index":(["cell"],next_cell_index_out)},
                       coords={"clat":coarse_grid_params_ds["clat"],
                               "clon":coarse_grid_params_ds["clon"],
                               "clon_bnds":(["cell","nv"],coarse_grid_params_ds["clon_vertices"].values),
                               "clat_bnds":(["cell","nv"],coarse_grid_params_ds["clat_vertices"].values)},
                       attrs={"number_of_grid_used":coarse_grid_params_ds.attrs["number_of_grid_used"],
                              "uuidOfHGrid":coarse_grid_params_ds.attrs["uuidOfHGrid"]})
        next_cell_index_out_ds["next_cell_index"].attrs["CDI_grid_type"] = "unstructured"
        next_cell_index_out_ds["clat"].attrs["bounds"] = "clat_bnds"
        next_cell_index_out_ds["clon"].attrs["bounds"] = "clon_bnds"
        next_cell_index_out_ds.\
            to_netcdf(self.args["output_rdirs_filepath"])
        if self.args["output_cell_numbers_filepath"] is not None:
            output_cell_numbers_ds = \
                xr.Dataset(data_vars={"cell_numbers":(["lat","lon"],cell_numbers_out)},
                           coords={"lat":fine_rdirs_in_ds["lat"],
                                   "lon":fine_rdirs_in_ds["lon"]})
            output_cell_numbers_ds.to_netcdf(self.args["output_cell_numbers_filepath"])

class Arguments:
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser(prog=' Cotat Plus LatLon to ICON Upscaling Tool',
                                     description='',
                                     epilog='')
    parser.add_argument("input_fine_rdirs_filepath",
                        metavar="Input_Fine_River_Direction_Filepath",
                        type=str,
                        help="Filepath to input fine river directions netCDF file")
    parser.add_argument("input_fine_total_cumulative_flow_filepath",
                        metavar="Input_File_Total_Cumulative_Flow_Filepath",
                        type=str,
                        help="Filepath to input fine total cumulative flow netCDF file")
    parser.add_argument("coarse_grid_params_filepath",
                        metavar="Coarse_Grid_Parameter_Filepath",
                        type=str,
                        help="Filepath of the input coarse atmospheric grid parameters"
                             " netCDF file")
    parser.add_argument("output_rdirs_filepath",
                        metavar="Output_River_Directions_Filepath",
                        type=str,
                        help="Target filepath for output river directions "
                             "netCDF file")
    parser.add_argument("input_fine_rdirs_fieldname",
                        metavar="Input_Fine_River_Directions_Fieldname",
                        type=str,
                        help="Fieldname of input river directions field")
    parser.add_argument("input_fine_total_cumulative_flow_fieldname",
                        metavar="Input_Fine_Total_Cumulative_Flow_Fieldname",
                        type=str,
                        help="Fieldname of input total cumulative flow field")
    parser.add_argument("cotat_parameters_filepath",
                        metavar="COTAT_Parameters_Filepath",
                        type=str,
                        help="Filepath of COTAT Plus parameters file to use")
    parser.add_argument("output_cell_numbers_filepath",
                        metavar="Output_Cell_Numbers_Filepath",
                        nargs="?",
                        default=None,
                        type=str,
                        help="Target output filepath for cell numbers"
                             " netCDF file (optional)")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_cotat_plus_latlon_to_icon_driver(args):
    driver = CotatPlusLatLonToIconDriver(vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_cotat_plus_latlon_to_icon_driver(args)
