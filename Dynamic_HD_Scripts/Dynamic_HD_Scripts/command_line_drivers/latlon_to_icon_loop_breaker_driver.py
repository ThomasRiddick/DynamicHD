import xarray as xr
import argparse
import numpy as np
from Dynamic_HD_Scripts.utilities.check_driver_inputs \
    import check_input_files, check_output_files
from Dynamic_HD_Scripts.tools.loop_breaker_driver \
    import loop_breaker_icon_icosohedral_cell_latlon_pixel

class LatLonToIconLoopBreakerDriver:

    def __init__(self,args):
        self.args = args

    def run(self):
        print("*** LatLon to Icon Loop Breaking Tool ***")
        print("Settings:")
        for key,value in self.args.items():
            print("{}: {}".format(key,value))
        check_input_files([self.args["input_fine_rdirs_filepath"],
                           self.args["input_fine_total_cumulative_flow_filepath"],
                           self.args["input_fine_cell_numbers_filepath"],
                           self.args["coarse_grid_params_filepath"],
                           self.args["input_coarse_catchments_filepath"],
                           self.args["input_coarse_cumulative_flow_filepath"],
                           self.args["input_coarse_rdirs_filepath"],
                           self.args["input_loop_log_filepath"]])
        check_output_files([self.args["output_rdirs_filepath"]])
        fine_rdirs_in_ds = xr.open_dataset(self.args["input_fine_rdirs_filepath"])
        fine_rdirs_in = \
            fine_rdirs_in_ds[self.args["input_fine_rdirs_fieldname"]].values
        pixel_center_lats = fine_rdirs_in_ds["lat"]
        pixel_center_lons = fine_rdirs_in_ds["lon"]
        fine_total_cumulative_flow_in_ds = \
            xr.open_dataset(self.args["input_fine_total_cumulative_flow_filepath"])
        fine_total_cumulative_flow_in = \
            fine_total_cumulative_flow_in_ds[
            self.args["input_fine_total_cumulative_flow_fieldname"]].values
        fine_cell_numbers_in_ds = \
            xr.open_dataset(self.args["input_fine_cell_numbers_filepath"])
        fine_cell_numbers_in = \
            fine_cell_numbers_in_ds[
            self.args["input_fine_cell_numbers_fieldname"]].values
        coarse_grid_params_ds = xr.open_dataset(self.args["coarse_grid_params_filepath"])
        cell_vertices_lats = coarse_grid_params_ds["clat_vertices"]
        cell_vertices_lons = coarse_grid_params_ds["clon_vertices"]
        neighboring_cell_indices_in = coarse_grid_params_ds["neighbor_cell_index"].values
        coarse_catchments_in_ds = \
            xr.open_dataset(self.args["input_coarse_catchments_filepath"])
        coarse_catchments_in = \
            coarse_catchments_in_ds[
            self.args["input_coarse_catchments_fieldname"]].values
        coarse_cumulative_flow_in_ds = \
            xr.open_dataset(self.args["input_coarse_cumulative_flow_filepath"])
        coarse_cumulative_flow_in = \
            coarse_cumulative_flow_in_ds[
            self.args["input_coarse_cumulative_flow_fieldname"]].values
        coarse_rdirs_in_ds = \
            xr.open_dataset(self.args["input_coarse_rdirs_filepath"])
        coarse_rdirs_in = coarse_rdirs_in_ds[
            self.args["input_coarse_rdirs_fieldname"]].values
        loop_nums_list = []
        with open(self.args["input_loop_log_filepath"],"r") as f:
            first_line=True
            for line in f:
                if first_line:
                    first_line=False
                    continue
                loop_nums_list.append(int(line.strip()))
        next_cell_index_out = \
            loop_breaker_icon_icosohedral_cell_latlon_pixel(input_fine_rdirs=
                                                            fine_rdirs_in,
                                                            input_fine_total_cumulative_flow=
                                                            fine_total_cumulative_flow_in,
                                                            input_cell_numbers=
                                                            fine_cell_numbers_in,
                                                            input_coarse_cumulative_flow=
                                                            coarse_cumulative_flow_in,
                                                            input_coarse_catchments=
                                                            coarse_catchments_in,
                                                            input_coarse_rdirs=
                                                            coarse_rdirs_in,
                                                            input_loop_nums_list=
                                                            loop_nums_list,
                                                            cell_neighbors=
                                                            neighboring_cell_indices_in.swapaxes(0,1),
                                                            pixel_center_lats=
                                                            pixel_center_lats,
                                                            pixel_center_lons=
                                                            pixel_center_lons,
                                                            cell_vertices_lats=
                                                            cell_vertices_lats,
                                                            cell_vertices_lons=
                                                            cell_vertices_lons)

        next_cell_index_out_ds = \
            xr.Dataset(data_vars={"next_cell_index":(["cell"],next_cell_index_out)},
                       coords={"clat":coarse_grid_params_ds["clat"],
                               "clon":coarse_grid_params_ds["clon"],
                               "clon_bnds":(["cell","nv"],coarse_grid_params_ds["clon_vertices"].values),
                               "clat_bnds":(["cell","nv"],coarse_grid_params_ds["clat_vertices"].values)},
                       attrs={"number_of_grid_used":coarse_grid_params_ds.attrs["number_of_grid_used"],
                              "ICON_grid_file_uri":coarse_grid_params_ds.attrs["ICON_grid_file_uri"],
                              "uuidOfHGrid":coarse_grid_params_ds.attrs["uuidOfHGrid"]})
        next_cell_index_out_ds["next_cell_index"].attrs["CDI_grid_type"] = "unstructured"
        next_cell_index_out_ds["clat"].attrs["bounds"] = "clat_bnds"
        next_cell_index_out_ds["clon"].attrs["bounds"] = "clon_bnds"
        next_cell_index_out_ds.\
            to_netcdf(self.args["output_rdirs_filepath"])

class Arguments:
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser(prog='Lat-Lon to Icon Loop Breaker',
                                     description='Break loops in river directions '
                                                 'upscaled from a fine lat-lon grid '
                                                 'to a coarse icon grid',
                                     epilog='')
    parser.add_argument("input_fine_total_cumulative_flow_filepath",
                        metavar="Input_Fine_Total_Cumulative_Flow_Filepath",
                        type=str,
                        help="Filepath to input fine accumulated flow "
                             " netCDF file")
    parser.add_argument("input_fine_rdirs_filepath",
                        metavar="Input_Fine_River_Directions_Filepath",
                        type=str,
                        help="Filepath to input fine river directions"
                             " netCDF file")
    parser.add_argument("input_fine_cell_numbers_filepath",
                        metavar="Input_Fine_Cell_Numbers_Filepath",
                        type=str,
                        help="Filepath to input fine lat-lon to coarse"
                             " icosahedral grid mapping netCDF file")
    parser.add_argument("coarse_grid_params_filepath",
                        metavar="Coarse_Grid_Parameters_Filepath",
                        type=str,
                        help="Filepath of the input coarse atmospheric"
                             " grid parameters netCDF file")
    parser.add_argument("output_rdirs_filepath",
                        metavar="Output_River_Directions_Filepath",
                        type=str,
                        help="Target filepath for output coarse"
                             " river directions")
    parser.add_argument("input_coarse_catchments_filepath",
                        metavar="Input_Coarse_Catchments_Filepath",
                        type=str,
                        help="Filepath to input coarse catchments"
                             " netCDF file")
    parser.add_argument("input_coarse_cumulative_flow_filepath",
                        metavar="Input_Coarse_Cumulative_Flow_Filepath",
                        type=str,
                        help="Filepath to input coarse accumulated flow"
                             " netCDF file")
    parser.add_argument("input_coarse_rdirs_filepath",
                        metavar="Input_Coarse_River_Directions_Filepath",
                        type=str,
                        help="Filepath to input coarse river directions"
                             " netCDF file")
    parser.add_argument("input_fine_rdirs_fieldname",
                        metavar="Input_Fine_River_Directions_Fieldname",
                        type=str,
                        help="Fieldname of the fine river directions"
                             " field")
    parser.add_argument("input_fine_total_cumulative_flow_fieldname",
                        metavar="Input_Fine_Total_Cumulative_Flow_Fieldname",
                        type=str,
                        help="Fieldname of the input fine accumulated flow"
                             " field")
    parser.add_argument("input_fine_cell_numbers_fieldname",
                        metavar="Input_Fine_Cell_Numbers_Fieldname",
                        type=str,
                        help="Fieldname of the input fine lat-lon to coarse"
                             " icosahedral grid mapping field")
    parser.add_argument("input_coarse_catchments_fieldname",
                        metavar="Coarse_Catchment_Fieldname",
                        type=str,
                        help="Fieldname of the input coarse"
                             " accumulated flow field")
    parser.add_argument("input_coarse_cumulative_flow_fieldname",
                        metavar="Coarse_Cumulative_Flow_Fieldname",
                        type=str,
                        help="Fieldname of the input coarse"
                             "accumulated flow field")
    parser.add_argument("input_coarse_rdirs_fieldname",
                        metavar="Coarse_River_Directions_Fieldname",
                        type=str,
                        help="Fieldname of the")
    parser.add_argument("input_loop_log_filepath",
                        metavar="Input_Loop_Log_Filepath",
                        type=str,
                        help="Filepath of the input loop log file listing"
                             " the loops to be removed")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_latlon_to_icon_loop_breaker_driver(args):
    driver = LatLonToIconLoopBreakerDriver(vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_latlon_to_icon_loop_breaker_driver(args)
