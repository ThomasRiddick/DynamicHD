import xarray as xr
from Dynamic_HD_Scripts.utilities.check_driver_inputs \
    import check_input_files, check_output_files

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
        input_fine_rdirs_in_ds = open_dataset(self.args["input_fine_rdirs_filepath"])
        input_fine_rdirs_in = \
            input_fine_rdirs_in_ds[self.args["input_fine_rdirs_fieldname"]].values
        input_fine_total_cumulative_flow_in_ds = \
            open_dataset(self.args["input_fine_total_cumulative_flow_filepath"])
        input_fine_total_cumulative_flow_in = \
            input_fine_total_cumulative_flow_in_ds[
            self.args["input_fine_total_cumulative_flow_fieldname"]].values
        input_fine_cell_numbers_in_ds = \
            open_dataset(self.args["input_fine_cell_numbers_filepath"])
        input_fine_cell_numbers_in = \
            input_fine_cell_numbers_in_ds[
            self.args["input_fine_cell_numbers_fieldname"]].values
        coarse_grid_params_ds = open_dataset(self.args["coarse_grid_params_filepath"])
        neighboring_cell_indices_in = grid_params_ds["neighbor_cell_index"]
        input_coarse_catchments_in_ds = \
            open_dataset(self.args["input_coarse_catchments_filepath"])
        input_coarse_catchments_in = \
            input_coarse_catchments_in_ds[
            self.args["input_coarse_catchments_fieldname"]].values
        input_coarse_cumulative_flow_in_ds = \
            open_dataset(self.args["input_coarse_cumulative_flow_filepath"])
        input_coarse_cumulative_flow_in = \
            input_coarse_cumulative_flow_in_ds[
            self.args["input_coarse_cumulative_flow_fieldname"]].values
        input_coarse_rdirs_in_ds = \
            open_dataset(self.args["input_coarse_rdirs_filepath"])
        input_coarse_rdirs_in = input_coarse_rdirs_in_ds[
            self.args["input_coarse_rdirs_fieldname"]].values
        loop_nums_list = []
        with open(input_loop_log_filepath,"r") as f:
            first_line=True
            for line in f:
                if first_line:
                    first_line=False
                    continue
                loop_nums_list.append(int(line.strip()))
        # do j = 1,3
        #   do i=1,ncells_coarse
        #     cell_vertices_lats(i,j) = cell_vertices_lats_raw(j,i)*(180.0/PI)
        #     if(cell_vertices_lats(i,j) > 90.0 - ABS_TOL_FOR_DEG) cell_vertices_lats(i,j) = 90.0
        #     if(cell_vertices_lats(i,j) < -90.0 + ABS_TOL_FOR_DEG) cell_vertices_lats(i,j) = -90.0
        #     cell_vertices_lons(i,j) = cell_vertices_lons_raw(j,i)*(180.0/PI)
        #   end do
        # end do

        # call break_loops_icon_icosohedral_cell_latlon_pixel(coarse_rdirs, &
        #                                                     coarse_cumulative_flow, &
        #                                                     coarse_catchments, &
        #                                                     input_fine_rdirs_int, &
        #                                                     input_fine_total_cumulative_flow, &
        #                                                     loop_nums_list, &
        #                                                     cell_neighbors, &
        #                                                     cell_vertices_lats, &
        #                                                     cell_vertices_lons, &
        #                                                     pixel_center_lats, &
        #                                                     pixel_center_lons, &
        #                                                     cell_numbers_data)
        number_of_outflows_out_ds = next_cell_index_in_ds.Copy(deep=True,
                                                               data={"num_outflows":
                                                                     number_of_outflows_out})
        number_of_outflows_out_ds.to_netcdf(self.args["output_number_of_outflows_filepath"])

class Arguments:
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser(prog='Icosahedral Grid River Bifurcation Tool',
                                     description='Add bifurcated river mouths to existing'
                                                 'river directions on icosahedral grids ',
                                     epilog='')
    parser.add_argument("input_fine_total_cumulative_flow_filepath",
                        metavar="Input Fine Total Cumulative Flow Filepath",
                        type=str,
                        help="Filepath to input fine accumulated flow "
                             " netCDF file")
    parser.add_argument("input_fine_rdirs_filepath",
                        metavar="Input Fine River Directions Filepath",
                        type=str,
                        help="Filepath to input fine river directions"
                             " netCDF file")
    parser.add_argument("input_fine_cell_numbers_filepath",
                        metavar="Input Fine Cell Numbers Filepath",
                        type=str,
                        help="Filepath to input fine lat-lon to coarse"
                             " icosahedral grid mapping netCDF file")
    parser.add_argument("coarse_grid_params_filepath",
                        metavar="Coarse Grid Parameters Filepath",
                        type=str,
                        help="Filepath of the input coarse atmospheric"
                             " grid parameters netCDF file")
    parser.add_argument("output_rdirs_filepath",
                        metavar="Output River Directions Filepath",
                        type=str,
                        help="Target filepath for output coarse"
                             " river directions")
    parser.add_argument("input_coarse_catchments_filepath",
                        metavar="Input Coarse Catchments Filepath",
                        type=str,
                        help="Filepath to input coarse catchments"
                             " netCDF file")
    parser.add_argument("input_coarse_cumulative_flow_filepath",
                        metavar="Input Coarse Cumulative Flow Filepath",
                        type=str,
                        help="Filepath to input coarse accumulated flow"
                             " netCDF file")
    parser.add_argument("input_coarse_rdirs_filepath",
                        metavar="Input Coarse River Directions Filepath",
                        type=str,
                        help="Filepath to input coarse river directions"
                             " netCDF file")
    parser.add_argument("input_fine_rdirs_fieldname",
                        metavar="Input Fine River Directions Fieldname",
                        type=str,
                        help="Fieldname of the fine river directions"
                             " field")
    parser.add_argument("input_fine_total_cumulative_flow_fieldname",
                        metavar="Input Fine Total Cumulative Flow Fieldname",
                        type=str,
                        help="Fieldname of the input fine accumulated flow"
                             " field")
    parser.add_argument("input_fine_cell_numbers_fieldname",
                        metavar="Input Fine Cell Numbers Fieldname",
                        type=str,
                        help="Fieldname of the input fine lat-lon to coarse"
                             " icosahedral grid mapping field")
    parser.add_argument("input_coarse_catchments_fieldname",
                        metavar="Coarse Catchment Fieldname",
                        type=str,
                        help="Fieldname of the input coarse"
                             " accumulated flow field")
    parser.add_argument("input_coarse_cumulative_fieldname",
                        metavar="Coarse Cumulative Fieldname",
                        type=str,
                        help="Fieldname of the input coarse"
                             "accumulated flow field")
    parser.add_argument("input_coarse_rdirs_fieldname",
                        metavar="Coarse River Directions Fieldname",
                        type=str,
                        help="Fieldname of the")
    parser.add_argument("input_loop_log_filename",
                        metavar="Input Loop Log Filepath",
                        type=str,
                        help="Filepath of the input loop log file listing"
                             " the loops to be removed")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_latlon_to_icon_loop_breaker_driver(args):
    driver = LatLonToIconLoopBreakerDriver(**vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_latlon_to_icon_loop_breaker_driver(args)
