import xarray as xr
from Dynamic_HD_Scripts.utilities.check_driver_inputs \
    import check_input_files, check_output_files

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
        fine_rdirs_in_ds = open_dataset(self.args["input_fine_rdirs_filepath"])
        fine_rdirs_in = \
            fine_rdirs_in_ds[self.args["input_fine_rdirs_fieldname"]].values
        fine_total_cumulative_flow_ds = open_dataset(self.args["input_fine_total_cumulative_flow_filepath"])
        fine_total_cumulative_flow_in = \
            fine_total_cumulative_flow_ds[self.args["input_fine_total_cumulative_flow_fieldname"]].values
        coarse_grid_params_ds = open_dataset(self.args["coarse_grid_params_filepath"])
        neighboring_cell_indices_in = grid_params_ds["neighbor_cell_index"]
        coarse_next_cell_index_out = np.zeros((????(VAR FROM COARSE GRID PARAMS).shape[0]),
                                              dtype=np.int64)
        # if(write_cell_numbers) then
        #   call cotat_plus_icon_icosohedral_cell_latlon_pixel(nint(input_fine_rdirs),&
        #                                                      input_fine_total_cumulative_flow,&
        #                                                      output_coarse_next_cell_index,&
        #                                                      pixel_center_lats,&
        #                                                      pixel_center_lons,&
        #                                                      cell_vertices_lats, &
        #                                                      cell_vertices_lons, &
        #                                                      cell_neighbors, .true., &
        #                                                      cotat_parameters_filepath,&
        #                                                      output_cell_numbers)
        # else
        #   call cotat_plus_icon_icosohedral_cell_latlon_pixel(nint(input_fine_rdirs),&
        #                                                      input_fine_total_cumulative_flow,&
        #                                                      output_coarse_next_cell_index,&
        #                                                      pixel_center_lats,&
        #                                                      pixel_center_lons,&
        #                                                      cell_vertices_lats, &
        #                                                      cell_vertices_lons, &
        #                                                      cell_neighbors, .true., &
        #                                                      cotat_parameters_filepath)
        # end if
        #         if (num_args == 10) then
        #   call get_command_argument(9,value=output_cell_numbers_filename)
        #   write_cell_numbers = .true.
        # else
        #   write_cell_numbers = .false.
        # end if

        # allocate(cell_vertices_lats(ncells_coarse,3))
        # allocate(cell_vertices_lons(ncells_coarse,3))
        # do j = 1,3
        #   do i=1,ncells_coarse
        #     cell_vertices_lats(i,j) = cell_vertices_lats_raw(j,i)*(180.0/PI)
        #     if(cell_vertices_lats(i,j) > 90.0 - ABS_TOL_FOR_DEG) cell_vertices_lats(i,j) = 90.0
        #     if(cell_vertices_lats(i,j) < -90.0 + ABS_TOL_FOR_DEG) cell_vertices_lats(i,j) = -90.0
        #     cell_vertices_lons(i,j) = cell_vertices_lons_raw(j,i)*(180.0/PI)
        #   end do
        # end do
        # allocate(output_coarse_next_cell_index(ncells_coarse))
        # if (write_cell_numbers) then
        #   write(*,*) "Writing Cell Numbers File"
        #   allocate(output_cell_numbers_processed(nlon,nlat))

        # end if
        next_cell_index_out_ds = coarse_grid_params_ds.Copy(deep=True,
                                                            data={"FDIR":
                                                                  next_cell_index_out})
        next_cell_index_out_ds.to_netcdf(self.args["output_rdirs_filepath"])

class Arguments:
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser(prog=' Cotat Plus LatLon to ICON Upscaling Tool',
                                     description='',
                                     epilog='')
    parser.add_argument("input_fine_rdirs_filepath",
                        metavar="Input Fine River Direction Filepath",
                        type=str,
                        help="Filepath to input fine river directions netCDF file")
    parser.add_argument("input_fine_total_cumulative_flow_filepath",
                        metavar="Input File Total Cumulative Flow Filepath",
                        type=str,
                        help="Filepath to input fine total cumulative flow netCDF file")
    parser.add_argument("coarse_grid_params_filepath",
                        metavar="Coarse Grid Parameter Filepath",
                        type=str,
                        help="Filepath of the input coarse atmospheric grid parameters"
                             " netCDF file")
    parser.add_argument("output_rdirs_filepath",
                        metavar="Output River Directions Filepath",
                        type=str,
                        help="Target filepath for output river directions "
                             "netCDF file")
    parser.add_argument("input_fine_rdirs_fieldname",
                        metavar="Input Fine River Directions Fieldname",
                        type=str,
                        help="Fieldname of input river directions field")
    parser.add_argument("input_fine_total_cumulative_flow_fieldname",
                        metavar="Input Fine Total Cumulative Flow Fieldname",
                        type=str,
                        help="Fieldname of input total cumulative flow field")
    parser.add_argument("cotat_parameters_filepath",
                        metavar="COTAT Parameters Filepath",
                        type=str,
                        help="Filepath of COTAT Plus parameters file to use")
    parser.add_argument("output_cell_numbers_filepath",
                        metavar="Output Cell Numbers Filepath",
                        default=None,
                        type=str,
                        help="Target output filepath for cell numbers"
                             " netCDF file (optional)")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_cotat_plus_latlon_to_icon_driver(args):
    driver = CotatPlusLatLonToIconDriver(**vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_cotat_plus_latlon_to_icon_driver(args)
