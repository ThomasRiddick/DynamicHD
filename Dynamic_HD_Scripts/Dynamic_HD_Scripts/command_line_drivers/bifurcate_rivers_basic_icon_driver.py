import xarray as xr

class BifurcateRiversBasicIconDriver:

    def __init__(self,args):
        self.args = args

    def run(self):
        next_cell_index_in_ds = open_dataset(self.args["next_cell_index_filepath"])
        next_cell_index_in = \
            next_cell_index_in_ds[self.args["next_cell_index_fieldname"]].values
        cumulative_flow_in_ds = open_dataset(self.args["cumulative_flow_filepath"])
        cumulative_flow_in = \
            cumulative_flow_in_ds[self.args["cumulative_flow_fieldname"]].values
        landsea_mask_in_int_ds = open_dataset(self.args["landsea_mask_filepath"])
        landsea_mask_in_int = \
            landsea_mask_in_int_ds[self.args["landsea_mask_fieldname"]].values
        print("ICON River Bifurcation Tool")
        grid_params_filepath
        print("Using minimum_cells_from_split_to_main_mouth= {}".
              format(self.args["minimum_cells_from_split_to_main_mouth"]))
        print("Using maximum_cells_from_split_to_main_mouth= {}".
              format(self.args["maximum_cells_from_split_to_main_mouth"]))
        print("Using cumulative_flow_threshold_fraction= {}".
              format(self.args["cumulative_flow_threshold_fraction"]))
        bifurcate_rivers_basic_icon_cpp(neighboring_cell_indices_in,
                                        next_cell_index_in,
                                        cumulative_flow_in,
                                        landsea_mask_in_int,
                                        number_of_outflows_out,
                                        bifurcations_next_cell_index_out,
                                        self.args["mouth_positions_filepath"],
                                        self.args["minimum_cells_from_split_to_main_mouth"],
                                        self.args["maximum_cells_from_split_to_main_mouth"],
                                        self.args["cumulative_flow_threshold_fraction"])
        number_of_outflows_out_ds = next_cell_index_in_ds.Copy(deep=True,
                                                               data={"num_outflows":
                                                                     number_of_outflows_out})
        number_of_outflows_out_ds.to_netcdf(output_number_of_outflows_filepath)
        next_cell_index_out_ds = next_cell_index_in_ds.Copy(deep=True,
                                                            data={"next_cell_index":
                                                                  next_cell_index_in})
        next_cell_index_out_ds.to_netcdf(output_next_cell_index_filepath)
        bifurcations_next_cell_index_out_ds = \
            next_cell_index_in_ds.Copy(deep=True,
                                       data={"bifurcated_next_cell_index":
                                             bifurcations_next_cell_index_out})
        bifurcations_next_cell_index_out_ds.\
            to_netcdf(output_bifurcated_next_cell_index_filepath)

class Arguments:
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser(prog='Icosahedral Grid River Bifurcation Tool',
                                     description='Add bifurcated river mouths to existing'
                                                 'river directions on icosahedral grids ',
                                     epilog='')
    parser.add_argument(next_cell_index_filepath,metavar='Next Cell Index Filepath',
                        type=str,
                        help="Filepath to input next cell index (river directions) netCDF file")
    parser.add_argument(cumulative_flow_filepath,metavar='Cumulative Flow Filepath',
                        type=str,
                        help="Filepath to input accumulated flow input netCDF file")
    parser.add_argument(landsea_mask_filepath,metavar='Landsea Filepath',
                        type=str,
                        help="Filepath to input binary landsea mask netCDF file")
    parser.add_argument(output_number_of_outflows_filepath,
                        metavar='Output Number of Outflows Filepath',
                        type=str,
                        help="Target filepath for output number of outflows netCDF file")
    parser.add_argument(output_next_cell_index_filepath,
                        metavar='Output Next Cell Index Filepath',
                        type=str,
                        help="Target filepath for output next cell index "
                             "(river directions) netCDF file")
    parser.add_argument(output_bifurcated_next_cell_index_filepath,
                        metavar='Output Bifurcated Next Cell Index Filepath',
                        type=str,
                        help="Target filepath for output bifurcated next cell index"
                             "(river directions) netCDF file")
    parser.add_argument(grid_params_filepath,
                        metavar='Grid Parameters Filepath',
                        type=str,
                        help="Filepath of the input atmospheric grid parameters"
                             " netCDF file")
    parser.add_argument(mouth_positions_filepath,
                        metavar='River Mouth Positions Filepath',
                        type=str,
                        help="Filepath of the input river mouth position file")
    parser.add_argument(next_cell_index_fieldname,
                        metavar='Next Cell Index Fieldname',
                        type=str,
                        help="Fieldname of the input next cell index"
                             "(river directions) input field")
    parser.add_argument(cumulative_flow_fieldname,
                        metavar='Cumulative Flow Fieldname',
                        type=str,
                        help="Fieldname of the accumulated flow input field")
    parser.add_argument(landsea_mask_fieldname,
                        metavar='Landsea Fieldname',
                        type=str,
                        help="Fieldname of the binary landsea mask input field")
    parser.add_argument(minimum_cells_from_split_to_main_mouth,
                        metavar='Minimum cells from split to main mouth',
                        type=int,
                        help="Minimum number of cells to follow a river upstream from the"
                             "primary river mouth before allowing bifurcation")
    parser.add_argument(maximum_cells_from_split_to_main_mouth,
                        metavar='Maximum cells from split to main mouth',
                        type=int,
                        help="Maximum number of cells to follow a river upstream from the"
                             "primary river mouth for which bifurcation is allowed")
    parser.add_argument(cumulative_flow_threshold_fraction,
                        metavar='Cumulative flow threshold fration',
                        type=double,
                        help="Protect any tributary from being broken if their accumulated "
                             "flow as a fraction of the flow of the main channel is above this "
                             "threshold")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_basic_river_bifurcation_icon_driver(args):
    driver = BifurcateRiversBasicIconDriver(**vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_basic_river_bifurcation_icon_driver(args)
