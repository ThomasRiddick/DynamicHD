import xarray as xr
import argparse
import numpy as np
from bifurcate_rivers_basic_icon_wrapper import bifurcate_rivers_basic_icon_cpp
from Dynamic_HD_Scripts.utilities.check_driver_inputs \
    import check_input_files, check_output_files

class BifurcateRiversBasicIconDriver:

    def __init__(self,args):
        self.args = args

    def run(self):
        print("*** ICON River Bifurcation Tool ***")
        print("Settings:")
        for key,value in self.args.items():
            print("{}: {}".format(key,value))
        check_input_files([self.args["next_cell_index_filepath"],
                           self.args["cumulative_flow_filepath"],
                           self.args["landsea_mask_filepath"],
                           self.args["grid_params_filepath"],
                           self.args["mouth_positions_filepath"]])
        check_output_files([self.args["output_number_of_outflows_filepath"],
                            self.args["output_next_cell_index_filepath"]])
        next_cell_index_in_ds = xr.open_dataset(self.args["next_cell_index_filepath"])
        next_cell_index_inout = \
            next_cell_index_in_ds[self.args["next_cell_index_fieldname"]].values
        cumulative_flow_in_ds = xr.open_dataset(self.args["cumulative_flow_filepath"])
        cumulative_flow_in = \
            cumulative_flow_in_ds[self.args["cumulative_flow_fieldname"]].values
        landsea_mask_in_int_ds = xr.open_dataset(self.args["landsea_mask_filepath"])
        landsea_mask_in_int = \
            landsea_mask_in_int_ds[self.args["landsea_mask_fieldname"]].values
        grid_params_ds = xr.open_dataset(self.args["grid_params_filepath"])
        neighboring_cell_indices_in = grid_params_ds["neighbor_cell_index"].values
        number_of_outflows_out = np.zeros(next_cell_index_inout.shape,
                                          dtype=np.int32)
        bifurcations_next_cell_index_out = np.zeros((next_cell_index_inout.shape[0]*11,),
                                                    dtype=np.int32)
        #Reassign variable to be the result of astype so that we retain reference
        #to output
        next_cell_index_inout = next_cell_index_inout.astype(np.int32)
        bifurcate_rivers_basic_icon_cpp(neighboring_cell_indices_in.\
                                        astype(np.int32).swapaxes(0,1).flatten(),
                                        next_cell_index_inout,
                                        cumulative_flow_in.astype(np.int32),
                                        landsea_mask_in_int.astype(np.int32),
                                        number_of_outflows_out,
                                        bifurcations_next_cell_index_out,
                                        self.args["mouth_positions_filepath"],
                                        self.args["minimum_cells_from_split_to_main_mouth"],
                                        self.args["maximum_cells_from_split_to_main_mouth"],
                                        self.args["cumulative_flow_threshold_fraction"],
                                        self.args["remove_main_channel"])
        number_of_outflows_out_ds = \
            xr.Dataset(data_vars={"num_outflows":(["cell"],number_of_outflows_out)},
                       coords={"clat":next_cell_index_in_ds["clat"],
                               "clon":next_cell_index_in_ds["clon"],
                               "clon_bnds":(["cell","nv"],next_cell_index_in_ds["clon_bnds"].values),
                               "clat_bnds":(["cell","nv"],next_cell_index_in_ds["clat_bnds"].values)},
                       attrs={"number_of_grid_used":next_cell_index_in_ds.attrs["number_of_grid_used"],
                              "grid_file_uri":next_cell_index_in_ds.attrs["grid_file_uri"],
                              "uuidOfHGrid":next_cell_index_in_ds.attrs["uuidOfHGrid"]})
        number_of_outflows_out_ds["num_outflows"].attrs["CDI_grid_type"] = "unstructured"
        number_of_outflows_out_ds.to_netcdf(self.args["output_number_of_outflows_filepath"])

        next_cell_index_out_ds = \
            xr.Dataset(data_vars={"next_cell_index":(["cell"],next_cell_index_inout)},
                       coords={"clat":next_cell_index_in_ds["clat"],
                               "clon":next_cell_index_in_ds["clon"],
                               "clon_bnds":(["cell","nv"],next_cell_index_in_ds["clon_bnds"].values),
                               "clat_bnds":(["cell","nv"],next_cell_index_in_ds["clat_bnds"].values)},
                       attrs={"number_of_grid_used":next_cell_index_in_ds.attrs["number_of_grid_used"],
                              "grid_file_uri":next_cell_index_in_ds.attrs["grid_file_uri"],
                              "uuidOfHGrid":next_cell_index_in_ds.attrs["uuidOfHGrid"]})
        next_cell_index_out_ds["next_cell_index"].attrs["CDI_grid_type"] = "unstructured"
        next_cell_index_out_ds.\
            to_netcdf(self.args["output_next_cell_index_filepath"])
        next_cell_index_out_ds.to_netcdf(self.args["output_next_cell_index_filepath"])

        bifurcations_next_cell_index_out_ds =  \
            xr.Dataset(data_vars={"bifurcated_next_cell_index":
                                  (["nlevels","cell"],
                                   bifurcations_next_cell_index_out.reshape(11,next_cell_index_inout.shape[0]))},
                       coords={"clat":next_cell_index_in_ds["clat"],
                               "clon":next_cell_index_in_ds["clon"],
                               "clon_bnds":(["cell","nv"],next_cell_index_in_ds["clon_bnds"].values),
                               "clat_bnds":(["cell","nv"],next_cell_index_in_ds["clat_bnds"].values)},
                       attrs={"number_of_grid_used":next_cell_index_in_ds.attrs["number_of_grid_used"],
                              "grid_file_uri":next_cell_index_in_ds.attrs["grid_file_uri"],
                              "uuidOfHGrid":next_cell_index_in_ds.attrs["uuidOfHGrid"]})
        bifurcations_next_cell_index_out_ds["bifurcated_next_cell_index"].\
            attrs["CDI_grid_type"] = "unstructured"
        bifurcations_next_cell_index_out_ds.\
            to_netcdf(self.args["output_bifurcated_next_cell_index_filepath"])

class Arguments:
    pass

def parse_arguments():

    args = Arguments()
    parser = argparse.ArgumentParser(prog='Icosahedral Grid River Bifurcation Tool',
                                     description='Add bifurcated river mouths to existing'
                                                 'river directions on icosahedral grids ',
                                     epilog='')
    parser.add_argument("next_cell_index_filepath",metavar='Next_Cell_Index_Filepath',
                        type=str,
                        help="Filepath to input next cell index (river directions) netCDF file")
    parser.add_argument("cumulative_flow_filepath",metavar='Cumulative_Flow_Filepath',
                        type=str,
                        help="Filepath to input accumulated flow input netCDF file")
    parser.add_argument("landsea_mask_filepath",metavar='Landsea_Filepath',
                        type=str,
                        help="Filepath to input binary landsea mask netCDF file")
    parser.add_argument("output_number_of_outflows_filepath",
                        metavar='Output_Number_of_Outflows_Filepath',
                        type=str,
                        help="Target filepath for output number of outflows netCDF file")
    parser.add_argument("output_next_cell_index_filepath",
                        metavar='Output_Next_Cell_Index_Filepath',
                        type=str,
                        help="Target filepath for output next cell index "
                             "(river directions) netCDF file")
    parser.add_argument("output_bifurcated_next_cell_index_filepath",
                        metavar='Output_Bifurcated_Next_Cell_Index_Filepath',
                        type=str,
                        help="Target filepath for output bifurcated next cell index"
                             "(river directions) netCDF file")
    parser.add_argument("grid_params_filepath",
                        metavar='Grid_Parameters_Filepath',
                        type=str,
                        help="Filepath of the input atmospheric grid parameters"
                             " netCDF file")
    parser.add_argument("mouth_positions_filepath",
                        metavar='River_Mouth_Positions_Filepath',
                        type=str,
                        help="Filepath of the input river mouth position file")
    parser.add_argument("next_cell_index_fieldname",
                        metavar='Next_Cell_Index_Fieldname',
                        type=str,
                        help="Fieldname of the input next cell index"
                             "(river directions) input field")
    parser.add_argument("cumulative_flow_fieldname",
                        metavar='Cumulative_Flow_Fieldname',
                        type=str,
                        help="Fieldname of the accumulated flow input field")
    parser.add_argument("landsea_mask_fieldname",
                        metavar='Landsea_Fieldname',
                        type=str,
                        help="Fieldname of the binary landsea mask input field")
    parser.add_argument("minimum_cells_from_split_to_main_mouth",
                        metavar='Minimum_Cells_From_Split_To_Main_Mouth',
                        type=int,
                        help="Minimum number of cells to follow a river upstream from the"
                             "primary river mouth before allowing bifurcation")
    parser.add_argument("maximum_cells_from_split_to_main_mouth",
                        metavar='Maximum_Cells_From_Split_To_Main_Mouth',
                        type=int,
                        help="Maximum number of cells to follow a river upstream from the"
                             "primary river mouth for which bifurcation is allowed")
    parser.add_argument("cumulative_flow_threshold_fraction",
                        metavar='Cumulative_Flow_Threshold_Fraction',
                        type=float,
                        help="Protect any tributary from being broken if their accumulated "
                             "flow as a fraction of the flow of the main channel is above this "
                             "threshold")
    parser.add_argument("-r","--remove-main-channel",
                        action="store_true",
                        default=False,
                        help="Remove original main channel when adding bifurcations")
    parser.parse_args(namespace=args)
    return args

def setup_and_run_basic_river_bifurcation_icon_driver(args):
    driver = BifurcateRiversBasicIconDriver(vars(args))
    driver.run()

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_basic_river_bifurcation_icon_driver(args)
