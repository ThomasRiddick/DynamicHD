'''
A program to generate small sets of lake model test data from an input orography and set of river directions and catchments

Created on Jan 9, 2022

@author: thomasriddick
'''
import numpy as np
from Dynamic_HD_Scripts.base import field
from Dynamic_HD_Scripts.base import grid
from Dynamic_HD_Scripts.tools.determine_river_directions import determine_river_directions
from Dynamic_HD_Scripts.tools.compute_catchments import compute_catchments_cpp
from Dynamic_HD_Scripts.interface.cpp_interface.libs \
    import evaluate_basins_wrapper

class LakeModelTestDataGenerator(object):

    def print_field_for_julia(self,field_in,name_in,type_in,grid_in):
        data_txt = ""
        raw_data = field_in.get_data()
        for row in raw_data:
            data_txt += " ".join(str(element) for element in row) + "\n"
        print("{name}::Field{{{type}}} ="
              "LatLonField{{{type}}}({grid},{type}[{data}]))\n".format(type=type_in,
                                                                       grid=grid_in,
                                                                       name=name_in,
                                                                       data=data_txt.rstrip("\n")))

    def generate_test_data(self):
        coarse_grid = grid.makeGrid(grid_type="LatLong",nlat=4,nlong=4)
        fine_grid = grid.makeGrid(grid_type="LatLong",nlat=20,nlong=20)
        fine_shape = (20,20)
        input_minima    = field.Field(np.array(
            [[False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,
              False,False,False,True,False, False,False, True,False,False, False,False,False,False,False,
              False, True,False,False,False,
              False, True,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,
              False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,
              False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,

              False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,
              False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,
              False,False,False,False,False, False,False,False,False,False, False, True,False,False,False,
              False, True,False,False,False,
              False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,
              False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,

              False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,
              False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,
              False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,
              False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,
              False, True,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,

              False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,
              False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,
              False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,
              False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False,
              False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False]]),fine_grid)
        input_raw_orography = field.Field(np.array(
            [[9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0],
             [9.0,9.0,9.0,3.0,3.0, 3.0,9.0,1.0,1.0,9.0, 1.0,1.0,1.0,1.0,9.0, 9.0,2.0,2.0,2.0,9.0],
             [9.0,1.0,9.0,3.0,3.0, 3.0,6.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,7.0, 7.0,2.0,2.0,2.0,9.0],
             [9.0,1.0,9.0,3.0,3.0, 3.0,9.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,9.0, 9.0,2.0,2.0,2.0,9.0],
             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,1.0,1.0,1.0,9.0, 9.0,2.0,2.0,2.0,9.0],

             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,1.0,1.0,1.0,9.0, 9.0,9.0,6.0,9.0,9.0],
             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,5.0,9.0,9.0, 9.0,3.0,3.0,3.0,9.0],
             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0],
             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0],
             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0],

             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0],
             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0],
             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0],
             [9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,8.0,9.0],
             [9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 9.0,9.0,9.0,9.0,7.0, 7.0,7.0,7.0,7.0,7.0],

             [9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0],
             [9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0],
             [9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,8.0,7.0, 0.0,0.0,0.0,0.0,0.0],
             [9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0],
             [9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0]]),fine_grid)
        input_corrected_orography = field.Field(np.array(
            [[9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0],
             [9.0,9.0,9.0,3.0,3.0, 3.0,9.0,1.0,1.0,9.0, 1.0,1.0,1.0,1.0,9.0, 9.0,2.0,2.0,2.0,9.0],
             [9.0,1.0,9.0,3.0,3.0, 3.0,6.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,7.0, 7.0,2.0,2.0,2.0,9.0],
             [9.0,1.0,9.0,3.0,3.0, 3.0,9.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,9.0, 9.0,2.0,2.0,2.0,9.0],
             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,1.0,1.0,1.0,9.0, 9.0,2.0,2.0,2.0,9.0],

             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,1.0,1.0,1.0,9.0, 9.0,9.0,6.0,9.0,9.0],
             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,5.0,9.0,9.0, 9.0,3.0,3.0,3.0,9.0],
             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0],
             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0],
             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0],

             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0],
             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0],
             [9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0],
             [9.0,7.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,8.0,9.0],
             [9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 9.0,9.0,9.0,9.0,7.0, 7.0,7.0,7.0,7.0,7.0],

             [9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0],
             [9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0],
             [9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,8.0,7.0, 0.0,0.0,0.0,0.0,0.0],
             [9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0],
             [9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0]]),fine_grid)
        input_cell_areas = field.Field(np.ones(fine_shape,dtype=np.float64,order='C'),fine_grid)
        input_lsmask    = field.Field(np.array(
            [[False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
              [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False]]),fine_grid)
        input_truesinks = field.Field(np.array(
            [[False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
              [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False],
             [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False,
              False,False,False,False,False]]),fine_grid)
        input_prior_fine_rdirs = determine_river_directions(input_corrected_orography,
                                                            input_lsmask,
                                                            input_truesinks,
                                                            always_flow_to_sea=True,
                                                            use_diagonal_nbrs=True,
                                                            mark_pits_as_true_sinks=True)
        input_prior_fine_catchments = \
            field.Field(compute_catchments_cpp(input_prior_fine_rdirs.get_data(),
                                               "/Users/thomasriddick/Documents/"
                                               "data/temp/loop_from_test_data_gen.log"),
                        fine_grid)
        input_coarse_catchment_nums = field.Field(np.array([[4,1,2,3],
                                                            [4,4,6,5],
                                                            [4,4,6,5],
                                                            [7,7,7,8]]),coarse_grid)
        input_coarse_rdirs = field.Field(np.array([[5,5,5,5],
                                                   [2,2,5,5],
                                                   [5,4,8,8],
                                                   [8,7,4,0]]),coarse_grid)
        connection_volume_thresholds = field.Field(np.zeros(fine_shape,dtype=np.float64,order='C'),fine_grid)
        flood_volume_thresholds = field.Field(np.zeros(fine_shape,dtype=np.float64,order='C'),fine_grid)
        flood_next_cell_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        flood_next_cell_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        connect_next_cell_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        connect_next_cell_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        flood_force_merge_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        flood_force_merge_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        connect_force_merge_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        connect_force_merge_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        flood_redirect_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        flood_redirect_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        connect_redirect_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        connect_redirect_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        additional_flood_redirect_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        additional_flood_redirect_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        additional_connect_redirect_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        additional_connect_redirect_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        flood_local_redirect = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        connect_local_redirect = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        additional_flood_local_redirect = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        additional_connect_local_redirect = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        merge_points = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        basin_catchment_numbers = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        evaluate_basins_wrapper.evaluate_basins(minima_in_int=
                                                np.ascontiguousarray(input_minima.get_data(),dtype=np.int32),
                                                raw_orography_in=
                                                np.ascontiguousarray(input_raw_orography.get_data(),
                                                                     dtype=np.float64),
                                                corrected_orography_in=
                                                np.ascontiguousarray(input_corrected_orography.get_data(),
                                                                     dtype=np.float64),
                                                cell_areas_in=
                                                np.ascontiguousarray(input_cell_areas.get_data(),
                                                                     dtype=np.float64),
                                                connection_volume_thresholds_in=
                                                connection_volume_thresholds.get_data(),
                                                flood_volume_thresholds_in=
                                                flood_volume_thresholds.get_data(),
                                                prior_fine_rdirs_in=
                                                np.ascontiguousarray(input_prior_fine_rdirs.get_data(),
                                                                     dtype=np.float64),
                                                prior_coarse_rdirs_in=
                                                np.ascontiguousarray(input_coarse_rdirs.get_data(),
                                                                     dtype=np.float64),
                                                prior_fine_catchments_in=
                                                np.ascontiguousarray(input_prior_fine_catchments.get_data(),
                                                                     dtype=np.int32),
                                                coarse_catchment_nums_in=
                                                np.ascontiguousarray(input_coarse_catchment_nums.get_data(),
                                                                     dtype=np.int32),
                                                flood_next_cell_lat_index_in=
                                                flood_next_cell_lat_index.get_data(),
                                                flood_next_cell_lon_index_in=
                                                flood_next_cell_lon_index.get_data(),
                                                connect_next_cell_lat_index_in=
                                                connect_next_cell_lat_index.get_data(),
                                                connect_next_cell_lon_index_in=
                                                connect_next_cell_lon_index.get_data(),
                                                flood_force_merge_lat_index_in=
                                                flood_force_merge_lat_index.get_data(),
                                                flood_force_merge_lon_index_in=
                                                flood_force_merge_lon_index.get_data(),
                                                connect_force_merge_lat_index_in=
                                                connect_force_merge_lat_index.get_data(),
                                                connect_force_merge_lon_index_in=
                                                connect_force_merge_lon_index.get_data(),
                                                flood_redirect_lat_index_in=
                                                flood_redirect_lat_index.get_data(),
                                                flood_redirect_lon_index_in=
                                                flood_redirect_lon_index.get_data(),
                                                connect_redirect_lat_index_in=
                                                connect_redirect_lat_index.get_data(),
                                                connect_redirect_lon_index_in=
                                                connect_redirect_lon_index.get_data(),
                                                additional_flood_redirect_lat_index_in=
                                                additional_flood_redirect_lat_index.get_data(),
                                                additional_flood_redirect_lon_index_in=
                                                additional_flood_redirect_lon_index.get_data(),
                                                additional_connect_redirect_lat_index_in=
                                                additional_connect_redirect_lat_index.get_data(),
                                                additional_connect_redirect_lon_index_in=
                                                additional_connect_redirect_lon_index.get_data(),
                                                flood_local_redirect_out_int=
                                                flood_local_redirect.get_data(),
                                                connect_local_redirect_out_int=
                                                connect_local_redirect.get_data(),
                                                additional_flood_local_redirect_out_int=
                                                additional_flood_local_redirect.get_data(),
                                                additional_connect_local_redirect_out_int=
                                                additional_connect_local_redirect.get_data(),
                                                merge_points_out_int=
                                                merge_points.get_data(),
                                                basin_catchment_numbers_in=
                                                basin_catchment_numbers.get_data())
        self.print_field_for_julia(connection_volume_thresholds,"connection_volume_thresholds",
                                   "Float64",grid_in="lake_grid")
        self.print_field_for_julia(flood_volume_thresholds,"flood_volume_thresholds",
                                   "Float64",grid_in="lake_grid")
        self.print_field_for_julia(flood_next_cell_lat_index,"flood_next_cell_lat_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(flood_next_cell_lon_index,"flood_next_cell_lon_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(connect_next_cell_lat_index,"connect_next_cell_lat_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(connect_next_cell_lon_index,"connect_next_cell_lon_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(flood_force_merge_lat_index,"flood_force_merge_lat_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(flood_force_merge_lon_index,"flood_force_merge_lon_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(connect_force_merge_lat_index,"connect_force_merge_lat_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(connect_force_merge_lon_index,"connect_force_merge_lon_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(flood_redirect_lat_index,"flood_redirect_lat_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(flood_redirect_lon_index,"flood_redirect_lon_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(connect_redirect_lat_index,"connect_redirect_lat_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(connect_redirect_lon_index,"connect_redirect_lon_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(additional_flood_redirect_lat_index,
                                   "additional_flood_redirect_lat_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(additional_flood_redirect_lon_index,
                                   "additional_flood_redirect_lon_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(additional_connect_redirect_lat_index,
                                   "additional_connect_redirect_lat_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(additional_connect_redirect_lon_index,
                                   "additional_connect_redirect_lon_index",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(flood_local_redirect,"flood_local_redirect",
                                   "Bool",grid_in="lake_grid")
        self.print_field_for_julia(connect_local_redirect,"connect_local_redirect",
                                   "Bool",grid_in="lake_grid")
        self.print_field_for_julia(additional_flood_local_redirect,
                                   "additional_flood_local_redirect",
                                   "Bool",grid_in="lake_grid")
        self.print_field_for_julia(additional_connect_local_redirect,
                                   "additional_connect_local_redirect",
                                   "Bool",grid_in="lake_grid")
        self.print_field_for_julia(merge_points,"merge_points",
                                   "Int64",grid_in="lake_grid")
        self.print_field_for_julia(input_minima,"input_minima",
                                   "Bool",grid_in="lake_grid")

def main():
    generator = LakeModelTestDataGenerator()
    generator.generate_test_data()

if __name__ == '__main__':
    main()
