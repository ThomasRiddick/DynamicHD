import numpy as np
import netCDF4
import cdo
import os
from os import path
from Dynamic_HD_Scripts.base.field import Field
from Dynamic_HD_Scripts.base import grid
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.tools.determine_river_directions import determine_river_directions
from Dynamic_HD_Scripts.tools.compute_catchments import compute_catchments_cpp
from Dynamic_HD_Scripts.utilities.basin_evaluation_algorithm_prototype import LatLonEvaluateBasin
import evaluate_basins_wrapper

class L2LakeModelTestDataGenerator:

    def generate_test_data(self):
      working_directory_path = "/Users/thomasriddick/Documents/data/temp"
      output_lakeparas_filename = path.join(working_directory_path,"l2_lake_model_para.nc")
      output_lakeparas_with_mask_filename = path.join(working_directory_path,"l2_lake_model_para_with_mask.nc")
      mask_filename = path.join(working_directory_path,"mask_temp.nc")
      array_filename =  path.join(working_directory_path,"array_temp.nc")
      fields_filename =  path.join(working_directory_path,"fields_temp.nc")
      coarse_catchment_nums_in = np.array([[5,5,5,5],
                                           [2,2,5,5],
                                           [5,4,8,8],
                                           [8,7,4,0]],
                                           dtype=np.int32)
      corrected_orography_in = np.array([
        [9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0],
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
        [9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0]],
        dtype=np.float64)
      raw_orography_in = np.array([
        [9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0],
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
        [9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0]],
        dtype=np.float64)
      minima_in = np.array([[False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False,  True, False,  False, False,  True, False, False,
                            False, False, False, False, False,   False,  True, False, False, False],
                            [False,  True, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False,  True, False, False, False,  False, True,  False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False,  True, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False],
                            [False, False, False, False, False,  False, False, False, False, False,
                             False, False, False, False, False,  False, False, False, False, False]],
                            dtype=np.int32)
      prior_fine_rdirs_in = np.array([
        [1., 4., 3., 2., 1., 1., 3., 2., 1., 1., 2., 1., 1., 1., 1., 3., 2., 1., 1., 1.],
        [3., 2., 1., 5., 4., 4., 6., 5., 4., 4., 1., 4., 4., 4., 4., 6., 5., 4., 4., 4.],
        [6., 5., 4., 8., 7., 7., 9., 8., 7., 7., 4., 7., 7., 7., 7., 9., 8., 7., 7., 7.],
        [9., 8., 7., 8., 7., 7., 9., 8., 7., 7., 7., 7., 7., 7., 7., 9., 8., 7., 7., 7.],
        [9., 8., 7., 8., 7., 7., 9., 8., 7., 7., 7., 7., 7., 7., 7., 9., 8., 7., 7., 7.],
        [9., 8., 7., 2., 1., 4., 4., 4., 4., 4., 9., 8., 7., 7., 7., 9., 8., 7., 7., 7.],
        [9., 8., 7., 1., 4., 7., 7., 7., 7., 7., 9., 8., 7., 7., 7., 6., 5., 4., 4., 4.],
        [9., 8., 7., 4., 7., 7., 7., 7., 7., 7., 6., 5., 4., 4., 4., 9., 8., 7., 7., 7.],
        [9., 8., 7., 7., 7., 7., 7., 7., 7., 7., 9., 8., 7., 7., 7., 9., 8., 7., 7., 7.],
        [9., 8., 7., 8., 7., 7., 7., 7., 7., 7., 9., 8., 7., 7., 7., 9., 8., 7., 7., 7.],
        [9., 8., 7., 8., 7., 7., 7., 7., 7., 7., 9., 8., 7., 7., 7., 9., 8., 7., 7., 7.],
        [9., 8., 7., 8., 7., 7., 7., 7., 7., 7., 9., 8., 7., 7., 7., 9., 8., 7., 7., 7.],
        [9., 8., 7., 8., 7., 7., 7., 7., 7., 7., 9., 8., 7., 7., 7., 9., 8., 7., 7., 7.],
        [9., 8., 7., 1., 1., 1., 1., 1., 1., 1., 9., 8., 7., 7., 7., 9., 8., 7., 7., 7.],
        [1., 5., 4., 4., 4., 4., 4., 4., 4., 4., 4., 1., 1., 1., 3., 2., 1., 1., 1., 1.],
        [4., 8., 7., 7., 7., 7., 7., 7., 7., 7., 7., 4., 4., 4., 6., 3., 2., 1., 1., 1.],
        [7., 8., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 9., 6., 3., 2., 1., 1.],
        [7., 8., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 9., 9., 6., 3., 2., 1.],
        [1., 8., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 9., 9., 9., 6., 3., 2.],
        [4., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 9., 9., 9., 9., 9., 9., 6., 0.]],
        dtype=np.float64)
      prior_coarse_rdirs_in = np.array([[5,5,5,5],
                                        [2,2,5,5],
                                        [5,4,8,8],
                                        [8,7,4,0]],
                                        dtype=np.float64)
      prior_fine_catchments_in = np.array([
        [3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [4, 4, 4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [4, 4, 4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [4, 4, 4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [4, 4, 4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5],
        [4, 4, 4, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5],
        [8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8],
        [8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8],
        [8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8],
        [8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8],
        [8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8],
        [8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8]],
        dtype=np.int32)
      cell_areas_in = np.array([[1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1],
                                [1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1]],
                                dtype=np.float64)
      landsea_in = np.array([
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,False],
        [False,False,False,False,False, False,False,False,False,False, False,False,False,False,False, False,False,False,False,True]],
        dtype=np.int32)
      output = \
        LatLonEvaluateBasin.evaluate_basins(landsea_in,
                                            minima_in,
                                            raw_orography_in,
                                            corrected_orography_in,
                                            cell_areas_in,
                                            prior_fine_rdirs_in,
                                            prior_fine_catchments_in,
                                            coarse_catchment_nums_in,
                                            return_algorithm_object=False)
      corresponding_surface_cell_lat_index = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                                                       [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                                                       [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                                                       [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                                                       [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                                                       [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                                                       [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                                                       [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                                                       [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                                                       [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                                                       [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                                                       [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                                                       [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                                                       [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                                                       [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                                                       [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                                                       [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                                                       [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                                                       [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                                                       [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]],
                                                       dtype=np.int32)
      corresponding_surface_cell_lon_index = np.array([[1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3],
                                                       [1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3]],
                                                       dtype=np.int32)
      binary_mask = np.array([[1,0,1],
                              [0,1,0],
                              [0,0,1]],dtype=np.int32)
      lake_grid=grid.LatLongGrid(nlat=20,nlong=20)
      fields_to_write = [Field(output["lake_mask"],grid=lake_grid),
                         Field(corresponding_surface_cell_lat_index,grid=lake_grid),
                         Field(corresponding_surface_cell_lon_index,grid=lake_grid)]
      fieldnames_for_fields_to_write = ["lake_mask",
                                        "corresponding_surface_cell_lat_index",
                                        "corresponding_surface_cell_lon_index", ]
      iodriver.advanced_field_writer(fields_filename,
                                     fields_to_write,
                                     fieldname=fieldnames_for_fields_to_write)
      with netCDF4.Dataset(array_filename,mode='w',format='NETCDF4') as dataset:
          dataset.createDimension("npoints",len(output["lakes_as_array"]))
          dataset.createDimension("scalar",1)
          field_values = dataset.createVariable("lakes_as_array",np.float64,
                                                ('npoints'))
          field_values[:] = output["lakes_as_array"]
          nlakes = dataset.createVariable("number_of_lakes",np.int64,
                                          ('scalar'))
          nlakes[0] =  output["number_of_lakes"]
      cdo_inst = cdo.Cdo()
      cdo_inst.merge(input=" ".join([array_filename,fields_filename]),
                    output=output_lakeparas_filename)
      surface_model_grid=grid.LatLongGrid(nlat=3,nlong=3)
      iodriver.advanced_field_writer(mask_filename,
                                     Field(binary_mask,grid=surface_model_grid),
                                     fieldname="binary_lake_mask")
      cdo_inst.merge(input=" ".join([output_lakeparas_filename,mask_filename]),
                    output=output_lakeparas_with_mask_filename)
      for file_to_remove in [fields_filename,array_filename,mask_filename]:
          os.remove(file_to_remove)

def main():
    generator = L2LakeModelTestDataGenerator()
    generator.generate_test_data()

if __name__ == '__main__':
    main()
