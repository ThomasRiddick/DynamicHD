'''
Unit tests for the compute_catchments module

Created on Oct 31, 2024

@author: thomasriddick
'''

import unittest
import numpy as np
from Dynamic_HD_Scripts.utilities.basin_evaluation_algorithm_prototype \
import LatLonBasinEvaluationAlgorithm,DisjointSetForest
#Note - copied library files from the latest version of master
#hence they don't match this version of the code (but can
#still be used here)
from fill_sinks_wrapper import fill_sinks_cpp_func

class DisjointSetTests(unittest.TestCase):

  def testDisjointSet(self):
    dsets = DisjointSetForest()
    dsets.add_set(1)
    dsets.add_set(2)
    dsets.add_set(5)
    dsets.add_set(6)
    dsets.add_set(9)
    dsets.add_set(10)
    dsets.add_set(13)
    dsets.add_set(14)
    dsets.add_set(17)
    dsets.add_set(18)
    dsets.add_set(19)
    dsets.add_set(20)
    dsets.make_new_link(2,5)
    dsets.make_new_link(5,9)
    dsets.make_new_link(5,10)
    dsets.make_new_link(13,1)
    dsets.make_new_link(1,14)
    dsets.make_new_link(14,17)
    dsets.make_new_link(14,18)
    dsets.make_new_link(13,19)
    dsets.make_new_link(20,18)
    self.assertTrue(dsets.check_subset_has_elements(10,[2, 5, 9, 10]));
    self.assertTrue(dsets.check_subset_has_elements(6,[6]));
    self.assertTrue(dsets.check_subset_has_elements(18,[20, 13, 1, 14, 17, 18, 19]));
    self.assertTrue(dsets.make_new_link(2,18));
    self.assertTrue(dsets.check_subset_has_elements(6,[6]));
    self.assertTrue(dsets.check_subset_has_elements(2,[2, 5, 9, 10, 20, 13, 1, 14, 17, 18, 19]));
    self.assertFalse(dsets.make_new_link(17,10));

class BasinEvaluationTests(unittest.TestCase):

  def testBasinEvaluationSingleLake(self):
    coarse_catchment_nums_in = np.array([[1, 1],
                                         [1, 1]])
    prior_coarse_rdirs_in = np.array([[5, 5],
                                      [5, 5]])
    corrected_orography_in = np.array([
      [10., 10., 10., 10., 10., 10.],
      [10., 10.,  8.,  8.,  5., 10.],
      [10.,  7.,  5.,  5.,  6., 10.],
      [10.,  7.,  5.,  5.,  6., 10.],
      [10., 10., 10., 10., 10., 10.],
      [10., 10., 10., 10., 10.,  0.]])

    raw_orography_in = np.array([
      [10., 10., 10., 10., 10., 10.],
      [10., 10.,  8.,  8.,  5., 10.],
      [10.,  7.,  5.,  5.,  6., 10.],
      [10.,  7.,  5.,  5.,  6., 10.],
      [10., 10., 10., 10., 10., 10.],
      [10., 10., 10., 10.,  0.,  0.]])

    minima_in = np.array([
      [False, False, False, False, False, False],
      [False, False, False, False, False, False],
      [False, False, False, False, False, False],
      [False, False, True,  False, False, False],
      [False, False, False, False, False, False],
      [False, False, False, False, False, False]])
    prior_fine_rdirs_in = np.array([
      [2, 2, 2, 2, 2, 2],
      [2, 2, 2, 2, 2, 2],
      [2, 6, 2, 1, 4, 2],
      [2, 6, 5, 4, 4, 2],
      [2, 2, 2, 2, 2, 2],
      [6, 6, 6, 6, 6, 0]])
    prior_fine_catchments_in = np.array([
      [1, 2, 2, 2, 2, 1],
      [1, 2, 2, 2, 2, 1],
      [1, 2, 2, 2, 2, 1],
      [1, 2, 2, 2, 2, 1],
      [1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1]])
    cell_areas_in = np.full((6,6),1.0)
    landsea_in = np.full((6,6),0,dtype=np.int32)
    landsea_in[5,5] = 1
    catchments_from_sink_filling_in = np.zeros((6,6),dtype=np.int32)
    fill_sinks_cpp_func(orography_array=
                        corrected_orography_in,
                        method = 4,
                        use_ls_mask = True,
                        landsea_in = landsea_in,
                        set_ls_as_no_data_flag = False,
                        use_true_sinks = False,
                        true_sinks_in = np.zeros((6,6),dtype=np.int32),
                        rdirs_in = np.zeros((6,6),dtype=np.float64),
                        next_cell_lat_index_in = np.zeros((6,6),dtype=np.int32),
                        next_cell_lon_index_in = np.zeros((6,6),dtype=np.int32),
                        catchment_nums_in = catchments_from_sink_filling_in,
                        prefer_non_diagonal_initial_dirs = False)
    alg = LatLonBasinEvaluationAlgorithm(minima_in,
                                         raw_orography_in,
                                         corrected_orography_in,
                                         cell_areas_in,
                                         prior_fine_rdirs_in,
                                         prior_fine_catchments_in,
                                         coarse_catchment_nums_in,
                                         catchments_from_sink_filling_in)
    alg.evaluate_basins()
    print("---")
    print([lake.filling_order for lake in alg.lakes])
    print([lake.primary_lake for lake in alg.lakes])
    print([lake.spill_points for lake in alg.lakes])
    print([lake.potential_exit_points for lake in alg.lakes])
    print([lake.outflow_points for lake in alg.lakes])

  def testBasinEvaluationTwoLakes(self):
    coarse_catchment_nums_in = np.array([[1, 1],
                                         [1, 1]])
    prior_coarse_rdirs_in = np.array([[5, 5],
                                      [5, 5]])
    corrected_orography_in = np.array([
      [10., 10., 10., 10., 10., 10.],
      [10., 10., 10., 10., 10., 10.],
      [10.,  5., 10., 10.,  5., 10.],
      [10.,  5.,  6.,  8.,  5., 10.],
      [10., 10., 10., 10., 10., 10.],
      [10., 10., 10., 10., 10.,  0.]])

    raw_orography_in = np.array([
      [10., 10., 10., 10., 10., 10.],
      [10., 10., 10., 10., 10., 10.],
      [10.,  5., 10., 10.,  5., 10.],
      [10.,  5.,  6.,  8.,  5., 10.],
      [10., 10., 10., 10., 10., 10.],
      [10., 10., 10., 10., 10.,  0.]])

    minima_in = np.array([
      [False, False, False, False, False, False],
      [False, False, False, False, False, False],
      [False, False, False, False, False, False],
      [False,  True, False, False,  True, False],
      [False, False, False, False, False, False],
      [False, False, False, False, False, False]])
    prior_fine_rdirs_in = np.array([
      [2, 2, 2, 2, 2, 2],
      [2, 2, 2, 2, 2, 2],
      [2, 6, 2, 3, 2, 2],
      [2, 5, 4, 6, 5, 2],
      [2, 2, 2, 2, 2, 2],
      [6, 6, 6, 6, 6, 0]])
    prior_fine_catchments_in = np.array([
      [1, 3, 3, 2, 2, 1],
      [1, 3, 3, 2, 2, 1],
      [1, 3, 3, 2, 2, 1],
      [1, 3, 3, 2, 2, 1],
      [1, 3, 3, 2, 2, 1],
      [1, 1, 1, 1, 1, 1]])
    cell_areas_in = np.full((6,6),1.0)
    landsea_in = np.full((6,6),0,dtype=np.int32)
    landsea_in[5,5] = 1
    catchments_from_sink_filling_in = np.zeros((6,6),dtype=np.int32)
    fill_sinks_cpp_func(orography_array=
                        corrected_orography_in,
                        method = 4,
                        use_ls_mask = True,
                        landsea_in = landsea_in,
                        set_ls_as_no_data_flag = False,
                        use_true_sinks = False,
                        true_sinks_in = np.zeros((6,6),dtype=np.int32),
                        rdirs_in = np.zeros((6,6),dtype=np.float64),
                        next_cell_lat_index_in = np.zeros((6,6),dtype=np.int32),
                        next_cell_lon_index_in = np.zeros((6,6),dtype=np.int32),
                        catchment_nums_in = catchments_from_sink_filling_in,
                        prefer_non_diagonal_initial_dirs = False)
    alg = LatLonBasinEvaluationAlgorithm(minima_in,
                                         raw_orography_in,
                                         corrected_orography_in,
                                         cell_areas_in,
                                         prior_fine_rdirs_in,
                                         prior_fine_catchments_in,
                                         coarse_catchment_nums_in,
                                         catchments_from_sink_filling_in)
    alg.evaluate_basins()
    print("---")
    print([lake.filling_order for lake in alg.lakes])
    print([lake.primary_lake for lake in alg.lakes])
    print([lake.spill_points for lake in alg.lakes])
    print([lake.potential_exit_points for lake in alg.lakes])
    print([lake.outflow_points for lake in alg.lakes])

  def testBasinEvaluationOne(self):
    coarse_catchment_nums_in = np.array([[3,3,2,2],
                                         [3,3,2,2],
                                         [1,1,1,2],
                                         [1,1,1,1]])
    prior_coarse_rdirs_in = np.array([[5.0,5.0,5.0,5.0],
                                      [5.0,5.0,5.0,5.0],
                                      [5.0,5.0,5.0,5.0],
                                      [5.0,5.0,5.0,5.0]])
    corrected_orography_in = np.array([
     [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0],
     [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0],
     [1.0, 8.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 6.0, 6.0, 5.0, 6.0,10.0,10.0,10.0, 1.0],
     [10.0, 3.0, 3.0,10.0,10.0, 7.0, 3.0,10.0,10.0,10.0,10.0, 6.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0,10.0,10.0],
     [10.0, 3.0, 3.0,10.0,10.0, 3.0, 3.0, 4.0, 3.0,10.0,10.0,10.0, 2.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0,10.0],
     [10.0, 3.0, 3.0, 6.0, 2.0, 1.0,10.0, 2.0, 3.0, 5.0, 3.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0],
     [4.0, 4.0, 3.0,10.0, 2.0, 1.0, 2.0, 2.0, 3.0,10.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0],
     [10.0, 4.0, 4.0,10.0,10.0, 2.0,10.0, 4.0,10.0,10.0,10.0, 3.0, 2.0, 3.0, 2.0, 3.0, 5.0, 9.0,10.0,10.0],
     [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 5.0, 3.0, 3.0, 3.0, 4.0, 5.0,10.0, 8.0,10.0],
     [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 5.0, 5.0,10.0,10.0,10.0,10.0, 7.0,10.0],
     [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 6.0,10.0],
     [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 5.0,10.0],
     [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 4.0,10.0],
     [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0, 2.0, 3.0, 3.0,10.0],
     [10.0,10.0,10.0, 3.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0, 3.0, 3.0,10.0,10.0],
     [10.0,10.0,10.0, 2.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 4.0,10.0,10.0,10.0,10.0],
     [10.0,10.0,10.0, 3.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 3.0,10.0,10.0,10.0,10.0],
     [10.0,10.0,10.0, 2.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0,10.0,10.0,10.0,10.0],
     [10.0,10.0,10.0, 2.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 1.0,10.0,10.0,10.0,10.0],
     [10.0,10.0,10.0, 1.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 0.0,10.0,10.0,10.0,10.0]])

    raw_orography_in = np.array([
     [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0],
     [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0],
     [1.0, 8.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 6.0, 6.0, 5.0, 6.0,10.0,10.0,10.0, 1.0],
     [10.0, 3.0, 3.0,10.0,10.0, 8.0, 7.0,10.0,10.0,10.0,10.0, 6.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0,10.0,10.0],
     [10.0, 3.0, 3.0,10.0,10.0, 3.0, 3.0, 4.0, 3.0,10.0,10.0,10.0, 2.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0,10.0],
     [10.0, 3.0, 3.0, 6.0, 2.0, 1.0,10.0, 2.0, 3.0,10.0, 3.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0],
     [4.0, 4.0, 3.0,10.0, 2.0, 1.0, 2.0, 2.0, 3.0,10.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0],
     [10.0, 4.0, 4.0,10.0,10.0, 2.0,10.0, 4.0,10.0,10.0,10.0, 3.0, 2.0, 3.0, 2.0, 3.0, 5.0, 9.0,10.0,10.0],
     [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 5.0, 3.0, 3.0, 3.0, 4.0, 5.0,10.0, 8.0,10.0],
     [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 5.0, 5.0,10.0,10.0,10.0,10.0, 7.0,10.0],
     [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 6.0,10.0],
     [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 5.0,10.0],
     [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 4.0,10.0],
     [10.0,10.0,10.0,10.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0, 2.0, 3.0, 3.0,10.0],
     [10.0,10.0,10.0, 3.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0, 3.0, 3.0,10.0,10.0],
     [10.0,10.0,10.0, 2.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 4.0,10.0,10.0,10.0,10.0],
     [10.0,10.0,10.0, 3.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 3.0,10.0,10.0,10.0,10.0],
     [10.0,10.0,10.0, 2.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0,10.0,10.0,10.0,10.0],
     [10.0,10.0,10.0, 2.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 1.0,10.0,10.0,10.0,10.0],
     [10.0,10.0,10.0, 1.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 0.0,10.0,10.0,10.0,10.0]])

    minima_in = np.array([
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [True, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,True, False,False,
      False,False,False,False],
    [False,False,False,False,False,True, False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,True, False,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [False,False,False,True, False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False],
    [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
      False,False,False,False]])
    prior_fine_rdirs_in = np.array([
      [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,3.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,3.0,3.0,2.0],
      [2.0,1.0,2.0,1.0,1.0,2.0,1.0,1.0,1.0,3.0,3.0,3.0,2.0,3.0,2.0,1.0,1.0,3.0,3.0,3.0],
      [5.0,4.0,1.0,1.0,3.0,2.0,1.0,1.0,1.0,1.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,6.0,6.0,6.0],
      [8.0,7.0,4.0,4.0,3.0,2.0,1.0,1.0,2.0,1.0,6.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0],
      [9.0,8.0,7.0,3.0,3.0,2.0,1.0,2.0,1.0,1.0,3.0,6.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0],
      [9.0,8.0,7.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,3.0,3.0,6.0,5.0,4.0,4.0,1.0,1.0,1.0,3.0],
      [9.0,8.0,7.0,6.0,6.0,5.0,4.0,4.0,4.0,4.0,6.0,6.0,9.0,8.0,7.0,1.0,1.0,4.0,4.0,6.0],
      [9.0,9.0,8.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0,9.0,9.0,8.0,7.0,5.0,4.0,4.0,7.0,7.0,9.0],
      [9.0,9.0,8.0,7.0,9.0,8.0,7.0,8.0,7.0,7.0,9.0,9.0,8.0,7.0,8.0,7.0,7.0,7.0,2.0,1.0],
      [9.0,9.0,8.0,9.0,9.0,8.0,7.0,7.0,7.0,9.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0,7.0,2.0,1.0],
      [1.0,1.0,9.0,9.0,9.0,8.0,7.0,7.0,7.0,9.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0,3.0,2.0,1.0],
      [1.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,9.0,8.0,3.0,3.0,2.0,1.0,3.0,2.0,1.0],
      [1.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,3.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0],
      [4.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,3.0,6.0,6.0,6.0,5.0,4.0,4.0,4.0,4.0],
      [7.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,3.0,3.0,3.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0],
      [7.0,3.0,6.0,5.0,4.0,1.0,1.0,1.0,1.0,1.0,3.0,3.0,3.0,3.0,9.0,8.0,7.0,7.0,7.0,7.0],
      [3.0,3.0,3.0,2.0,1.0,4.0,4.0,4.0,4.0,4.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,7.0,7.0],
      [3.0,3.0,3.0,2.0,1.0,7.0,7.0,7.0,7.0,7.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,1.0,7.0],
      [3.0,3.0,3.0,2.0,1.0,1.0,7.0,7.0,7.0,7.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,1.0,3.0],
      [6.0,6.0,6.0,0.0,4.0,4.0,4.0,7.0,7.0,7.0,6.0,6.0,6.0,6.0,6.0,0.0,4.0,4.0,4.0,6.0]])
    prior_fine_catchments_in = np.array([
      [11,11,11,11,11,11,13,13,12,12,12,12,12,12,12,12,12,11,11,11],
      [11,11,11,11,11,13,13,13,13,12,12,12,12,12,12,12,12,11,11,11],
      [11,11,11,11,13,13,13,13,13,13,12,12,12,12,12,12,12,11,11,11],
      [11,11,11,11,13,13,13,13,13,13,12,12,12,12,12,12,12,12,11,11],
      [11,11,11,13,13,13,13,13,13,13,12,12,12,12,12,12,12,14,14,11],
      [11,11,11,13,13,13,13,13,13,13,12,12,12,12,12,12,14,14,14,11],
      [11,11,11,13,13,13,13,13,13,13,12,12,12,12,12,14,14,14,14,11],
      [11,11,11,13,13,13,13,13,13,13,12,12,12,12,14,14,14,14,14,11],
      [11,11,11,11,13,13,13,13,13,13,12,12,12,12,14,14,14,14,15,15],
      [11,11,11,13,13,13,13,13,13,12,12,12,12,12,12,14,14,14,15,15],
      [15,15,13,13,13,13,13,13,13,12,12,12,12,12,12,12,14,15,15,15],
      [15,16,16,16,16,16,16,16,12,12,12,12,12,15,15,15,15,15,15,15],
      [15,16,16,16,16,16,16,16, 4,12,12,12,15,15,15,15,15,15,15,15],
      [15,16,16,16,16,16,16, 4, 4,12,12, 9,15,15,15,15,15,15,15,15],
      [15,16,16,16,16,16, 4, 4, 4, 4,10, 9, 9,15,15,15,15,15,15,15],
      [15, 4,16,16,16, 4, 4, 4, 4, 4, 7,10, 9, 9,15,15,15,15,15,15],
      [ 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7,10, 9, 9, 9, 9, 9,15,15],
      [ 2, 5, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7,10, 9, 9, 9, 8, 6,15],
      [ 2, 2, 5, 4, 3, 1, 4, 4, 4, 4, 7, 7, 7, 7,10, 9, 8, 6, 6, 2],
      [ 2, 2, 2, 0, 1, 1, 1, 4, 4, 4, 7, 7, 7, 7, 7, 0, 6, 6, 6, 2]])
    cell_areas_in = np.full((20,20),1.0)
    catchments_from_sink_filling_in = np.zeros((20,20),dtype=np.int32)
    fill_sinks_cpp_func(orography_array=
                        corrected_orography_in,
                        method = 4,
                        use_ls_mask = False,
                        landsea_in = np.full((20,20),0,dtype=np.int32),
                        set_ls_as_no_data_flag = False,
                        use_true_sinks = False,
                        true_sinks_in = np.zeros((20,20),dtype=np.int32),
                        rdirs_in = np.zeros((20,20),dtype=np.float64),
                        next_cell_lat_index_in = np.zeros((20,20),dtype=np.int32),
                        next_cell_lon_index_in = np.zeros((20,20),dtype=np.int32),
                        catchment_nums_in = catchments_from_sink_filling_in,
                        prefer_non_diagonal_initial_dirs = False)
    print(catchments_from_sink_filling_in)
    alg = LatLonBasinEvaluationAlgorithm(minima_in,
                                         raw_orography_in,
                                         corrected_orography_in,
                                         cell_areas_in,
                                         prior_fine_rdirs_in,
                                         prior_fine_catchments_in,
                                         coarse_catchment_nums_in,
                                         catchments_from_sink_filling_in)
    alg.evaluate_basins()
    print("---")
    print([lake.filling_order for lake in alg.lakes])
    print([lake.primary_lake for lake in alg.lakes])
    print([lake.spill_points for lake in alg.lakes])
    print([lake.outflow_points for lake in alg.lakes])

if __name__ == "__main__":
    unittest.main()
