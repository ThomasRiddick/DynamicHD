'''
Unit tests for the compute_catchments module

Created on Oct 31, 2024

@author: thomasriddick
'''

import unittest
import numpy as np
from Dynamic_HD_Scripts.utilities.basin_evaluation_algorithm_prototype \
import LatLonEvaluateBasin,DisjointSetForest


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
    coarse_catchment_nums_in = np.array([[1, 1, 1],
                                         [1, 1, 1],
                                         [1, 1, 1]])
    prior_coarse_rdirs_in = np.array([[6, 6,  2],
                                      [6, -2, 2],
                                      [6,  6, 0]])
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
    output,alg = \
      LatLonEvaluateBasin.evaluate_basins(landsea_in,
                                          minima_in,
                                          raw_orography_in,
                                          corrected_orography_in,
                                          cell_areas_in,
                                          prior_fine_rdirs_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          return_algorithm_object=True)
    print("---")
    print("one lake")
    print([lake.filling_order for lake in alg.lakes])
    print([lake.primary_lake for lake in alg.lakes])
    print([lake.spill_points for lake in alg.lakes])
    print([lake.potential_exit_points for lake in alg.lakes])
    print([lake.outflow_points for lake in alg.lakes])
    print(output)

  def testBasinEvaluationTwoLakes(self):
    coarse_catchment_nums_in = np.array([[2, 2, 1],
                                         [2, 2, 1],
                                         [2, 2, 1]])
    prior_coarse_rdirs_in = np.array([[2, 2,  2],
                                      [-2, 4, -2],
                                      [6,  6, 0]])
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
    output,alg = \
      LatLonEvaluateBasin.evaluate_basins(landsea_in,
                                          minima_in,
                                          raw_orography_in,
                                          corrected_orography_in,
                                          cell_areas_in,
                                          prior_fine_rdirs_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          return_algorithm_object=True)
    print("---")
    print("two lakes")
    print([lake.filling_order for lake in alg.lakes])
    print([lake.primary_lake for lake in alg.lakes])
    print([lake.spill_points for lake in alg.lakes])
    print([lake.potential_exit_points for lake in alg.lakes])
    print([lake.outflow_points for lake in alg.lakes])
    print(output)

  def testBasinEvaluationSingleLakeTwo(self):
    coarse_catchment_nums_in = np.array([[2, 1, 1],
                                         [2, 2, 1],
                                         [2, 2, 1]])
    prior_coarse_rdirs_in = np.array([[-2,6,2],
                                      [8,7,2 ],
                                      [8,7,0 ]])
    corrected_orography_in = np.array([
      [10.0,10.0,10.0, 3.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0,10.0, 3.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0, 2.0, 3.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0, 2.0, 2.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0, 2.0, 2.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 0.0]])

    raw_orography_in = np.array([
      [10.0,10.0,10.0, 3.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0,10.0, 3.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0, 1.0, 3.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0, 2.0, 2.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0, 2.0, 2.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 0.0]])

    minima_in = np.array([
      [False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False],
      [False,False, True,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False]])
    prior_fine_rdirs_in = np.array([
      [3, 2, 6, 6, 6, 6, 6, 6, 2],
      [3, 3, 8, 6, 6, 6, 6, 6, 2],
      [4, 4, 5, 4, 4, 4, 2, 2, 2],
      [8, 7, 8, 4, 4, 4, 2, 2, 2],
      [8, 7, 8, 4, 4, 4, 2, 2, 2],
      [8, 7, 8, 4, 4, 4, 2, 2, 2],
      [8, 8, 8, 4, 4, 4, 2, 2, 2],
      [8, 8, 8, 4, 4, 4, 2, 2, 2],
      [8, 8, 8, 4, 4, 4, 2, 6, 0]])
    prior_fine_catchments_in = np.array([
      [1, 1, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 2, 2, 2],
      [1, 1, 1, 1, 1, 1, 2, 6, 0]])
    cell_areas_in = np.full((9,9),1.0)
    landsea_in = np.full((9,9),0,dtype=np.int32)
    landsea_in[8,8] = 1
    output,alg = \
      LatLonEvaluateBasin.evaluate_basins(landsea_in,
                                          minima_in,
                                          raw_orography_in,
                                          corrected_orography_in,
                                          cell_areas_in,
                                          prior_fine_rdirs_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          return_algorithm_object=True)
    print("---")
    print("single lake 2")
    print([lake.filling_order for lake in alg.lakes])
    print([lake.primary_lake for lake in alg.lakes])
    print([lake.spill_points for lake in alg.lakes])
    print([lake.potential_exit_points for lake in alg.lakes])
    print([lake.outflow_points for lake in alg.lakes])
    print(output)

  def testBasinEvaluationSingleLakeThree(self):
    coarse_catchment_nums_in = np.array([[2, 1, 1],
                                         [2, 1, 1],
                                         [2, 1, 1]])
    prior_coarse_rdirs_in = np.array([[-2,6,2],
                                      [ 8,8,2],
                                      [ 8,8,0]])
    corrected_orography_in = np.array([
      [10.0,10.0,10.0, 5.0,8.375,8.0, 8.0, 8.0,8.0],
      [10.0,10.0,10.0, 5.0,10.0,10.0,10.0,10.0,8.0],
      [10.0,10.0, 2.0, 2.0,10.0,10.0,10.0,10.0,8.0],
      [10.0,10.0, 2.0, 2.0,10.0,10.0,10.0,10.0,8.0],
      [10.0,10.0, 2.0, 2.0,10.0,10.0,10.0,10.0,8.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,8.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,8.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,8.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,0.0]])

    raw_orography_in = np.array([
      [10.0,10.0,10.0, 5.0,8.375,8.0, 8.0, 8.0,8.0],
      [10.0,10.0,10.0, 5.0,10.0,10.0,10.0,10.0,8.0],
      [10.0,10.0, 2.0, 2.0,10.0,10.0,10.0,10.0,8.0],
      [10.0,10.0, 2.0, 2.0,10.0,10.0,10.0,10.0,8.0],
      [10.0,10.0, 2.0, 2.0,10.0,10.0,10.0,10.0,8.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,8.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,8.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,8.0],
      [10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,0.0]])

    minima_in = np.array([
      [False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False],
      [False,False, True,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False]])
    prior_fine_rdirs_in = np.array([
      [2, 2, 2, 6, 6, 6, 6, 6, 2],
      [6, 3, 2, 6, 6, 6, 6, 6, 2],
      [6, 6, 5, 8, 8, 8, 2, 2, 2],
      [8, 8, 8, 8, 8, 8, 2, 2, 2],
      [8, 8, 8, 8, 8, 8, 2, 2, 2],
      [8, 8, 8, 8, 8, 8, 2, 2, 2],
      [8, 8, 8, 8, 8, 8, 2, 2, 2],
      [8, 8, 8, 8, 8, 8, 2, 2, 2],
      [8, 8, 8, 8, 8, 8, 6, 6, 0]])
    prior_fine_catchments_in = np.array([
      [1, 1, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 2, 2, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 2, 2, 2, 2, 2],
      [1, 1, 1, 1, 2, 2, 2, 2, 0]])
    cell_areas_in = np.full((9,9),86400.0/9.0)
    landsea_in = np.full((9,9),0,dtype=np.int32)
    landsea_in[8,8] = 1
    output,alg = \
      LatLonEvaluateBasin.evaluate_basins(landsea_in,
                                          minima_in,
                                          raw_orography_in,
                                          corrected_orography_in,
                                          cell_areas_in,
                                          prior_fine_rdirs_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          return_algorithm_object=True)
    print("---")
    print("single lake 3")
    print([lake.filling_order for lake in alg.lakes])
    print([lake.primary_lake for lake in alg.lakes])
    print([lake.spill_points for lake in alg.lakes])
    print([lake.potential_exit_points for lake in alg.lakes])
    print([lake.outflow_points for lake in alg.lakes])
    print(output)

  def testBasinEvaluationOne(self):
    coarse_catchment_nums_in = np.array([[3,3,7,2],
                                         [3,6,7,2],
                                         [5,4,1,2],
                                         [5,4,1,1]])
    prior_coarse_rdirs_in = np.array([[5.0,5.0,5.0,5.0],
                                      [5.0,5.0,5.0,5.0],
                                      [5.0,5.0,5.0,5.0],
                                      [5.0,0.0,5.0,0.0]])
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
     [10.0,10.0,10.0, 1.0, 0.2, 0.1, 0.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 0.0,10.0,10.0,10.0,10.0]])

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
     [10.0,10.0,10.0, 1.0, 0.2, 0.1, 0.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 0.0,10.0,10.0,10.0,10.0]])

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
      [3.0,3.0,3.0,2.0,2.0,4.0,4.0,4.0,4.0,4.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,7.0,7.0],
      [3.0,3.0,3.0,2.0,2.0,7.0,7.0,7.0,7.0,7.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,1.0,7.0],
      [3.0,3.0,3.0,2.0,2.0,2.0,2.0,7.0,7.0,7.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,1.0,3.0],
      [6.0,6.0,6.0,6.0,6.0,6.0,0.0,7.0,7.0,7.0,6.0,6.0,6.0,6.0,6.0,0.0,4.0,4.0,4.0,6.0]])
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
      [ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7,10, 9, 9, 9, 9, 9,15,15],
      [ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7,10, 9, 9, 9, 8, 6,15],
      [ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7,10, 9, 8, 6, 6, 2],
      [ 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 7, 7, 7, 7, 7, 0, 6, 6, 6, 2]])
    landsea_in = np.zeros((20,20),dtype=np.int32)
    landsea_in[:,:] = False
    landsea_in[19,15] = True
    landsea_in[19,6] = True
    cell_areas_in = np.full((20,20),1.0)
    output,alg = \
      LatLonEvaluateBasin.evaluate_basins(landsea_in,
                                          minima_in,
                                          raw_orography_in,
                                          corrected_orography_in,
                                          cell_areas_in,
                                          prior_fine_rdirs_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          return_algorithm_object=True)
    print("---")
    print("Evaluation One")
    print([lake.filling_order for lake in alg.lakes])
    print([lake.primary_lake for lake in alg.lakes])
    print([lake.spill_points for lake in alg.lakes])
    print([lake.outflow_points for lake in alg.lakes])
    print(output)

  def testBasinEvaluationTwo(self):
    coarse_catchment_nums_in = np.array([[3,3,2,2],
                                         [3,3,2,2],
                                         [1,1,1,2],
                                         [1,1,1,1]],
                                         dtype=np.int32)
    corrected_orography_in = np.array([
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 1.0,10.0,20.0, 9.0, 20.0,20.0,20.0, 1.0,16.0, 20.0,20.0, 1.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 2.0,11.0,20.0, 8.0, 20.0,20.0,20.0, 2.0,15.0, 20.0,20.0, 2.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [18.0, 3.0,12.0,20.0, 7.0, 20.0,20.0,20.0, 3.0,14.0, 20.0,20.0, 3.0,18.0, 0.9,  0.8, 0.0, 0.7, 0.8, 0.9],
      [20.0, 4.0,13.0, 6.0, 5.0, 20.0,20.0,20.0, 4.0,13.0,  5.0, 5.0, 4.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 5.0,14.0,20.0, 4.0, 20.0,20.0,20.0, 5.0,12.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 6.0,15.0,20.0, 3.0, 20.0,20.0,20.0, 6.0,11.0, 20.0,20.0, 7.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 8.0,16.0,20.0, 2.0, 20.0,20.0,20.0, 7.0,10.0, 20.0,20.0, 8.0,20.0,20.0, 20.0,20.0, 5.0,20.0,20.0],
      [20.0, 9.0,17.0,20.0, 1.0, 20.0,20.0,20.0, 8.0, 9.0, 20.0,20.0, 9.0,20.0,20.0, 20.0, 4.0, 6.0,20.0,20.0],
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,  3.0, 7.0,20.0,20.0,20.0],
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0, 2.0,  8.0,20.0,20.0,20.0,20.0],
      [20.0, 1.0,16.0,20.0, 2.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 1.0, 9.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 2.0,15.0,20.0, 3.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,10.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 3.0,14.0,20.0, 4.0, 20.0,20.0, 1.0,20.0, 5.0, 20.0,20.0,20.0, 0.9,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 4.0,13.0, 6.0, 5.0, 20.0,20.0, 2.0, 4.0, 3.0, 20.0,20.0, 0.8,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 5.0,12.0,20.0, 7.0, 20.0,20.0, 3.0,20.0, 2.0, 20.0,20.0, 0.7,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 6.0,11.0,20.0, 8.0, 20.0,20.0, 4.0,20.0, 1.0, 20.0,20.0, 0.6,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 7.0,10.0,20.0, 9.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0, 0.5,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 8.0, 9.0,20.0,10.0, 18.0, 0.9, 0.8,20.0,20.0, 20.0, 0.4,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 0.0, 0.2,  0.3,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0]],
      dtype=np.float64)
    raw_orography_in = np.array([
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 1.0,10.0,20.0, 9.0, 20.0,20.0,20.0, 1.0,16.0, 20.0,20.0, 1.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 2.0,11.0,20.0, 8.0, 20.0,20.0,20.0, 2.0,15.0, 20.0,20.0, 2.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [18.0, 3.0,12.0,20.0, 7.0, 20.0,20.0,20.0, 3.0,14.0, 20.0,20.0, 3.0,18.0, 0.9,  0.8, 0.0, 0.7, 0.8, 0.9],
      [20.0, 4.0,13.0,20.0, 5.0, 20.0,20.0,20.0, 4.0,13.0,  5.0, 5.0, 4.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 5.0,14.0,20.0, 4.0, 20.0,20.0,20.0, 5.0,12.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 6.0,15.0,20.0, 3.0, 20.0,20.0,20.0, 6.0,11.0, 20.0,20.0, 7.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 8.0,16.0,20.0, 2.0, 20.0,20.0,20.0, 7.0,10.0, 20.0,20.0, 8.0,20.0,20.0, 20.0,20.0, 5.0,20.0,20.0],
      [20.0, 9.0,17.0,20.0, 1.0, 20.0,20.0,20.0, 8.0, 9.0, 20.0,20.0, 9.0,20.0,20.0, 20.0, 4.0, 6.0,20.0,20.0],
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,  3.0, 7.0,20.0,20.0,20.0],
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0, 2.0,  8.0,20.0,20.0,20.0,20.0],
      [20.0, 1.0,16.0,20.0, 2.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 1.0, 9.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 2.0,15.0,20.0, 3.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,10.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 3.0,14.0,20.0, 4.0, 20.0,20.0, 1.0,20.0, 5.0, 20.0,20.0,20.0, 0.9,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 4.0,13.0,20.0, 5.0, 20.0,20.0, 2.0, 4.0, 3.0, 20.0,20.0, 0.8,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 5.0,12.0,20.0, 7.0, 20.0,20.0, 3.0,20.0, 2.0, 20.0,20.0, 0.7,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 6.0,11.0,20.0, 8.0, 20.0,20.0, 4.0,20.0, 1.0, 20.0,20.0, 0.6,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 7.0,10.0,20.0, 9.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0, 0.5,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 8.0, 9.0,20.0,10.0, 18.0, 0.9, 0.8,20.0,20.0, 20.0, 0.4,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 0.0, 0.2,  0.3,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0]],
      dtype=np.float64)
    minima_in = np.array([
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False, True,False,False,False,False,False,False, True,False,False,False, True,False,False,False,False,False,False,False],
      [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False, False,False,False,False,False,False],
      [False,False,False,False,False,False, False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False, True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False, True,False,False, True,False,False,False,False,False,False,False,False, True,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False, True,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False, True,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]],
      dtype=np.int32)
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
      [6.0,6.0,6.0,0.0,4.0,4.0,4.0,7.0,7.0,7.0,6.0,6.0,6.0,6.0,6.0,0.0,4.0,4.0,4.0,6.0]],
      dtype=np.float64)
    prior_coarse_rdirs_in = np.array([[5.0,5.0,5.0,5.0],
                                      [5.0,5.0,5.0,5.0],
                                      [5.0,5.0,5.0,5.0],
                                      [5.0,5.0,5.0,5.0]],
                                      dtype=np.float64)
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
      [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7,10, 9, 9, 9, 9, 9,15,15],
      [2, 5, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7,10, 9, 9, 9, 8, 6,15],
      [2, 2, 5, 4, 3, 1, 4, 4, 4, 4, 7, 7, 7, 7,10, 9, 8, 6, 6, 2],
      [2, 2, 2, 0, 1, 1, 1, 4, 4, 4, 7, 7, 7, 7, 7, 0, 6, 6, 6, 2]],
      dtype=np.int32)
    landsea_in = np.array([
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False, False,False,False,False,False,False],
      [False,False,False,False,False,False, False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False, False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False, True,False,False,False,False,False,False,False,False,False,False,False]],
      dtype=np.int32)
    cell_areas_in = np.full((20,20),1.0)
    _,alg = \
      LatLonEvaluateBasin.evaluate_basins(landsea_in,
                                          minima_in,
                                          raw_orography_in,
                                          corrected_orography_in,
                                          cell_areas_in,
                                          prior_fine_rdirs_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          return_algorithm_object=True)
    print("---")
    print([lake.filling_order for lake in alg.lakes])
    print([lake.primary_lake for lake in alg.lakes])
    print([lake.spill_points for lake in alg.lakes])
    print([lake.outflow_points for lake in alg.lakes])

  def testBasinEvaluationThree(self):
    coarse_catchment_nums_in = np.array([[3,3,2,2],
                                         [3,3,2,2],
                                         [1,1,1,2],
                                         [1,1,1,1]],
                                         dtype=np.int32)
    corrected_orography_in = np.array([
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 1.0,10.0,20.0, 9.0, 20.0,20.0,20.0, 1.0,16.0, 20.0,20.0, 1.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 2.0,11.0,20.0, 8.0, 20.0,20.0,20.0, 2.0,15.0, 20.0,20.0, 2.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [18.0, 3.0,12.0,20.0, 7.0, 20.0,20.0,20.0, 3.0,14.0, 20.0,20.0, 3.0,18.0, 0.9,  0.8, 0.0, 0.7, 0.8, 0.9],
      [20.0, 4.0,13.0, 6.0, 5.0, 20.0,20.0,20.0, 4.0,13.0,  5.0, 5.0, 4.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 5.0,14.0,20.0, 4.0, 20.0,20.0,20.0, 5.0,12.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 6.0,15.0,20.0, 3.0, 20.0,20.0,20.0, 6.0,11.0, 20.0,20.0, 7.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 8.0,16.0,20.0, 2.0, 20.0,20.0,20.0, 7.0,10.0, 20.0,20.0, 8.0,20.0,20.0, 20.0,20.0, 5.0,20.0,20.0],
      [20.0, 9.0,17.0,20.0, 1.0, 20.0,20.0,20.0, 8.0, 9.0, 20.0,20.0, 9.0,20.0,20.0, 20.0, 4.0, 6.0,20.0,20.0],
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,  3.0, 7.0,20.0,20.0,20.0],
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0, 2.0,  8.0,20.0,20.0,20.0,20.0],
      [20.0, 1.0,16.0,20.0, 2.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 1.0, 9.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 2.0,15.0,20.0, 3.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,10.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 3.0,14.0,20.0, 4.0, 20.0,20.0, 1.0,20.0, 5.0, 20.0,20.0,20.0, 0.9,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 4.0,13.0, 6.0, 5.0, 20.0,20.0, 2.0, 4.0, 3.0, 20.0,20.0, 0.8,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 5.0,12.0,20.0, 7.0, 20.0,20.0, 3.0,20.0, 2.0, 20.0,20.0, 0.7,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 6.0,11.0,20.0, 8.0, 20.0,20.0, 4.0,20.0, 1.0, 20.0,20.0, 0.6,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 7.0,10.0,20.0, 9.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0, 0.5,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 8.0, 9.0,20.0,10.0, 18.0, 0.9, 0.8,20.0,20.0, 20.0, 0.4,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 0.0, 0.2,  0.3,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0]],
      dtype=np.float64)
    raw_orography_in = np.array([
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 1.0,10.0,20.0, 9.0, 20.0,20.0,20.0, 1.0,16.0, 20.0,20.0, 1.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 2.0,11.0,20.0, 8.0, 20.0,20.0,20.0, 2.0,15.0, 20.0,20.0, 2.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [18.0, 3.0,12.0,20.0, 7.0, 20.0,20.0,20.0, 3.0,14.0, 20.0,20.0, 3.0,18.0, 0.9,  0.8, 0.0, 0.7, 0.8, 0.9],
      [20.0, 4.0,13.0,20.0, 5.0, 20.0,20.0,20.0, 4.0,13.0,  5.0, 5.0, 4.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 5.0,14.0,20.0, 4.0, 20.0,20.0,20.0, 5.0,12.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 6.0,15.0,20.0, 3.0, 20.0,20.0,20.0, 6.0,11.0, 20.0,20.0, 7.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 8.0,16.0,20.0, 2.0, 20.0,20.0,20.0, 7.0,10.0, 20.0,20.0, 8.0,20.0,20.0, 20.0,20.0, 5.0,20.0,20.0],
      [20.0, 9.0,17.0,20.0, 1.0, 20.0,20.0,20.0, 8.0, 9.0, 20.0,20.0, 9.0,20.0,20.0, 20.0, 4.0, 6.0,20.0,20.0],
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,  3.0, 7.0,20.0,20.0,20.0],
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0, 2.0,  8.0,20.0,20.0,20.0,20.0],
      [20.0, 1.0,16.0,20.0, 2.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 1.0, 9.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 2.0,15.0,20.0, 3.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,10.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 3.0,14.0,20.0, 4.0, 20.0,20.0, 1.0,20.0, 5.0, 20.0,20.0,20.0, 0.9,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 4.0,13.0,20.0, 5.0, 20.0,20.0, 2.0, 4.0, 3.0, 20.0,20.0, 0.8,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 5.0,12.0,20.0, 7.0, 20.0,20.0, 3.0,20.0, 2.0, 20.0,20.0, 0.7,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 6.0,11.0,20.0, 8.0, 20.0,20.0, 4.0,20.0, 1.0, 20.0,20.0, 0.6,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 7.0,10.0,20.0, 9.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0, 0.5,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0, 8.0, 9.0,20.0,10.0, 18.0, 0.9, 0.8,20.0,20.0, 20.0, 0.4,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0],
      [20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 0.0, 0.2,  0.3,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0]],
      dtype=np.float64)
    minima_in = np.array([
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False, True,False,False,False,False,False,False, True,False,False,False, True,False,False,False,False,False,False,False],
      [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False, False,False,False,False,False,False],
      [False,False,False,False,False,False, False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False, True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False, True,False,False, True,False,False,False,False,False,False,False,False, True,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False, True,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False, True,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]],
      dtype=np.int32)
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
      [6.0,6.0,6.0,0.0,4.0,4.0,4.0,7.0,7.0,7.0,6.0,6.0,6.0,6.0,6.0,0.0,4.0,4.0,4.0,6.0]],
      dtype=np.float64)
    prior_coarse_rdirs_in = np.array([[5.0,5.0,5.0,5.0],
                                      [5.0,5.0,5.0,5.0],
                                      [5.0,5.0,5.0,5.0],
                                      [5.0,5.0,5.0,5.0]],
                                      dtype=np.float64)
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
      [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7,10, 9, 9, 9, 9, 9,15,15],
      [2, 5, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7,10, 9, 9, 9, 8, 6,15],
      [2, 2, 5, 4, 3, 1, 4, 4, 4, 4, 7, 7, 7, 7,10, 9, 8, 6, 6, 2],
      [2, 2, 2, 0, 1, 1, 1, 4, 4, 4, 7, 7, 7, 7, 7, 0, 6, 6, 6, 2]],
      dtype=np.int32)
    cell_areas_in = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                              [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                              [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                              [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],
                              [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
                              [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6],
                              [7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],
                              [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
                              [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
                              [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10],
                              [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10],
                              [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],
                              [8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
                              [7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7],
                              [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6],
                              [5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
                              [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],
                              [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
                              [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                              [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]],
                              dtype=np.float64)
    landsea_in = np.array([
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False, False,False,False,False,False,False],
      [False,False,False,False,False,False, False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False, False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False],
      [False,False,False,False,False,False,False,False, True,False,False,False,False,False,False,False,False,False,False,False]],
      dtype=np.int32)
    cell_areas_in = np.full((20,20),1.0)
    _,alg = \
      LatLonEvaluateBasin.evaluate_basins(landsea_in,
                                          minima_in,
                                          raw_orography_in,
                                          corrected_orography_in,
                                          cell_areas_in,
                                          prior_fine_rdirs_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          return_algorithm_object=True)
    print("---")
    print([lake.filling_order for lake in alg.lakes])
    print([lake.primary_lake for lake in alg.lakes])
    print([lake.spill_points for lake in alg.lakes])
    print([lake.outflow_points for lake in alg.lakes])

  # def testEvaluateBasinsFour(self):
  #   cell_neighbors = np.array([[5,7,2],
  #                              [1,10,3],
  #                              [2,13,4],
  #                              [3,16,5],
  #                              [4,19,1],
  #                              [20,21,7],
  #                              [1,6,8],
  #                              [7,23,9],
  #                              [8,25,10],
  #                              [2,9,11],
  #                              [10,27,12],
  #                              [11,29,13],
  #                              [3,12,14],
  #                              [13,31,15],
  #                              [14,33,16],
  #                              [4,15,17],
  #                              [16,35,18],
  #                              [17,37,19],
  #                              [5,18,20],
  #                              [19,39,6],
  #                              [6,40,22],
  #                              [21,41,23],
  #                              [8,22,24],
  #                              [23,43,25],
  #                              [24,26,9],
  #                              [25,45,27],
  #                              [11,26,28],
  #                              [27,47,29],
  #                              [12,28,30],
  #                              [29,49,31],
  #                              [14,30,32],
  #                              [31,51,33],
  #                              [15,32,34],
  #                              [33,53,35],
  #                              [17,34,36],
  #                              [35,55,37],
  #                              [18,36,38],
  #                              [37,57,39],
  #                              [20,38,40],
  #                              [39,59,21],
  #                              [22,60,42],
  #                              [41,61,43],
  #                              [24,42,44],
  #                              [43,63,45],
  #                              [26,44,46],
  #                              [45,64,47],
  #                              [28,46,48],
  #                              [47,66,49],
  #                              [30,48,50],
  #                              [49,67,51],
  #                              [32,50,52],
  #                              [51,69,53],
  #                              [34,52,54],
  #                              [53,70,55],
  #                              [36,54,56],
  #                              [55,72,57],
  #                              [38,56,58],
  #                              [57,73,59],
  #                              [40,58,60],
  #                              [59,75,41],
  #                              [42,75,62],
  #                              [61,76,63],
  #                              [44,62,64],
  #                              [46,63,65],
  #                              [64,77,66],
  #                              [48,65,67],
  #                              [50,66,68],
  #                              [67,78,69],
  #                              [52,68,70],
  #                              [54,69,71],
  #                              [70,79,72],
  #                              [56,71,73],
  #                              [58,72,74],
  #                              [73,80,75],
  #                              [60,74,61],
  #                              [62,80,77],
  #                              [65,76,78],
  #                              [68,77,79],
  #                              [71,78,80],
  #                              [74,79,76]],
  #                              dtype=np.int32)
  #   prior_fine_rdirs_in = np.array([[8],
  #                                   [13],
  #                                   [13],
  #                                   [13],
  #                                   [19],
  #                                   [8],
  #                                   [8],
  #                                   [24],
  #                                   [24],
  #                                   [13],
  #                                   [13],
  #                                   [13],
  #                                   [-2],
  #                                   [13],
  #                                   [13],
  #                                   [13],
  #                                   [36],
  #                                   [36],
  #                                   [37],
  #                                   [37],
  #                                   [8],
  #                                   [24],
  #                                   [24],
  #                                   [64],
  #                                   [45],
  #                                   [45],
  #                                   [45],
  #                                   [49],
  #                                   [49],
  #                                   [13],
  #                                   [13],
  #                                   [30],
  #                                   [52],
  #                                   [55],
  #                                   [55],
  #                                   [55],
  #                                   [55],
  #                                   [55],
  #                                   [37],
  #                                   [38],
  #                                   [61],
  #                                   [61],
  #                                   [64],
  #                                   [64],
  #                                   [64],
  #                                   [64],
  #                                   [64],
  #                                   [64],
  #                                   [-2],
  #                                   [49],
  #                                   [30],
  #                                   [54],
  #                                   [55],
  #                                   [55],
  #                                   [0],
  #                                   [55],
  #                                   [55],
  #                                   [38],
  #                                   [38],
  #                                   [59],
  #                                   [63],
  #                                   [64],
  #                                   [64],
  #                                   [-2],
  #                                   [64],
  #                                   [38],
  #                                   [49],
  #                                   [52],
  #                                   [55],
  #                                   [55],
  #                                   [55],
  #                                   [55],
  #                                   [56],
  #                                   [58],
  #                                   [58],
  #                                   [64],
  #                                   [64],
  #                                   [68],
  #                                   [71],
  #                                   [71]],
  #                                   dtype=np.int32)
  #   prior_coarse_rdirs_in = prior_fine_rdirs_in.copy()
  #   raw_orography_in = np.array([[10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [3.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [6.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [4.0],
  #                                [4.0],
  #                                [7.0],
  #                                [10.0],
  #                                [5.0],
  #                                [9.0],
  #                                [10.0],
  #                                [8.0],
  #                                [10.0],
  #                                [6.0],
  #                                [0.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [4.0],
  #                                [3.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [5.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0],
  #                                [10.0]],
  #                                dtype=np.float64)
  #   corrected_orography_in = raw_orography_in.copy()
  #   minima_in = np.array([[False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [True],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [True],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [True],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False],
  #                         [False]],
  #                         dtype=np.int32)
  #   catchments_from_sink_filling_in = np.zeros((80),dtype=np.int32)
  #   fill_sinks_cpp_func(orography_array=
  #                       corrected_orography_in,
  #                       method = 4,
  #                       use_ls_mask = False,
  #                       landsea_in = np.full((80),0,dtype=np.int32),
  #                       set_ls_as_no_data_flag = False,
  #                       use_true_sinks = False,
  #                       true_sinks_in = np.zeros((80),dtype=np.int32),
  #                       rdirs_in = np.zeros((80),dtype=np.float64),
  #                       next_cell_lat_index_in = np.zeros((80),dtype=np.int32),
  #                       next_cell_lon_index_in = np.zeros((80),dtype=np.int32),
  #                       catchment_nums_in = catchments_from_sink_filling_in,
  #                       prefer_non_diagonal_initial_dirs = False)
  #   alg = SingleIndexBasinEvaluationAlgorithm(minima_in,
  #                                             raw_orography_in,
  #                                             corrected_orography_in,
  #                                             cell_areas_in,
  #                                             prior_fine_rdirs_in,
  #                                             prior_fine_catchments_in,
  #                                             coarse_catchment_nums_in,
  #                                             catchments_from_sink_filling_in)
  #   alg.evaluate_basins()
  #   print("---")
  #   print([lake.filling_order for lake in alg.lakes])
  #   print([lake.primary_lake for lake in alg.lakes])
  #   print([lake.spill_points for lake in alg.lakes])
  #   print([lake.outflow_points for lake in alg.lakes])

  def testEvaluateBasinsFive(self):
    coarse_catchment_nums_in = np.array([[4,1,2,3],
                                         [4,4,6,5],
                                         [4,4,6,5],
                                         [7,7,7,8]],
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
    _,alg = \
      LatLonEvaluateBasin.evaluate_basins(landsea_in,
                                          minima_in,
                                          raw_orography_in,
                                          corrected_orography_in,
                                          cell_areas_in,
                                          prior_fine_rdirs_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          return_algorithm_object=True)
    print("---")
    print([lake.filling_order for lake in alg.lakes])
    print([lake.primary_lake for lake in alg.lakes])
    print([lake.spill_points for lake in alg.lakes])
    print([lake.outflow_points for lake in alg.lakes])

if __name__ == "__main__":
    unittest.main()
