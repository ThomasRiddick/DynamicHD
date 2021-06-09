'''
Test river direction determination algorithm
Created on February 26, 2019

@author: thomasriddick
'''
import unittest
from Dynamic_HD_Scripts.interface.cpp_interface.libs \
    import determine_river_directions_wrapper
import numpy as np

class TestRiverDirectionDeterminationDriver(unittest.TestCase):

    def testDetermineRiverDirectionsOne(self):
        nlat = 8
        nlon = 8
        orography = np.array([[10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0],
                              [10.0,10.0, 9.8,10.0,  0.0, 3.0,10.0,10.0],
                              [10.0, 9.0,10.0,10.0, 10.0,10.0, 4.0,10.0],
                              [10.0, 8.3,10.0,10.0, 10.0,10.0, 4.1,10.0],
                              [10.0, 7.3,10.0,10.0, 10.0,10.0, 4.8,10.0],
                              [10.0,10.0, 7.1, 6.0,  5.0, 4.9,10.0,10.0],
                              [10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0],
                              [10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0]],
                              dtype=np.float64)
        lsmask = np.array([[False,False,False,False, False,False,False,False],
                           [False,False,False,False,  True,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False]],
                           dtype=np.int32)
        truesinks = np.array([[False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False]],
                              dtype=np.int32)
        expected_rdirs_out = np.array([[6,   3,   2,   3,    2,   1,   1,   6],
                                       [3,   2,   1,   6,    0,   4,   4,   1],
                                       [3,   2,   1,   9,    8,   7,   7,   4],
                                       [3,   2,   1,   6,    9,   9,   8,   7],
                                       [6,   3,   3,   3,    3,   9,   8,   7],
                                       [9,   6,   6,   6,    6,   9,   8,   7],
                                       [6,   9,   9,   9,    9,   8,   7,   6],
                                       [9,   8,   7,   4,    4,   6,   9,   9]],
                                       dtype=np.float64)
        rdirs = np.zeros((8,8),dtype=np.float64)
        always_flow_to_sea_in = True
        use_diagonal_nbrs_in = True
        mark_pits_as_true_sinks_in = False
        determine_river_directions_wrapper.determine_river_directions(rdirs,
                                                                      orography,
                                                                      lsmask,
                                                                      truesinks,
                                                                      always_flow_to_sea_in,
                                                                      use_diagonal_nbrs_in,
                                                                      mark_pits_as_true_sinks_in)
        np.testing.assert_array_equal(rdirs,expected_rdirs_out)

    def testDetermineRiverDirectionsTwo(self):
        nlat = 8
        nlon = 8
        orography = np.array([[10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0],
                              [10.0,10.0, 9.8,10.0, -1.0, 3.0,10.0,10.0],
                              [10.0, 9.0,10.0,10.0, 10.0,10.0, 4.0,10.0],
                              [10.0, 8.3,10.0,10.0, 10.0,10.0, 4.1,10.0],
                              [10.0, 7.3,10.0,10.0, 10.0,10.0, 4.8,10.0],
                              [10.0,10.0, 7.1, 6.0,  5.0, 4.9,10.0,10.0],
                              [10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0],
                              [10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0]],
                              dtype=np.float64)
        lsmask = np.array([[False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False]],
                           dtype=np.int32)
        truesinks = np.array([[False,False,False,False, False,False,False,False],
                              [False,False,False,False,  True,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False]],
                              dtype=np.int32)
        expected_rdirs_out = np.array([[6,   3,   2,   3,    2,   1,   1,   6],
                                       [3,   2,   1,   6,    5,   4,   4,   1],
                                       [3,   2,   1,   9,    8,   7,   7,   4],
                                       [3,   2,   1,   6,    9,   9,   8,   7],
                                       [6,   3,   3,   3,    3,   9,   8,   7],
                                       [9,   6,   6,   6,    6,   9,   8,   7],
                                       [6,   9,   9,   9,    9,   8,   7,   6],
                                       [9,   8,   7,   4,    4,   6,   9,   9]],
                                       dtype=np.float64)
        rdirs = np.zeros((8,8),dtype=np.float64)
        always_flow_to_sea_in = True
        use_diagonal_nbrs_in = True
        mark_pits_as_true_sinks_in = False
        determine_river_directions_wrapper.determine_river_directions(rdirs,
                                    orography,
                                    lsmask,
                                    truesinks,
                                    always_flow_to_sea_in,
                                    use_diagonal_nbrs_in,
                                    mark_pits_as_true_sinks_in)
        np.testing.assert_array_equal(rdirs,expected_rdirs_out)

    def testDetermineRiverDirectionsThree(self):
        nlat = 8
        nlon = 8
        orography = np.array([[10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0],
                              [0.0,  3.0, 10.0,10.0, 10.0,10.0, 9.8,10.0],
                              [10.0,10.0, 4.0,10.0, 10.0, 9.0,10.0,10.0],
                              [10.0,10.0, 4.1,10.0, 10.0, 8.3,10.0,10.0],
                              [10.0,10.0, 4.8,10.0, 10.0, 7.3,10.0,10.0],
                              [5.0, 4.9,10.0,10.0, 10.0,10.0, 7.1, 6.0],
                              [10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0],
                              [10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0]],
                              dtype=np.float64)
        lsmask = np.array([[False,False,False,False, False,False,False,False],
                           [True,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False]],
                           dtype=np.int32)
        truesinks = np.array([[False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False]],
                              dtype=np.int32)
        expected_rdirs_out = np.array([[2,   1,   1,   6,    6,   3,   2,   3],
                                       [0,   4,   4,   1,    3,   2,   1,   6],
                                       [8,   7,   7,   4,    3,   2,   1,   9],
                                       [4,   9,   8,   7,    3,   2,   1,   7],
                                       [3,   9,   8,   7,    6,   3,   3,   3],
                                       [6,   9,   8,   7,    9,   6,   6,   6],
                                       [9,   8,   7,   1,    3,   9,   9,   9],
                                       [8,   7,   4,   4,    6,   6,   6,   9]],
                                       dtype=np.float64)
        rdirs = np.zeros((8,8),dtype=np.float64)
        always_flow_to_sea_in = True
        use_diagonal_nbrs_in = True
        mark_pits_as_true_sinks_in = False
        determine_river_directions_wrapper.determine_river_directions(rdirs,
                                    orography,
                                    lsmask,
                                    truesinks,
                                    always_flow_to_sea_in,
                                    use_diagonal_nbrs_in,
                                    mark_pits_as_true_sinks_in)
        np.testing.assert_array_equal(rdirs,expected_rdirs_out)

    def testDetermineRiverDirectionsFour(self):
        nlat = 8
        nlon = 8
        orography = np.array([[10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0],
                              [2.0,  3.0, 10.0,10.0, 10.0,10.0, 9.8,10.0],
                              [10.0,10.0, 4.0,10.0, 10.0, 9.0,10.0,10.0],
                              [10.0,10.0, 4.1,10.0, 10.0, 8.3,10.0,10.0],
                              [10.0,10.0, 4.8,10.0, 10.0, 7.3,10.0,10.0],
                              [5.0, 4.9,10.0,10.0, 10.0,10.0, 7.1, 6.0],
                              [10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0],
                              [10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0]],
                              dtype=np.float64)
        lsmask = np.array([[False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False]],
                           dtype=np.int32)
        truesinks = np.array([[False,False,False,False, False,False,False,False],
                              [True,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False]],
                              dtype=np.int32)
        expected_rdirs_out = np.array([[2,   1,   1,   6,    6,   3,   2,   3],
                                       [5,   4,   4,   1,    3,   2,   1,   6],
                                       [8,   7,   7,   4,    3,   2,   1,   9],
                                       [4,   9,   8,   7,    3,   2,   1,   7],
                                       [3,   9,   8,   7,    6,   3,   3,   3],
                                       [6,   9,   8,   7,    9,   6,   6,   6],
                                       [9,   8,   7,   1,    3,   9,   9,   9],
                                       [8,   7,   4,   4,    6,   6,   6,   9]],
                                       dtype=np.float64)
        rdirs = np.zeros((8,8),dtype=np.float64)
        always_flow_to_sea_in = True
        use_diagonal_nbrs_in = True
        mark_pits_as_true_sinks_in = False
        determine_river_directions_wrapper.determine_river_directions(rdirs,
                                    orography,
                                    lsmask,
                                    truesinks,
                                    always_flow_to_sea_in,
                                    use_diagonal_nbrs_in,
                                    mark_pits_as_true_sinks_in)
        np.testing.assert_array_equal(rdirs,expected_rdirs_out)

    def testDetermineRiverDirectionsFive(self):
        nlat = 8
        nlon = 8
        orography = np.array([[10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0],
                              [2.0,  3.0, 10.0,10.0, 10.0,10.0, 9.8,10.0],
                              [10.0,10.0, 4.0,10.0, 10.0, 9.0,10.0,10.0],
                              [10.0,10.0, 4.1,10.0, 10.0, 8.3,10.0, 1.0],
                              [10.0,10.0, 4.8,10.0, 10.0, 7.3,10.0,10.0],
                              [5.0, 4.9,10.0,10.0, 10.0,10.0, 7.1, 6.0],
                              [10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0],
                              [10.0,-1.0,10.0, 1.2,  1.3,10.0,10.0,10.0]],
                              dtype=np.float64)
        lsmask = np.array([[False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False]],
                           dtype=np.int32)
        truesinks = np.array([[False,False,False,False, False,False,False,False],
                              [True,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False],
                              [False,False,False,False, False,False,False,False]],
                              dtype=np.int32)
        expected_rdirs_out = np.array([[2,   1,   1,   6,    6,   3,   2,   3],
                                       [5,   4,   4,   1,    3,   2,   1,   6],
                                       [1,   7,   7,   4,    3,   2,   3,   2],
                                       [4,   9,   8,   7,    3,   2,   6,   5],
                                       [7,   9,   8,   7,    6,   3,   9,   8],
                                       [6,   9,   8,   7,    9,   6,   6,   6],
                                       [3,   2,   1,   2,    1,   1,   9,   9],
                                       [6,   5,   4,   5,    4,   4,   6,   9]],
                                       dtype=np.float64)
        rdirs = np.zeros((8,8),dtype=np.float64)
        always_flow_to_sea_in = True
        use_diagonal_nbrs_in = True
        mark_pits_as_true_sinks_in = True
        determine_river_directions_wrapper.determine_river_directions(rdirs,
                                    orography,
                                    lsmask,
                                    truesinks,
                                    always_flow_to_sea_in,
                                    use_diagonal_nbrs_in,
                                    mark_pits_as_true_sinks_in)
        np.testing.assert_array_equal(rdirs,expected_rdirs_out)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
