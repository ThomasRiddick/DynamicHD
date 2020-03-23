'''
Test stream following algorithm
Created on February 10, 2020

@author: thomasriddick
'''
import unittest
import Dynamic_HD_Scripts.libs.follow_streams_wrapper \
       as follow_streams_wrapper
import numpy as np

class TestFollowStreamsDriver(unittest.TestCase):

    def testFollowStreamsDriverOne(self):
        nlat = 16
        nlon = 16
        rdirs_in = np.array([[-1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1],
                             [-1, -1,  0,  0,  -1,  0, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1],
                             [-1, -1,  8,  8,   9, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1],
                             [-1, -1,  8,  8,   9, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1],
                             [-1, -1,  8,  4,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1],
                             [-1, -1, -1, -1,   7, -1,  1,  4,  -1, -1, -1, -1,   0, -1, -1, -1],
                             [-1, -1, -1, -1,  -1,  7,  2, -1,   7, -1, -1,  9,  -1, -1, -1, -1],
                             [-1, -1, -1, -1,  -1, -1,  2, -1,   8, -1,  9,  5,  -1, -1, -1, -1],
                             [-1, -1, -1, -1,  -1, -1,  2, -1,   8,  9,  0,  1,  -1, -1, -1, -1],
                             [-1, -1, -1, -1,  -1, -1,  2, -1,   6,  9,  0, -1,  -1, -1, -1, -1],
                             [-1, -1, -1, -1,  -1, -1,  2,  6,   6,  6,  6,  6,   0, -1, -1, -1],
                             [-1, -1, -1, -1,  -1, -1,  2,  3,  -1, -1, -1, -1,  -1, -1, -1, -1],
                             [-1, -1, -1, -1,  -1, -1,  2, -1,   3, -1, -1, -1,  -1, -1, -1, -1],
                             [-1, -1, -1, -1,  -1, -1,  0, -1,  -1,  3, -1, -1,  -1, -1, -1, -1],
                             [-1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1],
                             [-1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1]],
                             dtype=np.float64)
        cells_with_loop_in = np.array([
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False,  True,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False, True,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]],
            dtype=np.int32)
        downstream_cells_expected_out = np.array([
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False, True,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False, True,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False, True, True, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False,  True,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False, True,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]],
            dtype=np.int32)
        downstream_cells_out = np.zeros((nlat,nlon),dtype=np.int32)
        follow_streams_wrapper.follow_streams(rdirs_in,
                                              cells_with_loop_in,
                                              downstream_cells_out)
        np.testing.assert_array_equal(downstream_cells_out,
                                      downstream_cells_expected_out)

    def testFollowStreamsDriverTwo(self):
        nlat = 16
        nlon = 16
        rdirs_in = np.array([[-1, -1, -1, -1,  -1, -1, -1,  0,  -1, -1, -1, -1,  -1, -1, -1, -1],
                             [-1, -1,  0,  0,  -1,  0,  0,  8,  -1,  0, -1, -1,  -1, -1, -1, -1],
                             [-1, -1,  8,  8,   9,  8,  8,  8,   9,  2, -1, -1,  -1, -1, -1, -1],
                             [-1, -1,  8,  8,   8,  8,  8,  8,   8,  4, -1, -1,  -1, -1, -1, -1],
                             [-1, -1,  8,  4,   9,  8,  8,  8,   8, -1, -1, -1,  -1, -1, -1, -1],
                             [-1, -1, -1,  8,   7,  8,  1,  4,   8,  4,  4,  4,   0, -1,  1, -1],
                             [-1, -1, -1,  8,  -1,  7,  4, -1,   7, -1, -1,  9,  -1,  2, -1, -1],
                             [-1, -1, -1,  8,  -1, -1,  2,  7,   8, -1,  9,  5,  -1,  2, -1, -1],
                             [-1, -1, -1,  8,  -1, -1,  2,  8,   8,  9,  9,  1,  -1,  2, -1, -1],
                             [-1, -1, -1, -1,   7, -1,  2,  8,   6,  9,  8, -1,  -1,  2, -1, -1],
                             [-1, -1, -1, -1,  -1, -1,  2,  6,   6,  6,  6,  6,   6,  6,  0, -1],
                             [-1, -1, -1, -1,  -1, -1,  2,  3,   3, -1, -1, -1,  -1, -1, -1, -1],
                             [-1, -1, -1, -1,  -1, -1,  2, -1,   3,  3,  6,  6,   0, -1, -1, -1],
                             [-1, -1, -1, -1,  -1, -1,  0, -1,  -1,  3,  0, -1,  -1, -1, -1, -1],
                             [-1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1],
                             [-1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1]],
                             dtype=np.float64)
        cells_with_loop_in = np.array([
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False,  True,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False, True, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False, True, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]],
            dtype=np.int32)
        downstream_cells_expected_out = np.array([
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False, True,False, False,False,False,False,  True,False,False,False, False,False,False,False],
            [False,False, True,False, False,False,False,False,  True,False,False,False, False,False,False,False],
            [False,False, True, True, False,False,False,False,  True,False,False,False, False,False,False,False],
            [False,False,False,False,  True,False,False,False,  True,False,False,False, False,False,False,False],
            [False,False,False,False, False, True, True,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False, True, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False, True, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False, True, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False, True,  True, True, True, True,  True, True,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
            [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]],
            dtype=np.int32)
        downstream_cells_out = np.zeros((nlat,nlon),dtype=np.int32)
        follow_streams_wrapper.follow_streams(rdirs_in,
                                              cells_with_loop_in,
                                              downstream_cells_out)
        np.testing.assert_array_equal(downstream_cells_out,
                                      downstream_cells_expected_out)


if __name__ == "__main__":
    unittest.main()
