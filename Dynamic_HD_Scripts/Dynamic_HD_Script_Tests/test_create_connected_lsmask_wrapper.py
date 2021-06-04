'''
Unit test of the wrapper for the connected ls mask creation c++ code

Created on May 25, 2016

@author: thomasriddick
'''

import unittest
import numpy as np
from Dynamic_HD_Scripts.libs import create_connected_lsmask_wrapper as cc_lsmask_wrapper  #@UnresolvedImport

class Test(unittest.TestCase):
    """Unit test object"""

    def setUp(self):
        """Unit test setup function. Prepare test data"""
        self.landsea_in = np.asarray([[0,0,1,0,1,1,0,1,0,1],
                                      [1,0,0,0,0,1,0,1,1,1],
                                      [0,0,0,1,1,1,1,1,1,0],
                                      [0,0,1,0,1,0,0,1,1,1],
                                      [0,0,0,1,0,0,0,0,0,1],
                                      [0,1,0,1,0,0,1,0,1,0],
                                      [0,1,0,0,0,0,0,1,0,0],
                                      [0,0,0,0,0,1,0,0,0,0],
                                      [1,1,0,0,1,1,1,0,1,0],
                                      [1,1,1,0,0,0,0,0,0,0]],
                                     dtype=np.int32, order='C')

        self.ocean_seed_points = np.asarray([[0,0,1,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,1,0],
                                             [0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0],
                                             [0,1,0,0,0,0,0,0,0,0],
                                             [0,1,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0,0],
                                             [1,0,0,0,0,0,0,0,0,0]],
                                     dtype=np.int32, order='C')

        self.expected_output_using_diagonals  = np.asarray([[0,0,1,0,1,1,0,1,0,1],
                                                            [1,0,0,0,0,1,0,1,1,1],
                                                            [0,0,0,1,1,1,1,1,1,0],
                                                            [0,0,1,0,1,0,0,1,1,1],
                                                            [0,0,0,1,0,0,0,0,0,1],
                                                            [0,1,0,1,0,0,1,0,1,0],
                                                            [0,1,0,0,0,0,0,1,0,0],
                                                            [0,0,0,0,0,0,0,0,0,0],
                                                            [1,1,0,0,0,0,0,0,0,0],
                                                            [1,1,1,0,0,0,0,0,0,0]],
                                                           dtype=np.int32, order='C')

        self.expected_output_not_using_diagonals  = np.asarray([[0,0,1,0,1,1,0,1,0,1],
                                                                [1,0,0,0,0,1,0,1,1,1],
                                                                [0,0,0,1,1,1,1,1,1,0],
                                                                [0,0,0,0,1,0,0,1,1,1],
                                                                [0,0,0,0,0,0,0,0,0,1],
                                                                [0,1,0,0,0,0,0,0,0,0],
                                                                [0,1,0,0,0,0,0,0,0,0],
                                                                [0,0,0,0,0,0,0,0,0,0],
                                                                [1,1,0,0,0,0,0,0,0,0],
                                                                [1,1,1,0,0,0,0,0,0,0]],
                                                               dtype=np.int32, order='C')


    def testCreateConnectedLsmaskWrapperUsingDiagonals(self):
        """Test creating a connected ls mask including diagonal connections"""
        cc_lsmask_wrapper.create_connected_ls_mask(self.landsea_in, #@UndefinedVariable
                                                   self.ocean_seed_points,
                                                   True)
        np.testing.assert_array_equal(self.landsea_in,
                                      self.expected_output_using_diagonals,
                                      "Creating a connected ls mask using diagonal doesn't produce"
                                      " expected results")

    def testCreateConnectedLsmaskWrapperWithoutUsingDiagonals(self):
        """Test creating a connected ls mask not including diagonal connections"""
        cc_lsmask_wrapper.create_connected_ls_mask(self.landsea_in, #@UndefinedVariable
                                                   self.ocean_seed_points,
                                                   False)
        np.testing.assert_array_equal(self.landsea_in,
                                      self.expected_output_not_using_diagonals,
                                      "Creating a connected ls mask not using diagonals doesn't produce"
                                      " expected results")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
