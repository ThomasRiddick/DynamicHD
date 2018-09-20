'''
Test various dynamic hydrological discharge utility functions
Created on April 16, 2018

@author: thomasriddick
'''
import unittest
import Dynamic_HD_Scripts.field as field
import Dynamic_HD_Scripts.libs.lake_operators_wrapper as lake_operators_wrapper  #@UnresolvedImport
import Dynamic_HD_Scripts.libs.fill_sinks_wrapper as fill_sinks_wrapper
import Dynamic_HD_Scripts.libs.evaluate_basins_wrapper as evaluate_basins_wrapper
import numpy as np

class TestLocalMinimaFindingDriver(unittest.TestCase):

    def testLocalMinimaFinding(self):
        orography_field = np.array([[10.0,10.0,10.0,-2.3,10.0, 10.0,10.0,10.0,10.0,10.0],
                                    [-1.0,10.0,10.0,10.0,10.0, 10.0,15.0,16.0,18.0,10.0],
                                    [-2.5,10.0,10.0,10.1,10.1, 10.1,14.0,-1.0,19.0,10.0],
                                    [10.0,10.0,10.0,10.1,10.0, 10.1,17.0,17.0,19.0,10.0],
                                    [10.0, 9.9,10.0,10.1,10.1, 10.1,17.0,11.0,17.0,10.0],

                                    [10.0,10.0,10.0,10.0,10.0, 10.0,19.0,18.0,15.0,10.0],
                                    [10.0,10.0, 5.0, 5.0, 5.0,  5.0,10.0,10.0,10.0,10.0],
                                    [10.0,10.0, 5.0, 5.0, 5.0,  5.0,10.0,10.0, 3.0, 3.0],
                                    [10.0,10.0,10.0, 5.0, 5.0,  5.0,10.0,10.0,10.0,10.0],
                                    [10.0,-1.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,10.0]])
        expected_minima = np.array([[False,False,False, True,False,  True, True, True, True,False],
                                    [False,False,False,False,False,  True,False,False,False,False],
                                    [ True,False, True,False,False, False,False, True,False,False],
                                    [False,False,False,False, True, False,False,False,False,False],
                                    [False, True,False,False,False, False,False, True,False, True],

                                    [False,False,False,False,False, False,False,False,False, True],
                                    [False,False, True, True, True,  True,False,False,False,False],
                                    [False,False, True, True, True,  True,False,False, True, True],
                                    [False,False,False, True, True,  True,False,False,False,False],
                                    [False, True,False,False,False, False,False, True, True, True]])
        orography = field.Field(orography_field,grid="LatLong",nlat=10,nlong=10)
        local_minima = orography.find_all_local_minima()
        np.testing.assert_array_equal(expected_minima,local_minima.get_data())

    def testLocalMinimaFindingSinglePointTest(self):
        orography_field = np.array([[10.0,10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,10.0],
                                    [10.0,10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,10.0],
                                    [10.0,10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,10.0],
                                    [10.0,10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,10.0],
                                    [10.0,10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,10.0],

                                    [10.0,10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,10.0],
                                    [10.0,10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,10.0],
                                    [10.0,10.0,10.0,10.0,10.0, 10.0,10.0, 5.0,10.0,10.0],
                                    [10.0,10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,10.0],
                                    [10.0,10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,10.0]])
        expected_minima = np.array([[ True, True, True, True, True,  True, True, True, True, True],
                                    [ True, True, True, True, True,  True, True, True, True, True],
                                    [ True, True, True, True, True,  True, True, True, True, True],
                                    [ True, True, True, True, True,  True, True, True, True, True],
                                    [ True, True, True, True, True,  True, True, True, True, True],

                                    [ True, True, True, True, True,  True, True, True, True, True],
                                    [ True, True, True, True, True,  True,False,False,False, True],
                                    [ True, True, True, True, True,  True,False, True,False, True],
                                    [ True, True, True, True, True,  True,False,False,False, True],
                                    [ True, True, True, True, True,  True, True, True, True, True]])
        orography = field.Field(orography_field,grid="LatLong",nlat=10,nlong=10)
        local_minima = orography.find_all_local_minima()
        np.testing.assert_array_equal(expected_minima,local_minima.get_data())

class TestBurningCarvedRivers(unittest.TestCase):

    def testBurningCarvedRiversOne(self):
        nlat = 8;
        nlong = 8;
        orography = np.array([[1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                              [1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0],
                              [1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0],
                              [1.0,1.5,2.5,1.2, 1.3,2.5,1.5,1.0],
                              [1.0,1.5,2.5,1.3, 1.3,2.3,1.4,1.0],
                              [1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0],
                              [1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0],
                              [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0]],dtype=np.float64)
        rdirs = np.array([[7.0,8.0,8.0,8.0, 8.0,8.0,8.0,9.0],
                          [4.0,7.0,8.0,8.0, 7.0,9.0,7.0,6.0],
                          [4.0,7.0,3.0,2.0, 1.0,9.0,7.0,6.0],
                          [4.0,1.0,4.0,3.0, 3.0,3.0,6.0,6.0],
                          [7.0,7.0,4.0,6.0, 6.0,6.0,9.0,6.0],
                          [4.0,1.0,1.0,2.0, 1.0,6.0,6.0,9.0],
                          [4.0,2.0,2.0,3.0, 1.0,2.0,3.0,3.0],
                          [1.0,2.0,2.0,2.0, 3.0,2.0,3.0,3.0]],dtype=np.float64)
        minima = np.array([[False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False, True, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False]],dtype=np.int32)
        expected_orography_out = np.array([[1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                                           [1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0],
                                           [1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0],
                                           [1.0,1.5,2.5,1.2, 1.3,2.5,1.5,1.0],
                                           [1.0,1.5,2.5,1.3, 1.2,1.2,1.2,1.0],
                                           [1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0],
                                           [1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0],
                                           [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0]])
        lakemask = np.zeros((nlat,nlong),dtype=np.int32)
        self.assertFalse(np.array_equal(orography,expected_orography_out))
        lake_operators_wrapper.burn_carved_rivers(orography,rdirs,minima,lakemask)
        np.testing.assert_array_equal(orography,expected_orography_out)

    def testBurningCarvedRiversTwo(self):
        nlat = 8;
        nlong = 8;
        orography = np.array([[1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                              [1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0],
                              [1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0],
                              [1.0,1.5,2.5,1.2, 1.3,2.5,1.5,1.0],
                              [1.0,1.5,2.5,1.3, 1.4,2.5,1.4,1.0],
                              [1.0,1.5,2.5,1.4, 2.5,2.5,1.5,1.0],
                              [1.0,1.5,1.5,1.4, 1.5,1.5,1.5,1.0],
                              [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0]],
                              dtype=np.float64)
        rdirs = np.array([[7.0,8.0,8.0,8.0, 8.0,8.0,8.0,9.0],
                          [4.0,7.0,8.0,8.0, 7.0,9.0,7.0,6.0],
                          [4.0,7.0,3.0,2.0, 1.0,1.0,7.0,6.0],
                          [4.0,1.0,3.0,2.0, 3.0,4.0,6.0,6.0],
                          [7.0,7.0,3.0,2.0, 1.0,7.0,9.0,6.0],
                          [4.0,1.0,3.0,2.0, 1.0,7.0,6.0,9.0],
                          [4.0,2.0,2.0,1.0, 1.0,2.0,3.0,3.0],
                          [1.0,2.0,2.0,2.0, 3.0,2.0,3.0,3.0]],
                          dtype=np.float64)
        minima = np.array([[False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False, True,  False,False,False,False],
                           [False,False,False,False,  False,False,False,False],
                           [False,False,False,False,  False,False,False,False],
                           [False,False,False,False,  False,False,False,False],
                           [False,False,False,False,  False,False,False,False]],
                           dtype=np.int32)
        expected_orography_out = np.array([[1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                                           [1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0],
                                           [1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0],
                                           [1.0,1.5,2.5,1.2, 1.3,2.5,1.5,1.0],
                                           [1.0,1.5,2.5,1.2, 1.4,2.5,1.4,1.0],
                                           [1.0,1.5,2.5,1.2, 2.5,2.5,1.5,1.0],
                                           [1.0,1.5,1.5,1.2, 1.5,1.5,1.5,1.0],
                                           [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0]],
                                           dtype=np.float64)
        lakemask = np.zeros((nlat,nlong),dtype=np.int32)
        self.assertFalse(np.array_equal(orography,expected_orography_out))
        lake_operators_wrapper.burn_carved_rivers(orography,rdirs,minima,lakemask)
        np.testing.assert_array_equal(orography,expected_orography_out)

    def testBurningCarvedRiversThree(self):
        nlat = 8;
        nlong = 8;
        orography = np.array([[1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                              [1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0],
                              [1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0],
                              [0.5,1.5,1.3,1.2, 1.3,2.5,1.5,1.0],
                              [1.0,1.1,2.5,0.8, 0.7,2.5,1.4,1.0],
                              [1.0,1.5,2.5,1.8, 2.5,2.5,1.5,1.0],
                              [1.0,1.5,1.5,1.4, 1.5,1.5,1.5,1.0],
                              [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0]],
                              dtype=np.float64)
        rdirs = np.array([[7.0,8.0,8.0,8.0, 8.0,8.0,8.0,9.0],
                          [4.0,7.0,8.0,8.0, 7.0,9.0,7.0,6.0],
                          [4.0,1.0,2.0,1.0, 1.0,1.0,7.0,6.0],
                          [4.0,4.0,1.0,4.0, 3.0,1.0,6.0,6.0],
                          [7.0,7.0,4.0,7.0, 4.0,4.0,9.0,6.0],
                          [4.0,1.0,3.0,2.0, 1.0,7.0,6.0,9.0],
                          [4.0,2.0,2.0,1.0, 1.0,2.0,3.0,3.0],
                          [1.0,2.0,2.0,2.0, 3.0,2.0,3.0,3.0]],
                          dtype=np.float64)

        minima = np.array([[False,False,False,False, False,False,False,False,
                            False,False,False,False, False,False,False,False,
                            False,False,False,False, False,False,False,False,
                            False,False,False,False,  False,False,False,False,
                            False,False,False,False,  True,False,False,False,
                            False,False,False,False,  False,False,False,False,
                            False,False,False,False,  False,False,False,False,
                            False,False,False,False,  False,False,False,False]],
                            dtype=np.int32)
        expected_orography_out = np.array([[1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                                           [1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0],
                                           [1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0],
                                           [0.5,1.5,0.7,1.2, 1.3,2.5,1.5,1.0],
                                           [1.0,0.7,2.5,0.7, 0.7,2.5,1.4,1.0],
                                           [1.0,1.5,2.5,1.8, 2.5,2.5,1.5,1.0],
                                           [1.0,1.5,1.5,1.4, 1.5,1.5,1.5,1.0],
                                           [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0]],
                                           dtype=np.float64)
        lakemask = np.zeros((nlat,nlong),dtype=np.int32)
        self.assertFalse(np.array_equal(orography,expected_orography_out))
        lake_operators_wrapper.burn_carved_rivers(orography,rdirs,minima,lakemask)
        np.testing.assert_array_equal(orography,expected_orography_out)

    def testBurningCarvedRiversFour(self):
        nlat = 8;
        nlong = 8;
        orography = np.array([[1.0,1.0,1.0,1.0, 0.7,1.0,1.0,1.0],
                              [1.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.0],
                              [1.0,1.5,2.5,2.5, 2.5,1.2,1.5,1.0],
                              [1.0,1.5,1.4,0.7, 2.5,1.3,1.5,1.0],
                              [1.0,1.5,2.5,0.9, 1.0,2.5,1.4,1.0],
                              [1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0],
                              [1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0],
                              [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0]],
                             dtype=np.float64)
        rdirs = np.array([[7.0,8.0,8.0,8.0, 8.0,8.0,8.0,9.0],
                          [4.0,7.0,8.0,9.0, 8.0,8.0,7.0,6.0],
                          [4.0,1.0,2.0,9.0, 8.0,7.0,9.0,6.0],
                          [4.0,4.0,6.0,3.0, 9.0,8.0,6.0,6.0],
                          [7.0,7.0,4.0,6.0, 9.0,8.0,9.0,6.0],
                          [4.0,1.0,3.0,9.0, 8.0,7.0,6.0,9.0],
                          [4.0,2.0,2.0,1.0, 1.0,2.0,3.0,3.0],
                          [1.0,2.0,2.0,2.0, 3.0,2.0,3.0,3.0]],
                         dtype=np.float64)
        minima = np.array([[False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False, True, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False]],
                          dtype=np.int32)
        expected_orography_out = np.array([[1.0,1.0,1.0,1.0, 0.7,1.0,1.0,1.0],
                                           [1.0,1.5,1.5,1.5, 0.7,1.5,1.5,1.0],
                                           [1.0,1.5,2.5,2.5, 2.5,0.7,1.5,1.0],
                                           [1.0,1.5,1.4,0.7, 2.5,0.7,1.5,1.0],
                                           [1.0,1.5,2.5,0.9, 0.7,2.5,1.4,1.0],
                                           [1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0],
                                           [1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0],
                                           [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0]],
                                           dtype=np.float64)
        lakemask = np.zeros((nlat,nlong),dtype=np.int32)
        self.assertFalse(np.array_equal(orography,expected_orography_out))
        lake_operators_wrapper.burn_carved_rivers(orography,rdirs,minima,lakemask)
        np.testing.assert_array_equal(orography,expected_orography_out)

    def testBurningCarvedRiversFive(self):
        orography = np.array([[1.0,1.0,1.0,1.0, 0.7,1.0,1.0,1.0],
                              [1.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.0],
                              [1.0,1.5,2.5,2.5, 2.5,1.2,1.5,1.0],
                              [1.0,1.5,1.4,0.7, 2.5,1.3,1.5,1.0],
                              [1.0,1.5,2.5,0.9, 1.0,2.5,1.4,1.0],
                              [1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0],
                              [1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0],
                              [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0]],
                             dtype=np.float64)
        rdirs = np.array([[7.0,8.0,8.0,8.0, 8.0,8.0,8.0,9.0],
                          [4.0,7.0,8.0,9.0, 8.0,8.0,7.0,6.0],
                          [4.0,1.0,2.0,9.0, 8.0,7.0,9.0,6.0],
                          [4.0,4.0,6.0,3.0, 9.0,8.0,6.0,6.0],
                          [7.0,7.0,4.0,6.0, 9.0,8.0,9.0,6.0],
                          [4.0,1.0,3.0,9.0, 8.0,7.0,6.0,9.0],
                          [4.0,2.0,2.0,1.0, 1.0,2.0,3.0,3.0],
                          [1.0,2.0,2.0,2.0, 3.0,2.0,3.0,3.0]],
                         dtype=np.float64)
        minima = np.array([[False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False, True, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False],
                           [False,False,False,False, False,False,False,False]],
                          dtype=np.int32)
        expected_orography_out = np.array([[1.0,1.0,1.0,1.0, 0.7,1.0,1.0,1.0],
                                           [1.0,1.5,1.5,1.5, 0.7,1.5,1.5,1.0],
                                           [1.0,1.5,2.5,2.5, 2.5,1.2,1.5,1.0],
                                           [1.0,1.5,1.4,0.7, 2.5,0.7,1.5,1.0],
                                           [1.0,1.5,2.5,0.9, 1.0,2.5,1.4,1.0],
                                           [1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0],
                                           [1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0],
                                           [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0]],
                                          dtype=np.float64)
        lakemask = np.array([[False,False,False,False, False,False,False,False],
                             [False,False,False,False, False,False,False,False],
                             [False,False,False,False, False, True,False,False],
                             [False,False,False, True, False,False,False,False],
                             [False,False,False, True, True, False,False,False],
                             [False,False,False,False, False,False,False,False],
                             [False,False,False,False, False,False,False,False],
                             [False,False,False,False, False,False,False,False]],
                            dtype=np.int32)
        self.assertFalse(np.array_equal(orography,expected_orography_out))
        lake_operators_wrapper.burn_carved_rivers(orography,rdirs,minima,lakemask)
        np.testing.assert_array_equal(orography,expected_orography_out)

    def testBurningCarvedRiversSix(self):
        nlat = 16;
        nlong = 16;
        orography = \
        np.array([[0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
                  [0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,0.4,0.8, 2.0,2.0,1.0,2.0, 1.0,1.0,0.3,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 0.8,1.0,2.0,0.8, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,0.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,0.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.2,0.2,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,0.1,1.0, 1.0,0.5,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,2.0,1.0,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,2.0,1.0,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,2.0,1.0,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,1.5,1.5, 1.5,1.5,1.2,1.5, 1.5,1.1,1.5,1.5, 1.5,1.5,1.5,0.0],
                  [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0]],
                 dtype=np.float64)

        rdirs = np.zeros((nlat,nlong),dtype=np.float64)

        minima = np.array(
        [[False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False, True,False, False,False,False,False, False,False, True,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, True, False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False, True,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False, True,False, False,False,False,False, False,False, True,False, False, True,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False, True,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]]
         ,dtype=np.int32)
        expected_orography_out = \
        np.array([[0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
                  [0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,0.4,0.4, 2.0,2.0,0.2,2.0, 0.2,0.2,0.2,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 0.4,0.2,2.0,0.8, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,0.2,2.0,2.0, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,0.2,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,0.2,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,0.2,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,0.2,0.2,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,0.1,0.5, 0.5,0.5,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,2.0,0.2,2.0, 2.0,0.1,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,2.0,0.2,2.0, 2.0,0.1,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,2.0,2.0, 2.0,2.0,0.2,2.0, 2.0,0.1,2.0,2.0, 2.0,2.0,1.5,0.0],
                  [0.0,1.5,1.5,1.5, 1.5,1.5,0.2,1.5, 1.5,0.1,1.5,1.5, 1.5,1.5,1.5,0.0],
                  [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0]],
                  dtype=np.float64)

        lakemask = np.array(
        [[False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False, True, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False, True,False,False, False,False,False,False, False,False,False,False],
         [False,False, True,False, False, True,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
         [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]]
         ,dtype=np.int32)
        truesinks = np.zeros((nlat,nlong),dtype=np.int32)
        landsea = np.array([
        [True,True,True,True,    True,True,True,True,     True,True,True,True,        True,True,True,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,True,True,True,    True,True,True,True,     True,True,True,True,        True,True,True,True]],
        dtype=np.int32)
        empty = np.zeros((nlat,nlong),dtype=np.int32)
        fill_sinks_wrapper.fill_sinks_cpp_func(orography,4,True,landsea,False,True,truesinks,
                                               False,0.0,empty,empty,rdirs,empty,False)
        self.assertFalse(np.array_equal(orography,expected_orography_out))
        lake_operators_wrapper.burn_carved_rivers(orography,rdirs,minima,lakemask)
        np.testing.assert_array_equal(orography,expected_orography_out)

    def testBurningCarvedRiversSeven(self):
        nlat = 16;
        nlong = 16;
        orography = np.array([
        [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
        [0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,0.4,0.8, 2.0,2.0,1.0,2.0, 1.0,1.0,0.3,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 0.8,1.0,2.0,0.8, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,0.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,0.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.2,0.2,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,0.1,1.0, 1.0,0.5,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,2.0,1.0,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,2.0,1.0,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,2.0,1.0,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,1.5,1.5, 1.5,1.5,1.2,1.5, 1.5,1.1,1.5,1.5, 1.5,1.5,1.5,0.0],
        [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0]],
        dtype=np.float64)


        rdirs = np.zeros((nlat,nlong),dtype=np.float64)

        minima = np.array([
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False, True,False, False,False,False,False, False,False, True,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, True, False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False, True,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False, True,False, False,False,False,False, False,False, True,False, False, True,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False, True,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]],
        dtype=np.int32)

        expected_orography_out = np.array([
        [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
        [0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,0.4,0.4, 2.0,2.0,1.0,2.0, 1.0,1.0,0.2,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 0.4,1.0,2.0,0.8, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,0.4,2.0,2.0, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,0.4,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,0.4,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,0.4,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,0.2,0.2,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,0.1,0.5, 0.5,0.5,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,2.0,0.4,2.0, 2.0,0.1,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,2.0,0.4,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,2.0,2.0, 2.0,2.0,0.4,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0],
        [0.0,1.5,1.5,1.5, 1.5,1.5,1.2,1.5, 1.5,1.1,1.5,1.5, 1.5,1.5,1.5,0.0],
        [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0]],
        dtype=np.float64)

        lakemask = np.array([
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False, True, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False, True,False,False, False,False,False,False, False,False,False,False],
        [False,False, True,False, False, True,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
        [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]],
        dtype=np.int32)

        truesinks = np.zeros((nlat,nlong),dtype=np.int32)
        landsea = np.array([
        [True,True,True,True,    True,True,True,True,     True,True,True,True,     True,True,True,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
        [True,True,True,True,    True,True,True,True,     True,True,True,True,     True,True,True,True]],
        dtype=np.int32)
        empty = np.zeros((nlat,nlong),dtype=np.int32)
        fill_sinks_wrapper.fill_sinks_cpp_func(orography,4,True,landsea,False,True,truesinks,
                                               False,0.0,empty,empty,rdirs,empty,False)
        rdirs[13,9] = 5
        rdirs[13,6] = 0
        rdirs[3,10] = -1
        rdirs[10,11] = 5
        rdirs[11,9] = -1
        self.assertFalse(np.array_equal(orography,expected_orography_out))
        lake_operators_wrapper.burn_carved_rivers(orography,rdirs,minima,lakemask)
        np.testing.assert_array_equal(orography,expected_orography_out)

class TestConnectedAreaReduction(unittest.TestCase):

    def testConnectedAreaReductionOne(self):
      nlat = 16
      nlon = 16
      areas = \
        np.array([[True, False, True,False, False,False,False,False, False,True, True, False, False,True, False,False],
                  [True,  True,False,False, False,False,False,False, False,False,False,False, True, False, True,False],
                  [False,False, True,False, False,False,False,False, False,False,False,False, False, True,False,False],
                  [False, True,False, True, False,False,False,False, False,False,False,False, False,False, True,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False, True],
                  [False,False,False,False, False,False,False, True, False,False,False,False, False,False,False,False],
                  [True, False,False,False, True,  True, True,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False, True, True,False, False,False,False,False, False,False,False, True],
                  [False, True,False,False, False,False,False,False,  True,False,False,True, False,False,False,False],
                  [True,False,True, False, False,False,False,False, False, True,False,True, False,False,False,False],
                  [True,False,False, True, False,False,False,False, False, True,False,True, False,False,True ,False],
                  [False, True,False, True, False,False,False,False, False,False,True,False, False,False,False,False],
                  [False,False,False,False, False,True, False,False, False,False,False,False, True, False,False,True],
                  [True, False,False,False, False,False, True,False, False,False,False,False, True, False,False,True],
                  [False,False,False,False, False,False,False, True, False,False,False,False, True, False,False,True],
                  [True,False,False,False, False, True, True,False, False,True, True ,False, True, True, True, False]],
                  dtype=np.int32)
      expected_areas_out = \
        np.array([[True, False,False,False, False,False,False,False, False,True, False,False, False,True, False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False, True, False,False,False,False, False,False,False,False],
                  [True, False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False, True,False,False, False,False,False,False,  True,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,True ,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,True, False,False, False,False,False,False, True, False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,True, False,False, False,False,False,False]],
                                     dtype=np.int32)
      lake_operators_wrapper.reduce_connected_areas_to_points(areas,True)
      np.testing.assert_array_equal(areas,expected_areas_out)

    def testConnectedAreaReductionTwo(self):
      nlat = 16
      nlon = 16
      areas = \
        np.array([[False,False,False,False, False,False,False,False, False,False,True, False, False,False,False,False],
                  [False,True, True, True,  False, True,False,False, False,True, False,True,  False,False,False,False],
                  [False,True, False,False, False,False,True, False, False,True, False,True,  False,False,False,False],
                  [False,True, False,False, False,False,True, False, False,False,True, False, False,False,False,False],
                  [False,True, False,False, False,False,True, False, False,False,False,False, False,True, True, True],
                  [False,True, False,False, False,False,True, False, False,False,False,False, False,True, False,True],
                  [False,True, True, True,  True, True, False,False, False,False,False,False, False,True, False,True],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,True, True, True],
                  [False,False,False,False, False,False,False,False, False,True, True, True,  False,False,False,False],
                  [False,False,False,False, False,False,False,False, True, False,False,True,  False,False,False,False],
                  [False,False,False,False, False,False,False,False, True, False,False,True,  False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,True,  False,False,False,False],
                  [True, True, True, False, True, True, True, True,  False,False,False,False, False,False,True, False],
                  [True, False,False,False, False,False,False,True,  False,False,False,False, True, False,True, False],
                  [True, False,False,False, False,False,False,True,  False,False,False,False, True, False,True, False],
                  [False,True, True, False, True, True, True, True,  False,False,False,False, False,True, False,False]],
                        dtype=np.int32)
      expected_areas_out = \
        np.array([[False,False,False,False, False,False,False,False, False,False,True, False, False,False,False,False],
                  [False,True, False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,True, False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,True, False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [True, False,False,False, True, False,False,False, False,False,False,False, False,False,True, False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]],
                                     dtype=np.int32)
      lake_operators_wrapper.reduce_connected_areas_to_points(areas,True)
      np.testing.assert_array_equal(areas,expected_areas_out)

    def testConnectedAreaReductionThree(self):
      nlat = 16
      nlon = 16
      areas = \
        np.array([[False,False,False,False, False,True, True, False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, True, False,False,True,  False,False,False,False, False,False,False,False],
                  [False,False,False,False, True, False,False,True,  False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,True, True, False, False,False,False,False, False,False,False,False],
                  [True, False,False,False, False,False,False,False, True, True, True, True,  False,False,False,False],
                  [False,False,False,False, False,False,False,False, True, False,False,True,  False,False,False,True],
                  [True, False,False,False, False,False,False,False, True, False,False,True,  False,False,False,False],
                  [False,False,False,False, False,False,False,False, True, True, True, True,  False,False,False,False],
                  [False,False,False,False, False,True, False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, True, True, True, False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,True, False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, True, False,True, False, False,False,False,False],
                  [True, False,False,False, False,False,False,False, False,True, False,False, False,False,False,True],
                  [False,False,False,False, False,False,False,False, True, False,True, False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]],
                  dtype=np.int32)
      expected_areas_out = \
        np.array([[False,False,False,False, False,True, False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, True, False,False,True,  False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,True, False,False, False,False,False,False, False,False,False,False],
                  [True, False,False,False, False,False,False,False, True, False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [True, False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,True, False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, True, False,True, False, False,False,False,False],
                  [True, False,False,False, False,False,False,False, False,True, False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, True, False,True, False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]],
                                     dtype=np.int32)
      lake_operators_wrapper.reduce_connected_areas_to_points(areas,False)
      np.testing.assert_array_equal(areas,expected_areas_out)

    def testConnectedAreaReductionFour(self):
      nlat = 16
      nlon = 16
      areas = \
        np.array([[False,False,False,False, False,True, True, False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, True, False,False,True,  False,False,False,False, False, True, True,False],
                  [False,False,False,False, True, False,False,True,  False,False,False,False, False, True, True,False],
                  [False,False,False,False, False,True, True, False, False,False,False,False, False,False,False,False],
                  [True, False,False,False, False,False,False,False, True, True, True, True,  False,False,False,False],
                  [False,False,False,False, False,False,False,False, True, False,False,True,  False,False,False,True],
                  [True, False,False,False, False,False,False,False, True, False,False,True,  False,False,False,False],
                  [False,False,False,False, False,False,False,False, True, True, True, True,  False,False,False,False],
                  [False,False,False,False, False,True, False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, True, True, True, False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,True, False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, True, False,True, False, False,False,False,False],
                  [True, False,False,False, False,False,False,False, False,True, False,False, False,False,False,True],
                  [False,False,False,False, False,False,False,False, True, False,True, False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]],
                  dtype=np.int32)
      expected_areas_out = \
        np.array([[False,False,False,False, False, True,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False, True,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]],
                  dtype=np.int32)
      orography = np.array([[1.0,1.0,1.0,1.0, 1.0,0.1,0.1,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 0.1,1.0,1.0,0.1, 1.0,1.0,1.0,1.0, 1.0,0.5,0.5,1.0],
                            [1.0,1.0,1.0,1.0, 0.1,1.0,1.0,0.1, 1.0,1.0,1.0,1.0, 1.0,0.5,0.5,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,0.1,0.1,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                            [0.1,0.1,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,0.2,0.2,0.2, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,1.0,1.0,0.2, 1.0,1.0,1.0,0.1],
                            [0.1,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,1.0,1.0,0.2, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,0.2,0.2,0.2, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,0.1,1.0,1.0, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.1,1.0,0.1,1.0, 1.0,1.0,1.0,1.0],
                            [0.5,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,0.1,1.0,1.0, 1.0,1.0,1.0,0.5],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.1,1.0,0.1,1.0, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,0.1,1.0, 1.0,1.0,1.0,1.0]],
                            dtype=np.float64)
      lake_operators_wrapper.reduce_connected_areas_to_points(areas,True,orography,True)
      np.testing.assert_array_equal(areas,expected_areas_out)

    def testConnectedAreaReductionFive(self):
      nlat = 16
      nlon = 16
      areas = \
        np.array([[False,False,False,False, False,True, True, False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, True, False,False,True,  False,False,False,False, False, True, True,False],
                  [False,False,False,False, True, False,False,True,  False,False,False,False, False, True, True,False],
                  [False,False,False,False, False,True, True, False, False,False,False,False, False,False,False,False],
                  [True, False,False,False, False,False,False,False, True, True, True, True,  False,False,False,False],
                  [False,False,False,False, False,False,False,False, True, False,False,True,  False,False,False,True],
                  [True, False,False,False, False,False,False,False, True, False,False,True,  False,False,False,False],
                  [False,False,False,False, False,False,False,False, True, True, True, True,  False,False,False,False],
                  [False,False,False,False, False,True, False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, True, True, True, False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,True, False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, True, False,True, False, False,False,False,False],
                  [True, False,False,False, False,False,False,False, False,True, False,False, False,False,False,True],
                  [False,False,False,False, False,False,False,False, True, False,True, False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]],
                  dtype=np.int32)
      expected_areas_out = \
        np.array([[False,False,False,False, False, True,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False,  True,False,False, True, False,False,False,False, False, True,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False, True,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False, True],
                  [True,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False,  True,False, True,False, False,False,False,False],
                  [True,False,False,False, False,False,False,False, False, True,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False,  True,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]],
                  dtype=np.int32)
      orography = np.array([[1.0,1.0,1.0,1.0, 1.0,0.1,0.1,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 0.1,1.0,1.0,0.1, 1.0,1.0,1.0,1.0, 1.0,0.5,0.5,1.0],
                            [1.0,1.0,1.0,1.0, 0.1,1.0,1.0,0.1, 1.0,1.0,1.0,1.0, 1.0,0.5,0.5,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,0.1,0.1,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                            [0.1,0.1,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,0.2,0.2,0.2, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,1.0,1.0,0.2, 1.0,1.0,1.0,0.1],
                            [0.1,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,1.0,1.0,0.2, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,0.2,0.2,0.2, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,0.1,1.0,1.0, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.1,1.0,0.1,1.0, 1.0,1.0,1.0,1.0],
                            [0.5,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,0.1,1.0,1.0, 1.0,1.0,1.0,0.5],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.1,1.0,0.1,1.0, 1.0,1.0,1.0,1.0],
                            [1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,0.1,1.0, 1.0,1.0,1.0,1.0]],
                            dtype=np.float64)
      lake_operators_wrapper.reduce_connected_areas_to_points(areas,False,orography,True)
      np.testing.assert_array_equal(areas,expected_areas_out)

    def testLakeFillingOne(self):
      nlat = 16
      nlon = 16
      lake_minima = \
        np.array([[False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,True, False, False,False,False,False, False,False,False,False],
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
      lake_mask = \
        np.array([[False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [False,False,False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [False,False,False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [False,False,False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [False,False,False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [False,False,False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]],
                  dtype=np.int32)
      orography = np.array([[0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
                            [0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.5, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 0.6,0.2,0.2,0.3, 0.4,1.8,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 0.5,0.2,0.1,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 0.5,0.2,0.1,0.2, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 0.4,0.2,0.2,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 0.4,0.4,0.4,0.4, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,1.5,1.5, 1.5,1.5,1.2,1.5, 1.5,1.1,1.5,1.5, 1.5,1.5,1.5,0.0],
                            [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0]],
                            dtype=np.float64)
      expected_orography_out = np.array([[0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
                                         [0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,1.8,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,1.5,1.5, 1.5,1.5,1.2,1.5, 1.5,1.1,1.5,1.5, 1.5,1.5,1.5,0.0],
                                         [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0]],
                                         dtype=np.float64)
      lake_operators_wrapper.fill_lakes(lake_minima,lake_mask,orography,False)
      np.testing.assert_array_equal(orography,expected_orography_out)

    def testLakeFillingTwo(self):
      nlat = 16
      nlon = 16
      lake_minima = \
        np.array([[False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,True, False, False,False,False,False, False,False,False,False],
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
      lake_mask = \
        np.array([[False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [False,False,False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [False,False,False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [False,False,False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [False,False,False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [False,False,False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False]],
                  dtype=np.int32)
      orography = np.array([[0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
                            [0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.5, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 0.6,0.2,0.2,0.3, 0.4,1.8,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 0.5,0.2,0.1,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 0.5,0.2,0.1,0.2, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 0.4,0.2,0.2,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 0.4,0.4,0.4,0.4, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                            [0.0,1.5,1.5,1.5, 1.5,1.5,1.2,1.5, 1.5,1.1,1.5,1.5, 1.5,1.5,1.5,0.0],
                            [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0]],
                            dtype=np.float64)
      expected_orography_out = np.array([[0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0],
                                         [0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,1.8,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0],
                                         [0.0,1.5,1.5,1.5, 1.5,1.5,1.2,1.5, 1.5,1.1,1.5,1.5, 1.5,1.5,1.5,0.0],
                                         [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0]],
                                         dtype=np.float64)
      lake_operators_wrapper.fill_lakes(lake_minima,lake_mask,orography,True)
      np.testing.assert_array_equal(orography,expected_orography_out)

    def testLakeFillingThree(self):
      nlat = 16
      nlon = 16
      lake_minima = \
        np.array([[False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False, True],
                  [False,False,False,False, False,False,True, False, False,False,False,False, False,False,False,False],
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
      lake_mask = \
        np.array([[True, True, True, False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [True, True, True, True,  False,False,False,False, False,False,False,False, False,False,False,True],
                  [True, True, True, False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [True, True, False,False, True, True, True, True,  True, False,False,False, False,False,False,True],
                  [True, False,False,False, True, True, True, True,  True, False,False, True, False,False,False,True],
                  [True, False,False,False, True, True, True, True,  True, False, True,False, True, True, False,True],
                  [True, False,False,False, True, True, True, True,  True, False, True,False, False,True, True, True],
                  [True, False,False,False, True, True, True, True,  True, False,False,False, False,False,False,True],
                  [True, True, False,False, True, True, True, True,  True, False,False,False, False,False,False,True],
                  [False,True, True, False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,True, False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,True, True,  False,False,False,False, False,False,False,False, False,False,False,True]],
                  dtype=np.int32)
      orography = np.array([[-0.1,-0.0,0.3,2.0,2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1],
                            [0.2,0.1,0.2,0.5, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [0.2,0.5,0.3,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [0.2,0.2,2.0,2.0, 0.6,0.6,0.6,0.5, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [0.3,2.0,2.0,2.0, 0.6,0.2,0.2,0.3, 0.4,1.8,2.0,0.1, 2.0,2.0,2.0,0.2],
                            [0.3,2.0,2.0,2.0, 0.5,0.2,0.1,0.3, 0.4,2.0,0.2,2.0, 0.2,0.1,2.0,0.3],
                            [0.2,2.0,2.0,2.0, 0.5,0.2,0.1,0.2, 0.4,2.0,0.3,2.0, 2.0,0.2,0.1,0.2],
                            [0.1,2.0,2.0,2.0, 0.4,0.2,0.2,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.5],
                            [0.2,0.3,2.0,2.0, 0.4,0.4,0.4,0.4, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.3],
                            [2.0,0.1,0.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [2.0,2.0,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1],
                            [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3],
                            [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1],
                            [2.0,2.0,0.1,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [2.0,2.0,0.2,0.3, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3]],
                            dtype=np.float64)
      expected_orography_out = np.array([[1.7,1.7,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [1.7,1.7,1.7,1.7, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [1.7,1.7,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [1.7,1.7,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,1.8,2.0,1.7, 2.0,2.0,2.0,1.7],
                                         [1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,1.7,2.0, 1.7,1.7,2.0,1.7],
                                         [1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,1.7,2.0, 2.0,1.7,1.7,1.7],
                                         [1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [1.7,1.7,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [2.0,1.7,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [2.0,2.0,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [2.0,2.0,0.1,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [2.0,2.0,0.2,0.3, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7]],
                                         dtype=np.float64)
      lake_operators_wrapper.fill_lakes(lake_minima,lake_mask,orography,True)
      np.testing.assert_array_equal(orography,expected_orography_out)

    def testLakeFillingFour(self):
      nlat = 16
      nlon = 16
      lake_minima = \
        np.array([[False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False, True],
                  [False,False,False,False, False,False,True, False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,True, False, False,False,False,False, False,False,False,False, False,False,False,False]],
                              dtype=np.int32)
      lake_mask = \
        np.array([[True, True, True, False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [True, True, True, True,  False,False,False,False, False,False,False,False, False,False,False,True],
                  [True, True, True, False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [True, True, False,False, True, True, True, True,  True, False,False,False, False,False,False,True],
                  [True, False,False,False, True, True, True, True,  True, False,False, True, False,False,False,True],
                  [True, False,False,False, True, True, True, True,  True, False, True,False, True, True, False,True],
                  [True, False,False,False, True, True, True, True,  True, False, True,False, False,True, True, True],
                  [True, False,False,False, True, True, True, True,  True, False,False,False, False,False,False,True],
                  [True, True, False,False, True, True, True, True,  True, False,False,False, False,False,False,True],
                  [False,True, True, False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,True, False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,True, False,  False,False,False,False, False,False,False,False, False,False,False,True]],
                  dtype=np.int32)
      orography = np.array([[-0.1,-0.0,0.3,2.0,2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1],
                            [0.2,0.1,0.2,0.5, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [0.2,0.5,0.3,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [0.2,0.2,2.0,2.0, 0.6,0.6,0.6,0.5, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [0.3,2.0,2.0,2.0, 0.6,0.2,0.2,0.3, 0.4,1.8,2.0,0.1, 2.0,2.0,2.0,0.2],
                            [0.3,2.0,2.0,2.0, 0.5,0.2,0.1,0.3, 0.4,2.0,0.2,2.0, 0.2,0.1,2.0,0.3],
                            [0.2,2.0,2.0,2.0, 0.5,0.2,0.1,0.2, 0.4,2.0,0.3,2.0, 2.0,0.2,0.1,0.2],
                            [0.1,2.0,2.0,2.0, 0.4,0.2,0.2,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.5],
                            [0.2,0.3,2.0,2.0, 0.4,0.4,0.4,0.4, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.3],
                            [2.0,0.1,0.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [2.0,2.0,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1],
                            [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3],
                            [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1],
                            [2.0,2.0,0.1,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [2.0,2.0,0.2,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3]],
                            dtype=np.float64)
      expected_orography_out = np.array([[1.7,1.7,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [1.7,1.7,1.7,1.7, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [1.7,1.7,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [1.7,1.7,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,1.8,2.0,1.7, 2.0,2.0,2.0,1.7],
                                         [1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,1.7,2.0, 1.7,1.7,2.0,1.7],
                                         [1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,1.7,2.0, 2.0,1.7,1.7,1.7],
                                         [1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [1.7,1.7,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [2.0,1.7,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [2.0,2.0,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [2.0,2.0,0.3,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7],
                                         [2.0,2.0,0.3,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7]],
                                         dtype=np.float64)
      lake_operators_wrapper.fill_lakes(lake_minima,lake_mask,orography,True)
      np.testing.assert_array_equal(orography,expected_orography_out)

    def testLakeFillingSix(self):
      nlat = 16
      nlon = 16
      lake_minima = \
        np.array([[False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False, True],
                  [False,False,False,False, False,False,True, False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,True, False, False,False,False,False, False,False,False,False, False,False,False,False]],
                  dtype=np.int32)
      lake_mask = \
        np.array([[True, True, True, False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [True, True, True, True,  False,False,False,False, False,False,False,False, False,False,False,True],
                  [True, True, True, False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [True, True, False,False, True, True, True, True,  True, False,False,False, False,False,False,True],
                  [True, False,False,False, True, True, True, True,  True, False,False, True, False,False,False,True],
                  [True, False,False,False, True, True, True, True,  True, False, True,False, True, True, False,True],
                  [True, False,False,False, True, True, True, True,  True, False, True,False, False,True, True, True],
                  [True, False,False,False, True, True, True, True,  True, False,False,False, False,False,False,True],
                  [True, True, False,False, True, True, True, True,  True, False,False,False, False,False,False,True],
                  [False,True, True, False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,True, False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,True, False,  False,False,False,False, False,False,False,False, False,False,False,True]],
                  dtype=np.int32)
      orography = np.array([[-0.1,-0.0,0.3,2.0,2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1],
                            [0.2,0.1,0.2,0.5, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [0.2,0.5,0.3,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [0.2,0.2,2.0,2.0, 0.6,0.6,0.6,0.5, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [0.3,2.0,2.0,2.0, 0.6,0.2,0.2,0.3, 0.4,1.8,2.0,0.1, 2.0,2.0,2.0,0.2],
                            [0.3,2.0,2.0,2.0, 0.5,0.2,0.1,0.3, 0.4,2.0,0.2,2.0, 0.2,0.1,2.0,0.3],
                            [0.2,2.0,2.0,2.0, 0.5,0.2,0.1,0.2, 0.4,2.0,0.3,2.0, 2.0,0.2,0.1,0.2],
                            [0.1,2.0,2.0,2.0, 0.4,0.2,0.2,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.5],
                            [0.2,0.3,2.0,2.0, 0.4,0.4,0.4,0.4, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.3],
                            [2.0,0.1,0.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [2.0,2.0,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1],
                            [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3],
                            [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1],
                            [2.0,2.0,0.1,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [2.0,2.0,0.2,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3]],
                            dtype=np.float64)
      expected_orography_out = np.array([[0.7,0.7,0.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7],
                                         [0.7,0.7,0.7,0.7, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7],
                                         [0.7,0.7,0.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7],
                                         [0.7,0.7,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,2.0,0.7],
                                         [0.7,2.0,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,1.8,2.0,0.7, 2.0,2.0,2.0,0.7],
                                         [0.7,2.0,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,0.7,2.0, 0.7,0.7,2.0,0.7],
                                         [0.7,2.0,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,0.7,2.0, 2.0,0.7,0.7,0.7],
                                         [0.7,2.0,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,2.0,0.7],
                                         [0.7,0.7,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,2.0,0.7],
                                         [2.0,0.7,0.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7],
                                         [2.0,2.0,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7],
                                         [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7],
                                         [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7],
                                         [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7],
                                         [2.0,2.0,0.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7],
                                         [2.0,2.0,0.2,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7]],
                                         dtype=np.float64)
      lake_operators_wrapper.fill_lakes(lake_minima,lake_mask,orography,False)
      np.testing.assert_array_equal(orography,expected_orography_out)

    def testLakeFillingSeven(self):
      nlat = 16
      nlon = 16
      lake_minima = \
        np.array([[False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,True, False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,True, False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,True, False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,True, False, False,False,False,False, False,False,False,False, False,False,False,True]],
                  dtype=np.int32)
      lake_mask = \
        np.array([[True, True, True, False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [True, True, True, True,  False,False,True,False,  False,False,False,False, False,False,False,False],
                  [True, True, True, False, False,False,True,False,  False,False,False,False, False,False,False,False],
                  [True, True, False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [True, False,False,False, True, True, True, True,  True, False,False, True, False,False,False,False],
                  [True, False,False,False, True, True, True, True,  True, False, True,False, False,False,False,False],
                  [True, False,False,False, True, True, True, True,  True, False, True,False, False,False,False,False],
                  [True, False,False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [True, True, False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [False,True, True, False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,True, False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,True, False, False,False,False,False, False,False,False,False, False,False,False,True]],
                  dtype=np.int32)
      orography = np.array([[-0.1,-0.0,0.3,2.0,2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [0.2,0.1,0.2,0.5, 2.0,2.0,1.9,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [0.2,0.5,0.3,2.0, 2.0,2.0,1.9,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.4],
                            [0.2,0.2,2.0,2.0, 0.6,0.6,0.6,0.5, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [0.3,2.0,2.0,2.0, 0.6,0.2,0.2,0.3, 0.4,1.8,2.0,0.1, 2.0,2.0,2.0,2.0],
                            [0.3,2.0,2.0,2.0, 0.5,0.2,0.1,0.3, 0.4,2.0,0.2,2.0, 0.2,0.1,2.0,2.0],
                            [0.2,2.0,2.0,2.0, 0.5,0.2,0.1,0.2, 0.4,2.0,0.3,2.0, 2.0,0.2,0.1,2.0],
                            [0.1,2.0,2.0,2.0, 0.4,0.2,0.2,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [0.2,0.3,2.0,2.0, 0.4,0.4,0.4,0.4, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [2.0,0.1,0.7,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [2.0,2.0,1.7,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [2.0,2.0,0.1,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2],
                            [0.7,2.0,0.2,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3]],
                            dtype=np.float64)
      expected_orography_out = np.array([[1.4,1.4,1.4,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [1.4,1.4,1.4,1.4, 2.0,2.0,1.9,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [1.4,1.4,1.4,2.0, 2.0,2.0,1.9,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.4],
                                         [1.4,1.4,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,1.8,2.0,0.1, 2.0,2.0,2.0,2.0],
                                         [1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,0.2,2.0, 0.2,0.1,2.0,2.0],
                                         [1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,0.3,2.0, 2.0,0.2,0.1,2.0],
                                         [1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [1.4,1.4,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [2.0,1.4,1.4,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [2.0,2.0,1.7,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [2.0,2.0,0.3,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7],
                                         [0.7,2.0,0.3,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7]],
                                         dtype=np.float64)
      lake_operators_wrapper.fill_lakes(lake_minima,lake_mask,orography,True)
      np.testing.assert_array_equal(orography,expected_orography_out)

    def testLakeFillingEight(self):
      nlat = 16
      nlon = 16
      lake_minima = \
        np.array([[False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,True, False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,True, False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,True, False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,True, False, False,False,False,False, False,False,False,False],
                  [False,False,False,True,  False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,True, True,  False,True, False,False, False,False,False,False],
                  [False,False,False,False, False,False,True, True,  False,False,False,False, False,False,False,False],
                  [False,False,True, False, False,False,False,False, False,False,False,False, False,False,False,True]],
                  dtype=np.int32)
      lake_mask = \
        np.array([[True, True, True, False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [True, True, True, True,  False,False,True,False,  False,False,False,False, False,False,False,False],
                  [True, True, True, False, False,False,True,False,  False,False,False,False, False,False,False,False],
                  [True, True, False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [True, False,False,False, True, True, True, True,  True, False,False, True, False,False,False,False],
                  [True, False,False,False, True, True, True, True,  True, False, True,False, False,False,False,False],
                  [True, False,False,False, True, True, True, True,  True, False, True,False, False,False,False,False],
                  [True, False,False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [True, True, False,False, True, True, True, True,  True, False,False,False, False,False,False,False],
                  [False,True, True, False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,False,False, False,False,False,False, False,False,False,False, False,False,False,False],
                  [False,False,True, False, False,False,False,False, False,False,False,False, False,False,False,True],
                  [False,False,True, False, False,False,False,False, False,False,False,False, False,False,False,True]],
                  dtype=np.int32)
      orography = np.array([[-0.1,-0.0,0.3,2.0,2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [0.2,0.1,0.2,0.5, 2.0,2.0,1.9,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [0.2,0.5,0.3,2.0, 2.0,2.0,1.9,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.4],
                            [0.2,0.2,2.0,2.0, 0.6,0.6,0.6,0.5, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [0.3,2.0,2.0,2.0, 0.6,0.2,0.2,0.3, 0.4,1.8,2.0,0.1, 2.0,2.0,2.0,2.0],
                            [0.3,2.0,2.0,2.0, 0.5,0.2,0.1,0.3, 0.4,2.0,0.2,2.0, 0.2,0.1,2.0,2.0],
                            [0.2,2.0,2.0,2.0, 0.5,0.2,0.1,0.2, 0.4,2.0,0.3,2.0, 2.0,0.2,0.1,2.0],
                            [0.1,2.0,2.0,2.0, 0.4,0.2,0.2,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [0.2,0.3,2.0,2.0, 0.4,0.4,0.4,0.4, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [2.0,0.1,0.7,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [2.0,2.0,1.7,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                            [2.0,2.0,2.0,2.0, 2.0,2.0,0.2,0.1, 2.0,0.1,0.1,2.0, 2.0,2.0,2.0,2.0],
                            [2.0,2.0,0.1,2.0, 2.0,2.0,0.3,0.2, 2.0,0.2,0.2,2.0, 2.0,2.0,2.0,0.2],
                            [0.7,2.0,0.2,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3]],
                            dtype=np.float64)
      expected_orography_out = np.array([[1.4,1.4,1.4,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [1.4,1.4,1.4,1.4, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [1.4,1.4,1.4,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.4],
                                         [1.4,1.4,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,1.8,2.0,0.1, 2.0,2.0,2.0,2.0],
                                         [1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,0.2,2.0, 0.2,0.1,2.0,2.0],
                                         [1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,0.3,2.0, 2.0,0.2,0.1,2.0],
                                         [1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [1.4,1.4,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [2.0,1.4,1.4,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [2.0,2.0,1.7,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0],
                                         [2.0,2.0,2.0,2.0, 2.0,2.0,0.2,0.1, 2.0,0.1,0.1,2.0, 2.0,2.0,2.0,2.0],
                                         [2.0,2.0,0.3,2.0, 2.0,2.0,0.3,0.2, 2.0,0.2,0.2,2.0, 2.0,2.0,2.0,0.7],
                                         [0.7,2.0,0.3,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7]],
                                         dtype=np.float64)
      lake_operators_wrapper.fill_lakes(lake_minima,lake_mask,orography,True)
      np.testing.assert_array_equal(orography,expected_orography_out)

class BasinEvaluationDriver(unittest.TestCase):

  def testEvaluateBasinsOne(self):
    coarse_catchment_nums_in = np.array([[3,3,2,2],
                                         [3,3,2,2],
                                         [1,1,1,2],
                                         [1,1,1,1]],
                                         dtype=np.int32)
    corrected_orography_in =\
    np.array([[10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0],
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
              [10.0,10.0,10.0, 1.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 0.0,10.0,10.0,10.0,10.0]],
              dtype=np.float64)
    raw_orography_in =\
    np.array([[10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0],
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
              [10.0,10.0,10.0, 1.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 0.0,10.0,10.0,10.0,10.0]],
              dtype=np.float64)
    minima_in =\
    np.array([[False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
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
              False,False,False,False]],
              dtype=np.int32)
    prior_fine_rdirs_in = np.array([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,3.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,3.0,3.0,2.0],
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
    prior_fine_catchments_in = np.array([[11,11,11,11,11,11,13,13,12,12,12,12,12,12,12,12,12,11,11,11],
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
    connection_volume_thresholds_in = np.zeros((20,20),dtype=np.float64)
    connection_volume_thresholds_in.fill(0.0)
    flood_volume_thresholds_in = np.zeros((20,20),dtype=np.float64)
    flood_volume_thresholds_in.fill(0.0)
    flood_next_cell_lat_index_in = np.zeros((20,20),dtype=np.int32)
    flood_next_cell_lat_index_in.fill(-1)
    flood_next_cell_lon_index_in = np.zeros((20,20),dtype=np.int32)
    flood_next_cell_lon_index_in.fill(-1)
    connect_next_cell_lat_index_in = np.zeros((20,20),dtype=np.int32)
    connect_next_cell_lat_index_in.fill(-1)
    connect_next_cell_lon_index_in = np.zeros((20,20),dtype=np.int32)
    connect_next_cell_lon_index_in.fill(-1)
    flood_force_merge_lat_index_in = np.zeros((20,20),dtype=np.int32)
    flood_force_merge_lat_index_in.fill(-1)
    flood_force_merge_lon_index_in = np.zeros((20,20),dtype=np.int32)
    flood_force_merge_lon_index_in.fill(-1)
    connect_force_merge_lat_index_in = np.zeros((20,20),dtype=np.int32)
    connect_force_merge_lat_index_in.fill(-1)
    connect_force_merge_lon_index_in = np.zeros((20,20),dtype=np.int32)
    connect_force_merge_lon_index_in.fill(-1)
    flood_redirect_lat_index_in = np.zeros((20,20),dtype=np.int32)
    flood_redirect_lat_index_in.fill(-1)
    flood_redirect_lon_index_in = np.zeros((20,20),dtype=np.int32)
    flood_redirect_lon_index_in.fill(-1)
    connect_redirect_lat_index_in = np.zeros((20,20),dtype=np.int32)
    connect_redirect_lat_index_in.fill(-1)
    connect_redirect_lon_index_in = np.zeros((20,20),dtype=np.int32)
    connect_redirect_lon_index_in.fill(-1)
    flood_local_redirect_in = np.zeros((20,20),dtype=np.int32)
    flood_local_redirect_in.fill(False)
    connect_local_redirect_in = np.zeros((20,20),dtype=np.int32)
    connect_local_redirect_in.fill(False)
    merge_points_in = np.zeros((20,20),dtype=np.int32)
    merge_points_in.fill(0)
    flood_volume_thresholds_expected_out =\
    np.array([[-1,  -1,   -1,   -1,   -1, -1,   -1, -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1, -1,   -1],
      [-1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   5.0],
      [0.0, 262.0,5.0, -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,    111.0,111.0,56.0,111.0,-1,  -1,     -1,   2.0],
      [-1,   5.0,  5.0, -1,   -1,  340.0,262.0,-1, -1,  -1, -1,  111.0,1.0,  1.0,  56.0,56.0,-1,   -1,     -1,   -1],
      [-1,   5.0,  5.0, -1,   -1,  10.0, 10.0, 38.0,10.0,-1,-1, -1,    0.0,  1.0,  1.0, 26.0, 56.0,-1,     -1,   -1],
      [-1,   5.0,  5.0, 186.0, 2.0,2.0,  -1,    10.0,10.0,-1, 1.0,6.0,  1.0,  0.0,  1.0, 26.0, 26.0,111.0, -1,   -1],
      [16.0,16.0, 16.0,-1,    2.0,0.0,  2.0,  2.0, 10.0,-1, 1.0,0.0,  0.0,  1.0,  1.0, 1.0,  26.0,56.0,   -1,   -1],
      [-1,   46.0, 16.0,-1,   -1,  2.0, -1,    23.0,-1,  -1, -1,  1.0,  0.0,  1.0,  1.0, 1.0,  56.0,-1,    -1,   -1],
      [-1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1,  56.0, 1.0,  1.0,  1.0, 26.0, 56.0,-1,    -1,   -1],
      [-1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,    56.0, 56.0,-1,  -1,   -1,  -1,     -1,   -1],
      [-1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1],
      [-1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1],
      [-1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1],
      [-1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,   0.0,  3.0, 3.0,    10.0,-1],
      [-1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,   0.0,  3.0, 3.0,   -1,   -1],
      [-1,  -1,   -1,    1.0, -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1],
      [-1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1],
      [-1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1],
      [-1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1],
      [-1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1]],
      dtype=np.float64)
    connection_volume_thresholds_expected_out =\
       np.array([[-1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  186.0, 23.0,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, 56.0,-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
                 [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1]],
                 dtype=np.float64)
    flood_next_cell_lat_index_expected_out = \
    np.array([[-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,   3],
              [2,  3,   5,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,   2,   2,   4,   5,  -1,  -1, -1,   1],
              [-1,  4,   2,  -1,  -1,  7,  2,  -1,  -1,  -1, -1,  2,   5,   3,   9,   2,  -1,  -1, -1,  -1],
              [-1,  3,   4,  -1,  -1,  6,  4,   5,   3,  -1, -1, -1,   7,   3,   4,   3,   6,  -1, -1,  -1],
              [-1,  6,   5,   3,   6,  7, -1,   4,   4,  -1,  5,  6,   6,   6,   4,   8,   4,   3, -1,  -1],
              [7,  6,   6,  -1,   5,  5,  6,   5,   5,  -1,  5,  5,   4,   8,   6,   6,   5,   5, -1,  -1],
              [-1,  5,   7,  -1,  -1,  6, -1,   4,  -1,  -1, -1,  5,   6,   6,   8,   7,   5,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1,  8,   7,   7,   8,   6,   7,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,   8,   9,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  14,  14,  13, 15,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  13,  14,  13, -1,  -1],
              [-1, -1,  -1,  16,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1]],
              dtype=np.int32)
    flood_next_cell_lon_index_expected_out = \
    np.array([[-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,   1],
              [19,  5,   2,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  15,  12,  16,   3,  -1,  -1, -1,  19],
              [-1,  2,   2,  -1,  -1, 17,  1,  -1,  -1,  -1, -1, 13,  15,  12,  13,  14,  -1,  -1, -1,  -1],
              [-1,  2,   1,  -1,  -1,  8,  5,   9,   6,  -1, -1, -1,  12,  13,  13,  14,  17,  -1, -1,  -1],
              [-1,  2,   1,   5,   7,  5, -1,   6,   8,  -1, 14, 14,  10,  12,  14,  15,  15,  11, -1,  -1],
              [2,  0,   1,  -1,   4,  5,  4,   7,   8,  -1, 10, 11,  12,  12,  13,  14,  16,  17, -1,  -1],
              [-1,  3,   1,  -1,  -1,  6, -1,   7,  -1,  -1, -1, 12,  11,  15,  14,  13,   9,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, 16,  11,  15,  13,  16,  16,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  11,  12,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  15,  16,  18, 15,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  16,  17,  17, -1,  -1],
              [-1, -1,  -1,   4,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1],
              [-1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1]],
              dtype=np.int32)
    connect_next_cell_lat_index_expected_out = \
    np.array([[-1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,   3,       7,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1,  3,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1]],
              dtype=np.int32)
    connect_next_cell_lon_index_expected_out = \
    np.array([[-1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,   6,       7,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, 15,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1]],
              dtype=np.int32)
    flood_redirect_lat_index_expected_out = \
    np.array([[-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,   1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,   1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1,  7, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1,  7, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1,  1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1,  6, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1,  5,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1,  3, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1,  3, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1]],
              dtype=np.int32)
    flood_redirect_lon_index_expected_out = \
    np.array([[-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,   0, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,   3, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, 14, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, 14, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1,  0, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1,  5, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, 13,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1,  3, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1,  0, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1]],
              dtype=np.int32)
    connect_redirect_lat_index_expected_out = \
    np.array([[-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1]],
              dtype=np.int32)
    connect_redirect_lon_index_expected_out = \
    np.array([[-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1]],
              dtype=np.int32)
    flood_local_redirect_expected_out =  \
    np.array([[False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False, True,False,False, False,False,
              False,False,False,False],
              [False,False,False,False,False,False, False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False, False,
              True,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,
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
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False, False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False]],
              dtype=np.int32)
    connect_local_redirect_expected_out = \
    np.array([[False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False, False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False, False,False,
              False,False,False,False],
              [False,False,False,False,False,False, False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False, False,
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
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False, False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False],
              [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
              False,False,False,False]],
              dtype=np.int32)
    merge_points_expected_out = \
     np.array([[0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
               [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
               [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 7,0,0,0,0],
               [0,0,0,0,0, 8,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
               [0,0,0,0,0, 0,0,8,0,0,0,0,0,0,0, 0,0,0,0,0],

               [0,0,0,0,0, 0,0,0,0,0,0,8,0,0,0, 0,0,0,0,0],
               [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
               [0,8,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,7,0,0,0],
               [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,7, 0,0,0,0,0],
               [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],

               [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
               [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
               [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
               [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,8,0],
               [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],

               [0,0,0,8,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
               [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
               [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
               [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
               [0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0]],
               dtype=np.int32)
    flood_force_merge_lat_index_expected_out = \
    np.array([[-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,   6, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1,  6, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1,  7,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1]],
              dtype=np.int32)
    flood_force_merge_lon_index_expected_out = \
    np.array([[-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,   2, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1,  8, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, 12,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1]],
              dtype=np.int32)
    connect_force_merge_lat_index_expected_out = \
    np.array([[-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1]],
              dtype=np.int32)
    connect_force_merge_lon_index_expected_out = \
    np.array([[-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1]],
              dtype=np.int32)
    evaluate_basins_wrapper.evaluate_basins(minima_in,
                                            raw_orography_in,
                                            corrected_orography_in,
                                            connection_volume_thresholds_in,
                                            flood_volume_thresholds_in,
                                            prior_fine_rdirs_in,
                                            prior_fine_catchments_in,
                                            coarse_catchment_nums_in,
                                            flood_next_cell_lat_index_in,
                                            flood_next_cell_lon_index_in,
                                            connect_next_cell_lat_index_in,
                                            connect_next_cell_lon_index_in,
                                            flood_force_merge_lat_index_in,
                                            flood_force_merge_lon_index_in,
                                            connect_force_merge_lat_index_in,
                                            connect_force_merge_lon_index_in,
                                            flood_redirect_lat_index_in,
                                            flood_redirect_lon_index_in,
                                            connect_redirect_lat_index_in,
                                            connect_redirect_lon_index_in,
                                            flood_local_redirect_in,
                                            connect_local_redirect_in,
                                            merge_points_in)
    self.assertTrue(False)
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in)
    #           == field<double>(flood_volume_thresholds_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in)
    #           == field<double>(connection_volume_thresholds_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in)
    #           == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in)
    #           == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
    #           == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
    #           == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
    #           == field<int>(flood_redirect_lat_index_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
    #           == field<int>(flood_redirect_lon_index_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
    #           == field<int>(connect_redirect_lat_index_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
    #           == field<int>(connect_redirect_lon_index_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
    #           == field<bool>(flood_local_redirect_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
    #           == field<bool>(connect_local_redirect_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
    #           == field<merge_types>(merge_points_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
    #           == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
    #           == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
    #           == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in))
    # PROCESS BY HAND=>?np.testing.assert_array_equal.?EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
    #           == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
