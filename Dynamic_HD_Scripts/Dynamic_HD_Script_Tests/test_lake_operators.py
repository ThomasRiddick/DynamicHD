'''
Test various dynamic hydrological discharge utility functions
Created on April 16, 2018

@author: thomasriddick
'''
import unittest
import Dynamic_HD_Scripts.field as field
import Dynamic_HD_Scripts.libs.lake_operators_wrapper as lake_operators_wrapper  #@UnresolvedImport
import Dynamic_HD_Scripts.libs.fill_sinks_wrapper as fill_sinks_wrapper
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

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
