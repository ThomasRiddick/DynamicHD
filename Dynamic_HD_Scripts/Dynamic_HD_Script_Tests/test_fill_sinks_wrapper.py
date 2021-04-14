'''
Unit tests for both the C++ sink filling codes results and the Cython wrapper that
interfaces between it and python.

Created on Mar 31, 2016

@author: thomasriddick
'''

import unittest
import numpy as np
from ../Dynamic_HD_Scripts/libs import fill_sinks_wrapper


class TestAlgorithmOne(unittest.TestCase):
    """Tests of Algorithm 1 from Barnes et al (2014)"""

    ndv = np.finfo(dtype=np.float64).min #set the no data value

    def setUp(self):
        """Unit test setup function; prepare test pseudo-data and expected results"""
        self.input_array = np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                       [0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0],
                                       [0.0,1.1,2.0,2.0,1.2,2.0,2.0,2.0,1.1,0.0],
                                       [0.0,1.1,2.0,3.0,3.0,1.3,3.0,2.0,1.1,0.0],
                                       [0.0,1.1,2.0,3.0,0.1,0.1,3.0,2.0,1.1,0.0],
                                       [0.0,1.1,2.0,3.0,0.1,0.1,3.0,2.0,1.1,0.0],
                                       [0.0,1.1,2.0,3.0,3.0,3.0,3.0,2.0,1.1,0.0],
                                       [0.0,1.1,2.0,2.0,2.0,2.0,2.0,2.0,1.1,0.0],
                                       [0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0],
                                       [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                                      dtype=np.float64, order='C')

        self.expected_output_array = np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                                [0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0],
                                                [0.0,1.1,2.0,2.0,1.2,2.0,2.0,2.0,1.1,0.0],
                                                [0.0,1.1,2.0,3.0,3.0,1.3,3.0,2.0,1.1,0.0],
                                                [0.0,1.1,2.0,3.0,1.3,1.3,3.0,2.0,1.1,0.0],
                                                [0.0,1.1,2.0,3.0,1.3,1.3,3.0,2.0,1.1,0.0],
                                                [0.0,1.1,2.0,3.0,3.0,3.0,3.0,2.0,1.1,0.0],
                                                [0.0,1.1,2.0,2.0,2.0,2.0,2.0,2.0,1.1,0.0],
                                                [0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0],
                                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                                               dtype=np.float64, order='C')

        self.wrapped_input_array = np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                               [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                               [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,1.2],
                                               [1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                               [0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1],
                                               [0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1],
                                               [3.0,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                               [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,2.0],
                                               [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                                              dtype=np.float64, order='C')

        self.wrapped_input_array_with_slope = np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                                          [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                                          [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,1.2],
                                                          [1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                                          [0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1],
                                                          [0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.7],
                                                          [3.0,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                                          [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,2.0],
                                                          [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                                          [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                                                         dtype=np.float64, order='C')

        self.orography_in_wrapped_sink_with_slope = \
            np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                        [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,1.5],
                        [1.45,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                        [0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1],
                        [0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1],
                        [3.0,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                        [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,2.0],
                        [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                       dtype=np.float64, order='C')

        self.orography_in_with_slope = \
            np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0],
                        [0.0,1.1,2.0,2.0,1.2,2.0,2.0,2.0,1.1,0.0],
                        [0.0,1.1,2.0,0.1,0.1,0.1,0.1,2.0,1.1,0.0],
                        [0.0,1.1,2.0,0.1,0.1,0.1,0.1,1.4,1.1,0.0],
                        [0.0,1.1,2.0,0.1,0.1,0.1,0.1,2.0,1.1,0.0],
                        [0.0,1.1,2.0,0.1,0.1,0.1,0.1,2.0,1.1,0.0],
                        [0.0,1.1,2.0,2.0,2.0,1.3,2.0,2.0,1.1,0.0],
                        [0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                       dtype=np.float64, order='C')

        self.expected_wrapped_array_output = np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                                         [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                                         [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,1.2],
                                                         [1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                                         [1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,1.3],
                                                         [1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,1.3],
                                                         [3.0,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                                         [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,2.0],
                                                         [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                                         [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                                                        dtype=np.float64, order='C')

        self.expected_wrapped_array_output = np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                                         [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                                         [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,1.2],
                                                         [1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                                         [1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,1.3],
                                                         [1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,1.3],
                                                         [3.0,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                                         [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,2.0],
                                                         [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                                         [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                                                        dtype=np.float64, order='C')

        self.expected_orography_wrapped_sink_ls_filled_with_slope = \
            np.asarray([[0.3,0.2,0.1,0.0,0.0,0.0,0.0,0.1,0.2,0.3],
                        [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                        [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,1.5],
                        [1.6,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                        [1.7,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,1.7],
                        [1.8,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,1.8],
                        [3.0,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                        [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,2.0],
                        [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                        [0.3,0.2,0.1,0.0,0.0,0.0,0.0,0.1,0.2,0.3]],
                       dtype=np.float64, order='C')

        self.expected_orography_out_with_slope = \
            np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0],
                        [0.0,1.1,2.0,2.0,1.2,2.0,2.0,2.0,1.1,0.0],
                        [0.0,1.1,2.0,1.3,1.3,1.3,1.4,2.0,1.1,0.0],
                        [0.0,1.1,2.0,1.4,1.4,1.4,1.4,1.4,1.1,0.0],
                        [0.0,1.1,2.0,1.5,1.5,1.5,1.5,2.0,1.1,0.0],
                        [0.0,1.1,2.0,1.5,1.4,1.4,1.4,2.0,1.1,0.0],
                        [0.0,1.1,2.0,2.0,2.0,1.3,2.0,2.0,1.1,0.0],
                        [0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                       dtype=np.float64, order='C')

        self.true_sink_and_former_true_sinks_wrapped_with_lsmask_and_slope_expected_output = \
            np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                        [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,1.2],
                        [1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                        [0.7,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.7],
                        [0.7,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.7],
                        [3.0,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                        [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,2.0],
                        [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                       dtype=np.float64, order='C')

        self.expected_wrapped_array_nodata = np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                                         [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                                         [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,1.2],
                                                         [1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                                         [1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,1.3],
                                                         [1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,1.3],
                                                         [3.0,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                                         [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,2.0],
                                                         [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                                         [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                                                        dtype=np.float64, order='C')
        self.expected_wrapped_array_nodata[:,4:6] = self.ndv

        self.ls_mask = np.asarray([[False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False]],
                                  dtype=np.int32, order='C')

        self.dummy_mask = np.empty((1,1),dtype=np.int32)

        self.true_sinks_input_array = np.asarray([[False,False,False,False,False,False,False,False,False,False],
                                                  [False,False,False,False,False,False,False,False,False,False],
                                                  [False,False,False,False,False,False,False,False,False,False],
                                                  [False,False,False,False,False,False,False,False,False,False],
                                                  [False,False,False,False,False,False,False,False,False,False],
                                                  [False,False,False,False,True,False,False,False,False,False],
                                                  [False,False,False,False,False,False,False,False,False,False],
                                                  [False,False,False,False,False,False,False,False,False,False],
                                                  [False,False,False,False,False,False,False,False,False,False],
                                                  [False,False,False,False,False,False,False,False,False,False]],
                                                  dtype=np.int32, order='C')

        self.former_true_sinks_input_array = np.asarray([[False,False,False,False,False,False,False,False,False,False],
                                                         [False,False,False,False,False,False,False,False,False,False],
                                                         [False,False,False,False,False,False,False,False,False,False],
                                                         [False,False,False,False,False,False,False,False,False,False],
                                                         [False,False,False,False,False,False,False,False,False,False],
                                                         [False,False,False,False,False,False,False,False,False,False],
                                                         [False,False,False,False,False,False,True, False,False,False],
                                                         [False,False,False,False,False,False,False,False,False,False],
                                                         [False,False,False,False,False,False,False,False,False,False],
                                                         [False,False,False,False,False,False,False,False,False,False]],
                                                         dtype=np.int32, order='C')

        self.true_sinks_expected_output_orography_array = np.copy(self.input_array)

        self.true_sink_and_former_true_sinks_input_array = \
            np.asarray([[False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,True,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,True, False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False]],
                        dtype=np.int32, order='C')

        self.former_true_sinks_input_array_wrapped = \
            np.asarray([[False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,True],
                        [True,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,True,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,True, False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False]],
                        dtype=np.int32, order='C')


        self.true_sink_and_former_true_sinks_input_array_wrapped = \
            np.asarray([[False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,True],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,True,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,True],
                        [False,False,False,False,False,False,True, False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False]],
                        dtype=np.int32, order='C')

        self.true_sink_and_former_true_sinks_wrapped_with_lsmask_expected_output = \
            np.copy(self.wrapped_input_array)

    def testBasicCall(self):
        """Test a basic call to fill_sinks using algorithm 1"""
        fill_sinks_wrapper.fill_sinks_cpp_func(self.input_array,1) #@UndefinedVariable
        np.testing.assert_array_equal(self.input_array, self.expected_output_array,
                                      "Basic fill sinks call doesn't give expected results")

    def testCallWithLSMask(self):
        """Test call to fill_sinks using algorithm 1 with a landsea mask"""
        fill_sinks_wrapper.fill_sinks_cpp_func(self.wrapped_input_array,1,True,self.ls_mask) #@UndefinedVariable
        np.testing.assert_array_equal(self.wrapped_input_array, self.expected_wrapped_array_output,
                                      "Fill sinks algorithm 1 with land sea mask and wrapped orography doesn't"
                                      "give expected results")

    def testCallWithLSMaskSettingSeaAsNoData(self):
        """Test call to fill_sinks using algorithm 1 with a landsea mask; setting the sea to a no data value"""
        fill_sinks_wrapper.fill_sinks_cpp_func(self.wrapped_input_array,1,True,self.ls_mask,True) #@UndefinedVariable
        np.testing.assert_array_equal(self.wrapped_input_array, self.expected_wrapped_array_nodata,
                                      "Fill sinks algorithm 1 with land sea mask and wrapped orography doesn't"
                                      "give expected results")

    def testBasicCallWithTrueSink(self):
        """Test of a call with a true sink included"""
        fill_sinks_wrapper.fill_sinks_cpp_func(self.input_array,1,False,self.dummy_mask,False,
                                               use_true_sinks=True,
                                               true_sinks_in=self.true_sinks_input_array) #@UndefinedVariable
        np.testing.assert_array_equal(self.input_array, self.true_sinks_expected_output_orography_array,
                                      "Supplying a true sink to a basic algorithm 1 call is not producing"
                                      " the expected results")

    def testBasicCallWithFormerTrueSink(self):
        """Test of a call with a true sink (that is no longer active as it is not in a pit) included"""
        fill_sinks_wrapper.fill_sinks_cpp_func(self.input_array,1,False,self.dummy_mask,False,
                                               use_true_sinks=True,
                                               true_sinks_in=self.former_true_sinks_input_array) #@UndefinedVariable
        np.testing.assert_array_equal(self.input_array, self.expected_output_array,
                                      "Supplying a former true sink to a basic algorithm 1 call is not producing"
                                      " the expected results")

    def testBasicCallwithBothTrueSinkAndFormerTrueSink(self):
        """Test of a call with an active true and an inactive true sink included"""
        fill_sinks_wrapper.fill_sinks_cpp_func(self.input_array,1,False,self.dummy_mask,False,
                                               use_true_sinks=True,
                                               true_sinks_in=self.true_sink_and_former_true_sinks_input_array) #@UndefinedVariable
        np.testing.assert_array_equal(self.input_array, self.true_sinks_expected_output_orography_array,
                                      "Supplying a true and former true sink to a basic algorithm 1 call"
                                      " is not producing the expected results")

    def testLSCallWithFormerTrueSink(self):
        """Test of a call using a land-sea mask with an inactive true sink included"""
        fill_sinks_wrapper.fill_sinks_cpp_func(self.wrapped_input_array,1,True,self.ls_mask,use_true_sinks=True,
                                               true_sinks_in=self.former_true_sinks_input_array_wrapped) #@UndefinedVariable
        np.testing.assert_array_equal(self.wrapped_input_array,
                                      self.expected_wrapped_array_output,
                                      "Fill sinks algorithm 1 with land sea mask and wrapped orography doesn't"
                                      " give expected results when using former true sinks")

    def testLSCallWithBothTrueSinkAndFormerTrueSink(self):
        """Test of a call using a land-sea mask with an active true and an inactive true sink included"""
        fill_sinks_wrapper.fill_sinks_cpp_func(self.wrapped_input_array,1,True,self.ls_mask,use_true_sinks=True,
                                               true_sinks_in=self.\
                                               true_sink_and_former_true_sinks_input_array_wrapped) #@UndefinedVariable
        np.testing.assert_array_equal(self.wrapped_input_array,
                                      self.true_sink_and_former_true_sinks_wrapped_with_lsmask_expected_output,
                                      "Fill sinks algorithm 1 with land sea mask and wrapped orography doesn't"
                                      " give expected results when using true and former true sinks")

    def testLSCallWithBothTrueSinkOnSlopeAndFormerTrueSinkWrapped(self):
        """Test of a call using a ls mask with an active and an inactive true sink included that wraps across dateline"""
        fill_sinks_wrapper.fill_sinks_cpp_func(self.wrapped_input_array_with_slope,1,True,self.ls_mask,use_true_sinks=True,
                                               true_sinks_in=self.\
                                               true_sink_and_former_true_sinks_input_array_wrapped) #@UndefinedVariable
        np.testing.assert_array_equal(self.wrapped_input_array_with_slope,
                                      self.true_sink_and_former_true_sinks_wrapped_with_lsmask_and_slope_expected_output,
                                      "Fill sinks algorithm 1 with land sea mask and wrapped orography doesn't"
                                      " give expected results when using true (on a slope) and former true sinks")

    def testFillingSinksWithSlopeAddedToFill(self):
        """Test call on a wrapped grid where a slope is added to the filled sink"""
        fill_sinks_wrapper.fill_sinks_cpp_func(self.orography_in_wrapped_sink_with_slope,1,True,self.ls_mask,use_true_sinks=False,
                                               true_sinks_in=self.dummy_mask,add_slope=True,epsilon=0.1,);
        np.testing.assert_array_almost_equal(self.orography_in_wrapped_sink_with_slope,
                                             self.expected_orography_wrapped_sink_ls_filled_with_slope,
                                             decimal=8,
                                             err_msg="Testing sink filling where a slight slope to filled sinks doesn't produce"
                                             " expected results")

    def testFillingSinksWithSlopeAddedToFillMultipleEntry(self):
        """Test call on a grid where a slope is added to a filled sink with multiple entry points"""
        fill_sinks_wrapper.fill_sinks_cpp_func(self.orography_in_with_slope,1,False,self.ls_mask,False,
                                               use_true_sinks=False,true_sinks_in=self.dummy_mask,add_slope=True,
                                               epsilon=0.1);
        np.testing.assert_array_almost_equal(self.orography_in_with_slope,
                                             self.expected_orography_out_with_slope,
                                             decimal=8,
                                             err_msg="Testing sink filling where a slight slope to a filled sink with multiply"
                                                     " entry points doesn't produce expected results")

class TestAlgorithmFour(unittest.TestCase):
    """Tests of Algorithm 4 from Barnes et al (2014)"""

    def setUp(self):
        """Unit test setup function; prepare test pseudo-data and expected results"""
        self.input_array = np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                       [0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0],
                                       [0.0,1.1,2.0,2.0,1.2,2.0,2.0,2.0,1.1,0.0],
                                       [0.0,1.1,2.0,3.0,3.0,1.3,3.0,2.0,1.1,0.0],
                                       [0.0,1.1,2.0,3.0,0.1,0.1,3.0,2.0,1.1,0.0],
                                       [0.0,1.1,2.0,3.0,0.1,0.1,3.0,2.0,1.1,0.0],
                                       [0.0,1.1,2.0,3.0,3.0,3.0,3.0,2.0,1.1,0.0],
                                       [0.0,1.1,2.0,2.0,2.0,2.0,2.0,2.0,1.1,0.0],
                                       [0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0],
                                       [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                                      dtype=np.float64, order='C')

        self.expected_output_array = np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                                [0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0],
                                                [0.0,1.1,2.0,2.0,1.2,2.0,2.0,2.0,1.1,0.0],
                                                [0.0,1.1,2.0,3.0,3.0,1.3,3.0,2.0,1.1,0.0],
                                                [0.0,1.1,2.0,3.0,0.1,0.1,3.0,2.0,1.1,0.0],
                                                [0.0,1.1,2.0,3.0,0.1,0.1,3.0,2.0,1.1,0.0],
                                                [0.0,1.1,2.0,3.0,3.0,3.0,3.0,2.0,1.1,0.0],
                                                [0.0,1.1,2.0,2.0,2.0,2.0,2.0,2.0,1.1,0.0],
                                                [0.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,0.0],
                                                [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                                               dtype=np.float64, order='C')

        self.wrapped_input_array = np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                               [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                               [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,1.2],
                                               [1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                               [0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1],
                                               [0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1],
                                               [3.0,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                               [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,2.0],
                                               [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                                              dtype=np.float64, order='C')

        self.wrapped_input_array_with_slope = np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                                          [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                                          [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,1.2],
                                                          [1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                                          [0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1],
                                                          [0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.7],
                                                          [3.0,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                                          [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,2.0],
                                                          [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                                          [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                                                         dtype=np.float64, order='C')

        self.wrapped_output_array_with_slope = np.copy(self.wrapped_input_array_with_slope)

        self.expected_wrapped_array_output = np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                                         [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                                         [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,1.2],
                                                         [1.3,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                                         [0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1],
                                                         [0.1,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,0.1],
                                                         [3.0,3.0,2.0,1.1,0.0,0.0,1.1,2.0,3.0,3.0],
                                                         [2.0,2.0,2.0,1.1,0.0,0.0,1.1,2.0,2.0,2.0],
                                                         [1.1,1.1,1.1,1.1,0.0,0.0,1.1,1.1,1.1,1.1],
                                                         [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
                                                        dtype=np.float64, order='C')

        self.rdirs = np.zeros((10,10),dtype=np.float64,order='C')

        self.ls_mask = np.asarray([[False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False],
                                   [False,False,False,False,True,True,False,False,False,False]],
                                  dtype=np.int32, order='C')

        self.true_sink_and_former_true_sinks_input_array = \
            np.asarray([[False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,True,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,True, False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False]],
                        dtype=np.int32, order='C')

        self.former_true_sinks_input_array = \
            np.asarray([[True,False,False,False,False,False,False,False,False,False],
                        [False,False,False,True,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,True],
                        [True,False,False,False,False,True ,True ,False,True, True],
                        [False,False,False,True,False,False,False,False,True,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,True,False,False,False,True, False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,True,False,False,False,False,False,False]],
                        dtype=np.int32, order='C')

        self.wrapped_current_and_former_true_sinks_input_array = \
            np.asarray([[True,False,False,False,False,False,False,False,False,False],
                        [False,False,False,True,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,True],
                        [True,False,False,False,False,True ,True ,False,True, True],
                        [False,False,False,True,False,False,False,False,True,False],
                        [False,False,False,False,False,False,False,False,False,True],
                        [False,False,True,False,False,False,True, False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,False,False,False,False,False,False,False],
                        [False,False,False,True,False,False,False,False,False,False]],
                        dtype=np.int32, order='C')

        self.expected_rdirs_output = np.asarray([[4.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,6.0],
                                                [4.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,9.0,6.0],
                                                [4.0,7.0,7.0,7.0,7.0,7.0,7.0,9.0,9.0,6.0],
                                                [4.0,7.0,7.0,9.0,8.0,7.0,4.0,9.0,9.0,6.0],
                                                [4.0,7.0,7.0,6.0,9.0,8.0,7.0,9.0,9.0,6.0],
                                                [4.0,7.0,7.0,9.0,9.0,8.0,7.0,9.0,9.0,6.0],
                                                [4.0,7.0,7.0,9.0,9.0,8.0,7.0,9.0,9.0,6.0],
                                                [4.0,7.0,7.0,1.0,1.0,1.0,1.0,9.0,9.0,6.0],
                                                [4.0,7.0,1.0,1.0,1.0,1.0,1.0,1.0,9.0,6.0],
                                                [4.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,6.0]],
                                               dtype=np.float64, order='C')

        self.expected_catchment_num_output =  np.asarray([[1, 21,23,25,27,29,31,33,35, 2],
                                                          [3, 1, 21,23,25,27,29,31, 2, 4],
                                                          [5, 3, 1, 21,23,25,27, 2, 4, 6],
                                                          [7, 5, 3, 23,23,23,23, 4, 6, 8],
                                                          [9, 7, 5, 23,23,23,23, 6, 8,10],
                                                          [11,9, 7, 23,23,23,23, 8,10,12],
                                                          [13,11,9, 23,23,23,23,10,12,14],
                                                          [15,13,11,22,24,26,28,12,14,16],
                                                          [17,15,22,24,26,28,30,32,16,18],
                                                          [19,22,24,26,28,30,32,34,36,20]],
                                                         dtype=np.int32, order='C')

        self.expected_rdirs_output_with_true_and_former_true_sinks = np.asarray([[4.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,6.0],
                                                                                [4.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,9.0,6.0],
                                                                                [4.0,7.0,7.0,7.0,7.0,7.0,7.0,9.0,9.0,6.0],
                                                                                [4.0,7.0,7.0,3.0,2.0,1.0,1.0,9.0,9.0,6.0],
                                                                                [4.0,7.0,7.0,6.0,5.0,4.0,4.0,9.0,9.0,6.0],
                                                                                [4.0,7.0,7.0,9.0,8.0,7.0,7.0,9.0,9.0,6.0],
                                                                                [4.0,7.0,7.0,9.0,8.0,7.0,7.0,9.0,9.0,6.0],
                                                                                [4.0,7.0,7.0,1.0,1.0,1.0,1.0,9.0,9.0,6.0],
                                                                                [4.0,7.0,1.0,1.0,1.0,1.0,1.0,1.0,9.0,6.0],
                                                                                [4.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,6.0]],
                                                                               dtype=np.float64, order='C')

        self.expected_catchment_num_output_with_true_and_former_true_sinks =\
            np.asarray([[1, 21,23,25,27,29,31,33,35, 2],
                        [3, 1, 21,23,25,27,29,31, 2, 4],
                        [5, 3, 1, 21,23,25,27, 2, 4, 6],
                        [7, 5, 3, 37,37,37,37, 4, 6, 8],
                        [9, 7, 5, 37,37,37,37, 6, 8,10],
                        [11,9, 7, 37,37,37,37, 8,10,12],
                        [13,11,9, 37,37,37,37,10,12,14],
                        [15,13,11,22,24,26,28,12,14,16],
                        [17,15,22,24,26,28,30,32,16,18],
                        [19,22,24,26,28,30,32,34,36,20]],
                       dtype=np.int32, order='C')

        self.expected_wrapped_rdirs_output = np.asarray([[6.0,6.0,6.0,6.0,0.0,0.0,4.0,4.0,4.0,4.0],
                                                         [9.0,9.0,9.0,9.0,0.0,0.0,7.0,7.0,7.0,7.0],
                                                         [9.0,9.0,9.0,9.0,0.0,0.0,7.0,7.0,7.0,7.0],
                                                         [7.0,4.0,9.0,9.0,0.0,0.0,7.0,7.0,9.0,8.0],
                                                         [8.0,7.0,9.0,9.0,0.0,0.0,7.0,7.0,6.0,9.0],
                                                         [8.0,7.0,9.0,9.0,0.0,0.0,7.0,7.0,9.0,9.0],
                                                         [8.0,7.0,9.0,9.0,0.0,0.0,7.0,7.0,9.0,9.0],
                                                         [3.0,3.0,9.0,9.0,0.0,0.0,7.0,7.0,1.0,1.0],
                                                         [3.0,3.0,3.0,9.0,0.0,0.0,7.0,1.0,1.0,1.0],
                                                         [6.0,6.0,6.0,9.0,0.0,0.0,7.0,4.0,4.0,4.0]],
                                                        dtype=np.float64, order='C')

        self.expected_catchment_num_wrapped_output = np.asarray([[ 2, 2, 2, 2,0,0, 4, 4, 4, 4],
                                                                 [ 2, 2, 2, 1,0,0, 3, 4, 4, 4],
                                                                 [ 2, 2, 1, 5,0,0, 6, 3, 4, 4],
                                                                 [ 4, 4, 5, 7,0,0, 8, 6, 4, 4],
                                                                 [ 4, 4, 7, 9,0,0,10, 8, 4, 4],
                                                                 [ 4, 4, 9,11,0,0,12,10, 4, 4],
                                                                 [ 4, 4,11,13,0,0,14,12, 4, 4],
                                                                 [19,19,13,15,0,0,16,14,20,20],
                                                                 [19,19,19,17,0,0,18,20,20,20],
                                                                 [19,19,19,19,0,0,20,20,20,20]],
                                                                dtype=np.int32, order='C')

        self.expected_wrapped_rdirs_output_with_former_sink =\
            np.asarray([[6.0,6.0,6.0,6.0,0.0,0.0,4.0,4.0,4.0,4.0],
                        [9.0,9.0,9.0,9.0,0.0,0.0,7.0,7.0,7.0,7.0],
                        [9.0,9.0,9.0,9.0,0.0,0.0,7.0,7.0,7.0,7.0],
                        [7.0,4.0,9.0,9.0,0.0,0.0,7.0,7.0,9.0,8.0],
                        [8.0,7.0,9.0,9.0,0.0,0.0,7.0,7.0,6.0,9.0],
                        [8.0,7.0,9.0,9.0,0.0,0.0,7.0,7.0,9.0,9.0],
                        [8.0,7.0,9.0,9.0,0.0,0.0,7.0,7.0,9.0,9.0],
                        [3.0,3.0,9.0,9.0,0.0,0.0,7.0,7.0,1.0,1.0],
                        [3.0,3.0,3.0,9.0,0.0,0.0,7.0,1.0,1.0,1.0],
                        [6.0,6.0,6.0,9.0,0.0,0.0,7.0,4.0,4.0,4.0]],
                       dtype=np.float64, order='C')

        self.expected_wrapped_rdirs_output_with_current_sink_on_slope_and_former_sinks =\
            np.asarray([[6.0,6.0,6.0,6.0,0.0,0.0,4.0,4.0,4.0,4.0],
                        [9.0,9.0,9.0,6.0,0.0,0.0,4.0,7.0,7.0,7.0],
                        [9.0,9.0,9.0,6.0,0.0,0.0,4.0,7.0,7.0,7.0],
                        [1.0,1.0,9.0,6.0,0.0,0.0,4.0,7.0,3.0,2.0],
                        [1.0,1.0,9.0,6.0,0.0,0.0,4.0,7.0,3.0,2.0],
                        [4.0,4.0,9.0,6.0,0.0,0.0,4.0,7.0,6.0,5.0],
                        [7.0,7.0,9.0,6.0,0.0,0.0,4.0,7.0,9.0,8.0],
                        [3.0,3.0,9.0,6.0,0.0,0.0,4.0,7.0,1.0,1.0],
                        [3.0,3.0,3.0,6.0,0.0,0.0,4.0,1.0,1.0,1.0],
                        [6.0,6.0,6.0,6.0,0.0,0.0,4.0,4.0,4.0,4.0]],
                       dtype=np.float64, order='C')

        self.expected_wrapped_rdirs_output_initially_prefer_non_diags = \
            np.asarray([[6.0,6.0,6.0,6.0,0.0,0.0,4.0,4.0,4.0,4.0],
                        [9.0,9.0,9.0,6.0,0.0,0.0,4.0,7.0,7.0,7.0],
                        [9.0,9.0,9.0,6.0,0.0,0.0,4.0,7.0,7.0,7.0],
                        [7.0,4.0,9.0,6.0,0.0,0.0,4.0,7.0,9.0,8.0],
                        [8.0,7.0,9.0,6.0,0.0,0.0,4.0,7.0,6.0,9.0],
                        [8.0,7.0,9.0,6.0,0.0,0.0,4.0,7.0,9.0,9.0],
                        [8.0,7.0,9.0,6.0,0.0,0.0,4.0,7.0,9.0,9.0],
                        [3.0,3.0,9.0,6.0,0.0,0.0,4.0,7.0,1.0,1.0],
                        [3.0,3.0,3.0,6.0,0.0,0.0,4.0,1.0,1.0,1.0],
                        [6.0,6.0,6.0,6.0,0.0,0.0,4.0,4.0,4.0,4.0]],
                       dtype=np.float64, order='C')

        self.expected_wrapped_catchment_nums_with_current_sink_on_slope_and_former_sinks =\
            np.asarray([[ 2, 2, 2, 2,0,0, 4, 4, 4, 4],
                        [ 2, 2, 2, 1,0,0, 3, 4, 4, 4],
                        [ 2, 2, 1, 5,0,0, 6, 3, 4, 4],
                        [27,27, 5, 7,0,0, 8, 6,27,27],
                        [27,27, 7, 9,0,0,10, 8,27,27],
                        [27,27, 9,11,0,0,12,10,27,27],
                        [27,27,11,13,0,0,14,12,27,27],
                        [19,19,13,15,0,0,16,14,20,20],
                        [19,19,19,17,0,0,18,20,20,20],
                        [19,19,19,19,0,0,20,20,20,20]],
                       dtype=np.int32, order='C')

        self.catchment_nums_in = np.zeros((10,10),dtype=np.int32)

    def testBasicCall(self):
        """Test a basic call to fill_sinks using algorithm 4"""
        dummy_ls = np.empty((1,1),dtype=np.int32)
        dummy_ts = np.empty((1,1),dtype=np.int32)
        next_lat_indices = np.empty((10,10),dtype=np.int32)
        next_lon_indices = np.empty((10,10),dtype=np.int32)
        fill_sinks_wrapper.fill_sinks_cpp_func(self.input_array,4,False,dummy_ls,False,False,dummy_ts,
                                               False,0.0,next_lat_indices,next_lon_indices,
                                               self.rdirs,self.catchment_nums_in) #@UndefinedVariable
        np.testing.assert_array_equal(self.catchment_nums_in,self.expected_catchment_num_output,
                                      "Basic Algorithm 4 call not producing expected catchment number")
        np.testing.assert_array_equal(self.input_array, self.expected_output_array,
                                      "Basic Algorithm 4 call unexpectedly changes orography")
        np.testing.assert_array_equal(self.rdirs,self.expected_rdirs_output,
                                      "Basic Algorithm 4 call doesn't return expected river directions")

    def testCallWithLSMask(self):
        """Test a call to fill_sinks using algorithm 4 and a land sea mask"""
        dummy_ts = np.empty((1,1),dtype=np.int32)
        next_lat_indices = np.empty((10,10),dtype=np.int32)
        next_lon_indices = np.empty((10,10),dtype=np.int32)
        fill_sinks_wrapper.fill_sinks_cpp_func(self.wrapped_input_array,4,True,self.ls_mask,False,False,dummy_ts,
                                               False,0.0,next_lat_indices,next_lon_indices,self.rdirs,
                                               self.catchment_nums_in) #@UndefinedVariable
        np.testing.assert_array_equal(self.catchment_nums_in,self.expected_catchment_num_wrapped_output,
                                      "Fill sinks algorithm 4 with land sea mask doesn't produce expected river catchments")
        np.testing.assert_array_equal(self.rdirs,self.expected_wrapped_rdirs_output,
                                      "Fill sinks algorithm 4 with land sea mask doesn't produce expected river direction results")
        np.testing.assert_array_equal(self.wrapped_input_array, self.expected_wrapped_array_output,
                                      "Fill sinks algorithm 4 with land sea mask unexpected changes orography")

    def testCallWithLSMaskPreferNonDiagonalsInSettingInitialFlowDirections(self):
        """Test a call to fill_sinks using algorithm 4, a land sea mask and preferring non-diagonals in setting intitial flow directions"""
        dummy_ts = np.empty((1,1),dtype=np.int32)
        next_lat_indices = np.empty((10,10),dtype=np.int32)
        next_lon_indices = np.empty((10,10),dtype=np.int32)
        fill_sinks_wrapper.fill_sinks_cpp_func(self.wrapped_input_array,4,True,self.ls_mask,False,False,dummy_ts,
                                               False,0.0,next_lat_indices,next_lon_indices,self.rdirs,
                                               self.catchment_nums_in,True)  #@UndefinedVariable
        np.testing.assert_array_equal(self.catchment_nums_in,self.expected_catchment_num_wrapped_output,
                                      "Filling sinks algorithm 4 with land sea mask and prefering non-diagonal flow"
                                      "directions in the initial setup doesn't produce expected catchment areas")
        np.testing.assert_array_equal(self.rdirs,self.expected_wrapped_rdirs_output_initially_prefer_non_diags,
                                      "Filling sinks algorithm 4 with land sea mask and prefering non-diagonal flow"
                                      "directions in the initial setup doesn't produce expected river direction results")
        np.testing.assert_array_equal(self.wrapped_input_array, self.expected_wrapped_array_output,
                                      "Fill sinks algorithm 4 with land sea mask unexpected changes orography")

    def testBasicCallwithBothTrueSinkAndFormerTrueSinks(self):
        """Test a call to fill_sinks using algorithm 4 including both active and inactive true sinks"""
        dummy_ls = np.empty((1,1),dtype=np.int32)
        next_lat_indices = np.empty((10,10),dtype=np.int32)
        next_lon_indices = np.empty((10,10),dtype=np.int32)
        fill_sinks_wrapper.fill_sinks_cpp_func(self.input_array,4,False,dummy_ls,False,True,
                                               self.true_sink_and_former_true_sinks_input_array,
                                               False,0.0,next_lat_indices,next_lon_indices,
                                               self.rdirs,self.catchment_nums_in) #@UndefinedVariable
        np.testing.assert_array_equal(self.catchment_nums_in,
                                      self.expected_catchment_num_output_with_true_and_former_true_sinks,
                                      "Basic Algorithm 4 call (with true and former true sinks) unexpectedly"
                                      " changes catchments")
        np.testing.assert_array_equal(self.input_array, self.expected_output_array,
                                      "Basic Algorithm 4 call (with true and former true sinks) unexpectedly"
                                      " changes orography")
        np.testing.assert_array_equal(self.rdirs,self.expected_rdirs_output_with_true_and_former_true_sinks,
                                      "Basic Algorithm 4 call (with true and former true sinks) doesn't return"
                                      " expected river directions")

    def testLSCallwithFormerTrueSinkWrapped(self):
        """Test a call to fill_sinks using algorithm 4 including an inactive true sink"""
        next_lat_indices = np.empty((10,10),dtype=np.int32)
        next_lon_indices = np.empty((10,10),dtype=np.int32)
        fill_sinks_wrapper.fill_sinks_cpp_func(self.wrapped_input_array,4,True,self.ls_mask,False,True,
                                               self.former_true_sinks_input_array,
                                               False,0.0,next_lat_indices,next_lon_indices,
                                               self.rdirs,self.catchment_nums_in) #@UndefinedVariable
        np.testing.assert_array_equal(self.catchment_nums_in,self.expected_catchment_num_wrapped_output,
                                      "Fill sinks algorithm 4 with land sea mask (and a former true sink) doesn't"
                                      " produce expected river catchments")
        np.testing.assert_array_equal(self.rdirs,self.expected_wrapped_rdirs_output_with_former_sink,
                                      "Fill sinks algorithm 4 with land sea mask (and a former true sink) doesn't"
                                      " produce the expected river direction results")
        np.testing.assert_array_equal(self.wrapped_input_array, self.expected_wrapped_array_output,
                                      "Fill sinks algorithm 4 with land sea mask (and a former true sink) unexpected"
                                      " changes orography")

    def testLSCallWithBothTrueSinkOnSlopeAndFormerTrueSinkWrapped(self):
        """Test a call to fill_sinks using algorithm 4 including both an active (on a slope) and an inactive true sinks"""
        next_lat_indices = np.empty((10,10),dtype=np.int32)
        next_lon_indices = np.empty((10,10),dtype=np.int32)
        fill_sinks_wrapper.fill_sinks_cpp_func(self.wrapped_input_array_with_slope,4,True,self.ls_mask,
                                               False,True,self.wrapped_current_and_former_true_sinks_input_array,
                                               False,0.0,next_lat_indices,next_lon_indices,
                                               self.rdirs,self.catchment_nums_in,True)  #@UndefinedVariable
        np.testing.assert_array_equal(self.catchment_nums_in,
                                      self.expected_wrapped_catchment_nums_with_current_sink_on_slope_and_former_sinks,
                                      "Fill sinks algorithm 4 with land sea mask and prefering non-diagonal flow"
                                      " directions in the initial setup with both true sinks on slopes and former"
                                      " true sinks doesn't produce expected catchment numbers")
        np.testing.assert_array_equal(self.rdirs,
                                      self.expected_wrapped_rdirs_output_with_current_sink_on_slope_and_former_sinks,
                                      "Fill sinks algorithm 4 with land sea mask and prefering non-diagonal flow"
                                      " directions in the initial setup with both true sinks on slopes and former"
                                      " true sinks doesn't produce expected river direction results")
        np.testing.assert_array_equal(self.wrapped_input_array_with_slope, self.wrapped_output_array_with_slope,
                                      "Fill sinks algorithm 4 with land sea mask and prefering non-diagonal flow"
                                      " directions in the initial setup with both true sinks on slopes and former"
                                      " true sinks unexpectedly changes orography")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
