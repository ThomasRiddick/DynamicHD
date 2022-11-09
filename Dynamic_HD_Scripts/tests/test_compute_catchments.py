'''
Unit tests for the compute_catchments module

Created on Jan 29, 2016

@author: thomasriddick
'''

import unittest
import numpy as np
import textwrap
import os
from Dynamic_HD_Scripts.tools import compute_catchments as cc
from Dynamic_HD_Scripts.interface.fortran_interface import f2py_manager as f2py_mg
from Dynamic_HD_Script_Tests.context import fortran_source_path,data_dir

class RelabelCatchmentTestCase(unittest.TestCase):
    """Class containing tests of the relabelling of catchments into desceding order of size"""

    original_catchments = np.array([[1,2,3,3,3],
                                    [1,2,2,3,4],
                                    [1,2,3,3,5],
                                    [2,3,6,5,7],
                                    [2,8,3,9,7]],order='F',dtype=np.int32)
    old_to_new_label_map = np.array([0,140,78, 29,
                                      39,44,392,
                                      12, 3,-11])
    expected_results = np.array([[140,78,29,29,29],
                                 [140,78,78,29,39],
                                 [140,78,29,29,44],
                                 [78,29,392,44,12],
                                 [78,3,29,-11,12]],order='F',dtype=np.int32)
    expected_results_for_python_wrapper = np.array([[3,2,1,1,1],
                                                    [3,2,2,1,9],
                                                    [3,2,1,1,5],
                                                    [2,1,8,5,4],
                                                    [2,7,1,6,4]],order='F',dtype=np.int32)

    loop_logfile = os.path.join(data_dir,"temp/loop_log.txt")
    input_loop_list = ['1','2','3','6','7','9']
    loop_list_expected_results = ['3','2','1','8','4','6']

    def setUp(self):
        """Unit test setup function"""
        try:
            os.remove(self.loop_logfile)
        except OSError:
            pass
        with open(self.loop_logfile,'w') as f:
            f.write('Loops found in catchments:\n')
            f.writelines([loop+'\n' for loop in self.input_loop_list])

    def testRenumbering(self):
        """Test the core Fortran renumbering subroutine"""
        f2py_mngr = f2py_mg.f2py_manager(os.path.join(fortran_source_path,
                                                      "mod_compute_catchments.f90"),
                                         func_name="relabel_catchments")
        input_catchments = np.copy(self.original_catchments)
        f2py_mngr.run_current_function_or_subroutine(input_catchments,
                                                     self.old_to_new_label_map)
        np.testing.assert_equal(input_catchments,self.expected_results,
                                "Catchment renumbering not producing expected results")

    def testRenumberCatchmentsBySize(self):
        """Test the top level python renumbering function"""
        input_catchments = np.copy(self.original_catchments)
        renumbered_catchments = cc.renumber_catchments_by_size(input_catchments,
                                                               self.loop_logfile)
        np.testing.assert_array_equal(renumbered_catchments,
                                      self.expected_results_for_python_wrapper,
                                      "Catchment renumbering not producing expected results")

    def testLoopsRelabellingLoops(self):
        """Test the relabelling of loops found to reflect the new labels"""
        input_catchments = np.copy(self.original_catchments)
        cc.renumber_catchments_by_size(input_catchments,
                                       self.loop_logfile)
        with open(self.loop_logfile,'r') as f:
            next(f)
            new_loops_list = [loop.strip() for loop in f]
        np.testing.assert_array_equal(new_loops_list,
                                      self.loop_list_expected_results)

class compute_catchments(unittest.TestCase):
    """Perform general tests of catchment computation"""

    rivdir_test_data = np.swapaxes(np.array([[7,2,6,6,5,8,0,1,1],
                                             [4,3,6,9,8,9,8,4,1],
                                             [2,6,2,9,8,9,8,2,1],
                                             [5,6,2,9,8,7,8,2,2],
                                             [7,4,2,2,8,9,2,3,2],
                                             [8,2,3,2,3,1,3,8,2],
                                             [3,2,1,2,2,2,1,2,1],
                                             [3,2,1,3,2,1,2,3,6],
                                             [1,5,6,6,5,4,2,6,-1]]),0,1)

    rivdir_test_data_cpp = np.swapaxes(np.array([[7,2,6,6,5,8,0,1,1],
                                                 [4,3,6,9,8,9,8,4,1],
                                                 [2,6,2,9,8,9,8,2,1],
                                                 [5,6,2,9,8,7,8,2,2],
                                                 [7,4,2,2,8,9,2,3,2],
                                                 [8,2,3,2,3,1,3,8,2],
                                                 [3,2,1,2,2,2,1,2,1],
                                                 [3,2,1,3,2,1,2,3,6],
                                                 [1,5,6,6,5,4,2,6,0]]),0,1)

    expected_catchment_output = np.swapaxes(np.array([[1,6,7,7,7,8,9,9,9],
                                                      [2,6,7,7,7,9,9,9,2],
                                                      [3,6,6,7,7,9,9,2,2],
                                                      [3,6,6,7,7,7,9,2,2],
                                                      [2,2,6,6,7,9,2,2,2],
                                                      [2,4,6,6,6,6,2,2,2],
                                                      [4,4,4,6,6,6,6,2,2],
                                                      [4,4,4,6,6,6,10,2,4],
                                                      [5,4,6,6,6,6,10,2,2]]),0,1)

    expected_catchment_output_cpp = np.swapaxes(np.array([[7,5,1,1,1,8,2,2,2],
                                                          [6,5,1,1,1,2,2,2,6],
                                                          [3,5,5,1,1,2,2,6,6],
                                                          [3,5,5,1,1,1,2,6,6],
                                                          [6,6,5,5,1,2,6,6,6],
                                                          [6,4,5,5,5,5,6,6,6],
                                                          [4,4,4,5,5,5,5,6,6],
                                                          [4,4,4,5,5,5,9,6,4],
                                                          [10,4,5,5,5,5,9,6,6]]),0,1)

    expected_sink_type_count_output = np.array([1,1,4,0,4,0])
    loop_logfile = os.path.join(data_dir,"temp/loop_log.txt")
    loop_logfile_cpp = os.path.join(data_dir,"temp/loop_log_cpp.txt")

    def setUp(self):
        """Unit test setUp function"""
        self.f2py_mngr = f2py_mg.f2py_manager(os.path.join(fortran_source_path,
                                                           "mod_compute_catchments.f90"),
                                              func_name="compute_catchments")

    def testHypotheticalCatchment(self):
        """Test the key fortran function using some hypothetical river flow direction data"""
        sink_types_found, catchments_field = \
            self.f2py_mngr.run_current_function_or_subroutine(self.rivdir_test_data,2,
                                                              self.loop_logfile)
        np.testing.assert_array_equal(sink_types_found,
                                      self.expected_sink_type_count_output,
                                      "Sink types found count doesn't match expectation")
        np.testing.assert_array_equal(catchments_field,
                                      self.expected_catchment_output,
                                      "Catchments calculated for tests data don't match expectation")

    def testHypotheticalCatchmentUsingPythonWrapper(self):
        """Test the python wrapper/helper to the compute_catchments FORTRAN subroutine"""
        sink_types_found, catchments_field = cc.compute_catchments(np.swapaxes(self.rivdir_test_data,0,1),
                                                                   self.loop_logfile,
                                                                   circ_flow_check_period=2)
        np.testing.assert_array_equal(sink_types_found,
                                      self.expected_sink_type_count_output,
                                      "Sink types found count doesn't match expectation")
        np.testing.assert_array_equal(catchments_field,
                                      np.swapaxes(self.expected_catchment_output,0,1),
                                      "Catchments calculated for tests data don't match expectation")

    def testHypotheticalCatchmentUsingPythonWrapperCppVersion(self):
        """Test the calculating these catchment with the c++ code"""
        catchments_field = cc.compute_catchments_cpp(np.swapaxes(self.rivdir_test_data_cpp,0,1),
                                                     self.loop_logfile_cpp)
        np.testing.assert_array_equal(catchments_field,
                                      np.swapaxes(self.expected_catchment_output_cpp,0,1),
                                      "Catchments calculated for tests data don't match expectation")

class follow_river(unittest.TestCase):
    """Functions to test the follow_river subroutine"""

    rivdir_test_data = np.swapaxes(np.array([[7,2,6,6,5,8,0,1,1],
                                             [4,3,6,9,8,9,8,4,1],
                                             [2,6,2,9,8,9,8,2,1],
                                             [2,6,2,9,8,7,8,2,2],
                                             [5,4,2,2,8,9,2,3,2],
                                             [8,2,3,2,3,1,3,8,2],
                                             [3,2,1,2,2,2,1,2,1],
                                             [3,2,1,3,2,1,2,3,6],
                                             [1,5,6,6,5,4,2,6,-1]]),0,1)
    single_river_test_expected_output = np.swapaxes(np.array([[0,0,0,0,0,0,0,0,0],
                                                              [0,0,0,0,0,0,0,0,0],
                                                              [0,0,-18,0,0,0,0,0,0],
                                                              [0,0,-18,0,0,0,0,0,0],
                                                              [0,0,-18,0,0,0,0,0,0],
                                                              [0,0,-18,0,0,0,0,0,0],
                                                              [0,0,0,-18,0,0,0,0,0],
                                                              [0,0,0,-18,0,0,0,0,0],
                                                              [0,0,0,0,-18,0,0,0,0]]),0,1)

    adding_tributary_test_expected_output = np.swapaxes(np.array([[0,0,0,0,0,0,0,0,0],
                                                                  [0,0,0,0,0,0,0,0,0],
                                                                  [0,0,-18,0,0,0,0,0,0],
                                                                  [0,0,-18,0,0,0,0,0,0],
                                                                  [0,0,-18,-18,0,0,0,0,0],
                                                                  [0,0,-18,-18,0,0,0,0,0],
                                                                  [0,0,0,-18,0,0,0,0,0],
                                                                  [0,0,0,-18,0,0,0,0,0],
                                                                  [0,0,0,0,-18,0,0,0,0]]),0,1)
    left_right_boundary_wrap_test_expected_output = np.swapaxes(np.array([[0,0,0,0,0,0,0,0,0],
                                                                          [-18,0,0,0,0,0,0,0,-18],
                                                                          [0,0,0,0,0,0,0,-18,0],
                                                                          [0,0,0,0,0,0,0,-18,0],
                                                                          [0,0,0,0,0,0,0,-18,0],
                                                                          [0,0,0,0,0,0,0,0,-18],
                                                                          [0,0,0,0,0,0,0,0,-18],
                                                                          [0,0,0,0,0,0,0,-18,0],
                                                                          [0,0,0,0,0,0,0,0,-18]]),0,1)
    right_left_boundary_wrap_test_expected_output = np.swapaxes(np.array([[0,0,0,0,0,0,0,0,0],
                                                                          [0,0,0,0,0,0,0,0,0],
                                                                          [0,0,0,0,0,0,0,0,0],
                                                                          [0,0,0,0,0,0,0,0,0],
                                                                          [0,0,0,0,0,0,0,0,0],
                                                                          [0,0,0,0,0,0,0,0,0],
                                                                          [0,0,0,0,0,0,0,0,0],
                                                                          [-18,0,0,0,0,0,0,0,-18],
                                                                          [0,-18,0,0,0,0,0,0,0]]),0,1)
    flow_over_poles_set_to_sink_test_expected_output = np.swapaxes(np.array([[-18,0,0,0,0,-17,0,0,0],
                                                                             [0,0,0,0,0,0,0,0,0],
                                                                             [0,0,0,0,0,0,0,0,0],
                                                                             [0,0,0,0,0,0,0,0,0],
                                                                             [0,0,0,0,0,0,0,0,0],
                                                                             [0,0,0,0,0,0,0,0,0],
                                                                             [0,0,0,0,0,0,0,0,0],
                                                                             [0,0,0,0,0,0,-15,0,0],
                                                                             [-16,0,0,0,0,0,-15,0,0]]),0,1)

    rivdir_test_data_circular_flow = np.swapaxes(np.array([[6,6,2],
                                                           [8,5,2],
                                                           [8,4,4]]),0,1)

    rivdir_test_data_second_circular_flow = np.swapaxes(np.array([[6,6,2],
                                                                  [8,9,2],
                                                                  [8,8,4]]),0,1)
    circular_flow_expected_output = np.swapaxes(np.array([[-18,-18,-18],
                                                          [-18,0,-18],
                                                          [-18,-18,-18]]),0,1)

    second_circular_flow_expected_output = np.swapaxes(np.array([[-18,-18,-18],
                                                                 [-18,-18,-18],
                                                                 [-18,-18,-18]]),0,1)

    loop_logfile = os.path.join(data_dir,"temp/loop_log.txt")

    def setUp(self):
        """Unit test setUp function"""
        self.f2py_mngr = f2py_mg.f2py_manager(os.path.join(fortran_source_path,
                                                           "mod_compute_catchments.f90"),
                                              func_name="follow_river")
        self.catchment_number = np.asarray(-18)
        self.catchment_field = np.zeros((9,9),order='F',dtype=np.int32)
        try:
            os.remove(self.loop_logfile)
        except OSError:
            pass
        with open(self.loop_logfile,'w') as f:
            f.write('Loops found in catchments:\n')

    def testFirstRiver(self):
        """Test follow the first river found"""
        sink_type = self.f2py_mngr.run_current_function_or_subroutine(self.rivdir_test_data,
                                                                      self.catchment_field,
                                                                      self.catchment_number,
                                                                      3,3,2,self.loop_logfile,
                                                                      9,9)
        np.testing.assert_array_equal(self.catchment_field,
                                      self.single_river_test_expected_output,
                                      "Failed to follow a single river correctly")
        self.assertEqual(self.catchment_number,-17,
                         "Catchment Number not incremented correctly")
        self.assertEqual(sink_type, 3, "Sink type not set correctly")

    def testAddingTributary(self):
        """Test following a tributary that merges with another river"""
        self.f2py_mngr.run_current_function_or_subroutine(self.rivdir_test_data,
                                                          self.catchment_field,
                                                          self.catchment_number,
                                                          3,3,2,self.loop_logfile,
                                                          9,9)
        self.assertEqual(self.catchment_number,-17,
                         "Catchment Number not incremented correctly")
        sink_type = self.f2py_mngr.run_current_function_or_subroutine(self.rivdir_test_data,
                                                                      self.catchment_field,
                                                                      self.catchment_number,
                                                                      4,5,2,self.loop_logfile,
                                                                      9,9)
        np.testing.assert_array_equal(self.catchment_field,
                                      self.adding_tributary_test_expected_output,
                                      "Failed to add tributary correctly")
        self.assertEqual(self.catchment_number,-17,
                         "Catchment Number incremented incorrectly")
        self.assertEqual(sink_type, 0, "Sink type not set correctly")

    def testCircularFlow(self):
        """Test finding a circular flow"""
        circular_flow_catchment_field = np.zeros((3,3),order='F',dtype=np.int32)
        sink_type = self.f2py_mngr.run_current_function_or_subroutine(self.rivdir_test_data_circular_flow,
                                                                      circular_flow_catchment_field,
                                                                      self.catchment_number,
                                                                      1,2,2,self.loop_logfile,
                                                                      3,3)
        np.testing.assert_array_equal(circular_flow_catchment_field,
                                      self.circular_flow_expected_output,
                                      "Circular flow doesn't give correct catchment")
        self.assertEqual(sink_type, 6, "Sink type not set correctly")

    def testAnotherCircularFlow(self):
        """Test finding a second circular flow"""
        circular_flow_catchment_field = np.zeros((3,3),order='F',dtype=np.int32)
        sink_type = self.f2py_mngr.run_current_function_or_subroutine(self.rivdir_test_data_second_circular_flow,
                                                                      circular_flow_catchment_field,
                                                                      self.catchment_number,
                                                                      1,3,2,self.loop_logfile,
                                                                      3,3)
        np.testing.assert_array_equal(circular_flow_catchment_field,
                                      self.second_circular_flow_expected_output,
                                      "Second circular flow doesn't give correct catchment")
        self.assertEqual(sink_type, 6, "Sink type not set correctly")

    def testCircularFlowFileOutput(self):
        """Test printing out the catchment number of a loop to a log file"""
        circular_flow_catchment_field = np.zeros((3,3),order='F',dtype=np.int32)
        self.f2py_mngr.run_current_function_or_subroutine(self.rivdir_test_data_circular_flow,
                                                          circular_flow_catchment_field,
                                                          self.catchment_number,
                                                          1,2,2,self.loop_logfile,
                                                          3,3)
        #load and check the loop
        with open(self.loop_logfile) as f:
            next(f)
            loops = [int(line.strip()) for line in f]
        self.assertListEqual(loops,[-18],
                             "Circular flow diagnostic information is not being produced correctly")

    def testAnotherCircularFlowFileOutput(self):
        """Test printing out the catchment number of another loop to a log file"""
        circular_flow_catchment_field = np.zeros((3,3),order='F',dtype=np.int32)
        self.f2py_mngr.run_current_function_or_subroutine(self.rivdir_test_data_second_circular_flow,
                                                          circular_flow_catchment_field,
                                                          self.catchment_number,
                                                          1,3,2,self.loop_logfile,
                                                          3,3)
        #load and check the loop
        with open(self.loop_logfile) as f:
            next(f)
            loops = [int(line.strip()) for line in f]
        self.assertListEqual(loops,[-18],
                             "Circular flow diagnostic information is not being produced correctly")

    def testLeftRightCrossBorderWrap(self):
        """Test left right flow across border"""
        sink_type = self.f2py_mngr.run_current_function_or_subroutine(self.rivdir_test_data,
                                                                      self.catchment_field,
                                                                      self.catchment_number,
                                                                      1,2,2,self.loop_logfile,
                                                                      9,9)
        np.testing.assert_array_equal(self.catchment_field,
                                      self.left_right_boundary_wrap_test_expected_output,
                                      "Cross border wrapping from left to right failed")
        self.assertEqual(self.catchment_number,-17,
                         "Catchment Number incremented incorrectly")
        self.assertEqual(sink_type, 2, "Sink type not set correctly")

    def testRightLeftCrossBorderWrap(self):
        """Test right left flow across border"""
        sink_type = self.f2py_mngr.run_current_function_or_subroutine(self.rivdir_test_data,
                                                                      self.catchment_field,
                                                                      self.catchment_number,
                                                                      9,8,2,self.loop_logfile,
                                                                      9,9)
        self.assertEqual(self.catchment_number,-17,
                         "Catchment Number incremented incorrectly")
        np.testing.assert_array_equal(self.catchment_field,
                                      self.right_left_boundary_wrap_test_expected_output,
                                      "Cross border wrapping from left to right failed")
        self.assertEqual(sink_type, 3, "Sink type not set correctly")

    def testSetFlowOverPolesToSink(self):
        """Test setting flow over poles to a sink"""
        sink_type= self.f2py_mngr.run_current_function_or_subroutine(self.rivdir_test_data,
                                                                     self.catchment_field,
                                                                     self.catchment_number,
                                                                     1,1,2,self.loop_logfile,
                                                                     9,9)
        self.assertEqual(sink_type, 5, "Sink type not set correctly")
        sink_type= self.f2py_mngr.run_current_function_or_subroutine(self.rivdir_test_data,
                                                                     self.catchment_field,
                                                                     self.catchment_number,
                                                                     6,1,2,self.loop_logfile,
                                                                     9,9)
        self.assertEqual(sink_type, 5, "Sink type not set correctly")
        sink_type= self.f2py_mngr.run_current_function_or_subroutine(self.rivdir_test_data,
                                                                     self.catchment_field,
                                                                     self.catchment_number,
                                                                     1,9,2,self.loop_logfile,
                                                                     9,9)
        self.assertEqual(sink_type, 5, "Sink type not set correctly")
        sink_type= self.f2py_mngr.run_current_function_or_subroutine(self.rivdir_test_data,
                                                                     self.catchment_field,
                                                                     self.catchment_number,
                                                                     7,8,2,self.loop_logfile,
                                                                     9,9)
        self.assertEqual(sink_type, 5, "Sink type not set correctly")
        self.assertEqual(self.catchment_number,-14,
                         "Catchment Number incremented incorrectly")
        np.testing.assert_array_equal(self.catchment_field,
                                      self.flow_over_poles_set_to_sink_test_expected_output,
                                      'Failed to handle flow over poles correctly')

class LabelCatchmentTestCase(unittest.TestCase):
    """Tests for the catchment labelling routine"""

    #Note column are first axis in Fortran
    singlepointexpected_output = np.array([[0,0,0,0,0,0,0,0,0],
                                           [0,0,0,0,0,0,0,0,0],
                                           [0,0,0,0,0,0,0,0,0],
                                           [0,0,0,0,0,0,17,0,0],
                                           [0,0,0,0,0,0,0,0,0],
                                           [0,0,0,0,0,0,0,0,0],
                                           [0,0,0,0,0,0,0,0,0],
                                           [0,0,0,0,0,0,0,0,0],
                                           [0,0,0,0,0,0,0,0,0]])

    severalpointsexpected_output = np.array([[0, 0, 0,-7, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0,-7, 0,-7, 0, 0],
                                             [0, 0, 0, 0,-7,-7,-7, 0, 0],
                                             [0, 0, 0, 0, 0, 0,-7,-7,-7],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    def setUp(self):
        """Unit test setUp function"""
        self.catchment_field = np.zeros((9,9),order='F',dtype=np.int32)
        self.grid_points_in_catchment = np.zeros((2,9*9),order='F',dtype=np.int32)
        self.f2py_mngr = f2py_mg.f2py_manager(os.path.join(fortran_source_path,
                                                           "mod_compute_catchments.f90"),
                                              func_name="label_catchment")

    def testWithSinglePointInList(self):
        """Test labelling of a single catchment"""
        self.grid_points_in_catchment[:,0] = [4,7]
        self.f2py_mngr.run_current_function_or_subroutine(17,self.catchment_field,
                                                          self.grid_points_in_catchment,1)
        np.testing.assert_almost_equal(self.singlepointexpected_output,
                                       self.catchment_field,
                                       err_msg="Single point catchment is being labelled incorrectly")

    def testWithSeveralPointsInList(self):
        """Test labelling of several catchments"""
        self.grid_points_in_catchment[:,0:9] = np.swapaxes([[4,7],[4,8],[4,9],[3,7],[3,6],[3,5],[2,7],[2,5],[1,4]],0,1)
        self.f2py_mngr.run_current_function_or_subroutine(-7,self.catchment_field,
                                                          self.grid_points_in_catchment,9)
        np.testing.assert_almost_equal(self.severalpointsexpected_output,
                                       self.catchment_field,
                                       err_msg="Single point catchment is being labelled incorrectly")

class ComputeNextGridCellTestCase(unittest.TestCase):
    """Tests of the subroutine to compute the next grid cell"""

    river_flow_direction_input_values = [[7,8,9],
                                         [4,5,6],
                                         [1,2,3]]

    expected_output_coords_changes = [[ [-1,-1], [0,-1], [1,-1]],
                                      [ [-1, 0], [0, 0], [1, 0]],
                                      [ [-1, 1], [0, 1], [1, 1]]]

    def setUp(self):
        """Unit test setUp function"""
        self.f2py_mngr = f2py_mg.f2py_manager(os.path.join(fortran_source_path,
                                                           "mod_compute_catchments.f90"),
                                              func_name="compute_next_grid_cell")
        self.i = 52
        self.j = 47
        self.iasarray = np.array(self.i)
        self.jasarray = np.array(self.j)
        self.expected_output_coords = [[[base+mod for base,mod in zip([self.i,self.j],coordmods)] #@UndefinedVariable
                                          for coordmods in row] for row in self.expected_output_coords_changes] #@UnusedVariable

    def SinkPointReponseTestHelper(self,value,expected_return_value):
        """Help test the response to various sink types"""
        self.assertEqual(self.f2py_mngr.run_current_function_or_subroutine(value,self.i,self.j),
                    expected_return_value,
                    "Code {0} does not return sink type {1}".format(value,expected_return_value))
        self.assertEqual(self.iasarray,52,"Sink not returning original co-ordinates")
        self.assertEqual(self.jasarray,47,"Sink not returning original co-ordinates")

    def testOceanPointResponse(self):
        """Test the response to an ocean point"""
        self.SinkPointReponseTestHelper(-1,2)

    def testCoastPointResponse(self):
        """Test the reponse to a coastal point"""
        self.SinkPointReponseTestHelper(0,1)

    def testLocalSinkPointResponse(self):
        """Test the response to a local sink"""
        self.SinkPointReponseTestHelper(5,3)

    def testUnknownRiverDirectionResponse(self):
        """Test the reponse to an unknown river direction"""
        self.SinkPointReponseTestHelper(10,4)
        self.SinkPointReponseTestHelper(-177,4)

    def testRiverFlowDirectionResponse(self):
        """Test the reponse to the various possible flow directions"""
        for inputrow,outputrow in zip(self.river_flow_direction_input_values,
                                      self.expected_output_coords):
            for value,expected_output_value in zip(inputrow,outputrow):
                iasarray_local= np.copy(self.iasarray)
                jasarray_local= np.copy(self.jasarray)
                self.assertEqual(self.f2py_mngr.run_current_function_or_subroutine(value,iasarray_local,jasarray_local),
                                    (lambda value: 3 if value == 5 else 0)(value))
                self.assertListEqual([int(iasarray_local),int(jasarray_local)],expected_output_value,textwrap.dedent("""\
                                     Compute next grid cell returns coordinates {0} for value {1} that do not
                                     match expected coordinate values {2}""").format([int(iasarray_local),int(jasarray_local)],
                                                                                     value,expected_output_value))

if __name__ == "__main__":
    unittest.main()
