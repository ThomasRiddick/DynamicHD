'''
Unit tests for the flow_to_grid_cell module.
Created on Jan 20, 2016

@author: thomasriddick
'''

import unittest
import os
import numpy as np
from abc import ABCMeta, abstractmethod
from Dynamic_HD_Scripts.tools import flow_to_grid_cell
from Dynamic_HD_Scripts.interface.fortran_interface import f2py_manager
from Dynamic_HD_Script_Tests.context import fortran_source_path

class CreateHypotheticalRiverPathsMap(unittest.TestCase):
    """Tests of high level routine using small river directions maps"""

    flow_dirs =  np.array([[3,1,4,4,4,4],
                           [2,4,4,4,7,7],
                           [3,9,8,7,6,2],
                           [5,3,6,9,2,1],
                           [6,7,2,3,1,8],
                           [5,2,7,1,4,4]])

    flow_dirs_with_wrap =  np.array([[1,1,4,4,4,3],
                                     [2,4,4,4,7,7],
                                     [3,9,8,7,6,2],
                                     [4,3,6,9,5,4],
                                     [6,7,2,3,1,8],
                                     [7,2,7,1,4,9]])

    flow_dirs_with_loop = np.array([[1,1,4,4,4,6,3,5],
                                    [2,4,4,4,7,8,4,4],
                                    [3,9,8,7,6,8,5,8],
                                    [4,3,6,9,5,8,5,5],
                                    [6,7,2,3,1,8,7,5],
                                    [7,2,7,1,4,9,8,4],
                                    [5,5,5,5,5,5,5,5],
                                    [5,5,5,5,5,5,5,5]])

    ls_mask = np.array([[1,1,1,1,1,1],
                        [1,1,0,0,0,1],
                        [0,1,1,0,0,0],
                        [0,0,0,0,0,0],
                        [1,0,0,0,0,1],
                        [1,1,1,0,0,0]])

    expected_paths_map = np.array([[ 1, 7, 6, 5, 3, 1],
                                   [15, 7, 5, 1, 1, 1],
                                   [16, 1, 1, 1, 3, 4],
                                   [22,17, 1, 2, 1, 6],
                                   [ 1,21,18, 1, 8, 1],
                                   [ 1, 1,19,12, 3, 1]])

    expected_paths_map_with_wrap = np.array([[ 1, 7, 6, 5, 3, 1],
                                             [ 15, 6, 5, 1, 1, 2],
                                             [ 16, 1, 1, 1, 3, 4],
                                             [ 23, 17, 1, 2, 31, 30],
                                             [ 2, 22, 18, 1, 1, 2],
                                             [ 1, 1, 19, 4, 2, 1]])

    expected_paths_map_when_using_mask = np.array([[1, 1, 1, 2, 1, 1],
                                                   [1, 4, 3, 1, 1, 1],
                                                   [1, 1, 1, 1, 3, 4],
                                                   [2, 2, 1, 2, 1, 5],
                                                   [1, 1, 3, 1, 7, 1],
                                                   [1, 1, 4,11, 3, 1]])

    expected_paths_map_with_loop = np.array([[1, 5, 4, 3, 1, 0, 0, 1],
                                             [12, 6, 5, 1, 1,0, 0, 0],
                                             [13, 1, 1, 1, 3,10, 1, 1],
                                             [19, 14, 1, 2, 1, 6, 1, 20],
                                             [1, 18, 15, 1, 1, 1, 4, 2],
                                             [1, 1, 16, 4, 2, 1, 2, 1],
                                             [1, 2, 5, 1, 1, 1, 1, 1],
                                             [1, 1, 1, 1, 1, 1, 1, 1]])

    def testCreatingHypotheticalRiverMapsNoMask(self):
        """Test high level routine using small river direction map with no mask"""
        paths_maps = flow_to_grid_cell.create_hypothetical_river_paths_map(self.flow_dirs,
                                                                           lsmask=None,
                                                                           use_f2py_func=True,
                                                                           use_f2py_sparse_iterator=False,
                                                                           nlat=6,
                                                                           nlong=6)
        np.testing.assert_array_equal(paths_maps,
                                      self.expected_paths_map,
                                      "Paths map created doesn't match expectation")

    def testCreatingHypotheticalRiverMapsNoMaskWrapped(self):
        """Test high level routine using small river direction map (with directions wrapping E-W) with no mask"""
        paths_maps = flow_to_grid_cell.create_hypothetical_river_paths_map(self.flow_dirs_with_wrap,
                                                                           lsmask=None,
                                                                           use_f2py_func=True,
                                                                           use_f2py_sparse_iterator=False,
                                                                           nlat=6,
                                                                           nlong=6)
        np.testing.assert_array_equal(paths_maps,
                                      self.expected_paths_map_with_wrap,
                                      "Paths map created doesn't match expectation")

    def testCreatingHypotheticalRiverMapsNoMaskWrappedWithSparseIterator(self):
        """Test high level routine using small river direction map (with directions wrapping E-W) with no mask"""
        paths_maps = flow_to_grid_cell.create_hypothetical_river_paths_map(self.flow_dirs_with_wrap,
                                                                           lsmask=None,
                                                                           use_f2py_func=True,
                                                                           use_f2py_sparse_iterator=True,
                                                                           nlat=6,
                                                                           nlong=6)
        np.testing.assert_array_equal(paths_maps,
                                      self.expected_paths_map_with_wrap,
                                      "Paths map created doesn't match expectation")

    def testCreatingHypotheticalRiverMapsWithMask(self):
        """Test high level routine using small river direction map with a mask"""
        paths_maps = flow_to_grid_cell.create_hypothetical_river_paths_map(self.flow_dirs,
                                                                           lsmask=self.ls_mask,
                                                                           use_f2py_func=True,
                                                                           use_f2py_sparse_iterator=False,
                                                                           nlat=6,
                                                                           nlong=6)
        np.testing.assert_array_equal(paths_maps,
                                      self.expected_paths_map_when_using_mask,
                                      "Paths map created using land-sea mask doesn't match expectation")

    def testCreatingHypotheticalRiverMapsNoMaskWithSparseIterator(self):
        """Test high level routine using the sparse iterator without a mask"""
        paths_maps = flow_to_grid_cell.create_hypothetical_river_paths_map(self.flow_dirs,
                                                                           lsmask=None,
                                                                           use_f2py_func=True,
                                                                           use_f2py_sparse_iterator=True,
                                                                           nlat=6,
                                                                           nlong=6)
        np.testing.assert_array_equal(paths_maps,
                                      self.expected_paths_map,
                                      "Paths map created doesn't match expectation")

    def testCreatingHypotheticalRiverMapsWithMaskAndSparseIterator(self):
        """Test high level routine using the sparse iterator with a mask"""
        paths_maps = flow_to_grid_cell.create_hypothetical_river_paths_map(self.flow_dirs,
                                                                           lsmask=self.ls_mask,
                                                                           use_f2py_func=True,
                                                                           use_f2py_sparse_iterator=True,
                                                                           nlat=6,
                                                                           nlong=6)
        np.testing.assert_array_equal(paths_maps,
                                      self.expected_paths_map_when_using_mask,
                                      "Paths map created using land-sea mask doesn't match expectation")

    def testCreatingHypotheticalRiverMapsWithLoop(self):
        """Test high level routine including on a loop"""
        paths_maps = flow_to_grid_cell.create_hypothetical_river_paths_map(self.flow_dirs_with_loop,
                                                                           use_f2py_func=True,
                                                                           nlat=8,
                                                                           nlong=8,
                                                                           use_new_method=True)
        np.testing.assert_array_equal(paths_maps,
                                      self.expected_paths_map_with_loop,
                                      "Paths map created including loop doesn't match expectation")



class MainTestCaseHelper(object, metaclass=ABCMeta):
    """Helper class defining test helpers for the iterator"""

    flow_dirs =  np.array([[ 0,0,0,0,0,0 ],
                           [ 3,1,4,4,4,4 ],
                           [ 2,4,4,4,7,7 ],
                           [ 3,9,8,7,6,2 ],
                           [ 5,3,6,9,2,1 ],
                           [ 6,7,2,3,1,8 ],
                           [ 5,2,7,1,4,4 ],
                           [ 0,0,0,0,0,0 ]])

    expected_first_step_inflows = np.array([[ 1,1,1,1,1,1 ],
                                            [ 1,0,0,0,0,1 ],
                                            [ 0,0,0,1,1,1 ],
                                            [ 0,1,1,1,0,0 ],
                                            [ 0,0,1,2,1,0 ],
                                            [ 1,0,0,1,0,1 ],
                                            [ 1,1,0,0,0,1 ],
                                            [ 1,1,1,1,1,1 ]])

    expected_second_step_inflows = np.array([[ 1,1,1,1,1,1 ],
                                             [ 1,0,0,0,3,1 ],
                                             [ 0,0,5,1,1,1 ],
                                             [ 0,1,1,1,3,4 ],
                                             [ 0,0,1,2,1,6 ],
                                             [ 1,0,0,1,8,1 ],
                                             [ 1,1,0,0,3,1 ],
                                             [ 1,1,1,1,1,1 ]])

    expected_final_step_inflows = np.array([[ 1, 1, 1, 1, 1, 1 ],
                                            [ 1, 7, 6, 5, 3, 1 ],
                                            [15, 7, 5, 1, 1, 1 ],
                                            [16, 1, 1, 1, 3, 4 ],
                                            [22,17, 1, 2, 1, 6 ],
                                            [ 1,21,18, 1, 8, 1 ],
                                            [ 1, 1,19,12, 3, 1 ],
                                            [ 1, 1, 1, 1, 1, 1 ]])

    first_step_kernel_input_paths_map_section =  np.array([[0,0,1],
                                                           [0,0,0],
                                                           [0,0,0]])

    expected_first_step_kernel_output_for_source_cell = np.array([[0,0,1],
                                                                  [0,1,0],
                                                                  [0,0,0]])

    expected_first_step_kernel_output_for_non_source_cell = np.array([[0,0,1],
                                                                      [0,0,0],
                                                                      [0,0,0]])

    first_step_kernel_flow_dirs_for_source_cell = np.array([[2,1,4],
                                                            [5,5,6],
                                                            [8,9,4]])

    first_step_kernel_flow_dirs_for_non_source_cell = np.array([[2,1,1],
                                                                [6,1,1],
                                                                [8,9,4]])

    later_step_kernel_input_paths_map_section_for_source_cell =  np.array([[1,0,1],
                                                                           [4,1,0],
                                                                           [1,2,1]])

    later_step_kernel_input_paths_map_section_for_non_source_cell =  np.array([[1,0,1],
                                                                               [4,0,0],
                                                                               [1,2,1]])

    later_step_kernel_input_paths_map_section_for_unready_non_source_cell =  np.array([[1,0,1],
                                                                                       [0,0,0],
                                                                                       [5,2,1]])

    expected_later_step_kernel_output_for_source_cell = np.array([[1,0,1],
                                                                  [4,1,0],
                                                                  [1,2,1]])

    expected_later_step_kernel_output_for_non_source_cell = np.array([[1,0,1],
                                                                      [4,6,0],
                                                                      [1,2,1]])

    expected_later_step_kernel_output_for_unready_non_source_cell = np.array([[1,0,1],
                                                                              [0,0,0],
                                                                              [5,2,1]])

    later_step_kernel_flow_dirs_for_source_cell = np.array([[2,1,4],
                                                            [5,5,6],
                                                            [8,9,4]])

    later_step_kernel_flow_dirs_for_non_source_cell = np.array([[2,1,1],
                                                                [6,1,1],
                                                                [8,9,4]])


    def MainIteratorFirstIterationTestHelper(self,test_input_func):
        """Helper function for the testing the first pass of the iterator"""
        paths_map = np.zeros((8,6),dtype=np.int32,order='F')
        test_input_func(self.flow_dirs,paths_map,6,6)
        np.testing.assert_array_equal(paths_map,self.expected_first_step_inflows,
                                      "Iterator produces wrong results for first step")

    @abstractmethod
    def testMainIteratorFirstIteration(self):
        """Test the first pass of the iterator across the river flow directions"""
        pass

    def MainIteratorSecondIterationTestHelper(self,test_input_func):
        """Helper function for the testing the second pass of the iterator"""
        paths_map = self.expected_first_step_inflows.astype(dtype=np.int32,order='F')
        test_input_func(self.flow_dirs,paths_map,6,6)
        np.testing.assert_array_equal(paths_map,self.expected_second_step_inflows,
                                      "Iterator produces wrong results for second step")

    @abstractmethod
    def testMainIteratorSecondIteration(self):
        """Test the second pass of the iterator across the river flow directions"""
        pass

    def MainIteratorFinalIterationTestHelper(self,test_input_func):
        """Helper function for the testing the final pass of the iterator"""
        paths_map = np.zeros((8,6),dtype=np.int32,order='F')
        while test_input_func(self.flow_dirs,paths_map,6,6):
            pass
        np.testing.assert_array_equal(paths_map,self.expected_final_step_inflows,
                                      "Iterator produces wrong results for final step")

    @abstractmethod
    def testIteratorFinalIteration(self):
        """Test the final pass of the iterator across the river flow directions"""
        pass

    def KernelFirstStepSourceCellTestHelper(self,test_input_func):
        """Helper function for testing computation of the flow to the first source cell"""
        paths_map_section = self.first_step_kernel_input_paths_map_section.astype(dtype=np.int32,order='F')
        paths_map_section[1,1] = test_input_func(self.first_step_kernel_flow_dirs_for_source_cell,paths_map_section)
        np.testing.assert_array_equal(self.expected_first_step_kernel_output_for_source_cell,paths_map_section,
                                      "Kernel produces wrong results when run on for first step river source grid cell")

    @abstractmethod
    def testKernelFirstStepSourceCell(self):
        """Test computation of the flow to the first source cell"""
        pass

    def KernelFirstStepNonSourceCellTestHelper(self,test_input_func):
        """Helper function for testing computation of the flow to a non source cell"""
        paths_map_section = self.first_step_kernel_input_paths_map_section.astype(dtype=np.int32,order='F')
        test_input_func(self.first_step_kernel_flow_dirs_for_non_source_cell,paths_map_section)
        np.testing.assert_array_equal(self.expected_first_step_kernel_output_for_non_source_cell,paths_map_section,
                                      "Kernel produces wrong results when run on for first step downstream grid cell")

    @abstractmethod
    def testKernelFirstStepNonSourceCell(self):
        """Test the computation of the flow to a non source cell"""
        pass

    def KernelLaterStepSourceCellTestHelper(self,test_input_func):
        """Helper function for testing computation of the flow to a later source cell"""
        paths_map_section = self.later_step_kernel_input_paths_map_section_for_source_cell.astype(dtype=np.int32,order='F')
        paths_map_section[1,1] = test_input_func(self.first_step_kernel_flow_dirs_for_source_cell,paths_map_section)
        np.testing.assert_array_equal(self.expected_later_step_kernel_output_for_source_cell,paths_map_section,
                                      "Kernel produces wrong results when run on for later step source grid cell")

    @abstractmethod
    def testKernelLaterStepSourceCell(self):
        """Test the computation of the flow to a later source cell"""
        pass

    def KernelLaterStepNonSourceCellTestHelper(self,test_input_func):
        """Helper function for testing computation of the flow to a later non source cell"""
        paths_map_section = self.later_step_kernel_input_paths_map_section_for_non_source_cell.astype(dtype=np.int32,order='F')
        paths_map_section[1,1] = test_input_func(self.first_step_kernel_flow_dirs_for_non_source_cell,paths_map_section)
        np.testing.assert_array_equal(self.expected_later_step_kernel_output_for_non_source_cell,paths_map_section,
                                      "Kernel produces wrong results when run on for later step downstream grid cell")

    @abstractmethod
    def testKernelLaterStepNonSourceCell(self):
        """Test the computation of the flow to a later non source cell"""
        pass

    def KernelLaterStepUnreadyNonSourceCellTestHelper(self,test_input_func):
        """Helper function for testing computation of the flow to a unready non source cell"""
        paths_map_section = self.later_step_kernel_input_paths_map_section_for_unready_non_source_cell.astype(dtype=np.int32,order='F')
        test_input_func(self.first_step_kernel_flow_dirs_for_non_source_cell,paths_map_section)
        np.testing.assert_array_equal(self.expected_later_step_kernel_output_for_unready_non_source_cell,paths_map_section,
                                      "Kernel produces wrong results when run on for later step downstream grid cell that is not ready to be calculated")

    @abstractmethod
    def testKernelLaterStepUnreadyNonSourceCell(self):
        """Test the computation of the flow to a unready non source cell"""
        pass

class MainTestCasePython(unittest.TestCase,MainTestCaseHelper):
    """Implement the test functions of MainTestCaseHelper for the Python version of the code"""

    def testMainIteratorFirstIteration(self):
        self.MainIteratorFirstIterationTestHelper(flow_to_grid_cell.iterate_paths_map)

    def testMainIteratorSecondIteration(self):
        self.MainIteratorSecondIterationTestHelper(flow_to_grid_cell.iterate_paths_map)

    def testIteratorFinalIteration(self):
        self.MainIteratorFinalIterationTestHelper(flow_to_grid_cell.iterate_paths_map)

    def testKernelFirstStepSourceCell(self):
        self.KernelFirstStepSourceCellTestHelper(flow_to_grid_cell.count_accumulated_inflow)

    def testKernelFirstStepNonSourceCell(self):
        self.KernelFirstStepNonSourceCellTestHelper(flow_to_grid_cell.count_accumulated_inflow)

    def testKernelLaterStepSourceCell(self):
        self.KernelLaterStepSourceCellTestHelper(flow_to_grid_cell.count_accumulated_inflow)

    def testKernelLaterStepNonSourceCell(self):
        self.KernelLaterStepNonSourceCellTestHelper(flow_to_grid_cell.count_accumulated_inflow)

    def testKernelLaterStepUnreadyNonSourceCell(self):
        self.KernelLaterStepUnreadyNonSourceCellTestHelper(flow_to_grid_cell.count_accumulated_inflow)

class MainTestCaseFortran(unittest.TestCase,MainTestCaseHelper):
    """Implement the test functions of MainTestCaseHelper for the Fortran version of the code"""

    def setUp(self):
        """Unit test setup function"""
        self.f2py_mngr = f2py_manager.f2py_manager(os.path.join(fortran_source_path,'mod_iterate_paths_map.f90'))

    def testMainIteratorFirstIteration(self):
        self.f2py_mngr.set_function_or_subroutine_name('iterate_paths_map')
        self.MainIteratorFirstIterationTestHelper(self.f2py_mngr.run_current_function_or_subroutine)

    def testMainIteratorSecondIteration(self):
        self.f2py_mngr.set_function_or_subroutine_name('iterate_paths_map')
        self.MainIteratorSecondIterationTestHelper(self.f2py_mngr.run_current_function_or_subroutine)

    def testIteratorFinalIteration(self):
        self.f2py_mngr.set_function_or_subroutine_name('iterate_paths_map')
        self.MainIteratorFinalIterationTestHelper(self.f2py_mngr.run_current_function_or_subroutine)

    def testKernelFirstStepSourceCell(self):
        self.f2py_mngr.set_function_or_subroutine_name('count_accumulated_inflow')
        self.KernelFirstStepSourceCellTestHelper(self.f2py_mngr.run_current_function_or_subroutine)

    def testKernelFirstStepNonSourceCell(self):
        self.f2py_mngr.set_function_or_subroutine_name('count_accumulated_inflow')
        self.KernelFirstStepNonSourceCellTestHelper(self.f2py_mngr.run_current_function_or_subroutine)

    def testKernelLaterStepSourceCell(self):
        self.f2py_mngr.set_function_or_subroutine_name('count_accumulated_inflow')
        self.KernelLaterStepSourceCellTestHelper(self.f2py_mngr.run_current_function_or_subroutine)

    def testKernelLaterStepNonSourceCell(self):
        self.f2py_mngr.set_function_or_subroutine_name('count_accumulated_inflow')
        self.KernelLaterStepNonSourceCellTestHelper(self.f2py_mngr.run_current_function_or_subroutine)

    def testKernelLaterStepUnreadyNonSourceCell(self):
        self.f2py_mngr.set_function_or_subroutine_name('count_accumulated_inflow')
        self.KernelLaterStepUnreadyNonSourceCellTestHelper(self.f2py_mngr.run_current_function_or_subroutine)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
