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
from tests.context import fortran_source_path

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

class MainTestCaseNewAlgorithm(unittest.TestCase):

    def testCalculateCumlativeFlowLatLon(self):
        flow_dirs =  np.array([[3,1,4,4,4,4],
                               [2,4,4,4,7,7],
                               [3,9,8,7,6,2],
                               [5,3,6,9,2,1],
                               [6,7,2,3,1,8],
                               [5,0,7,0,4,4]],
                              dtype=np.int32)
        expected_cumulative_flow = np.array([[ 1, 7, 6, 5, 3, 1],
                                             [15, 7, 5, 1, 1, 1],
                                             [16, 1, 1, 1, 3, 4],
                                             [22,17, 1, 2, 1, 6],
                                             [1,21,18, 1, 8, 1],
                                             [1, 1,19,12, 3, 1]],
                                            dtype=np.int32)
        output_cumulative_flow = flow_to_grid_cell.\
            create_hypothetical_river_paths_map(flow_dirs,
                                                lsmask=None,
                                                use_f2py_func=True,
                                                nlat=6,nlong=6)
        np.testing.assert_array_equal(output_cumulative_flow,
                                      expected_cumulative_flow)

    def testCalculateCumlativeFlowLatLonWrapped(self):
        flow_dirs_with_wrap = np.array([[1,1,4,4,4,3],
                                        [2,4,4,4,7,7],
                                        [3,9,8,7,6,2],
                                        [4,3,6,9,5,4],
                                        [6,7,2,3,1,8],
                                        [7,0,7,0,4,9]],
                                       dtype=np.int32)
        expected_cumulative_flow_with_wrap = \
            np.array([[ 1, 7, 6, 5, 3, 1],
                      [15, 6, 5, 1, 1, 2],
                      [16, 1, 1, 1, 3, 4],
                      [23, 17, 1, 2, 31, 30],
                      [2, 22, 18, 1, 1, 2],
                      [1, 1, 19, 4, 2, 1 ]],
                     dtype=np.int32)
        output_cumulative_flow = flow_to_grid_cell.\
            create_hypothetical_river_paths_map(flow_dirs_with_wrap,
                                                lsmask=None,
                                                use_f2py_func=True,
                                                nlat=6,nlong=6)
        np.testing.assert_array_equal(output_cumulative_flow,
                                      expected_cumulative_flow_with_wrap)

    def testCalculateCumlativeFlowLatLonWithMask(self):
        flow_dirs =  np.array([[3,1,4,4,4,4],
                               [2,4,4,4,7,7],
                               [3,9,8,7,6,2],
                               [5,3,6,9,2,1],
                               [6,7,2,3,1,8],
                               [5,2,7,5,4,4]],
                              dtype=np.int32)
        expected_cumulative_flow_when_using_mask = \
            np.array([[1, 1, 1, 2, 1, 1],
                      [1, 4, 3, 1, 1, 1],
                      [1, 1, 1, 1, 3, 4],
                      [2, 2, 1, 2, 1, 5],
                      [1, 1, 3, 1, 7, 1],
                      [1, 1, 4,11, 3, 1]],
                     dtype=np.int32)
        ls_mask = np.array([[True,True,True,True,True,True],
                            [True,True,False,False,False,True],
                            [False,True,True,False,False,False],
                            [False,False,False,False,False,False],
                            [True,False,False,False,False,True],
                            [True,True,True,False,False,False]],
                            dtype=bool)
        flow_dirs[ls_mask] = 0
        output_cumulative_flow = flow_to_grid_cell.\
            create_hypothetical_river_paths_map(flow_dirs,
                                                lsmask=ls_mask,
                                                use_f2py_func=True,
                                                nlat=6,nlong=6)
        np.testing.assert_array_equal(output_cumulative_flow,
                                      expected_cumulative_flow_when_using_mask)


    @unittest.skip
    def testCalculateCumlativeFlowLatLonWithBasicLoop(self):
        flow_dirs_with_loop =  np.array([[6,4,2],
                                         [6,6,5],
                                         [6,6,5]],
                                          dtype=np.int32)

        expected_cumulative_flow_with_loop = \
                np.array([[0,0,1],
                          [1,2,4],
                          [1,2,3]],
                          dtype=np.int32)
        # output_cumulative_flow = flow_to_grid_cell.\
        #     create_hypothetical_river_paths_map(flow_dirs_with_loop,
        #                                         lsmask=None,
        #                                         use_f2py_func=True,
        #                                         nlat=3,nlong=3)
        #np.testing.assert_array_equal(output_cumulative_flow,
        #                              expected_cumulative_flow_with_loop)
        self.assertTrue(False)

    @unittest.skip
    def testCalculateCumlativeFlowLatLonWithLoop(self):
        flow_dirs_with_loop =  np.array([[1,1,4,4,4,6,3,5],
                                         [2,4,4,4,7,8,4,4],
                                         [3,9,8,7,6,8,5,8],
                                         [4,3,6,9,5,8,5,5],
                                         [6,7,2,3,1,8,7,5],
                                         [7,2,7,1,4,9,8,4],
                                         [5,5,5,5,5,5,5,5],
                                         [5,5,5,5,5,5,5,5]],
                                        dtype=np.int32)
        expected_cumulative_flow_with_loop = \
            np.array([[ 1, 5,   4, 3, 1, 0, 0,  0],
                      [ 12, 6,   5, 1, 1, 0, 0,  0],
                      [ 13, 1,   1, 1, 3,10, 0,  1],
                      [ 19, 14,  1, 2, 0, 6, 0, 20],
                      [ 1,  18, 15, 1, 1, 1, 4,  2],
                      [ 1,   1, 16, 4, 2, 1, 2,  1],
                      [ 0,   2,  5, 0, 0, 0, 0,  0],
                      [ 0,   0,  0, 0, 0, 0, 0,  0]],
                     dtype=np.int64)
        # output_cumulative_flow = flow_to_grid_cell.\
        #     create_hypothetical_river_paths_map(flow_dirs_with_loop,
        #                                         lsmask=None,
        #                                         use_f2py_func=True,
        #                                         nlat=8,nlong=8)
        #np.testing.assert_array_equal(output_cumulative_flow,
        #                              expected_cumulative_flow_with_loop)
        self.assertTrue(False)

    def testCalculateCumulativeFlow(self):
        input_river_directions = np.array([    2,10,  5,  5, -3,
                        23, 1, 9, 10, 13, 27, 11, 30, 33, 33, 4, 33, 5, 5, 6,
             6, 21, 24, -3, 26, 45, 10, 47, 28, 47, 33, 51, 32, -3, -3, -3, -3, -3, 40, 20,
             60, 24, 24, 62, 44, -3, 26, 46, 30, 49, 52, -3, -3, -3, -3, -3, -3, 73, 60, 40,
                       41, 61, 64, 65, 66, 48, 50, 78, 52, -3, -3, -3, 80, 75, 59,
                                      62, 78, 79, -3, 74 ], dtype=np.int32)
        expected_output_cumulative_flow = np.array([ 2, 3, 1, 2, 6,
                        38, 1, 1, 2, 9, 2, 1, 10, 1, 1, 1, 1, 1, 1, 35,
             2, 1, 39, 42, 1, 19, 3, 2, 1, 14, 1, 6, 5, 0, 0, 0, 0, 0, 1, 34,
            25, 1,  1, 21, 20, 6, 17, 5, 3, 2, 7, 9, 0, 0, 0, 0, 0, 1, 6, 32,
                        24, 23, 1, 2, 3, 4, 1, 1, 1, 0, 0, 0, 2, 4, 5,
                                             1, 1, 3, 4, 3 ], dtype=np.int32)
        cell_neighbors =  np.array([
            #1
            [5,7,2],
            #2
            [1,10,3],
            #3
            [2,13,4],
            #4
            [3,16,5],
            #5
            [4,19,1],
            #6
            [20,21,7],
            #7
            [1,6,8],
            #8
            [7,23,9],
            #9
            [8,25,10],
            #10
            [2,9,11],
            #11
            [10,27,12],
            #12
            [11,29,13],
            #13
            [3,12,14],
            #14
            [13,31,15],
            #15
            [14,33,16],
            #16
            [4,15,17],
            #17
            [16,35,18],
            #18
            [17,37,19],
            #19
            [5,18,20],
            #20
            [19,39,6],
            #21
            [6,40,22],
            #22
            [21,41,23],
            #23
            [8,22,24],
            #24
            [23,43,25],
            #25
            [24,26,9],
            #26
            [25,45,27],
            #27
            [11,26,28],
            #28
            [27,47,29],
            #29
            [12,28,30],
            #30
            [29,49,31],
            #31
            [14,30,32],
            #32
            [31,51,33],
            #33
            [15,32,34],
            #34
            [33,53,35],
            #35
            [17,34,36],
            #36
            [35,55,37],
            #37
            [18,36,38],
            #38
            [37,57,39],
            #39
            [20,38,40],
            #40
            [39,59,21],
            #41
            [22,60,42],
            #42
            [41,61,43],
            #43
            [24,42,44],
            #44
            [43,63,45],
            #45
            [26,44,46],
            #46
            [45,64,47],
            #47
            [28,46,48],
            #48
            [47,66,49],
            #49
            [30,48,50],
            #50
            [49,67,51],
            #51
            [32,50,52],
            #52
            [51,69,53],
            #53
            [34,52,54],
            #54
            [53,70,55],
            #55
            [36,54,56],
            #56
            [55,72,57],
            #57
            [38,56,58],
            #58
            [57,73,59],
            #59
            [40,58,60],
            #60
            [59,75,41],
            #61
            [42,75,62],
            #62
            [61,76,63],
            #63
            [44,62,64],
            #64
            [46,63,65],
            #65
            [64,77,66],
            #66
            [48,65,67],
            #67
            [50,66,68],
            #68
            [67,78,69],
            #69
            [52,68,70],
            #70
            [54,69,71],
            #71
            [70,79,72],
            #72
            [56,71,73],
            #73
            [58,72,74],
            #74
            [73,80,75],
            #75
            [60,74,61],
            #76
            [62,80,77],
            #77
            [65,76,78],
            #78
            [68,77,79],
            #79
            [71,78,80],
            #80
            [74,79,76 ]], dtype=np.int32)
        output_cumulative_flow = flow_to_grid_cell.\
            accumulate_flow_icon_single_index(cell_neighbors,
                                              input_river_directions)
        np.testing.assert_array_equal(output_cumulative_flow,
                                     expected_output_cumulative_flow)

    def testCalculateCumulativeFlowWithZeroBifurcations(self):
        input_river_directions = np.array([    6, -3, 11,  -3, 18,
                        21, 6, 6, 2, 2, 10, 13, 3, 13, 16, 4, 35, 37, 18, 39,
            21,41,22,25,26,45,11,29,30,49,30,50,15,33,36,-3,36,37,38, 38,
            42,43,24,63,46,47,28,49,50,51,-3,53,34,55,36,55,56,57,-3,-3,
                      62,63,-3,63,77,65,50,67,68,54,54,55,-3, -3, -3,
                                    62, -3, 68, 71, -3 ], dtype=np.int32)
        input_bifurcated_river_directions = np.zeros((80,11),dtype=np.int32)
        input_bifurcated_river_directions[:,:] = -9
        expected_output_cumulative_flow = np.array([ 1, 9, 4, 7, 1,
                        4, 1, 1, 1, 7, 6, 1, 3, 1, 5, 6, 1, 3, 1, 1,
             0, 2, 1, 6, 7, 8, 1, 12, 13, 15, 1, 1, 4, 3, 2, 20, 8, 4, 2, 1,
             3, 4,  5, 1, 9, 10, 11, 1, 17, 23, 24, 1, 2, 4, 9, 3, 2, 1, 0, 0,
                        1, 3, 6, 1, 2, 1, 4, 3, 1, 1, 2, 1, 0, 0, 0,
                                             1, 3, 1, 1, 0 ], dtype=np.int32)
        cell_neighbors = np.array([
            #1
            [5,7,2],
            #2
            [1,10,3],
            #3
            [2,13,4],
            #4
            [3,16,5],
            #5
            [4,19,1],
            #6
            [20,21,7],
            #7
            [1,6,8],
            #8
            [7,23,9],
            #9
            [8,25,10],
            #10
            [2,9,11],
            #11
            [10,27,12],
            #12
            [11,29,13],
            #13
            [3,12,14],
            #14
            [13,31,15],
            #15
            [14,33,16],
            #16
            [4,15,17],
            #17
            [16,35,18],
            #18
            [17,37,19],
            #19
            [5,18,20],
            #20
            [19,39,6],
            #21
            [6,40,22],
            #22
            [21,41,23],
            #23
            [8,22,24],
            #24
            [23,43,25],
            #25
            [24,26,9],
            #26
            [25,45,27],
            #27
            [11,26,28],
            #28
            [27,47,29],
            #29
            [12,28,30],
            #30
            [29,49,31],
            #31
            [14,30,32],
            #32
            [31,51,33],
            #33
            [15,32,34],
            #34
            [33,53,35],
            #35
            [17,34,36],
            #36
            [35,55,37],
            #37
            [18,36,38],
            #38
            [37,57,39],
            #39
            [20,38,40],
            #40
            [39,59,21],
            #41
            [22,60,42],
            #42
            [41,61,43],
            #43
            [24,42,44],
            #44
            [43,63,45],
            #45
            [26,44,46],
            #46
            [45,64,47],
            #47
            [28,46,48],
            #48
            [47,66,49],
            #49
            [30,48,50],
            #50
            [49,67,51],
            #51
            [32,50,52],
            #52
            [51,69,53],
            #53
            [34,52,54],
            #54
            [53,70,55],
            #55
            [36,54,56],
            #56
            [55,72,57],
            #57
            [38,56,58],
            #58
            [57,73,59],
            #59
            [40,58,60],
            #60
            [59,75,41],
            #61
            [42,75,62],
            #62
            [61,76,63],
            #63
            [44,62,64],
            #64
            [46,63,65],
            #65
            [64,77,66],
            #66
            [48,65,67],
            #67
            [50,66,68],
            #68
            [67,78,69],
            #69
            [52,68,70],
            #70
            [54,69,71],
            #71
            [70,79,72],
            #72
            [56,71,73],
            #73
            [58,72,74],
            #74
            [73,80,75],
            #75
            [60,74,61],
            #76
            [62,80,77],
            #77
            [65,76,78],
            #78
            [68,77,79],
            #79
            [71,78,80],
            #80
            [74,79,76 ]], dtype=np.int32)
        output_cumulative_flow = flow_to_grid_cell.\
            accumulate_flow_icon_single_index(cell_neighbors,
                                              input_river_directions,
                                              input_bifurcated_river_directions)
        np.testing.assert_array_equal(output_cumulative_flow,
                                      expected_output_cumulative_flow)

    def testCalculateCumulativeFlowWithBifurcations(self):
        input_river_directions = np.array([    6, -3, 11,  -3, 18,
                        21, 6, 6, 2, 2, 10, 13, 3, 13, 16, 4, 35, 37, 18, 39,
            21,41,22,25,26,45,11,29,30,49,30,50,15,33,36,-3,36,37,38, 38,
            42,43,24,63,46,47,28,49,50,51,-3,53,34,55,36,55,56,57,-3,-3,
                      62,63,-3,63,77,65,50,67,68,54,54,55,-3, -3, -3,
                                    62, -3, 68, 71, -3 ], dtype=np.int32)
        input_bifurcated_river_directions = np.zeros((80,11),np.int32)
        input_bifurcated_river_directions[:,:] = -9
        input_bifurcated_river_directions[10,0] = 9
        input_bifurcated_river_directions[33,0] = 35
        input_bifurcated_river_directions[42,0] = 44
        input_bifurcated_river_directions[45,0] = 65
        input_bifurcated_river_directions[45,1] = 27
        input_bifurcated_river_directions[45,2] = 64
        expected_output_cumulative_flow = np.array([1, 35, 4, 7, 1,
                        4, 1, 1, 17, 17, 16, 1, 3, 1, 5, 6, 1, 3, 1, 1,
             0, 2, 1, 6, 7, 8, 11, 12, 13, 15, 1, 1, 4, 3, 5, 23, 8, 4, 2, 1,
             3, 4,  5, 6, 9, 10, 11, 1, 17, 23, 24, 1, 2, 4, 9, 3, 2, 1, 0, 0,
                        1, 3, 21, 11, 12, 1, 4, 3, 1, 1, 2, 1, 0, 0, 0,
                                             1, 13, 1, 1, 0 ], dtype=np.int32)
        cell_neighbors = np.array([
            #1
            [5,7,2],
            #2
            [1,10,3],
            #3
            [2,13,4],
            #4
            [3,16,5],
            #5
            [4,19,1],
            #6
            [20,21,7],
            #7
            [1,6,8],
            #8
            [7,23,9],
            #9
            [8,25,10],
            #10
            [2,9,11],
            #11
            [10,27,12],
            #12
            [11,29,13],
            #13
            [3,12,14],
            #14
            [13,31,15],
            #15
            [14,33,16],
            #16
            [4,15,17],
            #17
            [16,35,18],
            #18
            [17,37,19],
            #19
            [5,18,20],
            #20
            [19,39,6],
            #21
            [6,40,22],
            #22
            [21,41,23],
            #23
            [8,22,24],
            #24
            [23,43,25],
            #25
            [24,26,9],
            #26
            [25,45,27],
            #27
            [11,26,28],
            #28
            [27,47,29],
            #29
            [12,28,30],
            #30
            [29,49,31],
            #31
            [14,30,32],
            #32
            [31,51,33],
            #33
            [15,32,34],
            #34
            [33,53,35],
            #35
            [17,34,36],
            #36
            [35,55,37],
            #37
            [18,36,38],
            #38
            [37,57,39],
            #39
            [20,38,40],
            #40
            [39,59,21],
            #41
            [22,60,42],
            #42
            [41,61,43],
            #43
            [24,42,44],
            #44
            [43,63,45],
            #45
            [26,44,46],
            #46
            [45,64,47],
            #47
            [28,46,48],
            #48
            [47,66,49],
            #49
            [30,48,50],
            #50
            [49,67,51],
            #51
            [32,50,52],
            #52
            [51,69,53],
            #53
            [34,52,54],
            #54
            [53,70,55],
            #55
            [36,54,56],
            #56
            [55,72,57],
            #57
            [38,56,58],
            #58
            [57,73,59],
            #59
            [40,58,60],
            #60
            [59,75,41],
            #61
            [42,75,62],
            #62
            [61,76,63],
            #63
            [44,62,64],
            #64
            [46,63,65],
            #65
            [64,77,66],
            #66
            [48,65,67],
            #67
            [50,66,68],
            #68
            [67,78,69],
            #69
            [52,68,70],
            #70
            [54,69,71],
            #71
            [70,79,72],
            #72
            [56,71,73],
            #73
            [58,72,74],
            #74
            [73,80,75],
            #75
            [60,74,61],
            #76
            [62,80,77],
            #77
            [65,76,78],
            #78
            [68,77,79],
            #79
            [71,78,80],
            #80
            [74,79,76 ]], dtype=np.int32)
        output_cumulative_flow = flow_to_grid_cell.\
            accumulate_flow_icon_single_index(cell_neighbors,
                                              input_river_directions,
                                              input_bifurcated_river_directions)
        np.testing.assert_array_equal(output_cumulative_flow,
                                     expected_output_cumulative_flow)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
