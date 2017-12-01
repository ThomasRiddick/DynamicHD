'''
Unit test of the driver for the Fortran loop breaking code
Created on Oct 30, 2016

@author: thomasriddick
'''

import unittest
import numpy as np
import Dynamic_HD_Scripts.field as field
import Dynamic_HD_Scripts.loop_breaker_driver as loop_breaker_driver
import Dynamic_HD_Scripts.context as scripts_context
import context
import subprocess
import re
from matplotlib.compat.subprocess import CalledProcessError

class Test(unittest.TestCase):
    """Unit test object"""
    
    show_output = False
    
    course_rdirs = np.array([[6,1,6,6,1],
                             [8,4,4,8,7],
                             [1,-1,5,0,6],
                             [2,4,9,8,8],
                             [9,6,8,7,6]],dtype=np.float64)
    
    course_cumulative_flow = np.array([[0,0,1,0,0],
                                       [0,2,1,0,1],
                                       [0,0,1,0,0],
                                       [0,0,0,1,0],
                                       [0,0,0,1,1]],dtype=np.float64)
    course_catchments = np.array([[25,25,61,61,61],
                                  [25,25,25,61,61],
                                  [13,0,14,58,13],
                                  [9,9,58,19,13],
                                  [9,58,58,58,9]],dtype=np.float64)
    fine_rdirs = np.array([[6,6,6, 6,6,2, 6,6,3, 6,6,6, 6,6,2],
                           [6,6,6, 6,6,6, 9,8,4, 4,6,6, 6,6,2],
                           [6,6,6, 6,6,8, 8,8,8,  6,6,6, 6,6,2],
                           [6,6,6, 6,6,8, 8,8,8, 6,6,6, 6,6,2],
                           [6,6,6, 6,6,8, 8,8,8, 6,6,6, 6,6,2],
                           [6,6,6, 6,6,3, 8,8,8, 6,6,6, 6,6,6],
                           [4,4,4, 0,1,2, -1,-1,-1, 5,5,5, 7,6,8],
                           [4,4,4, 0,1,2, 5,6,2,    1,1,1, 8,6,6],
                           [4,4,4, 0,1,2, 1,5,2,    2,3,8, 8,6,6],
                           [4,6,6, 6,9,6, 3,2,1, 1,2,3,  8,8,8],
                           [6,6,6, 6,8,4, 4,5,6, 4,5,6,  8,8,8],
                           [6,6,6, 6,8,4, 9,8,7, 7,8,9,  8,8,8],
                           [8,8,8, 6,6,6, 8,8,8, 3,3,3,  2,2,2],
                           [8,8,8, 6,6,6, 8,8,8, 3,3,3,  2,2,2],
                           [8,8,8, 6,6,6, 8,8,6, 3,3,3,  2,2,2 ]],dtype=np.float64)
    fine_cumulative_flow = np.array([[1,1,205, 1,1,1,   1,1,1, 1,1,1, 1,1,1],
                                     [1,1,1, 1,1,206, 207,1,1, 61,1,1, 1,1,1],
                                     [1,1,1,  1,1,1,    1,1,1, 1,1,1, 1,1,1],
                                     [1,1,1, 1,1,1,  1,1,1, 1,1,1, 1,1,1],
                                     [1,1,1, 1,1,1,  1,1,1, 1,1,1, 1,1,1],
                                     [1,1,1, 1,1,221,  1,1,1, 1,1,1, 1,1,1],
                                     [1,1,1, 1,1,1, 1,1,1, 1,1,1, 207,1,56],
                                     [1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1],
                                     [1,1,1, 1,1,1, 1,1,1, 1,1,1,   1,1,1],
                                     [32,1,1, 35,36,37, 1,1,1, 1,1,1, 1,1,1],
                                     [1,1,1, 1,1,1, 1,90,1, 1,1,1, 1,1,1],
                                     [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
                                     [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
                                     [1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
                                     [1,1,1, 1,1,1, 1,1,11, 1,1,1, 1,1,1]],dtype=np.float64)
    expected_results = np.array([[6,6,6,4,1],
                                 [8,4,4,8,7],
                                 [1,-1,5,0,7],
                                 [2,6,5,8,8],
                                 [9,6,8,7,6 ]],dtype=np.float64)
    loop_nums_list = [25,61,9,58,13]

    def testUsingSmallGrid(self):
        """
        Test using a small 5 by 5 grid as the course grid, 15 by 15 fine grid
        
        Same data was used in FRUIT unit testing
        """

        course_rdirs_field = field.makeField(self.course_rdirs, 
                                             field_type='RiverDirections',
                                             grid_type='LatLong',nlat=5,nlong=5)
        course_cumulative_flow_field = field.makeField(self.course_cumulative_flow,
                                                       field_type='CumulativeFlow',
                                                       grid_type='LatLong',nlat=5,nlong=5)
        course_catchments_field = field.makeField(self.course_catchments,
                                                  field_type='Generic',
                                                  grid_type='LatLong',nlat=15,nlong=15) 
        fine_rdirs_field = field.makeField(self.fine_rdirs,
                                           field_type='RiverDirections',
                                           grid_type='LatLong',nlat=15,nlong=15)
        fine_cumulative_flow_field = field.makeField(self.fine_cumulative_flow,
                                                     field_type='CumulativeFlow',
                                                     grid_type='LatLong',nlat=15,nlong=15)
        updated_output_course_river_directions =\
            loop_breaker_driver.run_loop_breaker(course_rdirs_field, course_cumulative_flow_field, 
                                                 course_catchments_field, fine_rdirs_field, 
                                                 fine_cumulative_flow_field, self.loop_nums_list, 
                                                 course_grid_type='LatLong',nlat=5,nlong=5)
        np.testing.assert_array_equal(updated_output_course_river_directions.get_data(), self.expected_results, 
                                      "Testing the loop breaking code with a small test grid doesn't give"
                                      " expected results")
        
    def testForMemoryLeaksWithValgrind(self):
        """Use Valgrind to find any new memory leaks that are occuring"""
        try:
            valgrind_output =  subprocess.check_output([context.valgrind_path,'--leak-check=full',
                                                        scripts_context.fortran_project_executable_path],
                                                       stderr=subprocess.STDOUT,
                                                       cwd=scripts_context.fortran_project_include_path)
        except CalledProcessError as cperror:
            raise RuntimeError("Failure in called process {0}; return code {1}; output:\n{2}".format(cperror.cmd,
                                                                                                     cperror.returncode,
                                                                                                     cperror.output))
        direct_mem_loss_match = re.search(r'definitely lost: ([,0-9]*)',valgrind_output)
        indirect_mem_loss_match = re.search(r'indirectly lost: ([,0-9]*)',valgrind_output)
        if self.show_output:
            print valgrind_output
        direct_mem_loss = int(direct_mem_loss_match.group(1).replace(',',''))
        indirect_mem_loss = int(indirect_mem_loss_match.group(1).replace(',',''))
        # 80 byte loss is a known problem that occurs sometimes related to using valgrind in python
        self.assertTrue((direct_mem_loss == 0 or direct_mem_loss == 80),"Direct memory leak detected")
        # 68 byte loss is a known problem that occurs sometimes related to using valgrind in python
        self.assertTrue((indirect_mem_loss == 0 or indirect_mem_loss == 68),"Indirect memory leak detected")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()