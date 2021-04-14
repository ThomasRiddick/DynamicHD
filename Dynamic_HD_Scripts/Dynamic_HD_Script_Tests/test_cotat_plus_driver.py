'''
Unit test of the COTAT plus driver using same data as top-level
Fortran FRUIT unit tests

Created on Oct 19, 2016

@author: thomasriddick
'''

import unittest
import numpy as np
import Dynamic_HD_Scripts.field as field
import Dynamic_HD_Scripts.cotat_plus_driver as cotat_plus_driver
import os
import re
from context import data_dir
import textwrap
import subprocess
from . import context
import Dynamic_HD_Scripts.context as scripts_context
from matplotlib.compat.subprocess import CalledProcessError

class Test(unittest.TestCase):
    """Unit test object"""
    
    show_output = False

    input_fine_river_directions_test_data = np.array([[-1,-1,-1, -1,0,4,   2,2,2, 2,2,2, 3,2,2],
                                                      [-1,-1,-1, -1,-1,-1, 0,4,4, 4,4,1, 4,4,1],
                                                      [-1,-1,-1, -1,-1,9,  8,8,8, 1,7,7, 4,4,4],
                                                      [-1,-1,-1, -1,-1,0, 4,4,4, 6,8,5, 8,4,8],
                                                      [-1,0,4, 4,4,5, 7,1,8, 4,6,7, 6,5,4],
                                                      [-1,0,4, 4,5,7, 4,4,1, 9,6,8, 1,6,2],
                                                      [-1,-1,0, 4,6,8, 4,4,6, 6,6,7, 4,2,5],
                                                      [-1,-1,-1, 7,6,8, 4,1,3, 9,9,8, 8,7,4],
                                                      [-1,0,0, 4,6,8, 4,4,5, 1,5,5, 9,1,7],
                                                      [0,8,7, 7,7,7, 7,4,4, 6,7,4, 4,1,2],
                                                      [8,8,8, 8,1,2, 6,7,5, 9,8,6, 8,7,4],
                                                      [9,2,7, 4,4,2, 9,8,7, 9,8,4, 9,8,8],
                                                      [6,6,8, 4,8,1, 2,1,8, 3,8,2, 3,5,8],
                                                      [4,6,8, 7,8,7, 4,4,3, 3,2,6, 6,5,8],
                                                      [9,8,8, 7,8,5, 8,9,8, 6,6,8, 9,8,7]],
                                                     dtype=np.int64)

    input_fine_total_cumulative_flow_test_data = np.array([[1,1,1,  1,1,1,  1,1,1,  1,1,1,  1,1,1],
                                                           [1,1,1,  1,1,1, 52,48,45, 42,11,6, 4,3,2],
                                                           [1,1,1, 1,1,1, 1,1,1, 1,29,9, 8,5,2],
                                                           [1,1,1, 1,1,8, 6,5,4, 1,22,1, 2,1,1],
                                                           [1,55,54, 53,52,1, 1,1,2, 1,2,20, 1,3,1],
                                                           [1,3,2, 1,1,51, 3,1,1, 1,16,17, 1,1,2],
                                                           [1,1,3, 1,1,47, 3,2,1, 2,4,15, 7,1,3],
                                                           [1,1,1, 1,1,42, 1,1,1, 1,1,1, 1,5,1],
                                                           [1,35,5, 2,2,39, 3,1,1, 24,1,1, 1,1,1],
                                                           [2,32,2, 2,1,1, 33,26,25, 1,22,14, 13,1,1],
                                                           [1,31,1, 1,1,1, 1,6,1, 1,5,1, 3,8,5],
                                                           [1,1,29, 15,13,2, 1,1,2, 1,3,1, 1,1,3],
                                                           [1,3,13, 1,12,3, 1,1,1, 1,1,1, 1,1,2],
                                                           [1,4,7, 1,5,6, 5,1,3, 1,2,11, 12,17,1],
                                                           [1,1,1, 1,1,1, 1,1,1, 4,8,9, 1,1,1]],
                                                          dtype=np.int64)
    
    small_grid_expected_result = np.array([[-1,6,0,4,4],
                                           [0,4,4,8,5],
                                           [0,7,4,8,7],
                                           [8,4,7,4,7],
                                           [8,7,7,6,5]],dtype=np.int64)
    
    directory = None
    
    def setUp(self):
        """Unit test setup. Creates a temporary directory for results if necessary"""
        #create files
        if False:
            self.directory = os.path.expanduser('~')+ '/temp'
        else:
            self.directory = data_dir + '/temp'
        try:
            os.stat(self.directory)
        except:
            os.mkdir(self.directory)
        self.cotat_params_file_path = os.path.join(self.directory,'cotat_plus_params_temp.nl')

    def testUsingSmallGrid(self):
        """
        Test using a small 5 by 5 grid
        
        Same data was used in FRUIT unit testing
        """

        input_fine_river_directions_test_field = field.makeField(self.input_fine_river_directions_test_data, 
                                                                 field_type='RiverDirections',
                                                                 grid_type='LatLong',nlat=15,nlong=15)
        input_fine_total_cumulative_flow_test_field = field.makeField(self.input_fine_total_cumulative_flow_test_data, 
                                                                      field_type='CumulativeFlow',
                                                                      grid_type='LatLong',nlat=15,nlong=15)
        cotat_params_text =\
            """
            &cotat_parameters
            MUFP = 1.5
            area_threshold = 9
            run_check_for_sinks = .True.
            /
            """
        with open(self.cotat_params_file_path,'w') as f:
            f.write(textwrap.dedent(cotat_params_text))
        output_course_river_directions = \
            cotat_plus_driver.run_cotat_plus(fine_rdirs_field=input_fine_river_directions_test_field, 
                                             fine_total_cumulative_flow_field=\
                                             input_fine_total_cumulative_flow_test_field, 
                                             cotat_plus_parameters_filepath=self.cotat_params_file_path, 
                                             course_grid_type='LatLong',nlat=5,nlong=5)
        np.testing.assert_array_equal(output_course_river_directions.get_data(),
                                      self.small_grid_expected_result,
                                      "Running scaling code over small 5 by 5 grid doesn't"
                                      " produce expected results")
        
    def testForMemoryLeaksWithValgrind(self):
        """Run valgrind to check no new memory leaks are occurring"""
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
            print(valgrind_output)
        direct_mem_loss = int(direct_mem_loss_match.group(1).replace(',',''))
        indirect_mem_loss = int(indirect_mem_loss_match.group(1).replace(',',''))
        # 80 byte loss is a known problem that occurs sometimes related to using valgrind in python
        self.assertTrue((direct_mem_loss == 0 or direct_mem_loss == 80),"Direct memory leak detected")
        # 68 byte loss is a known problem that occurs sometimes related to using valgrind in python
        self.assertTrue((indirect_mem_loss == 0 or indirect_mem_loss == 68),"Indirect memory leak detected")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
