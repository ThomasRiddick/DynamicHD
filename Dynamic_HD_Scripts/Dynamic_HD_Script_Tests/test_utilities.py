'''
Test various dynamic hydrological discharge utility functions
Created on May 2, 2016

@author: thomasriddick
'''

import unittest
import numpy as np
import os
import Dynamic_HD_Scripts.utilities as utilities
import Dynamic_HD_Scripts.field as field
from context import data_dir
from Dynamic_HD_Scripts import dynamic_hd

class TestReservoirSizeInitialisation(unittest.TestCase):
    """Test the initialisation of the size of HD reservoirs"""
   
    replacing_zeros_test_data = np.array([[0.0,0.0, 0.0, 0.0,0.1],
                                          [1.6,2.7, 0.0, 0.0,0.0],
                                          [0.0,0.5,-1.0,-1.0,0.0],
                                          [0.0,1.6, 0.0, 0.0,1.8],
                                          [0.0,0.1, 3.9, 0.0,0.0]])
    
    replacing_zeros_test_data_two = np.array([[0.0,-1.0, -1.0, -1.0,0.1],
                                              [1.6,-1.0, 0.0, -1.0,0.0],
                                              [0.0,-1.0,-1.0,-1.0,0.0],
                                              [0.0,0.0, 0.0, 0.0,1.8],
                                              [0.0,0.0, 0.0, 0.0,0.0]])
    
    replacing_zeros_expected_results = np.array([[1.5375,1.5375,1.5375, 1.5375,0.1],
                                                 [1.6,2.7, 1.5375, 1.5375,1.5375],
                                                 [1.5375,0.5,-1.0,-1.0,1.5375],
                                                 [1.5375,1.6, 1.5375, 1.5375,1.8],
                                                 [1.5375,0.1, 3.9, 1.5375,1.5375]])
    
    replacing_zero_with_highest_neighbor_expected_results = np.array([[2.7,2.7, 2.7, 0.1,0.1],
                                                                      [1.6,2.7, 2.7, 0.1,1.6],
                                                                      [2.7,0.5,-1.0,-1.0,1.8],
                                                                      [1.8,1.6, 3.9, 3.9,1.8],
                                                                      [1.8,0.1, 3.9, 3.9,1.8]])
    
    replacing_zero_with_highest_neighbor_expected_results_two = np.array([[1.6,-1.0, -1.0, -1.0,0.1],
                                                                          [1.6,-1.0, 0.0, -1.0,1.6],
                                                                          [1.8,-1.0,-1.0,-1.0,1.8],
                                                                          [1.8,0.0, 0.0, 1.8,1.8],
                                                                          [1.8,0.0, 0.0, 1.8,1.8]])
    
    hd_restart_file_prep_input_field =  np.array([[1.9,1.8,0.1, 0.1,1.0,0.1, -0.1,-0.2,-0.2],
                                                  [0.1,0.0,0.0, 0.0,0.6,0.2, -0.1,-0.05,-0.2],
                                                  [0.1,0.0,0.0, 0.0,0.5,0.3, -0.1,-0.2,-0.2],
                                                  [0.2,0.0,0.0, 0.0,0.3,0.3, 2.3,2.0,0.0],
                                                  [0.1,0.2,0.1, 8.1,0.9,0.3, 3.4,2.0,2.0],
                                                  [0.2,0.3,0.3, 0.0,1.1,0.0, 8.0,7.0,3.0],
                                                  [0.6,0.0,0.7, 1.0,1.1,8.0, 0.0,0.0,0.0],
                                                  [0.2,0.3,0.4, 0.7,9.0,0.0, 0.0,0.0,0.0],
                                                  [0.4,0.0,0.0, 0.6,5.0,0.0, 0.0,0.0,10.0]])
    
    hd_restart_file_prep_input_res_nums =  np.array([[5.5,5.5,5.5, 5.5,5.5,5.5, 5.5,5.5,5.5],
                                                     [5.5,1.1,1.1, 1.1,5.5,5.5, 5.5,5.5,5.5],
                                                     [5.5,1.1,1.1, 1.1,5.5,5.5, 5.5,5.5,1.1],
                                                     [5.5,1.1,1.1, 1.1,5.5,5.5, 5.5,5.5,0.0],
                                                     [5.5,5.5,5.5, 5.5,5.5,5.5, 5.5,1.1,5.5],
                                                     [5.5,5.5,5.5, 5.5,5.5,1.1, 5.5,5.5,5.5],
                                                     [5.5,1.1,5.5, 5.5,5.5,5.5, 0.0,0.0,0.0],
                                                     [5.5,5.5,5.5, 5.5,5.5,0.0, 0.0,0.0,0.0],
                                                     [0.0,0.0,5.5, 1.1,5.5,0.0, 0.0,0.0,0.0]])
    
    hd_restart_file_prep_input_ref_res_nums =  np.array([[5.5,5.5,5.5, 5.5,5.5,5.5, 5.5,5.5,5.5],
                                                         [5.5,1.1,1.1, 1.1,5.5,5.5, 5.5,5.5,5.5],
                                                         [5.5,1.1,1.1, 1.1,5.5,5.5, 5.5,5.5,1.1],
                                                         [5.5,1.1,1.1, 1.1,5.5,5.5, 5.5,5.5,0.0],
                                                         [5.5,5.5,5.5, 5.5,5.5,5.5, 5.5,1.1,5.5],
                                                         [5.5,5.5,5.5, 5.5,5.5,1.1, 5.5,5.5,5.5],
                                                         [5.5,1.1,5.5, 5.5,5.5,5.5, 0.0,0.0,0.0],
                                                         [5.5,5.5,5.5, 5.5,5.5,0.0, 0.0,0.0,0.0],
                                                         [0.0,0.0,1.1, 1.1,5.5,0.0, 0.0,0.0,0.0]])
    
    hd_restart_file_prep_expected_results = np.array([[1.9,1.8,0.1, 0.1,1.0,0.1, 0.2, 2.112698412698413,1.9],
                                                      [0.1,1.9,1.8, 1.0,0.6,0.2, 0.3, 2.112698412698413,1.9],
                                                      [0.1,0.2, 2.112698412698413, 0.6,0.5,0.3, 2.3,2.3,2.0],
                                                      [0.2,0.2,8.1, 8.1,0.3,0.3, 2.3,2.0,0.0],
                                                      [0.1,0.2,0.1, 8.1,0.9,0.3, 3.4,8.0,2.0],
                                                      [0.2,0.3,0.3, 0.0,1.1,8.0, 8.0,7.0,3.0],
                                                      [0.6,0.7,0.7, 1.0,1.1,8.0, 0.0,0.0,0.0],
                                                      [0.2,0.3,0.4, 0.7,9.0,0.0, 0.0,0.0,0.0],
                                                      [0.0,0.0,0.7, 9.0,5.0,0.0, 0.0,0.0,0.0]])
    
    hd_restart_file_overlandorbase_flow_input_field = np.array([[0.0,2.0,3.0,0.0],
                                                                [5.0,0.0,7.0,8.0],
                                                                [9.0,8.0,0.0,6.0],
                                                                [0.0,4.0,3.0,2.0]])
    
    hd_restart_file_overlandorbase_flow_res_nums = np.array([[0.0,1.1,1.1,1.1],
                                                             [1.1,1.1,1.1,1.1],
                                                             [0.0,1.1,0.0,1.1],
                                                             [0.0,1.1,1.1,1.1]])
    
    hd_restart_file_overlandorbase_flow_ref_res_nums = np.array([[0.0,1.1,1.1,1.1],
                                                                 [1.1,0.0,1.1,1.1],
                                                                 [0.0,0.0,1.1,1.1],
                                                                 [1.1,1.1,1.1,1.1]])

    hd_restart_file_overlandorbase_expected_results = np.array([[0.0,2.0,3.0,0.0],
                                                                [5.0,8.0,7.0,8.0],
                                                                [0.0,8.0,0.0,6.0],
                                                                [0.0,4.0,3.0,2.0]])
    
    def testReplacingZerosWithGlobalPostiveAverage(self):
        """Test utility for replacing zeros in a field with the global postive average of the field"""
        test_field = field.makeField(self.replacing_zeros_test_data.copy(),'ReservoirSize',
                                     grid_type='LatLong',nlat=5,nlong=5)
        test_field.replace_zeros_with_global_postive_value_average()
        np.testing.assert_almost_equal(test_field.get_data(),self.replacing_zeros_expected_results,
                                       decimal=12)
    
    def testReplacingZerosWithHighestValueNeighbor(self):
        """Test utility for replacing zeros with the nearest neighbor (if it is greater)"""
        test_field = field.makeField(self.replacing_zeros_test_data.copy(),'ReservoirSize',
                                     grid_type='LatLong',nlat=5,nlong=5)
        test_field.replace_zeros_with_highest_valued_neighbor()
        np.testing.assert_almost_equal(test_field.get_data(), self.replacing_zero_with_highest_neighbor_expected_results,
                                       decimal=12)
        
    def testReplacingZerosWithHighestValueNeighborTwo(self):
        """Test utility for replacing zeros with the nearest neighbor (if it is greater)"""
        test_field = field.makeField(self.replacing_zeros_test_data_two.copy(),'ReservoirSize',
                                     grid_type='LatLong',nlat=5,nlong=5)
        test_field.replace_zeros_with_highest_valued_neighbor()
        np.testing.assert_almost_equal(test_field.get_data(),
                                       self.replacing_zero_with_highest_neighbor_expected_results_two,
                                       decimal=12)
        
    def testHDRestartFieldreparation(self):
        """Test mid-level function that prepares one river reservoir field for a hd restart file"""
        test_field = field.makeField(self.hd_restart_file_prep_input_field,'ReservoirSize',
                                     grid_type='LatLong',nlat=9,nlong=9)
        test_res_nums = field.makeField(self.hd_restart_file_prep_input_res_nums,'Generic',
                                        grid_type='LatLong',nlat=9,nlong=9)
        test_ref_res_nums = field.makeField(self.hd_restart_file_prep_input_ref_res_nums,'Generic',
                                        grid_type='LatLong',nlat=9,nlong=9)
        results_field = utilities.prepare_hdrestart_field(input_field=test_field, 
                                                          resnum_riv_field=test_res_nums,
                                                          ref_resnum_field=test_ref_res_nums)
        np.testing.assert_almost_equal(results_field.get_data(),self.hd_restart_file_prep_expected_results,
                                       decimal=12)
        
    def testHDRestartFieldPreparationOverlandOrBaseflow(self):
        """Test mid-level function that prepares one overland/base flow reservoir field for a hd restart file"""
        test_field = field.makeField(self.hd_restart_file_overlandorbase_flow_input_field,'ReservoirSize',
                                     grid_type='LatLong',nlat=4,nlong=4)
        test_res_nums = field.makeField(self.hd_restart_file_overlandorbase_flow_res_nums,'Generic',
                                        grid_type='LatLong',nlat=4,nlong=4)
        test_ref_res_nums = field.makeField(self.hd_restart_file_overlandorbase_flow_ref_res_nums,'Generic',
                                        grid_type='LatLong',nlat=4,nlong=4)
        results_field = utilities.prepare_hdrestart_field(input_field=test_field, 
                                                          resnum_riv_field=test_res_nums,
                                                          ref_resnum_field=test_ref_res_nums,
                                                          is_river_res_field=False)
        np.testing.assert_almost_equal(results_field.get_data(),
                                       self.hd_restart_file_overlandorbase_expected_results,
                                       decimal=12)

class TestOrogCorrectionFieldCreationAndApplication(unittest.TestCase):
    """Test the creation and application of fields of relative catchment corrections"""
    
    original_field_input = np.array([[1.0,3.4,5.0],
                                     [7.9,8.4,2.0],
                                     [1.6,-1.0,100.10]])
    
    corrected_field_input = np.array([[20.0,3.4,4.0],
                                     [7.8,8.7,2.0],
                                     [1.6,2.0,1000.70]])
    
    orog_correction_field_expected_result = np.array([[19.0,0,-1.0],
                                                      [-0.1,0.3,0.0],
                                                      [0.0,3.0,900.60]])
    
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

    def testOrogCorrectionFieldCreation(self):
        """Test the creation of a field of relative orography corrections"""
        dynamic_hd.write_field(self.directory + '/orog_corr_test_input_original_field.nc', 
                               field.Field(self.original_field_input,'LatLong',nlat=3,nlong=3),'.nc')
        dynamic_hd.write_field(self.directory + '/orog_corr_test_input_corrected_field.nc',
                               field.Field(self.corrected_field_input,'LatLong',nlat=3,nlong=3),'.nc')
        utilities.generate_orog_correction_field(original_orography_filename=\
                                                 self.directory + '/orog_corr_test_input_original_field.nc',
                                                 corrected_orography_filename=\
                                                 self.directory + '/orog_corr_test_input_corrected_field.nc',
                                                 orography_corrections_filename=\
                                                 self.directory + '/orog_corr_test_output_corrs.nc'
                                                 ,grid_type='LatLong',nlat=3,nlong=3)
        corrs_in_file = dynamic_hd.load_field(self.directory + '/orog_corr_test_output_corrs.nc',
                                              '.nc', 'Generic', True, timeslice=None, grid_type='LatLong',
                                              nlat=3,nlong=3)
        np.testing.assert_array_almost_equal(corrs_in_file.get_data(), self.orog_correction_field_expected_result,
                                             decimal=12, err_msg="Generation of relative orography correction field"
                                             " doesn't produce expected result")
    
    def testOrogCorrectionFieldApplication(self):
        """Test the application of a field of relative orography corrections"""
        dynamic_hd.write_field(self.directory + '/orog_corr_test_input_original_field.nc', 
                               field.Field(self.original_field_input,'LatLong',nlat=3,nlong=3),'.nc')
        dynamic_hd.write_field(self.directory + '/orog_corr_test_input_original_field_copy.nc', 
                               field.Field(self.original_field_input,'LatLong',nlat=3,nlong=3),'.nc')
        dynamic_hd.write_field(self.directory + '/orog_corr_test_input_corrected_field.nc',
                               field.Field(self.corrected_field_input,'LatLong',nlat=3,nlong=3),'.nc')
        utilities.generate_orog_correction_field(original_orography_filename=\
                                                 self.directory + '/orog_corr_test_input_original_field.nc',
                                                 corrected_orography_filename=\
                                                 self.directory + '/orog_corr_test_input_corrected_field.nc',
                                                 orography_corrections_filename=\
                                                 self.directory + '/orog_corr_test_output_corrs.nc'
                                                 ,grid_type='LatLong',nlat=3,nlong=3)
        utilities.apply_orog_correction_field(original_orography_filename=\
                                              self.directory + '/orog_corr_test_input_original_field_copy.nc',
                                              orography_corrections_filename=self.directory + '/orog_corr_test_output_corrs.nc',
                                              corrected_orography_filename=\
                                              self.directory + '/orog_corr_test_output_corrected_field.nc',
                                              grid_type='LatLong',nlat=3,nlong=3)
        corrs_in_file = dynamic_hd.load_field(self.directory + '/orog_corr_test_output_corrected_field.nc',
                                              '.nc', 'Generic', True, timeslice=None, grid_type='LatLong',
                                              nlat=3,nlong=3)
        np.testing.assert_array_equal(corrs_in_file.get_data(), self.corrected_field_input,
                                      "Generating then applying an orography correction field doesn't produce"
                                      " expected result")

class TestIntelligentBurning(unittest.TestCase):
    """Object containing a set of unit tests for the intelligent burning utility function"""

    fine_fmap_field = np.array([[12,61,17,18,22,45,55,45, 8, 9, 1,34,45,23,37,32, 9,31],
                                [45, 8,62, 1,34,45,23,37,32, 9,31,38,21,38,29,28,14,17],
                                [34,12,63,48,18,36,28,23,27,35, 1, 4,47,58,52,35,37,32],
                                [12,13,17,64,22,45,55,45,70,71,72,34,45,23,37,32, 9,31],
                                [45, 8, 9, 1,65,45,23,69,32, 9,73,38,21,38,29,28,14,17],
                                [34,12,38,48,66,36,68,23,27,35,74, 4,47,58,52,35,37,32],
                                [12,13,17,18,22,67,55,45, 8, 9,75,76,45,23,37,32, 9,31],
                                [34,12,38,48,18,36,28,23,27,35, 1,77,47,58,52,35,37,32],
                                [45, 8, 9, 1,34,45,23,37,32, 9,31,78,79,80,29,28,14,17],
                                [12,13,17,18,22,45,55,45, 8, 9, 1,34,45,83,37,32, 9,31],
                                [34,12,38,48,18,36,28,23,27,35, 1,87,47,84,52,35,37,32],
                                [45, 8, 9, 1,34,45,23,37,32, 9,88,38,86,85,29,28,14,17],
                                [34,12,38,48,18,36,28,23,27,35,89, 4,47,58,52,35,37,32],
                                [12,13,17,18,22,45,55,45, 8, 9, 1,90,45,23,37,32, 9,31],
                                [34,12,38,48,18,36,28,23,27,35, 1, 4,91,58,52,35,37,32],
                                [45, 8, 9, 1,34,45,23,37,97, 96,31,38,92,38,29,28,14,17],
                                [45, 8, 9, 1,34,45,23,98,32, 95,94,93,21,38,29,28,14,17],
                                [12,13,17,18,22,45,55,99, 8, 9, 1,34,45,23,37,32, 9,31],
                                [34,12,38,48,18,36,28,101,103,35, 1, 4,47,58,52,35,37,32],
                                [34,12,38,48,18,36,28,23,27,107, 108, 109,47,58,52,35,37,32],
                                [45, 8, 9, 1,34,45,23,37,32, 9,31,38,111,112,113,114,115,17]])
    
    fine_orog_field = np.array([[2505,38,1504,1504,1504,1504,985,985,985,985,985,985,985,985,985,985,985,985],
                                [1293,985,52,985,985,985,985,985,985,985,985,985,985,985,985,985,985,985],
                                [1842,866,35,866,866,866,866,866,866,866,866,866,866,866,866,866,866,866],
                                [1820,866,866,61,866,866,866,866, 9, 1,21,866,866,866,866,866,866,866],
                                [1001,866,866,866,12,866,866, 2,866, 9,28,866,866,866,866,866,866,866],
                                [988,866,866,866,44, 866, 8,866,866,866,27,866,866,866,866,866,866,866],
                                [999,866,866,866,866,57,866,866,866,866,15,12,866,866,866,866,866,866],
                                [978,866,866,866,866,866,866,866,866,866,866, 8,866,866,866,866,866,866],
                                [1882,988,988,988,988,988,988,988,988,988,988,33,101,12,988,988,988,988],
                                [1828,1823,1823,1823,1823,1823,823,823,823,823,823,1823,1823,3,1823,823,823,823],
                                [837,  983, 983, 983, 983, 983,983,983,983,983,983,  18,983,32,983,983,983,983],
                                [983,  983, 818, 818, 818, 818,818,818,818,818,  9,818,16,31,818,818,818,818],
                                [982,818,818,818,818,818,818,818,818,818,23,818,818,818,818,818,818,818],
                                [839,839,975,975,975,975,975,975,975,975,975,113,975,975,975,975,975,975],
                                [874,975,975,975,975,975,975,975,975,975,975,975,32,975,975,975,975,975],
                                [838,1080,1080,1080,1080,1080,1080,1080, 27, 16,1080,1080, 82,1080,1080,1080,1080,1080],
                                [828, 914, 914, 914, 914, 914, 914,  28,914, 48,  18,  13,914, 914, 914, 914, 914, 914],
                                [928, 914, 914, 914, 914, 914, 914,  39,914,914, 914, 914,914, 914, 914, 914, 914, 914],
                                [918,914,914,914,914,914,914, 10, 10,914,914,914,914,914,914,914,914,914],
                                [920,914,914,914,914,914,914,914,914,11, 11, 11,914,914,914,914,914,914],
                                [983,914,914,914,914,914,914,914,914,914,914,914,15,15,108,16,33,914]])
    
    expected_orog_field = np.array([[ 52,358,387,452,385,384],
                                    [384, 61,  9, 28,350,450],
                                    [  7, 57,239, 33,101,323],
                                    [293,394,399, 18, 32,322],
                                    [283,493,377,113, 32,485],
                                    [394,261, 39, 48, 82,439],
                                    [230,473, 10, 11,108,33]])
    
    input_orog_field = np.array([[350,358,387,452,385,384],
                                 [384,460,358,360,350,450],
                                 [  7,345,239,350,400,323],
                                 [293,394,399,303,421,322],
                                 [283,493,377,384,384,485],
                                 [394,261,490,483,828,439],
                                 [230,473,402,372,340,329]])
    
    input_orog_field_some_points_lower_than_fine_orog = np.array([[52,358,387,452,385,384],
                                                                  [384,38,2,360,350,450],
                                                                  [  7,345,239,350,400,323],
                                                                  [293,394,399,303,421,322],
                                                                  [283,493,377,384,384,485],
                                                                  [394,261,490,483,828,439],
                                                                  [230,473,402,372,340,329]])
    
    input_orog_field_flipped_and_rotated = np.array([[372,340,329,230,473,402],
                                                     [483,828,439,394,261,490],
                                                     [384,384,485,283,493,377],
                                                     [303,421,322,293,394,399],
                                                     [350,400,323,  7,345,239],
                                                     [360,350,450,384,460,358],
                                                     [452,385,384,350,358,387]])
    
    expected_orog_field_bottomrightcorneronly = np.array([[350,358,387,452,385,384],
                                                          [384,460,358,360,350,450],
                                                          [  7,345,239,350,400,323],
                                                          [293,394,399,303,421,322],
                                                          [283,493,377,113, 32,485],
                                                          [394,261,490, 48, 82,439],
                                                          [230,473,402, 11,108, 33]])
    
    expected_orog_field_bottomleftcorneronly = np.array([[350,358,387,452,385,384],
                                                         [384,460,358,360,350,450],
                                                         [  7,345,239,350,400,323],
                                                         [293,394,399,303,421,322],
                                                         [283,493,377,113,384,485],
                                                         [394,261, 39, 48,828,439],
                                                         [230,473, 10, 11,340,329]])
    
    expected_orog_field_toponly = np.array([[52,358,387,452,385,384],
                                            [384,61,  9, 28,350,450],
                                            [  7,345,239,350,400,323],
                                            [293,394,399,303,421,322],
                                            [283,493,377,384,384,485],
                                            [394,261,490,483,828,439],
                                            [230,473,402,372,340,329]])
    
    expected_orog_field_toponly_some_points_lower_in_fine_orog = \
                                  np.array([[52,358,387,452,385,384],
                                            [384,38,  2, 28,350,450],
                                            [  7,345,239,350,400,323],
                                            [293,394,399,303,421,322],
                                            [283,493,377,384,384,485],
                                            [394,261,490,483,828,439],
                                            [230,473,402,372,340,329]])
    
    expected_orog_field_middleonly = np.array([[350,358,387,452,385,384],
                                               [384,460,358,360,350,450],
                                               [  7,345,239, 33,400,323],
                                               [293,394,399, 18,421,322],
                                               [283,493,377,113,384,485],
                                               [394,261,490,483,828,439],
                                               [230,473,402,372,340,329]])
    
    expected_orog_field_driver = np.array([[350,358,387,452,385,384],
                                           [384,460,358,360,350,450],
                                           [  7,345,239, 33,400,323],
                                           [293,394,399, 18,421,322],
                                           [283,493,377,113,384,485],
                                           [394,261,490,483,828,439],
                                           [230,473,402,372,108,33]])
    
    expected_orog_field_driver_flipped_and_rotated = np.array([[372,108,33 ,230,473,402],
                                                               [483,828,439,394,261,490],
                                                               [113,384,485,283,493,377],
                                                               [ 18,421,322,293,394,399],
                                                               [ 33,400,323,  7,345,239],
                                                               [360,350,450,384,460,358],
                                                               [452,385,384,350,358,387]])
    
    def testIntelligentBurningWholeArray(self):
        """Test intelligent burning across an entire array"""
        input_fine_orography_field = field.makeField(self.fine_orog_field, 
                                                     field_type='Orography', 
                                                     grid_type='LatLong',
                                                     nlat=21,nlong=18)
        course_orography_field = field.makeField(self.input_orog_field, 
                                                field_type='Orography', 
                                                grid_type='LatLong',
                                                nlat=7,nlong=6)
        input_fine_fmap = field.makeField(self.fine_fmap_field,
                                          field_type='CumulativeFlow',
                                          grid_type='LatLong',
                                          nlat=21,nlong=18)
        burnt_orog = utilities.intelligently_burn_orography(input_fine_orography_field, 
                                                            course_orography_field, 
                                                            input_fine_fmap, 
                                                            threshold=60, 
                                                            region={'lat_min':0,
                                                                    'lat_max':6,
                                                                    'lon_min':0,
                                                                    'lon_max':5}, 
                                                            course_grid_type='LatLong',
                                                            nlat=7,nlong=6)
        np.testing.assert_array_equal(self.expected_orog_field,
                                      burnt_orog.get_data(),
                                      "Testing intelligent burn on a whole array doesn't"
                                      " produce expected results")
        
    def testIntelligentBurningBottomRightCorner(self):
        """Test intelligent burning in the bottom right corner of an array only"""
        input_fine_orography_field = field.makeField(self.fine_orog_field, 
                                                     field_type='Orography', 
                                                     grid_type='LatLong',
                                                     nlat=21,nlong=18)
        course_orography_field = field.makeField(self.input_orog_field, 
                                                field_type='Orography', 
                                                grid_type='LatLong',
                                                nlat=7,nlong=6)
        input_fine_fmap = field.makeField(self.fine_fmap_field,
                                          field_type='CumulativeFlow',
                                          grid_type='LatLong',
                                          nlat=21,nlong=18)
        burnt_orog = utilities.intelligently_burn_orography(input_fine_orography_field, 
                                                            course_orography_field, 
                                                            input_fine_fmap, 
                                                            threshold=60, 
                                                            region={'lat_min':4,
                                                                    'lat_max':6,
                                                                    'lon_min':3,
                                                                    'lon_max':5}, 
                                                            course_grid_type='LatLong',
                                                            nlat=7,nlong=6)
        np.testing.assert_array_equal(burnt_orog.get_data(),
                                      self.expected_orog_field_bottomrightcorneronly,
                                      "Testing intelligent burn on bottom right corner of array"
                                      " only doesn't produce expected results")
        
    def testIntelligentBurningBottomLeftCorner(self):
        """Test intelligent burning in the bottom left corner of an array only"""
        input_fine_orography_field = field.makeField(self.fine_orog_field, 
                                                     field_type='Orography', 
                                                     grid_type='LatLong',
                                                     nlat=21,nlong=18)
        course_orography_field = field.makeField(self.input_orog_field, 
                                                field_type='Orography', 
                                                grid_type='LatLong',
                                                nlat=7,nlong=6)
        input_fine_fmap = field.makeField(self.fine_fmap_field,
                                          field_type='CumulativeFlow',
                                          grid_type='LatLong',
                                          nlat=21,nlong=18)
        burnt_orog = utilities.intelligently_burn_orography(input_fine_orography_field, 
                                                            course_orography_field, 
                                                            input_fine_fmap, 
                                                            threshold=60, 
                                                            region={'lat_min':4,
                                                                    'lat_max':6,
                                                                    'lon_min':0,
                                                                    'lon_max':3}, 
                                                            course_grid_type='LatLong',
                                                            nlat=7,nlong=6)
        np.testing.assert_array_equal(burnt_orog.get_data(),
                                      self.expected_orog_field_bottomleftcorneronly,
                                      "Testing intelligent burn on bottom left corner of array"
                                      " only doesn't produce expected results")
        
    def testIntelligentBurningMiddle(self):
        """Test intelligent burning in the middle section of an array only"""
        input_fine_orography_field = field.makeField(self.fine_orog_field, 
                                                     field_type='Orography', 
                                                     grid_type='LatLong',
                                                     nlat=21,nlong=18)
        course_orography_field = field.makeField(self.input_orog_field, 
                                                field_type='Orography', 
                                                grid_type='LatLong',
                                                nlat=7,nlong=6)
        input_fine_fmap = field.makeField(self.fine_fmap_field,
                                          field_type='CumulativeFlow',
                                          grid_type='LatLong',
                                          nlat=21,nlong=18)
        burnt_orog = utilities.intelligently_burn_orography(input_fine_orography_field, 
                                                            course_orography_field, 
                                                            input_fine_fmap, 
                                                            threshold=60, 
                                                            region={'lat_min':2,
                                                                    'lat_max':4,
                                                                    'lon_min':2,
                                                                    'lon_max':3}, 
                                                            course_grid_type='LatLong',
                                                            nlat=7,nlong=6)
        np.testing.assert_array_equal(burnt_orog.get_data(),
                                      self.expected_orog_field_middleonly,
                                      "Testing intelligent burn on middle of array"
                                      " only doesn't produce expected results")
        
    def testIntelligentBurningTop(self):
        """Test intelligent burning in the top section of an array only"""
        input_fine_orography_field = field.makeField(self.fine_orog_field, 
                                                     field_type='Orography', 
                                                     grid_type='LatLong',
                                                     nlat=21,nlong=18)
        course_orography_field = field.makeField(self.input_orog_field, 
                                                field_type='Orography', 
                                                grid_type='LatLong',
                                                nlat=7,nlong=6)
        input_fine_fmap = field.makeField(self.fine_fmap_field,
                                          field_type='CumulativeFlow',
                                          grid_type='LatLong',
                                          nlat=21,nlong=18)
        burnt_orog = utilities.intelligently_burn_orography(input_fine_orography_field, 
                                                            course_orography_field, 
                                                            input_fine_fmap, 
                                                            threshold=60, 
                                                            region={'lat_min':0,
                                                                    'lat_max':1,
                                                                    'lon_min':0,
                                                                    'lon_max':5}, 
                                                            course_grid_type='LatLong',
                                                            nlat=7,nlong=6)
        np.testing.assert_array_equal(burnt_orog.get_data(),
                                      self.expected_orog_field_toponly,
                                      "Testing intelligent burn on top two rows of array"
                                      " only doesn't produce expected results")
        
    def testIntelligentBurningTopSomePointsHigherInFineOrography(self):
        """Test intelligent burning in the top section of an array were some pointer are higher in the fine orography"""
        input_fine_orography_field = field.makeField(self.fine_orog_field, 
                                                     field_type='Orography', 
                                                     grid_type='LatLong',
                                                     nlat=21,nlong=18)
        course_orography_field = field.makeField(self.\
                                                 input_orog_field_some_points_lower_than_fine_orog, 
                                                 field_type='Orography', 
                                                 grid_type='LatLong',
                                                 nlat=7,nlong=6)
        input_fine_fmap = field.makeField(self.fine_fmap_field,
                                          field_type='CumulativeFlow',
                                          grid_type='LatLong',
                                          nlat=21,nlong=18)
        burnt_orog = utilities.intelligently_burn_orography(input_fine_orography_field, 
                                                            course_orography_field, 
                                                            input_fine_fmap, 
                                                            threshold=60, 
                                                            region={'lat_min':0,
                                                                    'lat_max':1,
                                                                    'lon_min':0,
                                                                    'lon_max':5}, 
                                                            course_grid_type='LatLong',
                                                            nlat=7,nlong=6)
        np.testing.assert_array_equal(burnt_orog.get_data(),
                                      self.\
                                      expected_orog_field_toponly_some_points_lower_in_fine_orog,
                                      "Testing intelligent burn on top two rows of array"
                                      " only doesn't produce expected results")
        
    def testIntelligentBurningDriver(self):
        """Test driver function for intelligent burning plus the entire procedure"""
        tmp_dir = data_dir + '/temp'
        test_regions_text_file = tmp_dir + '/test_regions_list.txt'
        test_fine_orog_file = tmp_dir + '/int_burn_test_fine_orog.nc'
        test_fine_fmap_file = tmp_dir + '/int_burn_test_fmap_orog.nc'
        test_course_orog_file = tmp_dir + '/int_burn_test_course_orog.nc'
        test_course_output_orog_file = tmp_dir + '/int_burn_test_course_output_orog.nc'
        #setup files
        try:
            os.stat(tmp_dir)
        except:
            os.mkdir(tmp_dir) 
        with open(test_regions_text_file,'w') as f:
            f.write("course_grid_type=LatLong \n"
                    "fine_grid_type=LatLong \n"
                    "course_grid_flipud=False \n"
                    "course_grid_rotate180lr=False \n" 
                    "fine_grid_flipud = False \n"
                    "fine_grid_rotate180lr = False \n"
                    "#Trial Regions \n"
                    "lat_min=2,lat_max=4,lon_min=2,lon_max=3,threshold=60 \n"
                    "lat_min=6,lat_max=6,lon_min=0,lon_max=5,threshold=113")
        dynamic_hd.write_field(test_fine_orog_file,
                               field.makeField(self.fine_orog_field, 
                                                     field_type='Orography', 
                                                     grid_type='LatLong',
                                                     nlat=21,nlong=18),
                               file_type=dynamic_hd.get_file_extension(test_fine_orog_file))
        dynamic_hd.write_field(test_fine_fmap_file,
                               field.makeField(self.fine_fmap_field, 
                                               field_type='CumulativeFlow', 
                                               grid_type='LatLong',
                                               nlat=21,nlong=18),
                               file_type=dynamic_hd.get_file_extension(test_fine_fmap_file))
        dynamic_hd.write_field(test_course_orog_file,
                               field.makeField(self.input_orog_field,
                                               field_type='Orography', 
                                               grid_type='LatLong',
                                               nlat=7,nlong=6),
                               file_type=dynamic_hd.get_file_extension(test_course_orog_file))
        #run intelligent burning code
        utilities.intelligent_orography_burning_driver(input_fine_orography_filename=test_fine_orog_file, 
                                                       input_course_orography_filename=test_course_orog_file, 
                                                       input_fine_fmap_filename=test_fine_fmap_file, 
                                                       output_course_orography_filename=test_course_output_orog_file, 
                                                       regions_to_burn_list_filename=test_regions_text_file, 
                                                       fine_grid_type='LatLong', 
                                                       course_grid_type='LatLong', 
                                                       fine_grid_kwargs={'nlat':21,'nlong':18},
                                                       **{'nlat':7,'nlong':6})
        #load results
        loaded_results = dynamic_hd.load_field(test_course_output_orog_file, 
                                               file_type=\
                                               dynamic_hd.get_file_extension(test_course_output_orog_file), 
                                               field_type='Generic', grid_type='LatLong',nlat=7,nlong=6)
        np.testing.assert_array_equal(loaded_results.get_data(), 
                                      self.expected_orog_field_driver,
                                      "Testing the intelligent burning driver didn't produce the expected"
                                      " results")
        
    def testIntelligentBurningDriverFlippedAndRotated(self):
        """Test the entire intelligent burning procedure where field needs flipping and rotation"""
        tmp_dir = data_dir + '/temp'
        test_regions_text_file = tmp_dir + '/test_regions_list.txt'
        test_fine_orog_file = tmp_dir + '/int_burn_test_fine_orog.nc'
        test_fine_fmap_file = tmp_dir + '/int_burn_test_fmap_orog.nc'
        test_course_orog_file = tmp_dir + '/int_burn_test_course_orog.nc'
        test_course_output_orog_file = tmp_dir + '/int_burn_test_course_output_orog.nc'
        #setup files
        try:
            os.stat(tmp_dir)
        except:
            os.mkdir(tmp_dir) 
        with open(test_regions_text_file,'w') as f:
            f.write("course_grid_type=LatLong \n"
                    "fine_grid_type=LatLong \n"
                    "course_grid_flipud=True \n"
                    "course_grid_rotate180lr=True \n" 
                    "fine_grid_flipud = False \n"
                    "fine_grid_rotate180lr = False \n"
                    "#Trial Regions \n"
                    "lat_min=2,lat_max=4,lon_min=2,lon_max=3,threshold=60 \n"
                    "lat_min=6,lat_max=6,lon_min=0,lon_max=5,threshold=113")
        dynamic_hd.write_field(test_fine_orog_file,
                               field.makeField(self.fine_orog_field, 
                                                     field_type='Orography', 
                                                     grid_type='LatLong',
                                                     nlat=21,nlong=18),
                               file_type=dynamic_hd.get_file_extension(test_fine_orog_file))
        dynamic_hd.write_field(test_fine_fmap_file,
                               field.makeField(self.fine_fmap_field, 
                                               field_type='CumulativeFlow', 
                                               grid_type='LatLong',
                                               nlat=21,nlong=18),
                               file_type=dynamic_hd.get_file_extension(test_fine_fmap_file))
        dynamic_hd.write_field(test_course_orog_file,
                               field.makeField(self.input_orog_field_flipped_and_rotated,
                                               field_type='Orography', 
                                               grid_type='LatLong',
                                               nlat=7,nlong=6),
                               file_type=dynamic_hd.get_file_extension(test_course_orog_file))
        #run intelligent burning code
        utilities.intelligent_orography_burning_driver(input_fine_orography_filename=test_fine_orog_file, 
                                                       input_course_orography_filename=test_course_orog_file, 
                                                       input_fine_fmap_filename=test_fine_fmap_file, 
                                                       output_course_orography_filename=test_course_output_orog_file, 
                                                       regions_to_burn_list_filename=test_regions_text_file, 
                                                       fine_grid_type='LatLong', 
                                                       course_grid_type='LatLong', 
                                                       fine_grid_kwargs={'nlat':21,'nlong':18},
                                                       **{'nlat':7,'nlong':6})
        #load results
        loaded_results = dynamic_hd.load_field(test_course_output_orog_file, 
                                               file_type=\
                                               dynamic_hd.get_file_extension(test_course_output_orog_file), 
                                               field_type='Generic', grid_type='LatLong',nlat=7,nlong=6)
        np.testing.assert_array_equal(loaded_results.get_data(), 
                                      self.expected_orog_field_driver_flipped_and_rotated,
                                      "Testing the intelligent burning driver didn't produce the expected"
                                      " results")

class TestLSMaskDownScaling(unittest.TestCase):
    """A set of tests of the utility to downscale a landsea mask"""
    
    nlat_test = 3
    nlon_test = 5
    theoretical_test_field = np.arange(0,nlat_test*nlon_test).reshape((nlat_test,
                                                                       nlon_test))
    theoretical_test_expected_results= np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4],
                                                [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4],
                                                [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4],
                                                [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4],
                                                [5,5,5,6,6,6,7,7,7,8,8,8,9,9,9],
                                                [5,5,5,6,6,6,7,7,7,8,8,8,9,9,9],
                                                [5,5,5,6,6,6,7,7,7,8,8,8,9,9,9],
                                                [5,5,5,6,6,6,7,7,7,8,8,8,9,9,9],
                                                [10,10,10,11,11,11,12,12,12,13,13,13,14,14,14],
                                                [10,10,10,11,11,11,12,12,12,13,13,13,14,14,14],
                                                [10,10,10,11,11,11,12,12,12,13,13,13,14,14,14],
                                                [10,10,10,11,11,11,12,12,12,13,13,13,14,14,14]])
    
    realistic_test_input_field = np.array([[0,0,0,1,1],
                                           [1,0,1,1,0],
                                           [0,0,0,1,0]])
    
    realistic_test_expected_results = np.array([[0,0,0,0,0,0,1,1,1,1],
                                                [0,0,0,0,0,0,1,1,1,1],
                                                [0,0,0,0,0,0,1,1,1,1],
                                                [1,1,0,0,1,1,1,1,0,0],
                                                [1,1,0,0,1,1,1,1,0,0],
                                                [1,1,0,0,1,1,1,1,0,0],
                                                [0,0,0,0,0,0,1,1,0,0],
                                                [0,0,0,0,0,0,1,1,0,0],
                                                [0,0,0,0,0,0,1,1,0,0]])
    
    def testLSMaskDownScalingUsingTheoreticalField(self):
        """Test using a field filled with a variety of numbers (i.e. not just 1's and 0's)"""
        local_theoretical_test_field = np.copy(self.theoretical_test_field)
        downscaled_ls_mask =  utilities.downscale_ls_mask(field.Field(local_theoretical_test_field,
                                                                      grid='LatLong',
                                                                      nlat=self.nlat_test,
                                                                      nlong=self.nlon_test),
                                                          fine_grid_type='LatLong',nlat=12,nlong=15)
        np.testing.assert_array_equal(local_theoretical_test_field, self.theoretical_test_field, 
                                      "LS mask downscaling theoretical results unexpected changes into field")
        np.testing.assert_array_equal(downscaled_ls_mask.get_data(),self.theoretical_test_expected_results,
                                      "LS mask downscaling theoretical test doesn't produce expected results")
    
    def testLSMaskDownScalingUsingRealisticField(self):
        """Test using a field filled with realistic numbers (i.e. 1's and 0's)"""
        local_realistic_test_field = np.copy(self.realistic_test_input_field)
        downscaled_ls_mask = utilities.downscale_ls_mask(field.Field(local_realistic_test_field,
                                                                     grid='LatLong',
                                                                     nlat=self.nlat_test,
                                                                     nlong=self.nlon_test),
                                                         fine_grid_type='LatLong',nlat=9,nlong=10)
        np.testing.assert_array_equal(downscaled_ls_mask.get_data(),self.realistic_test_expected_results,
                                      "LS mask downscaling realistic setup test doesn't produce" 
                                      " expected results") 

class TestTrueSinkDownScaling(unittest.TestCase):
    """Test of the down scaling of a the position of true sinks in a field"""
    
    find_area_minima_test_input_orog = np.array([[1.0,2.0,3.0,4.0, 4.0,4.0,5.0,6.0, 7.0,7.5,5.0,6.5],
                                                 [4.0,4.5,1.5,7.0, 4.0,3.0,3.0,7.0, 8.0,3.5,6.0,7.0],
                                                 [5.0,4.3,6.6,9.0, 5.0,4.5,8.0,3.3, 9.0,7.0,8.0,4.5],
                                                 [9.9,1.1,1.1,3.0, 3.0,3.0,9.1,7.0, 4.0,8.8,9.7,3.4],

                                                 [1.0,2.0,3.0,4.0, 4.0,4.0,5.0,6.0, 7.0,7.5,5.0,6.5],
                                                 [4.0,4.5,1.5,7.0, 4.0,3.0,3.0,7.0, 8.0,3.5,6.0,7.0],
                                                 [5.0,4.3,6.6,9.0, 5.0,4.5,8.0,3.3, 9.0,7.0,8.0,4.5],
                                                 [9.9,1.1,1.1,3.0, 3.0,3.0,9.1,7.0, 4.0,8.8,9.7,3.4],
                                                 
                                                 [1.0,2.0,3.0,4.0, 4.0,4.0,5.0,6.0, 7.0,7.5,5.0,6.5],
                                                 [4.0,4.5,1.5,7.0, 4.0,3.0,3.0,7.0, 8.0,2.5,6.0,7.0],
                                                 [5.0,4.3,6.6,9.0, 5.0,4.5,8.0,3.3, 9.0,7.0,8.0,4.5],
                                                 [9.9,0.9,1.1,3.0, 3.0,3.0,9.1,2.0, 4.0,8.8,9.7,3.4]])
    
    input_true_sinks_on_course_scale = np.array([[True,False,False],
                                                 [False,True,True],
                                                 [True,False,False]])

    find_area_minima_test_expected_output =  np.array([[True,False,False,False, False,False,False,False, False,False,False,False],
                                                       [False,False,False,False, False,False,False,False, False,False,False,False],
                                                       [False,False,False,False, False,False,False,False, False,False,False,False],
                                                       [False,False,False,False, False,False,False,False, False,False,False,False],

                                                       [False,False,False,False, False,False,False,False, False,False,False,False],
                                                       [False,False,False,False, False,True,False,False, False,False,False,False],
                                                       [False,False,False,False, False,False,False,False, False,False,False,False],
                                                       [False,False,False,False, False,False,False,False, False,False,False,True],
                                                 
                                                       [False,False,False,False, False,False,False,False, False,False,False,False],
                                                       [False,False,False,False, False,False,False,False, False,False,False,False],
                                                       [False,False,False,False, False,False,False,False, False,False,False,False],
                                                       [False,True,False,False, False,False,False,False,  False,False,False,False]])    
    
    def testTrueSinkDownScaling(self):
        """Test the downscale of a true sink within a field"""
        true_sinks_downscaled = utilities.downscale_true_sink_points(field.Orography(self.find_area_minima_test_input_orog,
                                                                                     grid='LatLong',nlat=12,nlong=12), 
                                                                     field.Field(self.input_true_sinks_on_course_scale,
                                                                                 grid='LatLong',nlat=3,nlong=3))
        np.testing.assert_array_equal(true_sinks_downscaled,self.find_area_minima_test_expected_output,
                                      "Downscaling of true sinks doesn't produce expected results")

class TestUpscaling(unittest.TestCase):
    """Tests of upscaling a generic field using a variety of methods"""

    integer_input_array = np.array([[ 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                    [11,12,13,14,15,16,17,18,19],
                                    [21,22,23,24,25,26,27,28,29],
                                    [61,62,63,64,65,66,67,68,69],
                                    [59,58,57,56,55,54,53,52,51], 
                                    [41,42,43,44,45,46,47,48,49],
                                    [31,32,33,34,35,36,37,38,39],
                                    [71,72,73,74,75,76,77,78,79],
                                    [89,88,87,86,85,84,83,82,81],
                                    [91,92,93,94,95,96,97,98,99]],dtype=np.int64)
    
    integer_expected_output_field_sum = np.array([[ 42, 60, 78],
                                                  [252,270,288],
                                                  [300,300,300],
                                                  [312,330,348],
                                                  [540,540,540]],dtype=np.int64)
    
    integer_expected_output_field_max = np.array([[13,16,19],
                                                  [63,66,69],
                                                  [59,56,53],
                                                  [73,76,79],
                                                  [93,96,99]],dtype=np.int64)
    
    float_input_array = np.array([[ 1.5, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [11,12,13,14,15,16,17,18,19],
                                  [21,22,23,24,25,26,28.7,28.3,28.5],
                                  [61,62,63,64,65,66,67,68,69],
                                  [59,58,57.5,56,55,54,53,52,51], 
                                  [41,42,43,44,45,46,47,48,49],
                                  [31,32.3,33,34,35,36,37,38,39],
                                  [71,72,73,74,75,76,77,78,79],
                                  [89,88,87,86,85,84,83,82,81],
                                  [91,92,93,94,95,96,97,98,99.9]],dtype=np.float64)

    float_expected_output_field_sum = np.array([[42.5, 60, 78],
                                                [252,270,289.5],
                                                [300.5,300,300],
                                                [312.3,330,348],
                                                [540,540,540.9]],dtype=np.float64)
    
    float_expected_output_field_max = np.array([[63,66,69],
                                                [93,96,99.9]],dtype=np.float64)
    
    def setUp(self):
        """Unit testing setup function. Set up input fields."""
        self.integer_input_field = field.Field(np.copy(self.integer_input_array),
                                               'LatLong',nlat=10,nlong=9)
        self.float_input_field = field.Field(np.copy(self.float_input_array),
                                               'LatLong',nlat=10,nlong=9)

    def testUpscaleFieldOfIntegerUsingMax(self):
        """Test upscaling a field of integers by taking the max value in each course cell"""
        output_field = utilities.upscale_field(input_field=self.integer_input_field,
                                               output_grid_type='LatLong', 
                                               method='Max', 
                                               output_grid_kwargs ={'nlat':5,'nlong':3})
        np.testing.assert_array_equal(output_field.get_data(),self.integer_expected_output_field_max) 
        
    def testUpscaleFieldOfIntegerUsingSum(self):
        """Test upscaling a field of integers by taking the summed value in each course cell"""
        output_field = utilities.upscale_field(input_field=self.integer_input_field,
                                               output_grid_type='LatLong', 
                                               method='Sum', 
                                               output_grid_kwargs ={'nlat':5,'nlong':3})
        np.testing.assert_array_equal(output_field.get_data(),self.integer_expected_output_field_sum) 
        
    def testUpscaleFieldOfFloatUsingSum(self):
        """Test upscaling a field of floats by taking the max value in each course cell"""
        output_field = utilities.upscale_field(input_field=self.float_input_field,
                                               output_grid_type='LatLong', 
                                               method='Sum', 
                                               output_grid_kwargs ={'nlat':5,'nlong':3})
        np.testing.assert_array_equal(output_field.get_data(),self.float_expected_output_field_sum)
        
    def testUpscaleFieldOfFloatUsingMax(self):
        """Test upscaling a field of floats by taking the summed value in each course cell"""
        output_field = utilities.upscale_field(input_field=self.float_input_field,
                                               output_grid_type='LatLong', 
                                               method='Max', 
                                               output_grid_kwargs ={'nlat':2,'nlong':3})
        np.testing.assert_array_equal(output_field.get_data(),self.float_expected_output_field_max)
        
    def testUpscaleFieldOfFloatUsingMaxWithScalingFactor(self):
        """Test upscaling a field of floats by taking the max value in each course cell scaling by cell size"""
        output_field = utilities.upscale_field(input_field=self.float_input_field,
                                               output_grid_type='LatLong', 
                                               method='Max', 
                                               output_grid_kwargs ={'nlat':2,'nlong':3},
                                               scalenumbers=True)
        np.testing.assert_array_equal(output_field.get_data(),self.float_expected_output_field_max/15.0)
        
    def testUpscaleFieldOfIntegerUsingSumWithScalingFactor(self):
        """Test upscaling a field of floats by taking the summed value in each course cell scaling by cell size"""
        output_field = utilities.upscale_field(input_field=self.integer_input_field,
                                               output_grid_type='LatLong', 
                                               method='Sum', 
                                               output_grid_kwargs ={'nlat':5,'nlong':3},
                                               scalenumbers=True)
        np.testing.assert_array_equal(output_field.get_data(),self.integer_expected_output_field_sum/6.0)       

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
