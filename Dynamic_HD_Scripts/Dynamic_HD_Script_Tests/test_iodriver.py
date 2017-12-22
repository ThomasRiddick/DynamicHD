'''
Unit tests for the iodriver module 

Created on Dec 7, 2017

@author: thomasriddick
'''

import unittest
import Dynamic_HD_Scripts.iodriver as iodriver
from context import data_dir
from os.path import join
import os
import textwrap


class Test(unittest.TestCase):


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
        self.half_degree_grid_desc = os.path.join(self.directory,"half_degree_grid_desc.txt")
        self.ten_minute_grid_desc  = os.path.join(self.directory,"ten_minute_grid_desc.txt")

    def testLoadingFileWithLatLonFields(self):
        """Test loading from a file with a set of lat-lon coords built in"""

        loaded_field = iodriver.advanced_field_loader(filename=join(data_dir,
                                                                    "HDdata",
                                                                    "orographys",
                                                                    "ice5g_v1_2_00_0k_10min.nc"),
                                                      adjust_orientation=True)
        self.assertEquals(loaded_field.data[231][0],-1425.0,"Field data has not been loaded and"
                                                          " oriented properly")
        self.assertAlmostEqual(loaded_field.grid.lon_points[0],-179.91666667,msg="Field data grid is"
                                                                                 " not correctly"
                                                                                 " oriented")
        self.assertAlmostEqual(loaded_field.grid.lat_points[0],89.91666412,msg="Field data grid is"
                                                                               " not correctly"
                                                                               " oriented")

    def testLoadingFileWithLatLonFieldsWithoutReorientation(self):
        """Test loading from a file with a set of lat-lon coords built in without reorienting"""
        loaded_field = iodriver.advanced_field_loader(filename=join(data_dir,
                                                                    "HDdata",
                                                                    "orographys",
                                                                    "ice5g_v1_2_00_0k_10min.nc"),
                                                      adjust_orientation=False)
        self.assertEquals(loaded_field.data[231][0],-2367.0,"Field data has not been loaded and"
                                                            " oriented properly")
        self.assertAlmostEqual(loaded_field.grid.lon_points[0],0.0,msg="Field data grid is"
                                                                       " not correctly"
                                                                       " oriented")
        self.assertAlmostEqual(loaded_field.grid.lat_points[0],-89.91667175,msg="Field data grid is"
                                                                                " not correctly"
                                                                                " oriented")
    
    def testLoadingFieldWithGlobalGridDesc(self):
        """Test loading from a file using a global_DXY grid description"""
        loaded_field = iodriver.advanced_field_loader(filename=join(data_dir,
                                                                    "HDdata",
                                                                    "catchmentmaps",
                                                                    "upscaled",
                                                                    "catchmentmap_unsorted_ICE5G_"
                                                                    "data_ALG4_sinkless_0k_"
                                                                    "20160714_121938.nc"),
                                                      grid_desc="global_0.5",
                                                      adjust_orientation=True)
        self.assertEquals(loaded_field.data[62][157],14149.0,"Field data has not been loaded and"
                                                            " oriented properly")
        self.assertAlmostEqual(loaded_field.grid.lon_points[0],-179.75,msg="Field data grid is"
                                                                          " not correctly"
                                                                          " oriented")
        self.assertAlmostEqual(loaded_field.grid.lat_points[0],89.75,msg="Field data grid is"
                                                                         " not correctly"
                                                                         " oriented")
        
    def testLoadingFieldWithRXGridDesc(self):
        """Test loading from a file using a rDXxDY grid description"""
        loaded_field = iodriver.advanced_field_loader(filename=join(data_dir,
                                                                    "HDdata",
                                                                    "catchmentmaps",
                                                                    "upscaled",
                                                                    "catchmentmap_unsorted_ICE5G_"
                                                                    "data_ALG4_sinkless_0k_"
                                                                    "20160714_121938.nc"),
                                                      grid_desc="r0.5x0.5",
                                                      adjust_orientation=True)
        self.assertEquals(loaded_field.data[62][517],14149.0,"Field data has not been loaded and"
                                                            " oriented properly")
        self.assertAlmostEqual(loaded_field.grid.lon_points[0],-179.75,msg="Field data grid is"
                                                                          " not correctly"
                                                                          " oriented")
        self.assertAlmostEqual(loaded_field.grid.lat_points[0],89.75,msg="Field data grid is"
                                                                         " not correctly"
                                                                         " oriented")
        
    def testLoadingFieldWithRXGridDescWithoutReorientation(self):
        """Test loading from a file using a rDXxDY grid description without reorienting"""
        loaded_field = iodriver.advanced_field_loader(filename=join(data_dir,
                                                                    "HDdata",
                                                                    "catchmentmaps",
                                                                    "upscaled",
                                                                    "catchmentmap_unsorted_ICE5G_"
                                                                    "data_ALG4_sinkless_0k_"
                                                                    "20160714_121938.nc"),
                                                      grid_desc="r0.5x0.5",
                                                      adjust_orientation=False)
        self.assertEquals(loaded_field.data[62][157],14149.0,"Field data has not been loaded and"
                                                            " oriented properly")
        self.assertAlmostEqual(loaded_field.grid.lon_points[0],0.0,msg="Field data grid is"
                                                                       " not correctly"
                                                                       " oriented")
        self.assertAlmostEqual(loaded_field.grid.lat_points[0],89.75,msg="Field data grid is"
                                                                         " not correctly"
                                                                         " oriented")

    def testLoadingFieldWithGridFileDesc(self):
        """Test loading from a file using a grid description file"""
        half_degree_grid_desc_text =\
            """
            gridtype = lonlat
            gridsize = 259200
            xsize    = 720
            ysize    = 360
            xfirst   = -179.75
            xinc     = 0.5
            yfirst   = 89.75
            yinc     = -0.5
            """
        with open(self.half_degree_grid_desc,'w') as f:
            f.write(textwrap.dedent(half_degree_grid_desc_text))
        loaded_field = iodriver.advanced_field_loader(filename=join(data_dir,
                                                                    "HDdata",
                                                                    "catchmentmaps",
                                                                    "upscaled",
                                                                    "catchmentmap_unsorted_ICE5G_"
                                                                    "data_ALG4_sinkless_0k_"
                                                                    "20160714_121938.nc"),
                                                      grid_desc_file=self.half_degree_grid_desc,
                                                      adjust_orientation=True)
        self.assertEquals(loaded_field.data[62][157],14149.0,"Field data has not been loaded and"
                                                             " oriented properly")
        self.assertAlmostEqual(loaded_field.grid.lon_points[0],-179.75,msg="Field data grid is"
                                                                          " not correctly"
                                                                          " oriented")
        self.assertAlmostEqual(loaded_field.grid.lat_points[0],89.75,msg="Field data grid is"
                                                                         " not correctly"
                                                                         " oriented")

    def testLoadingFieldWithGridFileDescWithNonStandardOrientation(self):
        """Test loading with a non standard grid from a file using a grid description file"""
        ten_minute_grid_desc_text =\
            """
            gridtype = lonlat
            gridsize = 2332800
            xsize    = 2160
            ysize    = 1080
            xfirst   = 0.0
            xinc     = 0.1666666666666667
            yfirst   = 89.9166666666666667
            yinc     = -0.1666666666666667
            """
        with open(self.ten_minute_grid_desc,'w') as f:
            f.write(textwrap.dedent(ten_minute_grid_desc_text))
        loaded_field = iodriver.advanced_field_loader(filename=join(data_dir,
                                                                    "HDdata",
                                                                    "catchmentmaps",
                                                                    "catchmentmap_unsorted_ICE6g"
                                                                    "_lgm_ALG4_sinkless_no_true_"
                                                                    "sinks_oceans_lsmask_plus_"
                                                                    "upscale_rdirs_tarasov_orog"
                                                                    "_corrs_20171015_031541.nc"),
                                                      grid_desc_file=self.ten_minute_grid_desc,
                                                      adjust_orientation=True)
        self.assertEquals(loaded_field.data[45][713],399,"Field data has not been loaded and"
                                                             " oriented properly")
        self.assertAlmostEqual(loaded_field.grid.lon_points[0],-179.916666667,msg="Field data grid is"
                                                                                  " not correctly"
                                                                                  " oriented")
        self.assertAlmostEqual(loaded_field.grid.lat_points[0],89.9166666667,msg="Field data grid is"
                                                                                 " not correctly"
                                                                                 " oriented")

    def testLoadingFieldWithGridFileDescWithNonStandardOrientationWithoutReorientation(self):
        """Test loading with a non standard grid using a grid description file and no reorientation"""
        ten_minute_grid_desc_text =\
            """
            gridtype = lonlat
            gridsize = 2332800
            xsize    = 2160
            ysize    = 1080
            xfirst   = 0.0
            xinc     = 0.1666666666666667
            yfirst   = 89.9166666666666667
            yinc     = -0.1666666666666667
            """
        with open(self.ten_minute_grid_desc,'w') as f:
            f.write(textwrap.dedent(ten_minute_grid_desc_text))
        loaded_field = iodriver.advanced_field_loader(filename=join(data_dir,
                                                                    "HDdata",
                                                                    "catchmentmaps",
                                                                    "catchmentmap_unsorted_ICE6g"
                                                                    "_lgm_ALG4_sinkless_no_true_"
                                                                    "sinks_oceans_lsmask_plus_"
                                                                    "upscale_rdirs_tarasov_orog"
                                                                    "_corrs_20171015_031541.nc"),
                                                      grid_desc_file=self.ten_minute_grid_desc,
                                                      adjust_orientation=False)
        self.assertEquals(loaded_field.data[45][1793],399,"Field data has not been loaded and"
                                                             " oriented properly")
        self.assertAlmostEqual(loaded_field.grid.lon_points[0],0.0,msg="Field data grid is"
                                                                                  " not correctly"
                                                                                  " oriented")
        self.assertAlmostEqual(loaded_field.grid.lat_points[0],89.9166666667,msg="Field data grid is"
                                                                                 " not correctly"
                                                                                 " oriented")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()