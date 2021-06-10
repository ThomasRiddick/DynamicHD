'''
Unit tests for the iodriver module

Created on Dec 7, 2017

@author: thomasriddick
'''

import unittest
import numpy as np
import textwrap
import os
from os.path import join
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.base import field
from context import data_dir


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
        self.assertEqual(loaded_field.data[231][0],-1425.0,"Field data has not been loaded and"
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
        self.assertEqual(loaded_field.data[231][0],-2367.0,"Field data has not been loaded and"
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
        self.assertEqual(loaded_field.data[62][157],14149.0,"Field data has not been loaded and"
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
        self.assertEqual(loaded_field.data[62][517],14149.0,"Field data has not been loaded and"
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
        self.assertEqual(loaded_field.data[62][157],14149.0,"Field data has not been loaded and"
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
        self.assertEqual(loaded_field.data[62][157],14149.0,"Field data has not been loaded and"
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
        self.assertEqual(loaded_field.data[45][713],399,"Field data has not been loaded and"
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
        self.assertEqual(loaded_field.data[45][1793],399,"Field data has not been loaded and"
                                                             " oriented properly")
        self.assertAlmostEqual(loaded_field.grid.lon_points[0],0.0,msg="Field data grid is"
                                                                                  " not correctly"
                                                                                  " oriented")
        self.assertAlmostEqual(loaded_field.grid.lat_points[0],89.9166666667,msg="Field data grid is"
                                                                                 " not correctly"
                                                                                 " oriented")

    def testFieldWritingAndLoadingWithLatLongField(self):
        example_field = field.makeEmptyField('Generic',dtype=np.int64,grid_type='HD')
        lat_points=np.linspace(89.75,-89.75,360,endpoint=True)
        lon_points=np.linspace(-179.75,179.75,720,endpoint=True)
        example_field.set_grid_coordinates((lat_points,lon_points))
        example_field.data[20,20] = 1
        example_field.data[200,20] = 2
        example_field.data[20,200] = 3
        example_field.data[200,200] = 4
        iodriver.advanced_field_writer(os.path.join(self.directory,
                                               "advancedfieldwritingandloadingtest.nc"),
                                       example_field,fieldname='test_field',
                                       clobber=True)
        loaded_field = iodriver.advanced_field_loader(os.path.join(self.directory,"advancedfieldwritingandloadingtest.nc"),
                                                      fieldname='test_field')
        np.testing.assert_array_equal(example_field.get_data(),loaded_field.get_data())

    def testFieldWritingAndLoadingWithLatLongFieldWithFlip(self):
        example_field = field.makeEmptyField('Generic',dtype=np.int64,grid_type='HD')
        lat_points=np.linspace(89.75,-89.75,360,endpoint=True)
        lon_points=np.linspace(-179.75,179.75,720,endpoint=True)
        example_field.set_grid_coordinates((lat_points,lon_points))
        example_field.data[20,20] = 1
        example_field.data[200,20] = 2
        example_field.data[20,200] = 3
        example_field.data[200,200] = 4
        example_field.flip_data_ud()
        iodriver.advanced_field_writer(os.path.join(self.directory,
                                               "advancedfieldwritingandloadingtest.nc"),
                                       example_field,fieldname='test_field',
                                       clobber=True)
        example_field.flip_data_ud()
        loaded_field = iodriver.advanced_field_loader(os.path.join(self.directory,"advancedfieldwritingandloadingtest.nc"),
                                                      fieldname='test_field')
        np.testing.assert_array_equal(example_field.get_data(),loaded_field.get_data())

    def testFieldWritingAndLoadingWithLatLongFieldWithRotation(self):
        example_field = field.makeEmptyField('Generic',dtype=np.int64,grid_type='HD')
        lat_points=np.linspace(89.75,-89.75,360,endpoint=True)
        lon_points=np.linspace(-179.75,179.75,720,endpoint=True)
        example_field.set_grid_coordinates((lat_points,lon_points))
        example_field.data[20,20] = 1
        example_field.data[200,20] = 2
        example_field.data[20,200] = 3
        example_field.data[200,200] = 4
        example_field.rotate_field_by_a_hundred_and_eighty_degrees()
        iodriver.advanced_field_writer(os.path.join(self.directory,
                                               "advancedfieldwritingandloadingtest.nc"),
                                       example_field,fieldname='test_field',
                                       clobber=True)
        example_field.rotate_field_by_a_hundred_and_eighty_degrees()
        loaded_field = iodriver.advanced_field_loader(os.path.join(self.directory,"advancedfieldwritingandloadingtest.nc"),
                                                      fieldname='test_field')
        np.testing.assert_array_equal(example_field.get_data(),loaded_field.get_data())

    def testFieldWritingAndLoadingWithLatLongFieldWithRotationNoAdjustment(self):
        example_field = field.makeEmptyField('Generic',dtype=np.int64,grid_type='HD')
        lat_points=np.linspace(89.75,-89.75,360,endpoint=True)
        lon_points=np.linspace(-179.75,179.75,720,endpoint=True)
        example_field.set_grid_coordinates((lat_points,lon_points))
        example_field.data[20,20] = 1
        example_field.data[200,20] = 2
        example_field.data[20,200] = 3
        example_field.data[200,200] = 4
        example_field.rotate_field_by_a_hundred_and_eighty_degrees()
        iodriver.advanced_field_writer(os.path.join(self.directory,
                                               "advancedfieldwritingandloadingtest.nc"),
                                       example_field,fieldname='test_field',
                                       clobber=True)
        loaded_field = iodriver.advanced_field_loader(os.path.join(self.directory,"advancedfieldwritingandloadingtest.nc"),
                                                      fieldname='test_field',adjust_orientation=False)
        np.testing.assert_array_equal(example_field.get_data(),loaded_field.get_data())

    def testFieldWritingAndLoadingWithLatLongFieldWithFlip(self):
        example_field = field.makeEmptyField('Generic',dtype=np.int64,grid_type='HD')
        lat_points=np.linspace(89.75,-89.75,360,endpoint=True)
        lon_points=np.linspace(-179.75,179.75,720,endpoint=True)
        example_field.set_grid_coordinates((lat_points,lon_points))
        example_field.data[20,20] = 1
        example_field.data[200,20] = 2
        example_field.data[20,200] = 3
        example_field.data[200,200] = 4
        example_field.flip_data_ud()
        iodriver.advanced_field_writer(os.path.join(self.directory,
                                               "advancedfieldwritingandloadingtest.nc"),
                                       example_field,fieldname='test_field',
                                       clobber=True)
        loaded_field = iodriver.advanced_field_loader(os.path.join(self.directory,"advancedfieldwritingandloadingtest.nc"),
                                                      fieldname='test_field',adjust_orientation=False)
        np.testing.assert_array_equal(example_field.get_data(),loaded_field.get_data())


    def testFieldWritingAndLoadingWithLatLongFloatingPointField(self):
        example_field = field.makeEmptyField('Generic',dtype=np.float64,grid_type='HD')
        lat_points=np.linspace(89.75,-89.75,360,endpoint=True)
        lon_points=np.linspace(-179.75,179.75,720,endpoint=True)
        example_field.set_grid_coordinates((lat_points,lon_points))
        example_field.data[20,20] = 1.5
        example_field.data[200,20] = 2.5
        example_field.data[20,200] = 3.5
        example_field.data[200,200] = 4.5
        iodriver.advanced_field_writer(os.path.join(self.directory,
                                               "advancedfieldwritingandloadingtest.nc"),
                                       example_field,fieldname='test_field',
                                       clobber=True)
        loaded_field = iodriver.advanced_field_loader(os.path.join(self.directory,"advancedfieldwritingandloadingtest.nc"),
                                                      fieldname='test_field')
        np.testing.assert_array_equal(example_field.get_data(),loaded_field.get_data())
        self.assertEqual(loaded_field.data[20,20],1.5)
        self.assertEqual(loaded_field.data[200,20],2.5)
        self.assertEqual(loaded_field.data[20,200],3.5)
        self.assertEqual(loaded_field.data[200,200],4.5)

    def testFieldWritingAndLoadingWithLatLongNoClobber(self):
        example_field = field.makeEmptyField('Generic',dtype=np.float64,grid_type='HD')
        lat_points=np.linspace(89.75,-89.75,360,endpoint=True)
        lon_points=np.linspace(-179.75,179.75,720,endpoint=True)
        example_field.set_grid_coordinates((lat_points,lon_points))
        example_field.data[20,20] = 1.5
        example_field.data[200,20] = 2.5
        example_field.data[20,200] = 3.5
        example_field.data[200,200] = 4.5
        iodriver.advanced_field_writer(os.path.join(self.directory,
                                               "advancedfieldwritingandloadingtest.nc"),
                                       example_field,fieldname='test_field',
                                       clobber=True)
        with self.assertRaisesRegex(RuntimeError,r"Target file /Users/thomasriddick/Documents/data/temp/advancedfieldwritingandloadingtest.nc already exists and clobbering is not set"):
          iodriver.advanced_field_writer(os.path.join(self.directory,
                                         "advancedfieldwritingandloadingtest.nc"),
                                         example_field,fieldname='test_field',
                                         clobber=False)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
