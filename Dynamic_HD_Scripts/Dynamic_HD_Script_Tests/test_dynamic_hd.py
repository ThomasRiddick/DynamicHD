'''
Unit test for the main dynamic HD module and a number of modules that it depends on.

Created on Dec 15, 2015

@author: triddick
'''

import unittest
import numpy as np
from ../Dynamic_HD_Scripts import dynamic_hd
from ../Dynamic_HD_Scripts import iohelper
from ../Dynamic_HD_Scripts import grid as gd
import sys
import scipy.io as scipyio
import os
from ../Dynamic_HD_Scripts import field as fld
from ../Dynamic_HD_Scripts import f2py_manager
from .context import fortran_source_path,data_dir

class MainTestCase(unittest.TestCase):
    """Test top-level functions of the dynamic HD scripts with real and hypothetical data"""

    printout = False
    directory = None
    kwargs_input = {'lat':36,
                    'gc_absolute_tolerance':999,
                    'gc_method':'dummy',
                    'long':72,
                    'gc_relative_tolerance':998,
                    'gc_frac_value_zero_offset':-999}

    grid_kwargs_expected_output = {'lat':36,'long':72}

    grad_change_kwargs_expected_output = {'gc_absolute_tolerance':999,'gc_method':'dummy',
                                          'gc_relative_tolerance':998,'gc_frac_value_zero_offset':-999}
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

    def create_files(self,filesdict):
        """Creates files from a dictionary of file paths and fields"""
        for filename,field in list(filesdict.items()):
            try:
                os.remove(filename)
            except:
                pass
            with scipyio.FortranFile(self.directory+ '/'+filename+".datx",'w') as f: #@UndefinedVariable
                f.write_record(field)

    def testUsingSingleModernHydrology(self):
        """Test generating river flow directions from Stefan's manually corrected modern day 0.5 degree orography only"""
        dynamic_hd.main(new_orography_file=data_dir + '/HDdata/orographys/topo_hd_vs1_9_data_from_stefan.dat',
                        updated_RFD_file=self.directory+'/updated_river_directions_from_real_current_orog_single_orog_test.datx',
                        recompute_changed_orography_only=False,grid_type='HD')
        with scipyio.FortranFile(self.directory+'/updated_river_directions_from_real_current_orog_single_orog_test.datx','r') as f: #@UndefinedVariable
            new_RFD_field = f.read_record(dtype=np.int64).reshape((360,720))
        true_base_RFD_file = data_dir + '/HDdata/rdirs/rivdir_vs_1_9_data_from_stefan.txt'
        true_base_RFD_field = iohelper.TextFileIOHelper.load_field(true_base_RFD_file,'HD')
        true_base_RFD_field = np.ma.masked_less(true_base_RFD_field,1,copy=False)
        land_sea_mask = np.ma.getmaskarray(true_base_RFD_field)
        true_base_RFD_field = true_base_RFD_field.filled(0.0)
        new_RFD_field = np.ma.array(new_RFD_field,mask=land_sea_mask).filled(0)
        self.assertAlmostEqual(np.count_nonzero(true_base_RFD_field - new_RFD_field),0,msg=
                               "Generated RFD field using current orography doesn't match RFD field from Stefan's script to within expected tolerance",
                               delta=4171)

    def testUsingCurrentOrography(self):
        """Test generating river flow directions from a Stefan's manually corrected 0.5 orography compared to a blank orography"""
        if self.printout:
            print("Starting prepartion of real data test")
        #setup the base orography as being 10E7 so there are changes everywhere
        base_test_orography = np.zeros((360,720),dtype=np.float64)
        base_test_orography.fill(1.0e+7)
        if self.printout:
            print(base_test_orography)
        #setup the base river direction as being -999 to indicate missing data
        base_river_directions = np.zeros((360,720),dtype=np.int64)
        base_river_directions.fill(-999)
        if self.printout:
            print(base_river_directions)
        filesdict = {"base_blank_orography":base_test_orography,
                     "base_blank_river_directions":base_river_directions}
        self.create_files(filesdict)
        #run main program
        dynamic_hd.main(base_orography_file=self.directory+'/base_blank_orography.datx',
                        new_orography_file=data_dir + '/HDdata/orographys/topo_hd_vs1_9_data_from_stefan.txt',
                        corrected_base_orography_file=data_dir + '/HDdata/orographys/topo_hd_vs1_9_data_from_stefan.txt',
                        updated_orography_file=self.directory+'/update_orography_from_real_current_orog_test_against_blank.datx',
                        base_RFD_file=self.directory+'/base_blank_river_directions.datx',
                        updated_RFD_file=self.directory+'/updated_river_directions_from_real_current_orog_test_against_blank.datx',
                        grid_type='HD')
        with scipyio.FortranFile(self.directory+'/updated_river_directions_from_real_current_orog_test_against_blank.datx','r') as f: #@UndefinedVariable
            new_RFD_field = f.read_record(dtype=np.int64).reshape((360,720))
        true_base_RFD_file = data_dir + '/HDdata/rdirs/rivdir_vs_1_9_data_from_stefan.txt'
        true_base_RFD_field = iohelper.TextFileIOHelper.load_field(true_base_RFD_file,'HD')
        true_base_RFD_field = np.ma.masked_less(true_base_RFD_field,1,copy=False)
        land_sea_mask = np.ma.getmaskarray(true_base_RFD_field)
        true_base_RFD_field = true_base_RFD_field.filled(0.0)
        new_RFD_field = np.ma.array(new_RFD_field,mask=land_sea_mask).filled(0)

        diff = new_RFD_field - true_base_RFD_field
        with scipyio.FortranFile(self.directory+'/difference_between_my_new_RFD_and_Stefans.datx','w') as f: #@UndefinedVariable
            f.write_record(diff)
        self.assertAlmostEqual(np.count_nonzero(true_base_RFD_field - new_RFD_field),0,msg=
                               "Generated RFD field using current orography doesn't match RFD field from Stefan's script to within expected tolerance",
                               delta=4171)
        with scipyio.FortranFile(self.directory+'/update_orography_from_real_current_orog_test_against_blank.datx','r') as f: #@UndefinedVariable
            update_orog_field = f.read_record(dtype=np.float64).reshape((360,720))
        new_orog_field = iohelper.TextFileIOHelper.load_field(data_dir + '/HDdata/orographys/topo_hd_vs1_9_data_from_stefan.txt',
                                                              'HD')
        np.testing.assert_array_equal(update_orog_field,new_orog_field,
                                      "Generated orography field using current orography doesn't match expectation")

    def testRealPreModernVsModernData(self):
        """Test generating the river flow direction by comparing a pre modern orography with a modern orography"""
        #run main program
        dynamic_hd.main(base_orography_file=data_dir + '/HDdata/orographys/topo_hd_vs1_9_data_from_stefan.txt',
                        new_orography_file=data_dir + '/HDdata/orographys/topo_created_with_uwe_scripts_method_with_mods.nc',
                        base_RFD_file=data_dir + '/HDdata/rdirs/rivdir_vs_1_9_data_from_stefan.txt',
                        corrected_base_orography_file=data_dir + '/HDdata/orographys/topo_hd_vs1_9_data_from_stefan.txt',
                        updated_orography_file=self.directory+'/updated_orography_from_real_test.datx',
                        updated_RFD_file=self.directory+'/updated_river_directions_from_real_test.datx',
                        grid_type='HD')
        with scipyio.FortranFile(self.directory+'/updated_river_directions_from_real_test.datx','r') as f: #@UndefinedVariable
            output_rdir_field = f.read_record(dtype=np.float64).reshape((360,720))
        with scipyio.FortranFile(data_dir + '/HDdata/rdirs/updated_river_directions_from_real_test_kgo.datx','r') as f: #@UndefinedVariable
            expected_output_rdir_field = f.read_record(dtype=np.float64).reshape((360,720))
        with scipyio.FortranFile(self.directory+'/updated_orography_from_real_test.datx','r') as f: #@UndefinedVariable
            output_orog_field = f.read_record(dtype=np.float64).reshape((360,720))
        with scipyio.FortranFile(data_dir + '/HDdata/orographys/updated_orography_from_real_test_kgo.datx','r') as f: #@UndefinedVariable
            expected_output_orog_field = f.read_record(dtype=np.float64).reshape((360,720))
        np.testing.assert_array_equal(output_rdir_field,expected_output_rdir_field,
                                      'Test of modern vs premodern data does not produce expected river direction output')
        np.testing.assert_array_equal(output_orog_field,expected_output_orog_field,
                                      'Test of modern vs premodern data does not produce expected orography output')

    def ComplexHypotheticalCaseTestHelper(self):
        """Setup common elments of a hypothetical test case on the HD grid"""
        if self.printout:
            print("Starting preperation of hypothetical data test")
        #setup the base orography
        base_test_orography = np.zeros((360,720),dtype=np.float64)
        ygrad = 0.1
        for i in range(0,360):
            for j in range(0,400):
                base_test_orography[i,j] = j*0.1 - i*ygrad
            for j in range(400,720):
                base_test_orography[i,j] = 40.0 - (j-400)*0.1 - i*ygrad
        if self.printout:
            print(base_test_orography)
        #add in some changes
        new_test_orography = np.copy(base_test_orography)
        for i in range(0,360):
            for j in range(400,720):
                new_test_orography[i,j] = j*0.1 - i*ygrad
        if self.printout:
            print(new_test_orography)
        #add in a band of orography in the corrected base orography to see if it
        #correctly appears where river directions have no been updated and doesn't
        #where they have been updated
        corrected_base_test_orography = np.copy(base_test_orography)
        corrected_base_test_orography[75:100,] = 150
        #create old river directions
        base_river_directions = np.zeros((360,720),dtype=np.int64)
        base_river_directions[:180,] = 4
        base_river_directions[180:,] = 6
        if self.printout:
            print(base_river_directions)
        filesdict = {"base_test_orography":base_test_orography,
                      "new_test_orography":new_test_orography,
                      "base_river_directions":base_river_directions,
                      "corrected_base_orography":corrected_base_test_orography}
        self.create_files(filesdict)

    def testComplexHypotheticalCaseOrography(self):
        """Test the orography generated by a complex hypothetical test case"""
        self.ComplexHypotheticalCaseTestHelper()
        #run main program
        dynamic_hd.main(base_orography_file=self.directory+'/base_test_orography.datx',
                        new_orography_file=self.directory+'/new_test_orography.datx',
                        corrected_base_orography_file=self.directory+'/corrected_base_orography.datx',
                        updated_orography_file=self.directory+'/updated_orography.datx',
                        base_RFD_file=self.directory+'/base_river_directions.datx',
                        updated_RFD_file=self.directory+'/updated_river_directions.datx',
                        grid_type='HD')
        #check the results
        with scipyio.FortranFile(self.directory+'/updated_orography.datx','r') as f: #@UndefinedVariable
            updated_orography_field = f.read_record(dtype=np.float64).reshape((360,720))
        with scipyio.FortranFile(self.directory+'/new_test_orography.datx','r') as f: #@UndefinedVariable
            expected_hypothetical_orography_field = f.read_record(dtype=np.float64).reshape((360,720))
        expected_hypothetical_orography_field[75:100,:401] = 150
        np.testing.assert_array_equal(updated_orography_field, expected_hypothetical_orography_field,
                                      "Generated orography field for hypothetical test case does not give expected results")

    def testComplexHypotheticalCase(self):
        """Test the flow directions generated by a complex hypothetical test case"""
        self.ComplexHypotheticalCaseTestHelper()
        #run main program
        dynamic_hd.main(base_orography_file=self.directory+'/base_test_orography.datx',
                        new_orography_file=self.directory+'/new_test_orography.datx',
                        corrected_base_orography_file=self.directory+'/corrected_base_orography.datx',
                        updated_orography_file=self.directory+'/updated_orography.datx',
                        base_RFD_file=self.directory+'/base_river_directions.datx',
                        updated_RFD_file=self.directory+'/updated_river_directions.datx',
                        grid_type='HD')
        #check the results
        with scipyio.FortranFile(self.directory+'/updated_river_directions.datx','r') as f: #@UndefinedVariable
            updated_RFD_field = f.read_record(dtype=np.int64).reshape((360,720))
        expected_hypothetical_RFD_field = np.zeros((360,720),dtype=np.int64)
        expected_hypothetical_RFD_field[:180,1:400] = 4
        expected_hypothetical_RFD_field[180:,1:400] = 6
        expected_hypothetical_RFD_field[:,400:-1] = 1
        expected_hypothetical_RFD_field[:-1,0] = 2
        expected_hypothetical_RFD_field[-1,0] = 5
        expected_hypothetical_RFD_field[:,-1:] = 3
        expected_hypothetical_RFD_field[-1:,400:] = 4
        expected_hypothetical_RFD_field[359,719] = 6
        np.testing.assert_array_equal(updated_RFD_field,expected_hypothetical_RFD_field,
                                      'Generated RFD field for hypothetical orography does not give expected results')

    def test_filter_kwargs(self):
        """Test seperating out grid kwargs from grad change kwargs"""
        grid_kwargs,grad_change_kwargs = dynamic_hd.filter_kwargs(self.kwargs_input)
        self.assertDictEqual(grid_kwargs,self.grid_kwargs_expected_output)
        self.assertDictEqual(grad_change_kwargs,self.grad_change_kwargs_expected_output)

    def RecomputeOnlySignificantGradientChangesTestHelper(self):
        """Setup test of recomputing significant gradient changes only"""
        if self.printout:
            "Starting preparation for testing recomputing only significant gradient changes"
        #setup the base orography
        base_test_orography = np.zeros((360,720),dtype=np.float64)
        xgrad = 1.0
        ygrad = 1.0
        for i in range(0,360):
            for j in range(0,720):
                base_test_orography[i,j] = xgrad*i + ygrad*j
        if self.printout:
            print(base_test_orography)
        #add in some changes
        new_test_orography = np.copy(base_test_orography)
        for i in range(0,360):
            for j in range(360,540):
                new_test_orography[i,j] = 2*xgrad*i + 2*ygrad*j
            for j in range(540,720):
                new_test_orography[i,j] = 10*xgrad*i + 10*ygrad*j
        new_test_orography = new_test_orography + 57.0
        if self.printout:
            print(new_test_orography)
        #add in a band of orography in the corrected base orography to see if it
        #correctly appears where river directions have no been updated and doesn't
        #where they have been updated
        corrected_base_test_orography = np.copy(base_test_orography)
        corrected_base_test_orography[75:100,] = 150
        #create old river directions
        base_river_directions = np.zeros((360,720),dtype=np.int64)
        base_river_directions[:180,] = 4
        base_river_directions[180:,] = 6
        if self.printout:
            print(base_river_directions)
        filesdict = {"base_test_orography_gc":base_test_orography,
                      "new_test_orography_gc":new_test_orography,
                      "base_river_directions_gc":base_river_directions,
                      "corrected_base_orography_gc":corrected_base_test_orography}
        self.create_files(filesdict)

    def testRecomputeOnlySignificantGradientChanges(self):
        """Test recomputing significant gradient changes only"""
        self.RecomputeOnlySignificantGradientChangesTestHelper()
        dynamic_hd.main(base_orography_file=self.directory+'/base_test_orography_gc.datx',
                        new_orography_file=self.directory+'/new_test_orography_gc.datx',
                        corrected_base_orography_file=self.directory+'/corrected_base_orography_gc.datx',
                        updated_orography_file=self.directory+'/updated_orography_gc.datx',
                        base_RFD_file=self.directory+'/base_river_directions_gc.datx',
                        updated_RFD_file=self.directory+'/updated_river_directions_gc.datx',
                        update_mask_file=self.directory+'/update_mask_file_gc.datx',
                        grid_type='HD', recompute_significant_gradient_changes_only=True,
                        gc_method='all_neighbours',gc_absolute_tol=2.5)
        #check the results
        with scipyio.FortranFile(self.directory+'/updated_river_directions_gc.datx','r') as f: #@UndefinedVariable
            updated_RFD_field = f.read_record(dtype=np.int64).reshape((360,720))
        with scipyio.FortranFile(self.directory+'/update_mask_file_gc.datx','r') as f: #@UndefinedVariable
            update_mask = f.read_record(dtype=np.int64).reshape((360,720))
        #define expected RFD field
        expected_hypothetical_RFD_field = np.zeros((360,720),dtype=np.int64)
        expected_hypothetical_RFD_field[:180,1:359]   = 4
        expected_hypothetical_RFD_field[:180,361:539] = 4
        expected_hypothetical_RFD_field[180:,1:359]   = 6
        expected_hypothetical_RFD_field[180:,361:539] = 6
        expected_hypothetical_RFD_field[1:,539:-1]    = 7
        #discontinuity at column 359-360 border
        expected_hypothetical_RFD_field[1:,359:361]   = 7
        #sides
        expected_hypothetical_RFD_field[0,359:361]    = 4
        expected_hypothetical_RFD_field[1:,719]       = 9
        expected_hypothetical_RFD_field[0,539:-1]     = 4
        expected_hypothetical_RFD_field[1:,0]         = 8
        #corners
        expected_hypothetical_RFD_field[0,-1]         = 6
        expected_hypothetical_RFD_field[0,0]          = 5

        #define expected update mask
        expected_hypothetical_update_mask  = np.zeros((360,720),dtype=np.int64)
        expected_hypothetical_update_mask[:,1:359] = 1
        expected_hypothetical_update_mask[:,361:539] = 1

        np.testing.assert_array_equal(update_mask,expected_hypothetical_update_mask,
                                      'The update mask for hypothetical orography processed using by '
                                      'recomputing only significant gradient changes using the all neighbours '
                                      'method was not as expected')
        np.testing.assert_array_equal(updated_RFD_field,expected_hypothetical_RFD_field,
                                      'The generated RFD field for hypothetical orography processed using by '
                                      'recomputing only significant gradient changes using the all neighbours '
                                      'method was not as expected')

class IOHelperTestCase(unittest.TestCase):
    """Tests of the input and output module"""

    field = None
    directory = data_dir + '/temp/'
    fortran_test_file= directory + 'test_temp.fortran_file'
    fortran_output_test_file= directory + 'test_output_temp.fortran_file'
    fortran_output_f2py_test_file= directory + 'test_output_f2py_temp.fortran_file'
    netcdf_output_test_file = directory + 'test_output_netcdf4_temp.nc'
    text_output_test_file = directory + 'test_output_text_temp.txt'

    def setUp(self):
        """Unit test setup. Prepare some test data"""
        self.field = np.linspace(0,1000,259200)

    def tearDown(self):
        """Unit test tear down. Remove any file that have been created"""
        try:
            os.remove(self.fortran_test_file)
        except OSError:
            pass
        try:
            os.remove(self.fortran_output_test_file)
        except OSError:
            pass

    def prep_fortran_file(self):
        """Helper that creates a file of data"""
        #Put setup and cleanup here instead of in setUp and tearDown as they are only
        #needed for some cases and slow down execution
        with scipyio.FortranFile(self.fortran_test_file,'w') as f: #@UndefinedVariable
            f.write_record(self.field)

    def testSciPyFortranLoadField(self):
        """Test loading a Fortran field using a method that utilizes the FortranFile module from SciPy"""
        self.prep_fortran_file()
        np.testing.assert_array_equal(self.field.reshape(360,720),
                                      iohelper.SciPyFortranFileIOHelper.load_field(self.fortran_test_file,'HD'),
                                      "Data not being loaded correctly by FortranFileIOHelper")

    def testF2PyFortranLoadField(self):
        """Test loading a Fortran field using Fortran code wrapped by F2Py"""
        field_from_dat = iohelper.F2PyFortranFileIOHelper.load_field(
         data_dir + "/HDdata/orographys/topo_hd_vs1_9_data_from_stefan.dat", 'HD')
        field_from_txt = iohelper.TextFileIOHelper.load_field(
         data_dir + "/HDdata/orographys/topo_hd_vs1_9_data_from_stefan.txt", 'HD')
        np.testing.assert_allclose(field_from_dat,field_from_txt,rtol=0.5e-14,atol=0.5e-14,
                                   err_msg="Fortran field loaded from big endian unformatted fortran data doesn't match field loaded from text")

    def testGetFileHelperFunction(self):
        """Tests the function that selects the FileIOHelper subclass required depending on the supplied file extension"""
        self.assertIs(iohelper.SciPyFortranFileIOHelper,iohelper.getFileHelper('.datx'),
                      "IO Helper child class selection problem")

    def testLoadFieldFunction(self):
        """Test the high level field loading function"""
        self.prep_fortran_file()
        field_instance = fld.Orography(self.field.reshape(360,720))
        loaded_field_instance = dynamic_hd.load_field(self.fortran_test_file,'.datx', 'Orography', 'HD')
        np.testing.assert_array_equal(loaded_field_instance.data,field_instance.data,"load_field function is loading incorrect data")
        self.assertEqual(type(field_instance),type(loaded_field_instance),"load_field function is producing the wrong type of object")

    def testSciPyFortranWriteField(self):
        """Test writing a Fortran field using a method that utilizes the FortranFile module from SciPy"""
        iohelper.SciPyFortranFileIOHelper.write_field(self.fortran_output_test_file,fld.Field(self.field))
        with scipyio.FortranFile(self.fortran_output_test_file,'r') as f: #@UndefinedVariable
            np.testing.assert_array_equal(f.read_record(dtype=np.float64),self.field,"Data is being corrupted during write-read")

    def testF2PyFortranWriteField(self):
        """Test writing a Fortran field using Fortran code wrapped by F2Py"""
        iohelper.F2PyFortranFileIOHelper.write_field(self.fortran_output_f2py_test_file,fld.Field(self.field.reshape(360,720)))
        field = iohelper.F2PyFortranFileIOHelper.load_field(self.fortran_output_f2py_test_file,'HD')
        np.testing.assert_allclose(field,self.field.reshape(360,720),rtol=1.0e-7,atol=1.0e-7,err_msg="Data is being corrupted during F2Py read write")

    def testNetCDF4FieldWriteAndRead(self):
        """Test reading and writing code using a function that utilizes NetCDF4 module"""
        field_as_float64 = self.field.astype(np.float64)
        field_instance = fld.Orography(field_as_float64.reshape(360,720))
        iohelper.NetCDF4FileIOHelper.write_field(self.netcdf_output_test_file,field_instance)
        field_after_write_read = iohelper.NetCDF4FileIOHelper.load_field(self.netcdf_output_test_file,'HD')
        np.testing.assert_array_equal(field_after_write_read, field_instance.get_data(),
                                      "Data is being corruted during NetCDF4 write-read")

    def testNetCDF4FieldReadWithTimeVariables(self):
        """Test reading a file with multiple times using a function that utilizes NetCDF4 module"""
        field_0 = iohelper.NetCDF4FileIOHelper.load_field(filename=data_dir + "/HDdata/orographys/topo-final-OR-from-virna.nc",
                                                          grid_type='LatLong5min',
                                                          unmask=True, timeslice=0)
        field_1 = iohelper.NetCDF4FileIOHelper.load_field(filename=data_dir + "/HDdata/orographys/topo-final-OR-from-virna.nc",
                                                          grid_type='LatLong5min',
                                                          unmask=True, timeslice=1)
        field_57 = iohelper.NetCDF4FileIOHelper.load_field(filename=data_dir + "/HDdata/orographys/topo-final-OR-from-virna.nc",
                                                          grid_type='LatLong5min',
                                                          unmask=True, timeslice=57)
        field_181 = iohelper.NetCDF4FileIOHelper.load_field(filename=data_dir + "/HDdata/orographys/topo-final-OR-from-virna.nc",
                                                            grid_type='LatLong5min',
                                                            unmask=True, timeslice=181)
        field_260 = iohelper.NetCDF4FileIOHelper.load_field(filename=data_dir + "/HDdata/orographys/topo-final-OR-from-virna.nc",
                                                            grid_type='LatLong5min',
                                                            unmask=True, timeslice=260)
        np.testing.assert_array_almost_equal(field_0[1577:1579,829:831], np.array([[562.4,577.4],[577.200046,600.200046]]),
                                             decimal= 6,err_msg="Loading a field from NetCDF file containing multiple time"
                                             " slices doesn't produce expected result")
        np.testing.assert_array_almost_equal(field_1[1577:1579,829:831], np.array([[562.8,577.8],[577.700023,600.700023]]),
                                             decimal= 6,err_msg="Loading a field from NetCDF file containing multiple time"
                                             " slices doesn't produce expected result")
        np.testing.assert_array_almost_equal(field_57[1577:1579,829:831], np.array([[559.1,574.1],[573.900046,596.900046]]),
                                             decimal= 6,err_msg="Loading a field from NetCDF file containing multiple time"
                                             " slices doesn't produce expected result")
        np.testing.assert_array_almost_equal(field_181[2158:2160,4318:4320], np.array([[-4256.5,-4256.5],[-4258.5,-4258.5]]),
                                             decimal= 6,err_msg="Loading a field from NetCDF file containing multiple time"
                                             " slices doesn't produce expected result")
        np.testing.assert_array_almost_equal(field_260[2158:2160,4318:4320], np.array([[-4288.0,-4288.0],[-4290.0,-4290.0]]),
                                             decimal= 6,err_msg="Loading a field from NetCDF file containing multiple time"
                                             " slices doesn't produce expected result")


    def testTextReadAndWrite(self):
        """Test writing to then reading from a text file"""
        iohelper.TextFileIOHelper.write_field(filename=self.text_output_test_file,
                                              field=fld.Field(self.field.reshape(360,720)),
                                              griddescfile=None)
        field_after_writing_and_reading_from_a_file = \
            iohelper.TextFileIOHelper.load_field(filename=self.text_output_test_file,
                                                 grid_type='HD')
        np.testing.assert_array_equal(self.field.reshape(360,720),field_after_writing_and_reading_from_a_file,
                                      "Writing to a text file and reading back from it doesn't produce expected"
                                      " result.")

class fieldOperationTestCase(unittest.TestCase):
    """Tests the various operations on field objects"""

    old_field =  np.array([[7,7,7,7,8],
                           [7,7,7,7,8],
                           [9,8,7,7,1],
                           [3,2,1,3,2],
                           [6,5,4,6,5]])

    new_field =  np.array([[9,3,4,1,6],
                           [7,4,5,3,7],
                           [9,8,3,7,2],
                           [3,2,1,3,2],
                           [6,1,2,6,5]])

    new_field_mask = np.array([[True ,False,True ,False,False],
                               [True ,False,False,True ,False],
                               [True ,False,True ,False,False],
                               [True ,True ,True ,False,True ],
                               [False,False,True ,False,False]])

    old_field_updated_by_partially_masked_data = np.array([[7,3,7,1,6],
                                                           [7,4,5,7,7],
                                                           [9,8,7,7,2],
                                                           [3,2,1,3,2],
                                                           [6,1,4,6,5]])

    empty_field = np.zeros((10,10),dtype=np.bool)

    flag_points_input = [(0,0),(2,0),(0,7),(2,2),(9,4),(5,9),(4,6),(9,9)]

    point_flagging_expected_output = np.array([[True, False,False,False,False,False,False,True, False,False],
                                               [False,False,False,False,False,False,False,False,False,False],
                                               [True, False,True, False,False,False,False,False,False,False],
                                               [False,False,False,False,False,False,False,False,False,False],
                                               [False,False,False,False,False,False,True ,False,False,False],
                                               [False,False,False,False,False,False,False,False,False,True ],
                                               [False,False,False,False,False,False,False,False,False,False],
                                               [False,False,False,False,False,False,False,False,False,False],
                                               [False,False,False,False,False,False,False,False,False,False],
                                               [False,False,False,False,True, False,False,False,False,True ]])

    def setUp(self):
        """Unit test setup method. Prepare some field objects"""
        self.new_field_masked = np.ma.array(self.new_field,mask=self.new_field_mask)
        self.field_instance     = fld.Field(self.old_field)
        self.new_field_instance = fld.Field(self.new_field)
        self.new_field_instance_masked = fld.Field(self.new_field_masked)

    def testFieldUpdateWithUnmaskedField(self):
        """Test updating a field with unmasked data"""
        self.field_instance.update_field_with_partially_masked_data(self.new_field_instance)
        np.testing.assert_array_equal(self.field_instance.get_data(),
                                      self.new_field)

    def testFieldupdateWithPartiallyMaskedField(self):
        """Test updating a field with partially masked data"""
        self.field_instance.update_field_with_partially_masked_data(self.new_field_instance_masked)
        np.testing.assert_array_equal(self.field_instance.get_data(),
                                      self.old_field_updated_by_partially_masked_data)

    def testMakeField(self):
        """Test creating a new field"""
        created_field_instance = fld.makeField(self.old_field,'Generic','HD')
        np.testing.assert_array_equal(self.field_instance.data,created_field_instance.data,
                                      "makeField is not producing a field correctly")
        self.assertEqual(type(self.field_instance),type(created_field_instance),
                         "makeField is producing wrong type of object")

    def testMarkingPointsUsingList(self):
        """Test marking points as true using a list of co-ordinates"""
        boolfld = fld.Field(self.empty_field,grid='HD')
        boolfld.flag_listed_points(self.flag_points_input)
        np.testing.assert_array_equal(boolfld.get_data(),
                                      self.point_flagging_expected_output,
                                      "Marking points from a list is not producing"
                                      " expected results")

class RiverDirectionsOperationsTestCase(unittest.TestCase):
    """Contains various test of the river direction class and of the cumulative flow class"""

    river_mouth_rdirs_input_array = np.array([[5.0,3.0,-1.0, -1.0,-1.0,-1.0, 8.0,-1.0,-1.0],
                                              [-1.0,-1.0,-1.0, -1.0,2.0,-1.0, -1.0,-1.0,5.0],
                                              [-1.0,-1.0,-1.0, 6.0,5.0,4.0, -1.0,-1.0,5.0],
                                              [-1.0,-1.0,-1.0, -1.0,8.0,-1.0, -1.0,-1.0,3.0],
                                              [-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, 9.0,-1.0,6.0],
                                              [1.0,-1.0,8.0, -1.0,2.0,1.0, 7.0,-1.0,9.0],
                                              [4.0,-1.0,8.0, -1.0,-1.0,-1.0, 4.0,-1.0,-1.0],
                                              [7.0,-1.0,8.0, -1.0,-1.0,-1.0, 6.0,-1.0,-1.0],
                                              [4.0,2.0,4.0,5.0,6.0,6.0,  6.0,7.0,6.0]])

    river_mouth_rdirs_input_array_no_sea_points = np.array([[5.0,3.0,5.0, 5.0,5.0,5.0, 8.0,5.0,5.0],
                                                            [5.0,5.0,5.0, 5.0,2.0,5.0, 5.0,5.0,5.0],
                                                            [5.0,5.0,5.0, 6.0,5.0,4.0, 5.0,5.0,5.0],
                                                            [5.0,5.0,5.0, 5.0,8.0,5.0, 5.0,5.0,3.0],
                                                            [5.0,5.0,5.0, 5.0,5.0,5.0, 9.0,5.0,6.0],
                                                            [1.0,5.0,8.0, 5.0,2.0,1.0, 7.0,5.0,9.0],
                                                            [4.0,5.0,8.0, 5.0,5.0,5.0, 4.0,1.0,9.0],
                                                            [7.0,2.0,8.0, 3.0,4.0,6.0, 6.0,7.0,8.0],
                                                            [4.0,2.0,4.0,5.0,6.0,6.0,  6.0,7.0,6.0]])

    ls_mask = np.array([[0.0,0.0,1.0, 1.0,1.0,1.0, 0.0,1.0,1.0],
                        [1.0,1.0,1.0, 1.0,0.0,1.0, 1.0,1.0,0.0],
                        [1.0,1.0,1.0, 0.0,0.0,0.0, 1.0,1.0,0.0],
                        [1.0,1.0,1.0, 1.0,0.0,1.0, 1.0,1.0,0.0],
                        [1.0,1.0,1.0, 1.0,1.0,1.0, 0.0,1.0,0.0],
                        [0.0,1.0,0.0, 1.0,0.0,0.0, 0.0,1.0,0.0],
                        [0.0,1.0,0.0, 1.0,1.0,1.0, 0.0,1.0,1.0],
                        [0.0,1.0,0.0, 1.0,1.0,1.0, 0.0,1.0,1.0],
                        [0.0,0.0,0.0,0.0,0.0,0.0,  0.0,0.0,0.0]])

    cumulative_flow_input_array = np.array([[1,2,3, 4,5,6, 7,8,9],
                                            [11,12,13, 14,15,16, 17,18,19],
                                            [21,22,23, 24,25,26, 27,28,29],
                                            [31,32,33, 34,35,36, 37,38,39],
                                            [41,42,43, 44,45,46, 47,48,49],
                                            [51,52,53, 54,55,56, 57,58,59],
                                            [61,62,63, 64,65,66, 67,68,69],
                                            [71,72,73, 74,75,76, 77,78,79],
                                            [81,82,83, 84,85,86, 87,88,89]])

    true_sinks_finding_test_input_array = np.array([[3.0,3.0,4.0, 7.0,5.0,5.0, 8.0,7.0,7.0],
                                                    [3.0,5.0,4.0, 5.0,2.0,5.0, 5.0,8.0,8.0],
                                                    [3.0,6.0,1.0, 6.0,5.0,4.0, 8.0,5.0,5.0],
                                                    [7.0,8.0,2.0, 5.0,8.0,7.0, 8.0,5.0,3.0],
                                                    [8.0,8.0,5.0, 7.0,5.0,5.0, 9.0,5.0,6.0],
                                                    [1.0,9.0,8.0, 6.0,2.0,1.0, 7.0,5.0,9.0],
                                                    [4.0,2.0,8.0, 8.0,5.0,5.0, 4.0,1.0,9.0],
                                                    [7.0,2.0,8.0, 3.0,4.0,6.0, 6.0,7.0,8.0],
                                                    [4.0,2.0,4.0,5.0,6.0,6.0,  6.0,7.0,6.0]])

    true_sinks_finding_test_expected_ouput_array = np.array([[False,False,False, False,True,True, False,False,False],
                                                             [False,True,False, True,False,True, True,False,False],
                                                             [False,False,False, False,True,False, False,True,True],
                                                             [False,False,False, True,False,False, False,True,False],
                                                             [False,False,True, False,True,True, False,True,False],
                                                             [False,False,False, False,False,False, False,True,False],
                                                             [False,False,False, False,True,True, False,False,False],
                                                             [False,False,False, False,False,False, False,False,False],
                                                             [False,False,False,True,False,False,  False,False,False]])

    cumulative_flow_at_river_mouth_expected_output = \
        np.array([[0,0,0, 0,0,0, 0,0,0],
                  [0,0,13, 0,0,0, 0,0,0],
                  [0,0,0, 0,0,0, 0,0,0],
                  [0,0,0, 0,0,0, 0,38,0],
                  [41,0,43, 0,0,46, 0,0,0],
                  [0,0,0, 0,0,0, 0,0,0],
                  [0,0,0, 0,65,66, 0,0,69],
                  [0,0,0, 0,0,0, 0,78,0],
                  [0,0,0, 0,0,0, 0,0,0]])


    river_mouth_rdirs_expected_output_array = np.array([[5.0,3.0,-1.0, -1.0,-1.0,-1.0, 8.0,-1.0,-1.0],
                                                        [-1.0,-1.0,0.0, -1.0,2.0,-1.0, -1.0,-1.0,5.0],
                                                        [-1.0,-1.0,-1.0, 6.0,5.0,4.0, -1.0,-1.0,5.0],
                                                        [-1.0,-1.0,-1.0, -1.0,8.0,-1.0, -1.0,0.0,3.0],
                                                        [0.0,-1.0,0.0, -1.0,-1.0,0.0, 9.0,-1.0,6.0],
                                                        [1.0,-1.0,8.0, -1.0,2.0,1.0, 7.0,-1.0,9.0],
                                                        [4.0,-1.0,8.0, -1.0,0.0,0.0, 4.0,-1.0,0.0],
                                                        [7.0,-1.0,8.0, -1.0,-1.0,-1.0, 6.0,0.0,-1.0],
                                                        [4.0,2.0,4.0,5.0,6.0,6.0,  6.0,7.0,6.0]])

    expected_river_mouth_points_output_array = np.array([[False,False,False, False,False,False, False,False,False],
                                                         [False,False,True, False,False,False, False,False,False],
                                                         [False,False,False, False,False,False, False,False,False],
                                                         [False,False,False, False,False,False, False,True,False],
                                                         [True,False,True, False,False,True, False,False,False],
                                                         [False,False,False, False,False,False, False,False,False],
                                                         [False,False,False, False,True,True, False,False,True],
                                                         [False,False,False, False,False,False, False,True,False],
                                                         [False,False,False,False,False,False,  False,False,False]])

    ls_mask_extraction_input_array = np.array([[5.0,3.0,-1.0, -1.0,-1.0,-1.0, 8.0,-1.0,-1.0],
                                               [-1.0,-1.0,0.0, -1.0,2.0,-1.0, -1.0,-1.0,5.0],
                                               [-1.0,-1.0,-1.0, 6.0,5.0,4.0, -1.0,-1.0,5.0],
                                               [-1.0,-1.0,-1.0, -1.0,8.0,-1.0, -1.0,0.0,3.0],
                                               [0.0,-1.0,0.0, -1.0,-1.0,0.0, 9.0,-1.0,6.0],
                                               [1.0,-1.0,8.0, -1.0,2.0,1.0, 7.0,-1.0,9.0],
                                               [4.0,-1.0,8.0, -1.0,0.0,0.0, 4.0,-1.0,0.0],
                                               [7.0,-1.0,8.0, -1.0,-1.0,-1.0, 6.0,0.0,-1.0],
                                               [4.0,2.0,4.0,5.0,6.0,6.0,  6.0,7.0,6.0]])

    ls_mask_expected_results = np.array([[0,0,1, 1,1,1, 0,1,1],
                                         [1,1,1, 1,0,1, 1,1,0],
                                         [1,1,1, 0,0,0, 1,1,0],
                                         [1,1,1, 1,0,1, 1,1,0],
                                         [1,1,1, 1,1,1, 0,1,0],
                                         [0,1,0, 1,0,0, 0,1,0],
                                         [0,1,0, 1,1,1, 0,1,1],
                                         [0,1,0, 1,1,1, 0,1,1],
                                         [0,0,0,0,0,0,  0,0,0]])

    def testMarkingRiverMouthsWithoutLSMask(self):
        """Test the river mouth marking routine without supplying a land sea mask"""
        rdirs_field = fld.makeField(self.river_mouth_rdirs_input_array,
                                    'RiverDirections','HD')
        rdirs_field.mark_river_mouths()
        np.testing.assert_array_equal(rdirs_field.get_data(),
                                      self.river_mouth_rdirs_expected_output_array,
                                      "River mouth marking does not produce expected results")

    def testMarkingRiverMouthsWithoutLSMaskNoSeaWarning(self):
        """Test the river mouth marking routine correctly warns when no river mouths are present"""
        rdirs_field = fld.makeField(self.river_mouth_rdirs_input_array_no_sea_points,
                                    'RiverDirections','HD')
        self.assertRaises(UserWarning,rdirs_field.mark_river_mouths)

    def testMarkingRiverMouthsWithLSMask(self):
        """Test the river mouth marking routine with a land sea mask"""
        rdirs_field = fld.makeField(self.river_mouth_rdirs_input_array_no_sea_points,
                                    'RiverDirections','HD')
        rdirs_field.mark_river_mouths(self.ls_mask)
        np.testing.assert_array_equal(rdirs_field.get_data(),
                                      self.river_mouth_rdirs_expected_output_array,
                                      "River mouth marking with land sea mask does not produce"
                                      " expected results")

    def testGettingRiverMouths(self):
        """Test getting the river mouths from a river direction field"""
        rdirs_field = fld.makeField(self.river_mouth_rdirs_input_array,
                                    'RiverDirections','HD')
        rdirs_field.mark_river_mouths()
        np.testing.assert_array_equal(rdirs_field.get_river_mouths(),
                                      self.expected_river_mouth_points_output_array,
                                      "Not finding existing river mouth points correctly")

    def testFindingCumulativeFlowAtRiverMouths(self):
        """Test getting the cumulative flow at river mouths from a cumulative flow field"""
        rdirs_field = fld.makeField(self.river_mouth_rdirs_input_array,
                                    'RiverDirections','HD')
        rdirs_field.mark_river_mouths()
        flow_to_cell = fld.makeField(self.cumulative_flow_input_array,
                                           'CumulativeFlow','HD')
        np.testing.assert_array_equal(flow_to_cell.\
                                      find_cumulative_flow_at_outlets(rdirs_field.\
                                                                      get_river_mouths()),
                                      self.cumulative_flow_at_river_mouth_expected_output,
                                      "Finding cumulative flow at outlets does not produce"
                                      " expected results")

    def testGettingLSMask(self):
        """Test creating a land-sea mask from a set of flow directions"""
        rdirs_field = fld.makeField(self.ls_mask_extraction_input_array,
                                    'RiverDirections', 'HD')
        np.testing.assert_array_equal(rdirs_field.get_lsmask(),self.ls_mask_expected_results)

    def testExtractingTrueSinks(self):
        """Test extracting the true sinks from an rdir field"""
        rdirs_field = fld.makeField(self.true_sinks_finding_test_input_array,
                                    'RiverDirections', 'HD')
        np.testing.assert_array_equal(rdirs_field.extract_truesinks(),
                                      self.true_sinks_finding_test_expected_ouput_array,
                                      "True sinks extraction function not producing expected"
                                      " results.")


class OrographyOperationsTestCase(unittest.TestCase):
    """Tests for various calculations relating to significant gradients"""

    fill_value = 999.0

    gradients_input_array         = np.array([[1.5, 2.5,3.5,5.5],
                                              [4.5,-0.5,6.5,8.0],
                                              [2.5, 9.0,0.5,7.5]])
    old_gradients_input_array     = np.array([[105.5,102.5,103.5,105.5],
                                              [104.5,99.5 ,106.5,108.0],
                                              [102.5,109.0,103.5,107.5]])
    calc_sig_grad_changes_expected_mask = np.array([[[False,False,True,False],
                                                     [False,False,True,False],
                                                     [True,True,True,True]],
                                                    [[False,False,True,False],
                                                     [False,False,True,False],
                                                     [True,True,True,True]],
                                                    [[False,False,True,False],
                                                     [False,False,True,False],
                                                     [True,True,True,True]],
                                                    [[False,False,True,False],
                                                     [False,False,True,False],
                                                     [True,True,True,True]],
                                                    [[False,False,True,False],
                                                     [False,False,True,False],
                                                     [True,True,True,True]],
                                                    [[False,False,True,False],
                                                     [False,False,True,False],
                                                     [True,True,True,True]],
                                                    [[False,False,True,False],
                                                     [False,False,True,False],
                                                     [True,True,True,True]],
                                                    [[False,False,True,False],
                                                     [False,False,True,False],
                                                     [True,True,True,True]],
                                                    [[False,False,True,False],
                                                     [False,False,True,False],
                                                     [True,True,True,True]]])

    calc_sig_grad_changes_expected_results = np.array([[[4.0,0.0,fill_value,0.0],
                                                        [0.0,0.0,fill_value,-3.0],
                                                        [fill_value,fill_value,fill_value,fill_value]],
                                                       [[4.0,0.0,fill_value,0.0],
                                                        [0.0,0.0,fill_value,0.0],
                                                        [fill_value,fill_value,fill_value,fill_value]],
                                                       [[4.0,0.0,fill_value,0.0],
                                                        [0.0,-3.0,fill_value,0.0],
                                                        [fill_value,fill_value,fill_value,fill_value]],
                                                       [[4.0,-4.0,fill_value,0.0],
                                                        [0.0,0.0,fill_value,0.0],
                                                        [fill_value,fill_value,fill_value,fill_value]],
                                                       [[0.0,0.0,fill_value,0.0],
                                                        [0.0,0.0,fill_value,0.0],
                                                        [fill_value,fill_value,fill_value,fill_value]],
                                                       [[4.0,0.0,fill_value,-4.0],
                                                        [0.0,0.0,fill_value,0.0],
                                                        [fill_value,fill_value,fill_value,fill_value]],
                                                       [[0.0,0.0,fill_value,0.0],
                                                        [0.0,-4.0,fill_value,0.0],
                                                        [fill_value,fill_value,fill_value,fill_value]],
                                                       [[0.0,0.0,fill_value,0.0],
                                                        [-4.0,0.0,fill_value,0.0],
                                                        [fill_value,fill_value,fill_value,fill_value]],
                                                       [[0.0,0.0,fill_value,0.0],
                                                        [0.0,0.0,fill_value,-4.0],
                                                        [fill_value,fill_value,fill_value,fill_value]]])

    calc_diff_from_base_orog_with_use_grads_option_expected_mask =  np.array([[False,False,True,False],
                                                                              [False,False,True,False],
                                                                              [True,True,True,True]])

    calc_diff_from_base_orog_with_use_grads_option_expected_results = np.array([[1.5,2.5,fill_value,5.5],
                                                                                [4.5,-0.5,fill_value,8.0],
                                                                                [fill_value,fill_value,fill_value,fill_value]])

    def setUp(self):
        """Unit test setup function. Prepare some arrays and fields"""
        gradients_input_array = np.copy(self.gradients_input_array)
        old_gradients_input_array = np.copy(self.old_gradients_input_array)
        self.new_orography_field = fld.makeField(gradients_input_array,
                                                 "Orography", 'HD')
        self.old_orography_field = fld.makeField(old_gradients_input_array,
                                                 "Orography", 'HD')

    def testCalculateSignificantGradientChanges(self):
        """Test calculating signficiant gradient changes"""
        significant_grad_changes = self.new_orography_field.\
                                    calculate_significant_gradient_changes(self.old_orography_field,
                                                                              gc_absolute_tol=3.0)
        np.testing.assert_array_equal(np.ma.getmaskarray(significant_grad_changes),
                                      self.calc_sig_grad_changes_expected_mask,
                                      "Mask calculated for calculate significant gradients changes doesn't match expected result")
        np.testing.assert_array_equal(significant_grad_changes.filled(self.fill_value),
                                      self.calc_sig_grad_changes_expected_results,
                                      "Filled in version of masked results for significant gradients changes calculation doesn't match expected result")

    def testMaskNewOrographyUsingBaseOrographyWithUseGradientsOption(self):
        """Test masking new orography where the gradient has change signficantly from the base orography"""
        self.new_orography_field.mask_new_orography_using_base_orography(self.old_orography_field,
                                                                         use_gradient=True,
                                                                         gc_absolute_tol=3.0)
        np.testing.assert_array_equal(self.new_orography_field.get_mask(),
                                      self.calc_diff_from_base_orog_with_use_grads_option_expected_mask,
                                      "Masking new orography with base orography using use gradients option doesn't give expected mask.")
        np.testing.assert_array_equal(self.new_orography_field.get_data().filled(self.fill_value),
                                      self.calc_diff_from_base_orog_with_use_grads_option_expected_results,
                                      "Masking new orography with base orography using use gradients option doesn't give expected results.")


class SquareNineByNineTestCase(unittest.TestCase):
    """Various tests on a 9 by 9 cell grid"""

    fill_value = -999.99
    orog_array_base = np.array([[1.1,1.2,1.3],
                                [2.3,2.2,2.3],
                                [3.3,3.2,3.3]])
    orog_array_new  = np.array([[1.4,1.1,1.3],
                                [2.3,2.2,2.3],
                                [3.3,3.2,3.3]])
    orog_array_differences = orog_array_new - orog_array_base
    orog_array_masked_differences = np.array([[0.3       ,-0.1      ,fill_value],
                                              [fill_value,fill_value,fill_value],
                                              [fill_value,fill_value,fill_value]])
    changed_orog_array_masked_extended = np.array([[1.4       ,1.1       ,1.3       ],
                                                   [2.3       ,2.2       ,2.3       ],
                                                   [fill_value,fill_value,fill_value]])

    def setUp(self):
        """Unit test setup function. Prepare some fields"""
        self.orog_instance_base = fld.Orography(self.orog_array_base)
        self.orog_instance_new = fld.Orography(self.orog_array_new)

    def testOrographyInitializationAndReturn(self):
        """Test that an orography field object is being intialized correctly"""
        self.assertTrue(np.array_equal(self.orog_instance_base.get_data(),self.orog_array_base),
                        "Orography object is not recieving and returning data correctly")

    def testCalculateDifferencesFromBaseOrographyMaskedOutputCorrect(self):
        """Test that calculation of the masked differences from a base orography is working correctly"""
        self.assertTrue(np.allclose(np.ma.filled(self.orog_instance_new.\
                                                    calculate_differences_from_base_orography(self.orog_instance_base),
                                                    self.fill_value),
                                       self.orog_array_masked_differences,sys.float_info.epsilon,sys.float_info.epsilon),
                        "Masked differences are incorrect")

    def testCalculateDifferencesFromBaseOrographyUnmaskedOutputCorrect(self):
        """Test that calculation of the unmasked differences from a base orography is working correctly"""
        self.assertTrue(np.array_equal(np.ma.getdata(self.orog_instance_new.\
                                                     calculate_differences_from_base_orography(self.orog_instance_base)),
                                       self.orog_array_differences),
                        "Unmasked differences are incorrect")

    def testMaskExtension(self):
        """Test that the one cell extension of the mask is working correctly"""
        self.orog_instance_new.mask_new_orography_using_base_orography(self.orog_instance_base)
        self.orog_instance_new.extend_mask()
        self.assertTrue(np.allclose(np.ma.filled(self.orog_instance_new.get_data(),self.fill_value),
                                    self.changed_orog_array_masked_extended,sys.float_info.epsilon,sys.float_info.epsilon),
                        "Extended masked differences are incorrect")

class HDGridFunctionsTestCase(unittest.TestCase):
    """Tests of the HD grid functions

    As the definition of the HD grid size is not enforce by these function it is possible
    to test these functions using much smaller fields of test data
    """

    old_mask = np.array([[True ,True ,False,True ,True ,True ,True ,True ,True ,True ],
                         [True ,True ,True ,True ,True ,True ,True ,True ,True ,False],
                         [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                         [False,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                         [True ,True ,True ,True ,True ,True ,True ,False,True ,True ],
                         [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                         [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                         [True ,True ,False,True ,True ,True ,True ,True ,True ,True ],
                         [True ,True ,True ,True ,True ,True ,True ,True ,True ,True ],
                         [True ,True ,True ,False,True ,True ,True ,True ,True ,False]])

    new_mask = np.array([[False,False,False,False,True ,True ,True ,True ,False,False],
                         [False,False,False,False,True ,True ,True ,True ,False,False],
                         [False,False,True ,True ,True ,True ,True ,True ,False,False],
                         [False,False,True ,True ,True ,True ,False,False,False,False],
                         [False,False,True ,True ,True ,True ,False,False,False,False],
                         [True ,True ,True ,True ,True ,True ,False,False,False,True ],
                         [True ,False,False,False,True ,True ,True ,True ,True ,True ],
                         [True ,False,False,False,True ,True ,True ,True ,True ,True ],
                         [False,False,False,False,False,True ,True ,True ,False,False],
                         [False,True ,False,False,False,True ,True ,True ,False,False]])
    kernel_input_array = [1.1,1.3,0.7,
                          6.9,1.7,1.8,
                          2.5,0.7,2.2]
    kernel_input_array_with_central_min = [0.7,1.3,0.7,
                                           6.9,0.7,1.8,
                                           0.7,0.7,0.7]

    kernel_flow_dir_input_array_not_sea = [3.0,2.0,1.0,
                                           6.0,5.0,4.0,
                                           9.0,2.0,7.0]

    kernel_flow_dir_input_array_sea_no_river_inflow= [6.0,7.0,4.0,
                                                      9.0,-1.0,3.0,
                                                      8.0,1.0,2.0]

    kernel_flow_dir_input_array_sea_no_river_inflow_with_sinks= [5.0,5.0,5.0,
                                                                 5.0,-1.0,5.0,
                                                                 5.0,5.0,5.0]

    river_mouth_rdirs_input_array = np.array([[5.0,3.0,-1.0, -1.0,-1.0,-1.0, 8.0,-1.0,-1.0],
                                              [-1.0,-1.0,-1.0, -1.0,2.0,-1.0, -1.0,-1.0,5.0],
                                              [-1.0,-1.0,-1.0, 6.0,5.0,4.0, -1.0,-1.0,5.0],
                                              [-1.0,-1.0,-1.0, -1.0,8.0,-1.0, -1.0,-1.0,3.0],
                                              [-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, 9.0,-1.0,6.0],
                                              [1.0,-1.0,8.0, -1.0,2.0,1.0, 7.0,-1.0,9.0],
                                              [4.0,-1.0,8.0, -1.0,-1.0,-1.0, 4.0,-1.0,-1.0],
                                              [7.0,-1.0,8.0, -1.0,-1.0,-1.0, 6.0,-1.0,-1.0],
                                              [4.0,2.0,4.0,5.0,6.0,6.0,  6.0,7.0,6.0]])

    flow_directions_input_array  =  np.array([[1.5,1.6,1.8,2.9,2.6],
                                              [2.6,1.8,1.9,3.5,7.7],
                                              [2.9,1.9,2.1,2.8,8.9],
                                              [3.6,2.4,2.3,2.6,9.1],
                                              [3.9,1.0,1.9,1.8,1.7]])
    flow_directions_output_array =  np.array([[5,4,4,4,6],
                                              [8,7,7,7,9],
                                              [9,8,7,7,1],
                                              [3,2,1,3,2],
                                              [6,5,4,6,5]])
    flow_directions_mask         =  np.array([[True ,False,True ,False,False],
                                              [True ,False,False,True ,False],
                                              [True ,False,True ,False,False],
                                              [True ,True ,True ,False,True ],
                                              [False,False,True ,False,False]])
    gradients_input_array         = np.array([[1.5, 2.5,3.5,5.5],
                                              [4.5,-0.5,6.5,8.0],
                                              [2.5, 9.0,0.5,7.5]])

    gradients_expected_output     = np.array([[[6.5,2.0,-4.0,1.0],
                                               [3.0,3.0,2.5,-7.5],
                                               [0.0,0.0,0.0,0.0]],
                                              [[3.0,-3.0,3.0,2.5],
                                               [-2,9.5,-6.0,-0.5],
                                               [0.0,0.0,0.0,0.0]],
                                              [[-2.0,4.0,4.5,-1.0],
                                               [4.5,1.0,1.0,-5.5],
                                               [0.0,0.0,0.0,0.0]],
                                              [[4.0,-1.0,-1.0,-2.0],
                                               [3.5,5.0,-7.0,-1.5],
                                               [5.0,-6.5,8.5,-7.0]],
                                              [[0.0,0.0,0.0,0.0],
                                               [0.0,0.0,0.0,0.0],
                                               [0.0,0.0,0.0,0.0]],
                                              [[1.0,1.0,2.0,-4.0],
                                               [-5.0,7.0,1.5,-3.5],
                                               [6.5,-8.5,7.0,-5.0]],
                                              [[0.0,0.0,0.0,0.0],
                                               [1.0,2.0,-4.0,-4.5],
                                               [5.5,-4.5,-1.0,-1.0]],
                                              [[0.0,0.0,0.0,0.0],
                                               [-3.0,3.0,-3.0,-2.5],
                                               [2.0,-9.5,6.0,0.5]],
                                              [[0.0,0.0,0.0,0.0],
                                               [-2.0,4.0,-1,-6.5],
                                               [-3.0,-2.5,7.5,-3.0]]])

    further_gradients_test_data    = np.array([[[4.0,-2.0,3.0,1.5],
                                               [3.5,5.0,1.5,-7.0],
                                               [0.0,0.0,0.0,0.0]],
                                              [[3.5,-3.5,1.0,1.5],
                                               [-3,9.0,-5.5,-1.5],
                                               [0.0,0.0,0.0,0.0]],
                                              [[-4.0,3.5,3.5,-0.5],
                                               [5.0,1.5,0.5,-3.5],
                                               [0.0,0.0,0.0,0.0]],
                                              [[4.5,-2.0,-3.0,-1.0],
                                               [4.0,7.0,-8.0,-2.5],
                                               [5.0,-6.5,8.0,-7.0]],
                                              [[0.0,0.0,0.0,0.0],
                                               [0.0,0.0,0.0,0.0],
                                               [0.0,0.0,0.0,0.0]],
                                              [[1.0,1.5,3.0,-4.0],
                                               [-5.0,7.0,-1.5,-3.5],
                                               [4.5,-7.5,5.0,-5.0]],
                                              [[0.0,0.0,0.0,0.0],
                                               [2.0,3.0,-4.0,-4.0],
                                               [5.0,-2.5,-2.0,-1.5]],
                                              [[0.0,0.0,0.0,0.0],
                                               [-3.0,3.0,-2.0,-1.5],
                                               [2.0,-9.0,4.0,1.5]],
                                              [[0.0,0.0,0.0,0.0],
                                               [-2.0,1.0,-1,-2.5],
                                               [-103.5,-3.5,7.0,3.0]]])

    small_gradient_change_array = np.array([[[1.5,2.0],
                                            [1.5,2.0]]]*9)

    zero_offset_change_expected_results = np.array([[[True,False],
                                                     [True,False]]]*9)

    all_gradients_methods_abs_tol_expected_results = np.array([[[True,False,False,True],
                                                                [True,False,False,False],
                                                                [False,True,True,False]],
                                                               [[True,False,False,True],
                                                                [True,False,False,False],
                                                                [False,True,True,False]],
                                                               [[True,False,False,True],
                                                                [True,False,False,False],
                                                                [False,True,True,False]],
                                                               [[True,False,False,True],
                                                                [True,False,False,False],
                                                                [False,True,True,False]],
                                                               [[True,False,False,True],
                                                                [True,False,False,False],
                                                                [False,True,True,False]],
                                                               [[True,False,False,True],
                                                                [True,False,False,False],
                                                                [False,True,True,False]],
                                                               [[True,False,False,True],
                                                                [True,False,False,False],
                                                                [False,True,True,False]],
                                                               [[True,False,False,True],
                                                                [True,False,False,False],
                                                                [False,True,True,False]],
                                                               [[True,False,False,True],
                                                                [True,False,False,False],
                                                                [False,True,True,False]]])

    all_gradients_methods_rel_tol_expected_results = np.array([[[True,False,False,False],
                                                                [True,False,False,False],
                                                                [True,True,True,False]],
                                                               [[True,False,False,False],
                                                                [True,False,False,False],
                                                                [True,True,True,False]],
                                                               [[True,False,False,False],
                                                                [True,False,False,False],
                                                                [True,True,True,False]],
                                                               [[True,False,False,False],
                                                                [True,False,False,False],
                                                                [True,True,True,False]],
                                                                [[True,False,False,False],
                                                                [True,False,False,False],
                                                                [True,True,True,False]],
                                                                [[True,False,False,False],
                                                                [True,False,False,False],
                                                                [True,True,True,False]],
                                                                [[True,False,False,False],
                                                                [True,False,False,False],
                                                                [True,True,True,False]],
                                                                [[True,False,False,False],
                                                                [True,False,False,False],
                                                                [True,True,True,False]],
                                                                [[True,False,False,False],
                                                                [True,False,False,False],
                                                                [True,True,True,False]]])

    all_gradients_methods_mixed_tol_expected_results = np.array([[[True,False,False,False],
                                                                  [True,False,False,False],
                                                                  [False,True,True,False]],
                                                                 [[True,False,False,False],
                                                                  [True,False,False,False],
                                                                  [False,True,True,False]],
                                                                 [[True,False,False,False],
                                                                  [True,False,False,False],
                                                                  [False,True,True,False]],
                                                                 [[True,False,False,False],
                                                                  [True,False,False,False],
                                                                  [False,True,True,False]],
                                                                 [[True,False,False,False],
                                                                  [True,False,False,False],
                                                                  [False,True,True,False]],
                                                                 [[True,False,False,False],
                                                                  [True,False,False,False],
                                                                  [False,True,True,False]],
                                                                 [[True,False,False,False],
                                                                  [True,False,False,False],
                                                                  [False,True,True,False]],
                                                                 [[True,False,False,False],
                                                                  [True,False,False,False],
                                                                  [False,True,True,False]],
                                                                 [[True,False,False,False],
                                                                  [True,False,False,False],
                                                                  [False,True,True,False]]])

    river_mouth_rdirs_expected_output_array = np.array([[5.0,3.0,-1.0, -1.0,-1.0,-1.0, 8.0,-1.0,-1.0],
                                                        [-1.0,-1.0,0.0, -1.0,2.0,-1.0, -1.0,-1.0,5.0],
                                                        [-1.0,-1.0,-1.0, 6.0,5.0,4.0, -1.0,-1.0,5.0],
                                                        [-1.0,-1.0,-1.0, -1.0,8.0,-1.0, -1.0,0.0,3.0],
                                                        [0.0,-1.0,0.0, -1.0,-1.0,0.0, 9.0,-1.0,6.0],
                                                        [1.0,-1.0,8.0, -1.0,2.0,1.0, 7.0,-1.0,9.0],
                                                        [4.0,-1.0,8.0, -1.0,0.0,0.0, 4.0,-1.0,0.0],
                                                        [7.0,-1.0,8.0, -1.0,-1.0,-1.0, 6.0,0.0,-1.0],
                                                        [4.0,2.0,4.0,5.0,6.0,6.0,  6.0,7.0,6.0]])

    flip_ud_test_input_data = np.array([[1,2],
                                        [3,4],
                                        [5,6],
                                        [7,8],
                                        [9,10]])

    flip_ud_test_expected_output = np.array([[9,10],
                                             [7,8],
                                             [5,6],
                                             [3,4],
                                             [1,2]])

    translation_test_input = np.array([[1,2,3,4],
                                       [11,12,13,14]])

    translation_test_expected_output = np.array([[3,4,1,2,],
                                                 [13,14,11,12]])

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

    find_area_minima_test_input_coords = [(0,0),(2*4,0),(1*4,1*4),(1*4,2*4)]

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

    find_flagged_points_input_data = np.copy(find_area_minima_test_expected_output)
    test_getting_flagged_points_coords_expected_output = [(0,0),(5,5),(7,11),(11,1)]

    def setUp(self):
        """Unit test setup method. Prepare a grid object and various fields"""
        self.hdgrid    = gd.makeGrid('HD')
        self.flow_directions_masked_input_array = np.ma.array(self.flow_directions_input_array,mask=self.flow_directions_mask)
        self.orography = fld.Orography(self.flow_directions_input_array)
        self.masked_orography = fld.Orography(self.flow_directions_masked_input_array)
        self.field     = fld.Field(self.flow_directions_output_array)
        self.flow_directions_output_field = fld.Field(self.flow_directions_output_array)

    def testCalculateGradients(self):
        """Test the calculation of gradients"""
        gradients = self.hdgrid.calculate_gradients(self.gradients_input_array)
        np.testing.assert_array_equal(gradients, self.gradients_expected_output,
                                      "Calculation of gradients not producing expected result")

    #This is a helper to help test a helper!
    def MaskHelperAllNeighboursMethodHelper(self):
        """Assist in testing the all neighbours method of the significant gradient mask helper class"""
        self.input_gradients =  np.copy(self.gradients_expected_output)
        self.input_old_gradients = np.copy(self.further_gradients_test_data)
        self.small_gradient_change_array_input = np.copy(self.small_gradient_change_array)

    def testMaskHelperAllNeighboursMethodNoTols(self):
        """Test the all neighbours method of the significant gradient mask helper class with no tolerance"""
        self.MaskHelperAllNeighboursMethodHelper()
        self.assertEqual(np.any(np.ma.getmask(gd.LatLongGridGradientChangeMaskingHelper.\
                                              all_neighbours_method(self.input_gradients - np.zeros((9,3,4)),
                                                                    np.zeros((9,3,4))))),
                         False,
                         "All gradients method with no tolerances supplied is not returning no masking as expected")

    def testMaskHelperAllNeighboursMethodAbsTols(self):
        """Test the all neighbours method of the significant gradient mask helper class with an absolute tolerance"""
        self.MaskHelperAllNeighboursMethodHelper()
        np.testing.assert_array_equal(np.ma.getmask(gd.LatLongGridGradientChangeMaskingHelper.\
                                                    all_neighbours_method(self.input_gradients-self.input_old_gradients,
                                                                          self.input_old_gradients,
                                                                          gc_absolute_tol=2.5)),
                                      self.all_gradients_methods_abs_tol_expected_results,
                                      "All gradients method with an absolute tolerance is not returning expected results")

    def testMaskHelperAllNeighboursMethodRelTols(self):
        """Test the all neighbours method of the significant gradient mask helper class with a relative tolerance"""
        self.MaskHelperAllNeighboursMethodHelper()
        np.testing.assert_array_equal(np.ma.getmask(gd.LatLongGridGradientChangeMaskingHelper.\
                                                    all_neighbours_method(self.input_gradients-self.input_old_gradients,
                                                                          self.input_old_gradients,
                                                                          gc_relative_tol=0.99)),
                                      self.all_gradients_methods_rel_tol_expected_results,
                                      "All gradients method with an relative tolerance is not returning expected results")

    def testMaskHelperAllNeighboursMethodMixedTols(self):
        """Test the all neighbours method with a combination of an absolute and a relative tolerance"""
        self.MaskHelperAllNeighboursMethodHelper()
        np.testing.assert_array_equal(np.ma.getmask(gd.LatLongGridGradientChangeMaskingHelper.\
                                                    all_neighbours_method(self.input_gradients-self.input_old_gradients,
                                                                          self.input_old_gradients,
                                                                          gc_absolute_tol=2.5,
                                                                          gc_relative_tol=0.99)),
                                      self.all_gradients_methods_mixed_tol_expected_results,
                                      "All gradients method with an relative and absolute tolerances is not returning expected results")

    def testMaskHelperAllNeighboursMethodRelTolsChangeZeroOffset(self):
        """Test the all neighbours method using a relative tolerance with the zero offset changed"""
        self.MaskHelperAllNeighboursMethodHelper()
        np.testing.assert_array_equal(np.ma.getmask(gd.LatLongGridGradientChangeMaskingHelper.\
                                                    all_neighbours_method(self.small_gradient_change_array_input,
                                                                          self.small_gradient_change_array_input,
                                                                          gc_relative_tol=0.99,
                                                                          gc_frac_value_zero_offset=2.0)),
                                      self.zero_offset_change_expected_results,
                                      "Changing the zero offset used to calculate relative gradients to apply tolerances to doesn't give expected results")

    def testMaskExtention(self):
        """Test the extension of a mask to include nearest neighbours"""
        np.testing.assert_array_equal(self.hdgrid.extend_mask_to_neighbours(self.old_mask),self.new_mask,"HD Mask not expanding correctly")

    def testflowdirectionkernel(self):
        """Test the flow direction kernel"""
        self.assertEqual(self.hdgrid.flow_direction_kernel(self.kernel_input_array),3.,"Flow direction kernel not operating correctly")

    def testfortranflowdirectionkernel(self):
        """Test the fortran version of the flow direction kernel"""
        f2py_mngr = f2py_manager.f2py_manager(os.path.join(fortran_source_path,'mod_grid_flow_direction_kernels.f90'),
                                              func_name='HDgrid_fdir_kernel')
        self.assertEqual(f2py_mngr.run_current_function_or_subroutine(self.kernel_input_array),3.,
                         "Fortran flow direction kernel not operating correctly")

    def testflowdirectionkernal_default_to_central_min_case(self):
        """Test that the flow direction kernel defaults to zero when center is also a minima"""
        self.assertEqual(self.hdgrid.flow_direction_kernel(self.kernel_input_array_with_central_min),5.,
                         "Flow direction kernel is not correctly defaulting to 5 (sink point) when central point has same minimum as a neighbour")

    def testfortranflowdirectionkernal_default_to_central_min_case(self):
        """Test that the Fortran flow direction kernel defaults to zero when center is also a minima"""
        f2py_mngr = f2py_manager.f2py_manager(os.path.join(fortran_source_path,'mod_grid_flow_direction_kernels.f90'),
                                              func_name='HDgrid_fdir_kernel')
        self.assertEqual(f2py_mngr.run_current_function_or_subroutine(self.kernel_input_array_with_central_min),5.,
                         "Fortran flow direction kernel is not correctly defaulting to 5 (sink point) when central point has same minimum as a neighbour")

    def testcomputingflowdirections(self):
        """Test computing the flow directions on a field"""
        np.testing.assert_array_equal(self.hdgrid.compute_flow_directions(self.flow_directions_input_array),self.flow_directions_output_array)

    def testgenerateflowdirectionfromorographywithoutmask(self):
        """Test computing the flow directions on a field at top level without changes mask"""
        np.testing.assert_array_equal(dynamic_hd.generate_flow_directions_from_orography(self.orography).get_data(),
                                      self.flow_directions_output_field.get_data())

    def testgenerateflowdirectionfromorographywithmask(self):
        """Test computing the flow directions on a field at top level with changes mask"""
        np.testing.assert_array_equal(np.ma.getmaskarray(dynamic_hd.generate_flow_directions_from_orography(self.masked_orography).get_data()),
                                      self.flow_directions_mask)
        np.testing.assert_array_equal(dynamic_hd.generate_flow_directions_from_orography(self.masked_orography).get_data(),
                                      self.flow_directions_output_field.get_data())

    def testfortranrivermouthkernel_non_sea_point(self):
        """Test the fortran river mouth kernel with a non sea point"""
        f2py_mngr = f2py_manager.f2py_manager(os.path.join(fortran_source_path,'mod_river_mouth_kernels.f90'),
                                              func_name='latlongrid_river_mouth_kernel')
        self.assertEqual(f2py_mngr.run_current_function_or_subroutine(self.kernel_flow_dir_input_array_not_sea),5.0,
                         "Response of river mouth finding kernel to non-sea point is incorrect")

    def testfortranrivermouthkernel_not_river_mouth(self):
        """Test the fortran river mouth kernel with a sea point that is not a river mouth"""
        f2py_mngr = f2py_manager.f2py_manager(os.path.join(fortran_source_path,'mod_river_mouth_kernels.f90'),
                                              func_name='latlongrid_river_mouth_kernel')
        self.assertEqual(f2py_mngr.run_current_function_or_subroutine(self.kernel_flow_dir_input_array_sea_no_river_inflow),
                         -1.0,"Response of river mouth finding kernel to non river mouth point is incorrect")

    def testfortranrivermouthkernel_not_river_mouth_with_sinks(self):
        """Test the fortran river mouth kernel with a sea point that is not a river mouth when surrounded by sinks"""
        f2py_mngr = f2py_manager.f2py_manager(os.path.join(fortran_source_path,'mod_river_mouth_kernels.f90'),
                                              func_name='latlongrid_river_mouth_kernel')
        self.assertEqual(f2py_mngr.\
                         run_current_function_or_subroutine(self.\
                                                            kernel_flow_dir_input_array_sea_no_river_inflow_with_sinks),
                         -1.0,
                         "Response of river mouth finding kernel to non river mouth point surrounded by sinks"
                         " is incorrect")

    def testfortranrivermouthkernel_river_mouth(self):
        """Test the fortran river mouth kernel correctly identifies a river mouth"""
        f2py_mngr = f2py_manager.f2py_manager(os.path.join(fortran_source_path,'mod_river_mouth_kernels.f90'),
                                              func_name='latlongrid_river_mouth_kernel')
        for i in range(9):
            if i == 4:
                continue
            rdirs_with_inflow = np.copy(self.kernel_flow_dir_input_array_sea_no_river_inflow)
            rdirs_with_inflow[i] = (3,2,1,6,5,4,9,8,7)[i]
            self.assertEqual(f2py_mngr.run_current_function_or_subroutine(rdirs_with_inflow),0.0,
                             "Response of river mouth finding kernel to a river mouth is incorrect")

    def test_marking_river_mouths(self):
        """Test the marking of river mouths with a small section of pseudo-data"""
        np.testing.assert_array_equal(self.hdgrid.mark_river_mouths(self.river_mouth_rdirs_input_array),
                                      self.river_mouth_rdirs_expected_output_array,
                                      "Test of river mouth marking with pseudo-data did not produce expected results")

    def test_flipping_data_ud(self):
        """Test flipping the data upside down"""
        np.testing.assert_array_equal(self.hdgrid.flip_ud(self.flip_ud_test_input_data),
                                      self.flip_ud_test_expected_output,
                                      "Flipping data upside down doesn't produce expected result")

    #include this here for convenience even though it actually acts on a field object
    def test_flipping_data_ud_on_field(self):
        """Test flipping a field upside down"""
        field = fld.Field(self.flip_ud_test_input_data,'HD')
        field.flip_data_ud()
        np.testing.assert_array_equal(field.get_data(),self.flip_ud_test_expected_output,
                                      "Flipping a field upside down doesn't produce expected result")

    def test_rotating_data_by_180_degrees(self):
        """Test applying a 180 degree longitude translation to the data"""
        np.testing.assert_array_equal(self.hdgrid.\
                                      one_hundred_eighty_degree_longitude_translation(self.translation_test_input),
                                      self.translation_test_expected_output,
                                      "Test of rotating globe by 180 degrees is not producing expected output")

    #include this here for convenience even though it actually acts on a field object
    def test_rotating_data_by_180_degrees_on_field(self):
        """Test applying a 180 degree rotation to a field"""
        field = fld.Field(self.translation_test_input,'HD')
        field.rotate_field_by_a_hundred_and_eighty_degrees()
        np.testing.assert_array_equal(field.get_data(),self.translation_test_expected_output,
                                      "Test of rotating field by 180 degrees in longitude direction is not"
                                      " producing the expected output")

    def test_finding_area_minima(self):
        """Test finding the minima within course scale cell boundaries on a fine orography"""
        truesinks_downscaled = self.hdgrid.find_area_minima(self.find_area_minima_test_input_orog,
                                                            self.find_area_minima_test_input_coords,(4,4))
        np.testing.assert_array_equal(truesinks_downscaled,self.find_area_minima_test_expected_output,
                                      "Finding minima of course scale cells is not producing expected results")

    def test_getting_flagged_point_coords(self):
        """Test getting the coordinates of flagged points within an array of booleans"""
        coords = self.hdgrid.get_flagged_points_coords(self.find_flagged_points_input_data)
        self.assertEqual(coords, self.test_getting_flagged_points_coords_expected_output,
                         "Finding coords of True points in boolean array is not functioning"
                         " correctly")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
