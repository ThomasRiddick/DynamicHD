'''
Classes for reading and writing fields from different types of files.
Created on Jan 13, 2016

@author: thomasriddick
'''

from . import grid as gd
import numpy as np
import scipy.io as scipyio
from abc import ABCMeta, abstractmethod
import netCDF4
from . import f2py_manager as f2py_mg
import os.path as path
import os
import cdo
from .context import fortran_source_path

class IOHelper(object, metaclass=ABCMeta):
    """Parent class for classes that load and write different types of file.

    Public methods (all abstract):
    load_field
    write_field

    Implementations of this class should make all their members class methods
    """

    @abstractmethod
    def load_field(self,filename,unmask=True,timeslice=None,
                   fieldname=None,check_for_grid_info=False,
                   grid_info=None,grid_type='HD',**grid_kwargs):
        """Load a field from a file

        Arguments:
        filename: full path of the file to load
        unmask (optional): boolean; flag to specifying whether the field is returned
            with any mask removed (True) or not (False)
        timeslice (optional): integer, which time slice to choose from file with multiple timeslices
        fieldname (optional): string, name of field to load from file with multiple fields
        check_for_grid_info: boolean; Search to see if file has grid info and use this
            to replace grid type specified if found
        grid_info (optional): grid within a one element list; can be used to return grid information
            via a grid object wrapped in a list if any has been found. Not used as an input to specify
            the grid.
        grid_type: keyword specifying what type of grid to use
        **grid_kwargs: keyword arguments giving parameters of the grid

        Implementations should return the loaded field and print out a notice
        the file has been read.
        """

        pass

    @abstractmethod
    def write_field(self,filename,field,griddescfile=None,fieldname=None):
        """Write a field to a file

        Arguments:
        filename: full path of the file to write to
        field: field to write
        griddescfile (optional): string; full path to the grid description metadata
            to add to file written out (if possible). Nothing is added if this is
            set to None
        fieldname: string; name of the output field to create and write to

        Implementation should print out a notice the file has been written to.
        """

        pass

class TextFileIOHelper(IOHelper):
    """Class to load and write plain text files

    Public methods:
    As for parent class
    """

    @classmethod
    def load_field(self, filename,unmask=True,timeslice=None,fieldname=None,
                   check_for_grid_info=False,grid_info=None,grid_type='HD',
                   **grid_kwargs):
        """Load a field from a given text file

        Arguments:
        filename: string; full path to the text file
        grid_type: string; keyword specifying the grid type to use
        **grid_kwargs: keyword dictionary; keyword arguments specifying parameter of the
            grid
        fieldname, timeslice, unmask, check_for_grid_info and grid_info are not used by this
            method and included only in order to match the abstract of method of IOFileHelper
        Returns:
        Field as a numpy array

        Loads a field from a text file using numpys load text fuction then reshapes it to
        the dimension of the given grid.
        """

        print("Reading input from {0}".format(filename))
        grid = gd.makeGrid(grid_type,**grid_kwargs)
        return np.loadtxt(filename,np.float64).reshape(grid.get_grid_dimensions())

    @classmethod
    def write_field(self,filename,field,griddescfile=None,fieldname=None):
        """Write a field to a file

        Arguments:
        filename: string, full path of the text file to write to
        field: A Field (or Field subclass object); field to write
        griddescfile is not used by this method and included only in
            order to match the abstract of method of IOFileHelper
        fieldname is not used by this method and included only in order to match the
            abstract of method of IOFileHelper
        Returns: nothing

        Write a field out such that it could be loaded by the load_field function above
        """

        print("Writing output to {0}".format(filename))
        np.savetxt(filename, field.get_data())

class NetCDF4FileIOHelper(IOHelper):
    """Class to load and write netcdf4 files

    Public methods:
    As for parent class
    copy_and_append_time_dimension_to_netcdf_dataset
    append_earlier_timeslice_to_dataset
    """

    @classmethod
    def load_field(self,filename,unmask=True,timeslice=None,fieldname=None,
                   check_for_grid_info=False,grid_info=None,grid_type='HD',
                   **grid_kwargs):
        """Load a field from a given NetCDF4 file

        Arguments:
        filename: string; full path to the NetCDF4 file
        grid_type: string; keyword specifying the grid type to use
        unmask (optional): boolean; flag to specifying whether the field is returned
            with any mask removed (True) or not (False)
        timeslice (optional): integer; which time slice to choose from file with multiple timeslices
        fieldname (optional): string; name of field to load from file with multiple fields
        check_for_grid_info: boolean; Search to see if file has grid info and use this
            to replace grid type specified if found
        grid_info (optional): grid in a list; can be used to return the grid object wrapped in a list
            if one was set by reading latitude and longitude parameters. Not used as input to specify
            the grid.
        **grid_kwargs: keyword dictionary; keyword arguments specifying parameter of the
            grid
        Returns:
        Either (depending on the flag supplied) a numpy masked array or a numpy array
        containing the field.

        Loads a field from a NetCDF4 file using numpys load text fuction then reshapes it to
        the dimension of the given grid.
        """

        if not check_for_grid_info:
            grid = gd.makeGrid(grid_type,**grid_kwargs)
        print("Reading input from {0}".format(filename))
        with netCDF4.Dataset(filename,mode='r',format='NETCDF4') as dataset:
            if check_for_grid_info:
                latitudes = None
                longitudes = None
                fields = dataset.get_variables_by_attributes(name='lat')
                if len(fields) == 1:
                    latitudes = fields[0][:]
                    for longitude_names in ['lon','long']:
                        fields = dataset.get_variables_by_attributes(name=longitude_names)
                        if len(fields) >= 1:
                            break
                    if len(fields) == 1:
                        longitudes = fields[0][:]
                    elif len(fields) > 1:
                        raise RuntimeError("File {0} contains"
                                           " multiple longitude fields".format(filename))
                elif len(fields) > 1:
                    raise RuntimeError("File {0} contains"
                                       " multiple latitude fields".format(filename))
                if longitudes is not None:
                    grid = gd.makeGrid('LatLong',nlat=len(latitudes),nlong=len(longitudes))
                    grid.set_latitude_points(np.asarray(latitudes))
                    grid.set_longitude_points(np.asarray(longitudes))
                    grid_info.append(grid)
                else:
                    grid = gd.makeGrid(grid_type,**grid_kwargs)
            fields = None
            if fieldname is None:
                potential_field_names = ['Topo','topo','field_value','orog','z','ICEM',
                                         'DEPTO','usurf','bats','slm','FDIR','lsmask',
                                         'lake_field','river_flow',
                                         'basin_catchment_numbers','rdirs','lsm']
            else:
                potential_field_names = [fieldname]
            for potential_field_name in potential_field_names:
                fields = dataset.get_variables_by_attributes(name=potential_field_name)
                if len(fields) >= 1:
                    break
            if len(fields) == 1:
                if timeslice is not None:
                    field_slice = fields[0][timeslice,:,:]
                else:
                    field_slice = fields[0][:]
                if unmask:
                    return np.asarray(field_slice.reshape(grid.get_grid_dimensions()))
                else:
                    return np.asanyarray(field_slice.reshape(grid.get_grid_dimensions()))
            elif len(fields) > 1:
                raise RuntimeError('File {0} contains multiple fields'.format(filename))
            else:
                raise RuntimeError('Field not found in file {0}'.format(filename))

    @classmethod
    def write_field(self, filename, field,griddescfile=None,fieldname=None):
        """Write a field to a given target NetCDF4 file

        Arguments:
        filename: full path of the netcdf file to write to
        field: A Field (or Field subclass object); field to write
        griddescfile (optional): string; full path to the grid description metadata
            to add to file written out. Nothing is added if this is
            set to None
        fieldname: string; name of the output field to create and write to
        Returns:nothing
        """

        nlat,nlong = field.get_grid().get_grid_dimensions()
        if fieldname is None:
            fieldname = 'field_value'
        print("Writing output to {0}".format(filename))
        if griddescfile is not None:
            output_filename=filename
            filename=path.splitext(filename)[0] + '_temp' + path.splitext(filename)[1]
        data_was_bool = False
        with netCDF4.Dataset(filename,mode='w',format='NETCDF4') as dataset:
            dataset.createDimension("latitude",nlat)
            dataset.createDimension("longitude",nlong)
            if field.get_data().dtype == np.bool:
                field.set_data(field.get_data().astype(np.int32))
                data_was_bool=True
            field_values = dataset.createVariable(fieldname,field.get_data().dtype,
                                                  ('latitude','longitude'))
            field_values[:,:] = field.get_data()
        if data_was_bool:
            field.set_data(field.get_data().astype(np.bool))
        if griddescfile is not None:
            cdo_instance = cdo.Cdo()
            cdo_instance.setgrid(griddescfile,input=filename,output=output_filename)
            os.remove(filename)

    @classmethod
    def write_fields(self, filename, fields,griddescfile=None,fieldnames=None):
        """Write a field to a given target NetCDF4 file

        Arguments:
        filename: full path of the netcdf file to write to
        fields: A list of Field (or Field subclass object); fields to write
        griddescfile (optional): string; full path to the grid description metadata
            to add to file written out. Nothing is added if this is
            set to None
        fieldnames: A list of strings; name of the output fields to create and write to
        Returns:nothing
        """

        nlat,nlong = fields[0].get_grid().get_grid_dimensions()
        if fieldnames is None:
            fieldnames = ['field_value']*len(fields)
        print("Writing output to {0}".format(filename))
        if griddescfile is not None:
            output_filename=filename
            filename=path.splitext(filename)[0] + '_temp' + path.splitext(filename)[1]
        with netCDF4.Dataset(filename,mode='w',format='NETCDF4') as dataset:
            dataset.createDimension("latitude",nlat)
            dataset.createDimension("longitude",nlong)
            for field,fieldname in zip(fields,fieldnames):
                data_was_bool = False
                if field.get_data().dtype == np.bool:
                    field.set_data(field.get_data().astype(np.int32))
                    data_was_bool=True
                field_values = dataset.createVariable(fieldname,field.get_data().dtype,
                                                      ('latitude','longitude'))
                field_values[:,:] = field.get_data()
                if data_was_bool:
                    field.set_data(field.get_data().astype(np.bool))
        if griddescfile is not None:
            cdo_instance = cdo.Cdo()
            cdo_instance.setgrid(griddescfile,input=filename,output=output_filename)
            os.remove(filename)

    @classmethod
    def copy_and_append_time_dimension_to_netcdf_dataset(self,dataset_in,dataset_out):
        """Make a copy of a input dataset while adding in an extra time dimension

        Arguments:
        dataset_in: netcdf4-python Dataset object; input dataset object without time dimension
        dataset_out: netcdf4-python Dataset object; empty dataset object to copy input dataset
            object to while adding a time dimension
        """

        for dim_name,dim_obj in list(dataset_in.dimensions.items()):
            dataset_out.createDimension(dim_name,len(dim_obj)
                                        if not dim_obj.isunlimited() else None)
        dataset_out.createDimension('time',None)
        times = dataset_out.createVariable("time",'f8',("time",))
        times.units = "years since 0001-01-01 00:00:00.0"
        times.calendar = "proleptic_gregorian"
        times[0] = np.array([0.0])
        for var_name, var_obj in list(dataset_in.variables.items()):
            new_var = dataset_out.createVariable(var_name,var_obj.datatype,var_obj.dimensions
                                                 if (len(var_obj.dimensions) <= 1
                                                     or var_name == 'AREA') else
                                                 ["time"] + list(var_obj.dimensions))
            if  len(var_obj.dimensions) <= 1 or var_name == 'AREA':
                new_var[:] = var_obj[:]
            else:
                new_var[0,:] = var_obj[:]
            new_var.setncatts({attr_name: var_obj.getncattr(attr_name) for attr_name in var_obj.ncattrs()})

    @classmethod
    def append_earlier_timeslice_to_dataset(self,main_dataset,dataset_to_append,slicetime):
        """Append a dataset containing a single earlier timeslice to another dataset with multiple timeslices

        Arguments:
        main_dataset: netcdf4-python Dataset object; the dataset object to add the timeslice to
        dataset_to_append: netcdf4-python Dataset object; the dataset object containing the timeslice to add
        slicetime: integer; time slice is for in years before present
        Returns: nothing

        The maindata set and the dataset to append must share the same structure other than the former
        must have a timedimension. An AREA diminsion in the datasets is ignore as this should stay
        constant in time. The timeslice to append therefore requires no AREA dimension.
        """

        var_obj = main_dataset.get_variables_by_attributes(name='time')[0]
        if slicetime >= var_obj[0]:
            raise RuntimeError("Trying to append a timeslice for a later date than the oldest timeslice"
                               " already present in the dataset")
        var_obj[1:] = var_obj[:]
        var_obj[0] = slicetime
        for var_name, var_obj in list(main_dataset.variables.items()):
            if var_name == 'time' or var_name == 'AREA':
                continue
            if  len(var_obj.dimensions) > 1:
                var_to_append = dataset_to_append.get_variables_by_attributes(name=var_name)[0]
                var_obj[1:,:] = var_obj[:-1,:]
                var_obj[0,:] = var_to_append[:]

class SciPyFortranFileIOHelper(IOHelper):
    """Class to load and write unformatted fortran files using SciPy Library.

    Public methods:
    As for parent class

    This class will work only for reading and writing 64-bit floats
    """

    data_type=np.float64

    @classmethod
    def load_field(self,filename,unmask=True,timeslice=None,
                   fieldname=None,check_for_grid_info=False,
                   grid_info=None,grid_type='HD',**grid_kwargs):
        """Load a field from a unformatted fortran file using a method from scipy

        Arguments:
        filename: string; full path of the file to load
        grid_type: string; keyword specifying what type of grid to use
        **grid_kwargs: keyword dictionary; keyword arguments giving parameters of the grid
        fieldname, timeslice, unmask, check_for_grid_info and grid_info are not used by
            this method and included only in order to match the abstract of method of
            IOFileHelper
        Returns:
        A numpy array containing the field

        This scipy method cannot read all Fortran file correctly; success or failure depends
        on the endianness and compiler used by the Fortran used to write the data. It will
        succeed in reading data written using FortranFile.write_record.
        """

        grid = gd.makeGrid(grid_type,**grid_kwargs)
        with scipyio.FortranFile(filename,mode='r') as f: #@UndefinedVariable:
            print("Reading input from {0}".format(filename))
            return f.read_record(self.data_type).reshape(grid.get_grid_dimensions())

    @classmethod
    def write_field(self, filename, field,griddescfile=None,fieldname=None):
        """Write a field to an unformatted fortran file using a method from scipy

        Arguments:
        filename: string; full path of the file to write to
        field: A Field (or Field subclass object); field to write
        griddescfile is ignored by this function but included to match the abstract
            function it is overriding
        fieldname is not used by this method and included only in order to match the
            abstract of method of IOFileHelper
        """

        with scipyio.FortranFile(filename,mode='w') as f: #@UndefinedVariable
            print("Writing output to {0}".format(filename))
            f.write_record(field.get_data())

class SciPyFortranFileIOHelperInt(SciPyFortranFileIOHelper):
    """A class to read and write 64-bit integers to/from unformatted fortran files

    Public methods:
    As for parent class

    All methods are inherited from SciPyFortranFileIOHelper. The only difference between
    this and it's parent class SciPyFortranFileIOHelper is to redefine the data type;
    this should allow integers to read.
    """

    data_type=np.int64

class F2PyFortranFileIOHelper(IOHelper):
    """Class to read and write unformatted fortran files using f2py.

    Public methods:
    As for parent class
    """

    @classmethod
    def load_field(self, filename,unmask=True,timeslice=None,
                   fieldname=None,check_for_grid_info=False,
                   grid_info=None,grid_type='HD',**grid_kwargs):
        """Load a field from a unformatted fortran file using f2py

        Arguments:
        filename: string; full path of the file to load
        grid_type: string; keyword specifying what type of grid to use
        **grid_kwargs: keyword dictionary; keyword arguments giving parameters of the grid
        fieldname, timeslice, unmask, check_for_grid_info and grid_info are not used by this
            method and included only in order to match the abstract of method of IOFileHelper
        Returns:
        A numpy array containing the field

        Loads a field using a specifically written fortran routine through f2py controlled
        by a f2py_manager instance. A certain amount of manipulation of the field is required
        to ensure the field is output in the same orientation as other loading methods.
        """

        grid = gd.makeGrid(grid_type,**grid_kwargs)
        print("Reading input from {0}".format(filename))
        mgnr = f2py_mg.f2py_manager(path.join(fortran_source_path,
                                              "mod_topo_io.f90"), func_name="read_topo")
        data = mgnr.run_current_function_or_subroutine(filename,*grid.get_grid_dimensions())
        #rotate the array 90 clockwise (i.e. 270 degrees anticlockwise); also flip
        #to match symmetry of other loading methods
        return np.fliplr(np.rot90(data,k=3))

    @classmethod
    def write_field(self, filename, field,griddescfile=None,fieldname=None):
        """Write a field to an unformatted fortran file using f2py

        Arguments:
        filename: string; full path of the file to write to
        field: A Field (or Field subclass object); field to write
        griddescfile is ignored by this function but included to match the abstract
            function it is overriding
        fieldname is not used by this method and included only in order to match the
            abstract of method of IOFileHelper

        Writes a field using a specifically written fortran routine through f2py controlled
        by a f2py_manager instance. The manipulation applied to the field in the load field
        method is reversed for consistency.
        """

        print("Writing output to {0}".format(filename))
        mgnr = f2py_mg.f2py_manager(path.join(fortran_source_path,
                                              "mod_topo_io.f90"), func_name="write_topo")
        #reverse the manipulation in the load_field method
        data = np.rot90(np.fliplr(field.get_data()))
        mgnr.run_current_function_or_subroutine(filename,data)

def getFileHelper(file_type):
    """Pseudo-factory function. Returns correct IOHelper subclass for a given filename extension

    Arguments:
    file_type: string; code for the IOHelper subclass to use
    Returns: a IOHelper subclass object of the appropriate subclass
    """

    try:
        return {'.nc'     :NetCDF4FileIOHelper,
                '.dat'    :F2PyFortranFileIOHelper,
                '.datx'   :SciPyFortranFileIOHelper,
                '.datxint':SciPyFortranFileIOHelperInt,
                '.txt'    :TextFileIOHelper,
                '.txtint' :TextFileIOHelper}[file_type]
    except KeyError:
        raise RuntimeError('Invalid File Type')
