'''
Improved loading/writing functions that deal with field orientation if possible and trys to
guess some information where possible
Created on Dec 4, 2017

@author: thomasriddick
'''
import dynamic_hd
import iohelper
import re
import os
import math
import numpy as np
import os.path as path
import tempfile
import xarray

global_regular_grid_desc_one = re.compile(r'^global_([-.0-9]+)$')
global_regular_grid_desc_two = re.compile(r'^r([-.0-9]+)x([-.0-9]+)$')
grid_desc_nlon = re.compile(r'^ *xsize *= *([-.0-9]+)')
grid_desc_nlat = re.compile(r'^ *ysize *= *([-.0-9]+)')
grid_desc_lon_first = re.compile(r'^ *xfirst *= *([-.0-9]+)')
grid_desc_lat_first = re.compile(r'^ *yfirst *= *([-.0-9]+)')
grid_desc_lon_inc = re.compile(r'^ *xinc *= *([-.0-9]+)')
grid_desc_lat_inc = re.compile(r'^ *yinc *= *([-.0-9]+)')

class Grid_Description(object):
    """Class to hold a single grid description parameter and the pattern that matches it"""

    def __init__(self,pattern):
        """Class Constructor.

        Arguments:
        pattern: re regular expression object; the pattern that matches this grid descriptor
            within a grid description file and captures its value in the regular expressions
            first (and only) capture group
        """

        self.pattern = pattern
        self.value   = None

    def check_pattern_was_found(self):
        """Check that this grid description parameter has been found

        Arguments: None
        Returns: A flag indicating if this parameter has been found; if it has the flag is True;
            otherwise it is false
        """

        return bool(self.value if not self.value == 0 else True)

    def get_value(self):
        """Get the value"""
        return self.value

class Grid_Descriptions(object):
    """Class to hold a collection of grid description parameters"""

    def __init__(self,grid_descs):
        """Add a set of grid descriptions to this classes name space

        Arguments:
        grid_descs: dictionary; a dictionary containing the required variable names of
            grid description parameters as keys and the grid description object for them
            as values
        """

        self.__dict__.update(grid_descs)

    def check_all_patterns_found(self):
        """Check that all the patterns stored in this object have been found

        Arguments: None
        Return: A flag indicating if all the pattern in this object have values associated
            with them (indicating they have been found) (True) or not (False)
        """

        for gd in vars(self).values():
            if not gd.check_pattern_was_found():
                return False
        return True

    def convert_to_values(self):
        """Convert all this objects from grid description objects to values

        Arguments:None
        Returns: Nothing
        """

        for varname,grid_desc_oj in vars(self).iteritems():
            self.__dict__[varname] = grid_desc_oj.get_value()


def advanced_field_loader(filename,time_slice=None,grid_desc_file=None,grid_desc=None,
                          field_type=None,fieldname=None,adjust_orientation=True):
    """Load a field, gather orientation information and potentially reorientate it

    Arguments:
        filename: string; the full path of the file to load the field from
        time_slice: integer; which timeslice to select from a (netCDF4) file with multiple time
            slices
        grid_desc_file: string; full path to a file describing the grid and its orientation in
            a cdo-style format
        grid_desc: string; a cdo-style grid description for the grid and its orientation
        field_type: string; the code for the field type to use; if none is specified then try
            and guess; if that is not possible then revert to Generic
        fieldname: string; which field to select from a (netCDF4) file with multiple fields
        adjust_orientation: boolean; Adjust the orientation to the standard orientation if
            it has been successful determined and is different from the standard orientation
    """

    if grid_desc and grid_desc_file:
        RuntimeError("Not possible to specify both a grid type directly and via a file")
    elif (not grid_desc) and (not grid_desc_file):
        check_for_grid_info=True
        grid_type='HD'
        grid_params={}
    else:
        check_for_grid_info=False
    if not field_type:
        if fieldname in ['Topo','topo','orog','z']:
            field_type = 'Orography'
        else:
            field_type = 'Generic'
    coordinates = None
    if grid_desc:
        match_one = global_regular_grid_desc_one.match(grid_desc)
        match_two = global_regular_grid_desc_two.match(grid_desc)
        if match_one:
            dxy = float(match_one.group(1))
            nlat = abs(int(round(180.0/dxy)))
            nlon = abs(int(round(360.0/dxy)))
            grid_type="LatLong"
            grid_params={"nlat":nlat,"nlong":nlon}
            lat_points = np.linspace(-dxy/2 + 90.0,dxy/2 - 90.0,num=nlat)
            lon_points = np.linspace(dxy/2 - 180.0,-dxy/2 + 180.0,num=nlon)
            coordinates = (lat_points,lon_points)
        elif match_two:
            dx = float(match_two.group(1))
            dy = float(match_two.group(2))
            nlat = abs(int(round(180.0/dy)))
            nlon = abs(int(round(360.0/dx)))
            grid_type="LatLong"
            grid_params={"nlat":nlat,"nlong":nlon}
            lat_points = np.linspace(-dy/2 + 90.,dy/2 - 90.0,num=nlat)
            lon_points = np.linspace(0.0,360.0-dx,num=nlon)
            coordinates = (lat_points,lon_points)
        else:
            RuntimeError("Grid description not recognised")
    if grid_desc_file:
        grid_desc_patterns = {'nlon'     :Grid_Description(grid_desc_nlon),
                              'nlat'     :Grid_Description(grid_desc_nlat),
                              'lon_first':Grid_Description(grid_desc_lon_first),
                              'lat_first':Grid_Description(grid_desc_lat_first),
                              'lon_inc'  :Grid_Description(grid_desc_lon_inc),
                              'lat_inc'  :Grid_Description(grid_desc_lat_inc)}
        with open(grid_desc_file,'r') as f:
            for line in f:
                for varname,grid_desc in grid_desc_patterns.iteritems():
                    match_for_pattern = None
                    match_for_pattern = grid_desc.pattern.match(line)
                    if match_for_pattern:
                        grid_desc_patterns[varname].value = float(match_for_pattern.group(1))
        grid_descs = Grid_Descriptions(grid_desc_patterns)
        if not grid_descs.check_all_patterns_found():
            raise RuntimeError("Grid description file does not contain all necessary"
                               " information")
        grid_descs.convert_to_values()
        grid_type="LatLong"
        grid_params={"nlat":int(grid_descs.nlat),"nlong":int(grid_descs.nlon)}
        lat_points = np.linspace(grid_descs.lat_first,
                                 grid_descs.lat_first+grid_descs.lat_inc*
                                 (grid_descs.nlat-1),
                                 num=int(grid_descs.nlat))
        lon_points = np.linspace(grid_descs.lon_first,
                                 grid_descs.lon_first+grid_descs.lon_inc*
                                 (grid_descs.nlon-1),
                                 num=int(grid_descs.nlon))
        coordinates = (lat_points,lon_points)
    field = dynamic_hd.load_field(filename,
                                  file_type=dynamic_hd.get_file_extension(filename),
                                  field_type=field_type,
                                  unmask=True,
                                  timeslice=time_slice,
                                  fieldname=fieldname,
                                  check_for_grid_info=check_for_grid_info,
                                  grid_type=grid_type,
                                  **grid_params)
    if coordinates is not None:
        if field.grid_has_coordinates():
            raise RuntimeError("Trying to set grid orientation information when this was already"
                               " specified in data file")
        field.set_grid_coordinates(coordinates)
    if adjust_orientation:
        field.orient_data()
    return field

def add_grid_information_to_field(target_filename,original_filename,
                                  target_fieldname='field_value',
                                  original_fieldname='field_value',time_slice=None,
                                  flip_ud_raw=False,rotate180lr_raw=False,
                                  grid_desc_file=None,grid_desc=None,clobber=False):
    field = advanced_field_loader(filename=original_filename,time_slice=time_slice,
                                  grid_desc_file=grid_desc_file,grid_desc=grid_desc,
                                  field_type=None,fieldname=original_fieldname,
                                  adjust_orientation=False)
    if flip_ud_raw:
      field.set_data(np.flipud(field.get_data()))
    if rotate180lr_raw:
      field.set_data(np.roll(field.get_data(),field.get_data().shape[1]/2,axis=1))
    advanced_field_writer(target_filename=target_filename,
                          field=field,
                          fieldname=target_fieldname,
                          clobber=clobber)

def advanced_field_writer(target_filename,field,fieldname='field_value',clobber=False):
    """Write a field to a file with grid information if any is available

    Arguments:
    target_filename: string; full path to target file to write field to
    field: field object; field to write to the file
    fieldname: string; name of the field to write to
    clobber: boolean; if true then overwrite any existing file at target_filename
        otherwise raise an error if target_filename already exists
    Returns: Nothing
    """
    if (path.exists(target_filename) and not clobber):
        raise RuntimeError("Target file {} already exists and clobbering is not set".
                           format(target_filename))
    temp_filename=None
    if (isinstance(field,list)):
      fields = field
      field = fields[0]
      fieldnames = fieldname
      multiple_fields = True
    else:
      multiple_fields = False
    threshold_for_using_xarray = 500000000
    if (field.get_grid().get_npoints() > threshold_for_using_xarray and
        field.grid.lon_points is not None and field.grid.lat_points is not None):
      if multiple_fields:
        RuntimeError("Cannot write multiple fields using x-array")
      data_array = field.to_data_array()
      dataset = xarray.Dataset({fieldname:data_array})
      print "Writing output to {0}".format(target_filename)
      dataset.to_netcdf(target_filename)
    else:
      try:
        if (field.grid.lon_points is not None and
            field.grid.lat_points is not None):
          temp_file, temp_filename = tempfile.mkstemp(suffix=".txt",text=False)
          with os.fdopen(temp_file,'w') as f:
              f.writelines([s + '\n' for s in
                           ["gridtype = lonlat",
                            "gridsize = {}".format(field.grid.nlat*field.grid.nlong),
                            "xsize = {}".format(field.grid.nlong),
                            "ysize = {}".format(field.grid.nlat),
                            "xfirst = {}".format(field.grid.lon_points[0]),
                            "xinc = {}".format(math.copysign(360.0/field.grid.nlong,
                                                             field.grid.lon_points[1]-
                                                             field.grid.lon_points[0])),
                            "yfirst = {}".format(field.grid.lat_points[0]),
                            "yinc = {}".format(math.copysign(180.0/field.grid.nlat,
                                                             field.grid.lat_points[1]-
                                                             field.grid.lat_points[0]))]])
        if not multiple_fields:
          dynamic_hd.write_field(filename=target_filename,
                                 field=field,
                                 file_type=dynamic_hd.get_file_extension(target_filename),
                                 griddescfile=temp_filename,
                                 fieldname=fieldname)
        else:
          iohelper.NetCDF4FileIOHelper.\
            write_fields(filename=target_filename,
                         fields=fields,
                         griddescfile=temp_filename,
                         fieldnames=fieldnames)
      finally:
        if temp_filename:
          os.remove(temp_filename)
