'''
High level routines of dynamic hydrology script.
June 4, 2021 - Obsolete code remove - now just load/write functions
Created on Dec 15, 2015

@author: triddick
'''

import os
from Dynamic_HD_Scripts import iohelper
from Dynamic_HD_Scripts import field as fld

def load_field(filename,file_type,field_type,unmask=True,timeslice=None,
               fieldname=None,check_for_grid_info=False,grid_type='HD',**grid_kwargs):
    """Inteface that loads a field as a raw array of data and returns it as a field object.

    Arguments:
    filename: string; the full path to the file to opened
    file_type: string; the code for this file type
    field_type: string; the code for this field type
    unmask (optional): boolean; flag to specifying whether the field is returned
            with any mask removed (True) or not (False)
    timeslice: integer; which timeslice to select from a (netCDF4) file with multiple time
        slices
    fieldname: string; which field to select from a (netCDF4) file with multiple fields
    check_for_grid_info: boolean; Search to see if file has grid info and use this
            to replace grid type specified if found
    grid_type: string; the code for this grid type
    **grid_kwargs: dictionary; key word arguments specifying parameters of the grid
    Returns:
    A field of object or an object of a field subclass

    Uses the getFileHelper pseudo-factory function to get the appropriate IOHelper subclass
    to load this file type and then uses this to loads the file; passing in any grid_kwargs
    supplied. Uses makeField factory function to create a field of the requested type with
    the loaded data and returns this object. Use a list to retrieve any grid information
    found by the loading function and add it to the field.
    """

    grid_info=[]
    raw_field = iohelper.getFileHelper(file_type).load_field(filename,unmask,timeslice,fieldname,
                                                             check_for_grid_info,grid_info,
                                                             grid_type,**grid_kwargs)
    if len(grid_info) == 0:
        return fld.makeField(raw_field,field_type,grid_type,**grid_kwargs)
    else:
        return fld.makeField(raw_field,field_type,grid_info[0])

def write_field(filename,field,file_type,griddescfile=None,
                fieldname=None):
    """Writes the given field object to the given file type.

    Arguments:
    filename: string; the full path of the target file
    field: field object; the field object containing the data to be written
    file_type: string; the code of the type of file that is to be written
    griddescfile (optional): string; full path to the grid description metadata to add to
            file written out (if possible). Nothing is added if this is set to
            None
    fieldname(optional): string; fieldname to use in a (netCDF4) file

    Uses the getFileHeper pseudo-factory function to get the appropriate IOHelper subclass
    to write this file type and then uses this to write the field object.
    """

    iohelper.getFileHelper(file_type).write_field(filename,field,griddescfile=griddescfile,
                                                  fieldname=fieldname)

def get_file_extension(filename):
    """Return the extension of a given filename"""
    return os.path.splitext(filename)[1]
