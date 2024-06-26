'''
Creates a connected land-sea mask using the C++ sink filling code from
a given set of ocean seed points and possibly unconnected input land-sea
mask
Created on May 28, 2016

@author: thomasriddick
'''

import numpy as np
import re
from Dynamic_HD_Scripts.base import field
import create_connected_lsmask_wrapper as cc_lsmask_wrapper #@UnresolvedImport
from Dynamic_HD_Scripts.base import iodriver

def advanced_connected_lsmask_creation_driver(input_lsmask_filename,
                                              output_lsmask_filename,
                                              input_lsmask_fieldname,
                                              output_lsmask_fieldname,
                                              input_ls_seed_points_filename=None,
                                              input_ls_seed_points_fieldname=None,
                                              input_ls_seed_points_list_filename = None,
                                              use_diagonals_in=True,
                                              rotate_seeds_about_polar_axis=False,
                                              flip_seeds_ud=False,
                                              adjust_lsmask_orientation=True,
                                              grid_type="LatLong10min",
                                              **grid_kwargs):
    lsmask = iodriver.advanced_field_loader(input_lsmask_filename,
                                            field_type='Generic',
                                            fieldname=input_lsmask_fieldname,
                                            adjust_orientation=adjust_lsmask_orientation)
    if input_ls_seed_points_filename:
        input_ls_seedpts = iodriver.advanced_field_loader(input_ls_seed_points_filename,
                                                          field_type='Generic',
                                                          fieldname=\
                                                          input_ls_seed_points_fieldname,
                                                          adjust_orientation=False)
    else:
        input_ls_seedpts = field.makeEmptyField('Generic',np.int32,
                                                grid_type=grid_type,**grid_kwargs)
    if input_ls_seed_points_list_filename:
        points_list = []
        print("Reading input from {0}".format(input_ls_seed_points_list_filename))
        comment_line_pattern = re.compile(r"^ *#.*$")
        with open(input_ls_seed_points_list_filename) as f:
            if "LatLong" not in f.readline().strip():
                raise RuntimeError("List of landsea points being loaded is not for correct grid-type")
            for line in f:
                if comment_line_pattern.match(line):
                    continue
                points_list.append(tuple(int(coord) for coord in line.strip().split(",")))
        input_ls_seedpts.flag_listed_points(points_list)
    if rotate_seeds_about_polar_axis:
        input_ls_seedpts.rotate_field_by_a_hundred_and_eighty_degrees()
    if flip_seeds_ud:
        input_ls_seedpts.flip_data_ud()
    if input_ls_seedpts.get_grid().has_orientation_information():
        input_ls_seedpts.orient_data()
    lsmask.change_dtype(np.int32)
    cc_lsmask_wrapper.create_connected_ls_mask(lsmask.get_data(),
                                               input_ls_seedpts.get_data().astype(dtype=np.int32,
                                                                                  order='C',
                                                                                  copy=False),
                                               use_diagonals_in)
    iodriver.advanced_field_writer(output_lsmask_filename,lsmask,
                                   fieldname=output_lsmask_fieldname)

def drive_connected_lsmask_creation(input_lsmask_filename,
                                    output_lsmask_filename,
                                    input_ls_seed_points_filename=None,
                                    input_ls_seed_points_list_filename = None,
                                    use_diagonals_in=True,
                                    rotate_seeds_about_polar_axis=False,
                                    flip_seeds_ud=False,
                                    flip_input_mask_ud=False,
                                    grid_type='HD',**grid_kwargs):
    """Drives the creation of a fully connected land-sea mask

    Argument:
    input_lsmask_filename: string; the full path of the input land-sea mask which
        may contain disconnected sea points
    output_lsmask_filename: string; the full path of the output land-sea mask without
        any disconnected sea point
    input_ls_seed_points_filename(optional): string; the full path of the file with an array with
        at least one point in the center of each ocean/sea to be included
    input_ls_seed_points_list_filename(optional): string; the full path of the file with a
        list of at least one point in the center of each ocean/sea to be included - these
        will be added to the contents of input_ls_seed_points_filename, normally only one of the
        two arguments would be used to initialise the seed points
    use_diagonals_in: boolean; if true count a diagonal connection as being connected
    flip_input_mask_ud: boolean; if true flip the input mask up down before processing
    rotate_seeds_about_polar_axis:  if true then rotate the seeds about the polar axis by 180
        degrees, i.e. shift the position of the zero longitude by 180 degrees when reading the
        seed points
    flip_seeds_upside_down: boolean; if true then flip the seeds upside down (i.e. about
                            the equator)
    grid_type: string; the code for this grid type
    **grid_kwargs:  dictionary; key word arguments specifying parameters of the grid
    Returns: Nothing

    Creates a land-sea masked based on the supplied land-sea mask with any sea points not connected
    to the supplied ocean seeds points (by other sea points) change to land points
    """

    lsmask = iodriver.load_field(input_lsmask_filename,
                                 file_type=iodriver.\
                                 get_file_extension(input_lsmask_filename),
                                 field_type='Generic',
                                 grid_type=grid_type,**grid_kwargs)
    if flip_input_mask_ud:
        lsmask.flip_data_ud()
    if input_ls_seed_points_filename:
        input_ls_seedpts = iodriver.load_field(input_ls_seed_points_filename,
                                               file_type=iodriver.\
                                               get_file_extension(input_ls_seed_points_filename),
                                               field_type='Generic',
                                               grid_type=grid_type,**grid_kwargs)
    else:
        input_ls_seedpts = field.makeEmptyField('Generic',np.int32,grid_type,**grid_kwargs)
    if input_ls_seed_points_list_filename:
        points_list = []
        print("Reading input from {0}".format(input_ls_seed_points_list_filename))
        comment_line_pattern = re.compile(r"^ *#.*$")
        with open(input_ls_seed_points_list_filename) as f:
            if f.readline().strip() != grid_type:
                raise RuntimeError("List of landsea points being loaded is not for correct grid-type")
            for line in f:
                if comment_line_pattern.match(line):
                    continue
                points_list.append(tuple(int(coord) for coord in line.strip().split(",")))
        input_ls_seedpts.flag_listed_points(points_list)
    if rotate_seeds_about_polar_axis:
        input_ls_seedpts.rotate_field_by_a_hundred_and_eighty_degrees()
    if flip_seeds_ud:
        input_ls_seedpts.flip_data_ud()
    lsmask.change_dtype(np.int32)
    cc_lsmask_wrapper.create_connected_ls_mask(lsmask.get_data(),
                                               input_ls_seedpts.get_data().astype(dtype=np.int32,
                                                                                  order='C',
                                                                                  copy=False),
                                               use_diagonals_in)
    iodriver.write_field(output_lsmask_filename,lsmask,
                         file_type=iodriver.\
                         get_file_extension(output_lsmask_filename))
