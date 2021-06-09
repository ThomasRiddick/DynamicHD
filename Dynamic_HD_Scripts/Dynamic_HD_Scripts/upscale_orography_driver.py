'''
Created on Apr 18, 2017

@author: thomasriddick
'''

from Dynamic_HD_Scripts import dynamic_hd
from Dynamic_HD_Scripts.dynamic_hd import get_file_extension
import numpy as np
from Dynamic_HD_Scripts import field
from Dynamic_HD_Scripts.interface.cpp_interface.libs \
    import upscale_orography_wrapper
import configparser
import gc
from Dynamic_HD_Scripts import iodriver
from Dynamic_HD_Scripts import coordinate_scaling_utilities

def drive_orography_upscaling(input_fine_orography_file,output_course_orography_file,
                              landsea_file=None,true_sinks_file=None,
                              upscaling_parameters_filename=None,
                              fine_grid_type='LatLong10min',course_grid_type='HD',
                              input_orography_field_name=None,flip_landsea=False,
                              rotate_landsea=False,flip_true_sinks=False,rotate_true_sinks=False,
                              fine_grid_kwargs={},**course_grid_kwargs):
    """Drive the C++ sink filling code base to make a tarasov-like orography upscaling

    Arguments:
    input_fine_orography_file: string; full path to input fine orography file
    output_course_orography_file: string; full path of target output course orography file
    landsea_file: string; full path to input fine landsea mask file (optional)
    true_sinks_file: string; full path to input fine true sinks file (optional)
    upscaling_parameters_filename: string; full path to the orography upscaling parameter
        file (optional)
    fine_grid_type: string; code for the fine grid type to be upscaled from  (optional)
    course_grid_type: string; code for the course grid type to be upscaled to (optional)
    input_orography_field_name: string; name of field in the input orography file (optional)
    flip_landsea: bool; flip the input landsea mask upside down
    rotate_landsea: bool; rotate the input landsea mask by 180 degrees along the horizontal axis
    flip_true_sinks: bool; flip the input true sinks field upside down
    rotate_true_sinks: bool; rotate the input true sinks field by 180 degrees along the
        horizontal axis
    fine_grid_kwargs:  keyword dictionary; the parameter of the fine grid to upscale
        from (if required)
    **course_grid_kwargs: keyword dictionary; the parameters of the course grid to upscale
        to (if required)
    Returns: Nothing.
    """

    if upscaling_parameters_filename:
        config = read_and_validate_config(upscaling_parameters_filename)
        method = config.getint("orography_upscaling_parameters","method")
        add_slope_in = config.getboolean("orography_upscaling_parameters","add_slope_in")
        epsilon_in = config.getfloat("orography_upscaling_parameters","epsilon_in")
        tarasov_separation_threshold_for_returning_to_same_edge_in =\
            config.getint("orography_upscaling_parameters",
                          "tarasov_separation_threshold_for_returning_to_same_edge_in")
        tarasov_min_path_length_in = config.getfloat("orography_upscaling_parameters",
                                                     "tarasov_min_path_length_in")
        tarasov_include_corners_in_same_edge_criteria_in = \
            config.getboolean("orography_upscaling_parameters",
                              "tarasov_include_corners_in_same_edge_criteria_in")
    else:
        #use defaults
        method = 1
        add_slope_in = False
        epsilon_in = 0.1
        tarasov_separation_threshold_for_returning_to_same_edge_in = 5
        tarasov_min_path_length_in = 2.0
        tarasov_include_corners_in_same_edge_criteria_in = False
    output_orography = field.makeEmptyField(field_type='Orography',dtype=np.float64,
                                            grid_type=course_grid_type,**course_grid_kwargs)
    input_orography = dynamic_hd.load_field(input_fine_orography_file,
                                            file_type=get_file_extension(input_fine_orography_file),
                                            field_type='Orography', unmask=True,
                                            fieldname=input_orography_field_name,
                                            grid_type=fine_grid_type,**fine_grid_kwargs)
    if landsea_file:
        landsea_mask = dynamic_hd.load_field(landsea_file,
                                             file_type=get_file_extension(landsea_file),
                                             field_type='Generic',unmask=True,
                                             grid_type=fine_grid_type,**fine_grid_kwargs)
        if flip_landsea:
            landsea_mask.flip_data_ud()
        if rotate_landsea:
            landsea_mask.rotate_field_by_a_hundred_and_eighty_degrees()
    else:
        landsea_mask = field.makeEmptyField(field_type='Generic',dtype=np.int32,
                                            grid_type=fine_grid_type,**fine_grid_kwargs)
    if true_sinks_file:
        true_sinks = dynamic_hd.load_field(true_sinks_file,
                                           file_type=get_file_extension(true_sinks_file),
                                           field_type='Generic',unmask=True,
                                           grid_type=fine_grid_type,**fine_grid_kwargs)
        if flip_true_sinks:
            true_sinks.flip_data_ud()
        if rotate_true_sinks:
            true_sinks.rotate_field_by_a_hundred_and_eighty_degrees()
    else:
        true_sinks = field.makeEmptyField(field_type='Generic',dtype=np.int32,
                                          grid_type=fine_grid_type,**fine_grid_kwargs)
    if not np.issubdtype(input_orography.get_data().dtype,np.float64()):
        input_orography.change_dtype(np.float64)
        #Make sure old data type array is flushed out of memory immediately
        gc.collect()
    upscale_orography_wrapper.upscale_orography(orography_in=input_orography.get_data(),
                                                orography_out=output_orography.get_data(),
                                                method=method,landsea_in=landsea_mask.get_data(),
                                                true_sinks_in=true_sinks.get_data(),
                                                add_slope_in=add_slope_in, epsilon_in=epsilon_in,
                                                tarasov_separation_threshold_for_returning_to_same_edge_in=\
                                                tarasov_separation_threshold_for_returning_to_same_edge_in,
                                                tarasov_min_path_length_in=tarasov_min_path_length_in,
                                                tarasov_include_corners_in_same_edge_criteria_in=\
                                                tarasov_include_corners_in_same_edge_criteria_in)
    dynamic_hd.write_field(output_course_orography_file,output_orography,
                           file_type=get_file_extension(output_course_orography_file))

def advanced_drive_orography_upscaling(input_fine_orography_file,output_course_orography_file,
                                       input_orography_fieldname,output_course_orography_fieldname,
                                       landsea_file=None,
                                       true_sinks_file=None,landsea_fieldname=None,
                                       true_sinks_fieldname=None,
                                       upscaling_parameters_filename=None,
                                       scaling_factor=3):
    if upscaling_parameters_filename:
        config = read_and_validate_config(upscaling_parameters_filename)
        method = config.getint("orography_upscaling_parameters","method")
        add_slope_in = config.getboolean("orography_upscaling_parameters","add_slope_in")
        epsilon_in = config.getfloat("orography_upscaling_parameters","epsilon_in")
        tarasov_separation_threshold_for_returning_to_same_edge_in =\
            config.getint("orography_upscaling_parameters",
                          "tarasov_separation_threshold_for_returning_to_same_edge_in")
        tarasov_min_path_length_in = config.getfloat("orography_upscaling_parameters",
                                                     "tarasov_min_path_length_in")
        tarasov_include_corners_in_same_edge_criteria_in = \
            config.getboolean("orography_upscaling_parameters",
                              "tarasov_include_corners_in_same_edge_criteria_in")
    else:
        #use defaults
        method = 1
        add_slope_in = False
        epsilon_in = 0.1
        tarasov_separation_threshold_for_returning_to_same_edge_in = 5
        tarasov_min_path_length_in = 2.0
        tarasov_include_corners_in_same_edge_criteria_in = False
    input_orography = iodriver.advanced_field_loader(input_fine_orography_file,
                                                     field_type='Orography',
                                                     fieldname=input_orography_fieldname)
    nlat_fine,nlon_fine = input_orography.get_grid_dimensions()
    lat_pts_fine,lon_pts_fine = input_orography.get_grid_coordinates()
    nlat_course,nlon_course,lat_pts_course,lon_pts_course = \
        coordinate_scaling_utilities.generate_course_coords(nlat_fine,nlon_fine,
                                                            lat_pts_fine,lon_pts_fine,
                                                            scaling_factor)
    output_orography = field.makeEmptyField(field_type='Orography',dtype=np.float64,
                                            grid_type='LatLong',nlat=nlat_course,
                                            nlong=nlon_course)
    output_orography.set_grid_coordinates([lat_pts_course,lon_pts_course])
    if landsea_file:
        landsea_mask = iodriver.advanced_field_loader(landsea_file,
                                                      field_type='Generic',
                                                      fieldname=landsea_fieldname)
    else:
        landsea_mask = field.makeEmptyField(field_type='Generic',dtype=np.int32,
                                            grid_type='LatLong',nlat=nlat_fine,
                                            nlong=nlon_fine)
    if true_sinks_file:
        true_sinks = iodriver.advanced_field_loader(true_sinks_file,
                                                    field_type='Generic',
                                                    fieldname=true_sinks_fieldname)
    else:
        true_sinks = field.makeEmptyField(field_type='Generic',dtype=np.int32,
                                          grid_type='LatLong',nlat=nlat_fine,
                                          nlong=nlon_fine)
    if not input_orography.get_data().dtype is np.float64():
        input_orography.change_dtype(np.float64)
        #Make sure old data type array is flushed out of memory immediately
        gc.collect()
    upscale_orography_wrapper.upscale_orography(orography_in=input_orography.get_data(),
                                                orography_out=output_orography.get_data(),
                                                method=method,landsea_in=landsea_mask.get_data(),
                                                true_sinks_in=true_sinks.get_data(),
                                                add_slope_in=add_slope_in, epsilon_in=epsilon_in,
                                                tarasov_separation_threshold_for_returning_to_same_edge_in=\
                                                tarasov_separation_threshold_for_returning_to_same_edge_in,
                                                tarasov_min_path_length_in=tarasov_min_path_length_in,
                                                tarasov_include_corners_in_same_edge_criteria_in=\
                                                tarasov_include_corners_in_same_edge_criteria_in)
    iodriver.advanced_field_writer(output_course_orography_file,output_orography)

def read_and_validate_config(upscaling_parameters_filename):
    """Reads and checks format of config file

    Arguments:
    upscaling_parameters_filename: string; full path to the orography upscaling parameter
        file (optional)
    Returns: ConfigParser object; the read and checked configuration
    """

    config = configparser.ConfigParser()
    print("Read orography upscaling options from file {0}".\
        format(upscaling_parameters_filename))
    config.read(upscaling_parameters_filename)
    valid_config = True
    valid_config = valid_config \
        if config.has_section("orography_upscaling_parameters") else False
    valid_config = valid_config \
        if config.has_option("orography_upscaling_parameters","method") else False
    valid_config = valid_config \
        if config.has_option("orography_upscaling_parameters","add_slope_in") else False
    valid_config = valid_config \
        if config.has_option("orography_upscaling_parameters","epsilon_in") else False
    valid_config = valid_config \
        if config.has_option("orography_upscaling_parameters",
                             "tarasov_separation_threshold_for_returning_to_same_edge_in") else False
    valid_config = valid_config \
        if config.has_option("orography_upscaling_parameters",
                             "tarasov_min_path_length_in") else False
    valid_config = valid_config \
        if config.has_option("orography_upscaling_parameters",
                             "tarasov_include_corners_in_same_edge_criteria_in") else False
    if not valid_config:
        raise RuntimeError("Invalid orography upscaling parameter configuration file supplied")
    return config
