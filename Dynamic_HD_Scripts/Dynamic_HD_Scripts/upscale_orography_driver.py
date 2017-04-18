'''
Created on Apr 18, 2017

@author: thomasriddick
'''

import dynamic_hd
import dynamic_hd.get_file_extension as get_file_extension
import numpy as np
import field
import libs.upscale_orography_wrapper as upscale_orography_wrapper  #@UnresolvedImport

def drive_orography_upscaling(input_fine_orography_file,output_course_orography_file,landsea_file,
                              true_sinks_file,fine_grid_type='LatLong10min',course_grid_type='HD',
                              input_orography_field_name=None,flip_landsea=False,
                              rotate_landsea=False,flip_true_sinks=False,rotate_true_sinks=False,
                              fine_grid_kwargs={},**course_grid_kwargs):
    """Drive the C++ sink filling code base to make a tarasov-like orography upscaling
    
    Arguments:
    
    Returns: Nothing.
    """

    #Put in ini file and complete doc
    #method = 1
    #add_slope_in = 0
    #epsilon_in = 0.1
    #tarasov_separation_threshold_for_returning_to_same_edge_in = 5
    #tarasov_min_path_length_in = 2.0
    #tarasov_include_corners_in_same_edge_criteria_in = 0
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
                                            grid_type=course_grid_type,**course_grid_kwargs) 
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
                                          grid_type=course_grid_type,**course_grid_kwargs) 
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