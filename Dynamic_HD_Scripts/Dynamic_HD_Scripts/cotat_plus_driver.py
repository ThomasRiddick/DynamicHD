'''
Drives the FORTRAN COTAT plus algorithm via f2py. The main function is cotat_plus_driver; this then
calls the function run_cotat_plus. 
Created on Oct 18, 2016

@author: thomasriddick
'''

import f2py_manager 
import os.path as path
from context import fortran_project_source_path,fortran_project_object_path,fortran_project_include_path
import field
import numpy as np
import grid
import dynamic_hd

def run_cotat_plus(fine_rdirs_field,fine_total_cumulative_flow_field,cotat_plus_parameters_filepath,
                   course_grid_type,**course_grid_kwargs):
    """Run the cotat plus fortran code using f2py for a lat-lon field
    
    Arguments:
    fine_rdirs_field: 2d ndarray; the fine river directions to be upscaled in 1-9 keypad format
    fine_total_cumulative_flow_field: 2d ndarray; the fine total cumulative flow (created from
        the fine_rdirs_field) to be used in upscaling
    cotat_plus_parameter_filepath: string; the file path containing the namelist with the parameters
        for the cotat plus upscaling algorithm
    course_grid_type: string; code for the course grid type to be upscaled to
    **course_grid_kwargs(optional): keyword dictionary; the parameter of the course grid to 
        upscale to (if required)
    Return: 2d ndarray; the upscaled river direction on the course grid
    
    Compiles and runs the COTAT plus algorithm in Fortran using f2py for a lat-lon field
    """
    additional_fortran_filenames = ["area_mod.o","coords_mod.o","cotat_parameters_mod.o","cotat_plus.o",
                                     "doubly_linked_list_mod.o","doubly_linked_list_link_mod.o",
                                     "field_section_mod.o","precision_mod.o","subfield_mod.o"]
    additional_fortran_filepaths = [path.join(fortran_project_object_path,filename) for filename in\
                                    additional_fortran_filenames]
    f2py_mngr = f2py_manager.f2py_manager(path.join(fortran_project_source_path,"cotat_plus_driver_mod.f90"),
                                          func_name="cotat_plus_latlon_f2py_wrapper",
                                          additional_fortran_files=additional_fortran_filepaths,
                                          include_path=fortran_project_include_path)
    course_grid = grid.makeGrid(course_grid_type,**course_grid_kwargs)
    course_rdirs_field_raw = f2py_mngr.\
        run_current_function_or_subroutine(fine_rdirs_field.get_data().astype(np.int64,order='F'),
                                           fine_total_cumulative_flow_field.get_data().astype(np.int64,order='F'),
                                           cotat_plus_parameters_filepath,*course_grid.get_grid_dimensions())
    course_rdirs_field = field.makeField(course_rdirs_field_raw.astype(np.float64),'RiverDirections',course_grid_type,
                                         **course_grid_kwargs)
    return course_rdirs_field

def cotat_plus_driver(input_fine_rdirs_filepath,input_fine_total_cumulative_flow_path,
                      output_course_rdirs_filepath,cotat_plus_parameters_filepath,
                      fine_grid_type,fine_grid_kwargs={},course_grid_type='HD',**course_grid_kwargs):
    """Top level driver for the cotat plus algorithm
    
    Arguments:
    input_fine_rdirs_filepath: string; path to the file with fine river directions to upscale
    input_fine_total_cumulative_flow_path: string; path to the file with the fine scale cumulative
        flow from the fine river directions
    output_course_rdirs_filepath: string; path to the file to write the upscaled course river directions to
    cotat_plus_parameters_filepath: string; the file path containing the namelist with the parameters
        for the cotat plus upscaling algorithm
    fine_grid_type: string; code for the fine grid type to upscale from
    **fine_grid_kwargs(optional): keyword dictionary; the parameter of the fine grid to 
        upscale from
    course_grid_type: string; code for the course grid type to be upscaled to
    **course_grid_kwargs(optional): keyword dictionary; the parameter of the course grid to 
        upscale to (if required)
    Returns: Nothing
    
    Compiles and runs the COTAT plus algorithm in Fortran using f2py for a lat-lon field. Writes
    output specified filename.
    """

    fine_rdirs_field = dynamic_hd.load_field(input_fine_rdirs_filepath, 
                                             file_type=dynamic_hd.\
                                             get_file_extension(input_fine_rdirs_filepath), 
                                             field_type='RiverDirections',
                                             grid_type=fine_grid_type,**fine_grid_kwargs)
    fine_total_cumulative_flow_field =\
        dynamic_hd.load_field(input_fine_total_cumulative_flow_path,
                              file_type=dynamic_hd.\
                              get_file_extension(input_fine_total_cumulative_flow_path),
                              field_type='CumulativeFlow',
                              grid_type=fine_grid_type,**fine_grid_kwargs)
    course_rdirs_field = run_cotat_plus(fine_rdirs_field, fine_total_cumulative_flow_field, 
                                        cotat_plus_parameters_filepath,course_grid_type,
                                        **course_grid_kwargs)
    dynamic_hd.write_field(output_course_rdirs_filepath, course_rdirs_field,
                           file_type=dynamic_hd.\
                           get_file_extension(output_course_rdirs_filepath))