'''
Drive the Fortran complex loop breaking code via f2py. The top level routine is
loop_breaker_driver; this then calls run_loop_breaker.
Created on Oct 30, 2016

@author: thomasriddick
'''

from . import f2py_manager
from . import field
import os.path as path
from context import fortran_project_source_path,fortran_project_object_path,fortran_project_include_path
import numpy as np
from . import dynamic_hd
import re

def run_loop_breaker(course_rdirs,course_cumulative_flow,course_catchments,fine_rdirs_field,
                     fine_cumulative_flow_field,loop_nums_list,course_grid_type,**course_grid_kwargs):
    """Run the Fortarn complex loop breaking code via f2py

    Arguments:
    course_rdirs: 2d ndarray; the course river directions to remove the loop from
    course_cumulative_flow: 2d ndarray; the course cumulative flow derived from the course
        river directions
    course_catchments: 2d ndarray; the set of catchments generated from the course river flow
        directions
    fine_rdirs_field: 2d ndarray; the original fine river directions the course river direction
        were upscaled from
    fine_cumulative_flow_field: 2d ndarray; the fine cumulative flow derived from the original
        fine river directions
    loop_nums_list: list of integers; the catchment numbers of catchments that have loops in them
        that need to be removed
    course_grid_type: string; code of the grid type of the course grid
    **course_grid_kwargs: keyword dictionary; key word arguments specifying parameters of the course
        grid
    Returns: the 2d ndarray of course river directions with the specified loops removed

    This will compile and run code that will remove the specified set of more complex loops (that
    may contain three or more cells).
    """

    additional_fortran_filenames = ["coords_mod.o","doubly_linked_list_mod.o","break_loops_mod.o",
                                    "doubly_linked_list_link_mod.o","field_section_mod.o",
                                    "loop_breaker_mod.o","unstructured_grid_mod.o",
                                    "map_non_coincident_grids_mod.o","subfield_mod.o"]
    additional_fortran_filepaths = [path.join(fortran_project_object_path,filename) for filename in\
                                    additional_fortran_filenames]
    f2py_mngr = f2py_manager.f2py_manager(path.join(fortran_project_source_path,"break_loops_driver_mod.f90"),
                                          func_name="break_loops_latlon_f2py_wrapper",
                                          additional_fortran_files=additional_fortran_filepaths,
                                          include_path=fortran_project_include_path)
    loop_nums_list_array = np.asarray(loop_nums_list)
    if len(loop_nums_list) == 0:
        print("List of loops to remove is empty!")
        return course_rdirs
    course_rdirs_field_raw = course_rdirs.get_data().astype(np.int32,order='F')
    f2py_mngr.run_current_function_or_subroutine(course_rdirs_field_raw,
                                                 course_cumulative_flow.get_data().astype(np.int64,order='F'),
                                                 course_catchments.get_data().astype(np.int64,order='F'),
                                                 fine_rdirs_field.get_data().astype(np.int64,order='F'),
                                                 fine_cumulative_flow_field.get_data().astype(np.int64,order='F'),
                                                 loop_nums_list_array.astype(np.int64,order='F'))
    course_rdirs_field = field.RiverDirections(course_rdirs_field_raw.astype(np.float64),
                                               grid=course_rdirs.get_grid())
    return course_rdirs_field

def loop_breaker_driver(input_course_rdirs_filepath,input_course_cumulative_flow_filepath,
                        input_course_catchments_filepath,input_fine_rdirs_filepath,
                        input_fine_cumulative_flow_filepath,output_updated_course_rdirs_filepath,
                        loop_nums_list_filepath,course_grid_type,fine_grid_type,
                        fine_grid_kwargs={},**course_grid_kwargs):
    """Drive the FORTRAN code to remove more complex loops from a field of river directions

    Arguments:
    input_course_rdirs_filepath: string, full path to input course river directions to remove
        loops from
    input_course_cumulative_flow_filepath: string, full path to the input course cumulative
        flow file
    input_course_catchments_filepath: string, full path to the course input catchments file
    input_fine_rdirs_filepath: string, full path to the fine input river directions the course
        input river directions were upscaled from
    input_fine_cumulative_flow_filepath: string, full path to the catchments generated from the
        fine input river directions
    output_updated_course_rdirs_filepath: string, full path to write the course river direction with
        the specified loops removed too
    loop_nums_list_filepath: string, full path to the file contain the catchment numbers of the
        loops to remove, one per line, see code below for correct format for the first line
    course_grid_type: string; code for the grid type of the course grid
    fine_grid_type: string; code for the grid type of the fine grid
    fine_grid_kwargs: keyword dictionary; key word arguments specifying parameters of the fine
        grid (if required)
    **course_grid_kwarg: keyword dictionary; key word arguments specifying parameters of the course
        grid (if required)
    Returns: nothing
    """

    input_course_rdirs_field = dynamic_hd.load_field(input_course_rdirs_filepath,
                                                     file_type=dynamic_hd.\
                                                     get_file_extension(input_course_rdirs_filepath),
                                                     field_type='RiverDirections',
                                                     grid_type=course_grid_type,**course_grid_kwargs)
    course_cumulative_flow_field =\
        dynamic_hd.load_field(input_course_cumulative_flow_filepath,
                              file_type=dynamic_hd.\
                              get_file_extension(input_course_cumulative_flow_filepath),
                              field_type='CumulativeFlow',
                              grid_type=course_grid_type,**course_grid_kwargs)
    course_catchments_field =\
        dynamic_hd.load_field(input_course_catchments_filepath,
                              file_type=dynamic_hd.\
                              get_file_extension(input_course_catchments_filepath),
                              field_type='Generic',
                              grid_type=course_grid_type,**course_grid_kwargs)
    fine_rdirs_field = dynamic_hd.load_field(input_fine_rdirs_filepath,
                                             file_type=dynamic_hd.\
                                             get_file_extension(input_fine_rdirs_filepath),
                                             field_type='RiverDirections',
                                             grid_type=fine_grid_type,**fine_grid_kwargs)
    fine_cumulative_flow_field =\
        dynamic_hd.load_field(input_fine_cumulative_flow_filepath,
                              file_type=dynamic_hd.\
                              get_file_extension(input_fine_cumulative_flow_filepath),
                              field_type='CumulativeFlow',
                              grid_type=fine_grid_type,**fine_grid_kwargs)
    loop_nums_list = []
    first_line_pattern = re.compile(r"^Loops found in catchments:$")
    with open(loop_nums_list_filepath,'r') as f:
        if not first_line_pattern.match(f.readline().strip()):
            raise RuntimeError("Format of the file with list of catchments to remove loops from"
                               " is invalid")
        for line in f:
            loop_nums_list.append(int(line.strip()))
    print('Removing loops from catchments: ' + ", ".join(str(value) for value in loop_nums_list))
    output_course_rdirs_field = run_loop_breaker(input_course_rdirs_field,course_cumulative_flow_field,
                                                 course_catchments_field,fine_rdirs_field,
                                                 fine_cumulative_flow_field,loop_nums_list,
                                                 course_grid_type,**course_grid_kwargs)
    dynamic_hd.write_field(output_updated_course_rdirs_filepath, output_course_rdirs_field,
                           file_type=dynamic_hd.\
                           get_file_extension(output_updated_course_rdirs_filepath))
