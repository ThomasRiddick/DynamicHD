'''
Created on Feb 12, 2018

@author: thomasriddick
'''
from Dynamic_HD_Scripts import f2py_manager
import os.path as path
from Dynamic_HD_Scripts.context import fortran_project_source_path,fortran_project_object_path,fortran_project_include_path

def run_flow(input_fine_river_directions,input_fine_total_cumulative_flow,
             cotat_parameters_filepath,course_grid_type,
             **course_grid_kwargs):

    additional_fortran_filenames = ["area_mod.o","coords_mod.o","cotat_parameters_mod.o","flow.o",
                                     "doubly_linked_list_mod.o","doubly_linked_list_link_mod.o",
                                     "field_section_mod.o","precision_mod.o","subfield_mod.o"]
    additional_fortran_filepaths = [path.join(fortran_project_object_path,filename) for filename in\
                                    additional_fortran_filenames]
    f2py_mngr = f2py_manager.f2py_manager(path.join(fortran_project_source_path,"flow_driver_mod.f90"),
                                          func_name="flow_latlon_f2py_wrapper",
                                          additional_fortran_files=additional_fortran_filepaths,
                                          include_path=fortran_project_include_path)

    output_course_river_directions_latlon_indices = f2py_mngr.\
        run_current_function_or_subroutine(),
