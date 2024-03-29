'''
Created on Feb 12, 2018

@author: thomasriddick
'''
import os.path as path
from Dynamic_HD_Scripts.interface.fortran_interface import f2py_manager
from Dynamic_HD_Scripts.context import fortran_project_source_path,fortran_project_object_path,fortran_project_include_path

def run_flow(input_fine_river_directions,input_fine_total_cumulative_flow,
             cotat_parameters_filepath,coarse_grid_type,
             **coarse_grid_kwargs):

    additional_fortran_filenames = ["base/area_mod.o",
                                    "base/coords_mod.o",
                                    "algorithms/cotat_parameters_mod.o",
                                    "algorithms/flow.o",
                                    "base/doubly_linked_list_mod.o",
                                    "base/doubly_linked_list_link_mod.o",
                                    "base/field_section_mod.o",
                                    "base/precision_mod.o",
                                    "base/subfield_mod.o"]
    additional_fortran_filepaths = [path.join(fortran_project_object_path,filename) for filename in\
                                    additional_fortran_filenames]
    f2py_mngr = f2py_manager.f2py_manager(path.join(fortran_project_source_path,
                                                    "drivers",
                                                    "flow_driver_mod.f90"),
                                          func_name="flow_latlon_f2py_wrapper",
                                          additional_fortran_files=additional_fortran_filepaths,
                                          include_path=fortran_project_include_path)

    output_coarse_river_directions_latlon_indices = f2py_mngr.\
        run_current_function_or_subroutine(),
