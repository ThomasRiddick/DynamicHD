import os.path as path
import numpy as np
from Dynamic_HD_Scripts.interface.fortran_interface import f2py_manager
from Dynamic_HD_Scripts.context import fortran_project_source_path,fortran_project_object_path,fortran_project_include_path

def cross_grid_mapper_latlon_to_icon(cell_neighbors,
                                     pixel_center_lats,
                                     pixel_center_lons,
                                     cell_vertices_lats,
                                     cell_vertices_lons):
    additional_fortran_filenames =  \
        ["Dynamic_HD_Fortran_Code_src_base_area_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_coords_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_doubly_linked_list_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_doubly_linked_list_link_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_field_section_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_precision_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_subfield_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_algorithms_cross_grid_mapper.f90.o",
         "Dynamic_HD_Fortran_Code_src_algorithms_map_non_coincident_grids_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_algorithms_cotat_parameters_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_unstructured_grid_mod.f90.o"]
    additional_fortran_filepaths = \
        [path.join("/Users/thomasriddick/Documents/workspace/worktrees/feature/improved-run-and-build-structure/build/libdyhd_fortran.a.p",filename)
        for filename in additional_fortran_filenames]
    f2py_mngr = f2py_manager.f2py_manager(
        path.join(fortran_project_source_path,"drivers",
                  "cross_grid_mapper_driver_mod.f90"),
                  func_name="cross_grid_mapper_latlon_to_icon_f2py_wrapper",
                  additional_fortran_files=additional_fortran_filepaths,
                  include_path=fortran_project_include_path)
    output_cell_numbers = f2py_mngr.\
        run_current_function_or_subroutine(cell_neighbors.\
                                           astype(np.int32,order='F'),
                                           pixel_center_lats.\
                                           astype(np.float64),
                                           pixel_center_lons.\
                                           astype(np.float64),
                                           cell_vertices_lats.\
                                           astype(np.float64,order='F'),
                                           cell_vertices_lons.\
                                           astype(np.float64,order='F'),
                                           len(pixel_center_lats),
                                           len(pixel_center_lons),
                                           cell_neighbors.shape[0])
    return output_cell_numbers
