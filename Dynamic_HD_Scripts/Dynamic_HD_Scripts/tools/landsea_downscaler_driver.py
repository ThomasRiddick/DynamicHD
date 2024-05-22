import os.path as path
import numpy as np
from Dynamic_HD_Scripts.utilities import coordinate_scaling_utilities
from Dynamic_HD_Scripts.context import fortran_project_source_path,fortran_project_object_path,fortran_project_include_path
from Dynamic_HD_Scripts.interface.fortran_interface \
    import f2py_manager as f2py_mg

def landsea_downscaler_icosohedral_cell_latlon_pixel(input_coarse_landsea_mask,
                                                     coarse_to_fine_cell_numbers_mapping):
    additional_fortran_filenames =  \
        ["Dynamic_HD_Fortran_Code_src_base_area_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_coords_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_doubly_linked_list_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_doubly_linked_list_link_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_field_section_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_precision_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_subfield_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_algorithms_icon_to_latlon_landsea_downscaler.f90.o",
         "Dynamic_HD_Fortran_Code_src_algorithms_map_non_coincident_grids_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_algorithms_cotat_parameters_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_unstructured_grid_mod.f90.o"]
    additional_fortran_filepaths = \
        [path.join("/Users/thomasriddick/Documents/workspace/worktrees/"
                   "feature/improved-run-and-build-structure/build/libdyhd_fortran.a.p",filename)
        for filename in additional_fortran_filenames]
    f2py_mngr = f2py_mg.f2py_manager(
        path.join(fortran_project_source_path,"drivers",
                  "landsea_downscaler_driver_mod.f90"),
                  func_name="landsea_downscaler_icosohedral_cell_latlon_pixel_f2py_wrapper",
                  additional_fortran_files=additional_fortran_filepaths,
                  include_path=fortran_project_include_path)
    fine_landsea_mask = \
        f2py_mngr.run_current_function_or_subroutine(
            np.asfortranarray(input_coarse_landsea_mask),
            np.asfortranarray(coarse_to_fine_cell_numbers_mapping),
            *coarse_to_fine_cell_numbers_mapping.shape,
            *input_coarse_landsea_mask.shape)
    return fine_landsea_mask
