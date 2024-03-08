import os.path as path
import numpy as np
from Dynamic_HD_Scripts.utilities import coordinate_scaling_utilities
from Dynamic_HD_Scripts.context import fortran_project_source_path,fortran_project_object_path,fortran_project_include_path

def landsea_downscaler_icosohedral_cell_latlon_pixel(input_coarse_landsea_mask,
                                                     cell_neighbors,
                                                     pixel_center_lats,
                                                     pixel_center_lons,
                                                     cell_vertices_lats,
                                                     cell_vertices_lons)
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
         "Dynamic_HD_Fortran_Code_src_base_unstructured_grid_mod.f90.o"]
    additional_fortran_filepaths = \
        [path.join("/Users/thomasriddick/Documents/workspace/"
                   "Dynamic_HD_Code/build/libdyhd_fortran.a.p",filename)
        for filename in additional_fortran_filenames]
    f2py_mngr = f2py_manager.f2py_manager(
        path.join(fortran_project_source_path,"drivers",
                  "landsea_downscaler_driver_mod.f90"),
                  func_name="landsea_downscaler_icosohedral_cell_latlon_pixel_f2py_wrappe",
                  additional_fortran_files=additional_fortran_filepaths,
                  include_path=fortran_project_include_path)
    fine_landsea_mask = \
        f2py_mngr.run_current_function_or_subroutine(input_coarse_landsea_mask.\
                                                     astype(np.int32,order='F'),
                                                     cell_neighbors.\
                                                     astype(np.int32,order='F'),
                                                     pixel_center_lats.\
                                                     astype(np.float32),
                                                     pixel_center_lons.\
                                                     astype(np.float32),
                                                     cell_vertices_lats.\
                                                     astype(np.float32,order='F'),
                                                     cell_vertices_lons.\
                                                     astype(np.float32,order='F'),
                                                     *input_fine_river_directions.shape,
                                                     cell_neighbors.shape[0])
    return fine_landsea_mask
