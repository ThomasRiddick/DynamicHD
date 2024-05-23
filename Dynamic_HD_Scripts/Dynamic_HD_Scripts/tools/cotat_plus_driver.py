'''
Drives the FORTRAN COTAT plus algorithm via f2py. The main function is cotat_plus_driver; this then
calls the function run_cotat_plus.
Created on Oct 18, 2016

@author: thomasriddick
'''

import os.path as path
import numpy as np
from mpi4py import MPI
from Dynamic_HD_Scripts.interface.fortran_interface import f2py_manager
from Dynamic_HD_Scripts.context import fortran_project_source_path,fortran_project_object_path,fortran_project_include_path
from Dynamic_HD_Scripts.base import field
from Dynamic_HD_Scripts.base import grid
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.utilities import coordinate_scaling_utilities
from Dynamic_HD_Scripts.utilities.process_manager import ProcessManager
from Dynamic_HD_Scripts.utilities.process_manager import using_mpi
from Dynamic_HD_Scripts.utilities.process_manager import MPICommands

def cotat_plus_icon_icosohedral_cell_latlon_pixel(input_fine_river_directions,
                                                  input_fine_total_cumulative_flow,
                                                  cell_neighbors,
                                                  pixel_center_lats,
                                                  pixel_center_lons,
                                                  cell_vertices_lats,
                                                  cell_vertices_lons,
                                                  cotat_plus_parameters_filepath=None,
                                                  output_cell_numbers=False):
    additional_fortran_filenames =  \
        ["Dynamic_HD_Fortran_Code_src_base_area_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_coords_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_algorithms_cotat_parameters_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_algorithms_cotat_plus.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_doubly_linked_list_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_doubly_linked_list_link_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_field_section_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_precision_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_subfield_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_algorithms_map_non_coincident_grids_mod.f90.o",
         "Dynamic_HD_Fortran_Code_src_base_unstructured_grid_mod.f90.o"]
    additional_fortran_filepaths = \
        [path.join("/Users/thomasriddick/Documents/workspace/worktrees/feature/improved-run-and-build-structure/build/libdyhd_fortran.a.p",filename)
        for filename in additional_fortran_filenames]
    f2py_mngr = f2py_manager.f2py_manager(
        path.join(fortran_project_source_path,"drivers",
                  "cotat_plus_driver_mod.f90"),
                  func_name="cotat_plus_icon_icosohedral_cell_latlon_pixel_f2py_wrapper",
                  additional_fortran_files=additional_fortran_filepaths,
                  include_path=fortran_project_include_path)
    output_coarse_next_cell_index,output_cell_numbers = f2py_mngr.\
        run_current_function_or_subroutine(input_fine_river_directions.\
                                           astype(np.float32,order='F'),
                                           input_fine_total_cumulative_flow.\
                                           astype(np.int32,order='F'),
                                           cell_neighbors.\
                                           astype(np.int32,order='F'),
                                           pixel_center_lats.\
                                           astype(np.float32),
                                           pixel_center_lons.\
                                           astype(np.float32),
                                           cell_vertices_lats.\
                                           astype(np.float64,order='F'),
                                           cell_vertices_lons.\
                                           astype(np.float64,order='F'),
                                           1,
                                           *input_fine_river_directions.shape,
                                           cell_neighbors.shape[0],
                                           cotat_plus_parameters_filepath)
    return output_coarse_next_cell_index,output_cell_numbers

def run_cotat_plus(fine_rdirs_field,fine_total_cumulative_flow_field,cotat_plus_parameters_filepath,
                   coarse_grid_type,**coarse_grid_kwargs):
    """Run the cotat plus fortran code using f2py for a lat-lon field

    Arguments:
    fine_rdirs_field: 2d ndarray; the fine river directions to be upscaled in 1-9 keypad format
    fine_total_cumulative_flow_field: 2d ndarray; the fine total cumulative flow (created from
        the fine_rdirs_field) to be used in upscaling
    cotat_plus_parameter_filepath: string; the file path containing the namelist with the parameters
        for the cotat plus upscaling algorithm
    coarse_grid_type: string; code for the coarse grid type to be upscaled to
    **coarse_grid_kwargs(optional): keyword dictionary; the parameter of the coarse grid to
        upscale to (if required)
    Return: 2d ndarray; the upscaled river direction on the coarse grid

    Compiles and runs the COTAT plus algorithm in Fortran using f2py for a lat-lon field
    """

    additional_fortran_filenames = ["base/area_mod.o",
                                    "base/coords_mod.o",
                                    "algorithms/cotat_parameters_mod.o",
                                    "algorithms/cotat_plus.o",
                                    "base/doubly_linked_list_mod.o",
                                    "base/doubly_linked_list_link_mod.o",
                                    "base/field_section_mod.o",
                                    "base/precision_mod.o",
                                    "base/subfield_mod.o",
                                    "algorithms/map_non_coincident_grids_mod.o",
                                    "base/unstructured_grid_mod.o"]
    additional_fortran_filepaths = [path.join(fortran_project_object_path,filename) for filename in\
                                    additional_fortran_filenames]
    f2py_mngr = f2py_manager.f2py_manager(path.join(fortran_project_source_path,
                                                    "drivers",
                                                    "cotat_plus_driver_mod.f90"),
                                          func_name="cotat_plus_latlon_f2py_wrapper",
                                          additional_fortran_files=additional_fortran_filepaths,
                                          include_path=fortran_project_include_path)
    coarse_grid = grid.makeGrid(coarse_grid_type,**coarse_grid_kwargs)
    if using_mpi():
        comm = MPI.COMM_WORLD
        comm.bcast(MPICommands.RUNCOTATPLUS, root=0)
    coarse_rdirs_field_raw = f2py_mngr.\
        run_current_function_or_subroutine(fine_rdirs_field.get_data().astype(np.int64,order='F'),
                                           fine_total_cumulative_flow_field.get_data().astype(np.int64,order='F'),
                                           cotat_plus_parameters_filepath,
                                           coarse_grid.get_grid_dimensions()[0],
                                           coarse_grid.get_grid_dimensions()[1])
    coarse_rdirs_field = field.makeField(coarse_rdirs_field_raw.astype(np.float64),'RiverDirections',coarse_grid_type,
                                         **coarse_grid_kwargs)
    if fine_rdirs_field.grid_has_coordinates():
      nlat_fine,nlon_fine = fine_rdirs_field.get_grid_dimensions()
      lat_pts_fine,lon_pts_fine = fine_rdirs_field.get_grid_coordinates()
      nlat_coarse,nlon_coarse = coarse_grid.get_grid_dimensions()
      lat_pts_coarse,lon_pts_coarse = \
        coordinate_scaling_utilities.generate_coarse_pts(nlat_fine,nlon_fine,
                                                         lat_pts_fine,lon_pts_fine,
                                                         nlat_coarse,nlon_coarse)
      coarse_rdirs_field.set_grid_coordinates([lat_pts_coarse,lon_pts_coarse])
    return coarse_rdirs_field

def cotat_plus_driver(input_fine_rdirs_filepath,input_fine_total_cumulative_flow_path,
                      output_coarse_rdirs_filepath,cotat_plus_parameters_filepath,
                      fine_grid_type,fine_grid_kwargs={},coarse_grid_type='HD',**coarse_grid_kwargs):
    """Top level driver for the cotat plus algorithm

    Arguments:
    input_fine_rdirs_filepath: string; path to the file with fine river directions to upscale
    input_fine_total_cumulative_flow_path: string; path to the file with the fine scale cumulative
        flow from the fine river directions
    output_coarse_rdirs_filepath: string; path to the file to write the upscaled coarse river directions to
    cotat_plus_parameters_filepath: string; the file path containing the namelist with the parameters
        for the cotat plus upscaling algorithm
    fine_grid_type: string; code for the fine grid type to upscale from
    **fine_grid_kwargs(optional): keyword dictionary; the parameter of the fine grid to
        upscale from
    coarse_grid_type: string; code for the coarse grid type to be upscaled to
    **coarse_grid_kwargs(optional): keyword dictionary; the parameter of the coarse grid to
        upscale to (if required)
    Returns: Nothing

    Compiles and runs the COTAT plus algorithm in Fortran using f2py for a lat-lon field. Writes
    output specified filename.
    """

    fine_rdirs_field = iodriver.load_field(input_fine_rdirs_filepath,
                                           file_type=iodriver.\
                                           get_file_extension(input_fine_rdirs_filepath),
                                           field_type='RiverDirections',
                                           grid_type=fine_grid_type,**fine_grid_kwargs)
    fine_total_cumulative_flow_field =\
        iodriver.load_field(input_fine_total_cumulative_flow_path,
                            file_type=iodriver.\
                            get_file_extension(input_fine_total_cumulative_flow_path),
                            field_type='CumulativeFlow',
                            grid_type=fine_grid_type,**fine_grid_kwargs)
    coarse_rdirs_field = run_cotat_plus(fine_rdirs_field, fine_total_cumulative_flow_field,
                                        cotat_plus_parameters_filepath,coarse_grid_type,
                                        **coarse_grid_kwargs)
    iodriver.write_field(output_coarse_rdirs_filepath, coarse_rdirs_field,
                         file_type=iodriver.\
                         get_file_extension(output_coarse_rdirs_filepath))

def advanced_cotat_plus_driver(input_fine_rdirs_filepath,input_fine_total_cumulative_flow_path,
                               output_coarse_rdirs_filepath, input_fine_rdirs_fieldname,
                               input_fine_total_cumulative_flow_fieldname,
                               output_coarse_rdirs_fieldname,
                               cotat_plus_parameters_filepath,scaling_factor):

    fine_rdirs_field = iodriver.advanced_field_loader(input_fine_rdirs_filepath,
                                                      field_type='RiverDirections',
                                                      fieldname=input_fine_rdirs_fieldname)
    fine_total_cumulative_flow_field =\
        iodriver.advanced_field_loader(input_fine_total_cumulative_flow_path,
                                       field_type='CumulativeFlow',
                                       fieldname=input_fine_total_cumulative_flow_fieldname)
    nlat_fine,nlon_fine = fine_rdirs_field.get_grid_dimensions()
    lat_pts_fine,lon_pts_fine = fine_rdirs_field.get_grid_coordinates()
    nlat_coarse,nlon_coarse,lat_pts_coarse,lon_pts_coarse = \
        coordinate_scaling_utilities.generate_coarse_coords(nlat_fine,nlon_fine,
                                                            lat_pts_fine,lon_pts_fine,
                                                            scaling_factor)
    coarse_rdirs_field = run_cotat_plus(fine_rdirs_field, fine_total_cumulative_flow_field,
                                        cotat_plus_parameters_filepath,
                                        coarse_grid_type="LatLong",nlat=nlat_coarse,
                                        nlong=nlon_coarse)
    coarse_rdirs_field.set_grid_coordinates([lat_pts_coarse,lon_pts_coarse])
    iodriver.advanced_field_writer(output_coarse_rdirs_filepath, coarse_rdirs_field,
                                   fieldname=output_coarse_rdirs_fieldname)
