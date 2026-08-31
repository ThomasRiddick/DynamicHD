import cython
from Cython.Shadow import bint
cimport numpy as np
import numpy as np


cdef extern from 'drivers/cross_grid_mapper.cpp':
    void cross_grid_mapper_latlon_to_icon(double* pixel_center_lats,
                                          double* pixel_center_lons,
                                          double* cell_vertices_lats,
                                          double* cell_vertices_lons,
                                          int* cell_neighbors,
                                          int* output_cell_numbers,
                                          int nlat_fine, int nlon_fine,
                                          int ncells_coarse,
                                          bool longitudal_range_centered_on_zero)

def cross_grid_mapper_latlon_to_icon_cpp(np.ndarray[double,ndim=1,mode='c'] pixel_center_lats,
                                         np.ndarray[double,ndim=1,mode='c'] pixel_center_lons,
                                         np.ndarray[double,ndim=1,mode='c'] cell_vertices_lats,
                                         np.ndarray[double,ndim=1,mode='c'] cell_vertices_lons,
                                         np.ndarray[int,ndim=1,mode='c'] cell_neighbors,
                                         np.ndarray[int,ndim=2,mode='c'] output_cell_numbers,
                                         bint longitudal_range_centered_on_zero):
    cdef int nlat_fine,nlon_fine
    nlat_fine, nlon_fine = pixel_center_lats.shape[0],pixel_center_lons.shape[0]
    cdef int ncells_coarse
    ncells_coarse = len(cell_neighbors)//3
    cross_grid_mapper_latlon_to_icon(&pixel_center_lats[0],
                                     &pixel_center_lons[0],
                                     &cell_vertices_lats[0],
                                     &cell_vertices_lons[0],
                                     &cell_neighbors[0],
                                     &output_cell_numbers[0,0],
                                     nlat_fine,nlon_fine,
                                     ncells_coarse,
                                     longitudal_range_centered_on_zero)
