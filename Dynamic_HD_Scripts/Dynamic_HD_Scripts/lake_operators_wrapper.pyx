'''Wrap C++ lake operators code using Cython'''
import cython
from Cython.Shadow import bint
cimport numpy as np
import numpy as np
from libcpp cimport bool

cdef extern from 'burn_carved_rivers.cpp':
    void latlon_burn_carved_rivers_cython_wrapper(double* orography_in,double* rdirs_in,
                                                  int* minima_in_int, int* lakemask_in_int,
                                                  int nlat_in,int nlon_in)
cdef extern from 'fill_lakes.cpp':
    void latlon_fill_lakes_cython_wrapper(int* lake_minima_in_int,int* lake_mask_in_int,
                                          double* orography_in,int nlat_in, int nlon_in,
                                          bint use_highest_possible_lake_water_level_in_int)
cdef extern from 'reduce_connected_areas_to_points.cpp':
    void latlon_reduce_connected_areas_to_points_cython_wrapper(int* areas_in_int,int nlat_in,int nlon_in,
                                                                bint use_diagonals_in_int,
                                                                double* orography_in,
                                                                bint check_for_false_minima_in)
    void latlon_reduce_connected_areas_to_points_cython_wrapper(int* areas_in_int,int nlat_in,int nlon_in,
                                                                bint use_diagonals_in_int)

def burn_carved_rivers(np.ndarray[double,ndim=2,mode='c'] orography_in,
                       np.ndarray[double,ndim=2,mode='c'] rdirs_in,
                       np.ndarray[int,ndim=2,mode='c'] minima_in_int,
                       np.ndarray[int,ndim=2,mode='c'] lakemask_in_int):
    """Call the C++ cython interface function from cython with appropriate arguments.

    Also find the required number of latitude and longitude points to pass in
    """

    cdef int nlat_in,nlon_in
    nlat_in, nlon_in = orography_in.shape[0],orography_in.shape[1]
    latlon_burn_carved_rivers_cython_wrapper(&orography_in[0,0],
                                             &rdirs_in[0,0],
                                             &minima_in_int[0,0],
                                             &lakemask_in_int[0,0],
                                             nlat_in,nlon_in)

def fill_lakes(np.ndarray[int,ndim=2,mode='c'] lake_minima_in_int,
               np.ndarray[int,ndim=2,mode='c'] lake_mask_in_int,
               np.ndarray[double,ndim=2,mode='c'] orography_in,
               bint use_highest_possible_lake_water_level_in_int):
    """Call the C++ cython interface function from cython with appropriate arguments.

    Also find the required number of latitude and longitude points to pass in
    """

    cdef int nlat_in,nlon_in
    nlat_in, nlon_in = orography_in.shape[0],orography_in.shape[1]
    latlon_fill_lakes_cython_wrapper(&lake_minima_in_int[0,0],&lake_mask_in_int[0,0],
                                     &orography_in[0,0],nlat_in,nlon_in,
                                     use_highest_possible_lake_water_level_in_int)

def reduce_connected_areas_to_points(np.ndarray[int,ndim=2,mode='c'] areas_in_int,
                                     bint use_diagonals_in_int,
                                     np.ndarray[double,ndim=2,mode='c'] orography_in = None,
                                     bint check_for_false_minima_in = False):
    """Call the C++ cython interface function from cython with appropriate arguments.

    Also find the required number of latitude and longitude points to pass in
    """

    cdef int nlat_in,nlon_in
    nlat_in, nlon_in = areas_in_int.shape[0],areas_in_int.shape[1]
    if not check_for_false_minima_in:
      latlon_reduce_connected_areas_to_points_cython_wrapper(&areas_in_int[0,0],
                                                             nlat_in,nlon_in,
                                                             use_diagonals_in_int)
    else:
      latlon_reduce_connected_areas_to_points_cython_wrapper(&areas_in_int[0,0],
                                                             nlat_in,nlon_in,
                                                             use_diagonals_in_int,
                                                             &orography_in[0,0],
                                                             check_for_false_minima_in)
