'''Wrap C++ lake operators code using Cython'''
import cython
from Cython.Shadow import bint
cimport numpy as np
import numpy as np
from libcpp cimport bool

cdef extern from 'drivers/burn_carved_rivers.cpp':
    void latlon_burn_carved_rivers_cython_wrapper(double* orography_in,double* rdirs_in,
                                                  int* minima_in_int, int* lakemask_in_int,
                                                  int nlat_in,int nlon_in,
                                                  int add_slope_in,
                                                  int max_exploration_range_in,
                                                  double minimum_height_change_threshold_in,
                                                  int short_path_threshold_in,
                                                  double short_minimum_height_change_threshold_in)
cdef extern from 'drivers/fill_lakes.cpp':
    void latlon_fill_lakes_cython_wrapper(int* lake_minima_in_int,int* lake_mask_in_int,
                                          double* orography_in,int nlat_in, int nlon_in,
                                          bint use_highest_possible_lake_water_level_in_int)
cdef extern from 'drivers/reduce_connected_areas_to_points.cpp':
    void latlon_reduce_connected_areas_to_points_cython_wrapper(int* areas_in_int,int nlat_in,int nlon_in,
                                                                bint use_diagonals_in_int,
                                                                double* orography_in,
                                                                bint check_for_false_minima_in)
    void latlon_reduce_connected_areas_to_points_cython_wrapper(int* areas_in_int,int nlat_in,int nlon_in,
                                                                bint use_diagonals_in_int)

cdef extern from 'drivers/redistribute_water.cpp':
    void latlon_redistribute_water_cython_wrapper(int* lake_numbers_in,
                                                  int* lake_centers_in_int,
                                                  double* water_to_redistribute_in,
                                                  double* water_redistributed_to_lakes_in,
                                                  double* water_redistributed_to_rivers_in,
                                                  int nlat_in, int nlon_in,
                                                  int coarse_nlat_in, int coarse_nlon_in)

cdef extern from 'drivers/filter_out_shallow_lakes.cpp':
    void latlon_filter_out_shallow_lakes(double* unfilled_orography,double* filled_orography,
                                         double minimum_depth_threshold,int nlat_in,int nlon_in)

def burn_carved_rivers(np.ndarray[double,ndim=2,mode='c'] orography_in,
                       np.ndarray[double,ndim=2,mode='c'] rdirs_in,
                       np.ndarray[int,ndim=2,mode='c'] minima_in_int,
                       np.ndarray[int,ndim=2,mode='c'] lakemask_in_int,
                       bint add_slope_in = False,
                       int max_exploration_range_in = 0,
                       double minimum_height_change_threshold_in = 0.0,
                       int short_path_threshold_in = 0,
                       double short_minimum_height_change_threshold_in = 0.0):
    """Call the C++ cython interface function from cython with appropriate arguments.

    Also find the required number of latitude and longitude points to pass in
    """

    cdef int nlat_in,nlon_in
    nlat_in, nlon_in = orography_in.shape[0],orography_in.shape[1]
    latlon_burn_carved_rivers_cython_wrapper(&orography_in[0,0],
                                             &rdirs_in[0,0],
                                             &minima_in_int[0,0],
                                             &lakemask_in_int[0,0],
                                             nlat_in,nlon_in,
                                             add_slope_in,
                                             max_exploration_range_in,
                                             minimum_height_change_threshold_in,
                                             short_path_threshold_in,
                                             short_minimum_height_change_threshold_in)

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

def redistribute_water(np.ndarray[int,ndim=2,mode='c'] lake_numbers_in,
                       np.ndarray[int,ndim=2,mode='c'] lake_centers_in_int,
                       np.ndarray[double,ndim=2,mode='c'] water_to_redistribute_in,
                       np.ndarray[double,ndim=2,mode='c'] water_redistributed_to_lakes_in,
                       np.ndarray[double,ndim=2,mode='c'] water_redistributed_to_rivers_in):
  cdef int nlat_in,nlon_in
  nlat_in, nlon_in = lake_numbers_in.shape[0],lake_numbers_in.shape[1]
  cdef int coarse_nlat_in,coarse_nlon_in
  coarse_nlat_in, coarse_nlon_in = (water_redistributed_to_rivers_in.shape[0],
                                    water_redistributed_to_rivers_in.shape[1])
  latlon_redistribute_water_cython_wrapper(&lake_numbers_in[0,0],
                                           &lake_centers_in_int[0,0],
                                           &water_to_redistribute_in[0,0],
                                           &water_redistributed_to_lakes_in[0,0],
                                           &water_redistributed_to_rivers_in[0,0],
                                           nlat_in, nlon_in,coarse_nlat_in,coarse_nlon_in)

def filter_out_shallow_lakes(np.ndarray[double,ndim=2,mode='c'] unfilled_orography,
                             np.ndarray[double,ndim=2,mode='c'] filled_orography,
                             double minimum_depth_threshold):
    cdef int nlat_in,nlon_in
    nlat_in, nlon_in = unfilled_orography.shape[0],unfilled_orography.shape[1]
    latlon_filter_out_shallow_lakes(&unfilled_orography[0,0],
                                    &filled_orography[0,0],
                                    minimum_depth_threshold,
                                    nlat_in,nlon_in)
