'''Wrap C++ orography upscaling code using Cython'''
import cython
from Cython.Shadow import bint
cimport numpy as np
import numpy as np
from libcpp cimport bool

cdef extern from 'upscale_orography.cpp':
    void latlon_upscale_orography_cython_interface(double* orography_in, int nlat_fine, int nlon_fine,
                                                   double* orography_out, int nlat_course, int nlon_course,
                                                   int method, int* landsea_in,int* true_sinks_in,
                                                   bint add_slope_in, double epsilon_in,
                                                   int tarasov_separation_threshold_for_returning_to_same_edge_in,
                                                   double tarasov_min_path_length_in,
                                                   bint tarasov_include_corners_in_same_edge_criteria_in,
                                                   bint prefer_non_diagonal_initial_dirs)
    
def upscale_orography(np.ndarray[double,ndim=2,mode='c'] orography_in, 
                      np.ndarray[double,ndim=2,mode='c'] orography_out, int method, 
                      np.ndarray[int,ndim=2,mode='c'] landsea_in,
                      np.ndarray[int,ndim=2,mode='c'] true_sinks_in,
                      bint add_slope_in, double epsilon_in,
                      int tarasov_separation_threshold_for_returning_to_same_edge_in,
                      double tarasov_min_path_length_in,
                      bint tarasov_include_corners_in_same_edge_criteria_in,
                      bint prefer_non_diagonal_initial_dirs=False):
    """Call the C++ cython interface function from cython with appropriate arguments.
   
    Also find the required number of latitude and longitude points to pass in 
    """
    cdef int nlat_fine,nlon_fine 
    nlat_fine, nlon_fine = orography_in.shape[0],orography_in.shape[1]
    cdef int nlat_course,nlon_course
    nlat_course, nlon_course = orography_out.shape[0],orography_out.shape[1]
    latlon_upscale_orography_cython_interface(&orography_in[0,0],nlat_fine,nlon_fine,
                                              &orography_out[0,0],nlat_course,nlon_course,
                                              method,&landsea_in[0,0],&true_sinks_in[0,0],
                                              add_slope_in,epsilon_in,
                                              tarasov_separation_threshold_for_returning_to_same_edge_in,
                                              tarasov_min_path_length_in,
                                              tarasov_include_corners_in_same_edge_criteria_in,
                                              prefer_non_diagonal_initial_dirs)