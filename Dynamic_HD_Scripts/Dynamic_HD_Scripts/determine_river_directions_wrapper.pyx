'''Wrap C++ river determination code using Cython'''
import cython
cimport numpy as np
import numpy as np

cdef extern from 'drivers/determine_river_directions.cpp':
    void latlon_determine_river_directions_cython_wrapper(double* rdirs_in,
                                                          double* orography_in,
                                                          int* land_sea_in_int,
                                                          int* true_sinks_in_int,
                                                          int nlat_in, int nlon_in,
                                                          int always_flow_to_sea_in_int,
                                                          int use_diagonal_nbrs_in_int,
                                                          int mark_pits_as_true_sinks_in_int)

def determine_river_directions(np.ndarray[double,ndim=2,mode='c'] rdirs_in,
                               np.ndarray[double,ndim=2,mode='c'] orography_in,
                               np.ndarray[int,ndim=2,mode='c'] land_sea_in_int,
                               np.ndarray[int,ndim=2,mode='c'] true_sinks_in_int,
                               int always_flow_to_sea_in_int,
                               int use_diagonal_nbrs_in_int,
                               int mark_pits_as_true_sinks_in_int):
    cdef int nlat_in,nlon_in
    nlat_in, nlon_in = rdirs_in.shape[0],rdirs_in.shape[1]
    latlon_determine_river_directions_cython_wrapper(&rdirs_in[0,0],
                                                     &orography_in[0,0],
                                                     &land_sea_in_int[0,0],
                                                     &true_sinks_in_int[0,0],
                                                     nlat_in,nlon_in,
                                                     always_flow_to_sea_in_int,
                                                     use_diagonal_nbrs_in_int,
                                                     mark_pits_as_true_sinks_in_int)
