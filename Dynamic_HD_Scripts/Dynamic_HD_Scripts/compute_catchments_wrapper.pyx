'''Wrap C++ catchment computation code using Cython'''
import cython
cimport numpy as np
import numpy as np

cdef extern from 'compute_catchments.cpp':
    void latlon_compute_catchments(int* catchment_numbers_in, double* rdirs_in,
                                   int nlat_in,int nlon_in)

def compute_catchments(np.ndarray[int,ndim=2,mode='c'] catchment_numbers_in,
                       np.ndarray[double,ndim=2,mode='c'] rdirs_in):
    """Call the C++ cython interface function from cython with appropriate arguments.

    Also find the required number of latitude and longitude points to pass in
    """

    cdef int nlat_in,nlon_in
    nlat_in, nlon_in = catchment_numbers_in.shape[0],catchment_numbers_in.shape[1]
    latlon_compute_catchments(&catchment_numbers_in[0,0],
                              &rdirs_in[0,0],
                              nlat_in,nlon_in)
