'''Wrap C++ catchment computation code using Cython'''
import cython
cimport numpy as np
from libcpp.string cimport string
import numpy as np

cdef extern from 'drivers/compute_catchments.cpp':
    void latlon_compute_catchments(int* catchment_numbers_in, double* rdirs_in,
                                   string loop_log_filepath,
                                   int nlat_in,int nlon_in)

def compute_catchments_cpp(np.ndarray[int,ndim=2,mode='c'] catchment_numbers_in,
                           np.ndarray[double,ndim=2,mode='c'] rdirs_in,
                           str loop_log_filepath):
    """Call the C++ cython interface function from cython with appropriate arguments.

    Also find the required number of latitude and longitude points to pass in
    """

    cdef int nlat_in,nlon_in
    nlat_in, nlon_in = catchment_numbers_in.shape[0],catchment_numbers_in.shape[1]
    cdef string loop_log_filepath_c = string(bytes(loop_log_filepath,'utf-8'))
    latlon_compute_catchments(&catchment_numbers_in[0,0],
                              &rdirs_in[0,0],
                              loop_log_filepath_c,
                              nlat_in,nlon_in)
