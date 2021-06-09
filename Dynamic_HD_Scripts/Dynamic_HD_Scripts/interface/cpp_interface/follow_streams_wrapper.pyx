import cython
cimport numpy as np
import numpy as np

cdef extern from 'drivers/follow_streams.cpp':
    void latlon_follow_streams_cython_wrapper(double* rdirs_in,int* cells_with_loop_in_int,
                                              int* downstream_cells_in_int,int nlat_in,int nlon_in)

def follow_streams(np.ndarray[double,ndim=2,mode='c'] rdirs_in,
                   np.ndarray[int,ndim=2,mode='c'] cells_with_loop_in_int,
                   np.ndarray[int,ndim=2,mode='c'] downstream_cells_in_int):
    cdef int nlat,nlon
    nlat,nlon = rdirs_in.shape[0],rdirs_in.shape[1]
    latlon_follow_streams_cython_wrapper(&rdirs_in[0,0],&cells_with_loop_in_int[0,0],
                                         &downstream_cells_in_int[0,0],nlat,nlon)
