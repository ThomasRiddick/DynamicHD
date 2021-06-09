'''Wrap C++ orography creation code using Cython'''
import cython
cimport numpy as np
import numpy as np

cdef extern from 'drivers/create_orography.cpp':
    void create_orography_cython_wrapper(int* landsea_in_int,double* inclines_in,
                                         double* orography_in,double sea_level_in,
                                         int nlat_in,int nlon_in)

def create_orography(np.ndarray[int,ndim=2,mode='c'] landsea_in_int,
                     np.ndarray[double,ndim=2,mode='c'] inclines_in,
                     np.ndarray[double,ndim=2,mode='c'] orography_in,
                     double sea_level_in=0.0):
    cdef int nlat,nlon
    nlat,nlon = landsea_in_int.shape[0],landsea_in_int.shape[1]
    create_orography_cython_wrapper(&landsea_in_int[0,0],&inclines_in[0,0],
                                    &orography_in[0,0],sea_level_in,nlat,nlon)
