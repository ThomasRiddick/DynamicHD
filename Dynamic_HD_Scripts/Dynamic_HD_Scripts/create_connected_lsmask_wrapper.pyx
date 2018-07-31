'''Wrap C++ connected land sea mask creation mode using Cython'''
import cython
from Cython.Shadow import bint
cimport numpy as np
import numpy as np
from libcpp cimport bool

cdef extern from 'create_connected_lsmask.cpp':
    void latlon_create_connected_lsmask_cython_wrapper(int* landsea_in_int, int* ls_seed_points_in_int,
                                                       bool use_diagonals_in_int, int nlat_in, int nlon_in)

def create_connected_ls_mask(np.ndarray[int,ndim=2,mode='c'] landsea_in_int,
                             np.ndarray[int,ndim=2,mode='c'] ls_seed_points_in_int,
                             bint use_diagonals_in_int):
    cdef int nlat,nlon
    nlat,nlon = landsea_in_int.shape[0],landsea_in_int.shape[1]
    latlon_create_connected_lsmask_cython_wrapper(&landsea_in_int[0,0], &ls_seed_points_in_int[0,0],
                                                  nlat, nlon, use_diagonals_in_int)
