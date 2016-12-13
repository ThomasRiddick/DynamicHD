'''Wrap C++ sink filling code using Cython'''
import cython
from Cython.Shadow import bint
cimport numpy as np
import numpy as np
from libcpp cimport bool

#Declare four versions of the C++ cython interface function for various optional argument combinations
cdef extern from "fill_sinks.cpp":
    void fill_sinks_cython_interface(double* orography_in, int nlat, int nlon, int method, bint use_ls_mask,
                                     int* landsea_in, bint set_ls_as_no_data_flag, int use_true_sinks, 
                                     int* true_sinks_in, double* rdirs_in, int* catchment_nums_in,
                                     bint prefer_non_diagonal_initial_dirs)
    void fill_sinks_cython_interface(double* orography_in, int nlat, int nlon, int method, bint use_ls_mask,
                                     int* landsea_in, bint set_ls_as_no_data_flag)
    void fill_sinks_cython_interface(double* orography_in, int nlat, int nlon, int method, bint use_ls_mask,
                                     int* landsea_in, bint set_ls_as_no_data_flag, int use_true_sinks,
                                     int* true_sinks_in)
    void fill_sinks_cython_interface(double* orography_in, int nlat, int nlon, int method)        

def fill_sinks_cpp_func(np.ndarray[double,ndim=2,mode='c'] orography_array, 
                        int method, bint use_ls_mask = False, np.ndarray[int,ndim=2,mode='c'] landsea_in = None, 
                        bint set_ls_as_no_data_flag = False, bint use_true_sinks = False,
                        np.ndarray[int,ndim=2,mode='c'] true_sinks_in = None, 
                        np.ndarray[double,ndim=2,mode='c'] rdirs_in = None,
                        np.ndarray[int,ndim=2,mode='c'] catchment_nums_in = None,
                        bint prefer_non_diagonal_initial_dirs = False):
    """Call the C++ cython interface function from cython with appropriate arguments.
   
    The version of the C++ function call depends on the arguments to this function; also find
    the required number of latitude and longitude points to pass in 
    """
    cdef int nlat,nlon
    nlat,nlon = orography_array.shape[0],orography_array.shape[1]
    if not use_ls_mask and not use_true_sinks and rdirs_in is None:
        #passing in the first element by reference is the prefer way to pass a numpy array to C++
        #in Cython
        fill_sinks_cython_interface(&orography_array[0,0],nlat,nlon,method)
    elif not use_true_sinks and rdirs_in is None:
        fill_sinks_cython_interface(&orography_array[0,0],nlat,nlon,method,use_ls_mask,
                                    &landsea_in[0,0],set_ls_as_no_data_flag) 
    elif rdirs_in is None:
        fill_sinks_cython_interface(&orography_array[0,0],nlat,nlon,method,use_ls_mask,
                                    &landsea_in[0,0],set_ls_as_no_data_flag,use_true_sinks, 
                                    &true_sinks_in[0,0])
    else:
        fill_sinks_cython_interface(&orography_array[0,0],nlat,nlon,method,use_ls_mask,
                                    &landsea_in[0,0],set_ls_as_no_data_flag,use_true_sinks,
                                    &true_sinks_in[0,0], &rdirs_in[0,0], &catchment_nums_in[0,0],
                                    prefer_non_diagonal_initial_dirs)