'''Wrap C++ sink filling code using Cython

Use low memory version for method 4
'''
import cython
from Cython.Shadow import bint
cimport numpy as np
#Need to import view or conversion of c++ pointer to numpy array will crash for unknown reasons
from cython cimport view
import numpy as np
from libcpp cimport bool

#Declare four versions of the C++ cython interface function for various optional argument combinations
cdef extern from "drivers/fill_sinks.cpp":
    double* latlon_fill_sinks_cython_interface_low_mem(double* orography_in, int nlat, int nlon, int method,
                                                       bint use_ls_mask,int* landsea_in,
                                                       bint set_ls_as_no_data_flag, int use_true_sinks,
                                                       int* true_sinks_in, int add_slope, double epsilon,
                                                       int* next_cell_lat_index_in,int* next_cell_lon_index_in,
                                                       int* catchment_nums_in,
                                                       bint prefer_non_diagonal_initial_dirs)
    void latlon_fill_sinks_cython_interface(double* orography_in, int nlat, int nlon, int method, bint use_ls_mask,
                                                    int* landsea_in, bint set_ls_as_no_data_flag)
    void latlon_fill_sinks_cython_interface(double* orography_in, int nlat, int nlon, int method, bint use_ls_mask,
                                                    int* landsea_in, bint set_ls_as_no_data_flag, int use_true_sinks,
                                                    int* true_sinks_in,int add_slope, double epsilon)
    void latlon_fill_sinks_cython_interface(double* orography_in, int nlat, int nlon, int method)

def fill_sinks_cpp_func(np.ndarray[double,ndim=2,mode='c'] orography_array,
                        int method, bint use_ls_mask = False, np.ndarray[int,ndim=2,mode='c'] landsea_in = None,
                        bint set_ls_as_no_data_flag = False, bint use_true_sinks = False,
                        np.ndarray[int,ndim=2,mode='c'] true_sinks_in = None,
                        bint add_slope = False, double epsilon = 0.0,
                        np.ndarray[int,ndim=2,mode='c'] next_cell_lat_index_in = None,
                        np.ndarray[int,ndim=2,mode='c'] next_cell_lon_index_in = None,
                        np.ndarray[int,ndim=2,mode='c'] catchment_nums_in = None,
                        bint prefer_non_diagonal_initial_dirs = False):
    """Call the C++ cython interface function from cython with appropriate arguments.

    The version of the C++ function call depends on the arguments to this function; also find
    the required number of latitude and longitude points to pass in
    """

    cdef int nlat,nlon
    cdef double* rdirs_out_pointer
    nlat,nlon = orography_array.shape[0],orography_array.shape[1]
    if not add_slope and not use_ls_mask and not use_true_sinks and method != 4:
        #passing in the first element by reference is the prefer way to pass a numpy array to C++
        #in Cython
        latlon_fill_sinks_cython_interface(&orography_array[0,0],nlat,nlon,method)
    elif not use_true_sinks and not add_slope and method != 4:
        latlon_fill_sinks_cython_interface(&orography_array[0,0],nlat,nlon,method,use_ls_mask,
                                           &landsea_in[0,0],set_ls_as_no_data_flag)
    elif method != 4:
        latlon_fill_sinks_cython_interface(&orography_array[0,0],nlat,nlon,method,use_ls_mask,
                                           &landsea_in[0,0],set_ls_as_no_data_flag,use_true_sinks,
                                           &true_sinks_in[0,0],add_slope,epsilon)
    else:
        rdirs_out_pointer = latlon_fill_sinks_cython_interface_low_mem(&orography_array[0,0],nlat,nlon,
                                                                      method,use_ls_mask,
                                                                      &landsea_in[0,0],
                                                                      set_ls_as_no_data_flag,
                                                                      use_true_sinks,
                                                                      &true_sinks_in[0,0],
                                                                      add_slope, epsilon,
                                                                      &next_cell_lat_index_in[0,0],
                                                                      &next_cell_lon_index_in[0,0],
                                                                      &catchment_nums_in[0,0],
                                                                      prefer_non_diagonal_initial_dirs)
        #Work only when you cimport view from cython
        rdirs_out = np.asarray(<double[:nlat,:nlon]>rdirs_out_pointer)
        return rdirs_out

