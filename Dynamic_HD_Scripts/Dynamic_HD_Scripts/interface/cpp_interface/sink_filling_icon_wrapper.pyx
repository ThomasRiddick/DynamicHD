import cython
from Cython.Shadow import bint
cimport numpy as np
import numpy as np

cdef extern from 'drivers/sink_filling_icon_interface.cpp':
    void sink_filling_icon_cython_interface(int ncells,
                                            int* neighboring_cell_indices_in,
                                            double* orography_inout,
                                            int* landsea_in_int,
                                            double* landsea_in_double,
                                            int* true_sinks_in_int,
                                            int fractional_landsea_mask_in_int,
                                            int set_ls_as_no_data_flag_in_int,
                                            int add_slope_in_int,
                                            double epsilon_in)

def sink_filling_icon_cpp(np.ndarray[int,ndim=1,mode='c'] neighboring_cell_indices_in,
                          np.ndarray[double,ndim=1,mode='c'] orography_inout,
                          np.ndarray[int,ndim=1,mode='c'] landsea_in_int,
                          np.ndarray[double,ndim=1,mode='c'] landsea_in_double,
                          np.ndarray[int,ndim=1,mode='c'] true_sinks_in_int,
                          bint fractional_landsea_mask_in,
                          bint set_ls_as_no_data_flag,
                          bint add_slope_in,
                          double epsilon_in):
    cdef int ncells
    ncells = len(orography_inout)
    sink_filling_icon_cython_interface(ncells,
                                       &neighboring_cell_indices_in[0],
                                       &orography_inout[0],
                                       &landsea_in_int[0],
                                       &landsea_in_double[0],
                                       &true_sinks_in_int[0],
                                       fractional_landsea_mask_in,
                                       set_ls_as_no_data_flag,
                                       add_slope_in,
                                       epsilon_in)
