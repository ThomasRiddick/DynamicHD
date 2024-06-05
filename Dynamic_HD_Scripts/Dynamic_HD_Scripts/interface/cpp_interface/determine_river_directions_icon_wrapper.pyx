import cython
from Cython.Shadow import bint
cimport numpy as np
import numpy as np

cdef extern from 'drivers/determine_river_directions_icon_interface.cpp':
    void determine_river_directions_icon_cython_interface(int ncells,
                                                          double* orography_in,
                                                          int* landsea_in_int,
                                                          double* landsea_in_double,
                                                          int* true_sinks_in_int,
                                                          int* neighboring_cell_indices_in,
                                                          int* next_cell_index_out,
                                                          int fractional_landsea_mask_in_int,
                                                          int always_flow_to_sea_in_int,
                                                          int mark_pits_as_true_sinks_in_int)

def determine_river_directions_icon_cpp(np.ndarray[double,ndim=1,mode='c'] orography_in,
                                        np.ndarray[int,ndim=1,mode='c'] landsea_in_int,
                                        np.ndarray[double,ndim=1,mode='c'] landsea_in_double,
                                        np.ndarray[int,ndim=1,mode='c'] true_sinks_in_int,
                                        np.ndarray[int,ndim=1,mode='c'] neighboring_cell_indices_in,
                                        np.ndarray[int,ndim=1,mode='c'] next_cell_index_out,
                                        bint fractional_landsea_mask_in,
                                        bint always_flow_to_lowest_in,
                                        bint mark_pits_as_true_sinks_in):
    cdef int ncells
    ncells = len(orography_in)
    cdef bint always_flow_to_sea = 0 if always_flow_to_lowest_in == 1 else 1
    determine_river_directions_icon_cython_interface(ncells,
                                                     &orography_in[0],
                                                     &landsea_in_int[0],
                                                     &landsea_in_double[0],
                                                     &true_sinks_in_int[0],
                                                     &neighboring_cell_indices_in[0],
                                                     &next_cell_index_out[0],
                                                     fractional_landsea_mask_in,
                                                     always_flow_to_sea,
                                                     mark_pits_as_true_sinks_in)
