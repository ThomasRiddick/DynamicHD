import cython
cimport numpy as np
import numpy as np
from libcpp.string cimport string
from Cython.Shadow import bint

cdef extern from 'drivers/bifurcate_rivers_basic_icon_interface.cpp':
    void bifurcate_rivers_basic_icon_cython_interface(int ncells,
                                                      int* neighboring_cell_indices_in,
                                                      int* next_cell_index_in,
                                                      int* cumulative_flow_in,
                                                      int* landsea_mask_in_int,
                                                      int* number_of_outflows_out,
                                                      int* bifurcations_next_cell_index_out,
                                                      string mouth_positions_filepath,
                                                      int minimum_cells_from_split_to_main_mouth_in,
                                                      int maximum_cells_from_split_to_main_mouth_in,
                                                      double cumulative_flow_threshold_fraction_in,
                                                      int remove_main_channel_in)

def bifurcate_rivers_basic_icon_cpp(np.ndarray[int,ndim=1,mode='c'] neighboring_cell_indices_in,
                                    np.ndarray[int,ndim=1,mode='c'] next_cell_index_in,
                                    np.ndarray[int,ndim=1,mode='c'] cumulative_flow_in,
                                    np.ndarray[int,ndim=1,mode='c'] landsea_mask_in_int,
                                    np.ndarray[int,ndim=1,mode='c'] number_of_outflows_out,
                                    np.ndarray[int,ndim=1,mode='c']
                                    bifurcations_next_cell_index_out,
                                    str mouth_positions_filepath,
                                    int minimum_cells_from_split_to_main_mouth_in,
                                    int maximum_cells_from_split_to_main_mouth_in,
                                    double cumulative_flow_threshold_fraction_in,
                                    bint remove_main_channel_in):
    cdef int ncells
    ncells = len(next_cell_index_in)
    cdef string mouth_positions_filepath_c = \
        string(bytes(mouth_positions_filepath,'utf-8'))
    bifurcate_rivers_basic_icon_cython_interface(ncells,
                                                 &neighboring_cell_indices_in[0],
                                                 &next_cell_index_in[0],
                                                 &cumulative_flow_in[0],
                                                 &landsea_mask_in_int[0],
                                                 &number_of_outflows_out[0],
                                                 &bifurcations_next_cell_index_out[0],
                                                 mouth_positions_filepath_c,
                                                 minimum_cells_from_split_to_main_mouth_in,
                                                 maximum_cells_from_split_to_main_mouth_in,
                                                 cumulative_flow_threshold_fraction_in,
                                                 remove_main_channel_in)
