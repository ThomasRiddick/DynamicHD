import cython
from Cython.Shadow import bint
cimport numpy as np
import numpy as np
from libcpp.string cimport string


cdef extern from 'drivers/compute_catchments_icon_simple_interface.cpp':
    void compute_catchments_icon_si_cython_interface(int ncells,
                                                     int* next_cell_index_in,
                                                     int* catchment_numbers_out,
                                                     int* neighboring_cell_indices_in,
                                                     int sort_catchments_by_size_in_int,
                                                     string loop_log_filepath,
                                                     int generate_selected_subcatchments_only_in_int,
                                                     string subcatchment_list_filepath)

def compute_catchments_icon_si_cpp(np.ndarray[int,ndim=1,mode='c'] next_cell_index_in,
                                   np.ndarray[int,ndim=1,mode='c'] catchment_numbers_out,
                                   np.ndarray[int,ndim=1,mode='c'] neighboring_cell_indices_in,
                                   bint sort_catchments_by_size,
                                   str loop_log_filepath,
                                   bint generate_selected_subcatchments_only,
                                   str subcatchment_list_filepath):
    cdef int ncells
    ncells = len(next_cell_index_in)
    cdef string loop_log_filepath_c = string(bytes(loop_log_filepath,'utf-8'))
    cdef string subcatchment_list_filepath_c = \
        string(bytes(subcatchment_list_filepath,'utf-8'))
    compute_catchments_icon_si_cython_interface(ncells,
                                                &next_cell_index_in[0],
                                                &catchment_numbers_out[0],
                                                &neighboring_cell_indices_in[0],
                                                sort_catchments_by_size,
                                                loop_log_filepath_c,
                                                generate_selected_subcatchments_only,
                                                subcatchment_list_filepath_c)
