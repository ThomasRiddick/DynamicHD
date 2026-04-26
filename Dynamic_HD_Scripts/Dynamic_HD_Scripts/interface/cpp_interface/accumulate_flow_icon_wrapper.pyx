import cython
cimport numpy as np
import numpy as np

cdef extern from 'drivers/accumulate_flow.cpp':
    void accumulate_flow_icon_single_index(int ncells,
                                           int* neighboring_cell_indices_in,
                                           int* input_next_cell_index,
                                           int* output_cumulative_flow,
                                           int* bifurcated_next_cell_index)
    void accumulate_flow_icon_single_index(int ncells,
                                           int* neighboring_cell_indices_in,
                                           int* input_next_cell_index,
                                           int* output_cumulative_flow)

def accumulate_flow_icon_cpp(
    np.ndarray[int,ndim=1,mode='c'] neighboring_cell_indices_in,
    np.ndarray[int,ndim=1,mode='c'] input_next_cell_index,
    np.ndarray[int,ndim=1,mode='c'] output_cumulative_flow,
    np.ndarray[int,ndim=1,mode='c'] bifurcated_next_cell_index=None):
    cdef int ncells
    ncells = len(input_next_cell_index)
    if bifurcated_next_cell_index is None:
        accumulate_flow_icon_single_index(ncells,
                                          & neighboring_cell_indices_in[0],
                                          & input_next_cell_index[0],
                                          & output_cumulative_flow[0])
    else:
        accumulate_flow_icon_single_index(ncells,
                                          & neighboring_cell_indices_in[0],
                                          & input_next_cell_index[0],
                                          & output_cumulative_flow[0],
                                          & bifurcated_next_cell_index[0])

