#ifndef INCLUDE_COMPUTE_CATCHMENTS_ICON_WRAPPER_HPP_
#define INCLUDE_COMPUTE_CATCHMENTS_ICON_WRAPPER_HPP_

void compute_catchments_icon_cython_interface(int ncells,
                                              int* next_cell_index_in,
                                              int* catchment_numbers_out,
                                              int* neighboring_cell_indices_in,
                                              int sort_catchments_by_size_in_int,
                                              string loop_log_filepath,
                                              int generate_selected_subcatchments_only_in_int,
                                              string subcatchment_list_filepath);

#endif /* INCLUDE_COMPUTE_CATCHMENTS_ICON_WRAPPER_HPP_ */
