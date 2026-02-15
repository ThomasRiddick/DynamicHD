#ifndef INCLUDE_BIFURCATE_RIVERS_BASIC_ICON_WRAPPER_HPP_
#define INCLUDE_BIFURCATE_RIVERS_BASIC_ICON_WRAPPER_HPP_

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
                                                  int remove_main_channel_in);

#endif /* INCLUDE_BIFURCATE_RIVERS_BASIC_ICON_WRAPPER_HPP_ */
