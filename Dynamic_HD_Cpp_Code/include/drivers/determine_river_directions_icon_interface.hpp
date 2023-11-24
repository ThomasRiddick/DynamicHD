#ifndef INCLUDE_DETERMINE_RIVER_DIRECTIONS_ICON_INTERFACE_WRAPPER_HPP_
#define INCLUDE_DETERMINE_RIVER_DIRECTIONS_ICON_INTERFACE_WRAPPER_HPP_

void determine_river_directions_icon_cython_interface(int ncells,
                                                      double* orography_in,
                                                      int* landsea_in_int,
                                                      double* landsea_in_double,
                                                      int* true_sinks_in_int,
                                                      int* neighboring_cell_indices_in,
                                                      int* next_cell_index_out,
                                                      int fractional_landsea_mask_in_int,
                                                      int always_flow_to_sea_in_int,
                                                      int mark_pits_as_true_sinks_in_int);

#endif /* INCLUDE_DETERMINE_RIVER_DIRECTIONS_ICON_INTERFACE_WRAPPER_HPP_ */
