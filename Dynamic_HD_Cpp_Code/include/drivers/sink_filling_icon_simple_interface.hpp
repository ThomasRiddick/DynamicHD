#ifndef INCLUDE_SINK_FILLING_ICON_SI_WRAPPER_HPP_
#define INCLUDE_SINK_FILLING_ICON_SI_WRAPPER_HPP_

void sink_filling_icon_si_cython_interface(int ncells,
                                           int* neighboring_cell_indices_in,
                                           double* orography_inout,
                                           int* landsea_in_int,
                                           double* landsea_in_double,
                                           int* true_sinks_in_int,
                                           int fractional_landsea_mask_in_int,
                                           int set_ls_as_no_data_flag_in_int,
                                           int add_slope_in_int,
                                           double epsilon_in);

#endif /* INCLUDE_SINK_FILLING_ICON_SI_WRAPPER_HPP_ */
