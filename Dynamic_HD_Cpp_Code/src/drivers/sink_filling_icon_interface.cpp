#include <iostream>
#include "base/grid.hpp"
#include "algorithms/sink_filling_algorithm.hpp"

using namespace std;

void sink_filling_icon_cython_interface(int ncells,
                                        int* neighboring_cell_indices_in,
                                        double* orography_inout,
                                        int* landsea_in_int,
                                        double* landsea_in_double,
                                        int* true_sinks_in_int,
                                        int fractional_landsea_mask_in_int,
                                        int set_ls_as_no_data_flag_in_int,
                                        int add_slope_in_int,
                                        double epsilon_in){
  bool fractional_landsea_mask_in = bool(fractional_landsea_mask_in_int);
  bool use_secondary_neighbors = true;
  int* secondary_neighboring_cell_indices = nullptr;
  auto grid_params_obj =
    new icon_single_index_grid_params(ncells,
                                      neighboring_cell_indices_in,
                                      use_secondary_neighbors,
                                      secondary_neighboring_cell_indices);
  auto landsea_in =  new bool[ncells];
  if (fractional_landsea_mask_in) {
    //invert landsea mask
    for (auto i = 0; i <ncells;i++){
      landsea_in[i] = ! bool(ceil(landsea_in_double[i]));
    }
  } else {
    for (auto i = 0; i <ncells;i++){
      landsea_in[i] = ! bool(landsea_in_int[i]);
    }
  }
  auto true_sinks_in = new bool[ncells];
  for (auto i = 0; i < ncells;i++){
    true_sinks_in[i] = bool(true_sinks_in_int[i]);
  }
  bool tarasov_mod = false;
  auto alg1 = sink_filling_algorithm_1_icon_single_index();
  alg1.setup_flags(bool(set_ls_as_no_data_flag_in_int),tarasov_mod,
                   bool(add_slope_in_int),epsilon_in);
  alg1.setup_fields(orography_inout,landsea_in,true_sinks_in,grid_params_obj);
  alg1.fill_sinks();
  delete grid_params_obj;
}
