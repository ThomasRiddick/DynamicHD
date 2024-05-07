// #include <iostream>
// #include <fstream>
// #include <unistd.h>
#include "base/grid.hpp"
#include "algorithms/river_direction_determination_algorithm.hpp"

using namespace std;

void determine_river_directions_icon_cython_interface(int ncells,
                                                      double* orography_in,
                                                      int* landsea_in_int,
                                                      double* landsea_in_double,
                                                      int* true_sinks_in_int,
                                                      int* neighboring_cell_indices_in,
                                                      int* next_cell_index_out,
                                                      int fractional_landsea_mask_in_int,
                                                      int always_flow_to_sea_in_int,
                                                      int mark_pits_as_true_sinks_in_int){
  bool use_secondary_neighbors = true;
  int* secondary_neighboring_cell_indices = nullptr;
  bool fractional_landsea_mask_in = bool(fractional_landsea_mask_in_int);
  auto grid_params_obj =
    new icon_single_index_grid_params(ncells,
                                      neighboring_cell_indices_in,
                                      use_secondary_neighbors,
                                      secondary_neighboring_cell_indices);
  bool* landsea_in = new bool[ncells];
  if (fractional_landsea_mask_in) {
    //invert landsea mask
    for (auto i = 0; i <ncells;i++){
      if (landsea_in_double[i] < 0.5) landsea_in_double[i] = 0.0;
      else landsea_in_double[i] = 1.0;
      landsea_in[i] = ! bool(landsea_in_double[i]);
    }
  } else {
    for (auto i = 0; i <ncells;i++){
      landsea_in[i] = ! bool(landsea_in_int[i]);
    }
  }
  bool* true_sinks_in = new bool[ncells];
  bool mark_pits_as_true_sinks_in = bool(mark_pits_as_true_sinks_in_int);
  for (auto i = 0; i < ncells;i++){
    mark_pits_as_true_sinks_in =  mark_pits_as_true_sinks_in || bool(true_sinks_in_int[i]);
    true_sinks_in[i] = false;
  }
  auto alg = river_direction_determination_algorithm_icon_single_index();
  alg.setup_flags(bool(always_flow_to_sea_in_int),use_secondary_neighbors,
                  mark_pits_as_true_sinks_in);
  alg.setup_fields(next_cell_index_out,orography_in,landsea_in,true_sinks_in,
                   grid_params_obj);
  alg.determine_river_directions();
  delete grid_params_obj;
}
