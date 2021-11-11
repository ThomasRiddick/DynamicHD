/*
 * birfurcate_rivers_basic.cpp
 *
 *  Created on: Nov 5, 2021
 *      Author: thomasriddick
 */

#include "drivers/bifurcate_rivers_basic.hpp"
#include "algorithms/basic_bifurcation_algorithm.hpp"

void latlon_bifurcate_rivers_basic(map<pair<int,int>,
                                       vector<pair<int,int>>> river_mouths_in,
                                   double* rdirs_in,
                                   double* bifurcations_rdirs_in,
                                   int* cumulative_flow_in,
                                   int* number_of_outflows_in,
                                   bool* landsea_mask_in,
                                   double cumulative_flow_threshold_fraction_in,
                                   int minimum_cells_from_split_to_main_mouth_in,
                                   int nlat_in,int nlon_in){
  cout << "Entering River Bifurcation C++ Code" << endl;
  auto alg = basic_bifurcation_algorithm_latlon();
  auto grid_params_in = new latlon_grid_params(nlat_in,nlon_in);
  alg.setup_fields(river_mouths_in,rdirs_in,
                   cumulative_flow_in,
                   number_of_outflows_in,
                   landsea_mask_in,
                   grid_params_in);
  alg.setup_flags(cumulative_flow_threshold_fraction_in,
                  minimum_cells_from_split_to_main_mouth_in);
  alg.bifurcate_rivers();
  double* bifurcations_rdirs_out = alg.get_bifurcation_rdirs();
  copy(bifurcations_rdirs_out,
       bifurcations_rdirs_out+(nlat_in*nlon_in*alg.get_maximum_bifurcations()),
       bifurcations_rdirs_in);
  delete[] bifurcations_rdirs_out;
  delete grid_params_in;
}

void icon_single_index_bifurcate_rivers_basic(map<int,vector<int>> river_mouths_in,
                                              int* next_cell_index_in,
                                              int* bifurcations_next_cell_index_in,
                                              int* cumulative_flow_in,
                                              int* number_of_outflows_in,
                                              bool* landsea_mask_in,
                                              double cumulative_flow_threshold_fraction_in,
                                              int minimum_cells_from_split_to_main_mouth_in,
                                              int ncells_in,
                                              int* neighboring_cell_indices_in){
  cout << "Entering River Bifurcation C++ Code" << endl;
  auto alg = basic_bifurcation_algorithm_icon_single_index();
  int* secondary_neighboring_cell_indices_in = new int[ncells_in*9];
  auto grid_params_in = new icon_single_index_grid_params(ncells_in,
                                                          neighboring_cell_indices_in,true,
                                                          secondary_neighboring_cell_indices_in);
  grid_params_in->icon_single_index_grid_calculate_secondary_neighbors();
  alg.setup_fields(river_mouths_in,
                   next_cell_index_in,
                   cumulative_flow_in,
                   number_of_outflows_in,
                   landsea_mask_in,
                   grid_params_in);
  alg.setup_flags(cumulative_flow_threshold_fraction_in,
                  minimum_cells_from_split_to_main_mouth_in);
  alg.bifurcate_rivers();
  int* bifurcations_next_cell_index_out = alg.get_bifurcation_next_cell_index();
  copy(bifurcations_next_cell_index_out,
       bifurcations_next_cell_index_out+(ncells_in*alg.get_maximum_bifurcations()),
       bifurcations_next_cell_index_in);
  delete[] secondary_neighboring_cell_indices_in;
  delete[] bifurcations_next_cell_index_out;
  delete grid_params_in;
}
