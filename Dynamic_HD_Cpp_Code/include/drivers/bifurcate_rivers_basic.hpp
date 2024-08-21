/*
 * bifurcate_rivers_basic.hpp
 *
 *  Created on: Nov 5, 2020
 *      Author: thomasriddick
 */

#ifndef INCLUDE_BIFURCATE_RIVERS_BASIC_HPP_
#define INCLUDE_BIFURCATE_RIVERS_BASIC_HPP_
#include <vector>
#include <map>
using namespace std;

void latlon_bifurcate_rivers_basic(map<pair<int,int>,
                                       vector<pair<int,int>>> river_mouths_in,
                                   double* rdirs_in,
                                   double* bifurcations_rdirs_in,
                                   int* cumulative_flow_in,
                                   int* number_of_outflows_in,
                                   bool* landsea_mask_in,
                                   double cumulative_flow_threshold_fraction_in,
                                   int minimum_cells_from_split_to_main_mouth_in,
                                   int maximum_cells_from_split_to_main_mouth_in,
                                   int nlat_in,int nlon_in,
                                   bool remove_main_channel_in = false);

void icon_single_index_bifurcate_rivers_basic(map<int,vector<int>> river_mouths_in,
                                              int* next_cell_index_in,
                                              int* bifurcations_next_cell_index_in,
                                              int* cumulative_flow_in,
                                              int* number_of_outflows_in,
                                              bool* landsea_mask_in,
                                              double cumulative_flow_threshold_fraction_in,
                                              int minimum_cells_from_split_to_main_mouth_in,
                                              int maximum_cells_from_split_to_main_mouth_in,
                                              int ncells_in,
                                              int* neighboring_cell_indices_in);

#endif /* INCLUDE_BIFURCATE_RIVERS_BASIC_HPP_ */
