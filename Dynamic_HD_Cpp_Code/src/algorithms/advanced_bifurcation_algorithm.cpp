/*
 * advanced_bifurcation_algorithm.cpp
 *
 *  Created on: Oct 8, 2022
 *      Author: Thomas Riddick
 */
// Decide join point by tracking river upstream
// Calc by orog (for water/no water just give no water 9999 with gradient up to this)
// Also give orog to block incoming major channel???

// Search outwards from starting point to designated join point...
// Trace lowest path and create join
// Repeat for each mouth point

//Need to make this derived class of bifurcation_algorithm...
//And basic bifurcation algorithm another one... then just need minor setup and
//implementation differences between the two and both can share the same core code

#include "algorithms/advanced_bifurcation_algorithm.hpp"
using namespace std;

void advanced_bifurcation_algorithm::push_cell(coords* cell_coords){
  q.push(new cell((*orography)(cell_coords),cell_coords));
}

//Still needs: river mask to orography converter
//             coast blocker

void advanced_bifurcation_algorithm_latlon::setup_fields(map<pair<int,int>,
                                                         vector<pair<int,int>>> river_mouths_in,
                                                         double* rdirs_in,
                                                         double* orography_in,
                                                         int* cumulative_flow_in,
                                                         int* number_of_outflows_in,
                                                         bool* landsea_mask_in,
                                                         grid_params* grid_params_in){
  orography = new field<double>(orography_in,_grid_params);
  bifurcation_algorithm_latlon::setup_fields(river_mouths_in,
                                             rdirs_in,
                                             cumulative_flow_in,
                                             number_of_outflows_in,
                                             landsea_mask_in,
                                             grid_params_in);
}

void advanced_bifurcation_algorithm_icon_single_index::
     setup_fields(map<int,vector<int>> river_mouths_in,
                  int* next_cell_index_in,
                  double* orography_in,
                  int* cumulative_flow_in,
                  int* number_of_outflows_in,
                  bool* landsea_mask_in,
                  grid_params* grid_params_in){
  orography = new field<double>(orography_in,_grid_params);
  bifurcation_algorithm_icon_single_index::setup_fields(river_mouths_in,
                                                        next_cell_index_in,
                                                        orography_in,
                                                        cumulative_flow_in,
                                                        number_of_outflows_in,
                                                        landsea_mask_in,
                                                        grid_params_in);
}
