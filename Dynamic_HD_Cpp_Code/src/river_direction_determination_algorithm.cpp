/*
 * river_direction_determination_algorithm.cpp
 *
 *  Created on: Dec 4, 2018
 *      Author: thomasriddick
 */

#include "river_direction_determination_algorithm.hpp"
#include <limits>
#include <cmath>

using namespace std;

/* IMPORTANT!
 * Note that a description of each functions purpose is provided with the function declaration in
 * the header file not with the function definition. The definition merely provides discussion of
 * details of the implementation.
 */


void river_direction_determination_algorithm::setup_flags(bool always_flow_to_sea_in,
                                                          bool use_diagonal_nbrs_in,
                                                          bool mark_pits_as_true_sinks_in) {
  always_flow_to_sea = always_flow_to_sea_in;
  use_diagonal_nbrs = use_diagonal_nbrs_in;
  mark_pits_as_true_sinks = mark_pits_as_true_sinks_in;
}

void river_direction_determination_algorithm::setup_fields(double* orography_in,
                                                           bool* land_sea_in,
                                                           bool* true_sinks_in,
                                                           grid_params* grid_params_in) {
  _grid_params = grid_params_in;
  _grid = grid_factory(_grid_params);
  orography = new field<double>(orography_in,_grid_params);
  land_sea = new field<bool>(land_sea_in,_grid_params);
  true_sinks = new field<bool>(true_sinks_in,_grid_params);
}

void river_direction_determination_algorithm::determine_river_directions() {
  _grid->for_all([&](coords* coords_in){
  if ((*true_sinks)(coords_in) &&
      ! (*land_sea)(coords_in)) mark_as_sink_point(coords_in);
  else if(! (*land_sea)(coords_in)) find_river_direction(coords_in);
  delete coords_in;
  });
  //Need the river directions first before marking ocean points with inflows
  _grid->for_all([&](coords* coords_in){
    if( (*land_sea)(coords_in)){
      if(point_has_inflows(coords_in)) mark_as_outflow_point(coords_in);
      else mark_as_ocean_point(coords_in);
    }
    delete coords_in;
  });
}

void river_direction_determination_algorithm::find_river_direction(coords* coords_in){
  double minimum_height = numeric_limits<double>::max();
  coords* minimum_height_nbr_coords = nullptr;
  bool sea_only = false;
  void (grid::*func)(coords*,function<void(coords*)>) = use_diagonal_nbrs ?
    &grid::for_all_nbrs : &grid::for_non_diagonal_nbrs;
  (_grid->*func)(coords_in,[&](coords* nbr_coords){
    bool is_sea = (*land_sea)(nbr_coords);
    double nbr_height = (*orography)(nbr_coords);
    if (sea_only && ! is_sea) {
    } else if(always_flow_to_sea && is_sea && ! sea_only){
      sea_only = true;
      minimum_height_nbr_coords = nbr_coords->clone();
      minimum_height = nbr_height;
    } else {
      minimum_height_nbr_coords = nbr_height < minimum_height ?
        nbr_coords->clone() : minimum_height_nbr_coords;
      minimum_height = min(minimum_height,nbr_height);
    }
    delete nbr_coords;
  });
  if (minimum_height > (*orography)(coords_in)) {
    if (mark_pits_as_true_sinks) mark_as_sink_point(coords_in);
    else throw runtime_error("Possible false sink found");
  } else mark_river_direction(coords_in,minimum_height_nbr_coords);
}

bool river_direction_determination_algorithm::point_has_inflows(coords* coords_in){
  bool inflow_found = false;
  void (grid::*func)(coords*,function<void(coords*)>) = use_diagonal_nbrs ?
    &grid::for_all_nbrs : &grid::for_non_diagonal_nbrs;
  (_grid->*func)(coords_in,[&](coords* nbr_coords){
    coords* cell_downstream_from_nbr = get_downstream_cell_coords(nbr_coords);
    if ((*cell_downstream_from_nbr) == (*coords_in)) inflow_found = true;
    delete nbr_coords;
  });
  return inflow_found;
}

void river_direction_determination_algorithm_latlon::setup_fields(double* rdirs_in,
                                                                  double* orography_in,
                                                                  bool* land_sea_in,
                                                                  bool* true_sinks_in,
                                                                  grid_params* grid_params_in){
  river_direction_determination_algorithm::setup_fields(orography_in,land_sea_in,
                                                        true_sinks_in,grid_params_in);
  rdirs = new field<double>(rdirs_in,_grid_params);
}

void river_direction_determination_algorithm_latlon::mark_river_direction(coords* initial_coords,
                                                                          coords* destination_coords){
  (*rdirs)(initial_coords) = _grid->calculate_dir_based_rdir(initial_coords,destination_coords);
}

void river_direction_determination_algorithm_latlon::mark_as_sink_point(coords* coords_in){
  (*rdirs)(coords_in) = sink_point_code;
}

void river_direction_determination_algorithm_latlon::mark_as_outflow_point(coords* coords_in){
  (*rdirs)(coords_in) = outflow_code;
}

void river_direction_determination_algorithm_latlon::mark_as_ocean_point(coords* coords_in){
  (*rdirs)(coords_in) = ocean_code;
}

coords* river_direction_determination_algorithm_latlon::get_downstream_cell_coords(coords* coords_in){
  double rdir = (*rdirs)(coords_in);
  return _grid->calculate_downstream_coords_from_dir_based_rdir(coords_in,rdir);
}

void river_direction_determination_algorithm_icon_single_index::
     setup_fields(int* next_cell_index_in,
                  double* orography_in,
                  bool* land_sea_in,
                  bool* true_sinks_in,
                  grid_params* grid_params_in){
  river_direction_determination_algorithm::setup_fields(orography_in,land_sea_in,
                                                        true_sinks_in,grid_params_in);
  next_cell_index = new field<int>(next_cell_index_in,_grid_params);
}

void river_direction_determination_algorithm_icon_single_index::
     mark_river_direction(coords* initial_coords,coords* destination_coords){
  auto destination_generic_1d_coords =
    static_cast<generic_1d_coords*>(destination_coords);
  (*next_cell_index)(initial_coords) = destination_generic_1d_coords->get_index();
}

void river_direction_determination_algorithm_icon_single_index::
     mark_as_sink_point(coords* coords_in){
  (*next_cell_index)(coords_in) = true_sink_value;
}

void river_direction_determination_algorithm_icon_single_index::
     mark_as_outflow_point(coords* coords_in){
  (*next_cell_index)(coords_in) = outflow_value;
}

void river_direction_determination_algorithm_icon_single_index::
     mark_as_ocean_point(coords* coords_in){
  (*next_cell_index)(coords_in) = ocean_value;
}

coords* river_direction_determination_algorithm_icon_single_index::
        get_downstream_cell_coords(coords* coords_in){
  icon_single_index_grid* _icon_single_index_grid =
    static_cast<icon_single_index_grid*>(_grid);
  return _icon_single_index_grid->
    convert_index_to_coords((*next_cell_index)(coords_in));
}