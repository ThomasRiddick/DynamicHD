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
  completed_cells = new field<bool>(_grid_params);
  completed_cells->set_all(false);
}

void river_direction_determination_algorithm::determine_river_directions() {
  _grid->for_all([&](coords* coords_in){
    if(! (*land_sea)(coords_in)){
      if ((*true_sinks)(coords_in)) {
        mark_as_sink_point(coords_in);
        (*completed_cells)(coords_in) = true;
      } else  find_river_direction(coords_in);
    }
    delete coords_in;
  });
  resolve_flat_areas();
  //Need the river directions first before marking ocean points with inflows
  _grid->for_all([&](coords* coords_in){
    if( (*land_sea)(coords_in)){
      if(point_has_inflows(coords_in)) mark_as_outflow_point(coords_in);
      else mark_as_ocean_point(coords_in);
    } else if( !(*completed_cells)(coords_in)){
      if (mark_pits_as_true_sinks) {
        mark_as_sink_point(coords_in);
        q.push(coords_in->clone());
        double flat_height = (*orography)(coords_in);
        (*completed_cells)(coords_in) = true;
        while (! q.empty()){
          coords* center_coords = q.front();
          q.pop();
          process_neighbors(center_coords,flat_height);
          delete center_coords;
        }
      } else throw runtime_error("Possible false sink found");
    }
    delete coords_in;
  });
}

void river_direction_determination_algorithm::find_river_direction(coords* coords_in){
  double minimum_height = numeric_limits<double>::max();
  coords* minimum_height_nbr_coords = nullptr;
  bool sea_only = false;
  bool potential_exit_point = false;
  double cell_height = (*orography)(coords_in);
  void (grid::*func)(coords*,function<void(coords*)>) = use_diagonal_nbrs ?
    &grid::for_all_nbrs_wrapped : &grid::for_non_diagonal_nbrs_wrapped;
  (_grid->*func)(coords_in,[&](coords* nbr_coords){
    bool is_sea = (*land_sea)(nbr_coords);
    double nbr_height = (*orography)(nbr_coords);
    if (! sea_only || is_sea) {
      if(always_flow_to_sea && is_sea && ! sea_only){
        sea_only = true;
        if(minimum_height_nbr_coords) delete minimum_height_nbr_coords;
        minimum_height_nbr_coords = nbr_coords->clone();
        minimum_height = nbr_height;
      } else {
        if(nbr_height < minimum_height){
          if( minimum_height_nbr_coords) delete  minimum_height_nbr_coords;
          minimum_height_nbr_coords = nbr_coords->clone();
        }
        minimum_height = min(minimum_height,nbr_height);
        if (nbr_height == cell_height) {
          if (!(*completed_cells)(nbr_coords) ) potential_exit_point = true;
        }
      }
    } else if (! is_sea) {
      if (nbr_height == cell_height) {
        if (!(*completed_cells)(nbr_coords) ) potential_exit_point = true;
      }
    }
    delete nbr_coords;
  });
  if (minimum_height > cell_height &&
      ! (sea_only && always_flow_to_sea)) {
    if (mark_pits_as_true_sinks) {
      mark_as_sink_point(coords_in);
      (*completed_cells)(coords_in) = true;
    } else throw runtime_error("Possible false sink found");
  } else if (minimum_height < cell_height ||
             (sea_only && always_flow_to_sea)) {
    mark_river_direction(coords_in,minimum_height_nbr_coords);
    (*completed_cells)(coords_in) = true;
    if (potential_exit_point) {
      potential_exit_points.push(new cell(cell_height,coords_in->clone()));
    }
  }
  delete minimum_height_nbr_coords;
}

bool river_direction_determination_algorithm::point_has_inflows(coords* coords_in){
  bool inflow_found = false;
  void (grid::*func)(coords*,function<void(coords*)>) = use_diagonal_nbrs ?
    &grid::for_all_nbrs_wrapped : &grid::for_non_diagonal_nbrs_wrapped;
  (_grid->*func)(coords_in,[&](coords* nbr_coords){
    coords* cell_downstream_from_nbr = get_downstream_cell_coords(nbr_coords);
    if ((*cell_downstream_from_nbr) == (*coords_in)) inflow_found = true;
    delete nbr_coords; delete cell_downstream_from_nbr;
  });
  return inflow_found;
}

void river_direction_determination_algorithm::resolve_flat_areas(){
  while (! potential_exit_points.empty()){
    cell* potential_exit_cell = potential_exit_points.top();
    potential_exit_points.pop();
    double flat_height = potential_exit_cell->get_orography();
    q.push(potential_exit_cell->get_cell_coords()->clone());
    while (! q.empty()){
      coords* center_coords = q.front();
      q.pop();
      process_neighbors(center_coords,flat_height);
      delete center_coords;
    }
    delete potential_exit_cell;
  }
}

void river_direction_determination_algorithm::process_neighbors(coords* center_coords,
                                                                double flat_height){
  void (grid::*func)(coords*,function<void(coords*)>) = use_diagonal_nbrs ?
                     &grid::for_all_nbrs_wrapped :
                     &grid::for_non_diagonal_nbrs_wrapped;
  (_grid->*func)(center_coords,[&](coords* nbr_coords){
    if((*orography)(nbr_coords) == flat_height &&
       ! (*completed_cells)(nbr_coords) &&
       ! (*land_sea)(nbr_coords)){
      (*completed_cells)(nbr_coords) = true;
      mark_river_direction(nbr_coords,center_coords);
      q.push(nbr_coords);
    } else delete nbr_coords;
  });
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
