/*
 * stream_follwing_algorithm.cpp
 *
 *  Created on: Feb 10, 2020
 *      Author: thomasriddick
 */

#include "algorithms/stream_following_algorithm.hpp"

void stream_following_algorithm::follow_streams_downstream(){
  _grid->for_all([&](coords* coords_in){
    if((*cells_with_loop)(coords_in)){
      follow_stream_downstream(coords_in);
      delete coords_in;
    }
  });
}

void stream_following_algorithm::follow_stream_downstream(coords* coords_in) {
  coords* next_cell_downstream = coords_in->clone();
  while(! is_outflow(next_cell_downstream)){
    (*downstream_cells)(next_cell_downstream) = true;
    next_cell_downstream = calculate_downstream_cell(next_cell_downstream);
  }
  if(include_downstream_outflow) (*downstream_cells)(next_cell_downstream) = true;
}

void stream_following_algorithm::setup_flags(bool include_downstream_outflow_in){
  include_downstream_outflow = include_downstream_outflow_in;
}

void stream_following_algorithm::setup_fields(bool* cells_with_loop_in,
                                              bool* downstream_cells_in,
                                              grid_params* grid_params_in){
  _grid_params = grid_params_in;
  _grid = grid_factory(grid_params_in);
  cells_with_loop = new field<bool>(cells_with_loop_in,_grid_params);
  downstream_cells = new field<bool>(downstream_cells_in,_grid_params);
  downstream_cells->set_all(false);
}

void dir_based_rdirs_stream_following_algorithm::setup_fields(double* rdirs_in,
                                                              bool* cells_with_loop_in,
                                                              bool* downstream_cells_in,
                                                              grid_params* grid_params_in){
  stream_following_algorithm::setup_fields(cells_with_loop_in,downstream_cells_in,
                                           grid_params_in);
  rdirs = new field<double>(rdirs_in,_grid_params);
}

coords* dir_based_rdirs_stream_following_algorithm::calculate_downstream_cell(coords* coords_in){
  return _grid->calculate_downstream_coords_from_dir_based_rdir(coords_in,
                                                                (*rdirs)(coords_in));
}

bool dir_based_rdirs_stream_following_algorithm::is_outflow(coords* coords_in){
  return ((*rdirs)(coords_in) == -1 || (*rdirs)(coords_in) == 0 || (*rdirs)(coords_in) == -2 );
}
