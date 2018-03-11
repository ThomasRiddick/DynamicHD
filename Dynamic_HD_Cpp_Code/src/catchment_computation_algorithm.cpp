/*
 * catchment_computation_algorithm.cpp
 *
 *  Created on: Feb 23, 2018
 *      Author: thomasriddick
 */

#include "catchment_computation_algorithm.hpp"

void catchment_computation_algorithm::setup_fields(int* catchment_numbers_in,
												   grid_params* grid_params_in) {
	_grid_params = grid_params_in;
	_grid = grid_factory(_grid_params);
	catchment_numbers = new field<int>(catchment_numbers_in,_grid_params);
	completed_cells = new field<bool>(_grid_params);
	completed_cells->set_all(false);
}

void catchment_computation_algorithm::compute_catchments() {
	add_outflows_to_queue();
	catchment_number = 1;
	while (! outflow_q.empty()) {
		outflow = outflow_q.front();
		outflow_q.pop();
		compute_catchment();
		catchment_number++;
	}
}

void catchment_computation_algorithm::add_outflows_to_queue() {
	_grid->for_all([&](coords* coords_in){
		if (check_if_neighbor_is_upstream(coords_in)) {
			outflow_q.push(new landsea_cell(coords_in));
		}
	});
}

void catchment_computation_algorithm::compute_catchment() {
	q.push(outflow);
	while ( ! q.empty()) {
		center_cell = q.front();
		q.pop();
		center_coords = center_cell->get_cell_coords();
		(*catchment_numbers)(center_coords) = catchment_number;
		(*completed_cells)(center_coords) = true;
		process_neighbors();
		delete center_cell;
	}
}

void catchment_computation_algorithm::process_neighbors() {
	neighbors_coords = completed_cells->get_neighbors_coords(center_coords,1);
	while( ! neighbors_coords->empty() ) {
		process_neighbor();
	}
	delete neighbors_coords;
}

void catchment_computation_algorithm::process_neighbor() {
	nbr_coords = neighbors_coords->back();
	if ( ! (*completed_cells)(nbr_coords)) {

		if ( check_if_neighbor_is_upstream()) {
			q.push(new landsea_cell(nbr_coords));
		}
	}
	neighbors_coords->pop_back();
	delete nbr_coords;
}

void catchment_computation_algorithm_latlon::setup_fields(int* catchment_numbers_in,double* rdirs_in,
		  	  	  	  	  	  	  	  	  	  	  	      grid_params* grid_params_in) {
	catchment_computation_algorithm::setup_fields(catchment_numbers_in,grid_params_in);
	rdirs = new field<double>(rdirs_in,_grid_params);
}

bool catchment_computation_algorithm_latlon::check_for_outflow(coords* cell_coords) {
	return ((*rdirs)(cell_coords) == 0);
}

bool catchment_computation_algorithm_latlon::check_if_neighbor_is_upstream() {
	coords* downstream_coords =
			_grid->calculate_downstream_coords_from_dir_based_rdir(nbr_coords,
																  (*rdirs)(nbr_coords));
	if (downstream_coords == center_coords) {
		delete downstream_coords;
		return true;
	}
	else {
		delete downstream_coords;
		return false;
	}
}
