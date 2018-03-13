/*
 * burn_carved_river_directions.cpp
 *
 *  Created on: Feb 19, 2018
 *      Author: thomasriddick
 */

#include "carved_river_direction_burning_algorithm.hpp"
#include "coords.hpp"

carved_river_direction_burning_algorithm::~carved_river_direction_burning_algorithm() {
	delete orography; delete minima; delete lakemask;
	delete _grid;
}

void carved_river_direction_burning_algorithm::setup_fields(double* orography_in,bool* minima_in,
		bool* lakemask_in,grid_params* grid_params_in) {
	_grid_params = grid_params_in;
	_grid = grid_factory(_grid_params);
	orography = new field<double>(orography_in,_grid_params);
	minima = new field<bool>(minima_in,_grid_params);
	lakemask = new field<bool>(lakemask_in,_grid_params);
}

void carved_river_direction_burning_algorithm_latlon::setup_fields(double* orography_in,
	double* rdirs_in,bool* minima_in,bool* lakemask_in,grid_params* grid_params_in) {
	carved_river_direction_burning_algorithm::setup_fields(orography_in,minima_in,lakemask_in,
														   grid_params_in);
	rdirs = new field<double>(rdirs_in,grid_params_in);
}

void carved_river_direction_burning_algorithm::burn_carved_river_directions() {
	add_minima_to_q();
	while (!q.empty()) {
		cell* minima = q.front();
		q.pop();
		double minima_height = minima->get_orography();
		coords* working_cell_coords = minima->get_cell_coords()->clone();
		delete minima;
		while(true) {
			working_cell_coords = get_next_cell_downstream(working_cell_coords);
			double working_cell_height = (*orography)(working_cell_coords);
			if ( working_cell_height > minima_height && ! (*lakemask)(working_cell_coords)){
				(*orography)(working_cell_coords) = minima_height;
			} else break;
		}
	}
};

void carved_river_direction_burning_algorithm::add_minima_to_q() {
	_grid->for_all([&](coords* coords_in){
		if ( (*minima)(coords_in) ) {
			q.push(new cell((*orography)(coords_in),coords_in));
		}
	});
}

coords* carved_river_direction_burning_algorithm_latlon::get_next_cell_downstream(coords* initial_coords) {
	coords* new_coords= _grid->calculate_downstream_coords_from_dir_based_rdir(initial_coords,
						(*rdirs)(initial_coords));
	delete initial_coords;
	return new_coords;
};

