/*
 * reduce_connected_areas_to_points_algorithm.cpp
 *
 *  Created on: Feb 7, 2018
 *      Author: thomasriddick
 */

#include "algorithms/reduce_connected_areas_to_points_algorithm.hpp"
#include <cmath>

reduce_connected_areas_to_points_algorithm::~reduce_connected_areas_to_points_algorithm() {
	delete completed_cells;
	delete areas;
	delete _grid;
	if (orography) delete orography;
}

void reduce_connected_areas_to_points_algorithm::setup_flags(bool use_diagonals_in,
                                                             bool check_for_false_minima_in)
{
	use_diagonals = use_diagonals_in;
	check_for_false_minima = check_for_false_minima_in;
}

void reduce_connected_areas_to_points_algorithm::setup_fields(bool* areas_in,double* orography_in,
		 	 	 	 	 	 	 	 	 	 	 	grid_params* grid_params_in)
{
	_grid_params = grid_params_in;
	_grid = grid_factory(_grid_params);
	areas = new field<bool>(areas_in,_grid_params);
	completed_cells = new field<bool>(_grid_params);
	completed_cells->set_all(false);
	if (check_for_false_minima) {
		orography = new field<double>(orography_in,_grid_params);
	}
}


void reduce_connected_areas_to_points_algorithm::iterate_over_field() {
	_grid->for_all([&](coords* coords_in){
		if ((*areas)(coords_in)) reduce_connected_area_to_point(coords_in);
		delete coords_in;
	});
}

void reduce_connected_areas_to_points_algorithm::reduce_connected_area_to_point(coords* point) {
	process_initial_point(point);
	while (!q.empty()){
		landsea_cell* center_cell = q.front();
		q.pop();
		center_coords = center_cell->get_cell_coords();
		process_neighbors();
		delete center_cell;
	}
	if (delete_initial_point) (*areas)(point) = false;
	else if (calculate_area_sizes) (*area_sizes)(point) = points_in_area;
}

void reduce_connected_areas_to_points_algorithm::process_initial_point(coords* point) {
	center_coords = point;
	(*completed_cells)(center_coords) = true;
	delete_initial_point = false;
	points_in_area = 1;
	process_neighbors();
}

void reduce_connected_areas_to_points_algorithm::process_neighbors() {
	neighbors_coords = areas->get_neighbors_coords(center_coords,4);
	diagonal_neighbors = floor(neighbors_coords->size()/2.0);
	while (!neighbors_coords->empty()) {
		process_neighbor();
	}
	delete neighbors_coords;
}

inline void reduce_connected_areas_to_points_algorithm::process_neighbor() {
	auto nbr_coords = neighbors_coords->back();
	if (use_diagonals || neighbors_coords->size() > diagonal_neighbors) {
		if(! (*completed_cells)(nbr_coords)) {
			if ((*areas)(nbr_coords)) {
				q.push(new landsea_cell(nbr_coords));
				(*areas)(nbr_coords) = false;
				(*completed_cells)(nbr_coords) = true;
				points_in_area += 1;
			} else if(check_for_false_minima) {
				//Note a neighbour of the same height that is not marked as a minima
				//implies that one of the neighbour's neighbours is at a lower height
				//and hence this is a false minima
				if ((*orography)(nbr_coords) <= (*orography)(center_coords)) {
					delete_initial_point = true;
				}
				delete nbr_coords;
			} else delete nbr_coords;
		} else delete nbr_coords;
	} else delete nbr_coords;
	neighbors_coords->pop_back();
}


