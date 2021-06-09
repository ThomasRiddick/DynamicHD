/*
 * lake_filling_algorithm.cpp
 *
 *  Created on: Feb 21, 2018
 *      Author: thomasriddick
 */

#include "algorithms/lake_filling_algorithm.hpp"

lake_filling_algorithm::~lake_filling_algorithm() {
	delete lake_minima; delete lake_mask; delete orography; delete lake_numbers;
	delete completed_cells; delete _grid;
}

void lake_filling_algorithm::
	 setup_flags(bool use_highest_possible_lake_water_level_in) {
	use_highest_possible_lake_water_level =
			use_highest_possible_lake_water_level_in;
}

void lake_filling_algorithm::setup_fields(bool* lake_minima_in,
										  bool* lake_mask_in,
		  	  	  	  	  	  	  	  	  double* orography_in,
										  grid_params* grid_params_in) {
	_grid_params = grid_params_in;
	_grid = grid_factory(_grid_params);
	lake_minima = new field<bool>(lake_minima_in,_grid_params);
	lake_mask = new field<bool>(lake_mask_in,_grid_params);
	orography = new field<double>(orography_in,_grid_params);
	completed_cells = new field<bool>(_grid_params);
	completed_cells->set_all(false);
	lake_numbers = new field<int>(_grid_params);
	lake_numbers->set_all(0);
}

void lake_filling_algorithm::fill_lakes(){
	add_lake_minima_to_queue();
	lake_number = 1;
	while (! minima_q.empty()) {
		minima = minima_q.top();
		minima_q.pop();
		if ( (*completed_cells)(minima->get_cell_coords())) {
			delete minima;
			continue;
		}
		fill_lake();
		lake_number++;
	}
}

void lake_filling_algorithm::add_lake_minima_to_queue() {
	_grid->for_all([&](coords* coords_in){
		if ( (*lake_minima)(coords_in) && (*lake_mask)(coords_in)) {
			minima_q.push(new cell((*orography)(coords_in),coords_in));
		} else delete coords_in;
	});
}

void lake_filling_algorithm::fill_lake(){
	q.push(minima);
	double previous_cell_height = 0.0;
	double center_cell_height = 0.0;
	double lake_water_level = 0.0;
	while( ! q.empty()){
		center_cell = q.top();
		q.pop();
		center_coords = center_cell->get_cell_coords();
		center_cell_height = center_cell->get_orography();
		if ( ! (*lake_mask)(center_coords)) {
			lake_water_level =  use_highest_possible_lake_water_level ?
					center_cell_height :
					previous_cell_height;
			adjust_lake_height(lake_number,lake_water_level);
			delete center_cell;
			break;
		}
		(*lake_numbers)(center_coords) = lake_number;
		previous_cell_height = center_cell_height;
		process_neighbors();
		delete center_cell;
	}
	while (! q.empty()){
		center_cell = q.top();
		q.pop();
		center_coords = center_cell->get_cell_coords();
		(*completed_cells)(center_coords) = false;
		delete center_cell;
	}
}

void lake_filling_algorithm::process_neighbors() {
	neighbors_coords = orography->get_neighbors_coords(center_coords,1);
	while( ! neighbors_coords->empty() ) {
		process_neighbor();
	}
	delete neighbors_coords;
}

void lake_filling_algorithm::process_neighbor() {
	coords* nbr_coords = neighbors_coords->back();
	neighbors_coords->pop_back();
	if ( ! (*completed_cells)(nbr_coords)) {
		double nbr_height = (*orography)(nbr_coords);
		q.push(new cell(nbr_height,nbr_coords));
		(*completed_cells)(nbr_coords) = true;
	} else delete nbr_coords;
}

void lake_filling_algorithm::adjust_lake_height(int lake_number,double height) {
	_grid->for_all([&](coords* coords_in){
		if ((*lake_numbers)(coords_in) == lake_number) {
			(*orography)(coords_in) = height;
		}
		delete coords_in;
	});
}
