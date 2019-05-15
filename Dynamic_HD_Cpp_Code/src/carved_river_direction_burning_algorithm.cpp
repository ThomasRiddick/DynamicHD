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

void carved_river_direction_burning_algorithm::setup_flags(bool add_slope_in,int max_exploration_range_in,
	                 																				 double minimum_height_change_threshold_in){
	add_slope = add_slope_in;
	max_exploration_range = max_exploration_range_in;
	minimum_height_change_threshold = minimum_height_change_threshold_in;
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
		cell* minima = q.top();
		q.pop();
		minima_height = minima->get_orography();
		working_cell_coords = minima->get_cell_coords()->clone();
		delete minima;
		while(true) {
			coords* new_working_cell_coords = get_next_cell_downstream(working_cell_coords);
			if (*new_working_cell_coords == *working_cell_coords ||
			    _grid->outside_limits(new_working_cell_coords)) {
				//Although value will be the same the object will be different (i.e. a clone
				//of the original)
				delete new_working_cell_coords;
				break;
			}
			delete working_cell_coords;
			working_cell_coords = new_working_cell_coords;
			double working_cell_height = (*orography)(working_cell_coords);
			if ( working_cell_height > minima_height && ! (*lakemask)(working_cell_coords)){
				(*orography)(working_cell_coords) = minima_height;
			}
			else if (working_cell_height == minima_height) continue;
			else if ((*lakemask)(working_cell_coords)) continue;
			else break;
			if (add_slope) reprocessing_q.push(working_cell_coords->clone());
		}
		if (add_slope) reprocess_path();
		delete working_cell_coords;
	}
};

void carved_river_direction_burning_algorithm::add_minima_to_q() {
	_grid->for_all([&](coords* coords_in){
		if ( (*minima)(coords_in) ) {
			q.push(new cell((*orography)(coords_in),coords_in));
		} else delete coords_in;
	});
}

void carved_river_direction_burning_algorithm::reprocess_path(){
	double exit_height = (*orography)(working_cell_coords);
	double change_in_height = minima_height - exit_height;
	if (change_in_height <= 0) {
		while (!reprocessing_q.empty()) {
			coords* working_cell_coords = reprocessing_q.top();
			reprocessing_q.pop();
			delete working_cell_coords;
		}
		return;
	} else if (change_in_height < minimum_height_change_threshold &&
	           ! (*lakemask)(working_cell_coords)){
		for(int i = 0;i < max_exploration_range;i++){
			coords* new_working_cell_coords = get_next_cell_downstream(working_cell_coords);
			if (*new_working_cell_coords == *working_cell_coords ||
			    _grid->outside_limits(new_working_cell_coords) ||
			    (*orography)(new_working_cell_coords) > exit_height ||
			    (*lakemask)(new_working_cell_coords)) {
				//Although value will be the same the object will be different (i.e. a clone
				//of the original)
				delete new_working_cell_coords;
				break;
			}
			reprocessing_q.push(working_cell_coords->clone());
			delete working_cell_coords;
			working_cell_coords = new_working_cell_coords;
			if (change_in_height >= minimum_height_change_threshold) break;
		}
		exit_height = (*orography)(working_cell_coords);
		change_in_height = minima_height - exit_height;
		delete working_cell_coords;
	}
	double change_in_height_per_cell = change_in_height/(double(reprocessing_q.size())+1.0);
	for(double i = 1.0; !reprocessing_q.empty();i++){
		coords* working_cell_coords = reprocessing_q.top();
		reprocessing_q.pop();
		if(!(*lakemask)(working_cell_coords)) (*orography)(working_cell_coords) =
																							minima_height - i*change_in_height_per_cell;
		delete working_cell_coords;
	}
}

coords* carved_river_direction_burning_algorithm_latlon::get_next_cell_downstream(coords* initial_coords) {
	coords* new_coords= _grid->calculate_downstream_coords_from_dir_based_rdir(initial_coords,
						(*rdirs)(initial_coords));
	coords* new_coords_wrapped = _grid->wrapped_coords(new_coords);
	if (! (*new_coords_wrapped == *new_coords)) delete new_coords;
	return new_coords_wrapped;
};

coords* carved_river_direction_burning_algorithm_icon_single_index::
	get_next_cell_downstream(coords* initial_coords) {
	coords* new_coords= new generic_1d_coords((*next_cell_index)(initial_coords));
	return new_coords;
};



