/*
 * catchment_computation_algorithm.cpp
 *
 *  Created on: Feb 23, 2018
 *      Author: thomasriddick
 */

#include "catchment_computation_algorithm.hpp"

using namespace std;

void catchment_computation_algorithm::setup_fields(int* catchment_numbers_in,
												   grid_params* grid_params_in) {
	_grid_params = grid_params_in;
	_grid = grid_factory(_grid_params);
	catchment_numbers = new field<int>(catchment_numbers_in,_grid_params);
	catchment_numbers->set_all(0);
	completed_cells = new field<bool>(_grid_params);
	completed_cells->set_all(false);
  searched_cells = new field<bool>(_grid_params);
  searched_cells->set_all(false);
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

vector<int>* catchment_computation_algorithm::identify_loops(){
	vector<int>* catchments_with_loops = new vector<int>;
	completed_cells->set_all(false);
	add_loop_cells_to_queue();
	//Reuse outflow to hold a loop cell
	while (! outflow_q.empty()) {
		outflow = outflow_q.front();
		outflow_q.pop();
		center_coords = outflow->get_cell_coords();
		if ((*completed_cells)(center_coords)) {
			delete outflow;
			continue;
		}
    find_loop_in_catchment();
		compute_catchment();
		catchments_with_loops->push_back(catchment_number);
		catchment_number++;
	}
	return catchments_with_loops;
}

void catchment_computation_algorithm::add_outflows_to_queue() {
	_grid->for_all([&](coords* coords_in){
		if (check_for_outflow(coords_in)) {
			outflow_q.push(new landsea_cell(coords_in));
		} else delete coords_in;
	});
}

void catchment_computation_algorithm::add_loop_cells_to_queue() {
	_grid->for_all([&](coords* coords_in){
		if (check_for_loops(coords_in)) {
			outflow_q.push(new landsea_cell(coords_in));
		} else delete coords_in;
	});
}

void catchment_computation_algorithm::find_loop_in_catchment() {
  coords* old_coords = nullptr;
  coords* new_coords = center_coords->clone();
  do {
    old_coords = new_coords;
    new_coords = calculate_downstream_coords(old_coords);
    (*searched_cells)(old_coords) = true;
    if(_grid->outside_limits(new_coords)) {
    	delete new_coords;
    	new_coords = old_coords;
    	break;
    }
    delete old_coords;
  } while (! (*searched_cells)(new_coords));
  delete outflow;
  outflow = new landsea_cell(new_coords);
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
		} else delete nbr_coords;
	} else delete nbr_coords;
	neighbors_coords->pop_back();
}

void catchment_computation_algorithm::test_compute_catchment(landsea_cell* outflow_in,
	                            															 int catchmentnumber_in){
	outflow = outflow_in;
	catchment_number = catchmentnumber_in;
	compute_catchment();
}

void catchment_computation_algorithm_latlon::setup_fields(int* catchment_numbers_in,double* rdirs_in,
		  	  	  	  	  	  	  	  	  	  	  	      grid_params* grid_params_in) {
	catchment_computation_algorithm::setup_fields(catchment_numbers_in,grid_params_in);
	rdirs = new field<double>(rdirs_in,_grid_params);
}

bool catchment_computation_algorithm_latlon::check_for_outflow(coords* cell_coords) {
	return ((*rdirs)(cell_coords) == 0.0 || (*rdirs)(cell_coords) == 5.0);
}

bool catchment_computation_algorithm_latlon::check_for_loops(coords* cell_coords) {
	return ((*catchment_numbers)(cell_coords) == 0 &&
	        ((*rdirs)(cell_coords) > 0.0 && (*rdirs)(cell_coords) != 5.0));
}

bool catchment_computation_algorithm_latlon::check_if_neighbor_is_upstream() {
	coords* downstream_coords =
			_grid->calculate_downstream_coords_from_dir_based_rdir(nbr_coords,
																  (*rdirs)(nbr_coords));
	if (*downstream_coords == *center_coords) {
		delete downstream_coords;
		return true;
	}
	else {
		delete downstream_coords;
		return false;
	}
}

coords* catchment_computation_algorithm_latlon::
        calculate_downstream_coords(coords* initial_coords){
  return _grid->calculate_downstream_coords_from_dir_based_rdir(initial_coords,
                                                               (*rdirs)(initial_coords));
}

void catchment_computation_algorithm_icon_single_index::
		 setup_fields(int* catchment_numbers_in,int* next_cell_index_in,
		  	  	  	  grid_params* grid_params_in) {
	catchment_computation_algorithm::setup_fields(catchment_numbers_in,grid_params_in);
	next_cell_index = new field<int>(next_cell_index_in,grid_params_in);
}

bool catchment_computation_algorithm_icon_single_index::check_for_outflow(coords* cell_coords) {
	int next_cell_index_local = (*next_cell_index)(cell_coords);
	return (next_cell_index_local == outflow_value ||
	        next_cell_index_local == true_sink_value);
}

bool catchment_computation_algorithm_icon_single_index::check_for_loops(coords* cell_coords) {
	int next_cell_index_local = (*next_cell_index)(cell_coords);
	return ((*catchment_numbers)(cell_coords) == 0 &&
	        next_cell_index_local != outflow_value &&
	        next_cell_index_local != ocean_value &&
	        next_cell_index_local != true_sink_value);
}

bool catchment_computation_algorithm_icon_single_index::check_if_neighbor_is_upstream() {
	icon_single_index_grid* _icon_single_index_grid =
		static_cast<icon_single_index_grid*>(_grid);
	coords* downstream_coords =
		_icon_single_index_grid->convert_index_to_coords((*next_cell_index)(nbr_coords));
	if (*downstream_coords == *center_coords) {
		delete downstream_coords;
		return true;
	}
	else {
		delete downstream_coords;
		return false;
	}
}

coords* catchment_computation_algorithm_icon_single_index::
        calculate_downstream_coords(coords* initial_coords){
  icon_single_index_grid* _icon_single_index_grid =
		static_cast<icon_single_index_grid*>(_grid);
  return _icon_single_index_grid->
  	convert_index_to_coords((*next_cell_index)(initial_coords));
}
