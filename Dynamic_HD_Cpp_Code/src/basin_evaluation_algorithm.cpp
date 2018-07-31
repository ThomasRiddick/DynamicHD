/*
 * basin_evaluation_algorithm.cpp
 *
 *  Created on: Feb 21, 2018
 *      Author: thomasriddick
 */

#include "basin_evaluation_algorithm.hpp"
#include <queue>

basin_evaluation_algorithm::~basin_evaluation_algorithm() {
	delete minima; delete coarse_minima; delete raw_orography;
	delete corrected_orography; delete connection_volume_thresholds;
	delete flood_volume_thresholds; delete _grid; delete _coarse_grid;
}

void basin_evaluation_algorithm::setup_fields(bool* minima_in, bool* coarse_minima_in,
		  	  	  	  	  	  	  	  	  				double* raw_orography_in,
		  	  	  	  	  	  	  	  	  				double* corrected_orography_in,
		  	  	  	  	  	  	  	  	  				double* connection_volume_thresholds_in,
		  	  	  	  	  	  	  	  	  				double* flood_volume_thresholds_in,
		  	  	  	  	  	  	  	  	  				int* prior_fine_catchments_in,
		  	  	  	  	  	  	  	  	  				int* coarse_catchment_nums_in,
		  	  	  	  	  	  	  	  	  				merge_types* merge_points_in,
										  												grid_params* grid_params_in,
										  												grid_params* coarse_grid_params_in) {
	_grid_params = grid_params_in;
	_coarse_grid_params = coarse_grid_params_in;
	_grid = grid_factory(_grid_params);
	_coarse_grid = grid_factory(_coarse_grid_params);
	minima = new field<bool>(minima_in,_grid_params);
	coarse_minima = new field<bool>(coarse_minima_in,_coarse_grid_params);
	raw_orography = new field<double>(raw_orography_in,_grid_params);
	corrected_orography =
		new field<double>(corrected_orography_in,_grid_params);
	connection_volume_thresholds = new field<double>(connection_volume_thresholds_in,_grid_params);
	flood_volume_thresholds = new field<double>(flood_volume_thresholds_in,_grid_params);
	prior_fine_catchments = new field<int>(prior_fine_catchments_in,_grid_params);
	merge_points = new field<merge_types>(merge_points_in,_grid_params);
	coarse_catchment_nums = new field<int>(coarse_catchment_nums_in,_coarse_grid_params);
	completed_cells = new field<bool>(_grid_params);
	search_completed_cells = new field<bool>(_grid_params);
	requires_redirect_indices = new field<bool>(_grid_params);
	basin_catchment_numbers = new field<int>(_grid_params);
}

void latlon_basin_evaluation_algorithm::setup_fields(bool* minima_in, bool* coarse_minima_in,
		  	  	  	  	  	  	  	  	  							 double* raw_orography_in,
		  	  	  	  	  	  	  	  	  							 double* corrected_orography_in,
		  	  	  	  	  	  	  	  	  							 double* connection_volume_thresholds_in,
		  	  	  	  	  	  	  	  	  							 double* flood_volume_thresholds_in,
		  	  	  	  	  	  	  	  	  							 double* prior_fine_rdirs_in,
		  	  	  	  	  	  	  	  	  							 int* prior_fine_catchments_in,
		  	  	  	  	  	  	  	  	  							 int* coarse_catchment_nums_in,
		  	  	  	  	  	  	  	  	  							 int* next_cell_lat_index_in,
		  	  	  	  	  	  	  	  	  							 int* next_cell_lon_index_in,
		  	  	  	  	  	  	  	  	  							 int* force_merge_lat_index_in,
		  	  	  	  	  	  	  	  	  							 int* force_merge_lon_index_in,
		  	  	  	  	  	  	  	  	  							 int* local_redirect_lat_index_in,
		  	  	  	  	  	  	  	  	  							 int* local_redirect_lon_index_in,
		  	  	  	  	  	  	  	  	  							 int* non_local_redirect_lat_index_in,
		  	  	  	  	  	  	  	  	  							 int* non_local_redirect_lon_index_in,
		  	  	  	  	  	  	  	  	  							 merge_types* merge_points_in,
										  															 grid_params* grid_params_in,
										  															 grid_params* coarse_grid_params_in)
{
	basin_evaluation_algorithm::setup_fields(minima_in,coarse_minima_in,raw_orography_in,
	                                         corrected_orography_in,
	                                         connection_volume_thresholds_in,
	                                         flood_volume_thresholds_in,
		  	  	   														 prior_fine_catchments_in,
		  	  	   														 coarse_catchment_nums_in,merge_points_in,
							 														 grid_params_in,coarse_grid_params_in);
	prior_fine_rdirs = new field<double>(prior_fine_rdirs_in,grid_params_in);
	next_cell_lat_index = new field<int>(next_cell_lat_index_in,grid_params_in);
	next_cell_lon_index = new field<int>(next_cell_lon_index_in,grid_params_in);
	force_merge_lat_index = new field<int>(force_merge_lat_index_in,grid_params_in);
	force_merge_lon_index = new field<int>(force_merge_lon_index_in,grid_params_in);
	local_redirect_lat_index = new field<int>(local_redirect_lat_index_in,grid_params_in);
	local_redirect_lon_index = new field<int>(local_redirect_lon_index_in,grid_params_in);
	non_local_redirect_lat_index = new field<int>(non_local_redirect_lat_index_in,grid_params_in);
	non_local_redirect_lon_index = new field<int>(non_local_redirect_lon_index_in,grid_params_in);
}

void basin_evaluation_algorithm::evaluate_basins(){
	add_minima_to_queue();
	basin_catchment_number = 1;
	while (! minima_q.empty()) {
		minimum = static_cast<basin_cell*>(minima_q.top());
		minima_q.pop();
		evaluate_basin();
		basin_catchment_number++;
	}
	set_remaining_redirects();
}

void basin_evaluation_algorithm::add_minima_to_queue() {
	_grid->for_all([&](coords* coords_in){
		if ((*minima)(coords_in)) {
			coords* coarse_minimum_coords =
				_coarse_grid->convert_fine_coords(coords_in,_grid_params);
			double height = 0.0;
			height_types height_type;
			if ((*coarse_minima)(coarse_minimum_coords)) {
		 		double raw_height = (*raw_orography)(coords_in);
				double corrected_height = (*corrected_orography)(coords_in);
				if(raw_height <= corrected_height) {
					height = raw_height;
					height_type = flood_height;
				} else {
					height = corrected_height;
					height_type = connection_height;
				}
				minima_q.push(new basin_cell(height,height_type,coords_in));
			}
			delete coarse_minimum_coords;
		} else delete coords_in;
	});
}

void basin_evaluation_algorithm::evaluate_basin(){
	q.push(minimum);
	completed_cells->set_all(false);
	double previous_cell_height = 0.0;
	center_cell_height = 0.0;
	previous_filled_cell_height = 0.0;
	cell_number = 0;
	center_cell_volume_threshold = 0.0;
	previous_center_coords = static_cast<basin_cell*>(q.top())->get_cell_coords();
	while( ! q.empty()){
		center_cell = static_cast<basin_cell*>(q.top());
		q.pop();
		center_coords = center_cell->get_cell_coords();
		center_cell_height = center_cell->get_orography();
		if ( center_cell_height < previous_cell_height) {
			if((*basin_catchment_numbers)(center_coords) == null_catchment) {
				(*merge_points)(previous_center_coords) = merge_as_secondary;
				set_secondary_redirect();
				basin_catchment_centers.push_back(minimum->get_cell_coords());
				delete center_cell;
				break;
			} else {
				(*merge_points)(previous_center_coords) = merge_as_primary;
				set_primary_redirect();
			}
		}
		process_center_cell();
		process_neighbors();
		previous_center_coords = center_coords;
		previous_cell_height = center_cell_height;
	}
	while (! q.empty()){
		center_cell = static_cast<basin_cell*>(q.top());
		q.pop();
		center_coords = center_cell->get_cell_coords();
		(*completed_cells)(center_coords) = false;
		delete center_cell;
	}
}

//notice that the previous filled cell is not necessarily
//the previous cell and the cell number doesn't include
//connection cells that are not yet filled
void basin_evaluation_algorithm::process_center_cell() {
	if ((*basin_catchment_numbers)(previous_center_coords)
	    != null_catchment) {
		if((*flood_volume_thresholds)(previous_center_coords) > 0.0) {
			cell_number++;
			return;
		}
	}
	(*basin_catchment_numbers)(previous_center_coords) =
			basin_catchment_number;
	center_cell_volume_threshold =
		center_cell_volume_threshold +
			cell_number*(center_cell_height-previous_filled_cell_height);
	height_types center_cell_height_type = center_cell->get_height_type();
	if (center_cell_height_type == connection_height) {
		(*connection_volume_thresholds)(previous_center_coords) =
			center_cell_volume_threshold;
		q.push(new basin_cell((*raw_orography)(center_coords),
		                      flood_height,center_coords));
	} else if (center_cell_height_type == flood_height) {
		(*flood_volume_thresholds)(previous_center_coords) =
			center_cell_volume_threshold;
		cell_number++;
	} else throw runtime_error("Cell type not recognized");
	previous_filled_cell_height = center_cell_height;
	set_previous_cells_next_cell_index();
}

void basin_evaluation_algorithm::process_neighbors() {
	neighbors_coords = raw_orography->get_neighbors_coords(center_coords,1);
	while( ! neighbors_coords->empty() ) {
		process_neighbor();
	}
	delete neighbors_coords;
}

void basin_evaluation_algorithm::process_neighbor() {
	coords* nbr_coords = neighbors_coords->back();
	neighbors_coords->pop_back();
	if ( ! (*completed_cells)(nbr_coords)) {
				double raw_height = (*raw_orography)(nbr_coords);
				double corrected_height = (*corrected_orography)(nbr_coords);
				double nbr_height;
				height_types nbr_height_type;
				if(raw_height <= corrected_height) {
					nbr_height = raw_height;
					nbr_height_type = flood_height;
				} else {
					nbr_height = corrected_height;
					nbr_height_type = connection_height;
				}
		q.push(new basin_cell(nbr_height,nbr_height_type,
		                		  nbr_coords));
		(*completed_cells)(nbr_coords) = true;
	} else delete nbr_coords;
}

void basin_evaluation_algorithm::set_secondary_redirect(){
	set_previous_cells_next_cell_index();
	(*requires_redirect_indices)(previous_center_coords) = true;
}

void basin_evaluation_algorithm::set_primary_redirect(){
	set_previous_cells_force_merge_index();
	coords* basin_catchment_center_coords =
		basin_catchment_centers[(*basin_catchment_numbers)(center_coords) - 1];
	if(_coarse_grid->fine_coords_in_same_cell(center_coords,
			basin_catchment_center_coords,_grid_params)) {
		  set_previous_cells_local_redirect_index(previous_center_coords,
		                                          basin_catchment_center_coords);
	} else {
		  find_and_set_previous_cells_non_local_redirect_index(previous_center_coords,
		                                                       center_coords,
		                                                       basin_catchment_center_coords);
	}
}

void basin_evaluation_algorithm::
	find_and_set_previous_cells_non_local_redirect_index(coords* initial_center_coords,
	                                                     coords* current_center_coords,
	                                                     coords* catchment_center_coords){
	coords* catchment_center_coarse_coords = _coarse_grid->convert_fine_coords(catchment_center_coords,
	                                                                           _grid_params);
	int coarse_catchment_num = (*coarse_catchment_nums)(catchment_center_coarse_coords);
	find_and_set_non_local_redirect_index_from_coarse_catchment_num(initial_center_coords,
	                                                                current_center_coords,
	                                                                coarse_catchment_num);
	delete catchment_center_coarse_coords;
}

void basin_evaluation_algorithm::
	find_and_set_non_local_redirect_index_from_coarse_catchment_num(coords* initial_center_coords,
	                                                                coords* current_center_coords,
	                                                     						int coarse_catchment_number){
	coords* center_coarse_coords = _coarse_grid->convert_fine_coords(current_center_coords,
	                                                                 _grid_params);
	if((*coarse_catchment_nums)(center_coarse_coords) == coarse_catchment_number) {
		set_previous_cells_non_local_redirect_index(initial_center_coords,
		                                            center_coarse_coords);
		delete center_coarse_coords;
	} else {
		search_q.push(new landsea_cell(center_coarse_coords));
		search_completed_cells->set_all(false);
		while (! search_q.empty()) {
			cell* search_cell = search_q.front();
			search_q.pop();
			search_coords = search_cell->get_cell_coords();
			if((*coarse_catchment_nums)(search_coords) ==
			   coarse_catchment_number){
				set_previous_cells_non_local_redirect_index(initial_center_coords,
		                                            	  search_coords);
				delete search_cell;
				break;
			}
			search_process_neighbors();
			delete search_cell;
		}
		while (! search_q.empty()) {
			cell* search_cell = search_q.front();
			search_q.pop();
			delete search_cell;
		}
	}
}

void basin_evaluation_algorithm::search_process_neighbors() {
	search_neighbors_coords =
		search_completed_cells->get_neighbors_coords(search_coords,1);
	while (!search_neighbors_coords->empty()){
		search_process_neighbor();
	}
	delete search_neighbors_coords;
}

void basin_evaluation_algorithm::search_process_neighbor() {
	auto search_nbr_coords = search_neighbors_coords->back();
	if (!(*search_completed_cells)(search_nbr_coords)) {
		search_q.push(new landsea_cell(search_nbr_coords));
		(*search_completed_cells)(search_nbr_coords) = true;
	}
	search_neighbors_coords->pop_back();
}

void basin_evaluation_algorithm::set_remaining_redirects() {
	_grid->for_all([&](coords* coords_in){
		if((*requires_redirect_indices)(coords_in)){
			coords* next_cell_coords = get_cells_next_cell_index_as_coords(coords_in);
			if((*basin_catchment_numbers)(next_cell_coords) != null_catchment){
				 coords* basin_catchment_center_coords =
				 	basin_catchment_centers[(*basin_catchment_numbers)(next_cell_coords) - 1];
					if(_coarse_grid->fine_coords_in_same_cell(next_cell_coords,
						basin_catchment_center_coords,_grid_params)) {
						set_previous_cells_local_redirect_index(coords_in,
						                                       	basin_catchment_center_coords);
					} else {
		  	 		find_and_set_previous_cells_non_local_redirect_index(coords_in,
						                                        						 next_cell_coords,
		  	 		                                                     basin_catchment_center_coords);
					}
			} else {
				int prior_fine_catchment_num = (*prior_fine_catchments)(next_cell_coords);
				coords* catchment_outlet_coarse_coords = nullptr;
				_grid->for_all([&](coords* coords_in_two){
					if((*prior_fine_catchments)(coords_in_two)
					   	== prior_fine_catchment_num) {
						if (check_for_sinks(coords_in_two)) {
							catchment_outlet_coarse_coords =
								_coarse_grid->convert_fine_coords(coords_in_two,
								                           _grid_params);
						}
					}
				});
				int coarse_catchment_number =
					(*coarse_catchment_nums)(catchment_outlet_coarse_coords);
				find_and_set_non_local_redirect_index_from_coarse_catchment_num(coords_in,
						                                        						 				next_cell_coords,
	                                                     							 		coarse_catchment_number);
				delete catchment_outlet_coarse_coords;
			}
		}
	});
}

priority_cell_queue basin_evaluation_algorithm::test_add_minima_to_queue(double* raw_orography_in,
                                                                         double* corrected_orography_in,
                                                                         bool* minima_in,
                                                                         bool* coarse_minima_in,
                                                                         grid_params* grid_params_in,
                                                                         grid_params* coarse_grid_params_in){
	_grid_params = grid_params_in;
	_coarse_grid_params =  coarse_grid_params_in;
	_grid = grid_factory(_grid_params);
	_coarse_grid = grid_factory(_coarse_grid_params);
	coarse_minima = new field<bool>(coarse_minima_in,_coarse_grid_params);
	minima = new field<bool>(minima_in,_grid_params);
	raw_orography = new field<double>(raw_orography_in,_grid_params);
	corrected_orography =
		new field<double>(corrected_orography_in,_grid_params);
	add_minima_to_queue();
	return minima_q;
}

priority_cell_queue basin_evaluation_algorithm::test_process_neighbors(coords* center_coords_in,
                                                                       bool*   completed_cells_in,
                                                                       double* raw_orography_in,
                                                                       double* corrected_orography_in,
                                                                       grid_params* grid_params_in){
	center_coords = center_coords_in;
	_grid_params = grid_params_in;
	_grid = grid_factory(_grid_params);
	raw_orography = new field<double>(raw_orography_in,_grid_params);
	corrected_orography =
		new field<double>(corrected_orography_in,_grid_params);
	completed_cells = new field<bool>(completed_cells_in,_grid_params);
	process_neighbors();
	return q;
}

queue<landsea_cell*> basin_evaluation_algorithm::
	test_search_process_neighbors(coords* search_coords_in,bool* search_completed_cells_in,
	                              grid_params* grid_params_in){
	_grid_params = grid_params_in;
	search_completed_cells = new field<bool>(search_completed_cells_in,_grid_params);
	search_coords = search_coords_in;
	search_process_neighbors();
	return search_q;
}

priority_cell_queue latlon_basin_evaluation_algorithm::test_process_center_cell(basin_cell* center_cell_in,
                                                                                coords* center_coords_in,
                                                                                coords* previous_center_coords_in,
                                               													 				double* flood_volume_thresholds_in,
                                               													 				double* connection_volume_thresholds_in,
                                               													 				double* raw_orography_in,
                                               													 				int* next_cell_lat_index_in,
                                               													 				int* next_cell_lon_index_in,
                                               													 				int* basin_catchment_numbers_in,
                                               													 				double& center_cell_volume_threshold_in,
                                               													 				int& cell_number_in,
                                               													 				int basin_catchment_number_in,
                                               													 				double center_cell_height_in,
                                               													 				double& previous_filled_cell_height_in,
                                               													 				grid_params* grid_params_in){
	previous_center_coords = previous_center_coords_in;
	center_coords = center_coords_in;
	center_cell = center_cell_in;
	flood_volume_thresholds = new field<double>(flood_volume_thresholds_in,grid_params_in);
	connection_volume_thresholds = new field<double>(connection_volume_thresholds_in,grid_params_in);
	raw_orography = new field<double>(raw_orography_in,grid_params_in);
	next_cell_lat_index = new field<int>(next_cell_lat_index_in,grid_params_in);
	next_cell_lon_index = new field<int>(next_cell_lon_index_in,grid_params_in);
	basin_catchment_numbers = new field<int>(basin_catchment_numbers_in,grid_params_in);
	center_cell_volume_threshold = center_cell_volume_threshold_in;
	cell_number = cell_number_in;
	basin_catchment_number = basin_catchment_number_in;
  center_cell_height = center_cell_height_in;
  previous_filled_cell_height = previous_filled_cell_height_in;
  process_center_cell();
	center_cell_volume_threshold_in = center_cell_volume_threshold;
	cell_number_in = cell_number;
	previous_filled_cell_height_in = previous_filled_cell_height;
	return q;
}

void latlon_basin_evaluation_algorithm::test_set_primary_redirect(vector<coords*> basin_catchment_centers_in,
                                                                  int* basin_catchment_numbers_in,
                                                                  int* coarse_catchment_nums_in,
                                                                  int* force_merge_lat_index_in,
                                                                  int* force_merge_lon_index_in,
                                                                  int* local_redirect_lat_index_in,
                                                                  int* local_redirect_lon_index_in,
                                                                  int* non_local_redirect_lat_index_in,
                                                                  int* non_local_redirect_lon_index_in,
                                                                  coords* center_coords_in,
                                                                  coords* previous_center_coords_in,
                                                                  grid_params* grid_params_in,
                                                                  grid_params* coarse_grid_params_in) {
	_grid_params = grid_params_in;
	_coarse_grid_params = coarse_grid_params_in;
	basin_catchment_centers = basin_catchment_centers_in;
	basin_catchment_numbers = new field<int>(basin_catchment_numbers_in,grid_params_in);
	coarse_catchment_nums = new field<int>(coarse_catchment_nums_in,coarse_grid_params_in);
	force_merge_lat_index = new field<int>(force_merge_lat_index_in,grid_params_in);
	force_merge_lon_index = new field<int>(force_merge_lon_index_in,grid_params_in);
	local_redirect_lat_index = new field<int>(local_redirect_lat_index_in,grid_params_in);
	local_redirect_lon_index = new field<int>(local_redirect_lon_index_in,grid_params_in);
	non_local_redirect_lat_index = new field<int>(non_local_redirect_lat_index_in,grid_params_in);
	non_local_redirect_lon_index = new field<int>(non_local_redirect_lon_index_in,grid_params_in);
	search_completed_cells = new field<bool>(coarse_grid_params_in);
	center_coords = center_coords_in;
	previous_center_coords = previous_center_coords_in;
	_grid = grid_factory(_grid_params);
	_coarse_grid = grid_factory(_coarse_grid_params);
	set_primary_redirect();
}

void latlon_basin_evaluation_algorithm::test_set_secondary_redirect(int* next_cell_lat_index_in,
                                                                    int* next_cell_lon_index_in,
                                                                    bool* requires_redirect_indices_in,
                                                                  	coords* center_coords_in,
                                                                  	coords* previous_center_coords_in,
                                                                  	grid_params* grid_params_in){
	center_coords = center_coords_in;
	previous_center_coords = previous_center_coords_in;
	next_cell_lat_index = new field<int>(next_cell_lat_index_in,grid_params_in);
	next_cell_lon_index = new field<int>(next_cell_lon_index_in,grid_params_in);
  requires_redirect_indices = new field<bool>(requires_redirect_indices_in,grid_params_in);
  set_secondary_redirect();
}

void latlon_basin_evaluation_algorithm::test_set_remaining_redirects(vector<coords*> basin_catchment_centers_in,
                                                                     double* prior_fine_rdirs_in,
                                                                     bool* requires_redirect_indices_in,
                                                                     int* basin_catchment_numbers_in,
                                                                     int* prior_fine_catchments_in,
                                                                     int* coarse_catchment_nums_in,
                                                                     int* next_cell_lat_index_in,
                                                                     int* next_cell_lon_index_in,
                                                                     int* local_redirect_lat_index_in,
                                                                     int* local_redirect_lon_index_in,
                                                                     int* non_local_redirect_lat_index_in,
                                    																 int* non_local_redirect_lon_index_in,
                                                                  	 grid_params* grid_params_in,
                                                                  	 grid_params* coarse_grid_params_in){
	_grid_params = grid_params_in;
	_coarse_grid_params = coarse_grid_params_in;
	basin_catchment_centers = basin_catchment_centers_in;
	prior_fine_rdirs = new field<double>(prior_fine_rdirs_in,grid_params_in);
	requires_redirect_indices = new field<bool>(requires_redirect_indices_in,grid_params_in);
	basin_catchment_numbers = new field<int>(basin_catchment_numbers_in,grid_params_in);
	prior_fine_catchments = new field<int>(prior_fine_catchments_in,grid_params_in);
	coarse_catchment_nums = new field<int>(coarse_catchment_nums_in,coarse_grid_params_in);
	next_cell_lat_index = new field<int>(next_cell_lat_index_in,grid_params_in);
	next_cell_lon_index = new field<int>(next_cell_lon_index_in,grid_params_in);
	local_redirect_lat_index = new field<int>(local_redirect_lat_index_in,grid_params_in);
	local_redirect_lon_index = new field<int>(local_redirect_lon_index_in,grid_params_in);
	non_local_redirect_lat_index = new field<int>(non_local_redirect_lat_index_in,grid_params_in);
	non_local_redirect_lon_index = new field<int>(non_local_redirect_lon_index_in,grid_params_in);
	search_completed_cells = new field<bool>(grid_params_in);
	_grid = grid_factory(_grid_params);
	_coarse_grid = grid_factory(_coarse_grid_params);
	set_remaining_redirects();
}

bool latlon_basin_evaluation_algorithm::check_for_sinks(coords* coords_in){
	double rdir = (*prior_fine_rdirs)(coords_in);
	return (rdir == 5 || rdir == 0);
}

void latlon_basin_evaluation_algorithm::
	set_previous_cells_next_cell_index(){
	latlon_coords* latlon_center_coords = static_cast<latlon_coords*>(center_coords);
	(*next_cell_lat_index)(previous_center_coords) = latlon_center_coords->get_lat();
	(*next_cell_lon_index)(previous_center_coords) = latlon_center_coords->get_lon();
}

void latlon_basin_evaluation_algorithm::
	set_previous_cells_force_merge_index(){
	latlon_coords* latlon_center_coords = static_cast<latlon_coords*>(center_coords);
	(*force_merge_lat_index)(previous_center_coords) = latlon_center_coords->get_lat();
	(*force_merge_lon_index)(previous_center_coords) = latlon_center_coords->get_lon();
}

void latlon_basin_evaluation_algorithm::
	set_previous_cells_local_redirect_index(coords* initial_fine_coords,coords* catchment_center_coords) {
	latlon_coords* latlon_catchment_center_coords =
		static_cast<latlon_coords*>(catchment_center_coords);
	(*local_redirect_lat_index)(initial_fine_coords) = latlon_catchment_center_coords->get_lat();
	(*local_redirect_lon_index)(initial_fine_coords) = latlon_catchment_center_coords->get_lon();
}

void latlon_basin_evaluation_algorithm::
	set_previous_cells_non_local_redirect_index(coords* initial_fine_coords,coords* target_coarse_coords) {
	latlon_coords* latlon_target_coarse_coords =
		static_cast<latlon_coords*>(target_coarse_coords);
	(*non_local_redirect_lat_index)(initial_fine_coords) = latlon_target_coarse_coords->get_lat();
	(*non_local_redirect_lon_index)(initial_fine_coords) = latlon_target_coarse_coords->get_lon();
}

coords* latlon_basin_evaluation_algorithm::get_cells_next_cell_index_as_coords(coords* coords_in){
		latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	return new latlon_coords((*next_cell_lat_index)(latlon_coords_in),
	                         (*next_cell_lon_index)(latlon_coords_in));
}
