/*
 * basin_evaluation_algorithm.cpp
 *
 *  Created on: Feb 21, 2018
 *      Author: thomasriddick
 */

#include "basin_evaluation_algorithm.hpp"
#include <queue>

basin_evaluation_algorithm::~basin_evaluation_algorithm() {
	delete minima; delete raw_orography; delete corrected_orography;
	delete connection_volume_thresholds; delete flood_volume_thresholds;
	delete basin_catchment_numbers; delete _grid; delete _coarse_grid;
	delete prior_fine_catchments; delete coarse_catchment_nums;
	delete requires_flood_redirect_indices;
	delete requires_connect_redirect_indices; delete flooded_cells;
	delete connected_cells; delete completed_cells;
	delete search_completed_cells; delete merge_points;
	delete flood_local_redirect; delete connect_local_redirect;
}

latlon_basin_evaluation_algorithm::~latlon_basin_evaluation_algorithm(){
	delete prior_fine_rdirs; delete flood_next_cell_lat_index;
	delete flood_next_cell_lon_index; delete connect_next_cell_lat_index;
	delete connect_next_cell_lon_index; delete flood_force_merge_lat_index;
	delete flood_force_merge_lon_index; delete connect_force_merge_lat_index;
	delete connect_force_merge_lon_index; delete flood_redirect_lat_index;
	delete flood_redirect_lon_index; delete connect_redirect_lat_index;
	delete connect_redirect_lon_index;
}

void basin_evaluation_algorithm::setup_fields(bool* minima_in,
		  	  	  	  	  	  	  	  	  				double* raw_orography_in,
		  	  	  	  	  	  	  	  	  				double* corrected_orography_in,
		  	  	  	  	  	  	  	  	  				double* connection_volume_thresholds_in,
		  	  	  	  	  	  	  	  	  				double* flood_volume_thresholds_in,
		  	  	  	  	  	  	  	  	  				int* prior_fine_catchments_in,
		  	  	  	  	  	  	  	  	  				int* coarse_catchment_nums_in,
		  	  	  	  	  	  	  	  	  				bool* flood_local_redirect_in,
		  	  	  	  	  	  	  	  	  				bool* connect_local_redirect_in,
		  	  	  	  	  	  	  	  	  				merge_types* merge_points_in,
										  												grid_params* grid_params_in,
										  												grid_params* coarse_grid_params_in) {
	_grid_params = grid_params_in;
	_coarse_grid_params = coarse_grid_params_in;
	_grid = grid_factory(_grid_params);
	_coarse_grid = grid_factory(_coarse_grid_params);
	minima = new field<bool>(minima_in,_grid_params);
	raw_orography = new field<double>(raw_orography_in,_grid_params);
	corrected_orography =
		new field<double>(corrected_orography_in,_grid_params);
	connection_volume_thresholds = new field<double>(connection_volume_thresholds_in,_grid_params);
	connection_volume_thresholds->set_all(-1.0);
	flood_volume_thresholds = new field<double>(flood_volume_thresholds_in,_grid_params);
	flood_volume_thresholds->set_all(-1.0);
	prior_fine_catchments = new field<int>(prior_fine_catchments_in,_grid_params);
	flood_local_redirect = new field<bool>(flood_local_redirect_in,_grid_params);
	connect_local_redirect = new field<bool>(connect_local_redirect_in,_grid_params);
	merge_points = new field<merge_types>(merge_points_in,_grid_params);
	merge_points->set_all(no_merge);
	coarse_catchment_nums = new field<int>(coarse_catchment_nums_in,_coarse_grid_params);
	completed_cells = new field<bool>(_grid_params);
	search_completed_cells = new field<bool>(_coarse_grid_params);
	requires_flood_redirect_indices = new field<bool>(_grid_params);
	requires_flood_redirect_indices->set_all(false);
	requires_connect_redirect_indices = new field<bool>(_grid_params);
	requires_connect_redirect_indices->set_all(false);
	basin_catchment_numbers = new field<int>(_grid_params);
	basin_catchment_numbers->set_all(null_catchment);
	flooded_cells = new field<bool>(_grid_params);
	flooded_cells->set_all(false);
	connected_cells = new field<bool>(_grid_params);
	connected_cells->set_all(false);
	basin_flooded_cells = new field<bool>(_grid_params);
	basin_connected_cells = new field<bool>(_grid_params);
}

void latlon_basin_evaluation_algorithm::setup_fields(bool* minima_in,
		  	  	  	  	  	  	  	  	  							 double* raw_orography_in,
		  	  	  	  	  	  	  	  	  							 double* corrected_orography_in,
		  	  	  	  	  	  	  	  	  							 double* connection_volume_thresholds_in,
		  	  	  	  	  	  	  	  	  							 double* flood_volume_thresholds_in,
		  	  	  	  	  	  	  	  	  							 double* prior_fine_rdirs_in,
		  	  	  	  	  	  	  	  	  							 int* prior_fine_catchments_in,
		  	  	  	  	  	  	  	  	  							 int* coarse_catchment_nums_in,
		  	  	  	  	  	  	  	  	  							 int* flood_next_cell_lat_index_in,
		  	  	  	  	  	  	  	  	  							 int* flood_next_cell_lon_index_in,
		  	  	  	  	  	  	  	  	  							 int* connect_next_cell_lat_index_in,
		  	  	  	  	  	  	  	  	  							 int* connect_next_cell_lon_index_in,
		  	  	  	  	  	  	  	  	  							 int* flood_force_merge_lat_index_in,
		  	  	  	  	  	  	  	  	  							 int* flood_force_merge_lon_index_in,
		  	  	  	  	  	  	  	  	  							 int* connect_force_merge_lat_index_in,
		  	  	  	  	  	  	  	  	  							 int* connect_force_merge_lon_index_in,
		  	  	  	  	  	  	  	  	  							 int* flood_redirect_lat_index_in,
		  	  	  	  	  	  	  	  	  							 int* flood_redirect_lon_index_in,
		  	  	  	  	  	  	  	  	  							 int* connect_redirect_lat_index_in,
		  	  	  	  	  	  	  	  	  							 int* connect_redirect_lon_index_in,
		  	  	  	  	  	  	  	  	  							 bool* flood_local_redirect_in,
		  	  	  	  	  	  	  	  	  							 bool* connect_local_redirect_in,
		  	  	  	  	  	  	  	  	  							 merge_types* merge_points_in,
										  															 grid_params* grid_params_in,
										  															 grid_params* coarse_grid_params_in)
{
	basin_evaluation_algorithm::setup_fields(minima_in,raw_orography_in,
	                                         corrected_orography_in,
	                                         connection_volume_thresholds_in,
	                                         flood_volume_thresholds_in,
		  	  	   														 prior_fine_catchments_in,
		  	  	   														 coarse_catchment_nums_in,
		  	  	   														 flood_local_redirect_in,
		  	  	  	  	  	  	  	  	  		 connect_local_redirect_in,
		  	  	  	  	  	  	  	  	  		 merge_points_in,
		  	  	  	  	  	  	  	  	  		 grid_params_in,
		  	  	  	  	  	  	  	  	  		 coarse_grid_params_in);
	prior_fine_rdirs = new field<double>(prior_fine_rdirs_in,grid_params_in);
	flood_next_cell_lat_index = new field<int>(flood_next_cell_lat_index_in,grid_params_in);
	flood_next_cell_lon_index = new field<int>(flood_next_cell_lon_index_in,grid_params_in);
	connect_next_cell_lat_index = new field<int>(connect_next_cell_lat_index_in,grid_params_in);
	connect_next_cell_lon_index = new field<int>(connect_next_cell_lon_index_in,grid_params_in);
	flood_force_merge_lat_index = new field<int>(flood_force_merge_lat_index_in,grid_params_in);
	flood_force_merge_lon_index = new field<int>(flood_force_merge_lon_index_in,grid_params_in);
	connect_force_merge_lat_index = new field<int>(connect_force_merge_lat_index_in,grid_params_in);
	connect_force_merge_lon_index = new field<int>(connect_force_merge_lon_index_in,grid_params_in);
	flood_redirect_lat_index = new field<int>(flood_redirect_lat_index_in,grid_params_in);
	flood_redirect_lon_index = new field<int>(flood_redirect_lon_index_in,grid_params_in);
	connect_redirect_lat_index = new field<int>(connect_redirect_lat_index_in,grid_params_in);
	connect_redirect_lon_index = new field<int>(connect_redirect_lon_index_in,grid_params_in);
}

void basin_evaluation_algorithm::evaluate_basins(){
	add_minima_to_queue();
	basin_catchment_number = 1;
	while (! minima_q.empty()) {
		minimum = static_cast<basin_cell*>(minima_q.top());
		minima_q.pop();
		evaluate_basin();
		basin_catchment_number++;
		delete minimum;
	}
	//set_remaining_redirects();
	while ( ! basin_catchment_centers.empty()){
		coords* catchment_center = basin_catchment_centers.back();
		basin_catchment_centers.pop_back();
		delete catchment_center;
	}
}

void basin_evaluation_algorithm::add_minima_to_queue() {
	_grid->for_all([&](coords* coords_in){
		if ((*minima)(coords_in)) {
			coords* coarse_minimum_coords =
				_coarse_grid->convert_fine_coords(coords_in,_grid_params);
			double height = 0.0;
			height_types height_type;
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
			delete coarse_minimum_coords;
		} else delete coords_in;
	});
}

void basin_evaluation_algorithm::evaluate_basin(){
	center_cell = minimum->clone();
	completed_cells->set_all(false);
	basin_flooded_cells->set_all(false);
	basin_connected_cells->set_all(false);
	center_cell_volume_threshold = 0.0;
	center_coords = center_cell->get_cell_coords()->clone();
	center_cell_height_type = center_cell->get_height_type();
	center_cell_height = center_cell->get_orography();
	surface_height = center_cell_height;
	previous_filled_cell_coords = center_coords->clone();
	previous_filled_cell_height_type = center_cell_height_type;
	previous_filled_cell_height = center_cell_height;
	read_new_center_cell_variables();
	if (center_cell_height_type == connection_height) {
		(*connected_cells)(center_coords) = true;
		(*basin_connected_cells)(center_coords) = true;
		cell_number = 0;
	} else if (center_cell_height_type == flood_height) {
		(*flooded_cells)(center_coords) = true;
		(*basin_flooded_cells)(center_coords) = true;
		cell_number = 1;
	} else throw runtime_error("Cell type not recognized");
	(*completed_cells)(center_coords) = true;
	//Make partial first iteration
	process_neighbors();
	delete center_cell;
	while( ! q.empty()){
		center_cell = static_cast<basin_cell*>(q.top());
		q.pop();
		//Call the newly loaded coordinates and height for the center cell 'new'
		//until after making the test for merges then relabel. Center cell height/coords
		//without the 'new' moniker refers to the previous center cell; previous center cell
		//height/coords the previous previous center cell
		read_new_center_cell_variables();
		if (possible_merge_point_reached()) {
			if ((*raw_orography)(center_coords) <= (*corrected_orography)(center_coords) ||
			    center_cell_height_type == connection_height) {
				if((*basin_catchment_numbers)(new_center_coords) == null_catchment) {
					set_merge_type(merge_as_secondary);
					set_secondary_redirect();
					basin_catchment_centers.push_back(minimum->get_cell_coords()->clone());
					if (center_cell_height_type == connection_height) (*connected_cells)(center_coords) = false;
					else (*flooded_cells)(center_coords) = false;
					delete center_cell; delete new_center_coords;
					delete center_coords; delete previous_filled_cell_coords;
					break;
				} else {
					set_primary_merge();
					set_primary_redirect();
					rebuild_secondary_basin(new_center_coords);
				}
			}
		}
		process_neighbors();
		//need height type to check for fetch here before checking for
		//skip and then updating all values for center cell to be those
		//of the new center cell
		if ( ! skipped_previous_center_cell) update_previous_filled_cell_variables();
		update_center_cell_variables();
		if (! skip_center_cell()) {
			process_center_cell();
			skipped_previous_center_cell = false;
		}
		delete center_cell;
	}
	while (! q.empty()){
		center_cell = static_cast<basin_cell*>(q.top());
		q.pop();
		center_coords = center_cell->get_cell_coords();
		(*completed_cells)(center_coords) = false;
		delete center_cell;
	}
}

inline void basin_evaluation_algorithm::read_new_center_cell_variables(){
		new_center_coords = center_cell->get_cell_coords()->clone();
		new_center_cell_height_type = center_cell->get_height_type();
		new_center_cell_height = center_cell->get_orography();
}

inline void basin_evaluation_algorithm::update_previous_filled_cell_variables(){
		if (previous_filled_cell_height_type == flood_height) {
			delete previous_filled_cell_coords;
		}
		previous_filled_cell_coords = center_coords;
		previous_filled_cell_height = center_cell_height;
		previous_filled_cell_height_type = center_cell_height_type;
		skipped_previous_center_cell = true;
}

inline void basin_evaluation_algorithm::update_center_cell_variables(){
		center_cell_height_type = new_center_cell_height_type;
		center_cell_height = new_center_cell_height;
		center_coords = new_center_coords;
		if (center_cell_height > surface_height) surface_height = center_cell_height;
}

inline bool basin_evaluation_algorithm::possible_merge_point_reached(){
	bool potential_catchment_edge_in_flat_area_found =
			    ((*flooded_cells)(new_center_coords) &&
		      		new_center_cell_height_type == flood_height) ||
		     	((*connected_cells)(new_center_coords) &&
		      	  new_center_cell_height_type == connection_height);
	bool already_in_basin = ((*basin_flooded_cells)(new_center_coords) &&
		       								new_center_cell_height_type == flood_height) ||
		     									((*basin_connected_cells)(new_center_coords) &&
		        							new_center_cell_height_type == connection_height);
	return ((new_center_cell_height < surface_height ||
		    	(new_center_cell_height == surface_height &&
		    	 potential_catchment_edge_in_flat_area_found))
		    	&& ! already_in_basin);
}

bool basin_evaluation_algorithm::skip_center_cell() {
	// here the variable center cell height type is actually
	// new center cell height type
	if((*basin_flooded_cells)(center_coords)) {
		cell_number++;
		//delete center_coords;
		return true;
	} else if (center_cell_height_type == connection_height &&
	           (*basin_connected_cells)(center_coords)){
		q.push(new basin_cell((*raw_orography)(center_coords),
					 							  flood_height,center_coords));
		return true;
	}
	return false;
}

//notice that the previous filled cell is not necessarily
//the previous cell and the cell number doesn't include
//connection cells that are not yet filled
void basin_evaluation_algorithm::process_center_cell() {
	if ((*basin_catchment_numbers)(previous_filled_cell_coords)
			== null_catchment) {
		(*basin_catchment_numbers)(previous_filled_cell_coords) =
			basin_catchment_number;
	}
	center_cell_volume_threshold =
		center_cell_volume_threshold +
			cell_number*(center_cell_height-previous_filled_cell_height);
	if (previous_filled_cell_height_type == connection_height) {
		(*connection_volume_thresholds)(previous_filled_cell_coords) =
			center_cell_volume_threshold;
		q.push(new basin_cell((*raw_orography)(previous_filled_cell_coords),
		                      flood_height,previous_filled_cell_coords));
		set_previous_cells_connect_next_cell_index();
	} else if (previous_filled_cell_height_type == flood_height) {
		(*flood_volume_thresholds)(previous_filled_cell_coords) =
			center_cell_volume_threshold;
		set_previous_cells_flood_next_cell_index();
	} else throw runtime_error("Cell type not recognized");
	if (center_cell_height_type == connection_height) {
		(*connected_cells)(center_coords) = true;
		(*basin_connected_cells)(center_coords) = true;
	}
	else if (center_cell_height_type == flood_height) {
		(*flooded_cells)(center_coords) = true;
		(*basin_flooded_cells)(center_coords) = true;
		cell_number++;
	} else throw runtime_error("Cell type not recognized");
}

void basin_evaluation_algorithm::process_neighbors() {
	neighbors_coords = raw_orography->get_neighbors_coords(new_center_coords,1);
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

void basin_evaluation_algorithm::set_primary_merge(){
	set_merge_type(merge_as_primary);
	if (previous_filled_cell_height_type == flood_height) {
		set_previous_cells_flood_force_merge_index();
	} else if (previous_filled_cell_height_type == connection_height) {
		set_previous_cells_connect_force_merge_index();
	} else throw runtime_error("Cell type not recognized");
}

void basin_evaluation_algorithm::set_merge_type(basic_merge_types current_merge_type) {
	merge_types prior_merge_type = (*merge_points)(previous_filled_cell_coords);
	merge_types new_merge_type;
	switch (prior_merge_type) {
		case no_merge:
			if (previous_filled_cell_height_type == flood_height) {
				if (current_merge_type == merge_as_primary)
					new_merge_type = connection_merge_not_set_flood_merge_as_primary;
				else if (current_merge_type == merge_as_secondary){
					new_merge_type = connection_merge_not_set_flood_merge_as_secondary;
				} else throw runtime_error("Merge type not recognized");
			} else if (previous_filled_cell_height_type == connection_height) {
				if (current_merge_type == merge_as_primary)
					new_merge_type = connection_merge_as_primary_flood_merge_not_set;
				else if (current_merge_type == merge_as_secondary){
					new_merge_type = connection_merge_as_secondary_flood_merge_not_set;
				} else throw runtime_error("Merge type not recognized");
			} else throw runtime_error("Cell type not recognized");
			break;
		case connection_merge_as_primary_flood_merge_not_set:
			if (previous_filled_cell_height_type == flood_height) {
				if (current_merge_type == merge_as_primary)
					new_merge_type = connection_merge_as_primary_flood_merge_as_primary;
				else if (current_merge_type == merge_as_secondary){
					new_merge_type = connection_merge_as_primary_flood_merge_as_secondary;
				} else throw runtime_error("Merge type not recognized");
			} else if (previous_filled_cell_height_type == connection_height) {
				throw runtime_error("Trying to overwrite connection merge type");
			} else throw runtime_error("Cell type not recognized");
			break;
		case connection_merge_as_secondary_flood_merge_not_set:
			if (previous_filled_cell_height_type == flood_height) {
				if (current_merge_type == merge_as_primary)
					new_merge_type = connection_merge_as_secondary_flood_merge_as_primary;
				else if (current_merge_type == merge_as_secondary){
					new_merge_type = connection_merge_as_secondary_flood_merge_as_secondary;
				} else throw runtime_error("Merge type not recognized");
			} else if (previous_filled_cell_height_type == connection_height) {
				throw runtime_error("Trying to overwrite connection merge type");
			} else throw runtime_error("Cell type not recognized");
			break;
		default:
		throw runtime_error("Merge type not recognized");
	}
	(*merge_points)(previous_filled_cell_coords) = new_merge_type;
}

basic_merge_types basin_evaluation_algorithm::get_merge_type(height_types height_type_in,coords* coords_in) {
	merge_types current_merge_type = (*merge_points)(coords_in);
	if (height_type_in == connection_height) {
		switch (current_merge_type) {
			case no_merge :
			case connection_merge_not_set_flood_merge_as_primary :
      case connection_merge_not_set_flood_merge_as_secondary :
      	return basic_no_merge;
			case connection_merge_as_primary_flood_merge_as_primary :
      case connection_merge_as_primary_flood_merge_as_secondary :
      case connection_merge_as_primary_flood_merge_not_set :
      	return merge_as_primary;
			case connection_merge_as_secondary_flood_merge_as_primary :
			case connection_merge_as_secondary_flood_merge_as_secondary :
			case connection_merge_as_secondary_flood_merge_not_set :
				return merge_as_secondary;
			case null_mtype:
				throw runtime_error("No merge type defined for these coordinates");
			default:
				throw runtime_error("Merge type not recognized");
		}
	} else if(height_type_in == flood_height) {
		switch (current_merge_type) {
			case no_merge :
			case connection_merge_as_primary_flood_merge_not_set :
			case connection_merge_as_secondary_flood_merge_not_set :
      	return basic_no_merge;
			case connection_merge_as_primary_flood_merge_as_primary :
			case connection_merge_as_secondary_flood_merge_as_primary :
			case connection_merge_not_set_flood_merge_as_primary :
      	return merge_as_primary;
      case connection_merge_as_primary_flood_merge_as_secondary :
			case connection_merge_as_secondary_flood_merge_as_secondary :
      case connection_merge_not_set_flood_merge_as_secondary :
      	return merge_as_secondary;
			case null_mtype:
				throw runtime_error("No merge type defined for these coordinates");
			default:
				throw runtime_error("Merge type not recognized");
		}
	} else throw runtime_error("Merge type not recognized");
}

void basin_evaluation_algorithm::rebuild_secondary_basin(coords* initial_coords){
	int secondary_basin_catchment_number = (*basin_catchment_numbers)(initial_coords);
	coords* current_coords = basin_catchment_centers[secondary_basin_catchment_number - 1];
	height_types current_height_type =  (*raw_orography)(current_coords) <= (*corrected_orography)(current_coords) ?
		 flood_height : connection_height;
	while(true){
		if(current_height_type == flood_height)
			(*basin_flooded_cells)(current_coords) = true;
		else if (current_height_type == connection_height)
			(*basin_connected_cells)(current_coords) = true;
		else throw runtime_error("Height type not recognized");
		basic_merge_types current_coords_basic_merge_type =
			get_merge_type(current_height_type,current_coords);
		if (current_coords_basic_merge_type == merge_as_primary){
			coords* new_initial_coords =
				get_cells_next_force_merge_index_as_coords(current_coords,
                                                   current_height_type);
			if((current_height_type == flood_height &&
			    ! (*basin_flooded_cells)(new_initial_coords)) ||
			   (current_height_type == connection_height &&
			    ! (*basin_connected_cells)(new_initial_coords))) {
				rebuild_secondary_basin(new_initial_coords);
			}
		} else if(current_coords_basic_merge_type == merge_as_secondary) {
			//this isn't the cells actual redirect index; the redirect index array is simply being
			//used as temporary storage
			coords* next_cell_coords = get_cells_redirect_index_as_coords(current_coords,
                                                      	 	 	 				current_height_type);
			if ((*basin_catchment_numbers)(next_cell_coords) == basin_catchment_number ||
			    ((*basin_flooded_cells)(next_cell_coords) &&
			     ((*basin_connected_cells)(next_cell_coords) ||
			      (*raw_orography)(next_cell_coords) <= (*corrected_orography)(next_cell_coords)))) {
				//delete next_cell_coords;
				return;
			}
			else if ((*basin_catchment_numbers)(next_cell_coords) == 0)
				return;
			else {
				//delete current_coords;
				rebuild_secondary_basin(next_cell_coords);
			}
		}
		current_coords = get_cells_next_cell_index_as_coords(current_coords,
                                                      	 current_height_type);
		if ((*raw_orography)(current_coords) <= (*corrected_orography)(current_coords) ||
		    (*basin_connected_cells)(current_coords)) current_height_type = flood_height;
		else  current_height_type = connection_height;
	}
}

void basin_evaluation_algorithm::set_secondary_redirect(){
	if (previous_filled_cell_height_type == flood_height) {
		set_previous_cells_flood_next_cell_index();
		(*requires_flood_redirect_indices)(previous_filled_cell_coords) = true;
	} else if (previous_filled_cell_height_type == connection_height){
		set_previous_cells_connect_next_cell_index();
		(*requires_connect_redirect_indices)(previous_filled_cell_coords) = true;
	} else throw runtime_error("Cell type not recognized");
	//Store information on the next downstream cell in the redirect index for
	//later recovery if necessary. We are not actually setting the redirect
	//index here merely using the array of redirect values as storage space
	set_previous_cells_redirect_index(previous_filled_cell_coords,
		                                new_center_coords,
		                                previous_filled_cell_height_type);
}

void basin_evaluation_algorithm::set_primary_redirect(){
	coords* basin_catchment_center_coords =
		basin_catchment_centers[(*basin_catchment_numbers)(new_center_coords) - 1];
	if(_coarse_grid->fine_coords_in_same_cell(center_coords,
			basin_catchment_center_coords,_grid_params)) {
		  set_previous_cells_redirect_type(previous_filled_cell_coords,
		                                   previous_filled_cell_height_type,
		                                   local_redirect);
		  set_previous_cells_redirect_index(previous_filled_cell_coords,
		                                    basin_catchment_center_coords,
		                                    previous_filled_cell_height_type);
	} else {
		  find_and_set_previous_cells_non_local_redirect_index(previous_filled_cell_coords,
		                                                       center_coords,
		                                                       basin_catchment_center_coords,
		                                                       previous_filled_cell_height_type);
	}
}

void basin_evaluation_algorithm::
	find_and_set_previous_cells_non_local_redirect_index(coords* initial_center_coords,
	                                                     coords* current_center_coords,
	                                                     coords* catchment_center_coords,
	                                                     height_types initial_center_height_type){
	coords* catchment_center_coarse_coords = _coarse_grid->convert_fine_coords(catchment_center_coords,
	                                                                           _grid_params);
	int coarse_catchment_num = (*coarse_catchment_nums)(catchment_center_coarse_coords);
	find_and_set_non_local_redirect_index_from_coarse_catchment_num(initial_center_coords,
	                                                                current_center_coords,
	                                                                initial_center_height_type,
	                                                                coarse_catchment_num);
	delete catchment_center_coarse_coords;
}

void basin_evaluation_algorithm::
	find_and_set_non_local_redirect_index_from_coarse_catchment_num(coords* initial_center_coords,
	                                                                coords* current_center_coords,
	                                                                height_types initial_center_height_type,
	                                                     						int coarse_catchment_number){
	coords* center_coarse_coords = _coarse_grid->convert_fine_coords(current_center_coords,
	                                                                 _grid_params);
	if((*coarse_catchment_nums)(center_coarse_coords) == coarse_catchment_number) {
		set_previous_cells_redirect_type(initial_center_coords,initial_center_height_type,
		                                 non_local_redirect);
		set_previous_cells_redirect_index(initial_center_coords,center_coarse_coords,
		                                  initial_center_height_type);
		delete center_coarse_coords;
	} else {
		search_q.push(new landsea_cell(center_coarse_coords));
		search_completed_cells->set_all(false);
		while (! search_q.empty()) {
			landsea_cell* search_cell = search_q.front();
			search_q.pop();
			search_coords = search_cell->get_cell_coords();
			if((*coarse_catchment_nums)(search_coords) ==
			   coarse_catchment_number){
				set_previous_cells_redirect_type(initial_center_coords,initial_center_height_type,
				                                 non_local_redirect);
				set_previous_cells_redirect_index(initial_center_coords,search_coords,
				                                  initial_center_height_type);
				delete search_cell;
				break;
			}
			search_process_neighbors();
			delete search_cell;
		}
		while (! search_q.empty()) {
			landsea_cell* search_cell = search_q.front();
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
	search_neighbors_coords->pop_back();
	if (!(*search_completed_cells)(search_nbr_coords)) {
		search_q.push(new landsea_cell(search_nbr_coords));
		(*search_completed_cells)(search_nbr_coords) = true;
	} else delete search_nbr_coords;
}

void basin_evaluation_algorithm::set_remaining_redirects() {
	_grid->for_all([&](coords* coords_in){
		bool cell_requires_flood_redirect_indices =
			(*requires_flood_redirect_indices)(coords_in);
		bool cell_requires_connect_redirect_indices =
			(*requires_connect_redirect_indices)(coords_in);
		while(cell_requires_flood_redirect_indices || cell_requires_connect_redirect_indices){
			height_types redirect_height_type;
			if (cell_requires_flood_redirect_indices) {
				redirect_height_type = flood_height;
				cell_requires_flood_redirect_indices = false;
			} else redirect_height_type = connection_height;
			//this isn't the cells actual redirect index; the redirect index array is simply being
			//used as temporary storage
			coords* first_cell_beyond_rim_coords = get_cells_redirect_index_as_coords(coords_in,
                                           												      	      redirect_height_type);
			if((*basin_catchment_numbers)(first_cell_beyond_rim_coords) != null_catchment){
				 coords* basin_catchment_center_coords =
				 	basin_catchment_centers[(*basin_catchment_numbers)(first_cell_beyond_rim_coords) - 1];
					if(_coarse_grid->fine_coords_in_same_cell(first_cell_beyond_rim_coords,
						basin_catchment_center_coords,_grid_params)) {
						set_previous_cells_redirect_type(coords_in,redirect_height_type,local_redirect);
						set_previous_cells_redirect_index(coords_in,basin_catchment_center_coords,
						                                  redirect_height_type);
					} else {
		  	 		find_and_set_previous_cells_non_local_redirect_index(coords_in,
						                                        						 first_cell_beyond_rim_coords,
		  	 		                                                     basin_catchment_center_coords,
		  	 		                                                     redirect_height_type);
					}
			} else {
				int prior_fine_catchment_num = (*prior_fine_catchments)(first_cell_beyond_rim_coords);
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
					delete coords_in_two;
				});
				int coarse_catchment_number =
					(*coarse_catchment_nums)(catchment_outlet_coarse_coords);
				find_and_set_non_local_redirect_index_from_coarse_catchment_num(coords_in,
						                                        						 				first_cell_beyond_rim_coords,
						                                        						 				redirect_height_type,
	                                                     							 		coarse_catchment_number);
				delete catchment_outlet_coarse_coords;
			}
		delete first_cell_beyond_rim_coords;
		}
		delete coords_in;
	});
}

reverse_priority_cell_queue basin_evaluation_algorithm::
														test_add_minima_to_queue(double* raw_orography_in,
                                                   	 double* corrected_orography_in,
                                                     bool* minima_in,
                                                     grid_params* grid_params_in,
                                                     grid_params* coarse_grid_params_in){
	_grid_params = grid_params_in;
	_coarse_grid_params =  coarse_grid_params_in;
	_grid = grid_factory(_grid_params);
	_coarse_grid = grid_factory(_coarse_grid_params);
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
	new_center_coords = center_coords_in;
	_grid_params = grid_params_in;
	_grid = grid_factory(_grid_params);
	raw_orography = new field<double>(raw_orography_in,_grid_params);
	corrected_orography =
		new field<double>(corrected_orography_in,_grid_params);
	completed_cells = new field<bool>(completed_cells_in,_grid_params);
	process_neighbors();
	delete raw_orography; delete corrected_orography; delete completed_cells; delete _grid;
	raw_orography = nullptr; corrected_orography = nullptr; completed_cells = nullptr;
	_grid = nullptr;
	return q;
}

queue<landsea_cell*> basin_evaluation_algorithm::
	test_search_process_neighbors(coords* search_coords_in,bool* search_completed_cells_in,
	                              grid_params* grid_params_in){
	_grid_params = grid_params_in;
	search_completed_cells = new field<bool>(search_completed_cells_in,_grid_params);
	search_coords = search_coords_in;
	search_process_neighbors();
	delete search_completed_cells;
	search_completed_cells = nullptr;
	return search_q;
}

void basin_evaluation_algorithm::
	set_previous_cells_redirect_type(coords* initial_fine_coords,height_types height_type,
	                                 redirect_type local_redirect) {
	if (height_type == connection_height) (*connect_local_redirect)(initial_fine_coords) = bool(local_redirect);
	else if (height_type == flood_height) (*flood_local_redirect)(initial_fine_coords) = bool(local_redirect);
	else throw runtime_error("Height type not recognized");
}

priority_cell_queue latlon_basin_evaluation_algorithm::test_process_center_cell(basin_cell* center_cell_in,
                                                                                coords* center_coords_in,
                                                                                coords* previous_filled_cell_coords_in,
                                               													 				double* flood_volume_thresholds_in,
                                               													 				double* connection_volume_thresholds_in,
                                               													 				double* raw_orography_in,
                                               													 				int* flood_next_cell_lat_index_in,
                                               													 				int* flood_next_cell_lon_index_in,
                                               													 				int* connect_next_cell_lat_index_in,
                                               													 				int* connect_next_cell_lon_index_in,
                                               													 				int* basin_catchment_numbers_in,
                                               													 				bool* flooded_cells_in,
                                               													 				bool* connected_cells_in,
                                               													 				double& center_cell_volume_threshold_in,
                                               													 				int& cell_number_in,
                                               													 				int basin_catchment_number_in,
                                               													 				double center_cell_height_in,
                                               													 				double& previous_filled_cell_height_in,
                                               													 				height_types& previous_filled_cell_height_type_in,
                                               													 				grid_params* grid_params_in){
	previous_filled_cell_coords = previous_filled_cell_coords_in;
	center_coords = center_coords_in;
	center_cell = center_cell_in;
	flood_volume_thresholds = new field<double>(flood_volume_thresholds_in,grid_params_in);
	connection_volume_thresholds = new field<double>(connection_volume_thresholds_in,grid_params_in);
	raw_orography = new field<double>(raw_orography_in,grid_params_in);
	flood_next_cell_lat_index = new field<int>(flood_next_cell_lat_index_in,grid_params_in);
	flood_next_cell_lon_index = new field<int>(flood_next_cell_lon_index_in,grid_params_in);
	connect_next_cell_lat_index = new field<int>(connect_next_cell_lat_index_in,grid_params_in);
	connect_next_cell_lon_index = new field<int>(connect_next_cell_lon_index_in,grid_params_in);
	basin_catchment_numbers = new field<int>(basin_catchment_numbers_in,grid_params_in);
	flooded_cells = new field<bool>(flooded_cells_in,grid_params_in);
	flooded_cells->set_all(false);
	connected_cells = new field<bool>(connected_cells_in,grid_params_in);
	connected_cells->set_all(false);
	basin_flooded_cells = new field<bool>(grid_params_in);
	basin_flooded_cells->set_all(false);
	basin_connected_cells = new field<bool>(grid_params_in);
	basin_connected_cells->set_all(false);
	center_cell_volume_threshold = center_cell_volume_threshold_in;
	cell_number = cell_number_in;
	basin_catchment_number = basin_catchment_number_in;
  center_cell_height = center_cell_height_in;
  previous_filled_cell_height = previous_filled_cell_height_in;
	center_cell_height_type = center_cell->get_height_type();
	previous_filled_cell_height_type = previous_filled_cell_height_type_in;
  process_center_cell();
	center_cell_volume_threshold_in = center_cell_volume_threshold;
	cell_number_in = cell_number;
	previous_filled_cell_height_in = previous_filled_cell_height;
	previous_filled_cell_height_type_in = previous_filled_cell_height_type;
	return q;
}

void latlon_basin_evaluation_algorithm::
		 test_set_primary_merge_and_redirect(vector<coords*> basin_catchment_centers_in,
                                         int* basin_catchment_numbers_in,
                                         int* coarse_catchment_nums_in,
                                         int* flood_force_merge_lat_index_in,
                                         int* flood_force_merge_lon_index_in,
                                         int* connect_force_merge_lat_index_in,
                                         int* connect_force_merge_lon_index_in,
                                         int* flood_redirect_lat_index_in,
                                         int* flood_redirect_lon_index_in,
                                         int* connect_redirect_lat_index_in,
                                         int* connect_redirect_lon_index_in,
                                         bool* flood_local_redirect_in,
		  	  	  	  	  	 							   bool* connect_local_redirect_in,
		  	  	  	  	  	 							   merge_types* merge_points_in,
                                         coords* new_center_coords_in,
                                         coords* center_coords_in,
                                         coords* previous_filled_cell_coords_in,
                                         height_types previous_filled_cell_height_type_in,
                                         grid_params* grid_params_in,
                                         grid_params* coarse_grid_params_in) {
	_grid_params = grid_params_in;
	_coarse_grid_params = coarse_grid_params_in;
	basin_catchment_centers = basin_catchment_centers_in;
	basin_catchment_numbers = new field<int>(basin_catchment_numbers_in,grid_params_in);
	coarse_catchment_nums = new field<int>(coarse_catchment_nums_in,coarse_grid_params_in);
	flood_force_merge_lat_index = new field<int>(flood_force_merge_lat_index_in,grid_params_in);
	flood_force_merge_lon_index = new field<int>(flood_force_merge_lon_index_in,grid_params_in);
	connect_force_merge_lat_index = new field<int>(connect_force_merge_lat_index_in,grid_params_in);
	connect_force_merge_lon_index = new field<int>(connect_force_merge_lon_index_in,grid_params_in);
	flood_redirect_lat_index = new field<int>(flood_redirect_lat_index_in,grid_params_in);
	flood_redirect_lon_index = new field<int>(flood_redirect_lon_index_in,grid_params_in);
	connect_redirect_lat_index = new field<int>(connect_redirect_lat_index_in,grid_params_in);
	connect_redirect_lon_index = new field<int>(connect_redirect_lon_index_in,grid_params_in);
	flood_local_redirect = new field<bool>(flood_local_redirect_in,_grid_params);
	connect_local_redirect = new field<bool>(connect_local_redirect_in,_grid_params);
	search_completed_cells = new field<bool>(coarse_grid_params_in);
	merge_points = new field<merge_types>(merge_points_in,_grid_params);
	center_coords = center_coords_in;
	new_center_coords = new_center_coords_in;
	previous_filled_cell_coords = previous_filled_cell_coords_in;
	previous_filled_cell_height_type = previous_filled_cell_height_type_in;
	_grid = grid_factory(_grid_params);
	_coarse_grid = grid_factory(_coarse_grid_params);
	set_primary_merge();
	set_primary_redirect();
}

void latlon_basin_evaluation_algorithm::test_set_secondary_redirect(int* flood_next_cell_lat_index_in,
                                                                    int* flood_next_cell_lon_index_in,
                                                                    int* connect_next_cell_lat_index_in,
                                                                    int* connect_next_cell_lon_index_in,
                                                                    int* flood_redirect_lat_in,
                                                                    int* flood_redirect_lon_in,
                                                                    int* connect_redirect_lat_in,
                                                                    int* connect_redirect_lon_in,
                                                                    bool* requires_flood_redirect_indices_in,
                                                                    bool* requires_connect_redirect_indices_in,
                                                                    double* raw_orography_in,
                    																								double* corrected_orography_in,
                    																								coords* new_center_coords_in,
                                                                  	coords* center_coords_in,
                                                                  	coords* previous_filled_cell_coords_in,
                                                                  	height_types& previous_filled_cell_height_type_in,
                                                                  	grid_params* grid_params_in){
	center_coords = center_coords_in;
	previous_filled_cell_coords = previous_filled_cell_coords_in;
	flood_next_cell_lat_index = new field<int>(flood_next_cell_lat_index_in,grid_params_in);
	flood_next_cell_lon_index = new field<int>(flood_next_cell_lon_index_in,grid_params_in);
	connect_next_cell_lat_index = new field<int>(connect_next_cell_lat_index_in,grid_params_in);
	connect_next_cell_lon_index = new field<int>(connect_next_cell_lon_index_in,grid_params_in);
  requires_flood_redirect_indices = new field<bool>(requires_flood_redirect_indices_in,grid_params_in);
  requires_connect_redirect_indices = new field<bool>(requires_connect_redirect_indices_in,grid_params_in);
  raw_orography = new field<double>(raw_orography_in,grid_params_in);
  corrected_orography = new field<double>(corrected_orography_in,grid_params_in);
  previous_filled_cell_height_type = previous_filled_cell_height_type_in;
  new_center_coords = new_center_coords_in;
  flood_redirect_lat_index =  new field<int>(flood_redirect_lat_in,grid_params_in);
  flood_redirect_lon_index =  new field<int>(flood_redirect_lon_in,grid_params_in);
  connect_redirect_lat_index =  new field<int>(connect_redirect_lat_in,grid_params_in);
  connect_redirect_lon_index =  new field<int>(connect_redirect_lon_in,grid_params_in);
  set_secondary_redirect();
}

void latlon_basin_evaluation_algorithm::test_set_remaining_redirects(vector<coords*> basin_catchment_centers_in,
                                                                     double* prior_fine_rdirs_in,
                                                                     bool* requires_flood_redirect_indices_in,
                                                                     bool* requires_connect_redirect_indices_in,
                                                                     int* basin_catchment_numbers_in,
                                                                     int* prior_fine_catchments_in,
                                                                     int* coarse_catchment_nums_in,
                                                                     int* flood_next_cell_lat_index_in,
                                                                     int* flood_next_cell_lon_index_in,
                                                                     int* connect_next_cell_lat_index_in,
                                                                     int* connect_next_cell_lon_index_in,
                                                                     int* flood_redirect_lat_index_in,
                                                                     int* flood_redirect_lon_index_in,
                                                                     int* connect_redirect_lat_index_in,
                                    																 int* connect_redirect_lon_index_in,
                                                                  	 bool* flood_local_redirect_in,
		  	  	  	  	  	  	  	  	  													  	 bool* connect_local_redirect_in,
                                                                  	 grid_params* grid_params_in,
                                                                  	 grid_params* coarse_grid_params_in){
	_grid_params = grid_params_in;
	_coarse_grid_params = coarse_grid_params_in;
	basin_catchment_centers = basin_catchment_centers_in;
	prior_fine_rdirs = new field<double>(prior_fine_rdirs_in,grid_params_in);
	requires_flood_redirect_indices = new field<bool>(requires_flood_redirect_indices_in,grid_params_in);
	requires_connect_redirect_indices = new field<bool>(requires_connect_redirect_indices_in,grid_params_in);
	basin_catchment_numbers = new field<int>(basin_catchment_numbers_in,grid_params_in);
	prior_fine_catchments = new field<int>(prior_fine_catchments_in,grid_params_in);
	coarse_catchment_nums = new field<int>(coarse_catchment_nums_in,coarse_grid_params_in);
	flood_next_cell_lat_index = new field<int>(flood_next_cell_lat_index_in,grid_params_in);
	flood_next_cell_lon_index = new field<int>(flood_next_cell_lon_index_in,grid_params_in);
	connect_next_cell_lat_index = new field<int>(connect_next_cell_lat_index_in,grid_params_in);
	connect_next_cell_lon_index = new field<int>(connect_next_cell_lon_index_in,grid_params_in);
	flood_redirect_lat_index = new field<int>(flood_redirect_lat_index_in,grid_params_in);
	flood_redirect_lon_index = new field<int>(flood_redirect_lon_index_in,grid_params_in);
	connect_redirect_lat_index = new field<int>(connect_redirect_lat_index_in,grid_params_in);
	connect_redirect_lon_index = new field<int>(connect_redirect_lon_index_in,grid_params_in);
	flood_local_redirect = new field<bool>(flood_local_redirect_in,_grid_params);
	connect_local_redirect = new field<bool>(connect_local_redirect_in,_grid_params);
	search_completed_cells = new field<bool>(grid_params_in);
	_grid = grid_factory(_grid_params);
	_coarse_grid = grid_factory(_coarse_grid_params);
	set_remaining_redirects();
}

bool latlon_basin_evaluation_algorithm::check_for_sinks(coords* coords_in){
	double rdir = (*prior_fine_rdirs)(coords_in);
	coords* next_coords = _grid->calculate_downstream_coords_from_dir_based_rdir(coords_in,rdir);
	double next_rdir = (*prior_fine_rdirs)(next_coords);
	delete next_coords;
	return (rdir == 5 || next_rdir == 0);
}

void latlon_basin_evaluation_algorithm::
	set_previous_cells_flood_next_cell_index(){
	latlon_coords* latlon_center_coords = static_cast<latlon_coords*>(center_coords);
	(*flood_next_cell_lat_index)(previous_filled_cell_coords) = latlon_center_coords->get_lat();
	(*flood_next_cell_lon_index)(previous_filled_cell_coords) = latlon_center_coords->get_lon();
}

void latlon_basin_evaluation_algorithm::
	set_previous_cells_connect_next_cell_index(){
	latlon_coords* latlon_center_coords = static_cast<latlon_coords*>(center_coords);
	(*connect_next_cell_lat_index)(previous_filled_cell_coords) = latlon_center_coords->get_lat();
	(*connect_next_cell_lon_index)(previous_filled_cell_coords) = latlon_center_coords->get_lon();
}

void latlon_basin_evaluation_algorithm::
	set_previous_cells_flood_force_merge_index(){
	latlon_coords* latlon_new_center_coords = static_cast<latlon_coords*>(new_center_coords);
	(*flood_force_merge_lat_index)(previous_filled_cell_coords) = latlon_new_center_coords->get_lat();
	(*flood_force_merge_lon_index)(previous_filled_cell_coords) = latlon_new_center_coords->get_lon();
}
void latlon_basin_evaluation_algorithm::
	set_previous_cells_connect_force_merge_index(){
	latlon_coords* latlon_new_center_coords = static_cast<latlon_coords*>(new_center_coords);
	(*connect_force_merge_lat_index)(previous_filled_cell_coords) = latlon_new_center_coords->get_lat();
	(*connect_force_merge_lon_index)(previous_filled_cell_coords) = latlon_new_center_coords->get_lon();

}

void latlon_basin_evaluation_algorithm::
	set_previous_cells_redirect_index(coords* initial_fine_coords,coords* target_coords,
	                                  height_types height_type) {
	latlon_coords* latlon_target_coords =
		static_cast<latlon_coords*>(target_coords);
	if (height_type == connection_height){
		(*connect_redirect_lat_index)(initial_fine_coords) = latlon_target_coords->get_lat();
		(*connect_redirect_lon_index)(initial_fine_coords) = latlon_target_coords->get_lon();
	} else if (height_type == flood_height) {
		(*flood_redirect_lat_index)(initial_fine_coords) = latlon_target_coords->get_lat();
		(*flood_redirect_lon_index)(initial_fine_coords) = latlon_target_coords->get_lon();
	} else throw runtime_error("Height type not recognized");
}

coords* latlon_basin_evaluation_algorithm::get_cells_next_cell_index_as_coords(coords* coords_in,
                                                                               height_types height_type_in){

	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	if (height_type_in == flood_height)
		return new latlon_coords((*flood_next_cell_lat_index)(latlon_coords_in),
	                         	 (*flood_next_cell_lon_index)(latlon_coords_in));
	else if (height_type_in == connection_height)
		return new latlon_coords((*connect_next_cell_lat_index)(latlon_coords_in),
	                         	 (*connect_next_cell_lon_index)(latlon_coords_in));
	else throw runtime_error("Height type not recognized");
}

coords* latlon_basin_evaluation_algorithm::get_cells_redirect_index_as_coords(coords* coords_in,
                                                                              height_types height_type_in){
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	if (height_type_in == flood_height)
		return new latlon_coords((*flood_redirect_lat_index)(latlon_coords_in),
	                         	 (*flood_redirect_lon_index)(latlon_coords_in));
	else if (height_type_in == connection_height)
		return new latlon_coords((*connect_redirect_lat_index)(latlon_coords_in),
	                         	 (*connect_redirect_lon_index)(latlon_coords_in));
	else throw runtime_error("Height type not recognized");
}

coords* latlon_basin_evaluation_algorithm::get_cells_next_force_merge_index_as_coords(coords* coords_in,
                                                                              				height_types height_type_in){
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	if (height_type_in == flood_height)
		return new latlon_coords((*flood_force_merge_lat_index)(latlon_coords_in),
	                         	 (*flood_force_merge_lon_index)(latlon_coords_in));
	else if (height_type_in == connection_height)
		return new latlon_coords((*connect_force_merge_lat_index)(latlon_coords_in),
	                         	 (*connect_force_merge_lon_index)(latlon_coords_in));
	else throw runtime_error("Height type not recognized");
}
