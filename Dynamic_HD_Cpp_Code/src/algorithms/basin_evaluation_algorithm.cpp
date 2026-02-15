/*
 * basin_evaluation_algorithm.cpp
 *
 *  Created on: Feb 21, 2018
 *      Author: thomasriddick
 */
// Note the algorithm works on 3 sets of cells. Each threshold is the threshold to
// start filling a new cell. The new cell can either be a flood or a connect cell;
// connect means it simply connects other cells by a set of lateral and diagonal channels
// connecting all 8 surrounding cells but whose width has negible effect on the cells
// ability to hold water. Flood means flooding the entire cell. The algorithm consider three
// cells at a time. The previous cell is actually the cell currently filling, the center cell
// is the cell target for beginning to fill while the new cell is using for testing if the
// center cell is the sill at the edge of a basin

// It is possible for more than one basin merge to occur at single level. Here level exploration is
// only done in a consitent manner once the first level merge is found. Then all other merges are
// searched for at the same level. Levels where no merge is found, i.e. all surrounding cells not
// in the part of the basin already processed are explored in an ad-hoc manner as this will still
// always give consitent results

#include <queue>
#include <algorithm>
#include <string>
#include <map>
#include "algorithms/basin_evaluation_algorithm.hpp"
using namespace std;

basin_evaluation_algorithm::~basin_evaluation_algorithm() {
	delete minima; delete raw_orography; delete corrected_orography;
	delete connection_volume_thresholds; delete flood_volume_thresholds;
	delete connection_heights; delete flood_heights;
	delete basin_numbers; delete _grid; delete _coarse_grid;
	delete prior_fine_catchments; delete coarse_catchment_nums;
	delete level_completed_cells; delete requires_flood_redirect_indices;
	delete requires_connect_redirect_indices; delete flooded_cells;
	delete connected_cells; delete completed_cells;
	delete search_completed_cells;
	delete basin_flooded_cells; delete basin_connected_cells;
  	delete null_coords; delete cell_areas;
  	delete basin_merges_and_redirects;
        for(vector<vector<pair<coords*,bool>*>*>::iterator i =
            basin_connect_and_fill_orders.begin();
            i != basin_connect_and_fill_orders.end(); ++i){
            for(vector<pair<coords*,bool>*>::iterator j =
                (*i)->begin();j != (*i)->end(); ++j){
		delete (*j)->first;
		delete (*j);
            }
	    delete (*i);
	}
}

latlon_basin_evaluation_algorithm::~latlon_basin_evaluation_algorithm(){
	delete prior_fine_rdirs; delete prior_coarse_rdirs;
	delete flood_next_cell_lat_index; delete flood_next_cell_lon_index;
	delete connect_next_cell_lat_index; delete connect_next_cell_lon_index;
}

icon_single_index_basin_evaluation_algorithm::~icon_single_index_basin_evaluation_algorithm(){
	delete prior_fine_rdirs; delete prior_coarse_rdirs;
	delete flood_next_cell_index;
	delete connect_next_cell_index;
}

void basin_evaluation_algorithm::setup_fields(bool* minima_in,
		  	  	  	  	  	  	  	  	  	  double* raw_orography_in,
		  	  	  	  	  	  	  	  	  	  double* corrected_orography_in,
		  	  	  	  	  	  	  	  	  	  double* cell_areas_in,
		  	  	  	  	  	  	  	  	  	  double* connection_volume_thresholds_in,
		  	  	  	  	  	  	  	  	  	  double* flood_volume_thresholds_in,
		  	  	  	  	  	  	  	  	  	  double* connection_heights_in,
		  	  	  	  	  	  	  	  	  	  double* flood_heights_in,
		  	  	  	  	  	  	  	  	  	  int* prior_fine_catchments_in,
		  	  	  	  	  	  	  	  	  	  int* coarse_catchment_nums_in,
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
	cell_areas = new field<double>(cell_areas_in,_grid_params);
	connection_volume_thresholds = new field<double>(connection_volume_thresholds_in,_grid_params);
	connection_volume_thresholds->set_all(-1.0);
	flood_volume_thresholds = new field<double>(flood_volume_thresholds_in,_grid_params);
	flood_volume_thresholds->set_all(-1.0);
	connection_heights =  new field<double>(connection_heights_in,_grid_params);
	connection_heights->set_all(0.0);
	flood_heights = new field<double>(flood_heights_in,_grid_params);
	flood_heights->set_all(0.0);
	prior_fine_catchments = new field<int>(prior_fine_catchments_in,_grid_params);
	coarse_catchment_nums = new field<int>(coarse_catchment_nums_in,_coarse_grid_params);
	completed_cells = new field<bool>(_grid_params);
	search_completed_cells = new field<bool>(_coarse_grid_params);
	level_completed_cells = new field<bool>(_grid_params);
	requires_flood_redirect_indices = new field<bool>(_grid_params);
	requires_flood_redirect_indices->set_all(false);
	requires_connect_redirect_indices = new field<bool>(_grid_params);
	requires_connect_redirect_indices->set_all(false);
	basin_numbers = new field<int>(_grid_params);
	basin_numbers->set_all(null_catchment);
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
  	  	  	  	  	  	  							 double* cell_areas_in,
  	  	  	  	  	  	  							 double* connection_volume_thresholds_in,
  	  	  	  	  	  	  							 double* flood_volume_thresholds_in,
		  	  	  	  	  	  	  	  	  			 double* connection_heights_in,
		  	  	  	  	  	  	  	  	  			 double* flood_heights_in,
  	  	  	  	  	  	  							 double* prior_fine_rdirs_in,
  	  	  	  	  	  	  							 double* prior_coarse_rdirs_in,
  	  	  	  	  	  	  							 int* prior_fine_catchments_in,
  	  	  	  	  	  	  							 int* coarse_catchment_nums_in,
  	  	  	  	  	  	  							 int* flood_next_cell_lat_index_in,
  	  	  	  	  	  	  							 int* flood_next_cell_lon_index_in,
  	  	  	  	  	  	  							 int* connect_next_cell_lat_index_in,
  	  	  	  	  	  	  							 int* connect_next_cell_lon_index_in,
													 grid_params* grid_params_in,
													 grid_params* coarse_grid_params_in)
{
	basin_evaluation_algorithm::setup_fields(minima_in,raw_orography_in,
	                                         corrected_orography_in,
	                                         cell_areas_in,
	                                         connection_volume_thresholds_in,
	                                         flood_volume_thresholds_in,
		  	  	  	  	  					 connection_heights_in,
		  	  	  	  	  	  	  	  	  	 flood_heights_in,
											 prior_fine_catchments_in,
											 coarse_catchment_nums_in,
	  	  	  	  	  	  	  	  	  		 grid_params_in,
	  	  	  	  	  	  	  	  	  		 coarse_grid_params_in);
	basin_merges_and_redirects =
		new merges_and_redirects(latlon_merge_and_redirect_indices_factory,grid_params_in);
	prior_fine_rdirs = new field<double>(prior_fine_rdirs_in,grid_params_in);
	prior_coarse_rdirs = new field<double>(prior_coarse_rdirs_in,coarse_grid_params_in);
	flood_next_cell_lat_index = new field<int>(flood_next_cell_lat_index_in,grid_params_in);
	flood_next_cell_lon_index = new field<int>(flood_next_cell_lon_index_in,grid_params_in);
	connect_next_cell_lat_index = new field<int>(connect_next_cell_lat_index_in,grid_params_in);
	connect_next_cell_lon_index = new field<int>(connect_next_cell_lon_index_in,grid_params_in);
  	null_coords = new latlon_coords(-1,-1);
  	//basin sink points are calculated on an as required basis - null coords mean they have not
  	//yet been calculated - after calculation these will be replaced with actual coords
	for (int i = 0; i < _grid->get_total_size(); i++){
		basin_sink_points.push_back(null_coords->clone());
	}
}

void icon_single_index_basin_evaluation_algorithm::
										 setup_fields(bool* minima_in,
							                    double* raw_orography_in,
							                    double* corrected_orography_in,
							                    double* cell_areas_in,
							                    double* connection_volume_thresholds_in,
							                    double* flood_volume_thresholds_in,
		  	  	  	  	  	  	  	  	  		double* connection_heights_in,
		  	  	  	  	  	  	  	  	  		double* flood_heights_in,
							                    int* prior_fine_rdirs_in,
							                    int* prior_coarse_rdirs_in,
							                    int* prior_fine_catchments_in,
							                    int* coarse_catchment_nums_in,
							                    int* flood_next_cell_index_in,
							                    int* connect_next_cell_index_in,
							                    grid_params* grid_params_in,
							                    grid_params* coarse_grid_params_in)
{
	basin_evaluation_algorithm::setup_fields(minima_in,raw_orography_in,
	                                         corrected_orography_in,
	                                         cell_areas_in,
	                                         connection_volume_thresholds_in,
	                                         flood_volume_thresholds_in,
		  	  	  	  	  	  	  	  	  	 connection_heights_in,
		  	  	  	  	  	  	  	  	  	 flood_heights_in,
											 prior_fine_catchments_in,
											 coarse_catchment_nums_in,
	  	  	  	  	  	  	  	  	  		 grid_params_in,
	  	  	  	  	  	  	  	  	  		 coarse_grid_params_in);
	basin_merges_and_redirects =
		new merges_and_redirects(icon_single_index_merge_and_redirect_indices_factory,
		                         grid_params_in);
	prior_fine_rdirs = new field<int>(prior_fine_rdirs_in,grid_params_in);
	prior_coarse_rdirs = new field<int>(prior_coarse_rdirs_in,coarse_grid_params_in);
	flood_next_cell_index = new field<int>(flood_next_cell_index_in,grid_params_in);
	connect_next_cell_index = new field<int>(connect_next_cell_index_in,grid_params_in);
  	null_coords = new generic_1d_coords(-1);
  	//basin sink points are calculated on an as required basis - null coords mean they have not
  	//yet been calculated - after calculation these will be replaced with actual coords
	for (int i = 0; i < _grid->get_total_size(); i++){
		basin_sink_points.push_back(null_coords->clone());
	}
}

void basin_evaluation_algorithm::
		 setup_sink_filling_algorithm(sink_filling_algorithm_4* sink_filling_alg_in){
		 	if(! minima)
		 		throw runtime_error("Trying to setup sink filling algorithm before setting minima");
			sink_filling_alg = sink_filling_alg_in;
			sink_filling_alg->set_catchments(-1);
			sink_filling_alg->setup_minima_q(minima);
}

void basin_evaluation_algorithm::evaluate_basins(){
	generate_minima();
	basin_number = 1;
	cout << "Starting to evaluate basins" << endl;
	//Runs on a catchment by catchment basis
	while (! minima_coords_queues.empty()){
		add_minima_for_catchment_to_queue();
		while (! minima_q.empty()) {
			minimum = static_cast<basin_cell*>(minima_q.front());
			minima_q.pop();
			evaluate_basin();
			basin_number++;
			delete minimum;
		}
	}
	cout << "Setting remaining redirects" << endl;
	set_remaining_redirects();
	//clean up
	while ( ! basin_catchment_centers.empty()){
		coords* catchment_center = basin_catchment_centers.back();
		basin_catchment_centers.pop_back();
		delete catchment_center;
	}
	while ( ! basin_sink_points.empty()){
		coords* basin_sink_point = basin_sink_points.back();
		basin_sink_points.pop_back();
		delete basin_sink_point;
	}
}

void basin_evaluation_algorithm::generate_minima() {
	sink_filling_alg->fill_sinks();
	catchments_from_sink_filling = sink_filling_alg->get_catchments();
	stack<coords*>* minima_coords_q = sink_filling_alg->get_minima_q();
	map<int,int> catchments_with_minima;
	int max_index = 0;
	//Convert queue given by sink filling algorithm to a seperate queue for
	//each catchment
	while (! minima_coords_q->empty()){
		coords* minima_coords = minima_coords_q->top();
		minima_coords_q->pop();
		int minima_catchment_num = (*catchments_from_sink_filling)(minima_coords);
		map<int,int>::iterator search =
			catchments_with_minima.find(minima_catchment_num);
		if(search != catchments_with_minima.end()){
			minima_coords_queues[search->second]->push(minima_coords);
		} else{
			//Swap from a stack to queue to account for reversal of order
			queue<coords*>* minima_coords_q_for_catchment = new queue<coords*>();
			minima_coords_q_for_catchment->push(minima_coords);
			minima_coords_queues.push_back(minima_coords_q_for_catchment);
			catchments_with_minima[minima_catchment_num] = max_index;
			max_index++;
		}
	}
	delete minima_coords_q;
}

void basin_evaluation_algorithm::add_minima_for_catchment_to_queue() {
	queue<coords*>* minima_coords_q = minima_coords_queues.back();
	minima_coords_queues.pop_back();
	while (! minima_coords_q->empty()){
		coords* minima_coords = minima_coords_q->front();
		minima_coords_q->pop();
		double height = 0.0;
		height_types height_type;
 		double raw_height = (*raw_orography)(minima_coords);
		double corrected_height = (*corrected_orography)(minima_coords);
		if(raw_height <= corrected_height) {
			height = raw_height;
			height_type = flood_height;
		} else {
			height = corrected_height;
			height_type = connection_height;
		}
		minima_q.push(new basin_cell(height,height_type,minima_coords));
	}
	delete minima_coords_q;
}

void basin_evaluation_algorithm::evaluate_basin(){
	initialize_basin();
	while(true){
		if (q.empty()) throw runtime_error("Basin outflow not found");
		center_cell = static_cast<basin_cell*>(q.top());
		q.pop();
		//Call the newly loaded coordinates and height for the center cell 'new'
		//until after making the test for merges then relabel. Center cell height/coords
		//without the 'new' moniker refers to the previous center cell; previous center cell
		//height/coords the previous previous center cell
		read_new_center_cell_variables();
		if (new_center_cell_height <= surface_height) {
			bool includes_secondary_merge = process_all_merges_at_given_level();
			if (includes_secondary_merge) break;
		}
		//this cell is in a subsumed basin already merge with this one
		if(((*basin_flooded_cells)(new_center_coords) &&
			 new_center_cell_height_type == flood_height) ||
		   ((*basin_connected_cells)(new_center_coords) &&
			 new_center_cell_height_type == connection_height)){
			delete new_center_coords;
			delete center_cell;
			continue;
		}
		process_neighbors();
		update_previous_filled_cell_variables();
		update_center_cell_variables();
		process_center_cell();
		delete center_cell;
	}
	delete previous_filled_cell_coords;
	//clean up
	while (! q.empty()){
		center_cell = static_cast<basin_cell*>(q.top());
		q.pop();
		delete center_cell;
	}
}

void basin_evaluation_algorithm::initialize_basin(){
	center_cell = minimum->clone();
	completed_cells->set_all(false);
	basin_flooded_cells->set_all(false);
	basin_connected_cells->set_all(false);
	center_cell_volume_threshold = 0.0;
	basin_connections.add_set(basin_number);
	basin_connect_and_fill_order = new vector<pair<coords*,bool>*>();
	center_coords = center_cell->get_cell_coords()->clone();
	center_cell_height_type = center_cell->get_height_type();
	center_cell_height = center_cell->get_orography();
	basin_catchment_centers.push_back(center_coords->clone());
	surface_height = center_cell_height;
	previous_filled_cell_coords = center_coords->clone();
	previous_filled_cell_height_type = center_cell_height_type;
	previous_filled_cell_height = center_cell_height;
  	catchments_from_sink_filling_catchment_num =
  		(*catchments_from_sink_filling)(center_coords);
	read_new_center_cell_variables();
	if (center_cell_height_type == connection_height) {
		(*connected_cells)(center_coords) = true;
		(*basin_connected_cells)(center_coords) = true;
		lake_area = 0.0;
	} else if (center_cell_height_type == flood_height) {
		(*flooded_cells)(center_coords) = true;
		(*basin_flooded_cells)(center_coords) = true;
		lake_area = (*cell_areas)(center_coords);
	} else throw runtime_error("Cell type not recognized");
	(*completed_cells)(center_coords) = true;
	//Make partial first and second iteration
	process_neighbors();
	delete center_cell;
	delete new_center_coords;
	center_cell = static_cast<basin_cell*>(q.top());
	q.pop();
	read_new_center_cell_variables();
	process_neighbors();
	update_center_cell_variables();
	process_center_cell();
	delete center_cell;
}

inline bool basin_evaluation_algorithm::
	process_all_merges_at_given_level(){
	level_q.push(new basin_cell(center_cell_height,center_cell_height_type,
	                            center_coords->clone()));
	level_completed_cells->set_all(false);
	height_types level_cell_height_type;
	basin_cell* level_cell;
	bool process_nbrs = false;
	secondary_merge_found = false;
	coords* secondary_merge_coords = nullptr;
	//Searches entire level including over catchment bounds (in case
	//the parts of the level in this catchment are in two seperate pockets)
	while (! (level_q.empty() && level_nbr_q.empty())) {
		if (! level_q.empty()) {
			level_cell = level_q.front();
			level_q.pop();
			process_nbrs = true;
		} else {
			level_cell = static_cast<basin_cell*>(level_nbr_q.top());
			level_nbr_q.pop();
			process_nbrs = false;
		}
		level_coords = level_cell->get_cell_coords();
		level_cell_height_type = level_cell->get_height_type();
		int level_cell_catchment = (*catchments_from_sink_filling)(level_coords);
		bool in_different_catchment =
	  		( level_cell_catchment != catchments_from_sink_filling_catchment_num) &&
	  		( level_cell_catchment != -1);
	  	if  (! in_different_catchment) {
			if (! already_in_basin(level_coords,level_cell_height_type)) {
				if ((*basin_numbers)(level_coords)  != 0 ) {
					int target_basin_number = (*basin_numbers)(level_coords);
					int root_target_basin_number = basin_connections.find_root(target_basin_number);
					if (root_target_basin_number == basin_number)
					throw runtime_error("Logic failure - trying to merge current basin with basin "
		      							"that is already connected to the current basin");
					rebuild_secondary_basin(root_target_basin_number);
					primary_merge_q.push(new pair<int,int>(target_basin_number,
					                                       root_target_basin_number));
				} else if((*basin_numbers)(level_coords)  == 0 &&
				          (! secondary_merge_found) &&
				          (! process_nbrs)){
					secondary_merge_found = true;
					//Bool false of true indicates this is a secondary merge
					secondary_merge_coords = level_coords->clone();
				}
			}
		}
		if (process_nbrs) process_level_neighbors();
		delete level_cell;
	}
	while (! primary_merge_q.empty()) {
		pair<int,int>* primary_merge_basin_numbers = primary_merge_q.front();
		primary_merge_q.pop();
		process_primary_merge(primary_merge_basin_numbers->first,
		                      primary_merge_basin_numbers->second);
                delete primary_merge_basin_numbers;
	}
	if (secondary_merge_found){
		process_secondary_merge(secondary_merge_coords);
		delete secondary_merge_coords;
	}
	return secondary_merge_found;
}

void basin_evaluation_algorithm::process_level_neighbors() {
	neighbors_coords = raw_orography->get_neighbors_coords(level_coords,1);
	while( ! neighbors_coords->empty() ) {
		process_level_neighbor();
	}
	delete neighbors_coords;
}

void basin_evaluation_algorithm::process_level_neighbor() {
	coords* nbr_coords = neighbors_coords->back();
	neighbors_coords->pop_back();
	if (! (*level_completed_cells)(nbr_coords)) {
				double raw_height = (*raw_orography)(nbr_coords);
				double corrected_height = (*corrected_orography)(nbr_coords);
				(*level_completed_cells)(nbr_coords) = true;
				double nbr_height;
				height_types nbr_height_type;
				if(raw_height <= corrected_height) {
					nbr_height = raw_height;
					nbr_height_type = flood_height;
				} else {
					nbr_height = corrected_height;
					nbr_height_type = connection_height;
				}
				if (nbr_height == surface_height){
					level_q.push(new basin_cell(nbr_height,nbr_height_type,
		                		  						nbr_coords));
				} else if (nbr_height < surface_height){
					level_nbr_q.push(new basin_cell(nbr_height,nbr_height_type,
		                		  							  nbr_coords));
				} else delete nbr_coords;
	} else delete nbr_coords;
}

inline void basin_evaluation_algorithm::read_new_center_cell_variables(){
		new_center_coords = center_cell->get_cell_coords()->clone();
		new_center_cell_height_type = center_cell->get_height_type();
		new_center_cell_height = center_cell->get_orography();
}

inline void basin_evaluation_algorithm::update_previous_filled_cell_variables(){
		delete previous_filled_cell_coords;
		previous_filled_cell_coords = center_coords->clone();
		previous_filled_cell_height = center_cell_height;
		previous_filled_cell_height_type = center_cell_height_type;
}

inline void basin_evaluation_algorithm::update_center_cell_variables(){
		center_cell_height_type = new_center_cell_height_type;
		center_cell_height = new_center_cell_height;
		delete center_coords;
		center_coords = new_center_coords;
		if (center_cell_height > surface_height) surface_height = center_cell_height;
}

inline bool basin_evaluation_algorithm::
	already_in_basin(coords* coords_in,height_types height_type_in){
	return (((*basin_flooded_cells)(coords_in) && height_type_in == flood_height) ||
		    ((*basin_connected_cells)(coords_in) && height_type_in == connection_height));
}

void basin_evaluation_algorithm::rebuild_center_cell(coords* coords_in,
                                                     height_types height_type_in) {
	if (height_type_in == flood_height) {
		lake_area += (*cell_areas)(coords_in);
	} else if (height_type_in == connection_height &&
	           ! (*basin_flooded_cells)(coords_in)) {
		q.push(new basin_cell((*raw_orography)(coords_in),
					 		  flood_height,coords_in->clone()));
	} else throw runtime_error("Height type not recognized");
}

void basin_evaluation_algorithm::process_center_cell() {
	set_previous_filled_cell_basin_number();
	center_cell_volume_threshold +=
			lake_area*(center_cell_height-previous_filled_cell_height);
	if (previous_filled_cell_height_type == connection_height) {
		(*connection_volume_thresholds)(previous_filled_cell_coords) =
			center_cell_volume_threshold;
		(*connection_heights)(previous_filled_cell_coords) =
			center_cell_height;
		q.push(new basin_cell((*raw_orography)(previous_filled_cell_coords),
		                      flood_height,previous_filled_cell_coords->clone()));
		set_previous_cells_connect_next_cell_index(center_coords);
		basin_connect_and_fill_order->
			push_back(new pair<coords*,bool>(previous_filled_cell_coords->clone(),false));
	} else if (previous_filled_cell_height_type == flood_height) {
		(*flood_volume_thresholds)(previous_filled_cell_coords) =
			center_cell_volume_threshold;
		(*flood_heights)(previous_filled_cell_coords) =
			center_cell_height;
		set_previous_cells_flood_next_cell_index(center_coords);
		basin_connect_and_fill_order->
			push_back(new pair<coords*,bool>(previous_filled_cell_coords->clone(),true));
	} else throw runtime_error("Cell type not recognized");
	if (center_cell_height_type == connection_height) {
		(*connected_cells)(center_coords) = true;
		(*basin_connected_cells)(center_coords) = true;
	}
	else if (center_cell_height_type == flood_height) {
		(*flooded_cells)(center_coords) = true;
		(*basin_flooded_cells)(center_coords) = true;
		lake_area += (*cell_areas)(center_coords);
	} else throw runtime_error("Cell type not recognized");
}

void inline  basin_evaluation_algorithm::set_previous_filled_cell_basin_number(){
	if ((*basin_numbers)(previous_filled_cell_coords)
			== null_catchment) {
		(*basin_numbers)(previous_filled_cell_coords) =
			basin_number;
	}
}

void basin_evaluation_algorithm::process_neighbors() {
	neighbors_coords = raw_orography->get_neighbors_coords(new_center_coords,1);
	while( ! neighbors_coords->empty() ) {
		process_neighbor();
	}
	delete neighbors_coords;
}

void basin_evaluation_algorithm::process_neighbors_when_rebuilding_basin(coords* coords_in) {
	neighbors_coords = raw_orography->get_neighbors_coords(coords_in,1);
	while( ! neighbors_coords->empty() ) {
		process_neighbor();
	}
	delete neighbors_coords;
}

void basin_evaluation_algorithm::process_neighbor() {
	coords* nbr_coords = neighbors_coords->back();
	neighbors_coords->pop_back();
	int nbr_catchment = (*catchments_from_sink_filling)(nbr_coords);
	bool in_different_catchment =
	  ( nbr_catchment != catchments_from_sink_filling_catchment_num) &&
	  ( nbr_catchment != -1);
	if ( (! (*completed_cells)(nbr_coords)) &&
	     (! in_different_catchment) )  {
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

void basin_evaluation_algorithm::process_secondary_merge(coords* other_basin_entry_coords){
	set_previous_filled_cell_basin_number();
	if (previous_filled_cell_height_type == flood_height) {
		basin_merges_and_redirects->set_unmatched_flood_merge(previous_filled_cell_coords);
	} else if (previous_filled_cell_height_type == connection_height) {
		basin_merges_and_redirects->set_unmatched_connect_merge(previous_filled_cell_coords);
	} else throw runtime_error("Cell type not recognized");
	basin_connect_and_fill_orders.push_back(basin_connect_and_fill_order);
	set_preliminary_secondary_redirect(other_basin_entry_coords);
	delete center_cell; delete new_center_coords;
	delete center_coords;
}

void basin_evaluation_algorithm::process_primary_merge(int target_basin_number,
                                                       int root_target_basin_number){

	coords* root_target_basin_center_coords =
		basin_catchment_centers[root_target_basin_number - 1];
	collected_merge_and_redirect_indices*
		working_collected_merge_and_redirect_indices =
		basin_merges_and_redirects->
			get_collected_merge_and_redirect_indices(previous_filled_cell_coords,
		                                           previous_filled_cell_height_type,
		                                           true);
	working_collected_merge_and_redirect_indices->
		set_next_primary_merge_target_index(root_target_basin_center_coords);
	reprocess_secondary_merge(root_target_basin_number,
	                          basin_number);

	basin_connections.make_new_link(basin_number,root_target_basin_number);
	merge_and_redirect_indices* working_redirect =
		working_collected_merge_and_redirect_indices->
			get_latest_primary_merge_and_redirect_indices();
	//Give redirect setting the original basin number not the root basin number
	set_primary_redirect(working_redirect,target_basin_number);
}

void basin_evaluation_algorithm::reprocess_secondary_merge(int root_secondary_basin_number,
                                                           int target_primary_basin_number){
	pair<coords*,bool>* last_filled_cell_coords_and_flag =
		basin_connect_and_fill_orders[root_secondary_basin_number -1]->back();
	coords* merge_coords = last_filled_cell_coords_and_flag->first;
	height_types merge_height_type = last_filled_cell_coords_and_flag->second ?
																	 flood_height : connection_height;
	collected_merge_and_redirect_indices*
		working_collected_merge_and_redirect_indices =
		basin_merges_and_redirects->get_collected_merge_and_redirect_indices(merge_coords,
		                                                                     merge_height_type,
		                                                                     false);
	if (working_collected_merge_and_redirect_indices->get_unmatched_secondary_merge()){
		working_collected_merge_and_redirect_indices->set_unmatched_secondary_merge(false);
	} else throw runtime_error("Unable to match primary merge with unmatched secondary merge");
	coords* target_primary_basin_center_coords =
		basin_catchment_centers[target_primary_basin_number - 1];
	working_collected_merge_and_redirect_indices->
		set_secondary_merge_target_coords(target_primary_basin_center_coords);
	set_secondary_redirect(working_collected_merge_and_redirect_indices->
	                       		get_secondary_merge_and_redirect_indices(),
                         merge_coords,
                         target_primary_basin_center_coords,
                         merge_height_type);
}

void basin_evaluation_algorithm::rebuild_secondary_basin(int root_secondary_basin_number){
	basin_connections.for_elements_in_set(root_secondary_basin_number,
	                                      [&](int working_basin_number){
	  vector<pair<coords*,bool>*>* working_basin_connect_and_fill_order =
	  	basin_connect_and_fill_orders[working_basin_number - 1];
  	for(vector<pair<coords*,bool>*>::iterator i =
  	    working_basin_connect_and_fill_order->begin();
        i != working_basin_connect_and_fill_order->end(); ++i){
	  	coords* working_coords = (*i)->first;
	  	(*completed_cells)(working_coords) = true;
	  	if((*i)->second) (*basin_flooded_cells)(working_coords) = true;
	  	else (*basin_connected_cells)(working_coords) = true;
	  }
	for(vector<pair<coords*,bool>*>::iterator i =
  	    working_basin_connect_and_fill_order->begin();
        i != working_basin_connect_and_fill_order->end(); ++i){
	  		coords* working_coords = (*i)->first;
	 		process_neighbors_when_rebuilding_basin(working_coords);
	  		rebuild_center_cell(working_coords,(*i)->second ? flood_height :
	  		                    							  connection_height);
	  }
	});
}

void basin_evaluation_algorithm::
	set_preliminary_secondary_redirect(coords* other_basin_entry_coords){
	if (previous_filled_cell_height_type == flood_height) {
		set_previous_cells_flood_next_cell_index(other_basin_entry_coords);
		(*requires_flood_redirect_indices)(previous_filled_cell_coords) = true;
	} else if (previous_filled_cell_height_type == connection_height){
		set_previous_cells_connect_next_cell_index(other_basin_entry_coords);
		(*requires_connect_redirect_indices)(previous_filled_cell_coords) = true;
	} else throw runtime_error("Cell type not recognized");
}

void basin_evaluation_algorithm::set_primary_redirect(merge_and_redirect_indices* working_redirect,
                                                      int target_basin_number_in){
	coords* target_basin_center_coords =
		basin_catchment_centers[target_basin_number_in - 1];
	if(_coarse_grid->fine_coords_in_same_cell(center_coords,
			target_basin_center_coords,_grid_params)) {
		  working_redirect->set_local_redirect();
		  working_redirect->set_redirect_indices(target_basin_center_coords);
	} else {
		  find_and_set_previous_cells_non_local_redirect_index(working_redirect,
		                                                       center_coords,
		                                                       target_basin_center_coords);
	}
}

void basin_evaluation_algorithm::set_secondary_redirect(merge_and_redirect_indices* working_redirect,
                                                        coords* redirect_coords,
                                                        coords* target_basin_center_coords,
                                                        height_types redirect_height_type){
	if (redirect_height_type == flood_height){
		if (! (*requires_flood_redirect_indices)(redirect_coords))
				throw runtime_error("Redirect logic error");
		(*requires_flood_redirect_indices)(redirect_coords) = false;
	} else if (redirect_height_type == connection_height){
		if (! (*requires_flood_redirect_indices)(redirect_coords))
				throw runtime_error("Redirect logic error");
		(*requires_connect_redirect_indices)(redirect_coords) = false;
	} else throw runtime_error("Cell type not recognized");
	coords* first_cell_beyond_rim_coords =
		get_cells_next_cell_index_as_coords(redirect_coords,redirect_height_type);
	if(_coarse_grid->fine_coords_in_same_cell(first_cell_beyond_rim_coords,
		target_basin_center_coords,_grid_params)) {
		  working_redirect->set_local_redirect();
		  working_redirect->set_redirect_indices(target_basin_center_coords);
	} else {
 		find_and_set_previous_cells_non_local_redirect_index(working_redirect,
		                                        			 first_cell_beyond_rim_coords,
 		                                                     target_basin_center_coords);
	}
	delete first_cell_beyond_rim_coords;
}

void basin_evaluation_algorithm::
	find_and_set_previous_cells_non_local_redirect_index(merge_and_redirect_indices* working_redirect,
	                                                     coords* current_center_coords,
	                                                     coords* target_basin_center_coords){
	coords* catchment_center_coarse_coords = _coarse_grid->convert_fine_coords(target_basin_center_coords,
	                                                                           _grid_params);
	if(coarse_cell_is_sink(catchment_center_coarse_coords)){
		int coarse_catchment_num = (*coarse_catchment_nums)(catchment_center_coarse_coords);
		find_and_set_non_local_redirect_index_from_coarse_catchment_num(working_redirect,
	                                                                	current_center_coords,
	                                                                	coarse_catchment_num);
	} else {
		//A non local connection is not possible so fall back on using a local connection
	  working_redirect->set_local_redirect();
	  working_redirect->set_redirect_indices(target_basin_center_coords);
	}
	delete catchment_center_coarse_coords;
}

void basin_evaluation_algorithm::
	find_and_set_non_local_redirect_index_from_coarse_catchment_num(merge_and_redirect_indices* working_redirect,
	                                                                coords* current_center_coords,
	                                                     						int coarse_catchment_number){
	coords* center_coarse_coords = _coarse_grid->convert_fine_coords(current_center_coords,
	                                                                 _grid_params);
	if((*coarse_catchment_nums)(center_coarse_coords) == coarse_catchment_number) {
		working_redirect->set_non_local_redirect();
		working_redirect->set_redirect_indices(center_coarse_coords);
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
				working_redirect->set_non_local_redirect();
				working_redirect->set_redirect_indices(search_coords);
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
			} else {
				redirect_height_type = connection_height;
				cell_requires_connect_redirect_indices = false;
			}
			coords* first_cell_beyond_rim_coords =
				get_cells_next_cell_index_as_coords(coords_in,redirect_height_type);
			coords* catchment_outlet_coarse_coords = nullptr;
			int prior_fine_catchment_num = (*prior_fine_catchments)(first_cell_beyond_rim_coords);
			if (prior_fine_catchment_num == 0){
				catchment_outlet_coarse_coords =
						_coarse_grid->convert_fine_coords(first_cell_beyond_rim_coords,
								                           		_grid_params);
			} else {
				catchment_outlet_coarse_coords = basin_sink_points[prior_fine_catchment_num - 1];
				if ((*catchment_outlet_coarse_coords)==(*null_coords)) {
					delete catchment_outlet_coarse_coords;
					catchment_outlet_coarse_coords = nullptr;
					coords* current_coords = first_cell_beyond_rim_coords->clone();
					while(true) {
							if (check_for_sinks_and_set_downstream_coords(current_coords)) {
								catchment_outlet_coarse_coords =
									_coarse_grid->convert_fine_coords(current_coords,
									                                  _grid_params);
								delete downstream_coords;
								delete current_coords;
								break;
							}
							delete current_coords;
							current_coords = downstream_coords;
					}
					if ( ! catchment_outlet_coarse_coords)
						throw runtime_error("Sink point for non local secondary redirect not found");
					basin_sink_points[prior_fine_catchment_num - 1] = catchment_outlet_coarse_coords;
				}
			}
			int coarse_catchment_number =
				(*coarse_catchment_nums)(catchment_outlet_coarse_coords);
			collected_merge_and_redirect_indices*
				working_collected_merge_and_redirect_indices =
				basin_merges_and_redirects->get_collected_merge_and_redirect_indices(coords_in,
		                                                                    		 redirect_height_type,
		                                                                    		 false);
			if (! working_collected_merge_and_redirect_indices->
			    	get_unmatched_secondary_merge())
				throw runtime_error("Missing unmatched secondary merge when setting remaining redirects");
			working_collected_merge_and_redirect_indices->set_unmatched_secondary_merge(false);
			working_collected_merge_and_redirect_indices->
				set_secondary_merge_target_coords(first_cell_beyond_rim_coords);
			merge_and_redirect_indices*  working_redirect =
				working_collected_merge_and_redirect_indices->
	        get_secondary_merge_and_redirect_indices();
			find_and_set_non_local_redirect_index_from_coarse_catchment_num(working_redirect,
					                                        				first_cell_beyond_rim_coords,
                                                     						coarse_catchment_number);
		if (prior_fine_catchment_num == 0) delete catchment_outlet_coarse_coords;
                delete first_cell_beyond_rim_coords;
		}
		delete coords_in;
	});
}

int* basin_evaluation_algorithm::retrieve_lake_numbers(){
	return basin_numbers->get_array();
}

queue<cell*> basin_evaluation_algorithm::
														test_add_minima_to_queue(double* raw_orography_in,
                                                   	 double* corrected_orography_in,
                                                     bool* minima_in,
                                                     int* prior_fine_catchments_in,
                                                     sink_filling_algorithm_4*
                                                     sink_filling_alg_in,
                                                     grid_params* grid_params_in,
                                                     grid_params* coarse_grid_params_in){
	_grid_params = grid_params_in;
	_coarse_grid_params =  coarse_grid_params_in;
	_grid = grid_factory(_grid_params);
	_coarse_grid = grid_factory(_coarse_grid_params);
	sink_filling_alg = sink_filling_alg_in;
	minima = new field<bool>(minima_in,_grid_params);
	raw_orography = new field<double>(raw_orography_in,_grid_params);
	corrected_orography =
		new field<double>(corrected_orography_in,_grid_params);
	prior_fine_catchments = new field<int>(prior_fine_catchments_in,_grid_params);
	sink_filling_alg->setup_minima_q(minima);
	generate_minima();
	add_minima_for_catchment_to_queue();
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
	catchments_from_sink_filling = new field<int>(_grid_params);
	catchments_from_sink_filling->set_all(0);
	catchments_from_sink_filling_catchment_num = 0;
	process_neighbors();
	delete raw_orography; delete corrected_orography; delete completed_cells; delete _grid;
	delete catchments_from_sink_filling;
	raw_orography = nullptr; corrected_orography = nullptr; completed_cells = nullptr;
	_grid = nullptr;
	return q;
}

queue<landsea_cell*> basin_evaluation_algorithm::
	test_search_process_neighbors(coords* search_coords_in,bool* search_completed_cells_in,
	                              grid_params* grid_params_in){
	_grid_params = grid_params_in;
	search_completed_cells = new field<bool>(search_completed_cells_in,_grid_params);
	catchments_from_sink_filling = new field<int>(_grid_params);
	catchments_from_sink_filling->set_all(0);
	catchments_from_sink_filling_catchment_num = 0;
	search_coords = search_coords_in;
	search_process_neighbors();
	delete search_completed_cells;
	delete catchments_from_sink_filling;
	search_completed_cells = nullptr;
	return search_q;
}

priority_cell_queue latlon_basin_evaluation_algorithm::test_process_center_cell(basin_cell* center_cell_in,
                                                                                coords* center_coords_in,
                                                                                coords* previous_filled_cell_coords_in,
                                               													 				double* flood_volume_thresholds_in,
                                               													 				double* connection_volume_thresholds_in,
                                               													 				double* raw_orography_in,
                                               													 				double* cell_areas_in,
                                               													 				int* flood_next_cell_lat_index_in,
                                               													 				int* flood_next_cell_lon_index_in,
                                               													 				int* connect_next_cell_lat_index_in,
                                               													 				int* connect_next_cell_lon_index_in,
                                               													 				int* basin_numbers_in,
                                               													 				bool* flooded_cells_in,
                                               													 				bool* connected_cells_in,
                                               													 				double& center_cell_volume_threshold_in,
                                               													 				double& lake_area_in,
                                               													 				int basin_number_in,
                                               													 				double center_cell_height_in,
                                               													 				double& previous_filled_cell_height_in,
                                               													 				height_types& previous_filled_cell_height_type_in,
                                               													 				grid_params* grid_params_in){
	previous_filled_cell_coords = previous_filled_cell_coords_in;
	center_coords = center_coords_in;
	center_cell = center_cell_in;
	flood_volume_thresholds = new field<double>(flood_volume_thresholds_in,grid_params_in);
	connection_volume_thresholds = new field<double>(connection_volume_thresholds_in,grid_params_in);
	flood_heights = new field<double>(grid_params_in);
	connection_heights = new field<double>(grid_params_in);
	raw_orography = new field<double>(raw_orography_in,grid_params_in);
	cell_areas = new field<double>(cell_areas_in,grid_params_in);
	flood_next_cell_lat_index = new field<int>(flood_next_cell_lat_index_in,grid_params_in);
	flood_next_cell_lon_index = new field<int>(flood_next_cell_lon_index_in,grid_params_in);
	connect_next_cell_lat_index = new field<int>(connect_next_cell_lat_index_in,grid_params_in);
	connect_next_cell_lon_index = new field<int>(connect_next_cell_lon_index_in,grid_params_in);
	basin_numbers = new field<int>(basin_numbers_in,grid_params_in);
	flooded_cells = new field<bool>(flooded_cells_in,grid_params_in);
	flooded_cells->set_all(false);
	connected_cells = new field<bool>(connected_cells_in,grid_params_in);
	connected_cells->set_all(false);
	basin_flooded_cells = new field<bool>(grid_params_in);
	basin_flooded_cells->set_all(false);
	basin_connected_cells = new field<bool>(grid_params_in);
	basin_connected_cells->set_all(false);
	basin_connect_and_fill_order = new vector<pair<coords*,bool>*>();
	basin_connect_and_fill_orders.push_back(basin_connect_and_fill_order);
	center_cell_volume_threshold = center_cell_volume_threshold_in;
	lake_area = lake_area_in;
	basin_number = basin_number_in;
  	center_cell_height = center_cell_height_in;
  	previous_filled_cell_height = previous_filled_cell_height_in;
	center_cell_height_type = center_cell->get_height_type();
	previous_filled_cell_height_type = previous_filled_cell_height_type_in;
  	process_center_cell();
	center_cell_volume_threshold_in = center_cell_volume_threshold;
	lake_area_in = lake_area;
	previous_filled_cell_height_in = previous_filled_cell_height;
	previous_filled_cell_height_type_in = previous_filled_cell_height_type;
	return q;
}

void latlon_basin_evaluation_algorithm::
		 test_set_primary_merge_and_redirect(vector<coords*> basin_catchment_centers_in,
		                                     double* prior_coarse_rdirs_in,
                                         int* basin_numbers_in,
                                         int* coarse_catchment_nums_in,
                                         coords* new_center_coords_in,
                                         coords* center_coords_in,
                                         coords* previous_filled_cell_coords_in,
                                         height_types previous_filled_cell_height_type_in,
                                         grid_params* grid_params_in,
                                         grid_params* coarse_grid_params_in) {
	_grid_params = grid_params_in;
	_coarse_grid_params = coarse_grid_params_in;
	basin_catchment_centers = basin_catchment_centers_in;
	basin_numbers = new field<int>(basin_numbers_in,grid_params_in);
	coarse_catchment_nums = new field<int>(coarse_catchment_nums_in,coarse_grid_params_in);
	prior_coarse_rdirs = new field<double>(prior_coarse_rdirs_in,coarse_grid_params_in);
	search_completed_cells = new field<bool>(coarse_grid_params_in);
	requires_flood_redirect_indices = new field<bool>(_grid_params);
	requires_flood_redirect_indices->set_all(false);
	requires_connect_redirect_indices = new field<bool>(_grid_params);
	requires_connect_redirect_indices->set_all(false);
	flood_next_cell_lat_index = new field<int>(grid_params_in);
	flood_next_cell_lon_index = new field<int>(grid_params_in);
	flood_next_cell_lat_index->set_all(-1);
	flood_next_cell_lon_index->set_all(-1);
	completed_cells = new field<bool>(grid_params_in);
	completed_cells->set_all(false);
	cell_areas = new field<double>(grid_params_in);
	cell_areas->set_all(1.0);
	raw_orography = new field<double>(grid_params_in);
	raw_orography->set_all(0.0);
	corrected_orography = new field<double>(grid_params_in);
	corrected_orography->set_all(0.0);
	catchments_from_sink_filling = new field<int>(grid_params_in);
	catchments_from_sink_filling->set_all(0);
	basin_flooded_cells = new field<bool>(grid_params_in);
	basin_flooded_cells->set_all(false);
	basin_connected_cells = new field<bool>(grid_params_in);
	basin_connected_cells->set_all(false);
	basin_connect_and_fill_order = new vector<pair<coords*,bool>*>();
	center_coords = center_coords_in;
	previous_filled_cell_coords = previous_filled_cell_coords_in;
	previous_filled_cell_height_type = previous_filled_cell_height_type_in;
	basin_number = (*basin_numbers)(previous_filled_cell_coords_in);
	basin_connections.add_set(basin_number);
	int target_basin_number = (*basin_numbers)(new_center_coords_in);
	basin_connections.add_set(target_basin_number);
	coords* target_basin_center_coords =
		basin_catchment_centers[target_basin_number -1];
	for (int i = 0;i<target_basin_number-1;i++){
		basin_connect_and_fill_orders.push_back(new vector<pair<coords*,bool>*>());
	}
	basin_connect_and_fill_order->push_back(new pair<coords*,bool>(target_basin_center_coords->clone(),
	                                                           true));
	basin_connect_and_fill_orders.push_back(basin_connect_and_fill_order);
	basin_merges_and_redirects =
		new merges_and_redirects(latlon_merge_and_redirect_indices_factory,grid_params_in);
	basin_merges_and_redirects->set_unmatched_flood_merge(target_basin_center_coords);
	(*requires_flood_redirect_indices)(target_basin_center_coords) = true;
	(*flood_next_cell_lat_index)(target_basin_center_coords) =
		static_cast<latlon_coords*>(previous_filled_cell_coords_in)->get_lat();
	(*flood_next_cell_lon_index)(target_basin_center_coords) =
		static_cast<latlon_coords*>(previous_filled_cell_coords_in)->get_lon();
	_grid = grid_factory(_grid_params);
	_coarse_grid = grid_factory(_coarse_grid_params);
	int root_target_basin_number = basin_connections.find_root(target_basin_number);
	rebuild_secondary_basin(root_target_basin_number);
	//process primary merge calls set primary redirect
	process_primary_merge(target_basin_number,root_target_basin_number);
        delete catchments_from_sink_filling;
        while (! q.empty()){
          center_cell = static_cast<basin_cell*>(q.top());
          q.pop();
          delete center_cell;
        }
}

void latlon_basin_evaluation_algorithm::test_set_secondary_redirect(double* prior_coarse_rdirs_in,
                                                                    int* flood_next_cell_lat_index_in,
                                                                    int* flood_next_cell_lon_index_in,
                                                                    int* connect_next_cell_lat_index_in,
                                                                    int* connect_next_cell_lon_index_in,
                                                                    int* coarse_catchment_nums_in,
                                                                    bool* requires_flood_redirect_indices_in,
                                                                    bool* requires_connect_redirect_indices_in,
                                                                    double* raw_orography_in,
                    												double* corrected_orography_in,
                    												coords* new_center_coords_in,
                                                                  	coords* center_coords_in,
                                                                  	coords* previous_filled_cell_coords_in,
                                                                  	coords* target_basin_center_coords,
                                                                  	height_types& previous_filled_cell_height_type_in,
                                                                  	grid_params* grid_params_in,
                                                                  	grid_params* coarse_grid_params_in){
	_grid_params = grid_params_in;
	_coarse_grid_params = coarse_grid_params_in;
	center_coords = center_coords_in;
	previous_filled_cell_coords = previous_filled_cell_coords_in;
	coarse_catchment_nums = new field<int>(coarse_catchment_nums_in,coarse_grid_params_in);
	prior_coarse_rdirs = new field<double>(prior_coarse_rdirs_in,coarse_grid_params_in);
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
	field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  	field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  	connect_merge_and_redirect_indices_index->set_all(-1);
  	flood_merge_and_redirect_indices_index->set_all(-1);
  	vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  	vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
	merge_and_redirect_indices* working_redirect = new latlon_merge_and_redirect_indices(new_center_coords_in);
	vector<merge_and_redirect_indices*>* primary_merges =
    	new vector<merge_and_redirect_indices*>;
  	collected_merge_and_redirect_indices*
  		collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                                     working_redirect,
                                                                     latlon_merge_and_redirect_indices_factory);
  	flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  	(*flood_merge_and_redirect_indices_index)(previous_filled_cell_coords) = 0;
	basin_merges_and_redirects = new merges_and_redirects(connect_merge_and_redirect_indices_index,
                               						      flood_merge_and_redirect_indices_index,
                               						      connect_merge_and_redirect_indices_vector,
                               						      flood_merge_and_redirect_indices_vector,
                               						      grid_params_in);
	_coarse_grid = grid_factory(_coarse_grid_params);
	set_secondary_redirect(working_redirect,previous_filled_cell_coords,target_basin_center_coords,
	                       previous_filled_cell_height_type_in);
	}

void latlon_basin_evaluation_algorithm::test_set_remaining_redirects(vector<coords*> basin_catchment_centers_in,
                                                                     double* prior_fine_rdirs_in,
                                                                     double* prior_coarse_rdirs_in,
                                                                     bool* requires_flood_redirect_indices_in,
                                                                     bool* requires_connect_redirect_indices_in,
                                                                     int* basin_numbers_in,
                                                                     int* prior_fine_catchments_in,
                                                                     int* coarse_catchment_nums_in,
                                                                     int* flood_next_cell_lat_index_in,
                                                                     int* flood_next_cell_lon_index_in,
                                                                     int* connect_next_cell_lat_index_in,
                                                                     int* connect_next_cell_lon_index_in,
                                                                  	 grid_params* grid_params_in,
                                                                  	 grid_params* coarse_grid_params_in){
	_grid_params = grid_params_in;
	_coarse_grid_params = coarse_grid_params_in;
	basin_catchment_centers = basin_catchment_centers_in;
	prior_fine_rdirs = new field<double>(prior_fine_rdirs_in,grid_params_in);
	prior_coarse_rdirs = new field<double>(prior_coarse_rdirs_in,coarse_grid_params_in);
	requires_flood_redirect_indices = new field<bool>(requires_flood_redirect_indices_in,grid_params_in);
	requires_connect_redirect_indices = new field<bool>(requires_connect_redirect_indices_in,grid_params_in);
	basin_numbers = new field<int>(basin_numbers_in,grid_params_in);
	prior_fine_catchments = new field<int>(prior_fine_catchments_in,grid_params_in);
	coarse_catchment_nums = new field<int>(coarse_catchment_nums_in,coarse_grid_params_in);
	flood_next_cell_lat_index = new field<int>(flood_next_cell_lat_index_in,grid_params_in);
	flood_next_cell_lon_index = new field<int>(flood_next_cell_lon_index_in,grid_params_in);
	connect_next_cell_lat_index = new field<int>(connect_next_cell_lat_index_in,grid_params_in);
	connect_next_cell_lon_index = new field<int>(connect_next_cell_lon_index_in,grid_params_in);
	search_completed_cells = new field<bool>(grid_params_in);
	_grid = grid_factory(_grid_params);
	_coarse_grid = grid_factory(_coarse_grid_params);
	null_coords = new latlon_coords(-1,-1);
	int highest_catchment_num = *std::max_element(prior_fine_catchments_in,
	                                    		 			prior_fine_catchments_in+_grid->get_total_size());
	for (int i = 0; i < highest_catchment_num; i++){
		basin_sink_points.push_back(null_coords->clone());
	}
	basin_merges_and_redirects =
		new merges_and_redirects(latlon_merge_and_redirect_indices_factory,grid_params_in);
	_grid->for_all([&](coords* coords_in){
	if ((*requires_flood_redirect_indices)(coords_in)){
			collected_merge_and_redirect_indices*
				working_collected_merge_and_redirect_indices =
					basin_merges_and_redirects->
						get_collected_merge_and_redirect_indices(coords_in,
                                             			   flood_height,
                                             			   true);
			working_collected_merge_and_redirect_indices->
				set_unmatched_secondary_merge(coords_in);
	}
	if ((*requires_connect_redirect_indices)(coords_in)){
		collected_merge_and_redirect_indices*
			working_collected_merge_and_redirect_indices =
				basin_merges_and_redirects->
					get_collected_merge_and_redirect_indices(coords_in,
                                         			   connection_height,
                                         		     true);
		working_collected_merge_and_redirect_indices->
			set_unmatched_secondary_merge(coords_in);
	}
  	delete coords_in;
  });
	set_remaining_redirects();
	while ( ! basin_sink_points.empty()){
		coords* basin_sink_point = basin_sink_points.back();
		basin_sink_points.pop_back();
		delete basin_sink_point;
	}
}

bool latlon_basin_evaluation_algorithm::check_for_sinks_and_set_downstream_coords(coords* coords_in){
	double rdir = (*prior_fine_rdirs)(coords_in);
	downstream_coords = _grid->calculate_downstream_coords_from_dir_based_rdir(coords_in,rdir);
	double next_rdir = (*prior_fine_rdirs)(downstream_coords);
	return (rdir == 5.0 || next_rdir == 0.0);
}

bool latlon_basin_evaluation_algorithm::coarse_cell_is_sink(coords* coords_in){
	double rdir = (*prior_coarse_rdirs)(coords_in);
	return (rdir == 5.0);
}

void latlon_basin_evaluation_algorithm::
	set_previous_cells_flood_next_cell_index(coords* coords_in){
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	(*flood_next_cell_lat_index)(previous_filled_cell_coords) = latlon_coords_in->get_lat();
	(*flood_next_cell_lon_index)(previous_filled_cell_coords) = latlon_coords_in->get_lon();
}

void latlon_basin_evaluation_algorithm::
	set_previous_cells_connect_next_cell_index(coords* coords_in){
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	(*connect_next_cell_lat_index)(previous_filled_cell_coords) = latlon_coords_in->get_lat();
	(*connect_next_cell_lon_index)(previous_filled_cell_coords) = latlon_coords_in->get_lon();
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

void latlon_basin_evaluation_algorithm::output_diagnostics_for_grid_section(int min_lat,int max_lat,
                                                                        	  int min_lon,int max_lon){
	cout << "Raw Orography" << endl;
	for (int i = min_lat; i <= max_lat; i++){
	  for (int j = min_lon; j <= max_lon; j++){
	    cout << setw(3) << (*raw_orography)(new latlon_coords(i,j)) << ", ";
	  }
	  cout << endl;
	}
	cout << "Corrected Orography" << endl;
	for (int i = min_lat; i <= max_lat; i++){
	  for (int j = min_lon; j <= max_lon; j++){
	    cout << setw(3) << (*corrected_orography)(new latlon_coords(i,j)) << ", ";
	  }
	  cout << endl;
	}
	cout << "Minima" << endl;
	string boolean_value;
	for (int i = min_lat; i <= max_lat; i++){
	  for (int j = min_lon; j <= max_lon; j++){
	  	boolean_value = (*minima)(new latlon_coords(i,j)) ? "true, " : "false, ";
	    cout << boolean_value;
	  }
	  cout << endl;
	}
	cout << "Fine Catchments" << endl;
	for (int i = min_lat; i <= max_lat; i++){
	  for (int j = min_lon; j <= max_lon; j++){
	    cout << setw(3) << (*prior_fine_catchments)(new latlon_coords(i,j)) << ", ";
	  }
	  cout << endl;
	}
	cout << "Fine River Directions" << endl;
	for (int i = min_lat; i <= max_lat; i++){
	  for (int j = min_lon; j <= max_lon; j++){
	    cout << setw(3) << (*prior_fine_rdirs)(new latlon_coords(i,j)) << ", ";
	  }
	  cout << endl;
	}
}

bool icon_single_index_basin_evaluation_algorithm::check_for_sinks_and_set_downstream_coords(coords* coords_in){
	int rdir = (*prior_fine_rdirs)(coords_in);
	downstream_coords = _grid->calculate_downstream_coords_from_index_based_rdir(coords_in,rdir);
	int next_rdir = (*prior_fine_rdirs)(downstream_coords);
	return (rdir == true_sink_value || next_rdir == outflow_value);
}

bool icon_single_index_basin_evaluation_algorithm::coarse_cell_is_sink(coords* coords_in){
	int rdir = (*prior_coarse_rdirs)(coords_in);
	return (rdir == true_sink_value);
}

void icon_single_index_basin_evaluation_algorithm::
	set_previous_cells_flood_next_cell_index(coords* coords_in){
	generic_1d_coords* generic_1d_coords_in = static_cast<generic_1d_coords*>(coords_in);
	(*flood_next_cell_index)(previous_filled_cell_coords) = generic_1d_coords_in->get_index();
}

void icon_single_index_basin_evaluation_algorithm::
	set_previous_cells_connect_next_cell_index(coords* coords_in){
	generic_1d_coords* generic_1d_coords_in = static_cast<generic_1d_coords*>(coords_in);
	(*connect_next_cell_index)(previous_filled_cell_coords) = generic_1d_coords_in->get_index();
}

coords* icon_single_index_basin_evaluation_algorithm::get_cells_next_cell_index_as_coords(coords* coords_in,
                                                                               height_types height_type_in){

	generic_1d_coords* generic_1d_coords_in = static_cast<generic_1d_coords*>(coords_in);
	if (height_type_in == flood_height)
		return new generic_1d_coords((*flood_next_cell_index)(generic_1d_coords_in));
	else if (height_type_in == connection_height)
		return new generic_1d_coords((*connect_next_cell_index)(generic_1d_coords_in));
	else throw runtime_error("Height type not recognized");
}

void icon_single_index_basin_evaluation_algorithm::
		 output_diagnostics_for_grid_section(int min_lat,int max_lat,
                                         int min_lon,int max_lon){
		 	throw runtime_error("Not implemented for icosohedral grid");
		 }
