/*
 * sink_filling_algorithm.cpp
 *
 *  Created on: May 20, 2016
 *      Author: thomasriddick
 */
#include <iostream>
#include <limits>
#include <algorithm>
#include "sink_filling_algorithm.hpp"

using namespace std;

/* IMPORTANT!
 * Note that a description of each functions purpose is provided with the function declaration in
 * the header file not with the function definition. The definition merely provides discussion of
 * details of the implementation.
 */

const double SQRT_TWO = sqrt(2);

//A variable equal to the smallest possible double used as a non data value
double sink_filling_algorithm::no_data_value = numeric_limits<double>::lowest();

sink_filling_algorithm::sink_filling_algorithm(field<double>* orography,grid_params* grid_params_in,
											   field<bool>* completed_cells,bool* landsea_in,
											   bool set_ls_as_no_data_flag,bool* true_sinks_in,
											   field<int>* catchment_nums_in) :
											   _grid_params(grid_params_in),
											   orography(orography),
											   completed_cells(completed_cells),
											   catchment_nums(catchment_nums_in),
											   set_ls_as_no_data_flag(set_ls_as_no_data_flag)
{
	_grid = grid_factory(_grid_params);
	landsea = landsea_in ? new field<bool>(landsea_in,_grid_params): nullptr;
	true_sinks = true_sinks_in ? new field<bool>(true_sinks_in,_grid_params): nullptr;
}

sink_filling_algorithm::~sink_filling_algorithm() {
	delete orography; delete completed_cells; delete true_sinks;
	delete _grid;
	if (tarasov_mod) {
		delete tarasov_path_initial_heights; delete tarasov_landsea_neighbors;
		delete tarasov_active_true_sink; delete tarasov_path_lengths;
		delete tarasov_maximum_separations_from_initial_edge;
		delete tarasov_initial_edge_nums;
		delete catchment_nums;
		while (!q.empty()){
			auto cell = q.top();
			q.pop();
			delete cell;
		}
	}
}

sink_filling_algorithm_1::sink_filling_algorithm_1(field<double>* orography,grid_params* grid_params_in,
												   field<bool>* completed_cells,bool* landsea_in,
												   bool set_ls_as_no_data_flag,bool add_slope,
												   double epsilon, bool* true_sinks_in)
	: sink_filling_algorithm(orography,grid_params_in,completed_cells,landsea_in,set_ls_as_no_data_flag,
							 true_sinks_in,nullptr), add_slope(add_slope), epsilon(epsilon) {}

sink_filling_algorithm_1_latlon::sink_filling_algorithm_1_latlon(field<double>* orography,grid_params* grid_params_in,
		   	   	   	   	   	   									 field<bool>* completed_cells,bool* landsea_in,
																 bool set_ls_as_no_data_flag,bool add_slope,
																 double epsilon,bool* true_sinks_in)
	: sink_filling_algorithm(orography,grid_params_in,completed_cells,landsea_in,set_ls_as_no_data_flag,
							 true_sinks_in,nullptr),
	  sink_filling_algorithm_1(orography,grid_params_in,completed_cells,landsea_in,set_ls_as_no_data_flag,
			  	  	  	  	   add_slope,epsilon,true_sinks_in)
{
	auto* grid_latlon = dynamic_cast<latlon_grid*>(_grid);
	nlat = grid_latlon->get_nlat(); nlon = grid_latlon->get_nlon();
}

sink_filling_algorithm_4::sink_filling_algorithm_4(field<double>* orography,grid_params* grid_params_in,
											       field<bool>* completed_cells,
												   bool* landsea_in, bool set_ls_as_no_data_flag,
												   field<int>* catchment_nums_in,
												   bool prefer_non_diagonal_initial_dirs,
												   bool* true_sinks_in)
	: sink_filling_algorithm(orography,grid_params_in,completed_cells,landsea_in,set_ls_as_no_data_flag,
			                 true_sinks_in,catchment_nums_in),
	  prefer_non_diagonal_initial_dirs(prefer_non_diagonal_initial_dirs) {}

sink_filling_algorithm_4_latlon::sink_filling_algorithm_4_latlon(field<double>* orography,grid_params* grid_params_in,
											       	   	         field<bool>* completed_cells,
																 bool* landsea_in, bool set_ls_as_no_data_flag,
																 field<int>* catchment_nums_in,
																 bool prefer_non_diagonal_initial_dirs,
																 bool index_based_rdirs_only_in,
																 field<int>* next_cell_lat_index_in,
																 field<int>* next_cell_lon_index_in,
																 bool* true_sinks_in,
																 field<short>* rdirs_in)
	: sink_filling_algorithm(orography,grid_params_in,completed_cells,landsea_in,set_ls_as_no_data_flag,
            true_sinks_in,catchment_nums_in),
	  sink_filling_algorithm_4(orography,grid_params_in,completed_cells,landsea_in,set_ls_as_no_data_flag,
			   	   	   	       catchment_nums_in,prefer_non_diagonal_initial_dirs,true_sinks_in),
	  rdirs(rdirs_in), next_cell_lat_index(next_cell_lat_index_in), next_cell_lon_index(next_cell_lon_index_in)
	{
		index_based_rdirs_only=index_based_rdirs_only_in;
		auto* grid_latlon = dynamic_cast<latlon_grid*>(_grid);
		nlat = grid_latlon->get_nlat(); nlon = grid_latlon->get_nlon();
	}

void sink_filling_algorithm::setup_flags(bool set_ls_as_no_data_flag_in, bool tarasov_mod_in,
										 int tarasov_separation_threshold_for_returning_to_same_edge_in,
										 double tarasov_min_path_length_in,
										 bool tarasov_include_corners_in_same_edge_criteria_in)
{
	set_ls_as_no_data_flag = set_ls_as_no_data_flag_in;
	tarasov_mod = tarasov_mod_in;
	tarasov_separation_threshold_for_returning_to_same_edge =
			tarasov_separation_threshold_for_returning_to_same_edge_in;
	tarasov_min_path_length = tarasov_min_path_length_in;
	tarasov_include_corners_in_same_edge_criteria = tarasov_include_corners_in_same_edge_criteria_in;
}

void sink_filling_algorithm_1::setup_flags(bool set_ls_as_no_data_flag_in, bool tarasov_mod_in,
										   bool add_slope_in, double epsilon_in,
										   int tarasov_separation_threshold_for_returning_to_same_edge_in,
										   double tarasov_min_path_length_in,
										   bool tarasov_include_corners_in_same_edge_criteria_in)
{
	sink_filling_algorithm::setup_flags(set_ls_as_no_data_flag_in,tarasov_mod_in,
										tarasov_separation_threshold_for_returning_to_same_edge_in,
										tarasov_min_path_length_in,
										tarasov_include_corners_in_same_edge_criteria_in);
	add_slope = add_slope_in; epsilon = epsilon_in;
}

void sink_filling_algorithm_4::setup_flags(bool set_ls_as_no_data_flag_in,
			     	 	 	 	 	 	   bool prefer_non_diagonal_initial_dirs_in,
										   bool tarasov_mod_in,
										   int tarasov_separation_threshold_for_returning_to_same_edge_in,
										   double tarasov_min_path_length_in,
										   bool tarasov_include_corners_in_same_edge_criteria_in)
{
	sink_filling_algorithm::setup_flags(set_ls_as_no_data_flag_in,tarasov_mod_in,
										tarasov_separation_threshold_for_returning_to_same_edge_in,
										tarasov_min_path_length_in,
										tarasov_include_corners_in_same_edge_criteria_in);
	prefer_non_diagonal_initial_dirs = prefer_non_diagonal_initial_dirs_in;
}

void sink_filling_algorithm_4_latlon::setup_flags(bool set_ls_as_no_data_flag_in,
			     	 	 	 	 	 	   	   	  bool prefer_non_diagonal_initial_dirs_in,
												  			bool tarasov_mod_in,
												  			bool index_based_rdirs_only_in,
												  			int tarasov_separation_threshold_for_returning_to_same_edge_in,
												  			double tarasov_min_path_length_in,
												  			bool tarasov_include_corners_in_same_edge_criteria_in)
{
	sink_filling_algorithm_4::setup_flags(set_ls_as_no_data_flag_in,prefer_non_diagonal_initial_dirs_in,
										  tarasov_mod_in,
										  tarasov_separation_threshold_for_returning_to_same_edge_in,
										  tarasov_min_path_length_in,
										  tarasov_include_corners_in_same_edge_criteria_in);
	index_based_rdirs_only = index_based_rdirs_only_in;
}

void sink_filling_algorithm::setup_fields(double* orography_in, bool* landsea_in,
									      bool* true_sinks_in,grid_params* grid_params)
{
	_grid_params = grid_params;
	if (tarasov_mod) _grid_params->set_nowrap(true);
	_grid = grid_factory(_grid_params);
	orography = new field<double>(orography_in,grid_params);
	completed_cells = new field<bool>(grid_params);  //Cells that have already been processed
	completed_cells->set_all(false);
	landsea = landsea_in ? new field<bool>(landsea_in,grid_params): nullptr;
	true_sinks = true_sinks_in ? new field<bool>(true_sinks_in,grid_params): nullptr;
	tarasov_path_initial_heights = tarasov_mod ? new field<double>(grid_params): nullptr;
	catchment_nums = tarasov_mod ? new field<int>(grid_params): nullptr;
	tarasov_landsea_neighbors = tarasov_mod ? new field<bool>(grid_params): nullptr;
	tarasov_active_true_sink = tarasov_mod ? new field<bool>(grid_params): nullptr;
	tarasov_path_lengths = tarasov_mod ? new field<double>(grid_params): nullptr;
	tarasov_maximum_separations_from_initial_edge = tarasov_mod ? new field<int>(grid_params): nullptr;
	tarasov_initial_edge_nums = tarasov_mod ? new field<int>(grid_params): nullptr;
	if (tarasov_mod) {
		tarasov_landsea_neighbors->set_all(false);
		tarasov_active_true_sink->set_all(false);
		if (!true_sinks) {
			//Tarasov code always uses true sinks
			true_sinks = new field<bool>(grid_params);
			true_sinks->set_all(false);
		}
	}
}

void sink_filling_algorithm_4::setup_fields(double* orography_in, bool* landsea_in,
								            bool* true_sinks_in, grid_params* grid_params_in,
											int* catchment_nums_in)
{
	sink_filling_algorithm::setup_fields(orography_in,landsea_in,true_sinks_in,grid_params_in);
	if (! catchment_nums) catchment_nums = new field<int>(catchment_nums_in,grid_params_in);
}

void sink_filling_algorithm_4_latlon::setup_fields(double* orography_in, bool* landsea_in,
								            	   bool* true_sinks_in, int * next_cell_lat_index_in,
												   int * next_cell_lon_index_in,
												   grid_params* grid_params_in,
												   short* rdirs_in, int* catchment_nums_in)
{
	sink_filling_algorithm_4::setup_fields(orography_in,landsea_in,true_sinks_in,grid_params_in,
										   catchment_nums_in);
	rdirs = new field<short>(rdirs_in,grid_params_in);
	next_cell_lat_index = new field<int>(next_cell_lat_index_in,grid_params_in);
	next_cell_lon_index = new field<int>(next_cell_lon_index_in,grid_params_in);
	auto* grid_latlon = dynamic_cast<latlon_grid*>(_grid);
	nlat = grid_latlon->get_nlat(); nlon = grid_latlon->get_nlon();
}

void sink_filling_algorithm_4_icon_single_index::setup_fields(double* orography_in, bool* landsea_in,
								            	   															bool* true_sinks_in, int * next_cell_index_in,
												   																	 	grid_params* grid_params_in,
												   																	  int* catchment_nums_in) {
	sink_filling_algorithm_4::setup_fields(orography_in,landsea_in,true_sinks_in,grid_params_in,
										   									catchment_nums_in);
	next_cell_index = new field<int>(next_cell_index_in,grid_params_in);
	auto* grid_icon_single_index = dynamic_cast<icon_single_index_grid*>(_grid);
	ncells = grid_icon_single_index->get_npoints();
}

void sink_filling_algorithm_1_latlon::setup_fields(double* orography_in, bool* landsea_in,
									      	  	   bool* true_sinks_in,grid_params* grid_params){
	sink_filling_algorithm::setup_fields(orography_in,landsea_in,true_sinks_in,grid_params);
	auto* grid_latlon = dynamic_cast<latlon_grid*>(_grid);
	nlat = grid_latlon->get_nlat(); nlon = grid_latlon->get_nlon();
}

void sink_filling_algorithm_1_icon_single_index::setup_fields(double* orography_in, bool* landsea_in,
									      	  	   						bool* true_sinks_in,grid_params* grid_params) {
	sink_filling_algorithm::setup_fields(orography_in,landsea_in,true_sinks_in,grid_params);
	auto* grid_icon_single_index = dynamic_cast<icon_single_index_grid*>(_grid);
	ncells = grid_icon_single_index->get_npoints();
}

//Implementation as per algorithm 1 of the paper cited in the preamble to this file
void sink_filling_algorithm::fill_sinks()
{
	//Reset the method variable of the base class
	method = get_method();
	//Set out-flow points
	if (!tarasov_mod) cout << "Setting out-flow points" << endl;
	add_edge_cells_to_q();
	if (true_sinks) add_true_sinks_to_q();
	//Algorithm is now finished with landsea (except for tarasov modification)
	if (!tarasov_mod) delete landsea;
	//Loop over non out-flow points searching and filling sinks (or assigning flow directions);
	//this is the main section of the program
	if (!tarasov_mod) cout << "Starting to loop over non out-flow points searching for sinks" << endl;
	while (! (q.empty() && pit_q.empty() && slope_q.empty()) ) {
		pop_center_cell();
		center_coords = center_cell->get_cell_coords();
		#if DEBUG
			cout << "q length: " << q.size() << endl;
			cout << "Processing central cell:" << endl;
			if (auto center_coords_latlon = dynamic_cast<latlon_coords*>(center_coords)){
				cout << "lat: " << center_coords_latlon->get_lat() << " lon: "<< center_coords_latlon->get_lon() << endl;
			} else if (auto center_coords_generic_1d = dynamic_cast<generic_1d_coords*>(center_coords)) {
				cout << "Cell " << *center_coords_generic_1d;
			}
			cout << "Cell orog: " << (*orography)(center_coords) << endl;
			if (tarasov_mod) {
				cout << "Warning follow values NOT valid for cell with ignored true sink" << endl;
				cout << "Center cell initial path height: " << center_cell->get_tarasov_path_initial_height() << endl;
				cout << "Center cell path length: " << center_cell->get_tarasov_path_length() << endl;
				cout << "Center cell max sep from initial edge: "
				     << center_cell->get_tarasov_maximum_separation_from_initial_edge() << endl;
			}
		#endif
		process_center_cell();
		if (tarasov_mod) {
			tarasov_update_maximum_separation_from_initial_edge();
			if(_grid->is_edge(center_coords) || (*tarasov_landsea_neighbors)(center_coords) ||
			   (*tarasov_active_true_sink)(center_coords)) {
				if (tarasov_is_shortest_permitted_path()) {
					tarasov_set_area_height();
					delete center_cell;
					break;
				}
			}
		}
		auto neighbors_coords = orography->get_neighbors_coords(center_coords,method);
		process_neighbors(neighbors_coords);
		delete neighbors_coords;
		delete center_cell;
	}
	//Tarasov like modification requires landsea throughout
	if (tarasov_mod) delete landsea;
}

//Handles the initial setup of the queue; there are two versions for each method (of which
//there are also currently two to give a total of four variants): one for if a land sea
//mask is supplied; one for if no land sea mask is supplied. The latter assumes outflow
//points around the edges
void sink_filling_algorithm::add_edge_cells_to_q()
{
	if (landsea && tarasov_mod){
		add_landsea_edge_cells_to_q();
		add_geometric_edge_cells_to_q();
	}
	//If a land sea mask is supplied
	else if (landsea) add_landsea_edge_cells_to_q();
	//No land sea mask supplied; use edges as out flow points; also assign
    //them flow direction if using algorithm 4
	else add_geometric_edge_cells_to_q();
}

void sink_filling_algorithm::add_landsea_edge_cells_to_q(){

	function<void(coords*)> add_edge_cell_to_q_func = [&](coords* coords_in){
		if((*landsea)(coords_in)){
			//set the land sea mask itself to no data if that option has been selected
			if (set_ls_as_no_data_flag) set_ls_as_no_data(coords_in);
			//set all points in the land sea mask as having been processed
			(*completed_cells)(coords_in) = true;
			if (tarasov_mod){
				tarasov_path_initial_height = (*orography)(coords_in);
				(*tarasov_path_initial_heights)(coords_in) =
						tarasov_path_initial_height;
				(*catchment_nums)(coords_in) = -1;
			}
			//Calculate and the process the neighbors of every landsea point and
			//add them to the queue (if they aren't already in the queue or themselves
			//land sea points
			//Note the get_neighbors_coords function is always called here using
			//method 1 technique as this is always appropriate (regardless of which
			//method is being run) for the set-up phase
			auto neighbors_coords = orography->get_neighbors_coords(coords_in);
			while (!neighbors_coords->empty()){
				nbr_coords = neighbors_coords->back();
				//If neither a land sea point nor a cell already in the queue
				//then add this cell to the queue (and possibly assign it
				//a flow direction if this is algorithm 4)
				if (!((*landsea)(nbr_coords) ||
					(*completed_cells)(nbr_coords))) {
					(*completed_cells)(nbr_coords) = true;
					if (tarasov_mod) {
						tarasov_neighbor_path_length =
								tarasov_calculate_neighbors_path_length_change(coords_in);
					}
					push_land_sea_neighbor();
				}
				neighbors_coords->pop_back();
				delete nbr_coords;
			}
			delete neighbors_coords;
		}
		delete coords_in;
	};
	_grid->for_all(add_edge_cell_to_q_func);
}

bool sink_filling_algorithm::tarasov_is_shortest_permitted_path(){
	if (tarasov_center_cell_path_length < tarasov_min_path_length) return false;
	else if (not tarasov_same_edge_criteria_met()) return false;
	else return true;
}

bool sink_filling_algorithm::tarasov_same_edge_criteria_met(){
	bool is_landsea_nbr = landsea ? (*tarasov_landsea_neighbors)(center_coords) : false;
	//Tarasov mod always uses true sinks
	bool is_true_sink   = (*tarasov_active_true_sink)(center_coords);
	if (_grid->check_if_cell_connects_two_landsea_or_true_sink_points(
			tarasov_center_cell_initial_edge_num,is_landsea_nbr,is_true_sink)) return false;
	else if (is_landsea_nbr || is_true_sink) return true;
	else if (_grid->check_if_cell_is_on_given_edge_number(center_coords,
			tarasov_center_cell_initial_edge_num)) {
		if (not tarasov_include_corners_in_same_edge_criteria &&
			_grid->is_corner_cell(center_coords)) return true;
		else if(tarasov_center_cell_maximum_separations_from_initial_edge >=
					tarasov_separation_threshold_for_returning_to_same_edge) return true;
		else return false;
	}
	else return true;
}

void sink_filling_algorithm::tarasov_update_maximum_separation_from_initial_edge(){
	int separation_from_initial_edge = _grid->get_separation_from_initial_edge(center_coords,
			tarasov_center_cell_initial_edge_num);
	if (tarasov_center_cell_maximum_separations_from_initial_edge <
			separation_from_initial_edge){
		center_cell->set_tarasov_maximum_separation_from_initial_edge(separation_from_initial_edge);
		tarasov_center_cell_maximum_separations_from_initial_edge = separation_from_initial_edge;
	}
}


void sink_filling_algorithm_latlon::add_geometric_edge_cells_to_q(){
	bool push_left = true;
	bool push_right = true;
	for (auto i = 0; i < nlat; i++) {
		if (tarasov_mod) {
			if (landsea) {
				auto left_coords = new latlon_coords(i,0);
				auto right_coords = new latlon_coords(i,nlon-1);
				push_left = ! (*landsea)(left_coords);
				push_right = ! (*landsea)(right_coords);
				delete left_coords; delete right_coords;
			} else {
				push_left 	= true;
				push_right = true;
			}
		} else {
			auto left_coords = new latlon_coords(i,0);
			auto right_coords = new latlon_coords(i,nlon-1);
			(*completed_cells)(left_coords) = true;
			(*completed_cells)(right_coords) = true;
			delete left_coords; delete right_coords;
		}
		push_vertical_edge(i,push_left,push_right);
	}
	for (auto j = 1; j < nlon-1; j++){
		bool push_top = true;
		bool push_bottom = true;
		if (tarasov_mod) {
			if (landsea) {
				auto top_coords = new latlon_coords(0,j);
				auto bottom_coords = new latlon_coords(nlat-1,j);
				push_top = ! (*landsea)(top_coords);
				push_bottom = ! (*landsea)(bottom_coords);
				delete top_coords; delete bottom_coords;
			} else {
				push_top 	= true;
				push_bottom = true;
			}
		} else {
			auto top_coords = new latlon_coords(0,j);
			auto bottom_coords = new latlon_coords(nlat-1,j);
			(*completed_cells)(top_coords) = true;
			(*completed_cells)(bottom_coords) = true;
			delete top_coords; delete bottom_coords;
		}
		push_horizontal_edge(j,push_top,push_bottom);
	}
}

void sink_filling_algorithm_icon_single_index::add_geometric_edge_cells_to_q(){
	throw runtime_error("Adding geometric edge cells to queue not implemented for ICON grid");
}

void sink_filling_algorithm::add_true_sinks_to_q()
{
	_grid->for_all([&](coords* coords_in){
			if ((*true_sinks)(coords_in)){
				if (landsea){
					if ((*landsea)(coords_in)) {
						delete coords_in;
						return;
					}
				}
				//ignore sinks next to landsea points... how such a situation could possible occur
				//and therefore the correct hydrology for it is not clear
				if(!(*completed_cells)(coords_in)) push_true_sink(coords_in);
				else {
					//if using tarasov style orography upscaling then delete true
					//sink as the field is used later and needs to be up-to-date
					if(tarasov_mod) (*true_sinks)(coords_in) = false;
					delete coords_in;
				}
			} else {
				delete coords_in;
			}
	});
}

inline void sink_filling_algorithm_1::set_ls_as_no_data(coords* coords_in)
{
	(*orography)(coords_in) = no_data_value;
}

inline void sink_filling_algorithm_4::set_ls_as_no_data(coords* coords_in)
{
	cout << "Setting sea points as no data is incompatible with method 4; ignoring flag" << endl;
}

inline void sink_filling_algorithm_1::push_land_sea_neighbor()
{
	if (tarasov_mod) {
		int new_catchment_num = q.get_next_k_value();
		q.push(new cell((*orography)(nbr_coords),nbr_coords->clone(),new_catchment_num,
		_grid->get_landsea_edge_num(),1,tarasov_neighbor_path_length,
		tarasov_path_initial_height));
		(*catchment_nums)(nbr_coords) = new_catchment_num;
		(*tarasov_path_initial_heights)(nbr_coords) = tarasov_path_initial_height;
		(*tarasov_landsea_neighbors)(nbr_coords) = true;
	}
	else q.push(new cell((*orography)(nbr_coords),nbr_coords->clone()));
}

inline void sink_filling_algorithm_4::push_land_sea_neighbor()
{
	int new_catchment_num = q.get_next_k_value();
	if (tarasov_mod) {
		q.push(new cell((*orography)(nbr_coords),nbr_coords->clone(),new_catchment_num,
			   	     	   (*orography)(nbr_coords),_grid->get_landsea_edge_num(),1,
						   tarasov_neighbor_path_length,tarasov_path_initial_height));
		(*tarasov_path_initial_heights)(nbr_coords) = tarasov_path_initial_height;
		(*tarasov_landsea_neighbors)(nbr_coords) = true;
	}
	else q.push(new cell((*orography)(nbr_coords),nbr_coords->clone(),new_catchment_num,
			   (*orography)(nbr_coords)));
	find_initial_cell_flow_direction();
	(*catchment_nums)(nbr_coords) = new_catchment_num;
}

//Require a minimum of 1 in-line function in sink_filling_algorithm_1_latlon
void sink_filling_algorithm_1_latlon::push_vertical_edge(int i, bool push_left, bool push_right)
{
	if (tarasov_mod) {
		int new_catchment_num = 0;
		if (push_left) {
			auto cell_coords = new latlon_coords(i,0);
			new_catchment_num = q.get_next_k_value();
			double cell_orog = (*orography)(cell_coords);
			q.push(new cell(cell_orog,cell_coords,new_catchment_num,
				   _grid->get_edge_number(cell_coords),0,
				   1.0,cell_orog));
			(*catchment_nums)(cell_coords) = new_catchment_num;
			(*tarasov_path_initial_heights)(cell_coords) = cell_orog;
		}
		if (push_right) {
			auto cell_coords = new latlon_coords(i,nlon-1);
			new_catchment_num = q.get_next_k_value();
			double cell_orog = (*orography)(cell_coords);
			q.push(new cell(cell_orog,cell_coords,new_catchment_num,
					        _grid->get_edge_number(cell_coords),0,
							1.0,cell_orog));
			(*catchment_nums)(cell_coords) = new_catchment_num;
			(*tarasov_path_initial_heights)(cell_coords) = cell_orog;
		}
	} else {
		if (push_left) {
			auto cell_coords = new latlon_coords(i,0);
			q.push(new cell((*orography)(cell_coords),cell_coords));
		}
		if (push_right) {
			auto cell_coords = new latlon_coords(i,nlon-1);
			q.push(new cell((*orography)(cell_coords),cell_coords));
		}
	}
}

void sink_filling_algorithm_1_icon_single_index::push_diagonal_edge(int i, bool push_left, bool push_right){
	throw runtime_error("Adding geometric edge cells to queue not implemented for ICON grid");
}

inline void sink_filling_algorithm_4_latlon::push_vertical_edge(int i, bool push_left, bool push_right)
{
	int new_catchment_num = 0;
	if (push_left) {
		auto cell_coords = new latlon_coords(i,0);
		new_catchment_num = q.get_next_k_value();
		if (tarasov_mod) {
			double cell_orog = (*orography)(cell_coords);
			q.push(new cell(cell_orog,cell_coords,new_catchment_num,cell_orog,
						    _grid->get_edge_number(cell_coords),0,
							1.0,cell_orog));
			(*tarasov_path_initial_heights)(cell_coords) = cell_orog;
		}
		else q.push(new cell((*orography)(cell_coords),cell_coords,new_catchment_num,
				             (*orography)(cell_coords)));
		(*catchment_nums)(cell_coords) = new_catchment_num;
		(*rdirs)(cell_coords) = 4;
	}
	if (push_right) {
		auto cell_coords = new latlon_coords(i,nlon-1);
		new_catchment_num = q.get_next_k_value();
		if (tarasov_mod) {
			double cell_orog = (*orography)(cell_coords);
			q.push(new cell(cell_orog,cell_coords,new_catchment_num,cell_orog,
							_grid->get_edge_number(cell_coords),0,
							1.0,cell_orog));
			(*tarasov_path_initial_heights)(cell_coords) = cell_orog;
		}
		else q.push(new cell((*orography)(cell_coords),cell_coords,new_catchment_num,
						     (*orography)(cell_coords)));
		(*catchment_nums)(cell_coords) = new_catchment_num;
		(*rdirs)(cell_coords) = 6;
	}
}

void sink_filling_algorithm_4_icon_single_index::push_diagonal_edge(int i, bool push_left, bool push_right){
	throw runtime_error("Adding geometric edge cells to queue not implemented for ICON grid");
}

inline void sink_filling_algorithm_1_latlon::push_horizontal_edge(int j, bool push_top, bool push_bottom)
{
	if (tarasov_mod) {
		int new_catchment_num = 0;
		if(push_top) {
			auto cell_coords = new latlon_coords(0,j);
			new_catchment_num = q.get_next_k_value();
			double cell_orog = (*orography)(cell_coords);
			q.push(new cell(cell_orog,cell_coords,new_catchment_num,
							_grid->get_edge_number(cell_coords),0,
							1.0,cell_orog));
			(*catchment_nums)(cell_coords) = new_catchment_num;
			(*tarasov_path_initial_heights)(cell_coords) = cell_orog;
		}
		if(push_bottom) {
			auto cell_coords = new latlon_coords(nlat-1,j);
			new_catchment_num = q.get_next_k_value();
			double cell_orog = (*orography)(cell_coords);
			q.push(new cell(cell_orog,cell_coords,new_catchment_num,
							_grid->get_edge_number(cell_coords),0,
							1.0,cell_orog));
			(*catchment_nums)(cell_coords) = new_catchment_num;
			(*tarasov_path_initial_heights)(cell_coords) = cell_orog;
		}
	} else {
		if (push_top) {
			auto cell_coords = new latlon_coords(0,j);
			q.push(new cell((*orography)(cell_coords),cell_coords));
		}
		if (push_bottom) {
			auto cell_coords = new latlon_coords(nlat-1,j);
			q.push(new cell((*orography)(cell_coords),cell_coords));
		}
	}
}

void sink_filling_algorithm_1_icon_single_index::push_horizontal_edge(int i){
	throw runtime_error("Adding geometric edge cells to queue not implemented for ICON grid");
}

inline void sink_filling_algorithm_4_latlon::push_horizontal_edge(int j, bool push_top, bool push_bottom)
{
	int new_catchment_num = 0;
	if(push_top) {
		auto cell_coords = new latlon_coords(0,j);
		new_catchment_num = q.get_next_k_value();
		if (tarasov_mod) {
			double cell_orog = (*orography)(cell_coords);
			q.push(new cell(cell_orog,cell_coords,new_catchment_num,cell_orog,
						    _grid->get_edge_number(cell_coords),0,
							1.0,cell_orog));
			(*tarasov_path_initial_heights)(cell_coords) = cell_orog;
		}
		else q.push(new cell((*orography)(cell_coords),cell_coords,new_catchment_num,
							 (*orography)(cell_coords)));
		(*catchment_nums)(cell_coords) = new_catchment_num;
		(*rdirs)(cell_coords) = 8;
	}
	if(push_bottom) {
		auto cell_coords = new latlon_coords(nlat-1,j);
		new_catchment_num = q.get_next_k_value();
		if (tarasov_mod) {
			double cell_orog = (*orography)(cell_coords);
			q.push(new cell(cell_orog,cell_coords,new_catchment_num,cell_orog,
							_grid->get_edge_number(cell_coords),0,
							1.0,cell_orog));
			(*tarasov_path_initial_heights)(cell_coords) = cell_orog;
		}
		else q.push(new cell((*orography)(cell_coords),cell_coords,new_catchment_num,
							 (*orography)(cell_coords)));
		(*catchment_nums)(cell_coords) = new_catchment_num;
		(*rdirs)(cell_coords) = 2;
	}
}

void sink_filling_algorithm_4_icon_single_index::push_horizontal_edge(int i){
	throw runtime_error("Adding geometric edge cells to queue not implemented for ICON grid");
}

inline void sink_filling_algorithm_1::push_true_sink(coords* coords_in)
{
	if (tarasov_mod) {
		int new_catchment_num = q.get_next_k_value();
		double cell_orog = (*orography)(coords_in);
		q.push_true_sink(new cell(cell_orog,coords_in,
					     	 	  new_catchment_num,_grid->get_true_sink_edge_num(),
								  0,1.0,cell_orog));
		(*tarasov_path_initial_heights)(coords_in) = cell_orog;
	}
	else q.push_true_sink(new cell((*orography)(coords_in),coords_in));
}

inline void sink_filling_algorithm_4::push_true_sink(coords* coords_in)
{
	int new_catchment_num = q.get_next_k_value();
	double cell_orog = (*orography)(coords_in);
	if (tarasov_mod) {
		q.push_true_sink(new cell(cell_orog,coords_in,new_catchment_num,
	  	     	 	 	 	 	 cell_orog,_grid->get_true_sink_edge_num(),
									   0,1.0,cell_orog));
		(*tarasov_path_initial_heights)(coords_in) = cell_orog;
	}
	else q.push_true_sink(new cell(cell_orog,coords_in,new_catchment_num,
							  	   cell_orog));
}

void sink_filling_algorithm::test_add_edge_cells_to_q(bool delete_ls_mask)
{
	add_edge_cells_to_q();
	if(delete_ls_mask) delete landsea;
}

void sink_filling_algorithm::test_add_true_sinks_to_q(bool delete_ls_mask) {
	add_true_sinks_to_q();
	if(delete_ls_mask) delete landsea;
}

//Calculates the flow direction for cell add to the queue as out flow points for algorithm 4;
//dealing with any wrapping. The out flow is the calculated being towards the lowest orography
//neighbor that is sea (in the land sea mask). In the case of the neighbors orography values
//being tied the first minimum is used unless a flag is set and then the last non-diagonal neighbor
//is used (and the first minimum again if there are no non diagonal neighbors). This function deals
//with longitudinal wrapping.
void sink_filling_algorithm_4::find_initial_cell_flow_direction(){
	double min_height = numeric_limits<double>::max();
	double direction = 5.0;
	coords* destination_coords = nbr_coords;
	//Default to 5 for land sea points
	if ((*landsea)(nbr_coords) == true){
		set_cell_to_true_sink_value(nbr_coords);
		return;
	}
	function<void(coords*)> find_init_rdir_func = [&](coords* coords_in){
		//check if out of latitude range
		if (_grid->outside_limits(coords_in)) {
			delete coords_in;
			return;
		}
		//deal with longitudinal wrapping
		auto coords_prime = _grid->wrapped_coords(coords_in);
		if ((*landsea)(coords_prime)){
			if (min_height > (*orography)(coords_prime)){
				min_height = (*orography)(coords_prime);
				if (destination_coords != nbr_coords) delete destination_coords;
				destination_coords = coords_prime;
				//Note the index here is correctly coords_in and not coords_prime!
				if (not index_based_rdirs_only) direction = _grid->calculate_dir_based_rdir(nbr_coords,coords_in);
			} else if(min_height == (*orography)(coords_prime) &&
					  prefer_non_diagonal_initial_dirs &&
					  //This block favors non diagonals if the appropriate flag is set
					  _grid->non_diagonal(nbr_coords,coords_prime)) {
				if (destination_coords != nbr_coords) delete destination_coords;
				destination_coords = coords_prime;
				//make this a part of latlon_grid then cast to that and give nbr_coors
				//and coords_in to that to generate this number, also below
				//Also require setter function below and a is non-diagonal
				//check function
				if(not index_based_rdirs_only) direction = _grid->calculate_dir_based_rdir(nbr_coords,coords_in);
			}
		}
		if (coords_in != coords_prime) delete coords_in;
		if (coords_prime != destination_coords) delete coords_prime;
	};
	_grid->for_all_nbrs(nbr_coords,find_init_rdir_func);
	set_index_based_rdirs(nbr_coords,destination_coords);
	if (not index_based_rdirs_only) static_cast<sink_filling_algorithm_4_latlon*>(this)->set_dir_based_rdir(nbr_coords,direction);
	if(destination_coords != nbr_coords) delete destination_coords;
}

inline void sink_filling_algorithm_1::pop_center_cell(){
	if(tarasov_mod || add_slope){
		center_cell = q.top();
		q.pop();
	} else {
		if (! pit_q.empty()){
			center_cell = pit_q.front();
			pit_q.pop();
			on_slope = false;
		} else if (! slope_q.empty()) {
			center_cell = slope_q.front();
			slope_q.pop();
			on_slope = true;
		} else {
			center_cell = q.top();
			q.pop();
			on_slope = false;
		}
	}
}

inline void sink_filling_algorithm_4::pop_center_cell(){
		center_cell = q.top();
		q.pop();
}

inline void sink_filling_algorithm_1::process_true_sink_center_cell() {
	if (tarasov_mod) {
		if (!(*completed_cells)(center_coords)) {
			(*completed_cells)(center_coords) = true;
			center_catchment_num = center_cell->get_catchment_num();
			(*catchment_nums)(center_coords) = center_catchment_num;
			(*tarasov_active_true_sink)(center_coords) = true;
			tarasov_get_center_cell_values();
			(*tarasov_path_lengths)(center_coords) =
					tarasov_center_cell_path_length;
			tarasov_set_field_values(center_coords);
		} else {
			center_catchment_num = (*catchment_nums)(center_coords);
			tarasov_get_center_cell_values_from_field();
			//If this cell is reprocessed in the future then reprocess it as
			//a normal cell
			(*true_sinks)(center_coords) = false;
		}
	} else (*completed_cells)(center_coords) = true;
}

inline void sink_filling_algorithm_4::process_true_sink_center_cell(){
	double cell_orog = (*orography)(center_coords);
	if (!(*completed_cells)(center_coords)){
		center_catchment_num = center_cell->get_catchment_num();
		if (cell_orog == no_data_value) set_cell_to_no_data_value(center_coords);
		else set_cell_to_true_sink_value(center_coords);
		(*completed_cells)(center_coords) = true;
		(*catchment_nums)(center_coords) = center_catchment_num;
		center_rim_height = cell_orog;
		if (tarasov_mod) {
			(*tarasov_active_true_sink)(center_coords) = true;
			tarasov_get_center_cell_values();
			(*tarasov_path_lengths)(center_coords) =
					tarasov_center_cell_path_length;
			tarasov_set_field_values(center_coords);
			//If this cell is reprocessed in the future then re-process it as
			//a normal cell
			(*true_sinks)(center_coords) = false;
		}
	} else {
		//If this a former true sink that now flow to a neighbor
		//then value in cell object is not the correct one (and
		//it couldn't be corrected as this would require searching
		//the queue for this cell in a previous step ) so have to
		//look it up in array instead
		center_catchment_num = (*catchment_nums)(center_coords);
		//Same for rim height
		center_rim_height = cell_orog;
		if (tarasov_mod) tarasov_get_center_cell_values_from_field();
	}
}

inline void sink_filling_algorithm_1::process_center_cell() {
	center_orography = center_cell->get_orography();
	#if PARALLELIZE
	if(parallelized){
		center_label = (*labels)(center_coords)
		if ( center_label == 0){
			auto neighbors_coords = orography->get_neighbors_coords(center_coords,method);
			bool needs_label = true;
			while (!neighbors_coords->empty()){
				nbr_coords = neighbors_coords->back();
				int nbr_coords_label = (*labels)(nbr_coords);
				if (needs_label &&  nbr_coords_label != 0) {
					(*labels)(center_coords) = nbr_coords_label;
					center_label = nbr_coords_label;
					needs_label = false;
				}
				delete nbr_coords;
			}
			if (needs_label) {
				(*labels)(center_coords) = unique_label_counter;
				center_label = unique_label_counter;
				unique_label_counter++;
			}
		}
	}
	#endif
	if (! (add_slope || tarasov_mod)) requeued = false;
	if (true_sinks){
		if((*true_sinks)(center_coords)) {
			process_true_sink_center_cell();
			return;
		}
	}
	if (tarasov_mod) {
		center_catchment_num = center_cell->get_catchment_num();
		tarasov_get_center_cell_values();
		//This may of not been set if this is an edge cell
		(*completed_cells)(center_coords) = true;
	}
}

inline void sink_filling_algorithm_4::process_center_cell()
{
	if (true_sinks){
		if ((*true_sinks)(center_coords)) {
			process_true_sink_center_cell();
			return;
		}
	}
	center_rim_height = center_cell->get_rim_height();
	center_catchment_num = center_cell->get_catchment_num();
	if (tarasov_mod) {
		tarasov_get_center_cell_values();
		//This may of not been set if this is an edge cell
		(*completed_cells)(center_coords) = true;
	}
	if (queue_minima){
		if ((*minima)(center_coords)) {
			minima_q->push(center_coords->clone());
		}
	}
}

inline void sink_filling_algorithm::tarasov_get_center_cell_values(){
	tarasov_center_cell_path_initial_height =
			center_cell->get_tarasov_path_initial_height();
	tarasov_center_cell_path_length =
			center_cell->get_tarasov_path_length();
	tarasov_center_cell_maximum_separations_from_initial_edge =
			center_cell->get_tarasov_maximum_separation_from_initial_edge();
	tarasov_center_cell_initial_edge_num =
			center_cell->get_tarasov_initial_edge_number();
}

inline void sink_filling_algorithm::tarasov_set_field_values(coords* coords_in){
	//Don't include setting path length as these are updated on neighbor and
	//not center cells sometimes so can be handled by argument free
	//universal function
	(*tarasov_path_initial_heights)(coords_in) =
			tarasov_center_cell_path_initial_height;
	(*tarasov_maximum_separations_from_initial_edge)(coords_in) =
			tarasov_center_cell_maximum_separations_from_initial_edge;
	(*tarasov_initial_edge_nums)(coords_in) =
			tarasov_center_cell_initial_edge_num;
}

inline void sink_filling_algorithm::tarasov_get_center_cell_values_from_field(){
	tarasov_center_cell_path_initial_height =
			(*tarasov_path_initial_heights)(center_coords);
	tarasov_center_cell_path_length =
			(*tarasov_path_lengths)(center_coords);
	tarasov_center_cell_maximum_separations_from_initial_edge =
			(*tarasov_maximum_separations_from_initial_edge)(center_coords);
	tarasov_center_cell_initial_edge_num =
			(*tarasov_initial_edge_nums)(center_coords);
}

//Process the neighbors of a cell; this is key high level function that contain a considerably
//section of the whichever algorithm is being used
void sink_filling_algorithm::process_neighbors(vector<coords*>* neighbors_coords){
	//Loop through the neighbors on the supplied list
	while (!neighbors_coords->empty()) {
				nbr_coords = neighbors_coords->back();
				//If a neighbor has already been proceed simply remove it and
				//move onto the next one
				if ((*completed_cells)(nbr_coords)) {
					bool nbr_processed=false;
					if (tarasov_mod) {
						if (((*tarasov_path_initial_heights)(nbr_coords)
							< tarasov_center_cell_path_initial_height)) nbr_processed = true;
						else if (center_catchment_num ==
								(*catchment_nums)(nbr_coords)) nbr_processed = true;
						//this prevents infinite loops between two starting points of the same
						//height
						else if ((*tarasov_path_initial_heights)(nbr_coords)
								== tarasov_center_cell_path_initial_height &&
								center_catchment_num >
								(*catchment_nums)(nbr_coords)) nbr_processed = true;
						else if (landsea) {
							if 	((*landsea)(nbr_coords)) nbr_processed = true;
						}
						tarasov_reprocessing_cell = ! nbr_processed;
					} else nbr_processed = true;
					if (nbr_processed){
						#if PARALLELIZE
						if(parallelized) update_spillover_elevation
						#endif
						neighbors_coords->pop_back();
						delete nbr_coords;
						continue;
					}
				} else if (tarasov_mod) tarasov_reprocessing_cell = false;
				//Process neighbors that haven't been processed yet in accordance with the algorithm
				//selected
				process_neighbor();
				neighbors_coords->pop_back();
				//For algorithm 4 might be faster calculate this on center
				//cells instead of neighbors, looking up the river direction
				//set previously; however to maintain unity of methods process
				//it here for both algorithm 1 and 4
				if (tarasov_mod) {
					tarasov_calculate_neighbors_path_length();
					(*tarasov_path_lengths)(nbr_coords) =
							tarasov_neighbor_path_length;
					tarasov_set_field_values(nbr_coords);
				}
				if (true_sinks) {
					if (tarasov_mod) {
					//If neighbor is a true sink then it is already on the queue,
					//unless it has already been processed in which case it is not
					//on the queue
						if ((*true_sinks)(nbr_coords) && ! tarasov_reprocessing_cell) {
							delete nbr_coords;
							continue;
						}
					}
					//If neighbor is a true sink then it is already on the queue
					else if ((*true_sinks)(nbr_coords)) {
						delete nbr_coords;
						continue;
					}
				}
				push_neighbor();
				delete nbr_coords;
			}
}
#if PARALLELIZE
void sink_filling_algorithm_1::update_spillover_elevation{
	int nbr_label = (*labels)(nbr_coords);
	if (center_label != nbr_label) {
		double spillover_elevation = max((*orography)(nbr_coords),
		                                 center_orography)
		if(spillover_elevations.get_value(center_label,nbr_label) >
		   spillover_elevation){
			spillover_elevations.push(new spillover_elevation(spillover_elevation,
		                                                 		center_label,
																										  	nbr_label))
		}
	}
}
#endif

void sink_filling_algorithm_1::process_neighbor(){
	nbr_orog = (*orography)(nbr_coords);
	if (! (tarasov_mod && add_slope)) {
		if (on_slope && center_orography >= nbr_orog) return;
	}
	if(add_slope){
		if (center_orography >= nbr_orog){
			nbr_orog = center_orography + epsilon;
		}
	} else nbr_orog = max(nbr_orog,center_orography);
	#if DEBUG
		cout << " Processing neighbor: " << endl;
		cout << " coords: " << *nbr_coords;
		cout << " center_orography: " << center_orography << endl;
		cout << " new_orography:    " << nbr_orog << endl;
		cout << " old orography:    " << (*orography)(nbr_coords) << endl;
	#endif
	(*orography)(nbr_coords) = nbr_orog;
	(*completed_cells)(nbr_coords) = true;
	if (tarasov_mod) (*catchment_nums)(nbr_coords) = center_catchment_num;
	#if PARALLELIZE
	if(parallelized){
		(*labels)(nbr_coords) = center_label;
	#endif
}

void sink_filling_algorithm_4::process_neighbor(){
	nbr_orog = (*orography)(nbr_coords);
	nbr_rim_height = max(nbr_orog,center_rim_height);
	if (nbr_orog == no_data_value) set_cell_to_no_data_value(nbr_coords);
	else {
		set_index_based_rdirs(nbr_coords,center_coords);
		if(not index_based_rdirs_only) {
			static_cast<sink_filling_algorithm_4_latlon*>(this)->calculate_direction_from_neighbor_to_cell();
		}
	}
	(*completed_cells)(nbr_coords) = true;
	(*catchment_nums)(nbr_coords) = center_catchment_num;
	#if DEBUG
		cout << " Processing neighbor: " << endl;
		cout << " coords: " << *nbr_coords << endl;
	#endif
}

//Calculate the river flow direction from a neighbor to a central cell dealing with longitudinal
//wrapping
inline void sink_filling_algorithm_4_latlon::calculate_direction_from_neighbor_to_cell(){
	(*rdirs)(nbr_coords) = _grid->calculate_dir_based_rdir(nbr_coords,center_coords);
}

inline void sink_filling_algorithm_1::push_neighbor()
{
	if (tarasov_mod) {
		q.push(new cell(nbr_orog,nbr_coords->clone(),
						center_catchment_num,
						tarasov_center_cell_initial_edge_num,
						tarasov_center_cell_maximum_separations_from_initial_edge,
						tarasov_neighbor_path_length,
						tarasov_center_cell_path_initial_height));
	} else if(! add_slope) {
		if (nbr_orog <= center_orography) {
			if (! on_slope) pit_q.push(new cell(nbr_orog,nbr_coords->clone()));
			else if (! requeued) {
				q.push(new cell(center_orography,center_coords->clone()));
				requeued = true;
			}
		} else slope_q.push(new cell(nbr_orog,nbr_coords->clone()));
	} else q.push(new cell(nbr_orog,nbr_coords->clone()));
}

inline void sink_filling_algorithm_4::push_neighbor()
{
	if (tarasov_mod) {
		q.push(new cell(nbr_orog,nbr_coords->clone(),center_catchment_num,nbr_rim_height,
						tarasov_center_cell_initial_edge_num,
						tarasov_center_cell_maximum_separations_from_initial_edge,
						tarasov_neighbor_path_length,
						tarasov_center_cell_path_initial_height));
	} else q.push(new cell(nbr_orog,nbr_coords->clone(),center_catchment_num,nbr_rim_height));
}

void sink_filling_algorithm_4::setup_minima_q(field<bool>* minima_in){
	minima = minima_in;
	queue_minima = true;
	minima_q = new stack<coords*>;
}

stack<coords*>* sink_filling_algorithm_4::get_minima_q(){
	return minima_q;
}

void sink_filling_algorithm_1::tarasov_set_area_height() {
	tarasov_area_height = center_cell->get_orography();
}

void sink_filling_algorithm_4::tarasov_set_area_height() {
	tarasov_area_height = center_cell->get_rim_height();
}

inline void sink_filling_algorithm::tarasov_calculate_neighbors_path_length() {
	tarasov_neighbor_path_length = tarasov_center_cell_path_length +
			tarasov_calculate_neighbors_path_length_change(center_coords);
}

double sink_filling_algorithm_latlon::tarasov_calculate_neighbors_path_length_change(coords* center_coords_in) {
	if (_grid->non_diagonal(nbr_coords,center_coords_in))  return 1.0;
	else return SQRT_TWO;
}

double sink_filling_algorithm_icon_single_index::
			 tarasov_calculate_neighbors_path_length_change(coords* center_coords_in) {
	throw runtime_error("Path length change function not yet implemented for triangular grid");
}

void sink_filling_algorithm_4_latlon::set_cell_to_no_data_value(coords* coords_in){
	(*next_cell_lat_index)(coords_in) = no_data_value;
	(*next_cell_lon_index)(coords_in) = no_data_value;
	if(not index_based_rdirs_only)(*rdirs)(coords_in) = 0.0;
}

void sink_filling_algorithm_4_icon_single_index::set_cell_to_no_data_value(coords* coords_in){
	(*next_cell_index)(coords_in) = no_data_value;
}

void sink_filling_algorithm_4_latlon::set_cell_to_true_sink_value(coords* coords_in){
	(*next_cell_lat_index)(coords_in) = true_sink_value;
	(*next_cell_lon_index)(coords_in) = true_sink_value;
	if(not index_based_rdirs_only)(*rdirs)(coords_in) = 5.0;
}

void sink_filling_algorithm_4_icon_single_index::set_cell_to_true_sink_value(coords* coords_in){
	(*next_cell_index)(coords_in) = true_sink_value;
}

void sink_filling_algorithm_4_latlon::set_index_based_rdirs(coords* start_coords,coords* dest_coords){
	auto latlon_dest_coords = static_cast<latlon_coords*>(dest_coords);
	(*next_cell_lat_index)(start_coords) = latlon_dest_coords->get_lat();
	(*next_cell_lon_index)(start_coords) = latlon_dest_coords->get_lon();
}

void sink_filling_algorithm_4_icon_single_index::set_index_based_rdirs(coords* start_coords,coords* dest_coords){
	auto icon_single_index_dest_coords = static_cast<generic_1d_coords*>(dest_coords);
	(*next_cell_index)(start_coords) = icon_single_index_dest_coords->get_index();
}

void sink_filling_algorithm_4_latlon::set_dir_based_rdir(coords* coords_in, double direction){
	(*rdirs)(coords_in) = direction;
}

double sink_filling_algorithm_4_latlon::test_find_initial_cell_flow_direction(coords* coords_in,
																	   	   	  grid_params* grid_params_in,
																			  field<double>* orography_in,
																			  field<bool>* landsea_in,
																			  bool prefer_non_diagonals_in)
{
	nbr_coords = coords_in; _grid_params = grid_params_in;
	orography = orography_in; landsea = landsea_in;
	prefer_non_diagonal_initial_dirs = prefer_non_diagonals_in;
	index_based_rdirs_only = false;
	rdirs = new field<short>(grid_params_in);
	next_cell_lat_index = new field<int>(grid_params_in);
	next_cell_lon_index = new field<int>(grid_params_in);
	_grid = grid_factory(grid_params_in);
	find_initial_cell_flow_direction();
	auto rdir = (*rdirs)(nbr_coords);
	delete rdirs; rdirs = nullptr;
	delete next_cell_lat_index; next_cell_lat_index = nullptr;
	delete next_cell_lon_index; next_cell_lon_index = nullptr;
	delete _grid; _grid = nullptr;
	return rdir;
}

double sink_filling_algorithm_4_latlon::test_calculate_direction_from_neighbor_to_cell(coords* nbr_coords_in,
																					   coords* coords_in,
																					   grid_params* grid_params_in)

{
	center_coords = coords_in;  nbr_coords = nbr_coords_in; _grid_params = grid_params_in;
	index_based_rdirs_only = false;
	rdirs = new field<short>(grid_params_in);
	next_cell_lat_index = new field<int>(grid_params_in);
	next_cell_lon_index = new field<int>(grid_params_in);
	_grid = grid_factory(grid_params_in);
	try {
		calculate_direction_from_neighbor_to_cell();
	} catch(runtime_error &err){
		delete rdirs; rdirs = nullptr;
		delete next_cell_lat_index; next_cell_lat_index = nullptr;
		delete next_cell_lon_index; next_cell_lon_index = nullptr;
		delete _grid; _grid = nullptr;
		throw err;
	}
	auto rdir = (*rdirs)(nbr_coords);
	delete rdirs; rdirs = nullptr;
	delete next_cell_lat_index; next_cell_lat_index = nullptr;
	delete next_cell_lon_index; next_cell_lon_index = nullptr;
	delete _grid; _grid = nullptr;
	return rdir;
}
