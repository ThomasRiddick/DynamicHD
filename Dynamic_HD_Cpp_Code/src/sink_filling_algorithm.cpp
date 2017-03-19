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

sink_filling_algorithm::sink_filling_algorithm(field<double>* orography,grid_params* grid_params_in,
											   field<bool>* completed_cells,bool* landsea_in,
											   bool set_ls_as_no_data_flag,bool* true_sinks_in) :
											   _grid_params(grid_params_in),
											   orography(orography),
											   completed_cells(completed_cells),
											   set_ls_as_no_data_flag(set_ls_as_no_data_flag)

{
	_grid = grid_factory(_grid_params);
	landsea = landsea_in ? new field<bool>(landsea_in,_grid_params): nullptr;
	true_sinks = true_sinks_in ? new field<bool>(true_sinks_in,_grid_params): nullptr;
}

sink_filling_algorithm::~sink_filling_algorithm() { delete orography; delete completed_cells;
													delete true_sinks; }

sink_filling_algorithm_1::sink_filling_algorithm_1(field<double>* orography,grid_params* grid_params_in,
												   field<bool>* completed_cells,bool* landsea_in,
												   bool set_ls_as_no_data_flag,bool add_slope,
												   double epsilon, bool* true_sinks_in)
	: sink_filling_algorithm(orography,grid_params_in,completed_cells,landsea_in,set_ls_as_no_data_flag,
							 true_sinks_in), add_slope(add_slope), epsilon(epsilon) {}

sink_filling_algorithm_1_latlon::sink_filling_algorithm_1_latlon(field<double>* orography,grid_params* grid_params_in,
		   	   	   	   	   	   									 field<bool>* completed_cells,bool* landsea_in,
																 bool set_ls_as_no_data_flag,bool add_slope,
																 double epsilon,bool* true_sinks_in)
	: sink_filling_algorithm(orography,grid_params_in,completed_cells,landsea_in,set_ls_as_no_data_flag,
							 true_sinks_in),
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
			                 true_sinks_in),
	  catchment_nums(catchment_nums_in),
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
																 field<double>* rdirs_in)
	: sink_filling_algorithm(orography,grid_params_in,completed_cells,landsea_in,set_ls_as_no_data_flag,
            true_sinks_in),
	  sink_filling_algorithm_4(orography,grid_params_in,completed_cells,landsea_in,set_ls_as_no_data_flag,
			   	   	   	       catchment_nums_in,prefer_non_diagonal_initial_dirs,true_sinks_in),
	  rdirs(rdirs_in), next_cell_lat_index(next_cell_lat_index_in), next_cell_lon_index(next_cell_lon_index_in)
	{
		index_based_rdirs_only=index_based_rdirs_only_in;
		auto* grid_latlon = dynamic_cast<latlon_grid*>(_grid);
		nlat = grid_latlon->get_nlat(); nlon = grid_latlon->get_nlon();
	}

void sink_filling_algorithm::setup_flags(bool set_ls_as_no_data_flag_in,bool debug_in)
{
	set_ls_as_no_data_flag = set_ls_as_no_data_flag_in;
	debug = debug_in;
}

void sink_filling_algorithm_1::setup_flags(bool set_ls_as_no_data_flag_in,bool debug_in,
										   bool add_slope_in, double epsilon_in)
{
	sink_filling_algorithm::setup_flags(set_ls_as_no_data_flag_in,debug_in);
	add_slope = add_slope_in; epsilon = epsilon_in;
}

void sink_filling_algorithm_4::setup_flags(bool set_ls_as_no_data_flag_in,
			     	 	 	 	 	 	   bool prefer_non_diagonal_initial_dirs_in,
										   bool debug_in)
{
	sink_filling_algorithm::setup_flags(set_ls_as_no_data_flag_in,debug_in);
	prefer_non_diagonal_initial_dirs = prefer_non_diagonal_initial_dirs_in;
}

void sink_filling_algorithm_4_latlon::setup_flags(bool set_ls_as_no_data_flag_in,
			     	 	 	 	 	 	   bool prefer_non_diagonal_initial_dirs_in,
										   bool debug_in, bool index_based_rdirs_only_in)
{
	sink_filling_algorithm_4::setup_flags(set_ls_as_no_data_flag_in,prefer_non_diagonal_initial_dirs_in,
			   	   	   	   	   	   	   	  debug_in);
	prefer_non_diagonal_initial_dirs = prefer_non_diagonal_initial_dirs_in;
	index_based_rdirs_only = index_based_rdirs_only_in;
}

void sink_filling_algorithm::setup_fields(double* orography_in, bool* landsea_in,
									      bool* true_sinks_in,grid_params* grid_params)
{
	_grid_params = grid_params;
	_grid = grid_factory(_grid_params);
	orography = new field<double>(orography_in,grid_params);
	completed_cells = new field<bool>(grid_params);  //Cells that have already been processed
	completed_cells->set_all(false);
	landsea = landsea_in ? new field<bool>(landsea_in,grid_params): nullptr;
	true_sinks = true_sinks_in ? new field<bool>(true_sinks_in,grid_params): nullptr;
}

void sink_filling_algorithm_4::setup_fields(double* orography_in, bool* landsea_in,
								            bool* true_sinks_in, grid_params* grid_params_in,
											int* catchment_nums_in)
{
	sink_filling_algorithm::setup_fields(orography_in,landsea_in,true_sinks_in,grid_params_in);
	catchment_nums = new field<int>(catchment_nums_in,grid_params_in);
}

void sink_filling_algorithm_4_latlon::setup_fields(double* orography_in, bool* landsea_in,
								            	   bool* true_sinks_in, int * next_cell_lat_index_in,
												   int * next_cell_lon_index_in,
												   grid_params* grid_params_in,
												   double* rdirs_in, int* catchment_nums_in)
{
	sink_filling_algorithm_4::setup_fields(orography_in,landsea_in,true_sinks_in,grid_params_in,
										   catchment_nums_in);
	rdirs = new field<double>(rdirs_in,grid_params_in);
	next_cell_lat_index = new field<int>(next_cell_lat_index_in,grid_params_in);
	next_cell_lon_index = new field<int>(next_cell_lon_index_in,grid_params_in);
	auto* grid_latlon = dynamic_cast<latlon_grid*>(_grid);
	nlat = grid_latlon->get_nlat(); nlon = grid_latlon->get_nlon();
}

void sink_filling_algorithm_1_latlon::setup_fields(double* orography_in, bool* landsea_in,
									      	  	   bool* true_sinks_in,grid_params* grid_params){
	sink_filling_algorithm::setup_fields(orography_in,landsea_in,true_sinks_in,grid_params);
	auto* grid_latlon = dynamic_cast<latlon_grid*>(_grid);
	nlat = grid_latlon->get_nlat(); nlon = grid_latlon->get_nlon();
}

//Implementation as per algorithm 1 of the paper cited in the preamble to this file
void sink_filling_algorithm::fill_sinks()
{
	//Reset the method variable of the base class
	method = get_method();
	//Set out-flow points
	cout << "Setting out-flow points" << endl;
	add_edge_cells_to_q();
	if (true_sinks) add_true_sinks_to_q();
	//Algorithm is now finished with landsea
	delete landsea;
	//Loop over non out-flow points searching and filling sinks (or assigning flow directions);
	//this is the main section of the program
	cout << "Starting to loop over non out-flow points searching for sinks" << endl;
	while (!q.empty()) {
		center_cell = q.top();
		q.pop();
		center_coords = center_cell->get_cell_coords();
		if (debug) {
			cout << "q length: " << q.size() << endl;
			cout << "Processing central cell:" << endl;
			if (auto center_coords_latlon = dynamic_cast<latlon_coords*>(center_coords)){
				cout << "lat: " << center_coords_latlon->get_lat() << " lon: "<< center_coords_latlon->get_lon() << endl;
			}
		}
		process_center_cell();
		if (tarasov_mod) {
			tarasov_update_maximum_separation_from_initial_edge();
			if (tarasov_is_shortest_permitted_path()) {
				tarasov_set_area_height();
				delete center_cell;
				break;
			}
		}
		auto neighbors_coords = orography->get_neighbors_coords(center_coords,method);
		process_neighbors(neighbors_coords);
		delete neighbors_coords;
		delete center_cell;
	}
}

//Handles the initial setup of the queue; there are two versions for each method (of which
//there are also currently two to give a total of four variants): one for if a land sea
//mask is supplied; one for if no land sea mask is supplied. The latter assumes outflow
//points around the edges
void sink_filling_algorithm::add_edge_cells_to_q()
{
	//If a land sea mask is supplied
	if (landsea) add_landsea_edge_cells_to_q();
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
					push_land_sea_neighbor();
				}
				neighbors_coords->pop_back();
				delete nbr_coords;
			}
			delete neighbors_coords;
		}
	};
	_grid->for_all(add_edge_cell_to_q_func);
}

bool sink_filling_algorithm::tarasov_is_shortest_permitted_path(){
	if (center_cell->get_tarasov_path_length() < tarasov_min_path_length) return false;
	else if (not tarasov_same_edge_criteria_met()) return false;
	else return true;
}

bool sink_filling_algorithm::tarasov_same_edge_criteria_met(){
	if (_grid->check_if_cell_is_on_given_edge_number(center_coords,
			center_cell->get_tarasov_initial_edge_number())) {
		if (not tarasov_include_corners_in_same_edge_criteria &&
			_grid->is_corner_cell(center_coords)) return true;
		else if(center_cell->get_tarasov_maximum_separation_from_initial_edge() >
					tarasov_seperation_threshold_for_returning_to_same_edge) return true;
		else return false;
	}
	else return true;
}

void sink_filling_algorithm::tarasov_update_maximum_separation_from_initial_edge(){
	int separation_from_initial_edge = _grid->get_separation_from_initial_edge(center_coords,
			center_cell->get_tarasov_initial_edge_number());
	if(center_cell->get_tarasov_maximum_separation_from_initial_edge() <
			separation_from_initial_edge){
		center_cell->set_tarasov_maximum_separation_from_initial_edge(separation_from_initial_edge);
	}
}

void sink_filling_algorithm_latlon::add_geometric_edge_cells_to_q(){
	for (auto i = 0; i < nlat; i++) {
		push_vertical_edge(i);
		(*completed_cells)(new latlon_coords(i,0)) = true;
		(*completed_cells)(new latlon_coords(i,nlon-1)) = true;
	}
	for (auto j = 1; j < nlon-1; j++){
		push_horizontal_edge(j);
		(*completed_cells)(new latlon_coords(0,j)) = true;
		(*completed_cells)(new latlon_coords(nlat-1,j)) = true;
	}
}

void sink_filling_algorithm::add_true_sinks_to_q()
{
	_grid->for_all([&](coords* coords_in){
			if ((*true_sinks)(coords_in)){
				if (landsea){
					if ((*landsea)(coords_in)) return;
				}
				//ignore sinks next to landsea points... how such a situation could possible occur
				//and therefore the correct hydrology for it is not clear
				if(!(*completed_cells)(coords_in)) push_true_sink(coords_in);
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
	q.push(new cell((*orography)(nbr_coords),nbr_coords->clone()));
}

inline void sink_filling_algorithm_4::push_land_sea_neighbor()
{
	int new_catchment_num = q.get_next_k_value();
	q.push(new cell((*orography)(nbr_coords),nbr_coords->clone(),new_catchment_num,
		   (*orography)(nbr_coords)));
	find_initial_cell_flow_direction();
	(*catchment_nums)(nbr_coords) = new_catchment_num;
}

//Require a minimum of 1 in-line function in sink_filling_algorithm_1_latlon
void sink_filling_algorithm_1_latlon::push_vertical_edge(int i)
{
	q.push(new cell((*orography)(new latlon_coords(i,0)),new latlon_coords(i,0)));
	q.push(new cell((*orography)(new latlon_coords(i,nlon-1)),new latlon_coords(i,nlon-1)));
}

inline void sink_filling_algorithm_4_latlon::push_vertical_edge(int i)
{
	int new_catchment_num = q.get_next_k_value();
	q.push(new cell((*orography)(new latlon_coords(i,0)),new latlon_coords(i,0),new_catchment_num,
					(*orography)(new latlon_coords(i,0))));
	(*catchment_nums)(new latlon_coords(i,0)) = new_catchment_num;
	new_catchment_num = q.get_next_k_value();
	q.push(new cell((*orography)(new latlon_coords(i,nlon-1)),new latlon_coords(i,nlon-1),new_catchment_num,
					(*orography)(new latlon_coords(i,nlon-1))));
	(*catchment_nums)(new latlon_coords(i,nlon-1)) = new_catchment_num;
	(*rdirs)(new latlon_coords(i,0)) = 4;
	(*rdirs)(new latlon_coords(i,nlon-1)) = 6;
}

inline void sink_filling_algorithm_1_latlon::push_horizontal_edge(int j)
{
	q.push(new cell((*orography)(new latlon_coords(0,j)),new latlon_coords(0,j)));
	q.push(new cell((*orography)(new latlon_coords(nlat-1,j)),new latlon_coords(nlat-1,j)));
}

inline void sink_filling_algorithm_4_latlon::push_horizontal_edge(int j)
{
	int new_catchment_num = q.get_next_k_value();
	q.push(new cell((*orography)(new latlon_coords(0,j)),new latlon_coords(0,j),new_catchment_num,
					(*orography)(new latlon_coords(0,j))));
	(*catchment_nums)(new latlon_coords(0,j)) = new_catchment_num;
	new_catchment_num = q.get_next_k_value();
	q.push(new cell((*orography)(new latlon_coords(nlat-1,j)),new latlon_coords(nlat-1,j),new_catchment_num,
					(*orography)(new latlon_coords(nlat-1,j))));
	(*catchment_nums)(new latlon_coords(nlat-1,j)) = new_catchment_num;
	(*rdirs)(new latlon_coords(0,j)) = 8;
	(*rdirs)(new latlon_coords(nlat-1,j)) = 2;
}

inline void sink_filling_algorithm_1::push_true_sink(coords* coords_in)
{
	q.push_true_sink(new cell((*orography)(coords_in),coords_in->clone()));
}

inline void sink_filling_algorithm_4::push_true_sink(coords* coords_in)
{
	int new_catchment_num = q.get_next_k_value();
	q.push_true_sink(new cell((*orography)(coords_in),coords_in->clone(),new_catchment_num,
			(*orography)(coords_in)));
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
		if (_grid->outside_limits(coords_in)) return;
		//deal with longitudinal wrapping
		auto coords_prime = _grid->wrapped_coords(coords_in);
		if ((*landsea)(coords_prime)){
			if (min_height > (*orography)(coords_prime)){
				min_height = (*orography)(coords_prime);
				set_index_based_rdirs(nbr_coords,coords_prime);
				destination_coords = coords_prime;
				//Note the index here is correctly coords_in and not coords_prime!
				if (not index_based_rdirs_only) direction = _grid->calculate_dir_based_rdir(nbr_coords,coords_in);
			} else if(min_height == (*orography)(coords_prime) &&
					  prefer_non_diagonal_initial_dirs &&
					  //This block favors non diagonals if the appropriate flag is set
					  _grid->non_diagonal(nbr_coords,coords_prime)) {
				destination_coords = coords_prime;
				//make this a part of latlon_grid then cast to that and give nbr_coors
				//and coords_in to that to generate this number, also below
				//Also require setter function below and a is non-diagonal
				//check function
				if(not index_based_rdirs_only) direction = _grid->calculate_dir_based_rdir(nbr_coords,coords_in);
			}
		}
	};
	_grid->for_all_nbrs(nbr_coords,find_init_rdir_func);
	set_index_based_rdirs(nbr_coords,destination_coords);
	if (not index_based_rdirs_only) static_cast<sink_filling_algorithm_4_latlon*>(this)->set_dir_based_rdir(nbr_coords,direction);
}

inline void sink_filling_algorithm_1::process_true_sink_center_cell() {
	(*completed_cells)(center_coords) = true;
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
	} else {
		//If this a former true sink that now flow to a neighbor
		//then value in cell object is not the correct one (and
		//it couldn't be corrected as this would require searching
		//the queue for this cell in a previous step ) so have to
		//look it up in array instead
		center_catchment_num = (*catchment_nums)(center_coords);
		//Same for rim height
		center_rim_height = cell_orog;
	}
}

inline void sink_filling_algorithm_1::process_center_cell() {
	center_orography = center_cell->get_orography();
	if (true_sinks) process_true_sink_center_cell();
}

inline void sink_filling_algorithm_4::process_center_cell()
{
	if (true_sinks) process_true_sink_center_cell();
	else {
		center_rim_height = center_cell->get_rim_height();
		center_catchment_num = center_cell->get_catchment_num();
	}
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
					neighbors_coords->pop_back();
					delete nbr_coords;
					continue;
				}
				//Process neighbors that haven't been processed yet in accordance with the algorithm
				//selected
				process_neighbor();
				neighbors_coords->pop_back();
				if (true_sinks) {
					//If neighbor is a true sink then it is already on the queue
					if ((*true_sinks)(nbr_coords)) {
						delete nbr_coords;
						continue;
					}
				}
				//For algorithm 4 might be faster calculate this on center
				//cells instead of neighbors, looking up the river direction
				//set previously; however to maintain unity of methods process
				//it here for both algorithm 1 and 4
				if (tarasov_mod) tarasov_calculate_neighbors_path_length();
				push_neighbor();
				delete nbr_coords;
			}
}

void sink_filling_algorithm_1::process_neighbor(){
	if(add_slope){
		nbr_orog = (*orography)(nbr_coords);
		if (center_orography >= nbr_orog){
			nbr_orog = center_orography + epsilon;
		}
	} else nbr_orog = max((*orography)(nbr_coords),center_orography);
	if (debug) {
		cout << " Processing neighbor: " << endl;
		if (auto nbr_coords_latlon = dynamic_cast<latlon_coords*>(nbr_coords)){
			cout << " lat: " <<  nbr_coords_latlon->get_lat() << " lon: "<< nbr_coords_latlon->get_lon() << endl;
		}
		cout << " center_orography: " << center_orography << endl;
		cout << " new_orography:    " << nbr_orog << endl;
		cout << " old orography:    " << (*orography)(nbr_coords) << endl;
	}
	(*orography)(nbr_coords) = nbr_orog;
	(*completed_cells)(nbr_coords) = true;
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
	if (debug){
		cout << " Processing neighbor: " << endl;
		if (auto nbr_coords_latlon = dynamic_cast<latlon_coords*>(nbr_coords)){
			cout << " lat: " <<  nbr_coords_latlon->get_lat() << " lon: "<< nbr_coords_latlon->get_lon() << endl;
		}
	}
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
						center_cell->get_tarasov_initial_edge_number(),
						center_cell->get_tarasov_maximum_separation_from_initial_edge(),
						tarasov_neighbor_path_length));
	} else q.push(new cell(nbr_orog,nbr_coords->clone()));
}

inline void sink_filling_algorithm_4::push_neighbor()
{
	if (tarasov_mod) {
		q.push(new cell(nbr_orog,nbr_coords->clone(),center_catchment_num,nbr_rim_height,
						center_cell->get_tarasov_initial_edge_number(),
						center_cell->get_tarasov_maximum_separation_from_initial_edge(),
						tarasov_neighbor_path_length));
	} else q.push(new cell(nbr_orog,nbr_coords->clone(),center_catchment_num,nbr_rim_height));
}

void sink_filling_algorithm_1::tarasov_set_area_height() {
	tarasov_area_height = center_cell->get_orography();
}

void sink_filling_algorithm_4::tarasov_set_area_height() {
	tarasov_area_height = center_cell->get_rim_height();
}

void sink_filling_algorithm::tarasov_calculate_neighbors_path_length() {
	tarasov_neighbor_path_length = center_cell->get_tarasov_path_length();
	if (_grid->non_diagonal(nbr_coords,center_coords)) tarasov_neighbor_path_length += SQRT_TWO;
	else tarasov_neighbor_path_length += 1.0;
}

void sink_filling_algorithm_4_latlon::set_cell_to_no_data_value(coords* coords_in){
	(*next_cell_lat_index)(coords_in) = no_data_value;
	(*next_cell_lon_index)(coords_in) = no_data_value;
	if(not index_based_rdirs_only)(*rdirs)(coords_in) = 0.0;
}

void sink_filling_algorithm_4_latlon::set_cell_to_true_sink_value(coords* coords_in){
	(*next_cell_lat_index)(coords_in) = true_sink_value;
	(*next_cell_lon_index)(coords_in) = true_sink_value;
	if(not index_based_rdirs_only)(*rdirs)(coords_in) = 5.0;
}

void sink_filling_algorithm_4_latlon::set_index_based_rdirs(coords* start_coords,coords* dest_coords){
	auto latlon_dest_coords = static_cast<latlon_coords*>(dest_coords);
	(*next_cell_lat_index)(start_coords) = latlon_dest_coords->get_lat();
	(*next_cell_lon_index)(start_coords) = latlon_dest_coords->get_lon();
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
	rdirs = new field<double>(grid_params_in);
	next_cell_lat_index = new field<int>(grid_params_in);
	next_cell_lon_index = new field<int>(grid_params_in);
	_grid = grid_factory(grid_params_in);
	find_initial_cell_flow_direction();
	return (*rdirs)(nbr_coords);
}

double sink_filling_algorithm_4_latlon::test_calculate_direction_from_neighbor_to_cell(coords* nbr_coords_in,
																					   coords* coords_in,
																					   grid_params* grid_params_in)

{
	center_coords = coords_in;  nbr_coords = nbr_coords_in; _grid_params = grid_params_in;
	index_based_rdirs_only = false;
	rdirs = new field<double>(grid_params_in);
	next_cell_lat_index = new field<int>(grid_params_in);
	next_cell_lon_index = new field<int>(grid_params_in);
	_grid = grid_factory(grid_params_in);
	calculate_direction_from_neighbor_to_cell();
	return (*rdirs)(nbr_coords);
}
