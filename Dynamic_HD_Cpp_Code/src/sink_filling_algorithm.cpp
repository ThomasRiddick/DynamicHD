/*
 * sink_filling_algorithm.cpp
 *
 *  Created on: May 20, 2016
 *      Author: thomasriddick
 */
#include <iostream>
#include <limits>
#include <algorithm>
#include "common_types.hpp"
#include "sink_filling_algorithm.hpp"

using namespace std;

/* IMPORTANT!
 * Note that a description of each functions purpose is provided with the function declaration in
 * the header file not with the function definition. The definition merely provides discussion of
 * details of the implementation.
 */

sink_filling_algorithm::sink_filling_algorithm(field<double>* orography,int nlat, int nlon,
											   field<bool>* completed_cells,bool* landsea_in,
											   bool set_ls_as_no_data_flag,bool* true_sinks_in) :
											   orography(orography),
											   completed_cells(completed_cells),
											   set_ls_as_no_data_flag(set_ls_as_no_data_flag),
											   nlat(nlat),nlon(nlon)
{
	landsea = landsea_in ? new field<bool>(landsea_in,nlat,nlon): nullptr;
	true_sinks = true_sinks_in ? new field<bool>(true_sinks_in,nlat,nlon): nullptr;
}

sink_filling_algorithm::~sink_filling_algorithm() { delete orography; delete completed_cells;
													delete true_sinks; }

sink_filling_algorithm_1::sink_filling_algorithm_1(field<double>* orography,int nlat, int nlon,
												   field<bool>* completed_cells,bool* landsea_in,
												   bool set_ls_as_no_data_flag,bool* true_sinks_in)
	: sink_filling_algorithm(orography,nlat,nlon,completed_cells,landsea_in,set_ls_as_no_data_flag,
							 true_sinks_in) {}

sink_filling_algorithm_4::sink_filling_algorithm_4(field<double>* orography,int nlat,int nlon,
											       field<double>* rdirs, field<bool>* completed_cells,
												   bool* landsea_in, bool set_ls_as_no_data_flag,
												   field<int>* catchment_nums_in,
												   bool prefer_non_diagonal_initial_dirs,
												   bool* true_sinks_in)
	: sink_filling_algorithm(orography,nlat,nlon,completed_cells,landsea_in,set_ls_as_no_data_flag,
			                 true_sinks_in),
	  rdirs(rdirs), catchment_nums(catchment_nums_in),
	  prefer_non_diagonal_initial_dirs(prefer_non_diagonal_initial_dirs) {}

void sink_filling_algorithm::setup_flags(bool set_ls_as_no_data_flag_in,bool debug_in)
{
	set_ls_as_no_data_flag = set_ls_as_no_data_flag_in;
	debug = debug_in;
}

void sink_filling_algorithm_4::setup_flags(bool set_ls_as_no_data_flag_in,
			     	 	 	 	 	 	   bool prefer_non_diagonal_initial_dirs_in,
										   bool debug_in)
{
	sink_filling_algorithm::setup_flags(set_ls_as_no_data_flag_in,debug_in);
	prefer_non_diagonal_initial_dirs = prefer_non_diagonal_initial_dirs_in;
}

void sink_filling_algorithm::setup_fields(double* orography_in, bool* landsea_in,
									      bool* true_sinks_in,int nlat_in,int nlon_in)
{
	nlat = nlat_in;
	nlon = nlon_in;
	orography = new field<double>(orography_in,nlat,nlon);
	completed_cells = new field<bool>(nlat,nlon);  //Cells that have already been processed
	completed_cells->set_all(false);
	landsea = landsea_in ? new field<bool>(landsea_in,nlat,nlon): nullptr;
	true_sinks = true_sinks_in ? new field<bool>(true_sinks_in,nlat,nlon): nullptr;
}

void sink_filling_algorithm_4::setup_fields(double* orography_in, bool* landsea_in,
								            bool* true_sinks_in, int nlat_in, int nlon_in,
											double* rdirs_in, int* catchment_nums_in)
{
	sink_filling_algorithm::setup_fields(orography_in,landsea_in,true_sinks_in,nlat_in,nlon_in);
	rdirs = new field<double>(rdirs_in,nlat,nlon);
	catchment_nums = new field<int>(catchment_nums_in,nlat,nlon);
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
			cout << "lat: " << center_coords.first << " lon: "<< center_coords.second << endl;
		}
		process_center_cell();
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
	if (landsea) {
		for (auto i = 0; i < nlat; i++){
			for (auto j = 0; j < nlon; j++){
				if((*landsea)(i,j)){
					//set the land sea mask itself to no data if that option has been selected
					if (set_ls_as_no_data_flag) set_ls_as_no_data(i,j);
					//set all points in the land sea mask as having been processed
					(*completed_cells)(i,j) = true;
					//Calculate and the process the neighbors of every landsea point and
					//add them to the queue (if they aren't already in the queue or themselves
					//land sea points
					//Note the get_neighbors_coords function is always called here using
					//method 1 technique as this is always appropriate (regardless of which
					//method is being run) for the set-up phase
					auto neighbors_coords = orography->get_neighbors_coords(i,j);
					while (!neighbors_coords->empty()){
						auto nbr_coords = neighbors_coords->back();
						nbr_lat = nbr_coords->first;
						nbr_lon = nbr_coords->second;
						//If neither a land sea point nor a cell already in the queue
						//then add this cell to the queue (and possibly assign it
						//a flow direction if this is algorithm 4)
						if (!((*landsea)(nbr_lat,nbr_lon) ||
							(*completed_cells)(nbr_lat,nbr_lon))) {
							(*completed_cells)(nbr_lat,nbr_lon) = true;
							push_land_sea_neighbor();
						}
						neighbors_coords->pop_back();
						delete nbr_coords;
					}
					delete neighbors_coords;
				}
			}
		}
	//No land sea mask supplied; use edges as out flow points; also assign
    //them flow direction if using algorithm 4
	} else {
		for (auto i = 0; i < nlat; i++) {
			push_vertical_edge(i);
			(*completed_cells)(i,0) = true;
			(*completed_cells)(i,nlon-1) = true;
		}
		for (auto j = 1; j < nlon-1; j++){
			push_horizontal_edge(j);
			(*completed_cells)(0,j) = true;
			(*completed_cells)(nlat-1,j) = true;
		} }
}

void sink_filling_algorithm::add_true_sinks_to_q()
{
	for (auto i = 0; i < nlat; i++){
		for (auto j = 0; j < nlon; j++){
			if ((*true_sinks)(i,j)){
				if (landsea){
					if ((*landsea)(i,j)) continue;
				}
				//ignore sinks next to landsea points... how such a situation could possible occur
				//and therefore the correct hydrology for it is not clear
				if(!(*completed_cells)(i,j)) push_true_sink(i,j);
			}
		}
	}
}

inline void sink_filling_algorithm_1::set_ls_as_no_data(int i, int j)
{
	(*orography)(i,j) = no_data_value;
}

inline void sink_filling_algorithm_4::set_ls_as_no_data(int i, int j)
{
	cout << "Setting sea points as no data is incompatible with method 4; ignoring flag" << endl;
}

inline void sink_filling_algorithm_1::push_land_sea_neighbor()
{
	q.push(new cell((*orography)(nbr_lat,nbr_lon),nbr_lat,nbr_lon));
}

inline void sink_filling_algorithm_4::push_land_sea_neighbor()
{
	int new_catchment_num = q.get_next_k_value();
	q.push(new cell((*orography)(nbr_lat,nbr_lon),nbr_lat,nbr_lon,new_catchment_num));
	(*rdirs)(nbr_lat,nbr_lon) =
			find_initial_cell_flow_direction();
	(*catchment_nums)(nbr_lat,nbr_lon) = new_catchment_num;
}

inline void sink_filling_algorithm_1::push_vertical_edge(int i)
{
	q.push(new cell((*orography)(i,0),i,0));
	q.push(new cell((*orography)(i,nlon-1),i,nlon-1));
}

inline void sink_filling_algorithm_4::push_vertical_edge(int i)
{
	int new_catchment_num = q.get_next_k_value();
	q.push(new cell((*orography)(i,0),i,0,new_catchment_num));
	(*catchment_nums)(i,0) = new_catchment_num;
	new_catchment_num = q.get_next_k_value();
	q.push(new cell((*orography)(i,nlon-1),i,nlon-1,new_catchment_num));
	(*catchment_nums)(i,nlon-1) = new_catchment_num;
	(*rdirs)(i,0) = 4;
	(*rdirs)(i,nlon-1) = 6;
}

inline void sink_filling_algorithm_1::push_horizontal_edge(int j)
{
	q.push(new cell((*orography)(0,j),0,j));
	q.push(new cell((*orography)(nlat-1,j),nlat-1,j));
}

inline void sink_filling_algorithm_4::push_horizontal_edge(int j)
{
	int new_catchment_num = q.get_next_k_value();
	q.push(new cell((*orography)(0,j),0,j,new_catchment_num));
	(*catchment_nums)(0,j) = new_catchment_num;
	new_catchment_num = q.get_next_k_value();
	q.push(new cell((*orography)(nlat-1,j),nlat-1,j,new_catchment_num));
	(*catchment_nums)(nlat-1,j) = new_catchment_num;
	(*rdirs)(0,j) = 8;
	(*rdirs)(nlat-1,j) = 2;
}

inline void sink_filling_algorithm_1::push_true_sink(int i, int j)
{
	q.push_true_sink(new cell((*orography)(i,j),i,j));
}

inline void sink_filling_algorithm_4::push_true_sink(int i, int j)
{
	int new_catchment_num = q.get_next_k_value();
	q.push_true_sink(new cell((*orography)(i,j),i,j,new_catchment_num));
}

//Calculates the flow direction for cell add to the queue as out flow points for algorithm 4;
//dealing with any wrapping. The out flow is the calculated being towards the lowest orography
//neighbor that is sea (in the land sea mask). In the case of the neighbors orography values
//being tied the first minimum is used unless a flag is set and then the last non-diagonal neighbor
//is used (and the first minimum again if there are no non diagonal neighbors). This function deals
//with longitudinal wrapping.
double sink_filling_algorithm_4::find_initial_cell_flow_direction(){
	double min_height = numeric_limits<double>::max();
	double directions[9] = {7.0,8.0,9.0,4.0,5.0,6.0,1.0,2.0,3.0};
	double direction  = 5.0;
	//Default to 5 for land sea points
	if ((*landsea)(nbr_lat,nbr_lon) == true) return direction;
	for (auto i = nbr_lat-1; i <= nbr_lat+1; i++){
		for (auto j = nbr_lon-1; j <= nbr_lon+1; j++){
			if (i == nbr_lat && j == nbr_lon) continue;
			if (i < 0 || i >= nlat) continue;
			int jprime;
			//deal with longitudinal wrapping
			if (j < 0 ) jprime = nlon + j;
			else if (j >= nlon) jprime = j - nlon;
			else jprime = j;
			if ((*landsea)(i,jprime)){
				if (min_height > (*orography)(i,jprime)){
					min_height = (*orography)(i,jprime);
					//Note the index here is correctly j and not jprime!
					direction = directions[3*(i+1-nbr_lat) + (j+1-nbr_lon)];
				} else if(min_height == (*orography)(i,jprime) &&
						  prefer_non_diagonal_initial_dirs &&
						  //This block favors non diagonals if the appropriate flag is set
						  (i == nbr_lat || j == nbr_lon)) {
					direction = directions[3*(i+1-nbr_lat) + (j+1-nbr_lon)];
				}
			}
		}
	}
	return direction;
}

inline void sink_filling_algorithm_1::process_true_sink_center_cell() {
	lat = center_coords.first;
	lon = center_coords.second;
	(*completed_cells)(lat,lon) = true;
}

inline void sink_filling_algorithm_4::process_true_sink_center_cell(){
	if (!(*completed_cells)(lat,lon)){
		double cell_orog = (*orography)(lat,lon);
		center_catchment_num = center_cell->get_catchment_num();
		if (cell_orog == no_data_value) (*rdirs)(lat,lon) = 0.0;
		else (*rdirs)(lat,lon) = 5.0;
		(*completed_cells)(lat,lon) = true;
		(*catchment_nums)(lat,lon) = center_catchment_num;
	} else {
		//If this a former true sink that now flow to a neighbor
		//then value in cell object is not the correct one (and
		//it couldn't be corrected as this would require searching
		//the queue for this cell in a previous step ) so have to
		//look it up in array instead
		center_catchment_num = (*catchment_nums)(lat,lon);
	}
}

inline void sink_filling_algorithm_1::process_center_cell() {
	center_orography = center_cell->get_orography();
	if (true_sinks) process_true_sink_center_cell();
}

inline void sink_filling_algorithm_4::process_center_cell()
{
	lat = center_coords.first;
	lon = center_coords.second;
	if (true_sinks) process_true_sink_center_cell();
	else center_catchment_num = center_cell->get_catchment_num();
}

//Process the neighbors of a cell; this is key high level function that contain a considerably
//section of the whichever algorithm is being used
void sink_filling_algorithm::process_neighbors(vector<integerpair*>* neighbors_coords){
	//Loop through the neighbors on the supplied list
	while (!neighbors_coords->empty()) {
				auto nbr_coords = neighbors_coords->back();
				nbr_lat = nbr_coords->first;
				nbr_lon = nbr_coords->second;
				//If a neighbor has already been proceed simply remove it and
				//move onto the next one
				if ((*completed_cells)(nbr_lat,nbr_lon)) {
					neighbors_coords->pop_back();
					delete nbr_coords;
					continue;
				}
				//Process neighbors that haven't been processed yet in accordance with the algorithm
				//selected
				process_neighbor();
				neighbors_coords->pop_back();
				delete nbr_coords;
				if (true_sinks) {
					//If neighbor is a true sink then it is already on the queue
					if ((*true_sinks)(nbr_lat,nbr_lon)) {
						continue;
					}
				}
				push_neighbor();
			}
}

void sink_filling_algorithm_1::process_neighbor(){
	nbr_orog = max((*orography)(nbr_lat,nbr_lon),center_orography);
	if (debug) {
		cout << " Processing neighbor: " << endl;
		cout << " lat: " << nbr_lat << " lon: "<< nbr_lon << endl;
		cout << " center_orography: " << center_orography << endl;
		cout << " new_orography:    " << nbr_orog << endl;
		cout << " old orography:    " << (*orography)(nbr_lat,nbr_lon) << endl;
	}
	(*orography)(nbr_lat,nbr_lon) = nbr_orog;
	(*completed_cells)(nbr_lat,nbr_lon) = true;
}

void sink_filling_algorithm_4::process_neighbor(){
	nbr_orog = (*orography)(nbr_lat,nbr_lon);
	if (nbr_orog == no_data_value) (*rdirs)(nbr_lat,nbr_lon) = 0.0;
	else (*rdirs)(nbr_lat,nbr_lon) = calculate_direction_from_neighbor_to_cell(nbr_lat,nbr_lon);
	(*completed_cells)(nbr_lat,nbr_lon) = true;
	(*catchment_nums)(nbr_lat,nbr_lon) = center_catchment_num;
	if (debug){
		cout << " Processing neighbor: " << endl;
		cout << " lat: " << nbr_lat << " lon: "<< nbr_lon << endl;
	}
}

//Calculate the river flow direction from a neighbor to a central cell dealing with longitudinal
//wrapping and defaulting to 5 if the neighbor and the central cells are not really neighbors
double sink_filling_algorithm_4::calculate_direction_from_neighbor_to_cell(int nbr_lat_loc,int nbr_lon_loc){
	double directions[9] = {3.0,2.0,1.0,6.0,5.0,4.0,9.0,8.0,7.0};
	//deal with wrapping longitude
	if (lon == 0 && nbr_lon_loc == nlon - 1) nbr_lon_loc = -1;
	if (nbr_lon_loc == 0 && lon == nlon - 1) nbr_lon_loc = nlon;
	//deal with the case of the neighbor and cell not actually being neighbors
	if ((abs(nbr_lat_loc - lat) > 1) || (abs(nbr_lon_loc - lon) > 1)) return 5.0;
	return directions[3*(nbr_lat_loc - lat) + (nbr_lon_loc - lon) + 4];
}


inline void sink_filling_algorithm_1::push_neighbor()
{
	q.push(new cell(nbr_orog,nbr_lat,nbr_lon));
}

inline void sink_filling_algorithm_4::push_neighbor()
{
	q.push(new cell(nbr_orog,nbr_lat,nbr_lon,center_catchment_num));
}

double sink_filling_algorithm_4::test_find_initial_cell_flow_direction(int lat_in,int lon_in,
																	   int nlat_in,int nlon_in,
																	   field<double>* orography_in,
																	   field<bool>* landsea_in,
																	   bool prefer_non_diagonals_in)
{
	nbr_lat = lat_in; nbr_lon = lon_in; nlat = nlat_in; nlon = nlon_in;
	orography = orography_in; landsea = landsea_in;
	prefer_non_diagonal_initial_dirs = prefer_non_diagonals_in;
	return find_initial_cell_flow_direction();
}

double sink_filling_algorithm_4::test_calculate_direction_from_neighbor_to_cell(int nbr_lat_in,
																				int nbr_lon_in,
																				int lat_in,
																				int lon_in,
																				int nlat_in,
																				int nlon_in)
{
	lat = lat_in; lon = lon_in; nlat = nlat_in; nlon = nlon_in;
	return calculate_direction_from_neighbor_to_cell(nbr_lat_in,nbr_lon_in);
}
