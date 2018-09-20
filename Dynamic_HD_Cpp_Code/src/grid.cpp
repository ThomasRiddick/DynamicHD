/*
 * grid.cpp

 *
 *  Created on: Dec 19, 2016
 *      Author: thomasriddick
 */

#include "grid.hpp"

/* IMPORTANT!
 * Note that a description of each functions purpose is provided with the function declaration in
 * the header file not with the function definition. The definition merely provides discussion of
 * details of the implementation.
 */

bool grid::check_if_cell_connects_two_landsea_or_true_sink_points(int edge_number, bool is_landsea_nbr,
																		 bool is_true_sink) {

	if ((edge_number == get_landsea_edge_num() ||
			edge_number == get_true_sink_edge_num()) &&
			(is_true_sink || is_landsea_nbr)) return true;
	else return false;
}

latlon_grid::latlon_grid(grid_params* params){
	grid_type = grid_types::latlon;
	if(latlon_grid_params* params_local = dynamic_cast<latlon_grid_params*>(params)){
		nlat = params_local->get_nlat();
		nlon = params_local->get_nlon();
		total_size = nlat*nlon;
		nowrap = params_local->get_nowrap();
	} else {
		throw runtime_error("latlon_grid constructor received wrong kind of grid parameters");
	}
};

void latlon_grid::for_diagonal_nbrs(coords* coords_in,function<void(coords*)> func){
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	for (auto i = latlon_coords_in->get_lat()-1; i<=latlon_coords_in->get_lat()+1;i=i+2){
		for (auto j = latlon_coords_in->get_lon()-1; j<=latlon_coords_in->get_lon()+1;j=j+2){
			func(new latlon_coords(i,j));
		}
	}
};


void latlon_grid::for_non_diagonal_nbrs(coords* coords_in,function<void(coords*)> func){
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	for (auto i = latlon_coords_in->get_lat()-1;i<=latlon_coords_in->get_lat()+1;i=i+2){
		func(new latlon_coords(i,latlon_coords_in->get_lon()));
	}
	for (auto j = latlon_coords_in->get_lon()-1; j<=latlon_coords_in->get_lon()+1;j=j+2){
		func(new latlon_coords(latlon_coords_in->get_lat(),j));
	}
};

void latlon_grid::for_all_nbrs(coords* coords_in,function<void(coords*)> func){
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	for (auto i = latlon_coords_in->get_lat()-1; i <= latlon_coords_in->get_lat()+1;i++){
		for (auto j = latlon_coords_in->get_lon()-1; j <=latlon_coords_in->get_lon()+1; j++){
			if (i == latlon_coords_in->get_lat() && j == latlon_coords_in->get_lon()) continue;
			func(new latlon_coords(i,j));
		}
	}
};

void latlon_grid::for_all(function<void(coords*)> func){
	for (auto i = 0; i < nlat; i++){
		for (auto j = 0; j < nlon; j++){
			func(new latlon_coords(i,j));
		}
	}
};

void latlon_grid::for_all_with_line_breaks(function<void(coords*,bool)> func){
	bool end_of_line;
	for (auto i = 0; i < nlat; i++){
		end_of_line = true;
		for (auto j = 0; j < nlon; j++){
			func(new latlon_coords(i,j),end_of_line);
			end_of_line = false;
		}
	}
};

latlon_coords* latlon_grid::latlon_wrapped_coords(latlon_coords* coords_in){
	if (coords_in->get_lon() < 0) {
		return new latlon_coords(coords_in->get_lat(),nlon + coords_in->get_lon());
	}
	else if (coords_in->get_lon() >= nlon) {
		return new latlon_coords(coords_in->get_lat(),coords_in->get_lon() - nlon);
	}
	else {
		return coords_in;
	}
};

bool latlon_grid::latlon_non_diagonal(latlon_coords* start_coords,latlon_coords* dest_coords){
	return (dest_coords->get_lat() == start_coords->get_lat() ||
			dest_coords->get_lon() == start_coords->get_lon());
}

bool latlon_grid::is_corner_cell(coords* coords_in){
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	return ((latlon_coords_in->get_lat() == 0 && latlon_coords_in->get_lon() == 0) ||
			(latlon_coords_in->get_lat() == 0 && latlon_coords_in->get_lon() == nlon-1) ||
			(latlon_coords_in->get_lat() == nlat-1 && latlon_coords_in->get_lon() == 0) ||
			(latlon_coords_in->get_lat() == nlat-1 && latlon_coords_in->get_lon() == nlon-1));
}



bool latlon_grid::check_if_cell_is_on_given_edge_number(coords* coords_in,int edge_number) {
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	if (get_edge_number(coords_in) == edge_number) return true;
	//deal with corner cells that are assigned horizontal edge edge numbers but should
	//also be considered to be a vertical edge
	else if (edge_number == left_vertical_edge_num &&
			 latlon_coords_in->get_lon() == 0) return true;
	else if (edge_number == right_vertical_edge_num &&
			 latlon_coords_in->get_lon() == 0 ) return true;
	else return false;
}

bool latlon_grid::is_edge(coords* coords_in) {
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	if (latlon_coords_in->get_lat() == 0 ||
		latlon_coords_in->get_lat() == nlat-1 ||
		latlon_coords_in->get_lon() == 0 ||
		latlon_coords_in->get_lon() == nlon-1) return true;
	else return false;
}

int latlon_grid::get_edge_number(coords* coords_in) {
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	if (latlon_coords_in->get_lat() == 0)      		return top_horizontal_edge_num;
	else if (latlon_coords_in->get_lat() == nlat-1) return bottom_horizontal_edge_num;
	else if (latlon_coords_in->get_lon() == 0)      return left_vertical_edge_num;
	else if (latlon_coords_in->get_lon() == nlon-1) return right_vertical_edge_num;
	else throw runtime_error("Internal logic broken - trying to get edge number of non-edge cell");
}

int latlon_grid::get_separation_from_initial_edge(coords* coords_in,int edge_number) {
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	if 		(edge_number == top_horizontal_edge_num)  	return latlon_coords_in->get_lat();
	else if (edge_number == bottom_horizontal_edge_num) return nlat - latlon_coords_in->get_lat() - 1;
	else if (edge_number == left_vertical_edge_num) 	return latlon_coords_in->get_lon();
	else if (edge_number == right_vertical_edge_num) 	return nlon - latlon_coords_in->get_lon() - 1;
	else if (edge_number == landsea_edge_num || edge_number == true_sink_edge_num) return 0;
	else throw runtime_error("Internal logic broken - invalid initial edge number used as input to "
						     "get_separation_from_initial_edge");
}

double latlon_grid::latlon_calculate_dir_based_rdir(latlon_coords* start_coords,latlon_coords* dest_coords){
	//deal with wrapping longitude
	int start_lon_loc = start_coords->get_lon();
	if (dest_coords->get_lon() == 0 && start_coords->get_lon() == nlon - 1) start_lon_loc = -1;
	if (start_coords->get_lon() == 0 && dest_coords->get_lon() == nlon - 1) start_lon_loc = nlon;
	//deal with the case of the neighbor and cell not actually being neighbors
	if ((abs(start_coords->get_lat() - dest_coords->get_lat()) > 1) ||
			(abs(start_lon_loc - dest_coords->get_lon()) > 1)) {
		throw runtime_error("Internal logic broken - trying to calculate direction between two non-neighboring cells");
	}
	return double(3*(start_coords->get_lat() - dest_coords->get_lat())
				 	 + (dest_coords->get_lon() - start_lon_loc) + 5);
}

icon_single_index_grid::icon_single_index_grid(grid_params* params){
	grid_type = grid_types::icon_single_index;
	if(icon_single_index_grid_params* params_local = dynamic_cast<icon_single_index_grid_params*>(params)){
		ncells = params_local->get_ncells();
		total_size = ncells;
		nowrap = params_local->get_nowrap();
		neighboring_cell_indices = params_local->get_neighboring_cell_indices();
		use_secondary_neighbors = params_local->get_use_secondary_neighbors();
		if (use_secondary_neighbors) {
			secondary_neighboring_cell_indices = params_local->get_secondary_neighboring_cell_indices();
		}
		if (nowrap) subgrid_mask = params_local->get_subgrid_mask();
	} else {
		throw runtime_error("latlon_grid constructor received wrong kind of grid parameters");
	}
};

void icon_single_index_grid::for_diagonal_nbrs(coords* coords_in,function<void(coords*)> func) {
	generic_1d_coords* generic_1d_coords_in = static_cast<generic_1d_coords*>(coords_in);
	if (use_secondary_neighbors) {
		for (auto i = 0; i < 9; i++) {
			int neighbor_index = get_cell_secondary_neighbors_index(generic_1d_coords_in,i);
			func(new generic_1d_coords(neighbor_index));
		}
	}
}

void icon_single_index_grid::for_non_diagonal_nbrs(coords* coords_in,function<void(coords*)> func) {
	generic_1d_coords* generic_1d_coords_in = static_cast<generic_1d_coords*>(coords_in);
	for (auto i = 0; i < 3; i++) {
		int neighbor_index = get_cell_neighbors_index(generic_1d_coords_in,i);
		func(new generic_1d_coords(neighbor_index));
	}
}

void icon_single_index_grid::for_all_nbrs(coords* coords_in,function<void(coords*)> func) {
	for_diagonal_nbrs(coords_in,func);
	for_non_diagonal_nbrs(coords_in,func);
}

void icon_single_index_grid::for_all(function<void(coords*)> func) {
	for (auto i = 0; i < ncells; i++) {
		func(new generic_1d_coords(i));
	}
}

void icon_single_index_grid::for_all_with_line_breaks(function<void(coords*,bool)> func){
	throw runtime_error("for_all_with_line_break not implemented for Icon grid");
}

generic_1d_coords* icon_single_index_grid::icon_single_index_wrapped_coords(generic_1d_coords* coords_in) {
	//ICON grid is naturally wrapped - this function should be optimized out by the compiler
	return coords_in;
}

bool icon_single_index_grid::icon_single_index_non_diagonal(generic_1d_coords* start_coords,
                                                            generic_1d_coords* dest_coords){
	int start_index = start_coords->get_index();
	for (auto i = 0; i < 3; i++) {
		if (get_cell_neighbors_index(dest_coords,i) == start_index) return true;
	}
	return false;
}

bool icon_single_index_grid::is_corner_cell(coords* coords_in){
	generic_1d_coords* generic_1d_coords_in = static_cast<generic_1d_coords*>(coords_in);
	int edge_num = edge_nums[generic_1d_coords_in->get_index()];
	return (edge_num > 3 && edge_num <= 6);
}

bool icon_single_index_grid::check_if_cell_is_on_given_edge_number(coords* coords_in,int edge_number) {
	generic_1d_coords* generic_1d_coords_in = static_cast<generic_1d_coords*>(coords_in);
	int cell_edge_number = edge_nums[generic_1d_coords_in->get_index()];
	if (cell_edge_number == edge_number) return true;
	//deal with corner cells that are assigned horizontal edge edge numbers but should
	//also be considered to be a vertical edge
	else if (cell_edge_number == top_corner  &&
			     edge_number == right_diagonal_edge_num) return true;
	else if (cell_edge_number == bottom_right_corner &&
			     edge_number == bottom_horizontal_edge_num) return true;
	else if (cell_edge_number == bottom_left_corner &&
			     edge_number == left_diagonal_edge_num) return true;
	else return false;
}

bool icon_single_index_grid::is_edge(coords* coords_in) {
	generic_1d_coords* generic_1d_coords_in = static_cast<generic_1d_coords*>(coords_in);
	return ( edge_nums[generic_1d_coords_in->get_index()] != no_edge );
}

int icon_single_index_grid::get_edge_number(coords* coords_in) {
	generic_1d_coords* generic_1d_coords_in = static_cast<generic_1d_coords*>(coords_in);
	int edge_num = edge_nums[generic_1d_coords_in->get_index()];
	if(edge_num < 1 || edge_num > 6) {
		throw runtime_error("Internal logic broken - trying to get edge number of non-edge cell");
	}
	if(edge_num > 3) return edge_num - 3;
	else             return edge_num;
}

int icon_single_index_grid::get_separation_from_initial_edge(coords* coords_in,int edge_number) {
	generic_1d_coords* generic_1d_coords_in = static_cast<generic_1d_coords*>(coords_in);
	int edge_num = edge_nums[generic_1d_coords_in->get_index()];
	if(edge_num < 1 || edge_num > 6) {
		runtime_error("Internal logic broken - invalid initial edge number used as input to "
						     	"get_separation_from_initial_edge");
	}
	if(edge_num > 3) edge_num -= 3;
	// Offset of -1 as edge numbers go from 1 to 3
	return edge_separations[generic_1d_coords_in->get_index()*3 + edge_num - 1];
}

grid* grid_factory(grid_params* grid_params_in){
	if(latlon_grid_params* latlon_params = dynamic_cast<latlon_grid_params*>(grid_params_in)){
		return new latlon_grid(latlon_params);
	} else {
		throw runtime_error("Grid type not known to field class, if it should be please add appropriate code to constructor");
	}
	return nullptr;
};

