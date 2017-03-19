/*
 * grid.cpp

 *
 *  Created on: Dec 19, 2016
 *      Author: thomasriddick
 */

#include "grid.hpp"

latlon_grid::latlon_grid(grid_params* params){
	grid_type = grid_types::latlon;
	if(latlon_grid_params* params_local = dynamic_cast<latlon_grid_params*>(params)){
		nlat = params_local->get_nlat();
		nlon = params_local->get_nlon();
		total_size = nlat*nlon;
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

bool latlon_grid::check_if_cell_is_on_given_edge_number(coords* coords_in,int edge_number){
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	if (get_edge_number(coords_in) == edge_number) return true;
	//deal with corner cells that are assigned vertical edge edge numbers but should
	//also be considered to be a horizontal edge
	else if ((edge_number == top_horizontal_edge_num) &&
			(latlon_coords_in->get_lon() == 0)) 		 return true;
	else if ((edge_number == bottom_horizontal_edge_num) &&
			(latlon_coords_in->get_lon() == nlon-1))     return true;
	else return false;
}

int latlon_grid::get_edge_number(coords* coords_in) {
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	if (latlon_coords_in->get_lat() == 0)      return left_vertical_edge_num;
	else if (latlon_coords_in->get_lat() == nlat-1) return right_vertical_edge_num;
	else if (latlon_coords_in->get_lon() == 0)      return top_horizontal_edge_num;
	else if (latlon_coords_in->get_lon() == nlon-1) return bottom_horizontal_edge_num;
	else throw runtime_error("Internal logic broken - trying to get edge number of non-edge cell");
}

int latlon_grid::get_separation_from_initial_edge(coords* coords_in,int edge_number) {
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	if (left_vertical_edge_num) return latlon_coords_in->get_lat();
	else if (right_vertical_edge_num) return latlon_coords_in->get_lat() - nlat + 1;
	else if (top_horizontal_edge_num) return latlon_coords_in->get_lon();
	else if (bottom_horizontal_edge_num) return latlon_coords_in->get_lon() - nlon + 1;
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

grid* grid_factory(grid_params* grid_params_in){
	if(latlon_grid_params* latlon_params = dynamic_cast<latlon_grid_params*>(grid_params_in)){
		return new latlon_grid(latlon_params);
	} else {
		throw runtime_error("Grid type not known to field class, if it should be please add appropriate code to constructor");
	}
	return nullptr;
};

