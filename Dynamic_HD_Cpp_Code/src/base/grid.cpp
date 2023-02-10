/*
 * grid.cpp

 *
 *  Created on: Dec 19, 2016
 *      Author: thomasriddick
 */

#include <cmath>
#include "base/grid.hpp"


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


bool grid::fine_coords_in_same_cell(coords* fine_coords_set_one,
	                               		coords* fine_coords_set_two,
	                               		grid_params* fine_grid_params){
	coords* coarse_coords_set_one = convert_fine_coords(fine_coords_set_one,fine_grid_params);
	coords* coarse_coords_set_two = convert_fine_coords(fine_coords_set_two,fine_grid_params);
	bool coords_in_same_cell = (*coarse_coords_set_one == *coarse_coords_set_two);
	delete coarse_coords_set_one; delete coarse_coords_set_two;
	return coords_in_same_cell;
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

void latlon_grid::for_all_nbrs_wrapped(coords* coords_in,function<void(coords*)> func){
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	for (auto i = latlon_coords_in->get_lat()-1; i <= latlon_coords_in->get_lat()+1;i++){
		for (auto j = latlon_coords_in->get_lon()-1; j <=latlon_coords_in->get_lon()+1; j++){
			if (i == latlon_coords_in->get_lat() && j == latlon_coords_in->get_lon()) continue;
			latlon_coords* nbr_coords = new latlon_coords(i,j);
			if (outside_limits(nbr_coords)) delete nbr_coords;
			else {
				coords* wrapped_coords = latlon_wrapped_coords(nbr_coords);
				if (wrapped_coords != nbr_coords) {
					delete nbr_coords;
				}
				func(wrapped_coords);
			}
		}
	}
}

void latlon_grid::for_non_diagonal_nbrs_wrapped(coords* coords_in,function<void(coords*)> func){
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	for (auto i = latlon_coords_in->get_lat()-1;i<=latlon_coords_in->get_lat()+1;i=i+2){
					latlon_coords* nbr_coords = new latlon_coords(i,latlon_coords_in->get_lon());
			if (outside_limits(nbr_coords)) delete nbr_coords;
			else {
				coords* wrapped_coords = latlon_wrapped_coords(nbr_coords);
				if (wrapped_coords != nbr_coords) delete nbr_coords;
				func(wrapped_coords);
			 }
	}
	for (auto j = latlon_coords_in->get_lon()-1; j<=latlon_coords_in->get_lon()+1;j=j+2){
			latlon_coords* nbr_coords = new latlon_coords(latlon_coords_in->get_lat(),j);
			if (outside_limits(nbr_coords)) delete nbr_coords;
			else {
				coords* wrapped_coords = latlon_wrapped_coords(nbr_coords);
				if (wrapped_coords != nbr_coords) delete nbr_coords;
				func(wrapped_coords);
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

coords* latlon_grid::calculate_downstream_coords_from_dir_based_rdir(coords* initial_coords,
		double rdir){
		latlon_coords* latlon_initial_coords = dynamic_cast<latlon_coords*>(initial_coords);
		if (rdir <= 0 || rdir == 5) return initial_coords->clone();
		int lat_offset = -int(ceil(rdir/3.0))+2;
		int lon_offset = rdir + 3*lat_offset - 5;
		if (nowrap) {
			return new latlon_coords(latlon_initial_coords->get_lat()+lat_offset,
														 	 latlon_initial_coords->get_lon()+lon_offset);
		} else {
			coords* unwrapped_downstream_coords =
				new latlon_coords(latlon_initial_coords->get_lat()+lat_offset,
													latlon_initial_coords->get_lon()+lon_offset);
			coords* downstream_coords = wrapped_coords(unwrapped_downstream_coords);
			if(downstream_coords != unwrapped_downstream_coords)
				delete unwrapped_downstream_coords;
			return downstream_coords;
		}
}

coords* latlon_grid::
	calculate_downstream_coords_from_index_based_rdir(coords* initial_coords,int rdir){
	throw runtime_error("Index based river directions not implemented for lat-lon grid");
	}

coords* latlon_grid::convert_fine_coords(coords* fine_coords,grid_params* fine_grid_params){
		latlon_coords* latlon_fine_coords = static_cast<latlon_coords*>(fine_coords);
		latlon_grid_params* latlon_fine_grid_params =
			static_cast<latlon_grid_params*>(fine_grid_params);
		int fine_cells_per_coarse_cell_lat = latlon_fine_grid_params->get_nlat()/nlat;
		int fine_cells_per_coarse_cell_lon = latlon_fine_grid_params->get_nlon()/nlon;
		//Although C++ rounds towards zero decide to specify floor explicitly in case of negative
		//latitude values
		int coarse_lat = floor(double(latlon_fine_coords->get_lat())/fine_cells_per_coarse_cell_lat);
		int coarse_lon = floor(double(latlon_fine_coords->get_lon())/fine_cells_per_coarse_cell_lon);
		return new latlon_coords(coarse_lat,coarse_lon);
}

void latlon_grid::for_all_fine_pixels_in_coarse_cell(coords* coarse_coords,
	                                         					 grid_params* coarse_grid_params,
	                                         					 function<void(coords*)> func){
		latlon_coords* latlon_coarse_coords = static_cast<latlon_coords*>(coarse_coords);
		latlon_grid_params* latlon_coarse_grid_params =
			static_cast<latlon_grid_params*>(coarse_grid_params);
		int fine_cells_per_coarse_cell_lat = nlat/latlon_coarse_grid_params->get_nlat();
		int fine_cells_per_coarse_cell_lon = nlon/latlon_coarse_grid_params->get_nlon();
		for(int i = latlon_coarse_coords->get_lat()-1*fine_cells_per_coarse_cell_lat;
		    i < latlon_coarse_coords->get_lat()*fine_cells_per_coarse_cell_lat; i++){
			for(int j = latlon_coarse_coords->get_lon()-1*fine_cells_per_coarse_cell_lon;
		      j < latlon_coarse_coords->get_lon()*fine_cells_per_coarse_cell_lon; j++){
				func(new latlon_coords(i,j));
			}
		}
}

int irregular_latlon_grid::get_edge_number(coords* coords_in) {
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	int edge_number = edge_mask[latlon_get_index(latlon_coords_in)];
	if (edge_number != 0) return edge_number;
	else throw runtime_error("Internal logic broken - trying to get edge number of non-edge cell");
}

int irregular_latlon_grid::get_separation_from_initial_edge(coords* coords_in,int edge_number) {
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	if (edge_number == landsea_edge_num || edge_number == true_sink_edge_num) return 0;
	else if(edge_number <= number_of_edges){
		return (edge_seperations[edge_number])[latlon_get_index(latlon_coords_in)];
	} else throw runtime_error("Internal logic broken - invalid initial edge number used as input to "
						     						 "get_separation_from_initial_edge");
}

void irregular_latlon_grid::for_diagonal_nbrs(coords* coords_in,function<void(coords*)> func){
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	for (auto i = latlon_coords_in->get_lat()-1; i<=latlon_coords_in->get_lat()+1;i=i+2){
		for (auto j = latlon_coords_in->get_lon()-1; j<=latlon_coords_in->get_lon()+1;j=j+2){
			latlon_coords* nbr_coords = new latlon_coords(i,j);
			if (grid_mask[latlon_get_index(nbr_coords)]) func(nbr_coords);
			else delete nbr_coords;
		}
	}
};

void irregular_latlon_grid::for_non_diagonal_nbrs(coords* coords_in,function<void(coords*)> func){
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	for (auto i = latlon_coords_in->get_lat()-1;i<=latlon_coords_in->get_lat()+1;i=i+2){
		latlon_coords* nbr_coords = new latlon_coords(i,latlon_coords_in->get_lon());
		if (grid_mask[latlon_get_index(nbr_coords)]) func(nbr_coords);
		else delete nbr_coords;
	}
	for (auto j = latlon_coords_in->get_lon()-1; j<=latlon_coords_in->get_lon()+1;j=j+2){
		latlon_coords* nbr_coords = new latlon_coords(latlon_coords_in->get_lat(),j);
		if (grid_mask[latlon_get_index(nbr_coords)]) func(nbr_coords);
		else delete nbr_coords;
	}
};

void irregular_latlon_grid::for_all_nbrs(coords* coords_in,function<void(coords*)> func){
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	for (auto i = latlon_coords_in->get_lat()-1; i <= latlon_coords_in->get_lat()+1;i++){
		for (auto j = latlon_coords_in->get_lon()-1; j <=latlon_coords_in->get_lon()+1; j++){
			if (i == latlon_coords_in->get_lat() && j == latlon_coords_in->get_lon()) continue;
			latlon_coords* nbr_coords = new latlon_coords(i,j);
			if (grid_mask[latlon_get_index(nbr_coords)]) func(nbr_coords);
			else delete nbr_coords;
		}
	}
};

// This function is not meaningful on a irregular grid; pass to for_all_nbrs
void irregular_latlon_grid::for_all_nbrs_wrapped(coords* coords_in,function<void(coords*)> func){
	for_all_nbrs(coords_in,func);
}

// This function is not meaningful on a irregular grid; pass to for_all_nbrs
void irregular_latlon_grid::for_non_diagonal_nbrs_wrapped(coords* coords_in,
                                                          function<void(coords*)> func){
	latlon_grid::for_non_diagonal_nbrs_wrapped(coords_in,func);
}


void irregular_latlon_grid::for_all(function<void(coords*)> func){
	for (auto i = 0; i < nlat; i++){
		for (auto j = 0; j < nlon; j++){
			latlon_coords* cell_coords = new latlon_coords(i,j);
			if (grid_mask[latlon_get_index(cell_coords)]) func(cell_coords);
			else delete cell_coords;
		}
	}
};

void irregular_latlon_grid::for_all_with_line_breaks(function<void(coords*,bool)> func){
	bool end_of_line;
	for (auto i = 0; i < nlat; i++){
		end_of_line = true;
		for (auto j = 0; j < nlon; j++){
			latlon_coords* cell_coords = new latlon_coords(i,j);
			if (grid_mask[latlon_get_index(cell_coords)]) {
				func(cell_coords,end_of_line);
				end_of_line = false;
			} else delete cell_coords;
		}
	}
};

bool irregular_latlon_grid::is_corner_cell(coords* coords_in){
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	return corner_mask[latlon_get_index(latlon_coords_in)];
}



bool irregular_latlon_grid::
	check_if_cell_is_on_given_edge_number(coords* coords_in,int edge_number) {
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	if (get_edge_number(coords_in) == edge_number) return true;
	//deal with corner cells that are assigned horizontal edge edge numbers but should
	//also be considered to be a vertical edge
	else if (is_corner_cell(coords_in)){
		if (secondary_edge_mask[latlon_get_index(latlon_coords_in)]
		    == edge_number) return true;
		else return false;
	}
	else return false;
}

bool irregular_latlon_grid::is_edge(coords* coords_in) {
	latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
	int cell_edge_number = edge_mask[latlon_get_index(latlon_coords_in)];
	if (cell_edge_number == 0) return false;
	else if((cell_edge_number == get_landsea_edge_num()) ||
					(cell_edge_number == get_true_sink_edge_num())) return false;
	else return true;
}

latlon_coords* irregular_latlon_grid::latlon_wrapped_coords(latlon_coords* coords_in){
	//Wrapping not defined on a irregular grid so just return input coords
	return coords_in;
}

void irregular_latlon_grid::generate_edge_and_corner_masks(){
	switch (geometry_type){
		case icosohedral:
			//Case needs its own block to declare local variables
			{
				bool* binary_edge_mask = new bool[get_total_size()];
				bool* faces_east_mask  = new bool[get_total_size()];
				bool* faces_west_mask = new bool[get_total_size()];
				bool* faces_north_mask = new bool[get_total_size()];
				bool* faces_south_mask = new bool[get_total_size()];
				int north_facing_corner_count = 0;
				int south_facing_corner_count = 0;
				int east_facing_corner_count = 0;
				int west_facing_corner_count = 0;

				latlon_grid::for_all([&](coords* coords_in){
					latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
					if( grid_mask[latlon_get_index(latlon_coords_in)] ){
						bool is_edge_cell = false;
						bool faces_east = false;
						bool faces_west = false;
						bool faces_north = false;
						bool faces_south = false;
						int outward_facing_direction_count = 0;
						latlon_grid::for_all_nbrs(coords_in,[&](coords* nbr_coords){
							latlon_coords* latlon_nbr_coords = static_cast<latlon_coords*>(nbr_coords);
							if(! grid_mask[latlon_get_index(latlon_nbr_coords)] ) {
								is_edge_cell = true;
								outward_facing_direction_count += 1;
								int rdir = int(calculate_dir_based_rdir(coords_in,nbr_coords));
								switch(rdir){
									case 1:
										faces_west = true;
										faces_south = true;
										break;
									case 2:
										faces_south = true;
										break;
									case 3:
										faces_south = true;
										faces_east = true;
										break;
									case 4:
										faces_west = true;
										break;
									case 5:
										throw runtime_error("");
										break;
									case 6:
										faces_east = true;
										break;
									case 7:
										faces_west = true;
										faces_north = true;
										break;
									case 8:
										faces_north = true;
										break;
									case 9:
										faces_north = true;
										faces_east = true;
										break;
									default:
										throw runtime_error("");
								}
							}
						});
						binary_edge_mask[latlon_get_index(latlon_coords_in)] = is_edge_cell;
						if (outward_facing_direction_count >= 5) {
							corner_mask[latlon_get_index(latlon_coords_in)] = true;
							if(faces_north) north_facing_corner_count += 1;
							if(faces_south) south_facing_corner_count += 1;
							if(faces_east) east_facing_corner_count += 1;
							if(faces_west) west_facing_corner_count += 1;
						}
						faces_east_mask[latlon_get_index(latlon_coords_in)] = faces_east;
						faces_west_mask[latlon_get_index(latlon_coords_in)] = faces_west;
						faces_north_mask[latlon_get_index(latlon_coords_in)] = faces_north;
						faces_south_mask[latlon_get_index(latlon_coords_in)] = faces_south;
						delete coords_in;
						}
				});
				latlon_grid::for_all([&](coords* coords_in){
					latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
					if( grid_mask[latlon_get_index(latlon_coords_in)] ){
						if (north_facing_corner_count == 2 && south_facing_corner_count == 2){
						// In this case there is one exactly vertical side
							if (faces_north_mask[latlon_get_index(latlon_coords_in)]){
								edge_mask[latlon_get_index(latlon_coords_in)] = top_edge_num;
							} else if (faces_south_mask[latlon_get_index(latlon_coords_in)]){
								edge_mask[latlon_get_index(latlon_coords_in)] = bottom_edge_num;
							} else if (west_facing_corner_count == 2 &&
							           faces_west_mask[latlon_get_index(latlon_coords_in)]) {
								edge_mask[latlon_get_index(latlon_coords_in)] = left_edge_num;
							} else if (east_facing_corner_count == 2 &&
							           faces_east_mask[latlon_get_index(latlon_coords_in)]) {
								edge_mask[latlon_get_index(latlon_coords_in)] = right_edge_num;
							} else throw runtime_error("");
							if (corner_mask[latlon_get_index(latlon_coords_in)]){
								if (faces_west_mask[latlon_get_index(latlon_coords_in)] &&
								    west_facing_corner_count == 2){
									secondary_edge_mask[latlon_get_index(latlon_coords_in)] = left_edge_num;
								} else if(faces_east_mask[latlon_get_index(latlon_coords_in)] &&
								          east_facing_corner_count == 2){
									secondary_edge_mask[latlon_get_index(latlon_coords_in)] = right_edge_num;
								} else if(faces_north_mask[latlon_get_index(latlon_coords_in)] &&
								          faces_south_mask[latlon_get_index(latlon_coords_in)]){
									secondary_edge_mask[latlon_get_index(latlon_coords_in)] = bottom_edge_num;
								} else throw runtime_error("");
							}
						} else if (north_facing_corner_count == 2){
							//The top is roughly horizontal
							if (faces_north_mask[latlon_get_index(latlon_coords_in)]){
								edge_mask[latlon_get_index(latlon_coords_in)] = top_edge_num;
							} else if (faces_west_mask[latlon_get_index(latlon_coords_in)]) {
								edge_mask[latlon_get_index(latlon_coords_in)] = left_edge_num;
							} else if (faces_east_mask[latlon_get_index(latlon_coords_in)]) {
								edge_mask[latlon_get_index(latlon_coords_in)] = right_edge_num;
							} else throw runtime_error("");
							if (corner_mask[latlon_get_index(latlon_coords_in)]){
								if(faces_north_mask[latlon_get_index(latlon_coords_in)]){
									if(faces_west_mask[latlon_get_index(latlon_coords_in)]){
										secondary_edge_mask[latlon_get_index(latlon_coords_in)] = left_edge_num;
									} else if(faces_east_mask[latlon_get_index(latlon_coords_in)]){
										secondary_edge_mask[latlon_get_index(latlon_coords_in)] = right_edge_num;
									} else throw runtime_error("");
								} else if(faces_west_mask[latlon_get_index(latlon_coords_in)]){
									if(faces_east_mask[latlon_get_index(latlon_coords_in)]){
										secondary_edge_mask[latlon_get_index(latlon_coords_in)] = right_edge_num;
									} else throw runtime_error("");
								} else throw runtime_error("");
							}
						} else if (south_facing_corner_count == 2) {
							//The bottom is roughly horizontal
							if (faces_south_mask[latlon_get_index(latlon_coords_in)]){
								edge_mask[latlon_get_index(latlon_coords_in)] = bottom_edge_num;
							} else if (faces_west_mask[latlon_get_index(latlon_coords_in)]) {
								edge_mask[latlon_get_index(latlon_coords_in)] = left_edge_num;
							} else if (faces_east_mask[latlon_get_index(latlon_coords_in)]) {
								edge_mask[latlon_get_index(latlon_coords_in)] = right_edge_num;
							} else throw runtime_error("");
							if (corner_mask[latlon_get_index(latlon_coords_in)]){
								if(faces_south_mask[latlon_get_index(latlon_coords_in)]){
									if(faces_west_mask[latlon_get_index(latlon_coords_in)]){
										secondary_edge_mask[latlon_get_index(latlon_coords_in)] = left_edge_num;
									} else if(faces_east_mask[latlon_get_index(latlon_coords_in)]){
										secondary_edge_mask[latlon_get_index(latlon_coords_in)] = right_edge_num;
									} else throw runtime_error("");
								} else if(faces_west_mask[latlon_get_index(latlon_coords_in)]){
									if(faces_east_mask[latlon_get_index(latlon_coords_in)]){
										secondary_edge_mask[latlon_get_index(latlon_coords_in)] = right_edge_num;
									} else throw runtime_error("");
								} else throw runtime_error("");
							}
						} else throw runtime_error("");
					}
				});
			}
			break;
		default:
			throw runtime_error("Unknown geometry type");
	}
}

// void irregular_latlon_grid::generate_edge_seperations(){
// 	q
// 	latlon_grid::for_all([&](coords* coords_in){
// 		if(edge_mask[latlon_get_index(coords_in)] != 0){
// 		q.push()
// 		}
// 	while q not zero
// 		pop
// 		find nbrs
// 		calc nbr length
// 		pop nbrs
// 	});
// }

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
		throw runtime_error("icon_single_index_grid constructor received wrong kind of grid parameters");
	}
};

void icon_single_index_grid::for_diagonal_nbrs(coords* coords_in,function<void(coords*)> func) {
	generic_1d_coords* generic_1d_coords_in = static_cast<generic_1d_coords*>(coords_in);
	if (use_secondary_neighbors) {
		for (auto i = 0; i < 9; i++) {
			int neighbor_index = get_cell_secondary_neighbors_index(generic_1d_coords_in,i);
			if (neighbor_index != no_neighbor) func(new generic_1d_coords(neighbor_index));
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

void icon_single_index_grid::for_all_nbrs_wrapped(coords* coords_in,function<void(coords*)> func){
	for_all_nbrs(coords_in,func);
}

void icon_single_index_grid::
		 for_non_diagonal_nbrs_wrapped(coords* coords_in,function<void(coords*)> func){
	for_non_diagonal_nbrs(coords_in,func);
}

void icon_single_index_grid::for_all(function<void(coords*)> func) {
	for (auto i = 0; i < ncells; i++) {
		func(new generic_1d_coords(i + array_offset));
	}
}

void icon_single_index_grid::for_all_with_line_breaks(function<void(coords*,bool)> func){
	for (auto i = 0; i < ncells; i++) {
		func(new generic_1d_coords(i + array_offset),false);
	}
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
	int edge_num = edge_nums[generic_1d_coords_in->get_index()-array_offset];
	return (edge_num > 3 && edge_num <= 6);
}

bool icon_single_index_grid::check_if_cell_is_on_given_edge_number(coords* coords_in,int edge_number) {
	generic_1d_coords* generic_1d_coords_in = static_cast<generic_1d_coords*>(coords_in);
	int cell_edge_number = edge_nums[generic_1d_coords_in->get_index() - array_offset];
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
	return ( edge_nums[generic_1d_coords_in->get_index() - array_offset] != no_edge );
}

int icon_single_index_grid::get_edge_number(coords* coords_in) {
	generic_1d_coords* generic_1d_coords_in = static_cast<generic_1d_coords*>(coords_in);
	int edge_num = edge_nums[generic_1d_coords_in->get_index() - array_offset];
	if(edge_num < 1 || edge_num > 6) {
		throw runtime_error("Internal logic broken - trying to get edge number of non-edge cell");
	}
	if(edge_num > 3) return edge_num - 3;
	else             return edge_num;
}

int icon_single_index_grid::get_separation_from_initial_edge(coords* coords_in,int edge_number) {
	generic_1d_coords* generic_1d_coords_in = static_cast<generic_1d_coords*>(coords_in);
	int edge_num = edge_nums[generic_1d_coords_in->get_index() - array_offset];
	if(edge_num < 1 || edge_num > 6) {
		runtime_error("Internal logic broken - invalid initial edge number used as input to "
						     	"get_separation_from_initial_edge");
	}
	if(edge_num > 3) edge_num -= 3;
	// Offset of -1 as edge numbers go from 1 to 3
	return edge_separations[(generic_1d_coords_in->get_index() - array_offset)*3 + edge_num - 1];
}

coords* icon_single_index_grid::convert_fine_coords(coords* fine_coords,grid_params* fine_grid_params){
	icon_single_index_grid_params* fine_grid_params_local =
	   static_cast<icon_single_index_grid_params*>(fine_grid_params);
	generic_1d_coords* fine_coords_local = static_cast<generic_1d_coords*>(fine_coords);
	int* fine_grid_mapping_to_coarse_grid = fine_grid_params_local->get_mapping_to_coarse_grid();
	if (fine_grid_mapping_to_coarse_grid) {
		return new generic_1d_coords(fine_grid_mapping_to_coarse_grid[fine_coords_local->get_index() -
		                             fine_grid_params_local->get_array_offset()]);
	} else throw runtime_error("No mapping between fine and coarse grid provided");
}

coords* icon_single_index_grid::
	calculate_downstream_coords_from_dir_based_rdir(coords* initial_coords,double rdir){
	throw runtime_error("Direction based river directions not compatible with Icon grid");
	}

coords* icon_single_index_grid::
	calculate_downstream_coords_from_index_based_rdir(coords* initial_coords,int rdir){
		return new generic_1d_coords(rdir);
	}

#if USE_NETCDFCPP
void icon_single_index_grid_params::icon_single_index_grid_read_params_file(){
	NcFile grid_params_file(icon_grid_params_filepath.c_str(), NcFile::read);
	NcDim dim;
	const int NCELLS_NAMES = 3;
	string potential_ncells_names[NCELLS_NAMES] = {"ncells","cells","cell"};
	for (int i = 0; dim.isNull();i++){
		dim = grid_params_file.getDim(potential_ncells_names[i]);
		if (i >= NCELLS_NAMES) throw runtime_error("Invalid ncells name");
	}
	ncells = int(dim.getSize());
	NcVar neighboring_cell_indices_var = grid_params_file.getVar("neighbor_cell_index");
	if (neighboring_cell_indices_var.isNull())
		throw runtime_error("Invalid data in specified grid file");
	neighboring_cell_indices = new int[3*ncells];
	int* neighboring_cell_indices_swapped_dims = new int[3*ncells];
	neighboring_cell_indices_var.getVar(neighboring_cell_indices_swapped_dims);
	for (int i = 0; i < ncells; i++){
		neighboring_cell_indices[3*i]   = neighboring_cell_indices_swapped_dims[i];
		neighboring_cell_indices[3*i+1] = neighboring_cell_indices_swapped_dims[ncells+i];
		neighboring_cell_indices[3*i+2] = neighboring_cell_indices_swapped_dims[2*ncells+i];
	}
	delete[] neighboring_cell_indices_swapped_dims;
	if (use_secondary_neighbors) icon_single_index_grid_calculate_secondary_neighbors();
}
#endif

void icon_single_index_grid_params::icon_single_index_grid_calculate_secondary_neighbors(){
	//Within this function work with c array indices rather than native icon indices so apply offset to
	//indices read from arrays
	secondary_neighboring_cell_indices = new int[9l*(long)ncells];
	calculated_secondary_neighboring_cell_indices = true;
	for(int index_over_grid=0;index_over_grid<ncells;index_over_grid++){
		//Six secondary neighbors are neighbors of primary neighbors
		for(int index_over_primary_nbrs=0; index_over_primary_nbrs < 3; index_over_primary_nbrs++){
			int primary_neighbor_index =
				neighboring_cell_indices[3*index_over_grid+index_over_primary_nbrs] - array_offset;
			int valid_secondary_nbr_count = 0;
			for(int index_over_secondary_nbrs=0; index_over_secondary_nbrs < 3; index_over_secondary_nbrs++){
				//2 rather than 3 times primary neighbor index as we miss out 1 secondary neighbor for each
				//primary neighbor
				int secondary_neighbor_index =
					neighboring_cell_indices[3*primary_neighbor_index+index_over_secondary_nbrs] - array_offset;
				if (secondary_neighbor_index != index_over_grid) {
					//Note this leaves gaps for the remaining three secondary neighbors
					secondary_neighboring_cell_indices[9l*long(index_over_grid)+3l*(long)index_over_primary_nbrs+
					                                   (long)valid_secondary_nbr_count] =
																							secondary_neighbor_index + array_offset;
					valid_secondary_nbr_count++;
				}
			}
		}
	//Three secondary neighbors are common neighbors of the existing secondary neighbors
	int gap_index = 2;
	//Last secondary neighbor is as yet unfilled so loop only up to an index of
	for(int index_over_secondary_nbrs=0; index_over_secondary_nbrs < 8;
	    index_over_secondary_nbrs++){
		//skip as yet unfilled entries in the secondary neighbors array
		if ((index_over_secondary_nbrs+1)%3 == 0) index_over_secondary_nbrs++;
		int first_secondary_neighbor_index =
			secondary_neighboring_cell_indices[9l*long(index_over_grid)+
		                                     long(index_over_secondary_nbrs)] - array_offset;
	  //Last secondary neighbor is as yet unfilled so loop only up to an index of 7
		for(int second_index_over_secondary_nbrs=index_over_secondary_nbrs+2;
		    second_index_over_secondary_nbrs < 8;
	    second_index_over_secondary_nbrs++){
			if ((second_index_over_secondary_nbrs+1)%3 == 0) second_index_over_secondary_nbrs++;
			int second_secondary_neighbor_index =
				secondary_neighboring_cell_indices[9l*(long)index_over_grid +
				                                   (long)second_index_over_secondary_nbrs]
					- array_offset;

				//Some tertiary neighbors are also secondary neighbors
				for(int index_over_tertiary_nbrs=0; index_over_tertiary_nbrs < 3; index_over_tertiary_nbrs++){
					int tertiary_neighbor_index =
						neighboring_cell_indices[first_secondary_neighbor_index*3 +
					                           index_over_tertiary_nbrs] - array_offset;
					//Test to see if this one of the twelve 5-point vertices in the grid
					if(second_secondary_neighbor_index == tertiary_neighbor_index) {
						secondary_neighboring_cell_indices[9l*(long)index_over_grid+
		                                           (long)gap_index] = no_neighbor;
						gap_index += 3;
						continue;
					}
					for(int second_index_over_tertiary_nbrs=0; second_index_over_tertiary_nbrs < 3;
			    second_index_over_tertiary_nbrs++){
						int second_tertiary_neighbor_index =
							neighboring_cell_indices[second_secondary_neighbor_index*3 +
					                             second_index_over_tertiary_nbrs] - array_offset;
						if(second_tertiary_neighbor_index == tertiary_neighbor_index){
							secondary_neighboring_cell_indices[9l*(long)index_over_grid+
		                                             (long)gap_index] = tertiary_neighbor_index + array_offset;
							gap_index += 3;
						}
					}
			  }
			}
		}
	}
}

grid* grid_factory(grid_params* grid_params_in){
	if(latlon_grid_params* latlon_params = dynamic_cast<latlon_grid_params*>(grid_params_in)){
		return new latlon_grid(latlon_params);
	} else if(icon_single_index_grid_params* icon_single_index_params =
	          dynamic_cast<icon_single_index_grid_params*>(grid_params_in)){
		return new icon_single_index_grid(icon_single_index_params);
	} else {
		throw runtime_error("Grid type not known to field class, if it should be please add appropriate code to constructor");
	}
	return nullptr;
};
