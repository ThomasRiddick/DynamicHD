/*
 * connected_lsmask_generation_algorithm.cpp
 *
 *  Created on: May 24, 2016
 *      Author: thomasriddick
 */

#include "connected_lsmask_generation_algorithm.hpp"
#include <cmath>

void create_connected_landsea_mask::setup_flags(bool use_diagonals_in)
{
	use_diagonals = use_diagonals_in;
}

void create_connected_landsea_mask::setup_fields(bool* landsea_in,
												 bool* ls_seed_points_in,
												 int nlat_in, int nlon_in)
{
	nlat = nlat_in;
	nlon = nlon_in;
	landsea = new field<bool>(landsea_in,nlat_in,nlon_in);
	ls_seed_points = new field<bool>(ls_seed_points_in,nlat_in,nlon_in);
	completed_cells = new field<bool>(nlat_in,nlon_in);
	completed_cells->set_all(false);
}

create_connected_landsea_mask::~create_connected_landsea_mask()
{
	delete landsea;
	delete ls_seed_points;
	delete completed_cells;
}

void create_connected_landsea_mask::generate_connected_mask()
{
	add_ls_seed_points_to_q();
	while (!q.empty()) {
		landsea_cell* center_cell = q.front();
		q.pop();
		center_coords = center_cell->get_cell_coords();
		process_neighbors();
		delete center_cell;
	}
	deep_copy_completed_cells_to_landsea();
}

void create_connected_landsea_mask::add_ls_seed_points_to_q()
{
	for (auto i = 0; i < nlat; i++){
		for (auto j = 0; j < nlon; j++){
			if ((*ls_seed_points)(i,j)) {
				q.push(new landsea_cell(i,j));
				(*completed_cells)(i,j) = true;
			}
		}
	}
}

void create_connected_landsea_mask::process_neighbors()
{
	neighbors_coords = landsea->get_neighbors_coords(center_coords,4);
	diagonal_neighbors = floor(neighbors_coords->size()/2.0);
	while (!neighbors_coords->empty()){
		process_neighbor();
	}
	delete neighbors_coords;
}

inline void create_connected_landsea_mask::process_neighbor()
{
	auto nbr_coords = neighbors_coords->back();
	nbr_lat = nbr_coords->first;
	nbr_lon = nbr_coords->second;
	if (use_diagonals || neighbors_coords->size() > diagonal_neighbors) {
		if ((*landsea)(nbr_lat,nbr_lon) && !((*completed_cells)(nbr_lat,nbr_lon))) {
			q.push(new landsea_cell(nbr_lat,nbr_lon));
			(*completed_cells)(nbr_lat,nbr_lon) = true;
		}
	}
	neighbors_coords->pop_back();
}

void create_connected_landsea_mask::deep_copy_completed_cells_to_landsea()
{
	for (auto i = 0; i < nlat; i++ ){
		for (auto j = 0; j < nlon; j++) {
			(*landsea)(i,j) = (*completed_cells)(i,j);
		}
	}
}
