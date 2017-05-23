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
												 grid_params* grid_params_in)
{
	_grid_params = grid_params_in;
	_grid = grid_factory(_grid_params);
	landsea = new field<bool>(landsea_in,_grid_params);
	ls_seed_points = new field<bool>(ls_seed_points_in,_grid_params);
	completed_cells = new field<bool>(_grid_params);
	completed_cells->set_all(false);
}

create_connected_landsea_mask::~create_connected_landsea_mask()
{
	delete landsea;
	delete ls_seed_points;
	delete completed_cells;
	delete _grid;
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
	function<void(coords*)> add_ls_point_to_q_func = [&](coords* coords_in) {
		if ((*ls_seed_points)(coords_in)) {
			q.push(new landsea_cell(coords_in));
			(*completed_cells)(coords_in) = true;
		}};
	_grid->for_all(add_ls_point_to_q_func);
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
	if (use_diagonals || neighbors_coords->size() > diagonal_neighbors) {
		if ((*landsea)(nbr_coords) && !((*completed_cells)(nbr_coords))) {
			q.push(new landsea_cell(nbr_coords));
			(*completed_cells)(nbr_coords) = true;
		}
	}
	neighbors_coords->pop_back();
}

void create_connected_landsea_mask::deep_copy_completed_cells_to_landsea()
{
	_grid->for_all([&](coords* coords_in){(*landsea)(coords_in) = (*completed_cells)(coords_in);
										  delete coords_in;});
}
