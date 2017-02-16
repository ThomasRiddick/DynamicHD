/*
 * connected_lsmask_generation_algorithm.hpp
 *
 *  Created on: May 24, 2016
 *      Author: thomasriddick
 */

#ifndef INCLUDE_CONNECTED_LSMASK_GENERATION_ALGORITHM_HPP_
#define INCLUDE_CONNECTED_LSMASK_GENERATION_ALGORITHM_HPP_

#include "cell.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "coords.hpp"
#include <queue>

class create_connected_landsea_mask{

public:
	~create_connected_landsea_mask();
	void setup_flags(bool use_diagonals_in);
	void setup_fields(bool* landsea_in, bool* ls_seed_points_in,
					  grid_params* grid_params_in);
	void generate_connected_mask();
	void add_ls_seed_points_to_q();
	void process_neighbors();
	void process_neighbor();
	void deep_copy_completed_cells_to_landsea();

private:
	queue<landsea_cell*> q;
	coords* center_coords;
	grid_params* _grid_params = nullptr;
	grid* _grid = nullptr;
	vector<coords*>* neighbors_coords = nullptr;
	field<bool>* landsea = nullptr;
	field<bool>* completed_cells = nullptr;
	field<bool>* ls_seed_points = nullptr;
	bool use_diagonals = true;

	int diagonal_neighbors = 0;
};

#endif /* INCLUDE_CONNECTED_LSMASK_GENERATION_ALGORITHM_HPP_ */
