/*
 * reduceconnectedareatopoint.h
 *
 *  Created on: Feb 7, 2018
 *      Author: thomasriddick
 */

#ifndef INCLUDE_REDUCE_CONNECTED_AREAS_TO_POINTS_ALGORITHM_HPP_
#define INCLUDE_REDUCE_CONNECTED_AREAS_TO_POINTS_ALGORITHM_HPP_

#include <queue>
#include "coords.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "cell.hpp"

class reduce_connected_areas_to_points_algorithm {
public:
	reduce_connected_areas_to_points_algorithm();
	~reduce_connected_areas_to_points_algorithm();
	void setup_flags(bool use_diagonals_in);
	void setup_fields(bool* areas_in,
					  grid_params* grid_params_in);
	void iterate_over_field();
	void reduce_connected_area_to_point(coords* point);
	void process_initial_point(coords* point);
	void process_neighbors();
	void process_neighbor();
private:
	queue<landsea_cell*> q;
	coords* center_coords;
	grid_params* _grid_params = nullptr;
	grid* _grid = nullptr;
	vector<coords*>* neighbors_coords = nullptr;
	field<bool>* areas = nullptr;
	field<bool>* completed_cells = nullptr;
	bool use_diagonals = true;
	int diagonal_neighbors = 0;
};

#endif /* INCLUDE_REDUCE_CONNECTED_AREAS_TO_POINTS_ALGORITHM_HPP_ */
