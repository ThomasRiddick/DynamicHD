/*
 * carved_river_direction_burning_algorithm.hpp
 *
 *  Created on: Feb 7, 2018
 *      Author: thomasriddick
 */

#ifndef INCLUDE_CARVED_RIVER_DIRECTIONS_BURNING_ALGORITHM_HPP_
#define INCLUDE_CARVED_RIVER_DIRECTIONS_BURNING_ALGORITHM_HPP_

#include <queue>
#include "coords.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "cell.hpp"
#include "priority_cell_queue.hpp"

class carved_river_direction_burning_algorithm {
public:
	carved_river_direction_burning_algorithm() {};
	virtual ~carved_river_direction_burning_algorithm();
	void setup_fields(double* orography_in,bool* minima_in,bool* lakemask_in,
					  grid_params* grid_params_in);
	void burn_carved_river_directions();
protected:
	void add_minima_to_q();
	virtual coords* get_next_cell_downstream(coords* initial_coords) = 0;
	priority_cell_queue q;
	grid_params* _grid_params = nullptr;
	grid* _grid = nullptr;
	field<double>* orography = nullptr;
	field<bool>* minima = nullptr;
	field<bool>* lakemask = nullptr;
};

class carved_river_direction_burning_algorithm_latlon : public carved_river_direction_burning_algorithm {
public:
	carved_river_direction_burning_algorithm_latlon() {};
	virtual ~carved_river_direction_burning_algorithm_latlon() {delete rdirs;};
	void setup_fields(double* orography_in, double* rdirs_in,bool* minima_in,bool* lakemask_in,
					  grid_params* grid_params_in);
protected:
	coords* get_next_cell_downstream(coords* initial_coords);
	field<double>* rdirs = nullptr;
};

#endif /* INCLUDE_REDUCE_CONNECTED_AREAS_TO_POINTS_HPP_ */
