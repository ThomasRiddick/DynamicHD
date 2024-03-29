/*
 * carved_river_direction_burning_algorithm.hpp
 *
 *  Created on: Feb 7, 2018
 *      Author: thomasriddick
 */

#ifndef INCLUDE_CARVED_RIVER_DIRECTIONS_BURNING_ALGORITHM_HPP_
#define INCLUDE_CARVED_RIVER_DIRECTIONS_BURNING_ALGORITHM_HPP_

#include <queue>
#include <stack>
#include "base/coords.hpp"
#include "base/field.hpp"
#include "base/grid.hpp"
#include "base/cell.hpp"
#include "base/priority_cell_queue.hpp"

class carved_river_direction_burning_algorithm {
public:
	carved_river_direction_burning_algorithm() {};
	virtual ~carved_river_direction_burning_algorithm();
	void setup_fields(double* orography_in,bool* minima_in,bool* lakemask_in,
					  grid_params* grid_params_in);
	void setup_flags(bool add_slope_in = false,int max_exploration_range_in = 0,
	                 double regular_minimum_height_change_threshold_in = 0.0,
	                 int short_path_threshold_in = 0,
	                 double short_minimum_height_change_threshold_in = 0.0);
	void burn_carved_river_directions();
protected:
	void add_minima_to_q();
	void reprocess_path();
	virtual coords* get_next_cell_downstream(coords* initial_coords) = 0;
	priority_cell_queue q;
	queue<coords*> reprocessing_q;
	grid_params* _grid_params = nullptr;
	grid* _grid = nullptr;
	field<double>* orography = nullptr;
	field<bool>* minima = nullptr;
	field<bool>* lakemask = nullptr;
	coords*  working_cell_coords = nullptr;
	double minima_height;
	bool add_slope = false;
	int max_exploration_range = 0;
	double regular_minimum_height_change_threshold = 0.0;
	int short_path_threshold = 0;
	double short_minimum_height_change_threshold = 0.0;
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

class carved_river_direction_burning_algorithm_icon_single_index : public carved_river_direction_burning_algorithm {
public:
	carved_river_direction_burning_algorithm_icon_single_index() {};
	virtual ~carved_river_direction_burning_algorithm_icon_single_index() {delete next_cell_index;};
protected:
	coords* get_next_cell_downstream(coords* initial_coords);
	field<double>* next_cell_index = nullptr;
};

#endif /* INCLUDE_REDUCE_CONNECTED_AREAS_TO_POINTS_HPP_ */
