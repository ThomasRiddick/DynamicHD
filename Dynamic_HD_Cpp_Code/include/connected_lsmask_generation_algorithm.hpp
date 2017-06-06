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

/**
 * Class containing mid-level code of connected landsea mask generation
 * algorithm
 */

class create_connected_landsea_mask{

public:
	///Class destructor
	~create_connected_landsea_mask();
	///Setup functions
	void setup_flags(bool use_diagonals_in);
	///Setup fields
	void setup_fields(bool* landsea_in, bool* ls_seed_points_in,
					  grid_params* grid_params_in);
	///Main function of connected landsea mask generation routine
	void generate_connected_mask();
	///Add the starting seed points (in the middle of ocean that are
	///required) to the q
	void add_ls_seed_points_to_q();
	///Process the neighbors of a particular cell
	void process_neighbors();
	///Process a particular neighbor of a cell
	void process_neighbor();
	///Make a deep copy of the list of completed cells to the output
	///land-sea array
	void deep_copy_completed_cells_to_landsea();

private:
	///The main queue - this isn't a priority queue just a normal queue...
	///a priority queue is not needed for connected land-sea mask production
	queue<landsea_cell*> q;
	///Coordinates of the central cell being processed
	coords* center_coords;
	///A input grid_parameter object to specify what kind of grid is required
	grid_params* _grid_params = nullptr;
	///The grid created from the input grid parameters
	grid* _grid = nullptr;
	///The coordinates of the neighboring cell current being processed
	vector<coords*>* neighbors_coords = nullptr;
	///A field containing first the original landsea mask which is then
	///replaced by the new landsea mask at the end of the algorithm
	field<bool>* landsea = nullptr;
	///A field containing the cells that have already been processed,
	///which slowly becomes the new landsea mask
	field<bool>* completed_cells = nullptr;
	///Point to start the sink filling algorithm from, at least one in
	///the center of every unconnected body of water
	field<bool>* ls_seed_points = nullptr;
	///Flag controlling whether to consider a diagonal neighbor directly
	///connected to a cell or not
	bool use_diagonals = true;

	///Number of a diagonal neighbor has (which will be 4 for an ordinary
	///latitude-longitude grid but not necessarily so for other grid)
	int diagonal_neighbors = 0;
};

#endif /* INCLUDE_CONNECTED_LSMASK_GENERATION_ALGORITHM_HPP_ */
