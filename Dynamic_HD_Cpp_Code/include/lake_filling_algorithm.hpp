/*
 * lake_filling_algorithm.hpp
 *
 *  Created on: Feb 21, 2018
 *      Author: thomasriddick
 */

#include <queue>
#include "cell.hpp"

class lake_filling_algorithm {
public:
	~lake_filling_algorithm();
	void setup_flags(bool use_highest_possible_lake_water_level_in);
	void setup_fields(bool* lake_minima_in,bool* lake_mask_in,
					  double* orography_in,grid_params* grid_params_in);
	void fill_lakes();
private:
	void fill_lake();
	void add_lake_minima_to_queue();
	void adjust_lake_height(int lake_number,double height);
	void process_neighbors();
	void process_neighbor();
	queue<cell*> minima_q;
	queue<cell*> q;
	grid_params* _grid_params = nullptr;
	grid* _grid = nullptr;
	field<bool>* completed_cells = nullptr;
	field<bool>* lake_minima = nullptr;
	field<bool>* lake_mask = nullptr;
	field<double>* orography = nullptr;
	field<int>* lake_numbers = nullptr;
	cell* minima = nullptr;
	cell* center_cell = nullptr;
	coords* center_coords = nullptr;
	vector<coords*>* neighbors_coords = nullptr;
	bool use_highest_possible_lake_water_level;
	int lake_number;
};
