/*
 * catchment_computation_algorithm.hpp
 *
 *  Created on: Feb 23, 2018
 *      Author: thomasriddick
 */

#include <queue>
#include <cell.hpp>

class catchment_computation_algorithm {
public:
	virtual ~catchment_computation_algorithm()
		{delete completed_cells; delete catchment_numbers;};
	void setup_fields(int* catchment_numbers_in,
					  grid_params* grid_params_in);
	void compute_catchments();
private:
	void process_neighbors();
	void process_neighbor();
	void compute_catchment();
	void add_outflows_to_queue();
	virtual bool check_if_neighbor_is_upstream();
	virtual bool check_for_outflow(coords* cell_coords);
	queue<landsea_cell*> outflow_q;
	queue<landsea_cell*> q;
	grid_params* _grid_params = nullptr;
	grid* _grid = nullptr;
	field<bool>* completed_cells = nullptr;
	field<int>* catchment_numbers = nullptr;
	landsea_cell* outflow;
	landsea_cell* center_cell;
	coords* center_coords;
	coords* nbr_coords;
	vector<coords*>* neighbors_coords;
	int catchment_number;
};

class catchment_computation_algorithm_latlon : public catchment_computation_algorithm {
public:
	virtual ~catchment_computation_algorithm_latlon() {delete rdirs;};
	void setup_fields(int* catchment_numbers_in,double* rdirs_in,
					  grid_params* grid_params_in);
private:
	bool check_if_neighbor_is_upstream();
	bool check_for_outflow(coords* cell_coords);
	field<double>* rdirs = nullptr;
};
