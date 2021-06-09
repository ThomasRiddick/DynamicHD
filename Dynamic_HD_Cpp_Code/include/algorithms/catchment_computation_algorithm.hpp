/*
 * catchment_computation_algorithm.hpp
 *
 *  Created on: Feb 23, 2018
 *      Author: thomasriddick
 */

#include <queue>
#include <tuple>
#include "base/cell.hpp"
#include "base/grid.hpp"
#include "base/field.hpp"

class catchment_computation_algorithm {
public:
	virtual ~catchment_computation_algorithm()
		{delete completed_cells; delete catchment_numbers; delete _grid;
		 delete searched_cells; };
	void setup_fields(int* catchment_numbers_in,
					  grid_params* grid_params_in);
	void compute_catchments();
	void renumber_catchments_by_size();
	vector<int>* identify_loops();
	void test_compute_catchment(landsea_cell* outflow_in,
	                            int catchmentnumber_in);
protected:
	void process_neighbors();
	void process_neighbor();
	void compute_catchment();
	void add_outflows_to_queue();
	void add_loop_cells_to_queue();
	void find_loop_in_catchment();
	virtual bool check_if_neighbor_is_upstream() = 0;
	virtual bool check_for_outflow(coords* cell_coords) = 0;
	virtual bool check_for_loops(coords* cell_coords) = 0;
	virtual coords* calculate_downstream_coords(coords* initial_coords) = 0;
	queue<landsea_cell*> outflow_q;
	queue<landsea_cell*> q;
	vector<pair<int,int> >* catchment_sizes = nullptr;
	grid_params* _grid_params = nullptr;
	grid* _grid = nullptr;
	field<bool>* completed_cells = nullptr;
	field<bool>* searched_cells = nullptr;
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
protected:
	bool check_if_neighbor_is_upstream();
	bool check_for_outflow(coords* cell_coords);
	bool check_for_loops(coords* cell_coords);
	coords* calculate_downstream_coords(coords* initial_coords);
	field<double>* rdirs = nullptr;
};

class catchment_computation_algorithm_icon_single_index :
	public catchment_computation_algorithm {
public:
	virtual ~catchment_computation_algorithm_icon_single_index() {delete next_cell_index;};
	void setup_fields(int* catchment_numbers_in,int* next_cell_index_in,
		  	  	  	  	grid_params* grid_params_in);
	bool check_for_outflow(coords* cell_coords);
	bool check_for_loops(coords* cell_coords);
	bool check_if_neighbor_is_upstream();
	coords* calculate_downstream_coords(coords* initial_coords);
	field<int>* next_cell_index = nullptr;
	const int true_sink_value = -5;
	const int outflow_value = -1;
	const int ocean_value   = -2;
};
