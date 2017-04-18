/*
 * sink_filling_algorithm.hpp
 *
 *  Created on: May 20, 2016
 *      Author: thomasriddick
 */

#ifndef INCLUDE_SINK_FILLING_ALGORITHM_HPP_
#define INCLUDE_SINK_FILLING_ALGORITHM_HPP_

#include <queue>
#include <limits>
#include <cmath>
#include "grid.hpp"
#include "field.hpp"
#include "priority_cell_queue.hpp"
using namespace std;

/*It would have been preferable to include the names of parameters in the function declarations
 *but unfortunately this was not done*/

class sink_filling_algorithm{

public:
	sink_filling_algorithm(){};
	sink_filling_algorithm(field<double>*,grid_params*, field<bool>*,bool*,bool,
						   bool* = nullptr, field<int>* = nullptr);
	virtual ~sink_filling_algorithm();
	void setup_flags(bool, bool = false, int = 1, double = 1.1, bool = false,
					 bool = false);
	void setup_fields(double*,bool*,bool*,grid_params*);
	void fill_sinks();
	void test_add_edge_cells_to_q() { add_edge_cells_to_q(); }
	void test_add_true_sinks_to_q() { add_true_sinks_to_q(); }
	priority_cell_queue get_q() {return q; }
	virtual int get_method() = 0;
	double tarasov_get_area_height() { return tarasov_area_height; }
	static double get_no_data_value() { return no_data_value; }

protected:

	static double no_data_value;
	//Process the neighbors of the current cell
	void process_neighbors(vector<coords*>*);
	//Handles the initial setup of the queue
	void add_edge_cells_to_q();
	void add_true_sinks_to_q();
	void add_landsea_edge_cells_to_q();
	void tarasov_calculate_neighbors_path_length();
	virtual void add_geometric_edge_cells_to_q() = 0;
	virtual void process_neighbor() = 0;
	virtual void process_center_cell() = 0;
	virtual void push_land_sea_neighbor() = 0;
	virtual void push_true_sink(coords*) = 0;
	virtual void set_ls_as_no_data(coords*) = 0;
	virtual void process_true_sink_center_cell() = 0;
	virtual void push_neighbor() = 0;
	virtual double tarasov_calculate_neighbors_path_length_change(coords*) = 0;
	void tarasov_get_center_cell_values();
	void tarasov_set_field_values(coords*);
	void tarasov_get_center_cell_values_from_field();
	void tarasov_update_maximum_separation_from_initial_edge();
	bool tarasov_is_shortest_permitted_path();
	bool tarasov_same_edge_criteria_met();
	virtual void tarasov_set_area_height() = 0;

	priority_cell_queue q;
	coords* center_coords = nullptr;
	coords* nbr_coords = nullptr;
	cell* center_cell = nullptr;
	grid_params* _grid_params = nullptr;
	grid* _grid = nullptr;
	field<double>* orography = nullptr;
	field<bool>* landsea = nullptr;
	field<bool>* true_sinks = nullptr;
	field<bool>* completed_cells = nullptr;
	field<double>* tarasov_path_initial_heights = nullptr;
	field<bool>* tarasov_landsea_neighbors = nullptr;
	field<bool>* tarasov_active_true_sink = nullptr;
	field<double>* tarasov_path_lengths = nullptr;
	field<int>* tarasov_maximum_separations_from_initial_edge = nullptr;
	field<int>* tarasov_initial_edge_nums = nullptr;
	//Used by Algorithm 4. Also used by both Algorithm when performing
	//orography upscaling
	field<int>* catchment_nums = nullptr;
	bool debug = false;
	bool set_ls_as_no_data_flag = false;
	double nbr_orog = 0.0;
	//Used by Algorithm 4. Also used by both Algorithm when performing
	//orography upscaling
	int center_catchment_num = 0;
	double tarasov_min_path_length = 1.0;
	double tarasov_area_height = no_data_value;
	bool tarasov_include_corners_in_same_edge_criteria = false;
	bool tarasov_reprocessing_cell = false;
	bool tarasov_mod = false;
	int tarasov_separation_threshold_for_returning_to_same_edge = 1;
	int method = 0;
	double tarasov_neighbor_path_length = 0.0;
	double tarasov_path_initial_height = 0.0;
	double tarasov_center_cell_path_initial_height = 0.0;
	double tarasov_center_cell_path_length = 0.0;
	int tarasov_center_cell_maximum_separations_from_initial_edge = 0;
	int tarasov_center_cell_initial_edge_num = 0;

};



class sink_filling_algorithm_latlon :  virtual public sink_filling_algorithm {
public:
	sink_filling_algorithm_latlon() {};
	void add_geometric_edge_cells_to_q();
	virtual void push_vertical_edge(int, bool = true, bool = true) = 0;
	virtual void push_horizontal_edge(int, bool = true, bool = true) = 0;
	virtual ~sink_filling_algorithm_latlon() {};
	double tarasov_calculate_neighbors_path_length_change(coords*);
protected:
	int nlat = 0;
	int nlon = 0;
};

class sink_filling_algorithm_1 : virtual public sink_filling_algorithm {

public:
	sink_filling_algorithm_1(){}
	sink_filling_algorithm_1(field<double>*, grid_params*, field<bool>*, bool*, bool,
							 bool, double, bool* = nullptr);
	void setup_flags(bool, bool = false, bool = false, bool = false, double = 0.1,
					 int = 1, double = 1.1, bool = false);
	virtual ~sink_filling_algorithm_1() {};
	int get_method() {return method;}

protected:
	const int method = 1;
	void process_neighbor();
	void set_ls_as_no_data(coords*);
	void push_land_sea_neighbor();
	void push_true_sink(coords*);
	void process_center_cell();
	void process_true_sink_center_cell();
	void push_neighbor();
	void tarasov_set_area_height();
	bool add_slope = false;
	double center_orography = 0.0;
	double epsilon = 0.1; //normally in meters

};

class sink_filling_algorithm_4 : virtual public sink_filling_algorithm {

public:
	sink_filling_algorithm_4(){}
	sink_filling_algorithm_4(field<double>*, grid_params*, field<bool>*, bool*,
							 bool, field<int>*, bool, bool* = nullptr);
	virtual ~sink_filling_algorithm_4() { if (not tarasov_mod) delete catchment_nums; }
	int get_method() {return method;}
protected:
	const int method = 4;
	void process_neighbor();
	void push_land_sea_neighbor();
	void set_ls_as_no_data(coords*);
	void process_center_cell();
	void process_true_sink_center_cell();
	void push_true_sink(coords*);
	void push_neighbor();
	void find_initial_cell_flow_direction();
	void setup_fields(double*, bool*, bool*, grid_params*, int*);
	void setup_flags(bool, bool = false, bool = false, int = 1, double = 1.1, bool = false,
			 	 	 bool = false);
	void tarasov_set_area_height();
	virtual void set_cell_to_no_data_value(coords*) = 0;
	virtual void set_cell_to_true_sink_value(coords*) = 0;
	virtual void set_index_based_rdirs(coords*,coords*) = 0;

	bool prefer_non_diagonal_initial_dirs = false;
	double nbr_rim_height = 0.0;
	double center_rim_height = 0.0;
	bool index_based_rdirs_only = true;
};

class sink_filling_algorithm_1_latlon : public sink_filling_algorithm_1, public sink_filling_algorithm_latlon{
	void push_vertical_edge(int,bool,bool);
	void push_horizontal_edge(int,bool,bool);
public:
	sink_filling_algorithm_1_latlon() {};
	sink_filling_algorithm_1_latlon(field<double>*, grid_params*, field<bool>*, bool*, bool,
					         	 	bool = false, double = 0.1, bool* = nullptr);
	void setup_fields(double*, bool*,bool*,grid_params*);
	virtual ~sink_filling_algorithm_1_latlon() {};
};

class sink_filling_algorithm_4_latlon : public sink_filling_algorithm_4, public sink_filling_algorithm_latlon{
	field<double>* rdirs = nullptr;
	field<int>* next_cell_lat_index = nullptr;
	field<int>* next_cell_lon_index = nullptr;
	const int true_sink_value = -5;
	const int no_data_value = -1;
	//Find the flow direction of an initial cell for algorithm 4
	void push_vertical_edge(int,bool,bool);
	void push_horizontal_edge(int,bool,bool);
	void set_cell_to_no_data_value(coords*);
	void set_cell_to_true_sink_value(coords*);
public:
	sink_filling_algorithm_4_latlon() {};
	sink_filling_algorithm_4_latlon(field<double>*, grid_params*, field<bool>*, bool*,
							 	 	bool, field<int>*, bool, bool, field<int>*,field<int>*,
									bool* = nullptr, field<double>* = nullptr);
	void set_dir_based_rdir(coords*, double);
	void setup_flags(bool, bool = false, bool = false, bool = false, bool = false, int = 1,
				     double = 1.1, bool = false);
	void setup_fields(double*, bool*, bool*, int*, int*, grid_params*, double*, int*);
	void calculate_direction_from_neighbor_to_cell();
	virtual ~sink_filling_algorithm_4_latlon() { delete rdirs; delete next_cell_lat_index;
												 delete next_cell_lon_index; };
	//Calculate the direction from a neighbor to a central cell (used by algorithm 4)
	double test_find_initial_cell_flow_direction(coords*,grid_params*,field<double>*,
												 field<bool>*, bool = false);
	double test_calculate_direction_from_neighbor_to_cell(coords*,coords*,grid_params*);
	void set_index_based_rdirs(coords*,coords*);
};

#endif /* INCLUDE_SINK_FILLING_ALGORITHM_HPP_ */
