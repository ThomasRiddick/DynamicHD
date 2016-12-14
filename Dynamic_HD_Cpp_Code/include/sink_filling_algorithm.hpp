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
#include "field.hpp"
#include "priority_cell_queue.hpp"
using namespace std;

/*It would have been preferable to include the names of parameters in the function declarations
 *but unfortunately this was not done*/

class sink_filling_algorithm{

public:
	sink_filling_algorithm(){}
	sink_filling_algorithm(field<double>*,int, int, field<bool>*,bool*,
			 	 	 	   bool,bool* = nullptr);
	virtual ~sink_filling_algorithm();
	void setup_flags(bool, bool = false);
	void setup_fields(double*,bool*,bool*,int,int);
	void fill_sinks();
	void test_add_edge_cells_to_q() { add_edge_cells_to_q(); }
	void test_add_true_sinks_to_q() { add_true_sinks_to_q(); }
	priority_cell_queue get_q() {return q; }
	virtual int get_method() { return method; }

protected:
	//A variable equal to the smallest possible double used as a non data value
	const double no_data_value = numeric_limits<double>::lowest();

	//Process the neighbors of the current cell
	void process_neighbors(vector<integerpair*>*);
	//Handles the initial setup of the queue
	void add_edge_cells_to_q();
	void add_true_sinks_to_q();
	virtual void process_neighbor() {}
	virtual void process_center_cell() {}
	virtual void push_land_sea_neighbor() {}
	virtual void push_vertical_edge(int) {};
	virtual void push_horizontal_edge(int) {};
	virtual void push_true_sink(int,int) {};
	virtual void set_ls_as_no_data(int i, int j) {};
	virtual void process_true_sink_center_cell() {};
	virtual void push_neighbor() {};

	priority_cell_queue q;
	integerpair center_coords;
	cell* center_cell = nullptr;
	field<double>* orography = nullptr;
	field<bool>* landsea = nullptr;
	field<bool>* true_sinks = nullptr;
	field<bool>* completed_cells = nullptr;
	bool debug = false;
	bool set_ls_as_no_data_flag = false;
	double nbr_orog = 0.0;
	int lat = 0;
	int lon = 0;
	int nlat = 0;
	int nlon = 0;
	int nbr_lat = 0;
	int nbr_lon = 0;
	int method = 0;

};

class sink_filling_algorithm_1 : public sink_filling_algorithm {

public:
	sink_filling_algorithm_1(){}
	sink_filling_algorithm_1(field<double>*, int, int, field<bool>*, bool*, bool,
					         bool* = nullptr);
	~sink_filling_algorithm_1() {}
	int get_method() {return method;}

protected:
	const int method = 1;
	void process_neighbor();
	void set_ls_as_no_data(int, int);
	void push_land_sea_neighbor();
	void push_vertical_edge(int);
	void push_horizontal_edge(int);
	void push_true_sink(int,int);
	void process_center_cell();
	void process_true_sink_center_cell();
	void push_neighbor();
	double center_orography = 0.0;

};

class sink_filling_algorithm_4 : public sink_filling_algorithm {

public:
	sink_filling_algorithm_4(){}
	sink_filling_algorithm_4(field<double>*, int, int, field<double>*, field<bool>*, bool*,
							 bool, field<int>*, bool, bool* = nullptr);
	~sink_filling_algorithm_4() { delete rdirs; delete catchment_nums; }
	int get_method() {return method;}
	void setup_flags(bool, bool = false, bool = false);
	void setup_fields(double*, bool*, bool*, int, int, double*, int*);
	double test_find_initial_cell_flow_direction(int, int, int, int, field<double>*,
												 field<bool>*, bool = false);
	double test_calculate_direction_from_neighbor_to_cell(int, int, int, int, int, int);

protected:
	const int method = 4;
	void process_neighbor();
	void push_land_sea_neighbor();
	//Calculate the direction from a neighbor to a central cell (used by algorithm 4)
	double calculate_direction_from_neighbor_to_cell(int,int);
	void set_ls_as_no_data(int,int);
	void process_center_cell();
	void process_true_sink_center_cell();
	void push_vertical_edge(int);
	void push_horizontal_edge(int);
	void push_true_sink(int,int);
	//Find the flow direction of an initial cell for algorithm 4
	double find_initial_cell_flow_direction();
	void push_neighbor();

	field<double>* rdirs = nullptr;
	field<int>* catchment_nums = nullptr;
	bool prefer_non_diagonal_initial_dirs = false;
	int center_catchment_num = 0;

};

#endif /* INCLUDE_SINK_FILLING_ALGORITHM_HPP_ */
