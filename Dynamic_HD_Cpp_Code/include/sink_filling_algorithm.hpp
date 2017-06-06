/*
 * sink_filling_algorithm.hpp
 *
 *  Created on: May 20, 2016
 *      Author: thomasriddick
 */

#ifndef INCLUDE_SINK_FILLING_ALGORITHM_HPP_
#define INCLUDE_SINK_FILLING_ALGORITHM_HPP_

/*! \mainpage
 * This code contains two variants of a priority flood based sink filling algorithm
 * and also a orography upscaling algorithm that can use either sink filling
 * algorithm. The main routines are defined in \ref sink_filling_algorithm supported
 * by the \ref field class which itself uses the \ref grid class. Coordinates are
 * handled generically by the \ref coords class and all the information on cells is
 * bundled into instances of the \ref cell class.
 *
 * All of the four possible algorithms (algorithms 1/4 and orography upscaling using
 * algorithm 1/4 as a base) follow the same basic structure based around a priority
 * queue.
 * 1. An initial queue and completed cells field is prepared by adding 'edges' to the
 * queue on a cell by cell basis. These could be true sinks, cell neighboring cell
 * points or in other situations the geometric edges of the grid.
 * 2. The highest priority (lowest orography) cell on the queue is processed.
 * 	1. A list of the appropriate neighbor of the cell to process is formed. For
 * 	the sink filling algorithm this is the unprocessed cells; for orography
 * 	upscaling this also includes processed cells with a higher starting height
 * 	than the center cell
 * 	2. For each neighbor on the list push the neighbor to the queue itself
 * 3. For sink filling algorithm continue till the queue finally empties and
 * every point is processed. For orography upscaling continue till a path reaches
 * an edge and meets the criteria for being a valid path and record the highest
 * point the path has passed over (this is recorded in the cell record and propagated
 * along the path) to as the new height of this cell on the course grid.
 *
 * The information stored in cell varies from algorithm to algorithm. For a sink filling
 * all that is in principle needed is a height. For algorithm 4, the carving algorithm it
 * is useful but not necessary to store also the highest height crossed and the catchment
 * number. Orography upscaling requires the highest height crossed, the farthest distance
 * traveled from the initial edge, the ID of the initial edge and the total length of the
 * path and the height of the paths starting point.
 */

#include <queue>
#include <limits>
#include <cmath>
#include "grid.hpp"
#include "field.hpp"
#include "priority_cell_queue.hpp"
using namespace std;

/*It would have been preferable to include the names of parameters in the function declarations
 *but unfortunately this was not done*/

/* Comments are optimized for Doxygen... this means slightly more comments than are
 * appears necessary if you are reading the code itself
 */

/**The abstract base class for running the sink filling and Tarasov upscaling algorithms. Has
 *both abstract sub-classes for different grid types and abstract subclasses for the different
 *sink filling algorithm possible. Concrete sub classes for a particular grid and algorithm inherit
 *sink from both of subclass types (multiple inheritance in a diamond pattern - the diamond problem).
 *This is solved by the standard technique of using virtual inheritance. The Tarasov-style upscaling
 *code is on a logical switch and doesn't not use further virtualization. The Tarasov-style upscaling
 *code can be used with any algorithm though it has only be tested with algorithm 1
 */

class sink_filling_algorithm{

public:
	///Constructor
	sink_filling_algorithm(){};
	///Constructor
	sink_filling_algorithm(field<double>*,grid_params*, field<bool>*,bool*,bool,
						   bool* = nullptr, field<int>* = nullptr);
	virtual ~sink_filling_algorithm();
	///Setup the necessary flags and parameters
	void setup_flags(bool, bool = false, int = 1, double = 1.1, bool = false,
					 bool = false);
	///Setup the necessary field given an input set of grid_params and 1D fields
	void setup_fields(double*,bool*,bool*,grid_params*);
	///Run the sink filling algorithm
	void fill_sinks();
	///Public wrapper for testing the protected add_edge_cell_to_q function
	void test_add_edge_cells_to_q(bool);
	///Public wrapper for testing the protected add_true_sinks_to_q function
	void test_add_true_sinks_to_q(bool);
	///Getter
	priority_cell_queue get_q() {return q; }
	///Getter
	virtual int get_method() = 0;
	///Getter
	double tarasov_get_area_height() { return tarasov_area_height; }
	///Getter
	static double get_no_data_value() { return no_data_value; }

protected:

	///Value to use to indicate no data
	static double no_data_value;
	///Process the neighbors of the current cell
	void process_neighbors(vector<coords*>*);
	///Handles the initial setup of the queue
	void add_edge_cells_to_q();
	///Handles adding potential true sinks to the queue
	void add_true_sinks_to_q();
	///Handles adding sea points to the queue
	void add_landsea_edge_cells_to_q();
	///Calculate the path length of a neighbor from the central
	///cells path and the river direction.
	void tarasov_calculate_neighbors_path_length();
	///Add the actual vertical and horizontal edges of a grid to the
	///initial queue.
	virtual void add_geometric_edge_cells_to_q() = 0;
	///Process the neighbor of the current center cell
	virtual void process_neighbor() = 0;
	///Process the current center cell
	virtual void process_center_cell() = 0;
	///Add a particular landsea neighbor the queue
	virtual void push_land_sea_neighbor() = 0;
	///Add a particular true sink to the queue
	virtual void push_true_sink(coords*) = 0;
	///Set sea points to the no data value
	virtual void set_ls_as_no_data(coords*) = 0;
	///Process a center cell that contains a true sink
	virtual void process_true_sink_center_cell() = 0;
	///Add a neighbor to the queue
	virtual void push_neighbor() = 0;
	///All variables below are only used by tarasov-style orography upscaling code
	///Calculate the change in path length from a neighbor to the central cell
	///(which is then used by tarasov_claculate_neigbors_path_length).
	virtual double tarasov_calculate_neighbors_path_length_change(coords*) = 0;
	///Load the values need for processing a center cell
	void tarasov_get_center_cell_values();
	///Set the field values necessary for when a true sink is passed over
	void tarasov_set_field_values(coords*);
	///Load the center cell values from the underlying field instead of using
	///the version that came from the cell on the queue (which is not correct
	///because the cell was a true sink).
	void tarasov_get_center_cell_values_from_field();
	///Update the maximum distance traveled from initial edge if that has increased,
	///otherwise leave it unchanged of course
	void tarasov_update_maximum_separation_from_initial_edge();
	///Check if a path that has reach an edge (and therefore is the path going over
	///the lowest height required - here called the shortest path but really meaning
	///the lowest path) fulfills.
	bool tarasov_is_shortest_permitted_path();
	///If a path has returned to the same edge check if it has traveled far enough
	///from the edge during it journey, if so return true else return false.
	///Also return true if it has gone to another edge or a true sink/landsea point
	///(unless connecting two landsea points or sinks in which case return false)
	bool tarasov_same_edge_criteria_met();
	///Set the value for height of area to height of highest point along shortest/
	///lowest valid path
	virtual void tarasov_set_area_height() = 0;

	///Main priority queue
	priority_cell_queue q;
	///Coordinates of center cell being processed
	coords* center_coords = nullptr;
	///Coordinates of neighboring cell being processed
	coords* nbr_coords = nullptr;
	///The center cell being processed
	cell* center_cell = nullptr;
	///Parameter of the grid to use
	grid_params* _grid_params = nullptr;
	///The grid of the fields used
	grid* _grid = nullptr;
	///The orography to use
	field<double>* orography = nullptr;
	///The landsea mask field to use
	field<bool>* landsea = nullptr;
	///The true sinks field to use
	field<bool>* true_sinks = nullptr;
	///A field to keep track of completed cells
	field<bool>* completed_cells = nullptr;
	///The remaining field are used only by the tarasov-style orography
	///upscaling code
	///Initial height path started from... this allows reversing direction
	///of path when it meets another path that started from a lower point
	field<double>* tarasov_path_initial_heights = nullptr;
	///Set of cells that neighbor a landsea cell
	field<bool>* tarasov_landsea_neighbors = nullptr;
	///Set of true sinks that are actually in a sink
	field<bool>* tarasov_active_true_sink = nullptr;
	///Field of length of the current path that is a passing through at each point
	field<double>* tarasov_path_lengths = nullptr;
	///Field of maximum separation from initial edge of path that is passing through at
	///each point
	field<int>* tarasov_maximum_separations_from_initial_edge = nullptr;
	///Field of initial edge number of path that is passing through at each point
	field<int>* tarasov_initial_edge_nums = nullptr;
	///Used by Algorithm 4. Also used by both algorithms when performing
	///orography upscaling
	///Field of the number assign to the catchment at each point
	field<int>* catchment_nums = nullptr;
	///Print debug printout
	bool debug = false;
	///Flag to switch on setting sea points as no data
	bool set_ls_as_no_data_flag = false;
	///Height of a neighbor
	double nbr_orog = 0.0;
	///The catchment number of the center cell
	///Used by Algorithm 4. Also used by both Algorithm when performing
	///orography upscaling
	int center_catchment_num = 0;
	///The algorithm in use. Zero is a dummy value; this is changed to
	///the correct value when a subclass is setup
	int method = 0;
	///The remaining variables are only used in Tarasov-style orography
	///upscaling
	///Minimum length of a path to accept
	double tarasov_min_path_length = 1.0;
	///The height to set the course cell being processed to
	double tarasov_area_height = no_data_value;
	///Are corner count as part of an edge or not for the
	///returning to the same edge criterion
	bool tarasov_include_corners_in_same_edge_criteria = false;
	///Has this cell being processed before and is now being
	///reprocessed as part of second path with a lower start
	bool tarasov_reprocessing_cell = false;
	///This is the master switch that switches on using
	///Tarasov style upscaling rather than a normal sink-filling/carving
	///algorithm
	bool tarasov_mod = false;
	///For a path return to the same edge to be counted it needs to of
	///at some point in its journey been further from the that edge
	///than this threshold
	int tarasov_separation_threshold_for_returning_to_same_edge = 1;
	///Path length of a neighbor
	double tarasov_neighbor_path_length = 0.0;
	///Initial height of a path
	double tarasov_path_initial_height = 0.0;
	///Initial height of the center cell being processed
	double tarasov_center_cell_path_initial_height = 0.0;
	///Path length of the initial cell being processed
	double tarasov_center_cell_path_length = 0.0;
	///Greatest separation of path of center cell from its initial edge
	int tarasov_center_cell_maximum_separations_from_initial_edge = 0;
	///Initial edge number of the path of the center cell
	int tarasov_center_cell_initial_edge_num = 0;
};

/**
 * Subclass of sink_filling_algorithm for a latitude longitude grids
 */


class sink_filling_algorithm_latlon :  virtual public sink_filling_algorithm {
public:
	///Constructor
	sink_filling_algorithm_latlon() {};
	///Implement concrete version of this for latitude longitude grid
	void add_geometric_edge_cells_to_q();
	///Push the two vertical cell edges onto the queue
	virtual void push_vertical_edge(int, bool = true, bool = true) = 0;
	///Push the two horizontal cell edges onto the queue
	virtual void push_horizontal_edge(int, bool = true, bool = true) = 0;
	///Destructor
	virtual ~sink_filling_algorithm_latlon() {};
	///Only used by Tarasov style orography upscaling code; calculate
	///change in path length from center cell to neighbor
	double tarasov_calculate_neighbors_path_length_change(coords*);
protected:
	///Number of latitude points in grid
	int nlat = 0;
	///Number of longitude points in grid
	int nlon = 0;
};

/**
 * A normal priority flood sink filling algorithm likes algorithms 1/3 of Barnes et al 2013
 * (Based on algorithm 1 of this paper but can add a small slope/epsilon factor to slopes like
 * algorithm 3
 */

class sink_filling_algorithm_1 : virtual public sink_filling_algorithm {

public:
	///Class constructor
	sink_filling_algorithm_1(){}
	///Class constructor
	sink_filling_algorithm_1(field<double>*, grid_params*, field<bool>*, bool*, bool,
							 bool, double, bool* = nullptr);
	///Set flags and parameters of a sink-filling run
	void setup_flags(bool, bool = false, bool = false, bool = false, double = 0.1,
					 int = 1, double = 1.1, bool = false);
	///Destructor
	virtual ~sink_filling_algorithm_1() {};
	///Getter
	int get_method() {return method;}

protected:
	///The number labeling the method being used
	const int method = 1;
	///Implement the various virtual functions of the base class
	void process_neighbor();
	void set_ls_as_no_data(coords*);
	void push_land_sea_neighbor();
	void push_true_sink(coords*);
	void process_center_cell();
	void process_true_sink_center_cell();
	void push_neighbor();
	void tarasov_set_area_height();
	///Add a slope/epsilon factor when filling sinks
	bool add_slope = false;
	///The orography of the central cell being processed
	double center_orography = 0.0;
	///If add_slope is true this is the slope/epsilon factor to add
	double epsilon = 0.1; //normally in meters

};

/** Algorithm 4 of Barnes et al 2013. A river carving algorithm, similar to sink
 * filling but instead of filling sinks river exit sinks uphill from the deepest
 * point to the lowest point on the lip of the sink. Points directly neighboring
 * this path flow into it. Other points follow the down-hill towards the deepest
 * point
 */

class sink_filling_algorithm_4 : virtual public sink_filling_algorithm {

public:
	///Constructor
	sink_filling_algorithm_4(){}
	///Constructor
	sink_filling_algorithm_4(field<double>*, grid_params*, field<bool>*, bool*,
							 bool, field<int>*, bool, bool* = nullptr);
	///Virtual Destructor
	virtual ~sink_filling_algorithm_4() { if (not tarasov_mod) delete catchment_nums; }
	///Getter
	int get_method() {return method;}
protected:
	///Label which method is being used
	const int method = 4;
	///Implement the various virtual functions of the base class
	void process_neighbor();
	void push_land_sea_neighbor();
	void set_ls_as_no_data(coords*);
	void process_center_cell();
	void process_true_sink_center_cell();
	void push_true_sink(coords*);
	void push_neighbor();
	///Find the flow direction of a landsea cell - which is down the steepest slope if
	///the cell touches the sea and the orography extends under water. Otherwise the
	///first non-diagonal neighbor is used or the first neighbor (depending on the
	///value of the prefer non diagonal initial dirs flag
	void find_initial_cell_flow_direction();
	///Setup required fields and grid
	void setup_fields(double*, bool*, bool*, grid_params*, int*);
	///Setup required parameter and flags
	void setup_flags(bool, bool = false, bool = false, int = 1, double = 1.1, bool = false,
			 	 	 bool = false);
	///Implement virtual function of base class only used by Tarasov code
	void tarasov_set_area_height();
	///Set the flow direction of a cell to no data
	virtual void set_cell_to_no_data_value(coords*) = 0;
	///Set the flow direction of a cell to be a true sink
	virtual void set_cell_to_true_sink_value(coords*) = 0;
	///Set the index based river direction of a cell. The arguments are
	///the starting cell's coordinates and the destination coordinates
	virtual void set_index_based_rdirs(coords*,coords*) = 0;
	///Prefer non-diagonal river direction for initial river direction
	///for points next to the sea when there is a choice
	bool prefer_non_diagonal_initial_dirs = false;
	///The height of the highest rim crossed so far for a neighboring cell
	double nbr_rim_height = 0.0;
	///The height of  the highest rim crossed for far for a center cell
	double center_rim_height = 0.0;
	///Only produced index based river direction and don't produce direction
	///based river directions
	bool index_based_rdirs_only = true;
};


/**
 * Algorithm 1 implemented for a latitude-longitude grid. Multiply inherits from derived classes of base class
 */

class sink_filling_algorithm_1_latlon : public sink_filling_algorithm_1, public sink_filling_algorithm_latlon{
	///Implement virtual function of sink_filling_algorithm_latlon
	void push_vertical_edge(int,bool,bool);
	void push_horizontal_edge(int,bool,bool);
public:
	///Constructor
	sink_filling_algorithm_1_latlon() {};
	///Constructor
	sink_filling_algorithm_1_latlon(field<double>*, grid_params*, field<bool>*, bool*, bool,
					         	 	bool = false, double = 0.1, bool* = nullptr);
	///Setup the necessary fields and grid
	void setup_fields(double*, bool*,bool*,grid_params*);
	///Destructor
	virtual ~sink_filling_algorithm_1_latlon() {};
};

/**
 * Algorithm 4 implemented for a latitude-longitude grid. Multiply inherits from derived classes of base class
 */

class sink_filling_algorithm_4_latlon : public sink_filling_algorithm_4, public sink_filling_algorithm_latlon{
	///Direction based river directions
	field<double>* rdirs = nullptr;
	///Index based river directions
	field<int>* next_cell_lat_index = nullptr;
	field<int>* next_cell_lon_index = nullptr;
	///River direction value to use for true sinks
	const int true_sink_value = -5;
	///River direction value to use for no data
	const int no_data_value = -1;
	///Implement virtual function of sink_filling_algorithm_latlon
	void push_vertical_edge(int,bool,bool);
	void push_horizontal_edge(int,bool,bool);
	///Implement virtual function of sink_filling_algorithm_4
	void set_cell_to_no_data_value(coords*);
	void set_cell_to_true_sink_value(coords*);
public:
	///Constructor
	sink_filling_algorithm_4_latlon() {};
	///Constructor
	sink_filling_algorithm_4_latlon(field<double>*, grid_params*, field<bool>*, bool*,
							 	 	bool, field<int>*, bool, bool, field<int>*,field<int>*,
									bool* = nullptr, field<double>* = nullptr);
	///Set the river direction of the river direction field for the cell at the given coordinate
	///to the given value (value is integer from 0-9 but written as double for historical reasons
	void set_dir_based_rdir(coords*, double);
	///Setup flags and parameters
	void setup_flags(bool, bool = false, bool = false, bool = false, bool = false, int = 1,
				     double = 1.1, bool = false);
	///Setup fields and grid
	void setup_fields(double*, bool*, bool*, int*, int*, grid_params*, double*, int*);
	///Calculate the direction from neighbor to cell; only need when uses direction based
	///river directions on a latitude-longitude grid. (Could potentially make direction based
	///river direction more abstract and more widely applicable.)
	void calculate_direction_from_neighbor_to_cell();
	///Destructor
	virtual ~sink_filling_algorithm_4_latlon() { delete rdirs; delete next_cell_lat_index;
												 delete next_cell_lon_index; };
	///Setup and test find_initial_cell_flow_direction function on a latitude-longitude grid
	double test_find_initial_cell_flow_direction(coords*,grid_params*,field<double>*,
												 field<bool>*, bool = false);
	///Setup and test calculate_direction_from_neighbor_to_cell function
	double test_calculate_direction_from_neighbor_to_cell(coords*,coords*,grid_params*);
	///Implement virtual function of sink_filling_algorithm_4
	void set_index_based_rdirs(coords*,coords*);
};

#endif /* INCLUDE_SINK_FILLING_ALGORITHM_HPP_ */
