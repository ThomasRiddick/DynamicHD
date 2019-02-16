/*
 * grid.hpp
 *
 *  Created on: Dec 18, 2016
 *      Author: thomasriddick
 */

#ifndef INCLUDE_GRID_HPP_
#define INCLUDE_GRID_HPP_

#include <sstream>
#include <functional>
#if USE_NETCDFCPP
#include <netcdfcpp.h>
#endif
#include "coords.hpp"
using namespace std;

/**
 * Abstract generic class for holding parameters of a grid. Real subclasses
 * are required for actual grids
 */

class grid_params {
protected:
	///Does this grid wrap east-west?
	bool nowrap = false;
public:
	///Constructor
	grid_params(bool nowrap_in) : nowrap(nowrap_in) {}
	///Destructor
	virtual ~grid_params() {};
	///Getter
	const int get_nowrap() { return nowrap; }
	///Setter
	void set_nowrap(bool nowrap_in) { nowrap = nowrap_in; }
};


/**
 * Generic grid class describing the relationship between cells on an abstract grid
 *  Subclasses of this describe specific grids. Uses hand written redirections to the
 *  appropriate subclasses functions via static casting instead of relying on virtual
 *  calls in time-critical cases (so that in-lining of these time critical function
 *  will be permitted). This is a abstract class.
 */

class grid {
public:
	virtual ~grid(){};
	///a enum with the title of the possible grid types
	enum grid_types {latlon,icon_single_index};
	///Getter
	int get_total_size() {return total_size;}
	///Getter. This return the index with any offset associated with using c++
	// and thus will likely differ from the offset free number recorded for
	// use with Fortran
	int get_index(coords*);
	///Getter
	virtual int get_edge_number(coords*) = 0;
	///Getter
	virtual int get_landsea_edge_num() = 0;
	///Getter
	virtual int get_true_sink_edge_num() = 0;
	///Getter
	virtual int get_separation_from_initial_edge(coords*,int) = 0;
	///Run the supplied function over the diagonal neighbors of the supplied
	///coordinates
	virtual void for_diagonal_nbrs(coords*,function<void(coords*)> ) = 0;
	///Run the supplied function over the non-diagonal neighbors of the
	///supplied coordinates
	virtual void for_non_diagonal_nbrs(coords*,function<void(coords*)>) = 0;
	///Run the supplied function over all the neighbors of the supplied
	///coordinates
	virtual void for_all_nbrs(coords*,function<void(coords*)>) = 0;
	///Run the supplied function over the entire field
	virtual void for_all(function<void(coords*)> func) = 0;
	///Run the supplied function over the entire field, supplying a boolean
	///at each point that flag whether the point is the end of a row (and
	///thus requires a line break) or not
	virtual void for_all_with_line_breaks(function<void(coords*,bool)>) = 0;
	///Return the specified coords wrapped if required (e.g. wrapped east-west
	///for a lat-lon grid
	coords* wrapped_coords(coords*);
	///Check if the supplied coordinates are outside the limits of the grid
	bool outside_limits(coords*);
	///Check if the two coordinates are non-diagonal neighbors of each other
	bool non_diagonal(coords*,coords*);
	///Used by Tarasov upscaling code - check if a cell is either a landsea
	///point or true sinks (two input booleans variables) and is also on a path
	///that started on a landsea or true sink point
	bool check_if_cell_connects_two_landsea_or_true_sink_points(int,bool,bool);
	///Used by Tarasov upscaling code - check if a cell is a corner cell
	virtual bool is_corner_cell(coords*) = 0;
	///Used by Tarasov upscaling code - check if a cell is on a the edge number
	///supplied as a second argument
	virtual bool check_if_cell_is_on_given_edge_number(coords*,int) = 0;
	///Check if supplied cell is an edge cell
	virtual bool is_edge(coords*) = 0;
	///Calculate the direction based river direction from the first cell to the
	///second
	double calculate_dir_based_rdir(coords*,coords*);
	bool fine_coords_in_same_cell(coords* fine_coords_set_one,
	                              coords* fine_coords_set_two,
	                              grid_params* fine_grid_params);
	virtual coords* calculate_downstream_coords_from_dir_based_rdir(coords* initial_coords,double rdir) = 0;
	virtual coords* convert_fine_coords(coords* fine_coords,grid_params* fine_grid_params) = 0;

protected:
	///The type of this grid object
	grid_types grid_type;
	///The total size of this grid if all dimensions are multiplied together
	int total_size;
	///Wrap coordinates around the global or not?
    ///No wrap actually effects the judging of what is inside and outside limits
	///and therefore wrapping; it doesn't affect wrapping directly merely ensures
	///that no is required when it is set to true
	bool nowrap = false;
};

/** A real class implementing functions on a latitude-longitude
 * grid
 */

class latlon_grid : public grid {
	int nlat;
	int nlon;
	//Used by the Tarasov Upscaling Code - numbers used
	//to identify the various kinds of edge path has
	//started from
	const int left_vertical_edge_num     = 1;
	const int right_vertical_edge_num    = 2;
	const int top_horizontal_edge_num    = 3;
	const int bottom_horizontal_edge_num = 4;
	//A landsea 'edge' is just a landsea point
	const int landsea_edge_num = 5;
	//A true sink 'edge' is just a true sink point
	const int true_sink_edge_num = 6;
public:
	///Constructor
	latlon_grid(grid_params*);
	virtual ~latlon_grid() {};
	///Getters
	int get_nlat() { return nlat; };
	///Getters
	int get_nlon() { return nlon; };
	///Getters
	int get_edge_number(coords*);
	///Getters
	int get_landsea_edge_num() { return landsea_edge_num; }
	///Getters
	int get_true_sink_edge_num() { return true_sink_edge_num; }
	///Getters
	int get_separation_from_initial_edge(coords*,int);
	/// This get the index of a set of coordinates within a 1D array
	/// representing at latitude-longitude field. It is not replacing
	/// a virtual function in the base class but instead endemic to
	/// latlon_grid and called from get_index in the base class through
	/// a select case statement coupled with static casting
	int latlon_get_index(latlon_coords* coords_in)
		{ return coords_in->get_lat()*nlon + coords_in->get_lon(); }
	//implement function that iterative apply a supplied function
	void for_diagonal_nbrs(coords*,function<void(coords*)> );
	void for_non_diagonal_nbrs(coords*,function<void(coords*)>);
	void for_all_nbrs(coords*,function<void(coords*)>);
	void for_all(function<void(coords*)>);
	void for_all_with_line_breaks(function<void(coords*,bool)>);
	//These next two functions are endemic to this subclass and are
	//called from the base class via a switch-case statement and
	//static casting
	///is given point outside limits of grid
	bool latlon_outside_limits(latlon_coords* coords);
	///Are two points linked directly by a non-diagonal
	bool latlon_non_diagonal(latlon_coords*, latlon_coords*);
	//Implementations of virtual functions of the base class
	bool is_corner_cell(coords*);
	bool is_edge(coords*);
	bool check_if_cell_is_on_given_edge_number(coords*,int);
	///These next two functions are endemic to this subclass and are
	///called from the base class via a switch-case statement and
	///static casting
	double latlon_calculate_dir_based_rdir(latlon_coords*,latlon_coords*);
	///Return wrapped version of supplied coordinates
	latlon_coords* latlon_wrapped_coords(latlon_coords*);
	coords* calculate_downstream_coords_from_dir_based_rdir(coords* initial_coords,double rdir);
	coords* convert_fine_coords(coords* fine_coords,grid_params* fine_grid_params);
};

/** A real class implementing functions on a ICON-style
 * grid without using blocks. Store index values as
 * FORTAN indices thus require an offset of one when
 * using them to index arrays
 */

class icon_single_index_grid : public grid {
	int ncells;
	//Used by the Tarasov Upscaling Code - numbers used
	//to identify the various kinds of edge path has
	//started from
	const int left_diagonal_edge_num     = 1;
	const int right_diagonal_edge_num    = 2;
	const int bottom_horizontal_edge_num = 3;
	//A landsea 'edge' is just a landsea point
	const int landsea_edge_num = 4;
	//A true sink 'edge' is just a true sink point
	const int true_sink_edge_num = 5;
	//Use neighbors that share only a vertex but not an
	//edge with the center cell
	bool use_secondary_neighbors = false;
	//An array with ncells*3 elements with the index of the
	//three neighbors of each cell
	int* neighboring_cell_indices = nullptr;
	//An array with ncells*9 elements with the index of the
	//nine secondary neighbors (neighbors of the cells vertices
	//that don't share an edge with the cell) of each cell
	int* secondary_neighboring_cell_indices = nullptr;
	//An array to use if no_wrap is set to allow a segment of a
	//global grid to be selected; inside area is true
	bool* subgrid_mask = nullptr;
	//Premarked edges for orography upscaling acts both lookup for number and mask
	bool* edge_nums = nullptr;
	//Value for non edges in edge nums array
	const int no_edge = 0;
	//Values for corners
	const int top_corner             = 4;
	const int bottom_right_corner    = 5;
	const int bottom_left_corner     = 6;
	//Precalculated edge seperation values (will be 3 entries per cell so size is ncells*3)
	double* edge_separations = nullptr;
	//Array offset
	const int array_offset = 1;
	const int no_neighbor = -1;
public:
	///Constructor
	icon_single_index_grid(grid_params*);
	virtual ~icon_single_index_grid() {};
	///Getters
	int get_npoints() { return ncells; };
	///Getters
	int get_edge_number(coords*);
	///Getters
	int get_landsea_edge_num() { return landsea_edge_num; }
	///Getters
	int get_true_sink_edge_num() { return true_sink_edge_num; }
	///Getters
	int get_separation_from_initial_edge(coords*,int);
	/// This get the index of a set of coordinates within a 1D array
	/// representing the ICON field. It is not replacing
	/// a virtual function in the base class but instead endemic to
	/// icon_single_index_grid and called from get_index in the base class through
	/// a select case statement coupled with static casting
	/// The index return includes the array offset
	int icon_single_index_get_index(generic_1d_coords* coords_in)
		{ return coords_in->get_index() - array_offset; }
	//setter
	void set_subgrid_mask(bool* subgrid_mask_in) { subgrid_mask = subgrid_mask_in; }
	//implement function that iterative apply a supplied function
	void for_diagonal_nbrs(coords*,function<void(coords*)> );
	void for_non_diagonal_nbrs(coords*,function<void(coords*)>);
	void for_all_nbrs(coords*,function<void(coords*)>);
	void for_all(function<void(coords*)>);
	void for_all_with_line_breaks(function<void(coords*,bool)>);
	//These next two functions are endemic to this subclass and are
	//called from the base class via a switch-case statement and
	//static casting
	///is given point outside limits of grid
	bool icon_single_index_outside_limits(generic_1d_coords* coords);
	///Are two points linked directly by a non-diagonal
	bool icon_single_index_non_diagonal(generic_1d_coords*, generic_1d_coords*);
	//Implementations of virtual functions of the base class
	bool is_corner_cell(coords*);
	bool is_edge(coords*);
	bool check_if_cell_is_on_given_edge_number(coords*,int);
	///This next function is endemic to this subclass and is
	///called from the base class via a switch-case statement and
	///static casting
	///Return wrapped version of supplied coordinates
	generic_1d_coords* icon_single_index_wrapped_coords(generic_1d_coords*);
	//Get the cell indices of the center cells three direct neighbors
	int get_cell_neighbors_index(generic_1d_coords* cell_coords,int neighbor_num)
		{ return neighboring_cell_indices[icon_single_index_get_index(cell_coords)*3 + neighbor_num]; }
	//Get the cell indices of the center cells nine secondary neighbors
	int get_cell_secondary_neighbors_index(generic_1d_coords* cell_coords,int neighbor_num)
		{ return secondary_neighboring_cell_indices[icon_single_index_get_index(cell_coords)*9 + neighbor_num]; }
	coords* convert_fine_coords(coords* fine_coords,grid_params* fine_grid_params);
	//Not implemeted for icon grid; return a runtime error
	coords* calculate_downstream_coords_from_dir_based_rdir(coords* initial_coords,double rdir);
	//Convert an index to coordinates. This is for EXTERNAL INDICES and does not include the array offset
	generic_1d_coords* convert_index_to_coords(int index)
		{ return new generic_1d_coords(index); }
};

/**
 * Concrete subclass containing the parameters for a latitude-longitude grid
 */

class latlon_grid_params : public grid_params {
	//Number of latitude points
	int nlat;
	//Number of longitude points
	int nlon;
public:
	virtual ~latlon_grid_params() {};
	///Class constructor
	latlon_grid_params(int nlat_in,int nlon_in)
	 	: grid_params(false),nlat(nlat_in), nlon(nlon_in){};
	///Class constructor
	latlon_grid_params(int nlat_in,int nlon_in,bool nowrap_in)
		: grid_params(nowrap_in), nlat(nlat_in), nlon(nlon_in) {};
	///Getter
	const int get_nlat() { return nlat; }
	///Getter
	const int get_nlon() { return nlon; }
};

/**
 * Concrete subclass containing the parameters for a ICON-style grid without using
 * blocks
 */

class icon_single_index_grid_params : public grid_params {
	//Number of points
	int ncells;
	//An array with ncells*3 elements with the index of the
	//three neighbors of each cell
	int* neighboring_cell_indices = nullptr;
	//Use neighbors that share only a vertex but not an
	//edge with the center cell
	bool use_secondary_neighbors = false;
	//An array with ncells*9 elements with the index of the
	//nine secondary neighbors (neighbors of the cells vertices
	//that don't share an edge with the cell) of each cell
	int* secondary_neighboring_cell_indices = nullptr;
	//If no wrap is set this can use to specify a subsection of the gri
	bool* subgrid_mask = nullptr;
	//Flag to show if this class calculated secondary neighbors and thus
	//needs to delete them
	bool calculated_secondary_neighboring_cell_indices = false;
	#if USE_NETCDFCPP
	//ICON grid parameter file path
	string icon_grid_params_filepath = "";
	#endif
	//Array offset
	const int array_offset = 1;
	//One corner of this cell is a five point vertices and thus it has
	//only 8 secondary neighbors and one entry in the array of secondary
	//neighbors is thus blank - use this value to signify that
	const int no_neighbor = -1;
public:
	virtual ~icon_single_index_grid_params()
	{if (calculated_secondary_neighboring_cell_indices)
		delete[] secondary_neighboring_cell_indices;};
	///Class constructor
	icon_single_index_grid_params(int ncells_in,int* neighboring_cell_indices_in,
			bool use_secondary_neighbors_in,int* secondary_neighboring_cell_indices_in)
	 	: grid_params(false),ncells(ncells_in),
		  neighboring_cell_indices(neighboring_cell_indices_in),
		  use_secondary_neighbors(use_secondary_neighbors_in),
		  secondary_neighboring_cell_indices(secondary_neighboring_cell_indices_in){};
	///Class constructor
	icon_single_index_grid_params(int ncells_in, int* neighboring_cell_indices_in,
			bool use_secondary_neighbors_in, int* secondary_neighboring_cell_indices_in,
			bool nowrap_in, bool* subgrid_mask_in)
		: grid_params(nowrap_in), ncells(ncells_in),
		  neighboring_cell_indices(neighboring_cell_indices_in),
		  use_secondary_neighbors(use_secondary_neighbors_in),
		  secondary_neighboring_cell_indices(secondary_neighboring_cell_indices_in),
		  subgrid_mask(subgrid_mask_in){};
	#if USE_NETCDFCPP
	icon_single_index_grid_params(string icon_grid_params_filepath_in,
	                              bool use_secondary_neighbors_in = true,
	                              bool read_params_file_immediately = true)
	: grid_params(true),
		use_secondary_neighbors(use_secondary_neighbors_in),
	  icon_grid_params_filepath(icon_grid_params_filepath_in)
	{ if (read_params_file_immediately) icon_single_index_grid_read_params_file(); }
	#endif
	///Getter
	const int get_ncells() { return ncells; }
	int* get_neighboring_cell_indices() { return neighboring_cell_indices; }
	const bool get_use_secondary_neighbors() { return use_secondary_neighbors; }
	int* get_secondary_neighboring_cell_indices() { return secondary_neighboring_cell_indices; }
	bool* get_subgrid_mask() { return subgrid_mask; }
	void icon_single_index_grid_calculate_secondary_neighbors();
	#if USE_NETCDFCPP
	void icon_single_index_grid_read_params_file();
	#endif
};

/*
 * A factory function that produces a grid subclass to match the
 * subclass of the input grid_params
 */

grid* grid_factory(grid_params*);

//avoid making this virtual so that it can be in-lined
inline int grid::get_index(coords* coords_in){
	switch(grid_type){
	case grid_types::latlon: {
			latlon_grid* this_as_latlon = static_cast<latlon_grid*>(this);
			latlon_coords* coords_in_latlon = static_cast<latlon_coords*>(coords_in);
			return this_as_latlon->latlon_get_index(coords_in_latlon); }
		break;
	case grid_types::icon_single_index: {
			icon_single_index_grid* this_as_icon_single_index = static_cast<icon_single_index_grid*>(this);
			generic_1d_coords* coords_in_generic_1d = static_cast<generic_1d_coords*>(coords_in);
			return this_as_icon_single_index->icon_single_index_get_index(coords_in_generic_1d); }
		break;
	default:
		throw runtime_error("Unknown grid type in get_index... need to add static casting to new grid types by hand");
	}
	//Prevents error messages
	return 0;
}

//the follow function definitions must go in the header to be worked
//if they are in-lined. This appears to be linked to a bug in clang

//avoid making this virtual so that it can be in-lined
inline coords* grid::wrapped_coords(coords* coords_in){
	switch(grid_type){
	case grid_types::latlon: {
			latlon_grid* this_as_latlon = static_cast<latlon_grid*>(this);
			latlon_coords* coords_in_latlon = static_cast<latlon_coords*>(coords_in);
			return static_cast<coords*>(this_as_latlon->latlon_wrapped_coords(coords_in_latlon)); }
		break;
	case grid_types::icon_single_index: {
			icon_single_index_grid* this_as_icon_single_index = static_cast<icon_single_index_grid*>(this);
			generic_1d_coords* coords_in_generic_1d = static_cast<generic_1d_coords*>(coords_in);
			return static_cast<coords*>(this_as_icon_single_index->
			                            icon_single_index_wrapped_coords(coords_in_generic_1d));}
		break;
	default:
		throw runtime_error("Unknown grid type in wrapped_coords... need to add static casting to new grid types by hand");
	}
	//Prevents error messages
	return 0;
}

//avoid making this virtual so that it can be in-lined
inline bool grid::outside_limits(coords* coords_in){
	switch(grid_type){
	case grid_types::latlon: {
			latlon_grid* this_as_latlon = static_cast<latlon_grid*>(this);
			latlon_coords* coords_in_latlon = static_cast<latlon_coords*>(coords_in);
			return this_as_latlon->latlon_outside_limits(coords_in_latlon); }
		break;
	case grid_types::icon_single_index: {
			icon_single_index_grid* this_as_icon_single_index = static_cast<icon_single_index_grid*>(this);
			generic_1d_coords* coords_in_generic_1d = static_cast<generic_1d_coords*>(coords_in);
			return this_as_icon_single_index->icon_single_index_outside_limits(coords_in_generic_1d);
	}
		break;
	default:
		throw runtime_error("Unknown grid type in wrapped_coords... need to add static casting to new grid types by hand");
	}
	//Prevents error messages
	return 0;
}

//avoid making this virtual so that it can be in-lined
inline bool grid::non_diagonal(coords* start_coords,coords* dest_coords){
	switch(grid_type){
	case grid_types::latlon: {
			latlon_grid* this_as_latlon = static_cast<latlon_grid*>(this);
			latlon_coords* start_coords_latlon = static_cast<latlon_coords*>(start_coords);
			latlon_coords* dest_coords_latlon = static_cast<latlon_coords*>(dest_coords);
			return this_as_latlon->latlon_non_diagonal(start_coords_latlon,dest_coords_latlon); }
		break;
	case grid_types::icon_single_index: {
			icon_single_index_grid* this_as_icon_single_index = static_cast<icon_single_index_grid*>(this);
			generic_1d_coords* start_coords_generic_1d = static_cast<generic_1d_coords*>(start_coords);
			generic_1d_coords* dest_coords_generic_1d = static_cast<generic_1d_coords*>(dest_coords);
			return this_as_icon_single_index->icon_single_index_non_diagonal(start_coords_generic_1d,
			                                                          			 dest_coords_generic_1d);
	}
		break;
	default:
		throw runtime_error("Unknown grid type in wrapped_coords... need to add static casting to new grid types by hand");
	}
	//Prevents error messages
	return 0;
}

//avoid making this virtual so that it can be in-lined
inline double grid::calculate_dir_based_rdir(coords* start_coords,coords* dest_coords){
	switch(grid_type){
	case grid_types::latlon: {
			latlon_grid* this_as_latlon = static_cast<latlon_grid*>(this);
			latlon_coords* start_coords_latlon = static_cast<latlon_coords*>(start_coords);
			latlon_coords* dest_coords_latlon = static_cast<latlon_coords*>(dest_coords);
			return this_as_latlon->latlon_calculate_dir_based_rdir(start_coords_latlon,dest_coords_latlon); }
		break;
	case grid_types::icon_single_index:
		throw runtime_error("Direction based river directions not defined for triangular icon grid");
	default:
		throw runtime_error("Unknown grid type in wrapped_coords... need to add static casting to new grid types by hand");
	}
	//Prevents error messages
	return 0;
}

//avoid making this virtual so that it can be in-lined
inline bool latlon_grid::latlon_outside_limits(latlon_coords* coords){
	if (nowrap) return (coords->get_lat() < 0 || coords->get_lat() >= nlat ||
			   		    coords->get_lon() < 0 || coords->get_lon() >= nlon);
	else return (coords->get_lat() < 0 || coords->get_lat() >= nlat);
}

//avoid making this virtual so that it can be in-lined
inline bool icon_single_index_grid::
	icon_single_index_outside_limits(generic_1d_coords* coords_in){
	//ICON grid is naturally wrapped. Get index takes care of the array offset
	if (nowrap && subgrid_mask) return subgrid_mask[icon_single_index_get_index(coords_in)];
	else return false;
}

#endif /* INCLUDE_GRID_HPP_ */
