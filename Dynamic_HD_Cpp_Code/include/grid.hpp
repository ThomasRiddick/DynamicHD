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
#include "coords.hpp"
using namespace std;

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
	enum grid_types {latlon};
	///Getter
	int get_total_size() {return total_size;}
	///Getter
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
	virtual coords* calculate_downstream_coords_from_dir_based_rdir(coords* initial_coords,double rdir) = 0;
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

#endif /* INCLUDE_GRID_HPP_ */
