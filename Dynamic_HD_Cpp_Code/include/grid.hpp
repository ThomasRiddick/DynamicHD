/*
 * grid.hpp
 *
 *  Created on: Dec 18, 2016
 *      Author: thomasriddick
 */

#ifndef INCLUDE_GRID_HPP_
#define INCLUDE_GRID_HPP_

#include <sstream>
#include "coords.hpp"
using namespace std;

class grid {
public:
	virtual ~grid(){};
	enum grid_types {latlon};
	int get_total_size() {return total_size;}
	int get_index(coords*);
	virtual int get_edge_number(coords*) = 0;
	virtual int get_landsea_edge_num() = 0;
	virtual int get_true_sink_edge_num() = 0;
	virtual int get_separation_from_initial_edge(coords*,int) = 0;
	virtual void for_diagonal_nbrs(coords*,function<void(coords*)> ) = 0;
	virtual void for_non_diagonal_nbrs(coords*,function<void(coords*)>) = 0;
	virtual void for_all_nbrs(coords*,function<void(coords*)>) = 0;
	virtual void for_all(function<void(coords*)> func) = 0;
	virtual void for_all_with_line_breaks(function<void(coords*,bool)>) = 0;
	coords* wrapped_coords(coords*);
	bool outside_limits(coords*);
	bool non_diagonal(coords*,coords*);
	bool check_if_cell_connects_two_landsea_or_true_sink_points(int,bool,bool);
	virtual bool is_corner_cell(coords*) = 0;
	virtual bool check_if_cell_is_on_given_edge_number(coords*,int) = 0;
	virtual bool is_edge(coords*) = 0;
	double calculate_dir_based_rdir(coords*,coords*);
protected:
	grid_types grid_type;
	int total_size;
    //No wrap actually effect the judging of what is inside and outside limits
	//and therefore wrapping; it doesn't affect wrapping directly merely ensures
	//that no is required when it is set to true
	bool nowrap = false;
};

class grid_params {
protected:
	bool nowrap = false;
public:
	grid_params(bool nowrap_in) : nowrap(nowrap_in) {}
	virtual ~grid_params() {};
	const int get_nowrap() { return nowrap; }
	void set_nowrap(bool nowrap_in) { nowrap = nowrap_in; }
};

class latlon_grid : public grid {
	int nlat;
	int nlon;
	const int left_vertical_edge_num     = 1;
	const int right_vertical_edge_num    = 2;
	const int top_horizontal_edge_num    = 3;
	const int bottom_horizontal_edge_num = 4;
	const int landsea_edge_num = 5;
	const int true_sink_edge_num = 6;
public:
	latlon_grid(grid_params*);
	virtual ~latlon_grid() {};
	int get_nlat() { return nlat; };
	int get_nlon() { return nlon; };
	int get_edge_number(coords*);
	int get_landsea_edge_num() { return landsea_edge_num; }
	int get_true_sink_edge_num() { return true_sink_edge_num; }
	int get_separation_from_initial_edge(coords*,int);
	int latlon_get_index(latlon_coords* coords_in)
		{ return coords_in->get_lat()*nlon + coords_in->get_lon(); }
	void for_diagonal_nbrs(coords*,function<void(coords*)> );
	void for_non_diagonal_nbrs(coords*,function<void(coords*)>);
	void for_all_nbrs(coords*,function<void(coords*)>);
	void for_all(function<void(coords*)>);
	void for_all_with_line_breaks(function<void(coords*,bool)>);
	bool latlon_outside_limits(latlon_coords* coords);
	bool latlon_non_diagonal(latlon_coords*, latlon_coords*);
	bool is_corner_cell(coords*);
	bool is_edge(coords*);
	bool check_if_cell_is_on_given_edge_number(coords*,int);
	double latlon_calculate_dir_based_rdir(latlon_coords*,latlon_coords*);
	latlon_coords* latlon_wrapped_coords(latlon_coords*);
};

class latlon_grid_params : public grid_params {
	int nlat;
	int nlon;
public:
	virtual ~latlon_grid_params() {};
	latlon_grid_params(int nlat_in,int nlon_in)
	 	: grid_params(false),nlat(nlat_in), nlon(nlon_in){};
	latlon_grid_params(int nlat_in,int nlon_in,bool nowrap_in)
		: grid_params(nowrap_in), nlat(nlat_in), nlon(nlon_in) {};
	const int get_nlat() { return nlat; }
	const int get_nlon() { return nlon; }
};

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
//if they are inlined. This appears to be linked to a bug in clang

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

inline bool latlon_grid::latlon_outside_limits(latlon_coords* coords){
	if (nowrap) return (coords->get_lat() < 0 || coords->get_lat() >= nlat ||
			   		    coords->get_lon() < 0 || coords->get_lon() >= nlon);
	else return (coords->get_lat() < 0 || coords->get_lat() >= nlat);
}

#endif /* INCLUDE_GRID_HPP_ */
