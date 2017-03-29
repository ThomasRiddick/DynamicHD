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

class grid {
public:
	virtual ~grid(){};
	enum grid_types {latlon};
	int get_total_size() {return total_size;}
	int get_index(coords*);
	virtual void for_diagonal_nbrs(coords*,function<void(coords*)> ) {};
	virtual void for_non_diagonal_nbrs(coords*,function<void(coords*)>) {};
	virtual void for_all_nbrs(coords*,function<void(coords*)>) {};
	virtual void for_all(function<void(coords*)> func) {};
	virtual void for_all_with_line_breaks(function<void(coords*,bool)>) {};
	coords* wrapped_coords(coords*);
	bool outside_limits(coords*);
	bool non_diagonal(coords*,coords*);
	double calculate_dir_based_rdir(coords*,coords*);
protected:
	grid_types grid_type;
	int total_size;
};

class grid_params {
public:
	virtual ~grid_params() {};
};

class latlon_grid : public grid {
	int nlat;
	int nlon;
public:
	latlon_grid(grid_params*);
	virtual ~latlon_grid() {};
	int get_nlat() { return nlat; };
	int get_nlon() { return nlon; };
	int latlon_get_index(latlon_coords* coords_in)
		{ return coords_in->get_lat()*nlon + coords_in->get_lon(); }
	void for_diagonal_nbrs(coords*,function<void(coords*)> );
	void for_non_diagonal_nbrs(coords*,function<void(coords*)>);
	void for_all_nbrs(coords*,function<void(coords*)>);
	void for_all(function<void(coords*)>);
	void for_all_with_line_breaks(function<void(coords*,bool)>);
	bool latlon_outside_limits(latlon_coords* coords)
		{ return (coords->get_lat() < 0 || coords->get_lat() >= nlat); }
	bool latlon_non_diagonal(latlon_coords*, latlon_coords*);
	double latlon_calculate_dir_based_rdir(latlon_coords*,latlon_coords*);
	latlon_coords* latlon_wrapped_coords(latlon_coords*);
};

class latlon_grid_params : public grid_params {
	int nlat;
	int nlon;
public:
	virtual ~latlon_grid_params() {};
	latlon_grid_params(int nlat_in,int nlon_in) : nlat(nlat_in), nlon(nlon_in){};
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
			//return this_as_latlon->latlon_get_index(coords_in_latlon); }
			int index = this_as_latlon->latlon_get_index(coords_in_latlon);
			return index;}
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

#endif /* INCLUDE_GRID_HPP_ */
