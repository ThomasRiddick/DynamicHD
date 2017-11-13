/*
 * coords.hpp
 *
 *  Created on: Dec 18, 2016
 *      Author: thomasriddick
 */

#ifndef INCLUDE_COORDS_HPP_
#define INCLUDE_COORDS_HPP_

#include <iostream>
#include <stdexcept>
using namespace std;

/**
 * Generic coordinates within a grid
 */

class coords {
public:
	///Destructor
	virtual ~coords(){};
	///Produce clone
	coords* clone();
	///Enumeration of subclass names
	enum coords_types {latlon,generic_1d};
	///Type of coordinate this coordinate instance is
	coords_types coords_type;
};

/**
 * Latitude-longitude coordinates within a latitude-longitude grid
 */
class latlon_coords : public coords{
	///Number of latitude points
	int lat;
	///Number of longitude points
	int lon;
public:
	///Destructor
	~latlon_coords(){};
	///Constructor
	latlon_coords(int lat_in, int lon_in) :
		lat(lat_in),lon(lon_in) {coords_type = coords_types::latlon; };
	///Constructor
	latlon_coords(const latlon_coords& coords_in) :
		lat(coords_in.lat), lon(coords_in.lon) {coords_type = coords_types::latlon; };
	///Getter
	const int get_lat() const { return lat;};
	///Getter
	const int get_lon() const { return lon;};
	///Overload equals operator
	bool operator== (const latlon_coords& rhs) const
			{ return (lat == rhs.get_lat() && lon == rhs.get_lon()); };
	///Overload ostream operator
	friend ostream& operator<< (ostream& out, latlon_coords& field_object)
		{ return out << "lat: " << field_object.lat << " lon: "
					 << field_object.lon << endl; };
};

/*
 * Single index within a flattened generic grid
 */

class generic_1d_coords : public coords{
	///Index
	int index;
public:
	///Destructor
	~generic_1d_coords(){};
	///Constructor
	generic_1d_coords(int index_in) :
		index(index_in) {coords_type = coords_types::generic_1d; };
	///Constructor
	generic_1d_coords(const generic_1d_coords& coords_in) :
		index(coords_in.index) {coords_type = coords_types::generic_1d; };
	///Getter
	const int get_index() const { return index;};
	///Overload equals operator
	bool operator== (const generic_1d_coords& rhs) const
			{ return (index == rhs.index()); };
	///Overload ostream operator
	friend ostream& operator<< (ostream& out, generic_1d_coords& field_object)
		{ return out << "Index: " << field_object.index << endl; };
};

/**
 * The clone class uses dynamic casting and a hand keyed switch to make
 * it possible to in-line the clone statement
 */
inline coords* coords::clone(){
	switch(coords_type){
		case coords_types::latlon:
			return new latlon_coords(*static_cast<latlon_coords*>(this));
			return new generic_1d_coords(*static_cast<generic_1d_coords*>(this));
		default:
			throw runtime_error("Undefined coord type.. need to add it to clone");
	}
}

#endif /* INCLUDE_COORDS_HPP_ */
