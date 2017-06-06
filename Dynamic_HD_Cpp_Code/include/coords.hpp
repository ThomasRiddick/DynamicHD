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
	enum coords_types {latlon};
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

/**
 * The clone class uses dynamic casting and a hand keyed switch to make
 * it possible to in-line the clone statement
 */
inline coords* coords::clone(){
	switch(coords_type){
		case coords_types::latlon:
			return new latlon_coords(*static_cast<latlon_coords*>(this));
		default:
			throw runtime_error("Undefined coord type.. need to add it to clone");
	}
}

#endif /* INCLUDE_COORDS_HPP_ */
