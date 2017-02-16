/*
 * coords.hpp
 *
 *  Created on: Dec 18, 2016
 *      Author: thomasriddick
 */

#ifndef INCLUDE_COORDS_HPP_
#define INCLUDE_COORDS_HPP_

#include <iostream>
using namespace std;

class coords {
public:
	virtual ~coords(){};
	coords* clone();
	enum coords_types {latlon};
	coords_types coords_type;
};

class latlon_coords : public coords{
	int lat;
	int lon;
public:
	~latlon_coords(){};
	latlon_coords(int lat_in, int lon_in) :
		lat(lat_in),lon(lon_in) {coords_type = coords_types::latlon; };
	latlon_coords(const latlon_coords& coords_in) :
		lat(coords_in.lat), lon(coords_in.lon) {coords_type = coords_types::latlon; };
	const int get_lat() const { return lat;};
	const int get_lon() const { return lon;};
	bool operator== (const latlon_coords& rhs) const
			{ return (lat == rhs.get_lat() && lon == rhs.get_lon()); };
	friend ostream& operator<< (ostream& out, latlon_coords& field_object)
		{ return out << "lat: " << field_object.lat << " lon: "
					 << field_object.lon << endl; };
};

inline coords* coords::clone(){
	switch(coords_type){
		case coords_types::latlon:
			return new latlon_coords(*static_cast<latlon_coords*>(this));
		default:
			throw runtime_error("Undefined coord type.. need to add it to clone");
	}
}

#endif /* INCLUDE_COORDS_HPP_ */
