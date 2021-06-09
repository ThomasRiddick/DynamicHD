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
	//Overload equals operator
	bool operator== (const coords& rhs){
		return this->equals(rhs);
	};
	///Overload not equals
	bool operator!= (const coords& rhs){
		return ! (*this==rhs);
	};
	friend ostream& operator<< (ostream& out, coords& field_object) {
		field_object.print(out);
		return out;
	};
protected:
	virtual bool equals(const coords& rhs) = 0;
	virtual void print(ostream& out) const = 0;
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
	friend ostream& operator<< (ostream& out, latlon_coords& field_object) {
		field_object.print(out);
		return out;
		};
	protected:
	virtual void print(ostream& out) const {
		out << "lat: " << this->lat << " lon: "
				<< this->lon << endl;
	};
	//For overloading equals on the base class (as operators can't be virtual)
	bool equals(const coords& rhs){
		const latlon_coords* latlon_rhs = dynamic_cast<const latlon_coords*>(&rhs);
		if(latlon_rhs) return *this == *latlon_rhs;
		else return false;
	};
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
			{ return (index == rhs.get_index()); };
	///Overload ostream operator
	friend ostream& operator<< (ostream& out, generic_1d_coords& field_object)
		{ return out << "Index: " << field_object.index << endl; };
protected:
	virtual void print(ostream& out) const {
		out << "index: " << this->index << endl;
	};
	//For overloading equals on the base class (as operators can't be virtual)
	bool equals(const coords& rhs){
		const generic_1d_coords* generic_1d_coords_rhs = dynamic_cast<const generic_1d_coords*>(&rhs);
		if(generic_1d_coords_rhs) return *this == *generic_1d_coords_rhs;
		else return false;
	};
};

/**
 * The clone class uses dynamic casting and a hand keyed switch to make
 * it possible to in-line the clone statement
 */
inline coords* coords::clone(){
	switch(coords_type){
		case latlon:
			return new latlon_coords(*static_cast<latlon_coords*>(this));
		case generic_1d:
			return new generic_1d_coords(*static_cast<generic_1d_coords*>(this));
		default:
			throw runtime_error("Undefined coord type.. need to add it to clone");
	}
}

#endif /* INCLUDE_COORDS_HPP_ */
