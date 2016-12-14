/*
 * cell.hpp
 *
 * Contains the declaration of the cell class and the compare cell class (which is
 * very short)
 *
 *  Created on: Mar 16, 2016
 *      Author: thomasriddick
 */

#ifndef CELL_HPP_
#define CELL_HPP_

#include <utility>
#include "common_types.hpp"

/*Stores the orography, position and order of addition for a grid cell as an object
 * (which can be queued)*/
class cell{
	//The value of orography of this cell
	double orography;
	//The latitude and longitude of this cell
	int lat,lon;
	//an identifier marking order of insertion to impose a stable queue order
	//when two orography values are the same
	int k;
	//an identifier of which catchment this cell is part of
	int catchment_num;

public:
	//Class Constructor
	cell(double orography_in,int lat_in,int lon_in,int catchment_num) : orography(orography_in),
		lat(lat_in), lon(lon_in), k(0), catchment_num(catchment_num) {}
	cell(double orography_in,int lat_in,int lon_in) : orography(orography_in),
		lat(lat_in), lon(lon_in), k(0), catchment_num(0) {}
	//Getters
	int get_lat() {return lat;}
	int get_lon() {return lon;}
	int get_catchment_num() {return catchment_num;}
	double get_orography() {return orography;}
	integerpair get_cell_coords() {return integerpair(lat,lon);}
	//Setters
	void set_orography(double value) {orography = value;}
	void set_k(int k_in) {k=k_in;}
	//Overloaded operators
	friend bool operator> (const cell&,const cell&);
	friend bool operator< (const cell&,const cell&);
	friend bool operator>= (const cell&,const cell&);
	friend bool operator<= (const cell&,const cell&);
};
/*
 * Contains an overloaded operator that compares two cell objects
 * for a priority queue
 */

class compare_cells{

public:
	//Operator that compares two cell objects
	bool operator() (cell* lhs,cell* rhs){return *lhs>*rhs;}
};

class landsea_cell : public cell {
public:
	landsea_cell(int lat_in,int lon_in) : cell(0.0,lat_in,lon_in){}
};

#endif /* CELL_HPP_ */
