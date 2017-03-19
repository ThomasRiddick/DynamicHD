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
#include "coords.hpp"

/*Stores the orography, position and order of addition for a grid cell as an object
 * (which can be queued)*/
class cell{
	//The value of orography of this cell
	double orography;
	//The coordinates of this cell
	coords* cell_coords = nullptr;
	//an identifier marking order of insertion to impose a stable queue order
	//when two orography values are the same
	int k;
	//an identifier of which catchment this cell is part of
	int catchment_num;
	//edge number of initial edge (only used by Tarasov modification)
	int tarasov_initial_edge_number = 0;
	//maximum separation along path traveled from initial edge (only used by Tarasov modification)
	int tarasov_maximum_separation_from_initial_edge = 0;
	//height of the highest (real) value of orography crossed on the way to this point
	double rim_height;
	//length of path from entry into area (only used by Tarasov modification)
	double tarasov_path_length = 0;

public:
	//Class Constructors
	cell(double orography_in,coords* cell_coords_in,int catchment_num, double rim_height_in)
	: orography(orography_in), cell_coords(cell_coords_in), k(0), catchment_num(catchment_num),
	  rim_height(rim_height_in) {}
	cell(double orography_in,coords* cell_coords_in) : orography(orography_in),
			cell_coords(cell_coords_in), k(0), catchment_num(0), rim_height(0.0)  {}
	cell(double orography_in,coords* cell_coords_in,int catchment_num, double rim_height_in,
		 int tarasov_initial_edge_number_in, int tarasov_maximum_separation_from_initial_edge_in,
		 double tarasov_path_length_in)
	: orography(orography_in), cell_coords(cell_coords_in), k(0), catchment_num(catchment_num),
	  tarasov_initial_edge_number(tarasov_initial_edge_number_in),
	  tarasov_maximum_separation_from_initial_edge(tarasov_maximum_separation_from_initial_edge_in),
	  rim_height(rim_height_in), tarasov_path_length(tarasov_path_length_in) {}
	cell(double orography_in,coords* cell_coords_in, int tarasov_initial_edge_number_in,
		 int tarasov_maximum_separation_from_initial_edge_in, double tarasov_path_length_in)
	: orography(orography_in), cell_coords(cell_coords_in), k(0), catchment_num(0),
	  tarasov_initial_edge_number(tarasov_initial_edge_number_in),
	  tarasov_maximum_separation_from_initial_edge(tarasov_maximum_separation_from_initial_edge_in),
	  rim_height(0.0), tarasov_path_length(tarasov_path_length_in)  {}
	//Class destructor
	~cell() { delete cell_coords; }
	//Getters
	coords* get_cell_coords() {return cell_coords;}
	int get_catchment_num() {return catchment_num;}
	double get_orography() {return orography;}
	double get_rim_height() {return rim_height;}
	double get_tarasov_path_length() {return tarasov_path_length;}
	int get_tarasov_initial_edge_number() {return tarasov_initial_edge_number;}
	int get_tarasov_maximum_separation_from_initial_edge()
		{return tarasov_maximum_separation_from_initial_edge;}
	//Setters
	void set_orography(double value) {orography = value;}
	void set_k(int k_in) {k=k_in;}
	void set_tarasov_maximum_separation_from_initial_edge(int value)
		{tarasov_maximum_separation_from_initial_edge = value;}
	//Overloaded operators
	cell operator= (const cell&);
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
	landsea_cell(coords* cell_coords) : cell(0.0,cell_coords){}
};

#endif /* CELL_HPP_ */
