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
#include "enums.hpp"

/** Stores the orography, position and order of addition for a grid cell as an object
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
	//height of initial point on this path (only used by Tarasov modification)
	double tarasov_path_initial_height = 0;

public:
	///Class Constructor
	cell(double orography_in,coords* cell_coords_in,int catchment_num, double rim_height_in)
	: orography(orography_in), cell_coords(cell_coords_in), k(0), catchment_num(catchment_num),
	  rim_height(rim_height_in) {}
	///Class Constructor
	cell(double orography_in,coords* cell_coords_in) : orography(orography_in),
			cell_coords(cell_coords_in), k(0), catchment_num(0), rim_height(0.0)  {}
	///Class Constructor
	cell(double orography_in,coords* cell_coords_in,int catchment_num_in, double rim_height_in,
		 int tarasov_initial_edge_number_in, int tarasov_maximum_separation_from_initial_edge_in,
		 double tarasov_path_length_in,double tarasov_path_initial_height_in)
	: orography(orography_in), cell_coords(cell_coords_in), k(0), catchment_num(catchment_num_in),
	  tarasov_initial_edge_number(tarasov_initial_edge_number_in),
	  tarasov_maximum_separation_from_initial_edge(tarasov_maximum_separation_from_initial_edge_in),
	  rim_height(rim_height_in), tarasov_path_length(tarasov_path_length_in),
	  tarasov_path_initial_height(tarasov_path_initial_height_in) {}
	///Class Constructor
	cell(double orography_in,coords* cell_coords_in, int catchment_num_in,
		 int tarasov_initial_edge_number_in,int tarasov_maximum_separation_from_initial_edge_in,
		 double tarasov_path_length_in,double tarasov_path_initial_height_in)
	: orography(orography_in), cell_coords(cell_coords_in), k(0), catchment_num(catchment_num_in),
	  tarasov_initial_edge_number(tarasov_initial_edge_number_in),
	  tarasov_maximum_separation_from_initial_edge(tarasov_maximum_separation_from_initial_edge_in),
	  rim_height(0.0), tarasov_path_length(tarasov_path_length_in),
	  tarasov_path_initial_height(tarasov_path_initial_height_in) {}
	//Class destructor
	~cell() {delete cell_coords;}
	///Getter
	coords* get_cell_coords() {return cell_coords;}
	///Getter
	int get_catchment_num() {return catchment_num;}
	///Getter
	double get_orography() {return orography;}
	///Getter
	double get_rim_height() {return rim_height;}
	///Getter
	double get_tarasov_path_length() {return tarasov_path_length;}
	///Getter
	double get_tarasov_path_initial_height() {return tarasov_path_initial_height;}
	///Getter
	int get_tarasov_initial_edge_number() {return tarasov_initial_edge_number;}
	///Getter
	int get_tarasov_maximum_separation_from_initial_edge()
		{return tarasov_maximum_separation_from_initial_edge;}
	///Setter
	void set_orography(double value) {orography = value;}
	///Setter
	void set_k(int k_in) {k=k_in;}
	///Setter
	void set_tarasov_maximum_separation_from_initial_edge(int value)
		{tarasov_maximum_separation_from_initial_edge = value;}
	//Clone operator
	cell* clone();
	///Overloaded equals operator
	cell operator= (const cell&);
	///Overloaded greater than operator
	friend bool operator> (const cell&,const cell&);
	///Overloaded less than operator
	friend bool operator< (const cell&,const cell&);
	///Overloaded greater than or equals to operator
	friend bool operator>= (const cell&,const cell&);
	///Overloaded less than or equals to operator
	friend bool operator<= (const cell&,const cell&);
	//Overloaded streaming operator
	friend ostream& operator<<(ostream& out, cell& cell_object) {
		return out << "Cell Orography: " << cell_object.orography << " "
							 << "Cell Coords: " << *cell_object.cell_coords;
	}
};


inline cell* cell::clone(){
	return new cell(orography,cell_coords->clone(),
	                catchment_num,tarasov_initial_edge_number,
	                tarasov_maximum_separation_from_initial_edge,
		 							tarasov_path_length,tarasov_path_initial_height);
}

/**
 * Contains an overloaded operator that compares two cell objects
 * for a priority queue
 */

class compare_cells{

public:
	///Operator that compares two cell objects
	bool operator() (cell* lhs,cell* rhs){return *lhs>*rhs;}
};

/**
 * Subclass by for land-sea mask generation. Basically a wrapper
 * that uses the superclass with the orography value set to 0.0
 */

class landsea_cell : public cell {
public:
	///Constructor
	landsea_cell(coords* cell_coords) : cell(0.0,cell_coords) {}
};

class basin_cell : public cell {
public:
	//Constructor
	basin_cell(double orography_in, height_types height_type_in,
	           coords* cell_coords_in) :
		cell(orography_in,cell_coords_in),
		height_type(height_type_in) {}
	//Clone operator
	basin_cell* clone();
	height_types get_height_type() {return height_type;}
	friend ostream& operator<<(ostream& out, basin_cell& cell_object) {
		return out << static_cast<cell&>(cell_object) << "Height Type: " << cell_object.height_type;
	}
private:
	height_types height_type;
};

inline basin_cell* basin_cell::clone(){
	return new basin_cell(get_orography(),height_type,
	           						get_cell_coords()->clone());
}

#endif /* CELL_HPP_ */
