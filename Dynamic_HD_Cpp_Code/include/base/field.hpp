/*
 * field.hpp
 *
 * Contains declarations and definitions (as it a template class) of the
 * field class
 *
 *  Created on: Mar 16, 2016
 *      Author: thomasriddick
 */

#ifndef FIELD_HPP_
#define FIELD_HPP_

#include <vector>
#include <utility>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include "base/grid.hpp"

using namespace std;

/**
 * This class contains a field data on a specified grid and provides a number
 * of functions on that field along with wrapping round the globe at the edges
 * where required. Is a template so it can be used for fields of any generic
 * data type
 * For a latitude-longitude grid this class is essentially a 2D array of a given
 * type with an additional set of function that calculate the neighbors of a
 * given cell. It is indexed by latitude then longitude (in keeping with
 * standard matrix index notation and in opposition to standard Cartesian
 * coordinate notation).
 */
template <typename field_type>
class field
{
	///Does this object own the underlying_array (for garbage keeping purposes)
	///or was it passed in when the object was constructed
	bool data_passed_in;
	///The data as a 1D array
	field_type* array = nullptr;
	///Assists with finding the neighbors of a cell by treating edge cases for a given
	///neighbor then pushing the relevant cell onto the list of neighbor
	void process_edge_cases_and_push_back(vector<coords*>*,coords*);
	///The grid object to use to interpret the underlying 1D
	grid* _grid = nullptr;
public:
	///Constructor when initializing without passing in underlying data
	field(grid_params* params) :
		data_passed_in(false), array(nullptr),
		_grid(grid_factory(params)){
		array = new field_type[_grid->get_total_size()];
	};
	///Constructor when initializing with passed in underlying data
	field(field_type* data_in, grid_params* params) :
			data_passed_in(true), array(data_in),
			_grid(grid_factory(params)) {};
	///Copy constructor
	field(const field<field_type>& field_object) :
		data_passed_in(true),array(field_object.array),
		_grid(field_object._grid){};
	///Destructor, clean up where necessary
	~field();
	///Overloaded bracket operator to be a generic indexing operator
	field_type& operator () (coords* coords_in) { return array[_grid->get_index(coords_in)]; }
	///Overloaded bracket operator to be a generic indexing operator (const version)
	field_type operator () (coords* coords_in) const { return array[_grid->get_index(coords_in)]; }
	///Get the underlying array
	field_type* get_array() {return array;}
	///Get the underlying array (const version)
	field_type* get_array() const {return array;}
	///Get a list of neighbors accounting for edge cases
	vector<coords*>* get_neighbors_coords(coords*);
	///Get a list of neighbors accounting for edge cases
	///specifying which algorithm is being used
	vector<coords*>* get_neighbors_coords(coords*,int);
	///Set all the cells/points in this field to a given value
	void set_all(field_type);
	///Overload the equality operator, defining equality as all corresponding entries in the two
	///field being the same
	bool operator== (const field<field_type>&) const;
	///check if two field are almost equal to within a tolerance
	bool almost_equal(const field<field_type>&, double = 1.0e-12) const;
	//Turn off deleting the data in destruction
	void switch_data_deletion_off() {data_passed_in=false;}
	//Get the maximum value in the field
	field_type get_max_element(){return *max_element(array,array+_grid->get_total_size());}
	///Overload the output stream operator to print out the values of all the entries in the field
	///and the values of nlat and nlon.
	template <typename friends_field_type>
	friend ostream& operator<< (ostream&, const field<friends_field_type>&);
};

//If this object created the underlying_array then delete it; always delete the
//grid object
template <typename field_type>
field<field_type>::~field(){
	if (!data_passed_in) delete[] array;
	delete _grid;
}

//Set all the cells/points in this field to a given value
template <typename field_type>
void field<field_type>::set_all(field_type value){
	for (auto i = 0; i < _grid->get_total_size(); i++){
		array[i] = value;
	}
}

//Assume method 1 is being used and call the version of this function with a method argument
template <typename field_type>
vector<coords*>* field<field_type>::get_neighbors_coords(coords* coords) {
	return get_neighbors_coords(coords,1);
}

//Works out the neighbors and returns them. If method 4 is being
//used then put non-diagonals at back of the vector (which will be processed first). Use the
//function process_edge_cases_and_push_back to deal with edge cases and actually perform the
//push onto the vector.
template <typename field_type>
vector<coords*>* field<field_type>::get_neighbors_coords(coords* coords_in, int method) {
	auto neighbors_coords = new vector<coords*>;
	function<void(coords*)> process_and_push_func = [&](coords* this_neighbors_coords)
							{process_edge_cases_and_push_back(neighbors_coords,this_neighbors_coords);};
	switch (method) {
		case 4:
			_grid->for_diagonal_nbrs(coords_in,process_and_push_func);
			//non-diagonals last (so they are at back of queue and processed first)
			_grid->for_non_diagonal_nbrs(coords_in,process_and_push_func);
			break;
		case 1:
		default:
			_grid->for_all_nbrs(coords_in,process_and_push_func);
			break;
		}
	return neighbors_coords;
}

//Assists with finding the neighbors of a cell by treating edge cases for a given
//neighbor then pushing the relevant cell onto the list of neighbor
template <typename field_type>
void field<field_type>::process_edge_cases_and_push_back(vector<coords*>* neighbors_coords,
																   coords* coords_in){
	if(_grid->outside_limits(coords_in)) {
		delete coords_in;
		return;
	}
	auto wrapped_coords = _grid->wrapped_coords(coords_in);
	neighbors_coords->push_back(wrapped_coords);
	if (wrapped_coords != coords_in) delete coords_in;
}

//provides no check that the array on the right hand side is of the same size
template <typename field_type>
bool field<field_type>::operator==(const field<field_type>& rhs) const {
	auto fields_are_equal = true;
	for (auto i = 0; i < _grid->get_total_size();++i){
	fields_are_equal = (array[i] == rhs.get_array()[i]) && fields_are_equal;
	}
	return fields_are_equal;
}

//check if two field are almost equal to within a tolerance
template <typename field_type>
bool field<field_type>::almost_equal(const field<field_type>& rhs, double absolute_tolerance) const {
	auto fields_are_almost_equal = true;
	for (auto i = 0; i < _grid->get_total_size();++i){
	fields_are_almost_equal = ((absolute_tolerance > (array[i] - rhs.get_array()[i])) &&
							  ((array[i] - rhs.get_array()[i]) > -1.0*absolute_tolerance) &&
							    fields_are_almost_equal);
	}
	return fields_are_almost_equal;

}

//Print out nlat and nlon and then the underlying data
template <typename field_type>
ostream& operator<< (ostream& out,const field<field_type>& field_object){
	field_object._grid->for_all_with_line_breaks([&](coords* coords, bool end_of_line){
		int width = 10;
		if (is_same<field_type,int>::value) width = 3;
		if (is_same<field_type,bool>::value) width = 2;
		if (end_of_line) out << endl;
		out <<  setw(width) << setprecision(2) << field_object(coords);
	});
	return out;
}

#endif /* FIELD_HPP_ */
