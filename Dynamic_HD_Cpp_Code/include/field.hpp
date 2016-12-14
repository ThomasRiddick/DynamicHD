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

#include <iostream>
#include <vector>
#include <utility>
#include <sstream>
#include <iomanip>
#include "common_types.hpp"

using namespace std;

/*
 * This class is essentially a 2D array of a given type with an additional
 * set of function that calculate the neighbors of a given cell. It is indexed
 * by latitude then longitude (in keeping with standard matrix index notation
 * and in opposition to standard Cartesian coordinate notation).
 */
template <typename field_type>
class field
{
	//Number of points/cells along the y (latitude) and x (longitude) axes
	int nlat, nlon;
	//Does this object own the underlying_array (for garbage keeping purposes)
	//or was it passed in when the object was constructed
	bool data_passed_in;
	//The underlying data as a 1D array
	field_type* underlying_array;
	//An array of pointers to the start of each row of the underlying_array
	field_type** array;
	//Sets up the array of pointers
	void set_array();
	//Assists with finding the neighbors of a cell by treating edge cases for a given
	//neighbor then pushing the relevant cell onto the list of neighbor
	void process_edge_cases_and_push_back(vector<integerpair*>*,int,int,int,int);
public:
	//Constructor when initializing without passing in underlying data
	field(int nlat_in, int nlon_in) : nlat(nlat_in), nlon(nlon_in),
		data_passed_in(false){
		underlying_array = new field_type[nlat*nlon];
		array = new field_type*[nlat];
		set_array();
	};
	//Constructor when initializing with passed in underlying data
	field(field_type* data_in, int nlat_in, int nlon_in) :
		 nlat(nlat_in), nlon(nlon_in),data_passed_in(true),
		 underlying_array(data_in){
		array = new field_type*[nlat];
		set_array();
	};
	//Destructor, clean up where necessary
	~field();
	//Overloaded bracket operator to be a 2D indexing operator
	field_type& operator () (int i, int j) {return array[i][j];}
	//Overloaded bracket operator to be a 2D indexing operator (const version)
	field_type operator () (int i, int j) const {return array[i][j];}
	//Get the underlying array
	field_type* get_array() {return underlying_array;}
	//Get the underlying array (const version)
	field_type* get_array() const {return underlying_array;}
	//Getters
	int get_nlat() {return nlat;}
	int get_nlon() {return nlon;}
	//Get a list of neighbors accounting for edge cases using an integerpair of coordinates
	vector<integerpair*>* get_neighbors_coords(integerpair);
	//Get a list of neighbors accounting for edge cases using an integerpair of coordinates
	//and specifying which algorithm is being used
	vector<integerpair*>* get_neighbors_coords(integerpair,int);
	//Get a list of neighbor accounting for edge cases using coordinates specifying as two
	//integers passed in directly
	vector<integerpair*>* get_neighbors_coords(int,int);
	//Set all the cells/points in this field to a given value
	void set_all(field_type);
	//Overload the equality operator, defining equality as all corresponding entries in the two
	//field being the same
	bool operator== (const field<field_type>&) const;
	//Overload the output stream operator to print out the values of all the entries in the field
	//and the values of nlat and nlon.
	template <typename friends_field_type>
	friend ostream& operator<< (ostream&, const field<friends_field_type>&);
};

template <typename field_type>
void field<field_type>::set_array(){
	for (auto i = 0; i < nlat; i++){
			array[i] = underlying_array + nlon*i;
		}
}

//If this object created the underlying_array then delete it; always delete the
//array of pointers
template <typename field_type>
field<field_type>::~field(){
	if (!data_passed_in) delete[] underlying_array;
	delete[] array;
}

template <typename field_type>
void field<field_type>::set_all(field_type value){
	for (auto i = 0; i < nlat*nlon; i++){
		underlying_array[i] = value;
	}
}

//Assume method 1 is being used and call the version of this function with a method argument
template <typename field_type>
vector<integerpair*>* field<field_type>::get_neighbors_coords(integerpair coords) {
	return get_neighbors_coords(coords,1);
}

//Works out the neighbors and returns them in a vector of integerpairs. If method 4 is being
//used then put non-diagonals at back of the vector (which will be processed first). Use the
//function process_edge_cases_and_push_back to deal with edge cases and actually perform the
//push onto the vector.
template <typename field_type>
vector<integerpair*>* field<field_type>::get_neighbors_coords(integerpair coords, int method) {
	auto lat = coords.first;
	auto lon = coords.second;
	auto neighbors_coords = new vector<integerpair*>;
	switch (method) {
		case 4:
			for (auto i = lat-1; i<=lat+1;i=i+2){
				for (auto j = lon-1; j<=lon+1;j=j+2){
					process_edge_cases_and_push_back(neighbors_coords,i,j,nlat,nlon);
				}
			}
			//non-diagonals last (so they are at back of queue and processed first)
			for (auto i = lat-1; i<=lat+1;i=i+2){
				auto j=lon;
				process_edge_cases_and_push_back(neighbors_coords,i,j,nlat,nlon);
			}
			for (auto j = lon-1; j<=lon+1;j=j+2){
				auto i=lat;
				process_edge_cases_and_push_back(neighbors_coords,i,j,nlat,nlon);
			}
			break;
		case 1:
		default:
			for (auto i = lat-1; i <= lat+1;i++){
				for (auto j = lon-1; j <=lon+1; j++){
					if (i == lat && j == lon) continue;
					process_edge_cases_and_push_back(neighbors_coords,i,j,nlat,nlon);
				}
			}
			break;
		}
	return neighbors_coords;
}

template <typename field_type>
void field<field_type>::process_edge_cases_and_push_back(vector<integerpair*>* neighbors_coords,
													    int i, int j, int nlat, int nlon){
	if (i < 0 || i >= nlat) return;
	if (j < 0) {
		neighbors_coords->push_back(new pair<int,int>(i,nlon + j));
	}
	else if (j >= nlon) {
		neighbors_coords->push_back(new pair<int,int>(i,j - nlon));
	}
	else {
		neighbors_coords->push_back(new pair<int,int>(i,j));
	}
}

//Assume method 1 is being used and call the version of this function with a method argument
template <typename field_type>
vector<integerpair*>* field<field_type>::get_neighbors_coords(int lat,int lon){
	return get_neighbors_coords(integerpair(lat,lon),1);
}

//provides no check that the array on the right hand side is of the same size
template <typename field_type>
bool field<field_type>::operator==(const field<field_type>& rhs) const {
	auto fields_are_equal = true;
	for (auto i = 0; i < nlat*nlon;++i){
	fields_are_equal = (underlying_array[i] == rhs.get_array()[i]) && fields_are_equal;
	}
	return fields_are_equal;
}

//Print out nlat and nlon and then the underlying data
template <typename field_type>
ostream& operator<< (ostream& out,const field<field_type>& field_object){
	out << "nlat:" << field_object.nlat << " nlon:" << field_object.nlon << endl;
	for (auto i=0; i < field_object.nlat;++i){
		for (auto j=0; j < field_object.nlon;++j){
			out << setw(10) << setprecision(2) << field_object.array[i][j];
		}
		out << endl;
	}
	return out;
}

#endif /* FIELD_HPP_ */
