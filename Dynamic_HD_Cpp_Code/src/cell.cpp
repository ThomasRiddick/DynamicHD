/*
 * cell.cpp
 *
 * Contain definitions for cell class
 *
 *  Created on: Mar 18, 2016
 *      Author: thomasriddick
 */
#include "cell.hpp"

//If the orography value of the cells is equal compare the k value
//(order of insertion)
bool operator> (const cell& lhs_cell,const cell& rhs_cell) {
	if (lhs_cell.orography != rhs_cell.orography) return lhs_cell.orography > rhs_cell.orography;
	else if (lhs_cell.k < 0 || rhs_cell.k < 0) return lhs_cell.k < rhs_cell.k;
	else return lhs_cell.k > rhs_cell.k;
}

//If the orography value of the cells is equal compare the k value
//(order of insertion)
bool operator< (const cell& lhs_cell,const cell& rhs_cell) {
	if (lhs_cell.orography != rhs_cell.orography) return lhs_cell.orography < rhs_cell.orography;
	else if (lhs_cell.k < 0 || rhs_cell.k < 0) return lhs_cell.k > rhs_cell.k;
	else return lhs_cell.k < rhs_cell.k;
}

//Do not implement the total order condition for >= and <= as there is no concept of equals for total order
//(if is functioning correctly)
bool operator>= (const cell& lhs_cell,const cell& rhs_cell) { return lhs_cell.orography >= rhs_cell.orography;}
bool operator<= (const cell& lhs_cell,const cell& rhs_cell) { return lhs_cell.orography <= rhs_cell.orography;}
