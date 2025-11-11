/*
 * basic_bifurcation_algorithm.cpp
 *
 *  Created on: Feb 21, 2018
 *      Author: thomasriddick
 */

#include "algorithms/basic_bifurcation_algorithm.hpp"
#include <limits>

using namespace std;

void basic_bifurcation_algorithm::push_cell(coords* cell_coords){
      q.push(new landsea_cell(cell_coords));
}

void basic_bifurcation_algorithm::push_coastal_cell(coords* cell_coords){
      q.push(new cell(numeric_limits<double>::max(),cell_coords));
}
