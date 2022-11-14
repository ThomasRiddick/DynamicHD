/*
 * basic_bifurcation_algorithm.cpp
 *
 *  Created on: Feb 21, 2018
 *      Author: thomasriddick
 */

#include "algorithms/basic_bifurcation_algorithm.hpp"

void basic_bifurcation_algorithm::push_cell(coords* cell_coords){
      q.push(new landsea_cell(cell_coords));
}
