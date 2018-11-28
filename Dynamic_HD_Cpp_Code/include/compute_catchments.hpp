/*
 * compute_catchments.hpp
 *
 *  Created on: Feb 26, 2018
 *      Author: thomasriddick
 */

#ifndef INCLUDE_COMPUTE_CATCHMENTS_HPP_
#define INCLUDE_COMPUTE_CATCHMENTS_HPP_

#include <iostream>
#include <fstream>
using namespace std;

void latlon_compute_catchments(int* catchment_numbers_in, double* rdirs_in,
                               string loop_log_filepath,
							                 int nlat_in,int nlon_in);

#endif /* INCLUDE_COMPUTE_CATCHMENTS_HPP_ */
