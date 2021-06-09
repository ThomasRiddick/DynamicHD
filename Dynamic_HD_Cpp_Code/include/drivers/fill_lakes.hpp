/*
 * fill_lakes.hpp
 *
 *  Created on: Feb 22, 2018
 *      Author: thomasriddick
 */

#ifndef INCLUDE_FILL_LAKES_HPP_
#define INCLUDE_FILL_LAKES_HPP_

void latlon_fill_lakes_cython_wrapper(int* lake_minima_in_int,int* lake_mask_in_int,
									  double* orography_in,int nlat_in, int nlon_in,
									  int use_highest_possible_lake_water_level_in_int);

void latlon_fill_lakes(bool* lake_minima_in,bool* lake_mask_in,double* orography_in,
					   int nlat_in, int nlon_in,bool use_highest_possible_lake_water_level_in);

#endif /* INCLUDE_FILL_LAKES_HPP_ */
