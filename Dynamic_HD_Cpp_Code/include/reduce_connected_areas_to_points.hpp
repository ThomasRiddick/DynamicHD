/*
 * reduce_connected_areas_to_points.hpp
 *
 *  Created on: Feb 23, 2018
 *      Author: thomasriddick
 */

#ifndef INCLUDE_REDUCE_CONNECTED_AREAS_TO_POINTS_HPP_
#define INCLUDE_REDUCE_CONNECTED_AREAS_TO_POINTS_HPP_

void latlon_reduce_connected_areas_to_points_cython_wrapper(int* areas_in_int,int nlat_in,int nlon_in,
		 	 	 	 	 	 	 	 	 	 	 	        int use_diagonals_in_int, double* orography_in=nullptr,
                                  int check_for_false_minima_in = 0);

void latlon_reduce_connected_areas_to_points(bool* areas_in,int nlat_in,int nlon_in,
		 	 	 	 	 	 	 	  	  	 bool use_diagonals_in,double* orography_in=nullptr,
                           bool check_for_false_minima_in = false);

#endif /* INCLUDE_REDUCE_CONNECTED_AREAS_TO_POINTS_HPP_ */
