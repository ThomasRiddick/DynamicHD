/*
 * reduce_connected_areas_to_points.hpp
 *
 *  Created on: Feb 23, 2018
 *      Author: thomasriddick
 */

void latlon_reduce_connected_areas_to_points_cython_wrapper(int* areas_in_int,int nlat_in,int nlon_in,
		 	 	 	 	 	 	 	 	 	 	 	        int use_diagonals_in_int);

void latlon_reduce_connected_areas_to_points(bool* areas_in,int nlat_in,int nlon_in,
		 	 	 	 	 	 	 	  	  	 bool use_diagonals_in);
