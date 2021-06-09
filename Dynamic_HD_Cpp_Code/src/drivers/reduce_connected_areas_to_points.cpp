/*
 * reduce_connected_areas_to_points.cpp
 *
 *  Created on: Feb 23, 2018
 *      Author: thomasriddick
 */

#include "drivers/reduce_connected_areas_to_points.hpp"
#include "algorithms/reduce_connected_areas_to_points_algorithm.hpp"

void latlon_reduce_connected_areas_to_points_cython_wrapper(int* areas_in_int,int nlat_in,int nlon_in,
		 	 	 	 	 	 	 	 	 	 	 	        int use_diagonals_in_int,double* orography_in,
                                  int check_for_false_minima_in) {
	auto areas_in = new bool[nlat_in*nlon_in];
	for (auto i = 0; i < nlat_in*nlon_in; i++) {
		areas_in[i] =  bool(areas_in_int[i]);
	}
	latlon_reduce_connected_areas_to_points(areas_in,nlat_in,nlon_in,
										 											bool(use_diagonals_in_int),
										 											orography_in,bool(check_for_false_minima_in));
	for (auto i = 0; i < nlat_in*nlon_in; i++){
		areas_in_int[i] = int(areas_in[i]);
	}
	delete [] areas_in;
}

void latlon_reduce_connected_areas_to_points(bool* areas_in,int nlat_in,int nlon_in,
											 bool use_diagonals_in, double* orography_in,
                       bool check_for_false_minima_in) {
	cout << "Entering Connected Area to Point Reduction C++ Code" << endl;
	auto alg = reduce_connected_areas_to_points_algorithm();
	auto grid_params_in = new latlon_grid_params(nlat_in,nlon_in);
	alg.setup_flags(use_diagonals_in,check_for_false_minima_in);
	alg.setup_fields(areas_in,orography_in,grid_params_in);
	alg.iterate_over_field();
	delete grid_params_in;
}


