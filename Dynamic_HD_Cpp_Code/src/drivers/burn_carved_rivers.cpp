/*
 * burn_carved_rivers.cpp
 *
 *  Created on: Feb 22, 2018
 *      Author: thomasriddick
 */

#include "drivers/burn_carved_rivers.hpp"
#include "algorithms/carved_river_direction_burning_algorithm.hpp"

void latlon_burn_carved_rivers_cython_wrapper(double* orography_in,double* rdirs_in,
											  int* minima_in_int, int* lakemask_in_int,
											  int nlat_in,int nlon_in,int add_slope_in,
							   				int max_exploration_range_in,
	               				double minimum_height_change_threshold_in,
	               				int short_path_threshold_in,
	                 			double short_minimum_height_change_threshold_in) {
	 auto minima_in = new bool[nlat_in*nlon_in];
	 auto lakemask_in = new bool[nlat_in*nlon_in];
	 for (auto i = 0; i < nlat_in*nlon_in; i++ ) {
		 minima_in[i]   = bool(minima_in_int[i]);
		 lakemask_in[i] = bool(lakemask_in_int[i]);
	 }
	 latlon_burn_carved_rivers(orography_in,rdirs_in,minima_in,
	 						   lakemask_in,nlat_in,nlon_in,bool(add_slope_in),
	 						   max_exploration_range_in,
	               minimum_height_change_threshold_in,
	               short_path_threshold_in,
	               short_minimum_height_change_threshold_in);
	 delete [] minima_in;
	 delete [] lakemask_in;
}

void latlon_burn_carved_rivers(double* orography_in,double* rdirs_in,bool* minima_in,
							   bool* lakemask_in,int nlat_in,int nlon_in,bool add_slope_in,
							   int max_exploration_range_in,
	               double minimum_height_change_threshold_in,
	               int short_path_threshold_in,
	               double short_minimum_height_change_threshold_in) {
	cout << "Entering C++ Carved River Burning Algorithm" << endl;
	auto alg = carved_river_direction_burning_algorithm_latlon();
	auto grid_params_in = new latlon_grid_params(nlat_in,nlon_in);
	alg.setup_flags(add_slope_in,max_exploration_range_in,
	                minimum_height_change_threshold_in,
	                short_path_threshold_in,
	                short_minimum_height_change_threshold_in);
	alg.setup_fields(orography_in,rdirs_in,minima_in,
					 lakemask_in,grid_params_in);
	alg.burn_carved_river_directions();
	delete grid_params_in;
}

