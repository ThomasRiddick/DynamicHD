/*
 * fill_lakes.cpp
 *
 *  Created on: Feb 22, 2018
 *      Author: thomasriddick
 */

#include "fill_lakes.hpp"
#include "lake_filling_algorithm.hpp"

void latlon_fill_lakes_cython_wrapper(int* lake_minima_in_int,int* lake_mask_in_int,
									  double* orography_in,int nlat_in, int nlon_in,
									  int use_highest_possible_lake_water_level_in_int) {
	auto lake_minima_in = new bool[nlat_in*nlon_in];
	auto lake_mask_in   = new bool[nlat_in*nlon_in];
	for (auto i = 0; i < nlat_in*nlon_in; i++ ) {
		lake_minima_in[i] = bool(lake_minima_in_int[i]);
		lake_mask_in[i]   = bool(lake_mask_in_int[i]);
	}
	latlon_fill_lakes(lake_minima_in,lake_mask_in,orography_in,nlat_in,nlon_in,
				 	  bool(use_highest_possible_lake_water_level_in_int));
	delete [] lake_minima_in;
	delete [] lake_mask_in;
}

void latlon_fill_lakes(bool* lake_minima_in,bool* lake_mask_in,double* orography_in,
					   int nlat_in, int nlon_in, bool use_highest_possible_lake_water_level_in) {
	cout << "Entering Lake Filling C++ Code" << endl;
	auto alg = lake_filling_algorithm();
	auto grid_params_in = new latlon_grid_params(nlat_in,nlon_in);
	alg.setup_flags(use_highest_possible_lake_water_level_in);
	alg.setup_fields(lake_minima_in,lake_mask_in,orography_in,grid_params_in);
	alg.fill_lakes();
	delete grid_params;
}
