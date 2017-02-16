/*
 * create_connected_lsmask.cpp
 *
 *  Created on: May 24, 2016
 *      Author: thomasriddick
 */

#include "connected_lsmask_generation_algorithm.hpp"
#include "create_connected_lsmask.hpp"
#include "grid.hpp"

void latlon_create_connected_lsmask_cython_wrapper(int* landsea_in_int, int* ls_seed_points_in_int,
  	  	  	  	  	  	  	  	  	  	  		   int nlat_in, int nlon_in, int use_diagonals_in_int)
{
	auto landsea_in = new bool[nlat_in*nlon_in];
	auto ls_seed_points_in = new bool[nlat_in*nlon_in];
	for (auto i = 0; i < nlat_in*nlon_in; i++){
		landsea_in[i] = bool(landsea_in_int[i]);
		ls_seed_points_in[i] = bool(ls_seed_points_in_int[i]);
	}
	latlon_create_connected_lsmask_main(landsea_in, ls_seed_points_in,
	  	  	  	  	  	  	     nlat_in, nlon_in, bool(use_diagonals_in_int));
	for (auto i = 0; i < nlat_in*nlon_in; i++){
		landsea_in_int[i] = int(landsea_in[i]);
	}
}

void latlon_create_connected_lsmask_main(bool* landsea_in, bool* ls_seed_points_in,
		  	  	  	  	  	  	  int nlat_in, int nlon_in, bool use_diagonals_in)
{
	auto alg = create_connected_landsea_mask();
	alg.setup_flags(use_diagonals_in);
	alg.setup_fields(landsea_in, ls_seed_points_in, new latlon_grid_params(nlat_in,nlon_in));
	alg.generate_connected_mask();
}
