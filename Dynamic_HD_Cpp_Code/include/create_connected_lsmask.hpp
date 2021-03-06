/*
 * create_connected_lsmask.hpp
 *
 *  Created on: May 24, 2016
 *      Author: thomasriddick
 */

#ifndef INCLUDE_CREATE_CONNECTED_LSMASK_HPP_
#define INCLUDE_CREATE_CONNECTED_LSMASK_HPP_

//The main create connect lsmask routine; sets up a connect ls-mask generation algorithm object and passes
//input to it
void latlon_create_connected_lsmask(bool* landsea_in, bool* ls_seed_points_in,
  	  	  	  	  	  	  	  	  int nlat_in, int nlon_in, bool use_diagonals_in);
//Interface function that takes integer as arguments and converts them to bool and after running main
//routine (above) then converts the boolean landsea mask back to an integer
void latlon_create_connected_lsmask_cython_wrapper(int* landsea_in_int, int* ls_seed_points_in_int,
  	  	  	  	  							int nlat_in, int nlon_in, int use_diagonals_in_int);

#endif /* INCLUDE_CREATE_CONNECTED_LSMASK_HPP_ */
