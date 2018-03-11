/*
 * burn_carved_rivers.hpp
 *
 *  Created on: Feb 22, 2018
 *      Author: thomasriddick
 */

#ifndef INCLUDE_BURN_CARVED_RIVERS_HPP_
#define INCLUDE_BURN_CARVED_RIVERS_HPP_

void latlon_burn_carved_rivers_cython_wrapper(double* orography_in,double* rdirs_in,
											  int* minima_in_int,int* lakemask_in_int,
											  int nlat_in,int nlon_in);

void latlon_burn_carved_rivers(double* orography_in,double* rdirs_in,bool* minima_in,
		   	   	   	   	   	   bool* lakemask_in,int nlat_in,int nlon_in);

#endif /* INCLUDE_BURN_CARVED_RIVERS_HPP_ */
