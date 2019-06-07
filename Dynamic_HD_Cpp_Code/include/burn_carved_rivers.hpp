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
											  int nlat_in,int nlon_in,
                        int add_slope_in=0,int max_exploration_range_in=0,
                        double minimum_height_change_threshold_in=0.0,
                        int short_path_threshold_in=0,
                        double short_minimum_height_change_threshold_in=0.0);

void latlon_burn_carved_rivers(double* orography_in,double* rdirs_in,bool* minima_in,
		   	   	   	   	   	   bool* lakemask_in,int nlat_in,int nlon_in,
                           bool add_slope_in=false,int max_exploration_range_in=0,
                           double minimum_height_change_threshold_in=0.0,
                           int short_path_threshold_in=0,
                           double short_minimum_height_change_threshold_in=0.0);

#endif /* INCLUDE_BURN_CARVED_RIVERS_HPP_ */
