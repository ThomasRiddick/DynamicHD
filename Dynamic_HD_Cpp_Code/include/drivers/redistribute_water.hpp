/*
 * redistribute_water.hpp
 *
 *  Created on: May 31, 2019
 *      Author: thomasriddick
 */


#ifndef INCLUDE_REDISTRIBUTE_WATER_HPP_
#define INCLUDE_REDISTRIBUTE_WATER_HPP_

void latlon_redistribute_water_cython_wrapper(int* lake_numbers_in,
                                              int* lake_centers_in_int,
                                              double* water_to_redistribute_in,
                                              double* water_redistributed_to_lakes_in,
                                              double* water_redistributed_to_rivers_in,
                                              int nlat_in, int nlon_in,
                                              int coarse_nlat_in, int coarse_nlon_in);

void latlon_redistribute_water(int* lake_numbers_in,
                               bool* lake_centers_in,
                               double* water_to_redistribute_in,
                               double* water_redistributed_to_lakes_in,
                               double* water_redistributed_to_rivers_in,
                               int nlat_in, int nlon_in,
                               int coarse_nlat_in, int coarse_nlon_in);

#endif /* INCLUDE_REDISTRIBUTE_WATER_HPP_ */
