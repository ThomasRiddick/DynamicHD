/*
 * determine_river_directions.hpp
 *
 *  Created on: Feb 16, 2019
 *      Author: thomasriddick
 */

#ifndef INCLUDE_DETERMINE_RIVER_DIRECTIONS_HPP_
#define INCLUDE_DETERMINE_RIVER_DIRECTIONS_HPP_

void latlon_determine_river_directions_cython_wrapper(double* rdirs_in,
                                                      double* orography_in,
                                                      int* land_sea_in_int,
                                                      int* true_sinks_in_int,
                                                      int nlat_in, int nlon_in,
                                                      int always_flow_to_sea_in_int,
                                                      int use_diagonal_nbrs_in_int,
                                                      int mark_pits_as_true_sinks_in_int);

void latlon_determine_river_directions(double* rdirs_in,
                                       double* orography_in,
                                       bool* land_sea_in,
                                       bool* true_sinks_in,
                                       int nlat_in, int nlon_in,
                                       bool always_flow_to_sea_in,
                                       bool use_diagonal_nbrs_in,
                                       bool mark_pits_as_true_sinks_in);

#endif /* INCLUDE_DETERMINE_RIVER_DIRECTIONS_HPP_ */
