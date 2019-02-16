/*
 * determine_river_directions.cpp
 *
 *  Created on: Feb 16, 2019
 *      Author: thomasriddick
 */

#include "determine_river_directions.hpp"
#include "river_direction_determination_algorithm.hpp"

void latlon_determine_river_directions_cython_wrapper(double* rdirs_in,
                                                      double* orography_in,
                                                      int* land_sea_in_int,
                                                      int* true_sinks_in_int,
                                                      int nlat_in, int nlon_in,
                                                      int always_flow_to_sea_in_int,
                                                      int use_diagonal_nbrs_in_int,
                                                      int mark_pits_as_true_sinks_in_int) {
  auto land_sea_in   = new bool[nlat_in*nlon_in];
  auto true_sinks_in = new bool[nlat_in*nlon_in];
  for (auto i = 0; i < nlat_in*nlon_in; i++) {
    land_sea_in[i]   =  bool(land_sea_in_int[i]);
    true_sinks_in[i] =  bool(true_sinks_in_int[i]);
  }
  latlon_determine_river_directions(rdirs_in,
                                    orography_in,
                                    land_sea_in,
                                    true_sinks_in,
                                    nlat_in,nlon_in,
                                    bool(always_flow_to_sea_in_int),
                                    bool(use_diagonal_nbrs_in_int),
                                    bool(mark_pits_as_true_sinks_in_int));
  delete [] land_sea_in;
  delete [] true_sinks_in;
}

void latlon_determine_river_directions(double* rdirs_in,
                                       double* orography_in,
                                       bool* land_sea_in,
                                       bool* true_sinks_in,
                                       int nlat_in, int nlon_in,
                                       bool always_flow_to_sea_in,
                                       bool use_diagonal_nbrs_in,
                                       bool mark_pits_as_true_sinks_in) {
  cout << "Entering River Determination C++ Code" << endl;
  auto alg = river_direction_determination_algorithm_latlon();
  auto grid_params_in = new latlon_grid_params(nlat_in,nlon_in);
  alg.setup_flags(always_flow_to_sea_in,use_diagonal_nbrs_in,
                  mark_pits_as_true_sinks_in);
  alg.setup_fields(rdirs_in,orography_in,land_sea_in,true_sinks_in,
                   grid_params_in);
  alg.determine_river_directions();
  delete grid_params_in;
}

