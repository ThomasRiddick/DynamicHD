/*
 * redistribute_water.cpp
 *
 *  Created on: May 30, 2019
 *      Author: thomasriddick
 */

/* IMPORTANT!
 * Note that a description of each functions purpose is provided with the function declaration in
 * the header file not with the function definition. The definition merely provides discussion of
 * details of the implementation.
 */

#include "drivers/redistribute_water.hpp"
#include "algorithms/water_redistribution_algorithm.hpp"

void latlon_redistribute_water_cython_wrapper(int* lake_numbers_in,
                                              int* lake_centers_in_int,
                                              double* water_to_redistribute_in,
                                              double* water_redistributed_to_lakes_in,
                                              double* water_redistributed_to_rivers_in,
                                              int nlat_in, int nlon_in,
                                              int coarse_nlat_in, int coarse_nlon_in){
  bool* lake_centers_in = new bool[nlat_in*nlon_in];
  for (auto i = 0; i < nlat_in*nlon_in; i++) {
    lake_centers_in[i] =  bool(lake_centers_in_int[i]);
  }
  latlon_redistribute_water(lake_numbers_in,
                            lake_centers_in,
                            water_to_redistribute_in,
                            water_redistributed_to_lakes_in,
                            water_redistributed_to_rivers_in,
                            nlat_in,nlon_in,coarse_nlat_in,
                            coarse_nlon_in);
  delete[] lake_centers_in;
}

void latlon_redistribute_water(int* lake_numbers_in,
                               bool* lake_centers_in,
                               double* water_to_redistribute_in,
                               double* water_redistributed_to_lakes_in,
                               double* water_redistributed_to_rivers_in,
                               int nlat_in, int nlon_in,
                               int coarse_nlat_in, int coarse_nlon_in){
  auto alg = water_redistribution_algorithm();
  auto grid_params_in = new latlon_grid_params(nlat_in,nlon_in);
  auto coarse_grid_params_in = new latlon_grid_params(coarse_nlat_in,
                                                      coarse_nlon_in);
  alg.setup_fields(lake_numbers_in,
                   lake_centers_in,
                   water_to_redistribute_in,
                   water_redistributed_to_lakes_in,
                   water_redistributed_to_rivers_in,
                   grid_params_in,
                   coarse_grid_params_in);
  alg. run_water_redistribution();
  delete grid_params_in;
  delete coarse_grid_params_in;
}
