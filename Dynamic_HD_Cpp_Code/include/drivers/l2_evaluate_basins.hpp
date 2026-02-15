/*
 * evaluate_basins.hpp
 *
 *  Created on: June 25, 2018
 *      Author: thomasriddick
 */

#ifndef EVALUATE_BASINS_HPP_
#define EVALUATE_BASINS_HPP_
#include <vector>
using namespace std;

double* latlon_evaluate_basins_cython_wrapper(int* landsea_in_int,
                                              int* minima_in_int,
                                              double* raw_orography_in,
                                              double* corrected_orography_in,
                                              double* cell_areas_in,
                                              int* prior_fine_rdirs_in,
                                              int* prior_fine_catchments_in,
                                              int* coarse_catchment_nums_in,
                                              int nlat_fine, int nlon_fine,
                                              int nlat_coarse,int nlon_coarse,
                                              int* lake_numbers_out_int,
                                              double* sinkless_rdirs_out_double,
                                              int* number_of_lakes_out_ptr,
                                              int* lake_mask_out_int,
                                              int* lakes_as_array_size);

vector<double>* latlon_evaluate_basins(bool* landsea_in,
                                       bool* minima_in,
                                       double* raw_orography_in,
                                       double* corrected_orography_in,
                                       double* cell_areas_in,
                                       int* prior_fine_rdirs_in,
                                       int* prior_fine_catchments_in,
                                       int* coarse_catchment_nums_in,
                                       int nlat_fine, int nlon_fine,
                                       int nlat_coarse,int nlon_coarse,
                                       int* lake_numbers_out,
                                       short* sinkless_rdirs_out,
                                       int &number_of_lakes_out,
                                       bool* lake_mask_out);

#endif /* EVALUATE_BASINS_HPP_ */
