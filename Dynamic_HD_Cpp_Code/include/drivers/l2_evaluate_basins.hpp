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

// void latlon_evaluate_basins_cython_wrapper(int* minima_in_int,
//                                            double* raw_orography_in,
//                                            double* corrected_orography_in,
//                                            double* cell_areas_in,
//                                            double* connection_volume_thresholds_in,
//                                            double* flood_volume_thresholds_in,
//                                            double* connection_heights_in,
//                                            double* flood_heights_in,
//                                            double* prior_fine_rdirs_in,
//                                            double* prior_coarse_rdirs_in,
//                                            int* prior_fine_catchments_in,
//                                            int* coarse_catchment_nums_in,
//                                            int* flood_next_cell_lat_index_in,
//                                            int* flood_next_cell_lon_index_in,
//                                            int* connect_next_cell_lat_index_in,
//                                            int* connect_next_cell_lon_index_in,
//                                            int* connect_merge_and_redirect_indices_index_in,
//                                            int* flood_merge_and_redirect_indices_index_in,
//                                            int nlat_fine, int nlon_fine,
//                                            int nlat_coarse,int nlon_coarse,
//                                            int* flood_merges_and_redirects_in,
//                                            int* connect_merges_and_redirects_in,
//                                            int* flood_merges_and_redirects_dims_in,
//                                            int* connect_merges_and_redirects_dims_in,
//                                            int* basin_catchment_numbers_in=nullptr,
//                                            int* sinkless_rdirs_in=nullptr);

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
