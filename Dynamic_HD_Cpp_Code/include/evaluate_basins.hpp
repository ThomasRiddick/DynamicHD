/*
 * evaluate_basins.hpp
 *
 *  Created on: June 25, 2018
 *      Author: thomasriddick
 */

#ifndef EVALUATE_BASINS_HPP_
#define EVALUATE_BASINS_HPP_

void latlon_evaluate_basins_cython_wrapper(int* minima_in_int,
                                           int* coarse_minima_in_int,
                                           double* raw_orography_in,
                                           double* corrected_orography_in,
                                           double* connection_volume_thresholds_in,
                                           double* flood_volume_thresholds_in,
                                           double* prior_fine_rdirs_in,
                                           int* prior_fine_catchments_in,
                                           int* coarse_catchment_nums_in,
                                           int* next_cell_lat_index_in,
                                           int* next_cell_lon_index_in,
                                           int* force_merge_lat_index_in,
                                           int* force_merge_lon_index_in,
                                           int* local_redirect_lat_index_in,
                                           int* local_redirect_lon_index_in,
                                           int* non_local_redirect_lat_index_in,
                                           int* non_local_redirect_lon_index_in,
                                           int* merge_points_out_int,
                                           int nlat_fine, int nlon_fine,
                                           int nlat_coarse,int nlon_coarse);

void latlon_evaluate_basins(bool* minima_in, bool* coarse_minima_in,
                            double* raw_orography_in,
                            double* corrected_orography_in,
                            double* connection_volume_thresholds_in,
                            double* flood_volume_thresholds_in,
                            double* prior_fine_rdirs_in,
                            int* prior_fine_catchments_in,
                            int* coarse_catchment_nums_in,
                            int* next_cell_lat_index_in,
                            int* next_cell_lon_index_in,
                            int* force_merge_lat_index_in,
                            int* force_merge_lon_index_in,
                            int* local_redirect_lat_index_in,
                            int* local_redirect_lon_index_in,
                            int* non_local_redirect_lat_index_in,
                            int* non_local_redirect_lon_index_in,
                            int* merge_points_out_int,
                            int nlat_fine, int nlon_fine,
                            int nlat_coarse,int nlon_coarse);

#endif /* EVALUATE_BASINS_HPP_ */
