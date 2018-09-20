/*
 * evaluate_basins.hpp
 *
 *  Created on: June 25, 2018
 *      Author: thomasriddick
 */

#ifndef EVALUATE_BASINS_HPP_
#define EVALUATE_BASINS_HPP_

void latlon_evaluate_basins_cython_wrapper(int* minima_in_int,
                                           double* raw_orography_in,
                                           double* corrected_orography_in,
                                           double* connection_volume_thresholds_in,
                                           double* flood_volume_thresholds_in,
                                           double* prior_fine_rdirs_in,
                                           int* prior_fine_catchments_in,
                                           int* coarse_catchment_nums_in,
                                           int* flood_next_cell_lat_index_in,
                                           int* flood_next_cell_lon_index_in,
                                           int* connect_next_cell_lat_index_in,
                                           int* connect_next_cell_lon_index_in,
                                           int* flood_force_merge_lat_index_in,
                                           int* flood_force_merge_lon_index_in,
                                           int* connect_force_merge_lat_index_in,
                                           int* connect_force_merge_lon_index_in,
                                           int* flood_redirect_lat_index_in,
                                           int* flood_redirect_lon_index_in,
                                           int* connect_local_redirect_lat_index_in,
                                           int* connect_local_redirect_lon_index_in,
                                           int* flood_local_redirect_out_int,
                                           int* connect_local_redirect_out_int,
                                           int* merge_points_out_int,
                                           int nlat_fine, int nlon_fine,
                                           int nlat_coarse,int nlon_coarse);

void latlon_evaluate_basins(bool* minima_in,
                            double* raw_orography_in,
                            double* corrected_orography_in,
                            double* connection_volume_thresholds_in,
                            double* flood_volume_thresholds_in,
                            double* prior_fine_rdirs_in,
                            int* prior_fine_catchments_in,
                            int* coarse_catchment_nums_in,
                            int* flood_next_cell_lat_index_in,
                            int* flood_next_cell_lon_index_in,
                            int* connect_next_cell_lat_index_in,
                            int* connect_next_cell_lon_index_in,
                            int* flood_force_merge_lat_index_in,
                            int* flood_force_merge_lon_index_in,
                            int* connect_force_merge_lat_index_in,
                            int* connect_force_merge_lon_index_in,
                            int* flood_redirect_lat_index_in,
                            int* flood_redirect_lon_index_in,
                            int* connect_redirect_lat_index_in,
                            int* connect_redirect_lon_index_in,
                            bool* flood_local_redirect_in,
                            bool* connect_local_redirect_in,
                            int* merge_points_out_int,
                            int nlat_fine, int nlon_fine,
                            int nlat_coarse,int nlon_coarse);

#endif /* EVALUATE_BASINS_HPP_ */
