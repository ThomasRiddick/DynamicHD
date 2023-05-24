/*
 * evaluate_basins.hpp
 *
 *  Created on: June 25, 2018
 *      Author: thomasriddick
 */

#ifndef EVALUATE_BASINS_HPP_
#define EVALUATE_BASINS_HPP_
#include <iostream>
using namespace std;

void latlon_evaluate_basins_cython_wrapper(int* minima_in_int,
                                           double* raw_orography_in,
                                           double* corrected_orography_in,
                                           double* cell_areas_in,
                                           double* connection_volume_thresholds_in,
                                           double* flood_volume_thresholds_in,
                                           double* connection_heights_in,
                                           double* flood_heights_in,
                                           double* prior_fine_rdirs_in,
                                           double* prior_coarse_rdirs_in,
                                           int* prior_fine_catchments_in,
                                           int* coarse_catchment_nums_in,
                                           int* flood_next_cell_lat_index_in,
                                           int* flood_next_cell_lon_index_in,
                                           int* connect_next_cell_lat_index_in,
                                           int* connect_next_cell_lon_index_in,
                                           int* connect_merge_and_redirect_indices_index_in,
                                           int* flood_merge_and_redirect_indices_index_in,
                                           int nlat_fine, int nlon_fine,
                                           int nlat_coarse,int nlon_coarse,
                                           string merges_filepath,
                                           int* basin_catchment_numbers_in=nullptr);

void latlon_evaluate_basins(bool* minima_in,
                            double* raw_orography_in,
                            double* corrected_orography_in,
                            double* cell_areas_in,
                            double* connection_volume_thresholds_in,
                            double* flood_volume_thresholds_in,
                            double* connection_heights_in,
                            double* flood_heights_in,
                            double* prior_fine_rdirs_in,
                            double* prior_coarse_rdirs_in,
                            int* prior_fine_catchments_in,
                            int* coarse_catchment_nums_in,
                            int* flood_next_cell_lat_index_in,
                            int* flood_next_cell_lon_index_in,
                            int* connect_next_cell_lat_index_in,
                            int* connect_next_cell_lon_index_in,
                            int* connect_merge_and_redirect_indices_index_in,
                            int* flood_merge_and_redirect_indices_index_in,
                            int nlat_fine, int nlon_fine,
                            int nlat_coarse,int nlon_coarse,
                            string merges_filepath,
                            int* basin_catchment_numbers_in=nullptr);

void icon_single_index_evaluate_basins(bool* minima_in,
                                       double* raw_orography_in,
                                       double* corrected_orography_in,
                                       double* cell_areas_in,
                                       double* connection_volume_thresholds_in,
                                       double* flood_volume_thresholds_in,
                                       int* prior_fine_rdirs_in,
                                       int* prior_coarse_rdirs_in,
                                       int* prior_fine_catchments_in,
                                       int* coarse_catchment_nums_in,
                                       int* flood_next_cell_index_in,
                                       int* connect_next_cell_index_in,
                                       int ncells_fine_in,
                                       int ncells_coarse_in,
                                       int* fine_neighboring_cell_indices_in,
                                       int* coarse_neighboring_cell_indices_in,
                                       int* fine_secondary_neighboring_cell_indices_in,
                                       int* coarse_secondary_neighboring_cell_indices_in,
                                       int* mapping_from_fine_to_coarse_grid,
                                       int* basin_catchment_numbers_in = nullptr);

#endif /* EVALUATE_BASINS_HPP_ */
