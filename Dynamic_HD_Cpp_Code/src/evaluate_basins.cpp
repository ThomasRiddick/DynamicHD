/*
 * evaluate_basins.cpp
 *
 *  Created on: June 25, 2018
 *      Author: thomasriddick
 */

#include "evaluate_basins.hpp"
#include "basin_evaluation_algorithm.hpp"
#include "enums.hpp"

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
                                           int* connect_redirect_lat_index_in,
                                           int* connect_redirect_lon_index_in,
                                           int* flood_local_redirect_out_int,
                                           int* connect_local_redirect_out_int,
                                           int* merge_points_out_int,
                                           int nlat_fine, int nlon_fine,
                                           int nlat_coarse,int nlon_coarse){
  auto minima_in = new bool[nlat_fine*nlon_fine];
  for (int i = 0; i < nlat_fine*nlon_fine; i++) {
    minima_in[i] = bool(minima_in_int[i]);
  }
  auto flood_local_redirect_in = new bool[nlat_fine*nlon_fine];
  std::fill_n(flood_local_redirect_in,nlat_fine*nlon_fine,false);
  auto connect_local_redirect_in = new bool[nlat_fine*nlon_fine];
  std::fill_n(connect_local_redirect_in,nlat_fine*nlon_fine,false);
  latlon_evaluate_basins(minima_in,
                         raw_orography_in,
                         corrected_orography_in,
                         connection_volume_thresholds_in,
                         flood_volume_thresholds_in,
                         prior_fine_rdirs_in,
                         prior_fine_catchments_in,
                         coarse_catchment_nums_in,
                         flood_next_cell_lat_index_in,
                         flood_next_cell_lon_index_in,
                         connect_next_cell_lat_index_in,
                         connect_next_cell_lon_index_in,
                         flood_force_merge_lat_index_in,
                         flood_force_merge_lon_index_in,
                         connect_force_merge_lat_index_in,
                         connect_force_merge_lon_index_in,
                         flood_redirect_lat_index_in,
                         flood_redirect_lon_index_in,
                         connect_redirect_lat_index_in,
                         connect_redirect_lon_index_in,
                         flood_local_redirect_in,
                         connect_local_redirect_in,
                         merge_points_out_int,
                         nlat_fine, nlon_fine,
                         nlat_coarse,nlon_coarse);
  for (int i = 0; i < nlat_fine*nlon_fine; i++) {
    flood_local_redirect_out_int[i] = int(flood_local_redirect_in[i]);
    connect_local_redirect_out_int[i] = int(connect_local_redirect_in[i]);
  }
}

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
                            int nlat_coarse,int nlon_coarse){
  cout << "Entering Basin Evaluation C++ Code" << endl;
  auto alg = latlon_basin_evaluation_algorithm();
  auto grid_params_in = new latlon_grid_params(nlat_fine,nlon_fine);
  auto coarse_grid_params_in = new latlon_grid_params(nlat_coarse,
                                                      nlon_coarse);
  merge_types* merge_points_in = new merge_types[nlat_fine*nlon_fine];
  cout << "nlat" << nlat_fine << endl;
  cout << "nlon" << nlon_fine << endl;
  cout << "nlat coarse" << nlat_coarse << endl;
  cout << "nlon coarse" << nlon_coarse << endl;
  alg.setup_fields(minima_in,
                   raw_orography_in,
                   corrected_orography_in,
                   connection_volume_thresholds_in,
                   flood_volume_thresholds_in,
                   prior_fine_rdirs_in,
                   prior_fine_catchments_in,
                   coarse_catchment_nums_in,
                   flood_next_cell_lat_index_in,
                   flood_next_cell_lon_index_in,
                   connect_next_cell_lat_index_in,
                   connect_next_cell_lon_index_in,
                   flood_force_merge_lat_index_in,
                   flood_force_merge_lon_index_in,
                   connect_force_merge_lat_index_in,
                   connect_force_merge_lon_index_in,
                   flood_redirect_lat_index_in,
                   flood_redirect_lon_index_in,
                   connect_redirect_lat_index_in,
                   connect_redirect_lon_index_in,
                   flood_local_redirect_in,
                   connect_local_redirect_in,
                   merge_points_in,
                   grid_params_in,
                   coarse_grid_params_in);
  alg.evaluate_basins();
  for (int i = 0; i < nlat_fine*nlon_fine; i++) {
    merge_points_out_int[i] = int(merge_points_in[i]);
  }
  delete grid_params_in;
  delete coarse_grid_params_in;
}
