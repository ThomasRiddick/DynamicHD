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
                                           int nlat_coarse,int nlon_coarse){
  auto minima_in = new bool[nlat_fine*nlon_fine];
  auto coarse_minima_in = new bool[nlat_coarse*nlon_coarse];
  for (int i = 0; i < nlat_fine*nlon_fine; i++) {
    minima_in[i] = bool(minima_in_int[i]);
  }
  for (int i = 0; i < nlat_coarse*nlon_coarse; i++) {
    coarse_minima_in[i] = bool(coarse_minima_in_int[i]);
  }
}

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
                            int nlat_coarse,int nlon_coarse){
  cout << "Entering Basin Evaluation C++ Code" << endl;
  auto alg = latlon_basin_evaluation_algorithm();
  auto grid_params_in = new latlon_grid_params(nlat_fine,nlon_fine);
  auto coarse_grid_params_in = new latlon_grid_params(nlat_coarse,
                                                      nlon_coarse);
  merge_types* merge_points_in = new merge_types[nlat_fine*nlon_fine];
  alg.setup_fields(minima_in,coarse_minima_in,
                   raw_orography_in,
                   corrected_orography_in,
                   connection_volume_thresholds_in,
                   flood_volume_thresholds_in,
                   prior_fine_rdirs_in,
                   prior_fine_catchments_in,
                   coarse_catchment_nums_in,
                   next_cell_lat_index_in,
                   next_cell_lon_index_in,
                   force_merge_lat_index_in,
                   force_merge_lon_index_in,
                   local_redirect_lat_index_in,
                   local_redirect_lon_index_in,
                   non_local_redirect_lat_index_in,
                   non_local_redirect_lon_index_in,
                   merge_points_in,
                   grid_params_in,
                   coarse_grid_params_in);
  alg.evaluate_basins();
  for (int i = 0; i < nlat_fine*nlon_fine; i++) {
    switch(merge_points_in[i]) {
      case merge_as_primary:
        merge_points_out_int[i] = 0;
        break;
      case merge_as_secondary:
        merge_points_out_int[i] = 1;
        break;
      case null_mtype:
      default:
        merge_points_out_int[i] = 2;
        break;
    }
  }
  delete grid_params_in;
  delete coarse_grid_params_in;
}
