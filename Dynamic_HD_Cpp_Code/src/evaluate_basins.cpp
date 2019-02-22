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
  int scale_factor = nlat_fine/nlat_coarse;
  auto grid_params_in = new latlon_grid_params((nlat_fine+2*scale_factor),nlon_fine);
  auto coarse_grid_params_in = new latlon_grid_params(nlat_coarse+2,
                                                      nlon_coarse);
  merge_types* merge_points_in = new merge_types[(nlat_fine+2*scale_factor)*nlon_fine];
  bool* minima_in_ext = new bool[(nlat_fine+2*scale_factor)*nlon_fine];
  double* raw_orography_in_ext = new double[(nlat_fine+2*scale_factor)*nlon_fine];
  double* corrected_orography_in_ext = new double[(nlat_fine+2*scale_factor)*nlon_fine];
  double* connection_volume_thresholds_in_ext = new double[(nlat_fine+2*scale_factor)*nlon_fine];
  double* flood_volume_thresholds_in_ext = new double[(nlat_fine+2*scale_factor)*nlon_fine];
  double* prior_fine_rdirs_in_ext = new double[(nlat_fine+2*scale_factor)*nlon_fine];
  int* prior_fine_catchments_in_ext = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* coarse_catchment_nums_in_ext = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* flood_next_cell_lat_index_in_ext = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* flood_next_cell_lon_index_in_ext = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* connect_next_cell_lat_index_in_ext = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* connect_next_cell_lon_index_in_ext = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* flood_force_merge_lat_index_in_ext  = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* flood_force_merge_lon_index_in_ext  = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* connect_force_merge_lat_index_in_ext  = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* connect_force_merge_lon_index_in_ext  = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* flood_redirect_lat_index_in_ext  = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* flood_redirect_lon_index_in_ext  = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* connect_redirect_lat_index_in_ext  = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* connect_redirect_lon_index_in_ext  = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  bool* flood_local_redirect_in_ext  = new bool[(nlat_fine+2*scale_factor)*nlon_fine];
  bool* connect_local_redirect_in_ext  = new bool[(nlat_fine+2*scale_factor)*nlon_fine];
  auto sink_filling_alg_4 = new sink_filling_algorithm_4_latlon();
  int* next_cell_lat_index_dummy_in = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* next_cell_lon_index_dummy_in = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  short* sinkless_rdirs_dummy_out = new short[(nlat_fine+2*scale_factor)*nlon_fine];
  int* catchment_nums_dummy_in = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  bool* true_sinks_in = new bool[(nlat_fine+2*scale_factor)*nlon_fine];
  fill_n(true_sinks_in,(nlat_fine+2*scale_factor)*nlon_fine,false);
  bool* landsea_in = new bool[(nlat_fine+2*scale_factor)*nlon_fine];
  fill_n(landsea_in,(nlat_fine+2*scale_factor)*nlon_fine,false);
  double maximum_double = std::numeric_limits<double>::max();
  double lowest_double = std::numeric_limits<double>::lowest();
  for (int i = 0; i < scale_factor; i++) {
    for (int j = 0; j < nlon_fine; j++){
      landsea_in[i*nlon_fine+j] = false;
      minima_in_ext[i*nlon_fine+j] = false;
      raw_orography_in_ext[i*nlon_fine+j] = lowest_double;
      corrected_orography_in_ext[i*nlon_fine+j] = lowest_double;
      connection_volume_thresholds_in_ext[i*nlon_fine+j] = 0.0;
      flood_volume_thresholds_in_ext[i*nlon_fine+j] = 0.0;
      prior_fine_rdirs_in_ext[i*nlon_fine+j] = -1.0;
      prior_fine_catchments_in_ext[i*nlon_fine+j] = 0;
      coarse_catchment_nums_in_ext[i*nlon_fine+j] = 0;
      flood_next_cell_lat_index_in_ext[i*nlon_fine+j] = 0;
      flood_next_cell_lon_index_in_ext[i*nlon_fine+j] = 0;
      connect_next_cell_lat_index_in_ext[i*nlon_fine+j] = 0;
      connect_next_cell_lon_index_in_ext[i*nlon_fine+j] = 0;
      flood_force_merge_lat_index_in_ext[i*nlon_fine+j] = 0;
      flood_force_merge_lon_index_in_ext[i*nlon_fine+j] = 0;
      connect_force_merge_lat_index_in_ext[i*nlon_fine+j] = 0;
      connect_force_merge_lon_index_in_ext[i*nlon_fine+j] = 0;
      flood_redirect_lat_index_in_ext[i*nlon_fine+j] = 0;
      flood_redirect_lon_index_in_ext[i*nlon_fine+j] = 0;
      connect_redirect_lat_index_in_ext[i*nlon_fine+j] = 0;
      connect_redirect_lon_index_in_ext[i*nlon_fine+j] = 0;
      flood_local_redirect_in_ext[i*nlon_fine+j] = false;
      connect_local_redirect_in_ext[i*nlon_fine+j] = false;
    }
  }
  for (int i = scale_factor; i < nlat_fine+scale_factor; i++){
    for (int j = 0; j < nlon_fine; j++){
      if (prior_fine_rdirs_in[i*nlon_fine+j-scale_factor*nlon_fine] == 0.0 ||
          prior_fine_rdirs_in[i*nlon_fine+j-scale_factor*nlon_fine] == -1.0 ) landsea_in[i*nlon_fine+j] = true;
      minima_in_ext[i*nlon_fine+j] = minima_in[i*nlon_fine+j-scale_factor*nlon_fine];
      raw_orography_in_ext[i*nlon_fine+j] = raw_orography_in[i*nlon_fine+j-scale_factor*nlon_fine];
      corrected_orography_in_ext[i*nlon_fine+j] =
        corrected_orography_in[i*nlon_fine+j-scale_factor*nlon_fine];
      connection_volume_thresholds_in_ext[i*nlon_fine+j] =
        connection_volume_thresholds_in[i*nlon_fine+j-scale_factor*nlon_fine];
      flood_volume_thresholds_in_ext[i*nlon_fine+j] =
        flood_volume_thresholds_in[i*nlon_fine+j-scale_factor*nlon_fine];
      prior_fine_rdirs_in_ext[i*nlon_fine+j] =
        prior_fine_rdirs_in_ext[i*nlon_fine+j-scale_factor*nlon_fine];
      prior_fine_catchments_in_ext[i*nlon_fine+j] =
        prior_fine_catchments_in[i*nlon_fine+j-scale_factor*nlon_fine];
      coarse_catchment_nums_in_ext[i*nlon_fine+j] =
        coarse_catchment_nums_in[i*nlon_fine+j-scale_factor*nlon_fine];
      flood_next_cell_lat_index_in_ext[i*nlon_fine+j] =
        flood_next_cell_lat_index_in[i*nlon_fine+j-scale_factor*nlon_fine];
      flood_next_cell_lon_index_in_ext[i*nlon_fine+j] =
        flood_next_cell_lon_index_in[i*nlon_fine+j-scale_factor*nlon_fine];
      connect_next_cell_lat_index_in_ext[i*nlon_fine+j] =
        connect_next_cell_lat_index_in[i*nlon_fine+j-scale_factor*nlon_fine];
      connect_next_cell_lon_index_in_ext[i*nlon_fine+j] =
        connect_next_cell_lon_index_in[i*nlon_fine+j-scale_factor*nlon_fine];
      flood_force_merge_lat_index_in_ext[i*nlon_fine+j] =
        flood_force_merge_lat_index_in[i*nlon_fine+j-scale_factor*nlon_fine];
      flood_force_merge_lon_index_in_ext[i*nlon_fine+j] =
       flood_force_merge_lon_index_in[i*nlon_fine+j-scale_factor*nlon_fine];
      connect_force_merge_lat_index_in_ext[i*nlon_fine+j] =
        connect_force_merge_lat_index_in[i*nlon_fine+j-scale_factor*nlon_fine];
      connect_force_merge_lon_index_in_ext[i*nlon_fine+j] =
        connect_force_merge_lon_index_in[i*nlon_fine+j-scale_factor*nlon_fine];
      flood_redirect_lat_index_in_ext[i*nlon_fine+j] =
        flood_redirect_lat_index_in[i*nlon_fine+j-scale_factor*nlon_fine];
      flood_redirect_lon_index_in_ext[i*nlon_fine+j] =
        flood_redirect_lon_index_in[i*nlon_fine+j-scale_factor*nlon_fine];
      connect_redirect_lat_index_in_ext[i*nlon_fine+j] =
        connect_redirect_lat_index_in[i*nlon_fine+j-scale_factor*nlon_fine];
      connect_redirect_lon_index_in_ext[i*nlon_fine+j] =
        connect_redirect_lon_index_in[i*nlon_fine+j-scale_factor*nlon_fine];
      flood_local_redirect_in_ext[i*nlon_fine+j] =
        flood_local_redirect_in[i*nlon_fine+j-scale_factor*nlon_fine];
      connect_local_redirect_in_ext[i*nlon_fine+j] =
        connect_local_redirect_in[i*nlon_fine+j-scale_factor*nlon_fine];
    }
  }
  for (int i = nlat_fine+scale_factor;
       i < nlat_fine+2*scale_factor; i++) {
    for (int j = 0; j < nlon_fine; j++){
      landsea_in[i*nlon_fine+j] = true;
      minima_in_ext[i*nlon_fine+j] = false;
      raw_orography_in_ext[i*nlon_fine+j] = maximum_double;
      corrected_orography_in_ext[i*nlon_fine+j] = maximum_double;
      connection_volume_thresholds_in_ext[i*nlon_fine+j] = 0.0;
      flood_volume_thresholds_in_ext[i*nlon_fine+j] = 0.0;
      prior_fine_rdirs_in_ext[i*nlon_fine+j] = 5.0;
      prior_fine_catchments_in_ext[i*nlon_fine+j] = 0;
      coarse_catchment_nums_in_ext[i*nlon_fine+j] = 0;
      flood_next_cell_lat_index_in_ext[i*nlon_fine+j] = 0;
      flood_next_cell_lon_index_in_ext[i*nlon_fine+j] = 0;
      connect_next_cell_lat_index_in_ext[i*nlon_fine+j] = 0;
      connect_next_cell_lon_index_in_ext[i*nlon_fine+j] = 0;
      flood_force_merge_lat_index_in_ext[i*nlon_fine+j] = 0;
      flood_force_merge_lon_index_in_ext[i*nlon_fine+j] = 0;
      connect_force_merge_lat_index_in_ext[i*nlon_fine+j] = 0;
      connect_force_merge_lon_index_in_ext[i*nlon_fine+j] = 0;
      flood_redirect_lat_index_in_ext[i*nlon_fine+j] = 0;
      flood_redirect_lon_index_in_ext[i*nlon_fine+j] = 0;
      connect_redirect_lat_index_in_ext[i*nlon_fine+j] = 0;
      connect_redirect_lon_index_in_ext[i*nlon_fine+j] = 0;
      flood_local_redirect_in_ext[i*nlon_fine+j] = false;
      connect_local_redirect_in_ext[i*nlon_fine+j] = false;
    }
  }
  sink_filling_alg_4->setup_flags(false,true,false,false);
  sink_filling_alg_4->setup_fields(corrected_orography_in_ext,
                                  landsea_in,
                                  true_sinks_in,
                                  next_cell_lat_index_dummy_in,
                                  next_cell_lon_index_dummy_in,
                                  grid_params_in,
                                  sinkless_rdirs_dummy_out,
                                  catchment_nums_dummy_in);
  alg.setup_fields(minima_in_ext,
                   raw_orography_in_ext,
                   corrected_orography_in_ext,
                   connection_volume_thresholds_in_ext,
                   flood_volume_thresholds_in_ext,
                   prior_fine_rdirs_in_ext,
                   prior_fine_catchments_in_ext,
                   coarse_catchment_nums_in_ext,
                   flood_next_cell_lat_index_in_ext,
                   flood_next_cell_lon_index_in_ext,
                   connect_next_cell_lat_index_in_ext,
                   connect_next_cell_lon_index_in_ext,
                   flood_force_merge_lat_index_in_ext,
                   flood_force_merge_lon_index_in_ext,
                   connect_force_merge_lat_index_in_ext,
                   connect_force_merge_lon_index_in_ext,
                   flood_redirect_lat_index_in_ext,
                   flood_redirect_lon_index_in_ext,
                   connect_redirect_lat_index_in_ext,
                   connect_redirect_lon_index_in_ext,
                   flood_local_redirect_in_ext,
                   connect_local_redirect_in_ext,
                   merge_points_in,
                   grid_params_in,
                   coarse_grid_params_in);
  alg.setup_sink_filling_algorithm(sink_filling_alg_4);
  alg.evaluate_basins();
  for (int i = scale_factor; i < (nlat_fine+scale_factor)*nlon_fine; i++) {
    merge_points_out_int[i] = int(merge_points_in[i]);
  }
  delete grid_params_in;
  delete coarse_grid_params_in;
}
