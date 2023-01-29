/*
 * evaluate_basins.cpp
 *
 *  Created on: June 25, 2018
 *      Author: thomasriddick
 */

#include "base/enums.hpp"
#include "algorithms/basin_evaluation_algorithm.hpp"
#include "drivers/evaluate_basins.hpp"
#if USE_NETCDFCPP
#include <netcdf>
#endif
using namespace std;
#if USE_NETCDFCPP
using namespace netCDF;
#endif

void latlon_evaluate_basins_cython_wrapper(int* minima_in_int,
                                           double* raw_orography_in,
                                           double* corrected_orography_in,
                                           double* cell_areas_in,
                                           double* connection_volume_thresholds_in,
                                           double* flood_volume_thresholds_in,
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
                                           int* basin_catchment_numbers_in){
  auto minima_in = new bool[nlat_fine*nlon_fine];
  for (int i = 0; i < nlat_fine*nlon_fine; i++) {
    minima_in[i] = bool(minima_in_int[i]);
  }
  latlon_evaluate_basins(minima_in,
                         raw_orography_in,
                         corrected_orography_in,
                         cell_areas_in,
                         connection_volume_thresholds_in,
                         flood_volume_thresholds_in,
                         prior_fine_rdirs_in,
                         prior_coarse_rdirs_in,
                         prior_fine_catchments_in,
                         coarse_catchment_nums_in,
                         flood_next_cell_lat_index_in,
                         flood_next_cell_lon_index_in,
                         connect_next_cell_lat_index_in,
                         connect_next_cell_lon_index_in,
                         connect_merge_and_redirect_indices_index_in,
                         flood_merge_and_redirect_indices_index_in,
                         nlat_fine, nlon_fine,
                         nlat_coarse,nlon_coarse,
                         merges_filepath,
                         basin_catchment_numbers_in);
}

void latlon_evaluate_basins(bool* minima_in,
                            double* raw_orography_in,
                            double* corrected_orography_in,
                            double* cell_areas_in,
                            double* connection_volume_thresholds_in,
                            double* flood_volume_thresholds_in,
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
                            int* basin_catchment_numbers_in){
  cout << "Entering Basin Evaluation C++ Code" << endl;
  auto alg = latlon_basin_evaluation_algorithm();
  int scale_factor = nlat_fine/nlat_coarse;
  auto grid_params_in = new latlon_grid_params((nlat_fine+2*scale_factor),nlon_fine);
  auto coarse_grid_params_in = new latlon_grid_params(nlat_coarse+2,
                                                      nlon_coarse);
  bool* minima_in_ext = new bool[(nlat_fine+2*scale_factor)*nlon_fine];
  double* raw_orography_in_ext = new double[(nlat_fine+2*scale_factor)*nlon_fine];
  double* corrected_orography_in_ext = new double[(nlat_fine+2*scale_factor)*nlon_fine];
  double* cell_areas_in_ext = new double[(nlat_fine+2*scale_factor)*nlon_fine];
  double* connection_volume_thresholds_in_ext = new double[(nlat_fine+2*scale_factor)*nlon_fine];
  fill_n(connection_volume_thresholds_in_ext,(nlat_fine+2*scale_factor)*nlon_fine,0.0);
  double* flood_volume_thresholds_in_ext = new double[(nlat_fine+2*scale_factor)*nlon_fine];
  fill_n(flood_volume_thresholds_in_ext,(nlat_fine+2*scale_factor)*nlon_fine,0.0);
  double* prior_fine_rdirs_in_ext = new double[(nlat_fine+2*scale_factor)*nlon_fine];
  int* prior_fine_catchments_in_ext = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* coarse_catchment_nums_in_ext = new int[(nlat_coarse+2)*nlon_coarse];
  double* prior_coarse_rdirs_in_ext = new double[(nlat_coarse+2)*nlon_coarse];
  int* flood_next_cell_lat_index_in_ext = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  fill_n(flood_next_cell_lat_index_in_ext,(nlat_fine+2*scale_factor)*nlon_fine,-1);
  int* flood_next_cell_lon_index_in_ext = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  fill_n(flood_next_cell_lon_index_in_ext,(nlat_fine+2*scale_factor)*nlon_fine,-1);
  int* connect_next_cell_lat_index_in_ext = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  fill_n(connect_next_cell_lat_index_in_ext,(nlat_fine+2*scale_factor)*nlon_fine,-1);
  int* connect_next_cell_lon_index_in_ext = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  fill_n(connect_next_cell_lon_index_in_ext,(nlat_fine+2*scale_factor)*nlon_fine,-1);
  int* basin_catchment_numbers_in_ext = nullptr;
  if (basin_catchment_numbers_in){
      basin_catchment_numbers_in_ext  = new int[(nlat_fine+2*scale_factor)*nlon_fine];
      fill_n(basin_catchment_numbers_in_ext,(nlat_fine+2*scale_factor)*nlon_fine,-1);
  }
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
  for (int j = 0; j < nlon_coarse; j++) {
    coarse_catchment_nums_in_ext[j] = 0;
    prior_coarse_rdirs_in_ext[j] = -1.0;
  }
  for (int i = 0; i < scale_factor; i++) {
    for (int j = 0; j < nlon_fine; j++){
      landsea_in[i*nlon_fine+j] = true;
      minima_in_ext[i*nlon_fine+j] = false;
      raw_orography_in_ext[i*nlon_fine+j] = lowest_double;
      corrected_orography_in_ext[i*nlon_fine+j] = lowest_double;
      cell_areas_in_ext[i*nlon_fine+j] = 1.0;
      prior_fine_rdirs_in_ext[i*nlon_fine+j] = -1.0;
      prior_fine_catchments_in_ext[i*nlon_fine+j] = 0;
    }
  }
  for (int i = 1; i < nlat_coarse+1; i++) {
    for (int j = 0; j < nlon_coarse; j++) {
      coarse_catchment_nums_in_ext[i*nlon_coarse+j] =
        coarse_catchment_nums_in[nlon_coarse*(i-1)+j];
      prior_coarse_rdirs_in_ext[i*nlon_coarse+j] =
        prior_coarse_rdirs_in[nlon_coarse*(i-1)+j];
    }
  }
  for (int i = scale_factor; i < nlat_fine+scale_factor; i++){
    for (int j = 0; j < nlon_fine; j++){
      if (prior_fine_rdirs_in[i*nlon_fine+j-scale_factor*nlon_fine] == 0.0 ||
          prior_fine_rdirs_in[i*nlon_fine+j-scale_factor*nlon_fine] == -1.0 ) landsea_in[i*nlon_fine+j] = true;
      minima_in_ext[i*nlon_fine+j] = minima_in[i*nlon_fine+j-scale_factor*nlon_fine];
      raw_orography_in_ext[i*nlon_fine+j] = raw_orography_in[i*nlon_fine+j-scale_factor*nlon_fine];
      cell_areas_in_ext[i*nlon_fine+j] = cell_areas_in[i*nlon_fine+j-scale_factor*nlon_fine];
      corrected_orography_in_ext[i*nlon_fine+j] =
        corrected_orography_in[i*nlon_fine+j-scale_factor*nlon_fine];
      prior_fine_rdirs_in_ext[i*nlon_fine+j] =
        prior_fine_rdirs_in[i*nlon_fine+j-scale_factor*nlon_fine];
      prior_fine_catchments_in_ext[i*nlon_fine+j] =
        prior_fine_catchments_in[i*nlon_fine+j-scale_factor*nlon_fine];
    }
  }
  for (int j = 0; j < nlon_coarse; j++){
    coarse_catchment_nums_in_ext[(nlat_coarse+1)*nlon_coarse + j] = 0;
    prior_coarse_rdirs_in_ext[(nlat_coarse+1)*nlon_coarse + j] = 5.0;
  }
  for (int i = nlat_fine+scale_factor;
       i < nlat_fine+2*scale_factor; i++) {
    for (int j = 0; j < nlon_fine; j++){
      landsea_in[i*nlon_fine+j] = false;
      minima_in_ext[i*nlon_fine+j] = false;
      raw_orography_in_ext[i*nlon_fine+j] = maximum_double;
      corrected_orography_in_ext[i*nlon_fine+j] = maximum_double;
      cell_areas_in_ext[i*nlon_fine+j] = 1.0;
      prior_fine_rdirs_in_ext[i*nlon_fine+j] = 5.0;
      prior_fine_catchments_in_ext[i*nlon_fine+j] = 0;
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
                   cell_areas_in_ext,
                   connection_volume_thresholds_in_ext,
                   flood_volume_thresholds_in_ext,
                   prior_fine_rdirs_in_ext,
                   prior_coarse_rdirs_in_ext,
                   prior_fine_catchments_in_ext,
                   coarse_catchment_nums_in_ext,
                   flood_next_cell_lat_index_in_ext,
                   flood_next_cell_lon_index_in_ext,
                   connect_next_cell_lat_index_in_ext,
                   connect_next_cell_lon_index_in_ext,
                   grid_params_in,
                   coarse_grid_params_in);
  alg.setup_sink_filling_algorithm(sink_filling_alg_4);
  alg.evaluate_basins();
  merges_and_redirects* merges_and_redirects_out =
    alg.get_basin_merges_and_redirects();
  int* flood_merge_and_redirect_indices_index_in_ext =
    merges_and_redirects_out->get_flood_merge_and_redirect_indices_index()->get_array();
  int* connect_merge_and_redirect_indices_index_in_ext =
    merges_and_redirects_out->get_connect_merge_and_redirect_indices_index()->get_array();
  #if USE_NETCDFCPP
  pair<tuple<int,int,int>*,int*>* array_and_dimensions =
    merges_and_redirects_out->get_merges_and_redirects_as_array(true);
  NcFile merges_and_redirects_file(merges_filepath.c_str(), NcFile::newFile);
  NcDim first_index =
    merges_and_redirects_file.addDim("first_index",get<0>(*array_and_dimensions->first));
  NcDim second_index =
    merges_and_redirects_file.addDim("second_index",get<1>(*array_and_dimensions->first));
  NcDim third_index =
    merges_and_redirects_file.addDim("third_index",get<2>(*array_and_dimensions->first));
  vector<NcDim> dims;
  vector.push_back(first_index);
  vector.push_back(second_index);
  vector.push_back(third_index);
  NcVar flood_merges_and_redirects_out_var =
    merges_and_redirects_file.addVar("flood_merges_and_redirects",ncInt,dims);
  flood_merges_and_redirects_out_var.putVar(array_and_dimensions->second);
  array_and_dimensions =
    merges_and_redirects_out->get_merges_and_redirects_as_array(false);
  NcVar connect_merges_and_redirects_out_var =
  merges_and_redirects_file.addVar("connect_merges_and_redirects",ncInt,dims);
  connect_merges_and_redirects_out_var.putVar(array_and_dimensions->second);
  #endif
  if(basin_catchment_numbers_in){
    basin_catchment_numbers_in_ext = alg.retrieve_lake_numbers();
  }
  for (int i = scale_factor*nlon_fine; i < (nlat_fine+scale_factor)*nlon_fine; i++) {
    connection_volume_thresholds_in[i-scale_factor*nlon_fine] = connection_volume_thresholds_in_ext[i];
    flood_volume_thresholds_in[i-scale_factor*nlon_fine] = flood_volume_thresholds_in_ext[i];
    flood_next_cell_lat_index_in[i-scale_factor*nlon_fine]  =
      max(flood_next_cell_lat_index_in_ext[i] - scale_factor,-1);
    flood_next_cell_lon_index_in[i-scale_factor*nlon_fine]  = flood_next_cell_lon_index_in_ext[i];
    connect_next_cell_lat_index_in[i-scale_factor*nlon_fine] =
      max(connect_next_cell_lat_index_in_ext[i] - scale_factor,-1);
    connect_next_cell_lon_index_in[i-scale_factor*nlon_fine] = connect_next_cell_lon_index_in_ext[i];
    connect_merge_and_redirect_indices_index_in[i-scale_factor*nlon_fine] =
      connect_merge_and_redirect_indices_index_in_ext[i];
    flood_merge_and_redirect_indices_index_in[i-scale_factor*nlon_fine] =
      flood_merge_and_redirect_indices_index_in_ext[i];
    if(basin_catchment_numbers_in){
      basin_catchment_numbers_in[i-scale_factor*nlon_fine] =
        basin_catchment_numbers_in_ext[i];
    }
  }
  delete grid_params_in;
  delete coarse_grid_params_in;
}

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
                                       int* basin_catchment_numbers_in){
  cout << "Entering Basin Evaluation C++ Code" << endl;
  auto grid_params_in = new icon_single_index_grid_params(ncells_fine_in,
                                                          fine_neighboring_cell_indices_in,true,
                                                          fine_secondary_neighboring_cell_indices_in);
  auto coarse_grid_params_in = new icon_single_index_grid_params(ncells_coarse_in,
                                                                 coarse_neighboring_cell_indices_in,true,
                                                                 coarse_secondary_neighboring_cell_indices_in);
  grid_params_in->set_mapping_to_coarse_grid(mapping_from_fine_to_coarse_grid);
  auto alg = icon_single_index_basin_evaluation_algorithm();
  fill_n(connection_volume_thresholds_in,ncells_fine_in,0.0);
  fill_n(flood_volume_thresholds_in,ncells_fine_in,0.0);
  fill_n(flood_next_cell_index_in,ncells_fine_in,-1);
  fill_n(connect_next_cell_index_in,ncells_fine_in,-1);
  auto sink_filling_alg_4 = new sink_filling_algorithm_4_icon_single_index();
  int* next_cell_index_dummy_in = new int[ncells_fine_in];
  int* catchment_nums_dummy_in = new int[ncells_fine_in];
  bool* true_sinks_in = new bool[ncells_fine_in];
  fill_n(true_sinks_in,ncells_fine_in,false);
  bool* landsea_in = new bool[ncells_fine_in];
  fill_n(landsea_in,ncells_fine_in,false);
  for (int i = 0; i < ncells_fine_in; i++){
    if (prior_fine_rdirs_in[i] == 0 ||
        prior_fine_rdirs_in[i] == -1 ) landsea_in[i] = true;
  }
  sink_filling_alg_4->setup_flags(false,true,false,false);
  sink_filling_alg_4->setup_fields(corrected_orography_in,
                                  landsea_in,
                                  true_sinks_in,
                                  next_cell_index_dummy_in,
                                  grid_params_in,
                                  catchment_nums_dummy_in);
  alg.setup_fields(minima_in,
                   raw_orography_in,
                   corrected_orography_in,
                   cell_areas_in,
                   connection_volume_thresholds_in,
                   flood_volume_thresholds_in,
                   prior_fine_rdirs_in,
                   prior_coarse_rdirs_in,
                   prior_fine_catchments_in,
                   coarse_catchment_nums_in,
                   flood_next_cell_index_in,
                   connect_next_cell_index_in,
                   grid_params_in,
                   coarse_grid_params_in);
  alg.setup_sink_filling_algorithm(sink_filling_alg_4);
  alg.evaluate_basins();
  if(basin_catchment_numbers_in){
    std::copy_n(alg.retrieve_lake_numbers(),ncells_fine_in,
                basin_catchment_numbers_in);
  }
  delete grid_params_in;
  delete coarse_grid_params_in;
  delete sink_filling_alg_4;
}
