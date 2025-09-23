/*
 * evaluate_basins.cpp
 *
 *  Created on: June 25, 2018
 *      Author: thomasriddick
 */

#include "algorithms/l2_basin_evaluation_algorithm.hpp"
#include "drivers/l2_evaluate_basins.hpp"
#include "drivers/fill_sinks.hpp"
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
                                              int* lake_numbers_out,
                                              double* sinkless_rdirs_out_double,
                                              int* number_of_lakes_out_ptr,
                                              int* lake_mask_out_int,
                                              int* lakes_as_array_size){
  auto landsea_in = new bool[nlat_fine*nlon_fine];
  for (int i = 0; i < nlat_fine*nlon_fine; i++) {
    landsea_in[i] = bool(landsea_in_int[i]);
  }
  auto minima_in = new bool[nlat_fine*nlon_fine];
  for (int i = 0; i < nlat_fine*nlon_fine; i++) {
    minima_in[i] = bool(minima_in_int[i]);
  }
  short* sinkless_rdirs_out = new short[nlat_fine*nlon_fine];
  bool* lake_mask_out = new bool[nlat_fine*nlon_fine];
  fill_n(sinkless_rdirs_out,nlat_fine*nlon_fine,0);
  int number_of_lakes_out;
  vector<double>* lakes_as_vector_of_doubles =
    latlon_evaluate_basins(landsea_in,
                           minima_in,
                           raw_orography_in,
                           corrected_orography_in,
                           cell_areas_in,
                           prior_fine_rdirs_in,
                           prior_fine_catchments_in,
                           coarse_catchment_nums_in,
                           nlat_fine,nlon_fine,
                           nlat_coarse,nlon_coarse,
                           lake_numbers_out,
                           sinkless_rdirs_out,
                           number_of_lakes_out,
                           lake_mask_out);
  *number_of_lakes_out_ptr = number_of_lakes_out;
  *lakes_as_array_size = lakes_as_vector_of_doubles->size();
  double* lakes_as_array = new double[*lakes_as_array_size];
  int i = 0;
  for (double element : *lakes_as_vector_of_doubles) {
    lakes_as_array[i] = element;
    i++;
  }
  for (int i = 0; i < nlat_fine*nlon_fine; i++) {
    lake_mask_out_int[i] = int(lake_mask_out[i]);
  }
  for (auto i = 0; i < nlat_fine*nlon_fine;i++){
    sinkless_rdirs_out_double[i] = sinkless_rdirs_out[i];
  }
  delete[] sinkless_rdirs_out;
  return lakes_as_array;
}

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
                                       bool* lake_mask_out){
  cout << "Entering Basin Evaluation C++ Code" << endl;
  int scale_factor = nlat_fine/nlat_coarse;
  auto grid_params_in = new latlon_grid_params((nlat_fine+2*scale_factor),nlon_fine);
  auto coarse_grid_params_in = new latlon_grid_params(nlat_coarse+2,
                                                      nlon_coarse);
  bool* minima_in_ext = new bool[(nlat_fine+2*scale_factor)*nlon_fine];
  bool* true_sinks_ext = new bool[(nlat_fine+2*scale_factor)*nlon_fine];
  double* raw_orography_in_ext = new double[(nlat_fine+2*scale_factor)*nlon_fine];
  double* corrected_orography_in_ext = new double[(nlat_fine+2*scale_factor)*nlon_fine];
  double* cell_areas_in_ext = new double[(nlat_fine+2*scale_factor)*nlon_fine];
  int* prior_fine_rdirs_in_ext = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* prior_fine_catchments_in_ext = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* coarse_catchment_nums_in_ext = new int[(nlat_coarse+2)*nlon_coarse];
  int* next_cell_lat_index_dummy_in = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  int* next_cell_lon_index_dummy_in = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  short* sinkless_rdirs_out_ext = new short[(nlat_fine+2*scale_factor)*nlon_fine];
  fill_n(sinkless_rdirs_out_ext,(nlat_fine+2*scale_factor)*nlon_fine,0);
  int* catchments_from_sink_filling_in = new int[(nlat_fine+2*scale_factor)*nlon_fine];
  bool* true_sinks_in = new bool[(nlat_fine+2*scale_factor)*nlon_fine];
  fill_n(true_sinks_in,(nlat_fine+2*scale_factor)*nlon_fine,false);
  bool* landsea_in_ext = new bool[(nlat_fine+2*scale_factor)*nlon_fine];
  double maximum_double = std::numeric_limits<double>::max();
  for (int j = 0; j < nlon_coarse; j++) {
    coarse_catchment_nums_in_ext[j] = -1;
  }
  for (int i = 0; i < scale_factor; i++) {
    for (int j = 0; j < nlon_fine; j++){
      landsea_in_ext[i*nlon_fine+j] = false;
      minima_in_ext[i*nlon_fine+j] = false;
      true_sinks_ext[i*nlon_fine+j] = false;
      raw_orography_in_ext[i*nlon_fine+j] = maximum_double;
      corrected_orography_in_ext[i*nlon_fine+j] = maximum_double;
      cell_areas_in_ext[i*nlon_fine+j] = 1.0;
      prior_fine_rdirs_in_ext[i*nlon_fine+j] = 2.0;
      prior_fine_catchments_in_ext[i*nlon_fine+j] = -1;
    }
  }
  for (int i = 1; i < nlat_coarse+1; i++) {
    for (int j = 0; j < nlon_coarse; j++) {
      coarse_catchment_nums_in_ext[i*nlon_coarse+j] =
        coarse_catchment_nums_in[nlon_coarse*(i-1)+j];
    }
  }
  for (int i = scale_factor; i < nlat_fine+scale_factor; i++){
    for (int j = 0; j < nlon_fine; j++){
      landsea_in_ext[i*nlon_fine+j] = landsea_in[i*nlon_fine+j-scale_factor*nlon_fine];
      minima_in_ext[i*nlon_fine+j] = minima_in[i*nlon_fine+j-scale_factor*nlon_fine];
      true_sinks_ext[i*nlon_fine+j] = false;
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
    coarse_catchment_nums_in_ext[(nlat_coarse+1)*nlon_coarse + j] = -1;
  }
  for (int i = nlat_fine+scale_factor;
       i < nlat_fine+2*scale_factor; i++) {
    for (int j = 0; j < nlon_fine; j++){
      landsea_in_ext[i*nlon_fine+j] = false;
      minima_in_ext[i*nlon_fine+j] = false;
      true_sinks_ext[i*nlon_fine+j] = false;
      raw_orography_in_ext[i*nlon_fine+j] = maximum_double;
      corrected_orography_in_ext[i*nlon_fine+j] = maximum_double;
      cell_areas_in_ext[i*nlon_fine+j] = 1.0;
      prior_fine_rdirs_in_ext[i*nlon_fine+j] = 8.0;
      prior_fine_catchments_in_ext[i*nlon_fine+j] = -1;
    }
  }
  latlon_fill_sinks(corrected_orography_in_ext,nlat_fine+2*scale_factor,
                    nlon_fine,4,landsea_in_ext,false,
                    true_sinks_ext,
                    false,0.0,
                    next_cell_lat_index_dummy_in,
                    next_cell_lon_index_dummy_in,
                    sinkless_rdirs_out_ext,
                    catchments_from_sink_filling_in,false);
  latlon_basin_evaluation_algorithm alg =
    latlon_basin_evaluation_algorithm(minima_in_ext,
                                      raw_orography_in_ext,
                                      corrected_orography_in_ext,
                                      cell_areas_in_ext,
                                      prior_fine_rdirs_in_ext,
                                      prior_fine_catchments_in_ext,
                                      coarse_catchment_nums_in_ext,
                                      catchments_from_sink_filling_in,
                                      -scale_factor,
                                      grid_params_in,
                                      coarse_grid_params_in);
  alg.evaluate_basins();
  vector<lake_variables*> lakes = alg.get_lakes();
  vector<double>* lakes_as_array = alg.get_lakes_as_array();
  number_of_lakes_out = alg.get_number_of_lakes();
  int* lake_numbers_out_ext = alg.get_lake_numbers()->get_array();
  bool* lake_mask_out_ext = alg.get_lake_mask()->get_array();
  for (int i = scale_factor*nlon_fine; i < (nlat_fine+scale_factor)*nlon_fine; i++) {
    lake_numbers_out[i-scale_factor*nlon_fine] = lake_numbers_out_ext[i];
    sinkless_rdirs_out[i-scale_factor*nlon_fine] = sinkless_rdirs_out_ext[i];
    lake_mask_out[i-scale_factor*nlon_fine] = lake_mask_out_ext[i];
  }
  delete grid_params_in;
  delete coarse_grid_params_in;
  delete[] minima_in_ext;
  delete[] true_sinks_ext;
  delete[] raw_orography_in_ext;
  delete[] corrected_orography_in_ext;
  delete[] cell_areas_in_ext;
  delete[] prior_fine_rdirs_in_ext;
  delete[] prior_fine_catchments_in_ext;
  delete[] coarse_catchment_nums_in_ext;
  delete[] next_cell_lat_index_dummy_in;
  delete[] next_cell_lon_index_dummy_in;
  delete[] sinkless_rdirs_out_ext;
  delete[] catchments_from_sink_filling_in;
  delete[] true_sinks_in;
  delete[] landsea_in_ext;
  return lakes_as_array;
}
