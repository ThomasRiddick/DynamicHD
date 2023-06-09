'''Wrap C++ orography upscaling code using Cython'''
import cython
from Cython.Shadow import bint
cimport numpy as np
import numpy as np
from libcpp cimport bool
from libcpp.string cimport string

cdef extern from 'drivers/evaluate_basins.cpp':
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
                                               string merges_filepath)
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
                                               int* basin_catchment_numbers_in)

def evaluate_basins(np.ndarray[int,ndim=2,mode='c'] minima_in_int,
                    np.ndarray[double,ndim=2,mode='c'] raw_orography_in,
                    np.ndarray[double,ndim=2,mode='c'] corrected_orography_in,
                    np.ndarray[double,ndim=2,mode='c'] cell_areas_in,
                    np.ndarray[double,ndim=2,mode='c'] connection_volume_thresholds_in,
                    np.ndarray[double,ndim=2,mode='c'] flood_volume_thresholds_in,
                    np.ndarray[double,ndim=2,mode='c'] connection_heights_in,
                    np.ndarray[double,ndim=2,mode='c'] flood_heights_in,
                    np.ndarray[double,ndim=2,mode='c'] prior_fine_rdirs_in,
                    np.ndarray[double,ndim=2,mode='c'] prior_coarse_rdirs_in,
                    np.ndarray[int,ndim=2,mode='c'] prior_fine_catchments_in,
                    np.ndarray[int,ndim=2,mode='c'] coarse_catchment_nums_in,
                    np.ndarray[int,ndim=2,mode='c'] flood_next_cell_lat_index_in,
                    np.ndarray[int,ndim=2,mode='c'] flood_next_cell_lon_index_in,
                    np.ndarray[int,ndim=2,mode='c'] connect_next_cell_lat_index_in,
                    np.ndarray[int,ndim=2,mode='c'] connect_next_cell_lon_index_in,
                    np.ndarray[int,ndim=2,mode='c'] connect_merge_and_redirect_indices_index_in,
                    np.ndarray[int,ndim=2,mode='c'] flood_merge_and_redirect_indices_index_in,
                    str merges_filepath,
                    np.ndarray[int,ndim=2,mode='c'] basin_catchment_numbers_in=None):
    cdef int nlat_fine,nlon_fine
    nlat_fine, nlon_fine = raw_orography_in.shape[0],raw_orography_in.shape[1]
    cdef int nlat_coarse,nlon_coarse
    nlat_coarse, nlon_coarse = coarse_catchment_nums_in.shape[0],coarse_catchment_nums_in.shape[1]
    cdef string merges_filepath_c = string(bytes(merges_filepath,'utf-8'))
    if basin_catchment_numbers_in is None:
      latlon_evaluate_basins_cython_wrapper(&minima_in_int[0,0],
                                      &raw_orography_in[0,0],
                                      &corrected_orography_in[0,0],
                                      &cell_areas_in[0,0],
                                      &connection_volume_thresholds_in[0,0],
                                      &flood_volume_thresholds_in[0,0],
                                      &connection_heights_in[0,0],
                                      &flood_heights_in[0,0],
                                      &prior_fine_rdirs_in[0,0],
                                      &prior_coarse_rdirs_in[0,0],
                                      &prior_fine_catchments_in[0,0],
                                      &coarse_catchment_nums_in[0,0],
                                      &flood_next_cell_lat_index_in[0,0],
                                      &flood_next_cell_lon_index_in[0,0],
                                      &connect_next_cell_lat_index_in[0,0],
                                      &connect_next_cell_lon_index_in[0,0],
                                      &connect_merge_and_redirect_indices_index_in[0,0],
                                      &flood_merge_and_redirect_indices_index_in[0,0],
                                      nlat_fine, nlon_fine,
                                      nlat_coarse, nlon_coarse,
                                      merges_filepath_c)
    else:
      latlon_evaluate_basins_cython_wrapper(&minima_in_int[0,0],
                                            &raw_orography_in[0,0],
                                            &corrected_orography_in[0,0],
                                            &cell_areas_in[0,0],
                                            &connection_volume_thresholds_in[0,0],
                                            &flood_volume_thresholds_in[0,0],
                                            &connection_heights_in[0,0],
                                            &flood_heights_in[0,0],
                                            &prior_fine_rdirs_in[0,0],
                                            &prior_coarse_rdirs_in[0,0],
                                            &prior_fine_catchments_in[0,0],
                                            &coarse_catchment_nums_in[0,0],
                                            &flood_next_cell_lat_index_in[0,0],
                                            &flood_next_cell_lon_index_in[0,0],
                                            &connect_next_cell_lat_index_in[0,0],
                                            &connect_next_cell_lon_index_in[0,0],
                                            &connect_merge_and_redirect_indices_index_in[0,0],
                                            &flood_merge_and_redirect_indices_index_in[0,0],
                                            nlat_fine, nlon_fine,
                                            nlat_coarse, nlon_coarse,
                                            merges_filepath_c,
                                            &basin_catchment_numbers_in[0,0])
