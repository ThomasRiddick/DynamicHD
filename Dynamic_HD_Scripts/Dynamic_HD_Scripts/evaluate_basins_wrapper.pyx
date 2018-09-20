'''Wrap C++ orography upscaling code using Cython'''
import cython
from Cython.Shadow import bint
cimport numpy as np
import numpy as np
from libcpp cimport bool

cdef extern from 'evaluate_basins.cpp':
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
                                               int nlat_coarse,int nlon_coarse)

def evaluate_basins(np.ndarray[int,ndim=2,mode='c'] minima_in_int,
                    np.ndarray[double,ndim=2,mode='c'] raw_orography_in,
                    np.ndarray[double,ndim=2,mode='c'] corrected_orography_in,
                    np.ndarray[double,ndim=2,mode='c'] connection_volume_thresholds_in,
                    np.ndarray[double,ndim=2,mode='c'] flood_volume_thresholds_in,
                    np.ndarray[double,ndim=2,mode='c'] prior_fine_rdirs_in,
                    np.ndarray[int,ndim=2,mode='c'] prior_fine_catchments_in,
                    np.ndarray[int,ndim=2,mode='c'] coarse_catchment_nums_in,
                    np.ndarray[int,ndim=2,mode='c'] flood_next_cell_lat_index_in,
                    np.ndarray[int,ndim=2,mode='c'] flood_next_cell_lon_index_in,
                    np.ndarray[int,ndim=2,mode='c'] connect_next_cell_lat_index_in,
                    np.ndarray[int,ndim=2,mode='c'] connect_next_cell_lon_index_in,
                    np.ndarray[int,ndim=2,mode='c'] flood_force_merge_lat_index_in,
                    np.ndarray[int,ndim=2,mode='c'] flood_force_merge_lon_index_in,
                    np.ndarray[int,ndim=2,mode='c'] connect_force_merge_lat_index_in,
                    np.ndarray[int,ndim=2,mode='c'] connect_force_merge_lon_index_in,
                    np.ndarray[int,ndim=2,mode='c'] flood_redirect_lat_index_in,
                    np.ndarray[int,ndim=2,mode='c'] flood_redirect_lon_index_in,
                    np.ndarray[int,ndim=2,mode='c'] connect_local_redirect_lat_index_in,
                    np.ndarray[int,ndim=2,mode='c'] connect_local_redirect_lon_index_in,
                    np.ndarray[int,ndim=2,mode='c'] flood_local_redirect_out_int,
                    np.ndarray[int,ndim=2,mode='c'] connect_local_redirect_out_int,
                    np.ndarray[int,ndim=2,mode='c'] merge_points_out_int):
    cdef int nlat_fine,nlon_fine
    nlat_fine, nlon_fine = raw_orography_in.shape[0],raw_orography_in.shape[1]
    cdef int nlat_coarse,nlon_coarse
    nlat_coarse, nlon_coarse = coarse_catchment_nums_in.shape[0],coarse_catchment_nums_in.shape[1]
    latlon_evaluate_basins_cython_wrapper(&minima_in_int[0,0],
                                          &raw_orography_in[0,0],
                                          &corrected_orography_in[0,0],
                                          &connection_volume_thresholds_in[0,0],
                                          &flood_volume_thresholds_in[0,0],
                                          &prior_fine_rdirs_in[0,0],
                                          &prior_fine_catchments_in[0,0],
                                          &coarse_catchment_nums_in[0,0],
                                          &flood_next_cell_lat_index_in[0,0],
                                          &flood_next_cell_lon_index_in[0,0],
                                          &connect_next_cell_lat_index_in[0,0],
                                          &connect_next_cell_lon_index_in[0,0],
                                          &flood_force_merge_lat_index_in[0,0],
                                          &flood_force_merge_lon_index_in[0,0],
                                          &connect_force_merge_lat_index_in[0,0],
                                          &connect_force_merge_lon_index_in[0,0],
                                          &flood_redirect_lat_index_in[0,0],
                                          &flood_redirect_lon_index_in[0,0],
                                          &connect_local_redirect_lat_index_in[0,0],
                                          &connect_local_redirect_lon_index_in[0,0],
                                          &flood_local_redirect_out_int[0,0],
                                          &connect_local_redirect_out_int[0,0],
                                          &merge_points_out_int[0,0],
                                          nlat_fine, nlon_fine,
                                          nlat_coarse, nlon_coarse)
