import cython
from Cython.Shadow import bint
cimport numpy as np
import numpy as np
from cython cimport view
from libcpp cimport bool
from libcpp.string cimport string

cdef extern from 'drivers/l2_evaluate_basins.cpp':
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
                                                int* lake_numbers_out_int,
                                                double* sinkless_rdirs_out_double,
                                                int* number_of_lakes_out_ptr,
                                                int* lake_mask_out_int,
                                                int* lakes_as_array_size)

def evaluate_basins(np.ndarray[int,ndim=2,mode='c'] landsea_in_int,
                    np.ndarray[int,ndim=2,mode='c'] minima_in_int,
                    np.ndarray[double,ndim=2,mode='c'] raw_orography_in,
                    np.ndarray[double,ndim=2,mode='c'] corrected_orography_in,
                    np.ndarray[double,ndim=2,mode='c'] cell_areas_in,
                    np.ndarray[int,ndim=2,mode='c'] prior_fine_rdirs_in,
                    np.ndarray[int,ndim=2,mode='c'] prior_fine_catchments_in,
                    np.ndarray[int,ndim=2,mode='c'] coarse_catchment_nums_in,
                    np.ndarray[int,ndim=2,mode='c'] lake_numbers_out,
                    np.ndarray[double,ndim=2,mode='c'] sinkless_rdirs_out_double,
                    np.ndarray[int,ndim=2,mode='c'] lake_mask_out_int):
    cdef int nlat_fine,nlon_fine
    nlat_fine, nlon_fine = raw_orography_in.shape[0],raw_orography_in.shape[1]
    cdef int nlat_coarse,nlon_coarse
    nlat_coarse, nlon_coarse = coarse_catchment_nums_in.shape[0],coarse_catchment_nums_in.shape[1]
    cdef int number_of_lakes_out
    cdef int lakes_as_array_size
    cdef double* lake_as_array_pointer = \
      latlon_evaluate_basins_cython_wrapper(&landsea_in_int[0,0],
                                            &minima_in_int[0,0],
                                            &raw_orography_in[0,0],
                                            &corrected_orography_in[0,0],
                                            &cell_areas_in[0,0],
                                            &prior_fine_rdirs_in[0,0],
                                            &prior_fine_catchments_in[0,0],
                                            &coarse_catchment_nums_in[0,0],
                                            nlat_fine,nlon_fine,
                                            nlat_coarse,nlon_coarse,
                                            &lake_numbers_out[0,0],
                                            &sinkless_rdirs_out_double[0,0],
                                            &number_of_lakes_out,
                                            &lake_mask_out_int[0,0],
                                            &lakes_as_array_size)
    #Work only when you cimport view from cython
    lake_as_array = np.asarray(<double[:lakes_as_array_size]>lake_as_array_pointer)
    return number_of_lakes_out,lake_as_array
