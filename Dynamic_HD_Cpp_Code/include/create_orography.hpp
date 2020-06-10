#ifndef INCLUDE_CREATE_OROGRAPHY_HPP_
#define INCLUDE_CREATE_OROGRAPHY_HPP_

void create_orography_cython_wrapper(int* landsea_in_int,double* inclines_in,
                                     double* orography_in,double sea_level_in,
                                     int nlat_in,int nlon_in);

void create_orography(bool* landsea_in,double* inclines_in,
                      double* orography_in,double sea_level_in,
                      int nlat_in,int nlon_in);

#endif /* INCLUDE_CREATE_OROGRAPHY_HPP_ */
