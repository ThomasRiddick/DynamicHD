#include "create_orography.hpp"
#include "orography_creation_algorithm.hpp"
#include <iostream>
#include <string>
#include "grid.hpp"
using namespace std;

void create_orography_cython_wrapper(int* landsea_in_int,double* inclines_in,
                                     double* orography_in,double sea_level_in,
                                     int nlat_in,int nlon_in){
  auto landsea_in = new bool[nlat_in*nlon_in];
  for (auto i = 0; i < nlat_in*nlon_in; i++){
    landsea_in[i] = bool(landsea_in_int[i]);
  }
  create_orography(landsea_in,inclines_in,orography_in,
                   sea_level_in,nlat_in,nlon_in);
}

void create_orography(bool* landsea_in,double* inclines_in,
                      double* orography_in,double sea_level_in,
                      int nlat_in,int nlon_in) {
  cout << "Entering C++ Orography Creation Algorithm" << endl;
  auto alg = orography_creation_algorithm();
  grid_params* grid_params_in = new latlon_grid_params(nlat_in,nlon_in);
  alg.setup_flags(sea_level_in);
  alg.setup_fields(landsea_in,inclines_in,
                   orography_in,grid_params_in);
  alg.create_orography();
  alg.reset();
  double* bathymetry_in = new double[nlat_in*nlon_in];
  bool* landsea_inv_in = new bool[nlat_in*nlon_in];
  for (int i = 0; i < nlat_in*nlon_in;i++){
    landsea_inv_in[i] = ! landsea_in[i];
  }
  alg.setup_flags(-sea_level_in);
  alg.setup_fields(landsea_inv_in,inclines_in,
                   bathymetry_in,grid_params_in);
  alg.create_orography();
  for (int i = 0; i < nlat_in*nlon_in;i++){
    if (landsea_in[i]) {
      orography_in[i] = - bathymetry_in[i];
    }
  }
  delete grid_params_in;
}
