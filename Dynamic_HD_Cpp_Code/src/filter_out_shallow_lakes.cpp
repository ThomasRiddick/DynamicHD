#include "filter_out_shallow_lakes.hpp"
#include "reduce_connected_areas_to_points.hpp"
#include "create_connected_lsmask.hpp"
#include <iostream>
#include <iomanip>
using namespace std;

void latlon_filter_out_shallow_lakes(double* unfilled_orography,double* filled_orography,
                                     double minimum_depth_threshold,int nlat_in,int nlon_in){
  cout << "Entering lake filtering C++ code now" << endl;
  double* orography_difference = new double[nlat_in*nlon_in];
  bool* lakes = new bool[nlat_in*nlon_in];
  fill_n(lakes,nlat_in*nlon_in,false);
  bool* all_lakes = new bool[nlat_in*nlon_in];
  fill_n(all_lakes,nlat_in*nlon_in,false);
  bool* deep_lakes = new bool[nlat_in*nlon_in];
  fill_n(deep_lakes,nlat_in*nlon_in,false);
  for (auto i = 0; i < nlat_in*nlon_in; i++){
    orography_difference[i] = filled_orography[i] - unfilled_orography[i];
    if (orography_difference[i] > 0){
      lakes[i] = true;
      all_lakes[i] = true;
    }
    if (orography_difference[i] > minimum_depth_threshold) deep_lakes[i] = true;
  }
  latlon_reduce_connected_areas_to_points(deep_lakes,nlat_in,nlon_in,true);
  latlon_create_connected_lsmask(lakes,deep_lakes,nlat_in,nlon_in,true);
  for (auto i = 0; i < nlat_in*nlon_in; i++){
    if (all_lakes[i] && ! lakes[i]){
      unfilled_orography[i] = filled_orography[i];
    }
  }
}
