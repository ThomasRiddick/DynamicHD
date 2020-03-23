/*
 * follow_streams.cpp
 *
 *  Created on: Feb 10, 2020
 *      Author: thomasriddick
 */

#include "follow_streams.hpp"
#include "stream_following_algorithm.hpp"
#include "grid.hpp"

void latlon_follow_streams_cython_wrapper(double* rdirs_in,int* cells_with_loop_in_int,
                                          int* downstream_cells_in_int,int nlat_in,int nlon_in){
   auto cells_with_loop_in = new bool[nlat_in*nlon_in];
   auto downstream_cells_in = new bool[nlat_in*nlon_in];
   fill_n(downstream_cells_in,nlat_in*nlon_in,false);
   for (auto i = 0; i < nlat_in*nlon_in; i++ ) {
     cells_with_loop_in[i]   = bool(cells_with_loop_in_int[i]);
   }
  latlon_follow_streams(rdirs_in,cells_with_loop_in,downstream_cells_in,nlat_in,nlon_in);
  for (auto i = 0; i < nlat_in*nlon_in; i++ ) {
      downstream_cells_in_int[i]   = int(downstream_cells_in[i]);
   }
}

void latlon_follow_streams(double* rdirs_in,bool* cells_with_loop_in,bool* downstream_cells_in,
                           int nlat_in,int nlon_in) {
  auto grid_params_in = new latlon_grid_params(nlat_in,nlon_in);
  dir_based_rdirs_stream_following_algorithm alg = dir_based_rdirs_stream_following_algorithm();
  alg.setup_fields(rdirs_in,cells_with_loop_in,downstream_cells_in,grid_params_in);
  alg.follow_streams_downstream();
}
