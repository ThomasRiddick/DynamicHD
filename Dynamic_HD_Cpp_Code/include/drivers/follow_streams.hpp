/*
 * follow_streams.hpp
 *
 *  Created on: Feb 10, 2020
 *      Author: thomasriddick
 */

void latlon_follow_streams_cython_wrapper(double* rdirs_in,int* cells_with_loop_in,
                                          int* downstream_cells_in,int nlat_in,int nlon_in,
                                          int include_downstream_outflow_in_int = 0);

void latlon_follow_streams(double* rdirs_in,bool* cells_with_loop_in,bool* downstream_cells_in,
                           int nlat_in,int nlon_in,bool include_downstream_outflow_in = false);
