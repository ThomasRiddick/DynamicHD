/*
 * fill_sinks.h
 *
 * Contains function declaration and global variables for the fill_sinks program
 *
 *  Created on: Mar 15, 2016
 *      Author: thomasriddick
 */

#ifndef FILL_SINKS_HPP_
#define FILL_SINKS_HPP_

//The main fill sinks routine; selects which algorithm to use and passes the input to it
void latlon_fill_sinks(double*, int, int, int,bool* = nullptr, bool = true,bool* = nullptr,
		bool = false, double = 0.1, int* = nullptr, int* = nullptr, short* = nullptr,
		int* = nullptr, bool=false, bool=false, bool* = nullptr);
//Interface function that takes integer as arguments and converts them to bool and also deal with creating a
//null pointer for the land sea mask argument of fill_sinks when the land sea mask is not being used
void latlon_fill_sinks_cython_interface(double*,int,int,int,int = 0,int* = nullptr,int = 1, int = 0, int* = nullptr,int=0,
										double=0.0,int* = nullptr, int* = nullptr, double* = nullptr, int* = nullptr, int=0, int* = nullptr);
//Low memory version of interface function
double* latlon_fill_sinks_cython_interface_low_mem(double* orography_in, int nlat, int nlon, int method, int use_ls_mask,
                                                   int* landsea_in_int, int set_ls_as_no_data_flag, int use_true_sinks,
                                                   int* true_sinks_in_int, int add_slope, double epsilon,
                                                   int* next_cell_lat_index_in=nullptr,
                                                   int* next_cell_lon_index_in=nullptr,
                                                   int* catchment_nums_in=nullptr,
                                                   int prefer_non_diagonal_initial_dirs=0);

#endif /* FILL_SINKS_HPP_ */
