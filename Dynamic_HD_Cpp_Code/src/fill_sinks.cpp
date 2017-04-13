/*
 * fill_sinks.cpp
 *
 * Contain the main functions of the fill sinks program. The algorithms are taken from
 * Barnes R., Lehman C., and Mulla D. (2014) paper 'Priority-flood: An optimal depression
 * -filling and watershed-labeling algorithm for digital elevation models
 *
 *  Created on: Mar 15, 2016
 *      Author: thomasriddick
 */
#include <iostream>
#include "fill_sinks.hpp"
#include "sink_filling_algorithm.hpp"
#include "grid.hpp"

using namespace std;

/* IMPORTANT!
 * Note that a description of each functions purpose is provided with the function declaration in
 * the header file not with the function definition. The definition merely provides discussion of
 * details of the implementation.
 */

//If necessary the function converts the land sea mask to a boolean data type; otherwise it calls
//the function with landsea_in set as a nullptr
void latlon_fill_sinks_cython_interface(double* orography_in, int nlat, int nlon, int method, int use_ls_mask,
								 int* landsea_in_int, int set_ls_as_no_data_flag, int use_true_sinks,
								 int* true_sinks_in_int, int add_slope, double epsilon,
								 int* next_cell_lat_index_in,int* next_cell_lon_index_in, double* rdirs_in,
								 int* catchment_nums_in,int prefer_non_diagonal_initial_dirs)
{
	if(!use_ls_mask) landsea_in_int = nullptr;
	if(!use_true_sinks) true_sinks_in_int = nullptr;
	bool* landsea_in = nullptr;
	//The value of a non null pointer is true; the value of a null pointer is false
	if(landsea_in_int){
		landsea_in =  new bool[nlat*nlon];
		for (auto i = 0; i < nlat*nlon;i++){
			landsea_in[i] = bool(landsea_in_int[i]);
		}
	}
	bool* true_sinks_in = nullptr;
	if(true_sinks_in_int){
		true_sinks_in = new bool[nlat*nlon];
		for (auto i = 0; i < nlat*nlon;i++){
			true_sinks_in[i] = bool(true_sinks_in_int[i]);
		}
	}
	latlon_fill_sinks(orography_in, nlat, nlon, method, landsea_in, bool(set_ls_as_no_data_flag),
			   	   	  true_sinks_in,bool(add_slope),epsilon,next_cell_lat_index_in,next_cell_lon_index_in,
					  rdirs_in,catchment_nums_in,bool(prefer_non_diagonal_initial_dirs));
}

void latlon_fill_sinks(double* orography_in, int nlat, int nlon, int method,
				bool* landsea_in, bool set_ls_as_no_data_flag,bool* true_sinks_in,
				bool add_slope_in, double epsilon_in, int* next_cell_lat_index_in,
				int* next_cell_lon_index_in, double* rdirs_in,
				int* catchment_nums_in, bool prefer_non_diagonal_initial_dirs,
				bool index_based_rdirs_only_in)
{
	cout << "Entering sink filling C++ code now" << endl;
	const bool debug = false;
	const bool tarasov_mod = false;
	switch(method){
		case 1:
			{
				cout << "Using Algorithm 1" << endl;
				auto alg1 = sink_filling_algorithm_1_latlon();
				alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in);
				alg1.setup_fields(orography_in,landsea_in,true_sinks_in,new latlon_grid_params(nlat,nlon));
				alg1.fill_sinks();
			}
			break;
		case 4:
			{
				cout << "Using Algorithm 4" << endl;
				auto alg4 = sink_filling_algorithm_4_latlon();
				alg4.setup_flags(set_ls_as_no_data_flag,prefer_non_diagonal_initial_dirs,tarasov_mod,debug,
								 index_based_rdirs_only_in);
				alg4.setup_fields(orography_in,landsea_in,true_sinks_in,next_cell_lat_index_in,
							      next_cell_lon_index_in,new latlon_grid_params(nlat,nlon),
								  rdirs_in,catchment_nums_in);
				alg4.fill_sinks();
			}
			break;
		default:
			cout << "Algorithm selected does not exist" << endl;
			break;
	}

}
