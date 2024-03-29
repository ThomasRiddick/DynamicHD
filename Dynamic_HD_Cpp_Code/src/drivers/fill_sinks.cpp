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
#include "drivers/fill_sinks.hpp"
#include "algorithms/sink_filling_algorithm.hpp"
#include "base/grid.hpp"

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
								 int* next_cell_lat_index_in,int* next_cell_lon_index_in, double* rdirs_in_double,
								 int* catchment_nums_in,int prefer_non_diagonal_initial_dirs,
								 int* no_data_in_int)
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
	bool* no_data_in = nullptr;
	if (no_data_in_int){
		no_data_in = new bool[nlat*nlon];
		for (auto i = 0; i < nlat*nlon;i++){
			no_data_in[i] = bool(no_data_in_int[i]);
		}
	}
	short* rdirs_in = nullptr;
	if(rdirs_in_double){
		rdirs_in = new short[nlat*nlon];
		fill_n(rdirs_in,nlat*nlon,0);
	}
	latlon_fill_sinks(orography_in, nlat, nlon, method, landsea_in, bool(set_ls_as_no_data_flag),
			   	   	  true_sinks_in,bool(add_slope),epsilon,next_cell_lat_index_in,next_cell_lon_index_in,
					  rdirs_in,catchment_nums_in,bool(prefer_non_diagonal_initial_dirs),false,
					  no_data_in);
	if (rdirs_in) {
		for (auto i = 0; i < nlat*nlon;i++){
			rdirs_in_double[i] = rdirs_in[i];
		}
		delete rdirs_in;
	}
	if (landsea_in) delete[] landsea_in;
	if (true_sinks_in) delete[] true_sinks_in;
	if (no_data_in) delete[] no_data_in;
}

double* latlon_fill_sinks_cython_interface_low_mem(double* orography_in, int nlat, int nlon, int method, int use_ls_mask,
								 															     int* landsea_in_int, int set_ls_as_no_data_flag, int use_true_sinks,
								 																	 int* true_sinks_in_int, int add_slope, double epsilon,
								 																	 int* next_cell_lat_index_in,int* next_cell_lon_index_in,
								 																	 int* catchment_nums_in,int prefer_non_diagonal_initial_dirs)
{
	cout << "Running version of sink filling code with reduced memory usage" << endl;
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
	short* rdirs_out = new short[nlat*nlon];
	fill_n(rdirs_out,nlat*nlon,0);
	latlon_fill_sinks(orography_in, nlat, nlon, method, landsea_in, bool(set_ls_as_no_data_flag),
			   	   	  true_sinks_in,bool(add_slope),epsilon,next_cell_lat_index_in,next_cell_lon_index_in,
					  rdirs_out,catchment_nums_in,bool(prefer_non_diagonal_initial_dirs));

	if (landsea_in) delete[] landsea_in;
	if (true_sinks_in) delete[] true_sinks_in;
	double* rdirs_out_double = new double[nlat*nlon];
	for (auto i = 0; i < nlat*nlon;i++){
		rdirs_out_double[i] = rdirs_out[i];
	}
	delete[] rdirs_out;
	return rdirs_out_double;
}

void latlon_fill_sinks(double* orography_in, int nlat, int nlon, int method,
				bool* landsea_in, bool set_ls_as_no_data_flag,bool* true_sinks_in,
				bool add_slope_in, double epsilon_in, int* next_cell_lat_index_in,
				int* next_cell_lon_index_in, short* rdirs_in,
				int* catchment_nums_in, bool prefer_non_diagonal_initial_dirs,
				bool index_based_rdirs_only_in,bool* no_data_in)
{
	cout << "Entering sink filling C++ code now" << endl;
	const bool tarasov_mod = false;
	latlon_grid_params* grid_params = new latlon_grid_params(nlat,nlon);
	switch(method){
		case 1:
			{
				cout << "Using Algorithm 1" << endl;
				auto alg1 = sink_filling_algorithm_1_latlon();
				alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,add_slope_in,epsilon_in);
				alg1.setup_fields(orography_in,landsea_in,true_sinks_in,grid_params,
				                  no_data_in);
				alg1.fill_sinks();
			}
			break;
		case 4:
			{
				cout << "Using Algorithm 4" << endl;
				auto alg4 = sink_filling_algorithm_4_latlon();
				alg4.setup_flags(set_ls_as_no_data_flag,prefer_non_diagonal_initial_dirs,tarasov_mod,
								 index_based_rdirs_only_in);
				alg4.setup_fields(orography_in,landsea_in,true_sinks_in,next_cell_lat_index_in,
							      next_cell_lon_index_in,grid_params,
								  rdirs_in,catchment_nums_in,no_data_in);
				alg4.fill_sinks();
			}
			break;
		default:
			cout << "Algorithm selected does not exist" << endl;
			break;
	}
	delete grid_params;
}
