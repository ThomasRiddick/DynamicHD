/*
 * upscale_orography.cpp
 *
 *  Created on: Apr 4, 2017
 *      Author: thomasriddick
 */

#include <iostream>
#include "upscale_orography.hpp"
#include "sink_filling_algorithm.hpp"
#include "grid.hpp"

/* IMPORTANT!
 * Note that a description of each functions purpose is provided with the function declaration in
 * the header file not with the function definition. The definition merely provides discussion of
 * details of the implementation.
 */

void latlon_upscale_orography_cython_interface(double* orography_in, int nlat_fine, int nlon_fine,
							  	  	  	       double* orography_out, int nlat_course, int nlon_course,
											   int method, int* landsea_in_int,int* true_sinks_in_int,
											   int add_slope_in, double epsilon_in,
											   int tarasov_separation_threshold_for_returning_to_same_edge_in,
											   double tarasov_min_path_length_in,
											   int tarasov_include_corners_in_same_edge_criteria_in,
											   int prefer_non_diagonal_initial_dirs){
	bool* landsea_in =  new bool[nlat_fine*nlon_fine];
	for (auto i = 0; i < nlat_fine*nlon_fine;i++){
		landsea_in[i] = bool(landsea_in_int[i]);
	}
	bool* true_sinks_in = new bool[nlat_fine*nlon_fine];
	for (auto i = 0; i < nlat_fine*nlon_fine;i++){
		true_sinks_in[i] = bool(true_sinks_in_int[i]);
	}
	latlon_upscale_orography(orography_in,nlat_fine,nlon_fine,orography_out,nlat_course,nlon_course,
							 method,landsea_in,true_sinks_in,add_slope_in,epsilon_in,
							 tarasov_separation_threshold_for_returning_to_same_edge_in,
							 tarasov_min_path_length_in,tarasov_include_corners_in_same_edge_criteria_in,
							 prefer_non_diagonal_initial_dirs);
	delete [] landsea_in;
	delete [] true_sinks_in;
}

void latlon_upscale_orography(double* orography_in, int nlat_fine, int nlon_fine,
							  double* orography_out, int nlat_course, int nlon_course,
							  int method, bool* landsea_in,bool* true_sinks_in,
							  bool add_slope_in, double epsilon_in,
							  int tarasov_separation_threshold_for_returning_to_same_edge_in,
							  double tarasov_min_path_length_in,
							  bool tarasov_include_corners_in_same_edge_criteria_in,
							  bool prefer_non_diagonal_initial_dirs) {
	cout << "Entering orography upscaling C++ code now" << endl;
	cout << "Parameters:" << endl;
	cout << "Minimum Path Length Threshold: " << to_string(tarasov_min_path_length_in) << endl;
	cout << "Include Corners in Same Edge Criteria: "
			<< to_string(tarasov_include_corners_in_same_edge_criteria_in) << endl;
	cout << "Separation Threshold for Return to Same Edge: "
			<<  to_string(tarasov_separation_threshold_for_returning_to_same_edge_in) << endl;
	const bool tarasov_mod = true;
	const bool set_ls_as_no_data_flag = false;
	const bool index_based_rdirs_only_in = false;
	int scale_factor_lat = nlat_fine/nlat_course;
	int scale_factor_lon = nlon_fine/nlon_course;
	auto grid_params = new latlon_grid_params(scale_factor_lat,scale_factor_lon);
	switch(method){
		case 1:
			{
				cout << "Using Algorithm 1" << endl;
				function<double(double*,bool*,bool*)> run_alg_1 = [&](double* orography_section,
															          bool* landsea_section,
																	  bool* true_sinks_section) {
					bool all_sea_points = true;
					for (auto i = 0; i < scale_factor_lat*scale_factor_lon;i++){
						if (! landsea_section[i]) all_sea_points = false;
					}
					auto alg1 = sink_filling_algorithm_1_latlon();
					if (all_sea_points) return sink_filling_algorithm::get_no_data_value();
					alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,add_slope_in,epsilon_in,
							  	  	 tarasov_separation_threshold_for_returning_to_same_edge_in,
									 tarasov_min_path_length_in,
									 tarasov_include_corners_in_same_edge_criteria_in);
					alg1.setup_fields(orography_section,landsea_section,true_sinks_section,
									  grid_params);
					alg1.fill_sinks();
					return alg1.tarasov_get_area_height();
				};
				partition_fine_orography(orography_in,landsea_in,true_sinks_in,
										 nlat_fine,nlon_fine,orography_out,
										 nlat_course,nlon_course,scale_factor_lat,
										 scale_factor_lon,run_alg_1);
			}
			break;
		case 4:
			{
				cout << "Using Algorithm 4" << endl;
				function<double(double*,bool*,bool*)> run_alg_4 = [&](double* orography_section,
																	  bool* landsea_section,
																	  bool* true_sinks_section) {
					bool all_sea_points = true;
					for (auto i = 0; i < scale_factor_lat*scale_factor_lon;i++){
						if (! landsea_section[i]) all_sea_points = false;
					}
					auto alg4 = sink_filling_algorithm_4_latlon();
					if (all_sea_points) sink_filling_algorithm::get_no_data_value();
					auto next_cell_lat_index_in = new int[scale_factor_lat*scale_factor_lon];
					auto next_cell_lon_index_in = new int[scale_factor_lat*scale_factor_lon];
					auto rdirs_in				= new short[scale_factor_lat*scale_factor_lon];
					auto catchment_nums_in   	= new int[scale_factor_lat*scale_factor_lon];
					alg4.setup_flags(set_ls_as_no_data_flag,prefer_non_diagonal_initial_dirs,tarasov_mod,
									 index_based_rdirs_only_in,
									 tarasov_separation_threshold_for_returning_to_same_edge_in,
									 tarasov_min_path_length_in,
									 tarasov_include_corners_in_same_edge_criteria_in);
					alg4.setup_fields(orography_in,landsea_section,true_sinks_section,next_cell_lat_index_in,
									  next_cell_lon_index_in,grid_params,
									  rdirs_in,catchment_nums_in);
					alg4.fill_sinks();
					delete[] next_cell_lat_index_in;
					delete[] next_cell_lon_index_in;
					delete[] rdirs_in;
					delete[] catchment_nums_in;
					return alg4.tarasov_get_area_height();
				};
				partition_fine_orography(orography_in,landsea_in,true_sinks_in,
										 nlat_fine,nlon_fine,orography_out,
										 nlat_course,nlon_course,scale_factor_lat,
										 scale_factor_lon,run_alg_4);
			}
			break;
		default:
			cout << "Algorithm selected does not exist" << endl;
			break;
	}
	delete grid_params;
}

void partition_fine_orography(double* orography_in, bool* landsea_in, bool* true_sinks_in,int nlat_fine,
							  int nlon_fine, double* orography_out, int nlat_course, int nlon_course,
							  int scale_factor_lat, int scale_factor_lon, function<double(double*,bool*,bool*)> func) {

	double* orography_section = new double[scale_factor_lat*scale_factor_lon];
	bool* landsea_section     = new bool[scale_factor_lat*scale_factor_lon];
	bool* true_sinks_section  = new bool[scale_factor_lat*scale_factor_lon];
	int tenth_of_total_points = 10;
	if (nlat_course >= 10) tenth_of_total_points = nlat_course/10;
	if (nlat_course >= 10) cout << "Each dot represents 10% completion:" << endl;
	for (auto i = 0; i < nlat_course; i++) {
		for (auto j = 0; j < nlon_course; j++) {
			for (auto ifine = 0; ifine < scale_factor_lat; ifine++){
				for (auto jfine = 0; jfine < scale_factor_lon; jfine++) {
					orography_section[ifine*scale_factor_lon + jfine] =
							orography_in[(i*scale_factor_lat + ifine)*nlon_fine + (j*scale_factor_lon + jfine)];
					landsea_section[ifine*scale_factor_lon + jfine] =
							landsea_in[(i*scale_factor_lat + ifine)*nlon_fine + (j*scale_factor_lon + jfine)];
					true_sinks_section[ifine*scale_factor_lon + jfine] =
							true_sinks_in[(i*scale_factor_lat + ifine)*nlon_fine + (j*scale_factor_lon + jfine)];
				}
			}
		orography_out[i*nlon_course + j] = func(orography_section,landsea_section,true_sinks_section);
		}
		if (i%tenth_of_total_points == 0 && ! (i == 0)) cout << ".";
	}
	if (nlat_course >= 10) cout << "." << endl;
	delete[] orography_section;
	delete[] landsea_section;
	delete[] true_sinks_section;
}
