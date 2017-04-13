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
	cout << "Minimum Path Length Threshold: " << tarasov_min_path_length_in << endl;
	cout << "Include Corners in Same Edge Criteria: "
			<< tarasov_include_corners_in_same_edge_criteria_in << endl;
	cout << "Separation Threshold for Return to Same Edge: "
			<<  tarasov_separation_threshold_for_returning_to_same_edge_in << endl;
	const bool debug = false;
	const bool tarasov_mod = true;
	const bool set_ls_as_no_data_flag = false;
	const bool index_based_rdirs_only_in = false;
	int scale_factor_lat = nlat_fine/nlat_course;
	int scale_factor_lon = nlon_fine/nlon_course;
	switch(method){
		case 1:
			{
				cout << "Using Algorithm 1" << endl;
				function<double(double*,bool*,bool*)> run_alg_1 = [&](double* orography_section,
															          bool* landsea_section,
																	  bool* true_sinks_section) {
					auto alg1 = sink_filling_algorithm_1_latlon();
					alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,debug,add_slope_in,epsilon_in,
							  	  	 tarasov_separation_threshold_for_returning_to_same_edge_in,
									 tarasov_min_path_length_in,
									 tarasov_include_corners_in_same_edge_criteria_in);
					alg1.setup_fields(orography_section,landsea_section,true_sinks_section,
								  	  new latlon_grid_params(scale_factor_lat,scale_factor_lon));
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
					auto next_cell_lat_index_in = new int[scale_factor_lat*scale_factor_lon];
					auto next_cell_lon_index_in = new int[scale_factor_lat*scale_factor_lon];
					auto rdirs_in				= new double[scale_factor_lat*scale_factor_lon];
					auto catchment_nums_in   	= new int[scale_factor_lat*scale_factor_lon];
					auto alg4 = sink_filling_algorithm_4_latlon();
					alg4.setup_flags(set_ls_as_no_data_flag,prefer_non_diagonal_initial_dirs,tarasov_mod,debug,
									 index_based_rdirs_only_in,
									 tarasov_separation_threshold_for_returning_to_same_edge_in,
									 tarasov_min_path_length_in,
									 tarasov_include_corners_in_same_edge_criteria_in);
					alg4.setup_fields(orography_in,landsea_section,true_sinks_section,next_cell_lat_index_in,
									  next_cell_lon_index_in,new latlon_grid_params(scale_factor_lat,scale_factor_lon),
									  rdirs_in,catchment_nums_in);
					alg4.fill_sinks();
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
}

void partition_fine_orography(double* orography_in, bool* landsea_in, bool* true_sinks_in,int nlat_fine,
							  int nlon_fine, double* orography_out, int nlat_course, int nlon_course,
							  int scale_factor_lat, int scale_factor_lon, function<double(double*,bool*,bool*)> func) {

	double* orography_section = new double[scale_factor_lat*scale_factor_lon];
	bool* landsea_section     = new bool[scale_factor_lat*scale_factor_lon];
	bool* true_sinks_section  = new bool[scale_factor_lat*scale_factor_lon];
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
	}
}
