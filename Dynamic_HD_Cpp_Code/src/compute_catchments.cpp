/*
 * compute_catchments.cpp
 *
 *  Created on: Feb 26, 2018
 *      Author: thomasriddick
 */

#include "compute_catchments.hpp"
#include "catchment_computation_algorithm.hpp"

void latlon_compute_catchments(int* catchment_numbers_in, double* rdirs_in,
							   int nlat_in,int nlon_in) {
	cout << "Entering C++ Catchment Computation Algorithm" << endl;
	auto alg = catchment_computation_algorithm_latlon();
	auto grid_params_in = new latlon_grid_params(nlat_in,nlon_in);
	alg.setup_fields(catchment_numbers_in,
					 rdirs_in,grid_params_in)
	alg.compute_catchments();
	delete grid_params;
}
