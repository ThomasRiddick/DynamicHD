/*
 * compute_catchments.cpp
 *
 *  Created on: Feb 26, 2018
 *      Author: thomasriddick
 */

#include "drivers/compute_catchments.hpp"
#include "algorithms/catchment_computation_algorithm.hpp"
#include <string>
using namespace std;

void latlon_compute_catchments(int* catchment_numbers_in, double* rdirs_in,
                               string loop_log_filepath,
							                 int nlat_in,int nlon_in) {
	cout << "Entering C++ Catchment Computation Algorithm" << endl;
	auto alg = catchment_computation_algorithm_latlon();
	auto grid_params_in = new latlon_grid_params(nlat_in,nlon_in);
	alg.setup_fields(catchment_numbers_in,
					 rdirs_in,grid_params_in);
	alg.compute_catchments();
  vector<int>* loop_numbers = alg.identify_loops();
  ofstream loop_log_file;
  loop_log_file.open(loop_log_filepath);
  loop_log_file << "Loops found in catchments:" << endl;
  for (auto i = loop_numbers->begin(); i != loop_numbers->end(); ++i)
    loop_log_file << to_string(*i) << endl;
  loop_log_file.close();
  delete loop_numbers;
	delete grid_params_in;
}
