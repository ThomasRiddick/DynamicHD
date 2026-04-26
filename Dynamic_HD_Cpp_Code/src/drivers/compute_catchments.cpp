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

void latlon_relabel_catchments(int* catchment_numbers_in,
                               int* old_to_new_label_map_in,
                               int nlat_in,int nlon_in){

  //Relabel catchments according to size. Take a 1D array filled of new catchment
  //label numbers index by old label number and loop over a field of catchment
  //replacing the old label number with the new ones

  //old_to_new_label_map_in - An array with the new labels sorted so the
  //index is the old label

  //Iterate overall all cells and look up value in old_to_new_label_map
  //(which index by old label number) and replace the label with the
  //value found; which is the new label number
  for (int i = 0; i < nlat_in*nlon_in; i++){
    if (catchment_numbers_in[i] != 0) {
        catchment_numbers_in[i] =
          old_to_new_label_map_in[catchment_numbers_in[i]];
    }
  }
}
