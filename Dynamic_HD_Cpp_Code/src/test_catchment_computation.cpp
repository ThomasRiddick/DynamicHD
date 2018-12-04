/*
 * test_catchment_computation.cpp
 *
 * Unit test for the catchment computation C++ code using Google's
 * Google Test Framework
 *
 *  Created on: Mar 11, 2018
 *      Author: thomasriddick using Google's recommended template code
 */

#include "compute_catchments.hpp"
#include "catchment_computation_algorithm.hpp"
#include "gtest/gtest.h"
#include "cell.hpp"
#include "context.hpp"
#include <string>

using namespace std;

namespace {

class CatchmentComputationTest : public ::testing::Test {
 protected:

  CatchmentComputationTest() {
  };

  virtual ~CatchmentComputationTest() {
  };

  // Common object can go here

};

TEST_F(CatchmentComputationTest, CatchmentComputationSingleCatchmentTest){
  int nlat = 8;
  int nlon = 8;
  int* catchment_numbers = new int[8*8] {0};
  double* rdirs = new double[8*8] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    0.0,7.0,8.0,8.0, 7.0,9.0,7.0,0.0,
    0.0,7.0,3.0,2.0, 1.0,9.0,7.0,0.0,
    0.0,1.0,4.0,3.0, 3.0,3.0,6.0,0.0,
    0.0,7.0,4.0,6.0, 6.0,6.0,9.0,0.0,
    0.0,1.0,1.0,2.0, 1.0,6.0,6.0,0.0,
    0.0,2.0,2.0,3.0, 1.0,2.0,3.0,0.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0
  };
  int* expected_catchments_number_out = new int[8*8] {
    207,0,0,0, 0,0,0,0,
    0,207,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0,
  };
  auto alg = catchment_computation_algorithm_latlon();
  auto grid_params_in = new latlon_grid_params(nlat,nlon);
  landsea_cell* outflow_in = new landsea_cell(new latlon_coords(0,0));
  alg.setup_fields(catchment_numbers,
                   rdirs,grid_params_in);
  alg.test_compute_catchment(outflow_in,207);
  delete grid_params_in;
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(expected_catchments_number_out[i],catchment_numbers[i]);
  }
  delete[] expected_catchments_number_out;
  delete[] catchment_numbers;
  delete[] rdirs;
}

TEST_F(CatchmentComputationTest, CatchmentComputationGeneralTestOne) {
  int nlat = 8;
  int nlon = 8;
  int* catchment_numbers = new int[8*8] {0};
  double* rdirs = new double[8*8] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    0.0,7.0,8.0,8.0, 7.0,9.0,7.0,0.0,
    0.0,7.0,3.0,2.0, 1.0,9.0,7.0,0.0,
    0.0,1.0,4.0,3.0, 3.0,3.0,6.0,0.0,
    0.0,7.0,4.0,6.0, 6.0,6.0,9.0,0.0,
    0.0,1.0,1.0,2.0, 1.0,6.0,6.0,0.0,
    0.0,2.0,2.0,3.0, 1.0,2.0,3.0,0.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0
  };
  int* expected_catchments_number_out = new int[8*8] {
    1, 2, 3, 4,  5, 6, 7, 8,
    9, 1, 3, 4,  4, 7, 6,10,
   11, 9,14,14, 14, 6, 7,12,
   13,15,15,14, 14,14,14,14,
   15,13,13,14, 14,14,14,16,
   17,19,22,25, 25,18,18,18,
   19,22,23,25, 24,26,28,20,
   21,22,23,24, 25,26,27,28
  };
  string log_file_path = datadir + "/temp/loop_log_cpp_null.txt";
  latlon_compute_catchments(catchment_numbers,rdirs,
                            log_file_path,
                            nlat,nlon);
  int count = 0;
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(expected_catchments_number_out[i],catchment_numbers[i]);
    if(expected_catchments_number_out[i] != catchment_numbers[i]) count++;
  }
  delete[] expected_catchments_number_out;
  delete[] catchment_numbers;
  delete[] rdirs;
}

TEST_F(CatchmentComputationTest, CatchmentComputationGeneralWithComplexLoop) {
  int nlat = 8;
  int nlon = 8;
  int* catchment_numbers = new int[8*8] {0};
  double* rdirs = new double[8*8] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    0.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.0,
    0.0,2.0,2.0,2.0, 2.0,2.0,8.0,0.0,
    0.0,6.0,6.0,6.0, 2.0,4.0,4.0,0.0,
    0.0,8.0,6.0,8.0, 4.0,8.0,4.0,0.0,
    0.0,8.0,8.0,8.0, 8.0,8.0,4.0,0.0,
    0.0,8.0,8.0,8.0, 8.0,4.0,4.0,0.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0
  };
  int* expected_catchments_number_out = new int[8*8] {
    1, 2, 3, 4,  5, 6, 7, 8,
    9,29,29,29, 29,29,30,10,
   11,29,29,29, 29,29,30,12,
   13,29,29,29, 29,29,29,14,
   15,29,29,29, 29,29,29,16,
   17,29,29,29, 29,29,29,18,
   19,29,29,29, 29,29,29,20,
   21,22,23,24, 25,26,27,28
  };
  string log_file_path = datadir + "/temp/loop_log_cpp_cloops.txt";
  latlon_compute_catchments(catchment_numbers,rdirs,
                            log_file_path,
                            nlat,nlon);
  int count = 0;
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(expected_catchments_number_out[i],catchment_numbers[i]);
    if(expected_catchments_number_out[i] != catchment_numbers[i]) count++;
  }
  ifstream logfile;
  string output_line_one;
  string output_line_two;
  string output_line_three;
  logfile.open(log_file_path);
  getline(logfile,output_line_one);
  logfile >> output_line_two;
  logfile >> output_line_three;
  cout << output_line_one << endl;
  int loop_catchment_out_one = stoi(output_line_two);
  int loop_catchment_out_two = stoi(output_line_three);
  logfile.close();
  EXPECT_EQ(loop_catchment_out_one,29);
  EXPECT_EQ(loop_catchment_out_two,30);
  delete[] expected_catchments_number_out;
  delete[] catchment_numbers;
  delete[] rdirs;
}

TEST_F(CatchmentComputationTest, CatchmentComputationLoopFindingTestOne) {
  int nlat = 3;
  int nlon = 3;
  int* catchment_numbers = new int[3*3] {0};
  double* rdirs = new double[3*3] {
    6,6,2,
    8,5,2,
    8,4,4
  };
  int* expected_catchments_number_out = new int[3*3] {
    2,2,2,
    2,1,2,
    2,2,2
  };
  string log_file_path = datadir + "/temp/loop_log_cpp_loop_test.txt";
  remove(log_file_path.c_str());
  latlon_compute_catchments(catchment_numbers,rdirs,
                            log_file_path,
                            nlat,nlon);
  int count = 0;
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(expected_catchments_number_out[i],catchment_numbers[i]);
    if(expected_catchments_number_out[i] != catchment_numbers[i]) count++;
  }
  ifstream logfile;
  string output_line_one;
  string output_line_two;
  logfile.open(log_file_path);
  getline(logfile,output_line_one);
  logfile >> output_line_two;
  int loop_catchment_out = stoi(output_line_two);
  logfile.close();
  EXPECT_EQ(loop_catchment_out,2);
  delete[] expected_catchments_number_out;
  delete[] catchment_numbers;
  delete[] rdirs;
}

}  // namespace
