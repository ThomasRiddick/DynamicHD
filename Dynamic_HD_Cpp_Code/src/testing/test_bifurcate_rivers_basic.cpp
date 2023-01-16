/*
 * test_bifurcate_rivers_basic.cpp
 *
 * Unit test for the river bifurcation C++ code using Google's
 * Google Test Framework
 *
 *  Created on: Mar 11, 2018
 *      Author: thomasriddick using Google's recommended template code
 */

#include <climits>
#include "drivers/bifurcate_rivers_basic.hpp"
#include "base/grid.hpp"
#include "base/field.hpp"
#include "gtest/gtest.h"

using namespace std;

namespace {

class BifurcateRiversBasicTest : public ::testing::Test {
 protected:

  BifurcateRiversBasicTest() {
  };

  virtual ~BifurcateRiversBasicTest() {
  };

  // Common object can go here

};

TEST_F(BifurcateRiversBasicTest, BifurcateRiversBasicTestOne){
  int nlat = 8;
  int nlon = 8;
  auto grid_params_in = new latlon_grid_params(nlat,nlon);
  double* original_rdirs = new double[8*8] {
    6.0,6.0,6.0,6.0, 6.0,0.0,0.0,0.0,
    2.0,2.0,2.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,6.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,6.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,6.0,0.0, 0.0,0.0,0.0,0.0,
    3.0,6.0,6.0,3.0, 3.0,0.0,0.0,0.0,
    2.0,3.0,1.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,6.0,6.0, 9.0,0.0,0.0,0.0
  };
  int* cumulative_flow_in = new int[8*8]{
      1,  2,  3,  4,   5,  0,  0,  0,
      1,  1,  1,  1,   0,  0,  0,  0,
      2,  4,  6,  8,   0,  0,  0,  0,
      1,  2,  3,  4,   0,  0,  0,  0,
      1,  2,  3,  4,   0,  0,  0,  0,
     31,  1,  2,  3,   1,  0,  0,  0,
      1, 32,  1,  1,   0,  0,  0,  0,
    103,105,108,110, 111,  0,  0,  0
  };
  bool* landsea_mask_in = new bool[8*8] {
    false,false,false,false,  true, true, true, true,
    false,false,false, true,  true, true, true, true,
    false,false,false, true,  true, true, true, true,
    false,false,false, true,  true, true, true, true,
    false,false,false, true,  true, true, true, true,
    false,false,false,false,  true, true, true, true,
    false,false,false, true,  true, true, true, true,
    false,false,false,false,  true, true, true, true
  };
  int* expected_number_of_outflows_out = new int[8*8]{
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  2,  1,   1,  1,  1,  1
  };
  double* expected_rdirs_out = new double[8*8] {
    6.0,6.0,6.0,3.0, 6.0,0.0,0.0,0.0,
    2.0,2.0,9.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,8.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,8.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,8.0,0.0, 0.0,0.0,0.0,0.0,
    3.0,6.0,6.0,7.0, 3.0,0.0,0.0,0.0,
    2.0,3.0,9.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,6.0,6.0, 9.0,0.0,0.0,0.0
  };
  double* expected_bifurcations_rdirs_out_slice_one = new double[8*8] {
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0, 8.0,-9.0, -9.0,-9.0,-9.0,-9.0
  };
  double* expected_bifurcations_rdirs_out_other_slices = new double[8*8] {
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0
  };
  map<pair<int,int>,vector<pair<int,int>>> river_mouths_in;
  vector<pair<int,int>> additional_river_mouths;
  additional_river_mouths.push_back(pair<int,int>(1,4));
  river_mouths_in[pair<int,int>(6,5)] = additional_river_mouths;
  double* bifurcations_rdirs_in = new double[8*8*7];
  fill_n(bifurcations_rdirs_in,8*8*7,-9.0);
  int* number_of_outflows_in = new int[8*8];
  fill_n(number_of_outflows_in,8*8,1);
  double cumulative_flow_threshold_fraction_in = 0.25;
  int minimum_cells_from_split_to_main_mouth_in = 2;
  int maximum_cells_from_split_to_main_mouth_in = INT_MAX;
  latlon_bifurcate_rivers_basic(river_mouths_in,
                                original_rdirs,
                                bifurcations_rdirs_in,
                                cumulative_flow_in,
                                number_of_outflows_in,
                                landsea_mask_in,
                                cumulative_flow_threshold_fraction_in,
                                minimum_cells_from_split_to_main_mouth_in,
                                maximum_cells_from_split_to_main_mouth_in,
                                nlat,nlon);
  EXPECT_TRUE(field<int>(number_of_outflows_in,grid_params_in) ==
              field<int>(expected_number_of_outflows_out,grid_params_in));
  EXPECT_TRUE(field<double>(original_rdirs,grid_params_in) ==
              field<double>(expected_rdirs_out,grid_params_in));
  for (int i = 0; i < nlat*nlon;i++){
    EXPECT_TRUE(bifurcations_rdirs_in[i] == expected_bifurcations_rdirs_out_slice_one[i]);
  }
  for (int j = 1; j < 7;j++){
    for (int i = 0; i < nlat*nlon;i++){
      EXPECT_TRUE(bifurcations_rdirs_in[i+(nlat*nlon*j)] ==
                  expected_bifurcations_rdirs_out_other_slices[i]);
    }
  }
  delete[] landsea_mask_in;
  delete[] original_rdirs;
  delete[] expected_rdirs_out;
  delete[] expected_number_of_outflows_out;
  delete[] bifurcations_rdirs_in;
  delete[] expected_bifurcations_rdirs_out_slice_one;
  delete[] expected_bifurcations_rdirs_out_other_slices;
  delete grid_params_in;
}

TEST_F(BifurcateRiversBasicTest, BifurcateRiversBasicTestTwo){
  int nlat = 8;
  int nlon = 8;
  auto grid_params_in = new latlon_grid_params(nlat,nlon);
  double* original_rdirs = new double[8*8] {
    6.0,6.0,6.0,6.0, 6.0,0.0,0.0,0.0,
    2.0,2.0,2.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,6.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,6.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,6.0,0.0, 0.0,0.0,0.0,0.0,
    3.0,6.0,6.0,3.0, 3.0,0.0,0.0,0.0,
    2.0,3.0,1.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,6.0,6.0, 9.0,0.0,0.0,0.0
  };
  int* cumulative_flow_in = new int[8*8]{
      1,  2,  3,  4,   5,  0,  0,  0,
      1,  1,  1,  1,   0,  0,  0,  0,
      2,  4,  6,  8,   0,  0,  0,  0,
      1,  2,  3,  4,   0,  0,  0,  0,
      1,  2,  3,  4,   0,  0,  0,  0,
     31,  1,  2,  3,   1,  0,  0,  0,
      1, 32,  1,  1,   0,  0,  0,  0,
    103,105,108,110, 111,  0,  0,  0
  };
  bool* landsea_mask_in = new bool[8*8] {
    false,false,false,false,  true, true, true, true,
    false,false,false, true,  true, true, true, true,
    false,false,false, true,  true, true, true, true,
    false,false,false, true,  true, true, true, true,
    false,false,false, true,  true, true, true, true,
    false,false,false,false,  true, true, true, true,
    false,false,false, true,  true, true, true, true,
    false,false,false,false,  true, true, true, true
  };
  int* expected_number_of_outflows_out = new int[8*8]{
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      2,  1,  1,  1,   1,  1,  1,  1
  };
  double* expected_rdirs_out = new double[8*8] {
    6.0,6.0,6.0,3.0, 6.0,0.0,0.0,0.0,
    2.0,2.0,9.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,8.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,8.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,8.0,0.0, 0.0,0.0,0.0,0.0,
    3.0,9.0,6.0,3.0, 3.0,0.0,0.0,0.0,
    9.0,3.0,1.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,6.0,6.0, 9.0,0.0,0.0,0.0
  };
  double* expected_bifurcations_rdirs_out_slice_one = new double[8*8] {
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
     8.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0
  };
  double* expected_bifurcations_rdirs_out_other_slices = new double[8*8] {
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0
  };
  map<pair<int,int>,vector<pair<int,int>>> river_mouths_in;
  vector<pair<int,int>> additional_river_mouths;
  additional_river_mouths.push_back(pair<int,int>(1,4));
  river_mouths_in[pair<int,int>(6,5)] = additional_river_mouths;
  double* bifurcations_rdirs_in = new double[8*8*7];
  fill_n(bifurcations_rdirs_in,8*8*7,-9.0);
  int* number_of_outflows_in = new int[8*8];
  fill_n(number_of_outflows_in,8*8,1);
  double cumulative_flow_threshold_fraction_in = 0.25;
  int minimum_cells_from_split_to_main_mouth_in = 4;
  int maximum_cells_from_split_to_main_mouth_in = INT_MAX;
  latlon_bifurcate_rivers_basic(river_mouths_in,
                                original_rdirs,
                                bifurcations_rdirs_in,
                                cumulative_flow_in,
                                number_of_outflows_in,
                                landsea_mask_in,
                                cumulative_flow_threshold_fraction_in,
                                minimum_cells_from_split_to_main_mouth_in,
                                maximum_cells_from_split_to_main_mouth_in,
                                nlat,nlon);
  EXPECT_TRUE(field<int>(number_of_outflows_in,grid_params_in) ==
              field<int>(expected_number_of_outflows_out,grid_params_in));
  EXPECT_TRUE(field<double>(original_rdirs,grid_params_in) ==
              field<double>(expected_rdirs_out,grid_params_in));
  for (int i = 0; i < nlat*nlon;i++){
    EXPECT_TRUE(bifurcations_rdirs_in[i] == expected_bifurcations_rdirs_out_slice_one[i]);
  }
  for (int j = 1; j < 7;j++){
    for (int i = 0; i < nlat*nlon;i++){
      EXPECT_TRUE(bifurcations_rdirs_in[i+(nlat*nlon*j)] ==
                  expected_bifurcations_rdirs_out_other_slices[i]);
    }
  }
  delete[] landsea_mask_in;
  delete[] original_rdirs;
  delete[] expected_rdirs_out;
  delete[] expected_number_of_outflows_out;
  delete[] bifurcations_rdirs_in;
  delete[] expected_bifurcations_rdirs_out_slice_one;
  delete[] expected_bifurcations_rdirs_out_other_slices;
  delete grid_params_in;
}

TEST_F(BifurcateRiversBasicTest, BifurcateRiversBasicTestThree){
  int nlat = 8;
  int nlon = 8;
  auto grid_params_in = new latlon_grid_params(nlat,nlon);
  double* original_rdirs = new double[8*8] {
    6.0,6.0,6.0,6.0, 6.0,0.0,0.0,0.0,
    2.0,2.0,2.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,6.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,6.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,6.0,0.0, 0.0,0.0,0.0,0.0,
    3.0,6.0,6.0,3.0, 3.0,0.0,0.0,0.0,
    2.0,3.0,1.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,6.0,6.0, 9.0,0.0,0.0,0.0
  };
  int* cumulative_flow_in = new int[8*8]{
      1,  2,  3,  4,   5,  0,  0,  0,
      1,  1,  1,  1,   0,  0,  0,  0,
      2,  4,  6,  8,   0,  0,  0,  0,
      1,  2,  3,  4,   0,  0,  0,  0,
      1,  2,  3,  4,   0,  0,  0,  0,
     31,  1,  2,  3,   1,  0,  0,  0,
      1, 32,  1,  1,   0,  0,  0,  0,
    103,105,108,110, 111,  0,  0,  0
  };
  bool* landsea_mask_in = new bool[8*8] {
    false,false,false,false,  true, true, true, true,
    false,false,false, true,  true, true, true, true,
    false,false,false, true,  true, true, true, true,
    false,false,false, true,  true, true, true, true,
    false,false,false, true,  true, true, true, true,
    false,false,false,false,  true, true, true, true,
    false,false,false, true,  true, true, true, true,
    false,false,false,false,  true, true, true, true
  };
  int* expected_number_of_outflows_out = new int[8*8]{
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,
      1,  3,  1,  1,   1,  1,  1,  1
  };
  double* expected_rdirs_out = new double[8*8] {
    6.0,6.0,6.0,3.0, 6.0,0.0,0.0,0.0,
    2.0,2.0,9.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,8.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,8.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,8.0,0.0, 0.0,0.0,0.0,0.0,
    3.0,9.0,6.0,8.0, 3.0,0.0,0.0,0.0,
    9.0,3.0,9.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,6.0,6.0,6.0, 9.0,0.0,0.0,0.0
  };
  double* expected_bifurcations_rdirs_out_slice_one = new double[8*8] {
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0, 9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0
  };
  double* expected_bifurcations_rdirs_out_slice_two = new double[8*8] {
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0, 7.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0
  };
  double* expected_bifurcations_rdirs_out_other_slices = new double[8*8] {
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0
  };
  map<pair<int,int>,vector<pair<int,int>>> river_mouths_in;
  vector<pair<int,int>> additional_river_mouths;
  additional_river_mouths.push_back(pair<int,int>(4,3));
  additional_river_mouths.push_back(pair<int,int>(1,4));
  river_mouths_in[pair<int,int>(6,5)] = additional_river_mouths;
  double* bifurcations_rdirs_in = new double[8*8*7];
  fill_n(bifurcations_rdirs_in,8*8*7,-9.0);
  int* number_of_outflows_in = new int[8*8];
  fill_n(number_of_outflows_in,8*8,1);
  double cumulative_flow_threshold_fraction_in = 0.25;
  int minimum_cells_from_split_to_main_mouth_in = 3;
  int maximum_cells_from_split_to_main_mouth_in = INT_MAX;
  latlon_bifurcate_rivers_basic(river_mouths_in,
                                original_rdirs,
                                bifurcations_rdirs_in,
                                cumulative_flow_in,
                                number_of_outflows_in,
                                landsea_mask_in,
                                cumulative_flow_threshold_fraction_in,
                                minimum_cells_from_split_to_main_mouth_in,
                                maximum_cells_from_split_to_main_mouth_in,
                                nlat,nlon);
  EXPECT_TRUE(field<int>(number_of_outflows_in,grid_params_in) ==
              field<int>(expected_number_of_outflows_out,grid_params_in));
  EXPECT_TRUE(field<double>(original_rdirs,grid_params_in) ==
              field<double>(expected_rdirs_out,grid_params_in));
  for (int i = 0; i < nlat*nlon;i++){
    EXPECT_TRUE(bifurcations_rdirs_in[i] == expected_bifurcations_rdirs_out_slice_one[i]);
  }
  for (int i = 0; i < nlat*nlon;i++){
    EXPECT_TRUE(bifurcations_rdirs_in[i+(nlat*nlon*1)] ==
                expected_bifurcations_rdirs_out_slice_two[i]);
  }
  for (int j = 2; j < 7;j++){
    for (int i = 0; i < nlat*nlon;i++){
      EXPECT_TRUE(bifurcations_rdirs_in[i+(nlat*nlon*j)] ==
                  expected_bifurcations_rdirs_out_other_slices[i]);
    }
  }
  delete[] landsea_mask_in;
  delete[] original_rdirs;
  delete[] expected_rdirs_out;
  delete[] expected_number_of_outflows_out;
  delete[] bifurcations_rdirs_in;
  delete[] expected_bifurcations_rdirs_out_slice_one;
  delete[] expected_bifurcations_rdirs_out_slice_two;
  delete[] expected_bifurcations_rdirs_out_other_slices;
  delete grid_params_in;
}

TEST_F(BifurcateRiversBasicTest, BifurcateRiversBasicTestFour){
  int nlat = 9;
  int nlon = 9;
  auto grid_params_in = new latlon_grid_params(nlat,nlon);
  double* original_rdirs = new double[9*9] {
    0.0,3.0,2.0,2.0, 1.0,2.0,2.0,4.0,4.0,
    0.0,6.0,0.0,0.0, 4.0,1.0,2.0,4.0,4.0,
    0.0,4.0,0.0,0.0, 0.0,1.0,1.0,4.0,4.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,4.0,4.0,4.0,
    0.0,0.0,0.0,0.0, 0.0,4.0,7.0,4.0,4.0,
    0.0,7.0,0.0,0.0, 0.0,4.0,8.0,4.0,4.0,
    0.0,9.0,8.0,8.0, 8.0,9.0,8.0,4.0,4.0,
    0.0,9.0,8.0,8.0, 9.0,8.0,8.0,4.0,4.0,
    0.0,4.0,8.0,8.0, 8.0,8.0,8.0,4.0,4.0
  };
  int* cumulative_flow_in = new int[9*9]{
      0,  3,  1,  1,   1,  1,  7,  1,  1,
      0,  3,  0,  0,   4,  3,  8,  1,  1,
      0,  1,  0,  0,   0,  4,  9,  1,  1,
      0,  0,  0,  0,   0,  0,  2,  1,  1,
      0,  0,  0,  0,   0,  2, 501, 1,  1,
      0,  3,  0,  0,   0,  8, 500, 1,  1,
      0,  4,  5,  4,   4,100, 400, 1,  1,
      0,  1,  3,  3,  99,  6, 399, 1,  1,
      1,  1,  1,  1,  98,  1, 398, 1,  1,
  };
  bool* landsea_mask_in = new bool[9*9] {
    true,false,false,false, false,false,false,false, false,
    true,false, true, true, false,false,false,false, false,
    true, false, true, true,  true,false,false,false, false,
    true,  true, true, true,  true, true,false,false, false,
    true,  true, true, true,  true,false,false,false, false,
    true,false, true, true,  true,false,false,false, false,
    true,false,false,false, false,false,false,false, false,
    true,false,false,false, false,false,false,false, false,
    true,false,false,false, false,false,false,false,false
  };
  int* expected_number_of_outflows_out = new int[9*9]{
      1,  1,  1,  1,   1,  1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  1,  1,  1,
      1,  1,  1,  1,   1,  1,  2,  1,  1,
      1,  1,  1,  1,   1,  1,  3,  1,  1
  };
  double* expected_rdirs_out = new double[9*9] {
    0.0,3.0,1.0,4.0, 4.0,2.0,2.0,4.0,4.0,
    0.0,2.0,0.0,0.0, 1.0,7.0,2.0,4.0,4.0,
    0.0,2.0,0.0,0.0, 0.0,7.0,7.0,4.0,4.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,7.0,7.0,4.0,
    0.0,0.0,0.0,0.0, 0.0,4.0,7.0,7.0,7.0,
    0.0,8.0,0.0,0.0, 0.0,4.0,8.0,9.0,7.0,
    0.0,9.0,7.0,8.0, 1.0,9.0,8.0,9.0,7.0,
    0.0,9.0,8.0,7.0, 9.0,7.0,8.0,9.0,4.0,
    0.0,4.0,8.0,8.0, 8.0,8.0,8.0,4.0,4.0
  };
  double* expected_bifurcations_rdirs_out_slice_one = new double[9*9] {
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0, 9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0, 9.0,-9.0,-9.0
  };
  double* expected_bifurcations_rdirs_out_slice_two = new double[9*9] {
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0, 7.0,-9.0,-9.0
  };
  double* expected_bifurcations_rdirs_out_other_slices = new double[9*9] {
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0,
    -9.0,-9.0,-9.0,-9.0, -9.0,-9.0,-9.0,-9.0,-9.0
  };
  map<pair<int,int>,vector<pair<int,int>>> river_mouths_in;
  vector<pair<int,int>> additional_river_mouths;
  additional_river_mouths.push_back(pair<int,int>(2,3));
  additional_river_mouths.push_back(pair<int,int>(3,1));
  additional_river_mouths.push_back(pair<int,int>(4,1));
  river_mouths_in[pair<int,int>(3,5)] = additional_river_mouths;
  double* bifurcations_rdirs_in = new double[9*9*7];
  fill_n(bifurcations_rdirs_in,9*9*7,-9.0);
  int* number_of_outflows_in = new int[9*9];
  fill_n(number_of_outflows_in,9*9,1);
  double cumulative_flow_threshold_fraction_in = 0.1;
  int minimum_cells_from_split_to_main_mouth_in = 3;
  int maximum_cells_from_split_to_main_mouth_in = INT_MAX;
  latlon_bifurcate_rivers_basic(river_mouths_in,
                                original_rdirs,
                                bifurcations_rdirs_in,
                                cumulative_flow_in,
                                number_of_outflows_in,
                                landsea_mask_in,
                                cumulative_flow_threshold_fraction_in,
                                minimum_cells_from_split_to_main_mouth_in,
                                maximum_cells_from_split_to_main_mouth_in,
                                nlat,nlon);
  EXPECT_TRUE(field<int>(number_of_outflows_in,grid_params_in) ==
              field<int>(expected_number_of_outflows_out,grid_params_in));
  EXPECT_TRUE(field<double>(original_rdirs,grid_params_in) ==
              field<double>(expected_rdirs_out,grid_params_in));
  for (int i = 0; i < nlat*nlon;i++){
    EXPECT_TRUE(bifurcations_rdirs_in[i] == expected_bifurcations_rdirs_out_slice_one[i]);
  }
  for (int i = 0; i < nlat*nlon;i++){
    EXPECT_TRUE(bifurcations_rdirs_in[i+(nlat*nlon*1)] ==
                expected_bifurcations_rdirs_out_slice_two[i]);
  }
  for (int j = 2; j < 7;j++){
    for (int i = 0; i < nlat*nlon;i++){
      EXPECT_TRUE(bifurcations_rdirs_in[i+(nlat*nlon*j)] ==
                  expected_bifurcations_rdirs_out_other_slices[i]);
    }
  }
  delete[] landsea_mask_in;
  delete[] original_rdirs;
  delete[] expected_rdirs_out;
  delete[] expected_number_of_outflows_out;
  delete[] bifurcations_rdirs_in;
  delete[] expected_bifurcations_rdirs_out_slice_one;
  delete[] expected_bifurcations_rdirs_out_slice_two;
  delete[] expected_bifurcations_rdirs_out_other_slices;
  delete grid_params_in;
}

TEST_F(BifurcateRiversBasicTest, BifurcateRiversBasicTestFive) {
  int ncells = 80;
  int* cell_neighbors = new int[80*3] {
    //1
    5,7,2,
    //2
    1,10,3,
    //3
    2,13,4,
    //4
    3,16,5,
    //5
    4,19,1,
    //6
    20,21,7,
    //7
    1,6,8,
    //8
    7,23,9,
    //9
    8,25,10,
    //10
    2,9,11,
    //11
    10,27,12,
    //12
    11,29,13,
    //13
    3,12,14,
    //14
    13,31,15,
    //15
    14,33,16,
    //16
    4,15,17,
    //17
    16,35,18,
    //18
    17,37,19,
    //19
    5,18,20,
    //20
    19,39,6,
    //21
    6,40,22,
    //22
    21,41,23,
    //23
    8,22,24,
    //24
    23,43,25,
    //25
    24,26,9,
    //26
    25,45,27,
    //27
    11,26,28,
    //28
    27,47,29,
    //29
    12,28,30,
    //30
    29,49,31,
    //31
    14,30,32,
    //32
    31,51,33,
    //33
    15,32,34,
    //34
    33,53,35,
    //35
    17,34,36,
    //36
    35,55,37,
    //37
    18,36,38,
    //38
    37,57,39,
    //39
    20,38,40,
    //40
    39,59,21,
    //41
    22,60,42,
    //42
    41,61,43,
    //43
    24,42,44,
    //44
    43,63,45,
    //45
    26,44,46,
    //46
    45,64,47,
    //47
    28,46,48,
    //48
    47,66,49,
    //49
    30,48,50,
    //50
    49,67,51,
    //51
    32,50,52,
    //52
    51,69,53,
    //53
    34,52,54,
    //54
    53,70,55,
    //55
    36,54,56,
    //56
    55,72,57,
    //57
    38,56,58,
    //58
    57,73,59,
    //59
    40,58,60,
    //60
    59,75,41,
    //61
    42,75,62,
    //62
    61,76,63,
    //63
    44,62,64,
    //64
    46,63,65,
    //65
    64,77,66,
    //66
    48,65,67,
    //67
    50,66,68,
    //68
    67,78,69,
    //69
    52,68,70,
    //70
    54,69,71,
    //71
    70,79,72,
    //72
    56,71,73,
    //73
    58,72,74,
    //74
    73,80,75,
    //75
    60,74,61,
    //76
    62,80,77,
    //77
    65,76,78,
    //78
    68,77,79,
    //79
    71,78,80,
    //80
    74,79,76
  };
  int* original_next_cell_index_in = new int[80] {
    //1
    -2,
    //2
    -1,
    //3
     2,
    //4
    16,
    //5
    -2,
    //6
    -2,
    //7
    -2,
    //8
    -2,
    //9
    -2,
    //10
    -2,
    //11
    27,
    //12
    29,
    //13
    30,
    //14
    30,
    //15
    33,
    //16
    15,
    //17
    18,
    //18
    19,
    //19
    -1,
    //20
    -2,
    //21
    -2,
    //22
    -2,
    //23
    -2,
    //24
    -2,
    //25
    -2,
    //26
    -2,
    //27
    -1,
    //28
    27,
    //29
    -1,
    //30
    -1,
    //31
    30,
    //32
    30,
    //33
    32,
    //34
    54,
    //35
    37,
    //36
    37,
    //37
    -1,
    //38
    -1,
    //39
    -2,
    //40
    -2,
    //41
    -2,
    //42
    -2,
    //43
    -2,
    //44
    -2,
    //45
    -2,
    //46
    -1,
    //47
    -1,
    //48
    -1,
    //49
    -1,
    //50
    49,
    //51
    49,
    //52
    53,
    //53
    54,
    //54
    -1,
    //55
    38,
    //56
    57,
    //57
    -1,
    //58
    -2,
    //59
    -2,
    //60
    -2,
    //61
    -2,
    //62
    -2,
    //63
    -2,
    //64
    -2,
    //65
    46,
    //66
    48,
    //67
    68,
    //68
    69,
    //69
    52,
    //70
    54,
    //71
    72,
    //72
    -1,
    //73
    -2,
    //74
    -2,
    //75
    -2,
    //76
    -2,
    //77
    78,
    //78
    68,
    //79
    72,
    //80
    -2
  };
  int* cumulative_flow_in= new int[80] {
    //1
    0,
    //2
    0,
    //3
    1,
    //4
    100,
    //5
    0,
    //6
    0,
    //7
    0,
    //8
    0,
    //9
    0,
    //10
    0,
    //11
    1,
    //12
    1,
    //13
    1,
    //14
    1,
    //15
    102,
    //16
    101,
    //17
    1,
    //18
    2,
    //19
    0,
    //20
    0,
    //21
    0,
    //22
    0,
    //23
    0,
    //24
    0,
    //25
    0,
    //26
    0,
    //27
    0,
    //28
    1,
    //29
    0,
    //30
    0,
    //31
    1,
    //32
    104,
    //33
    103,
    //34
    1,
    //35
    1,
    //36
    1,
    //37
    0,
    //38
    0,
    //39
    0,
    //40
    0,
    //41
    0,
    //42
    0,
    //43
    0,
    //44
    0,
    //45
    0,
    //46
    0,
    //47
    0,
    //48
    0,
    //49
    0,
    //50
    1,
    //51
    1,
    //52
    55,
    //53
    56,
    //54
    0,
    //55
    1,
    //56
    1,
    //57
    0,
    //58
    0,
    //59
    0,
    //60
    0,
    //61
    0,
    //62
    0,
    //63
    0,
    //64
    0,
    //65
    1,
    //66
    1,
    //67
    1,
    //68
    53,
    //69
    54,
    //70
    1,
    //71
    1,
    //72
    0,
    //73
    0,
    //74
    0,
    //75
    0,
    //76
    0,
    //77
    50,
    //78
    51,
    //79
    1,
    //80
    0
  };
  bool* landsea_mask_in= new bool[80] {
    //1
    true,
    //2
    true,
    //3
    false,
    //4
    false,
    //5
    true,
    //6
    true,
    //7
    true,
    //8
    true,
    //9
    true,
    //10
    true,
    //11
    false,
    //12
    false,
    //13
    false,
    //14
    false,
    //15
    false,
    //16
    false,
    //17
    false,
    //18
    false,
    //19
    true,
    //20
    true,
    //21
    true,
    //22
    true,
    //23
    true,
    //24
    true,
    //25
    true,
    //26
    true,
    //27
    true,
    //28
    false,
    //29
    true,
    //30
    true,
    //31
    false,
    //32
    false,
    //33
    false,
    //34
    false,
    //35
    false,
    //36
    false,
    //37
    true,
    //38
    true,
    //39
    true,
    //40
    true,
    //41
    true,
    //42
    true,
    //43
    true,
    //44
    true,
    //45
    true,
    //46
    true,
    //47
    true,
    //48
    true,
    //49
    true,
    //50
    false,
    //51
    false,
    //52
    false,
    //53
    false,
    //54
    true,
    //55
    false,
    //56
    false,
    //57
    true,
    //58
    true,
    //59
    true,
    //60
    true,
    //61
    true,
    //62
    true,
    //63
    true,
    //64
    true,
    //65
    false,
    //66
    false,
    //67
    false,
    //68
    false,
    //69
    false,
    //70
    false,
    //71
    false,
    //72
    true,
    //73
    true,
    //74
    true,
    //75
    true,
    //76
    true,
    //77
    false,
    //78
    false,
    //79
    false,
    //80
    true
  };
   int* expected_number_of_outflows_out = new int[ncells]{
    //1
    1,
    //2
    1,
    //3
    1,
    //4
    2,
    //5
    1,
    //6
    1,
    //7
    1,
    //8
    1,
    //9
    1,
    //10
    1,
    //11
    1,
    //12
    1,
    //13
    1,
    //14
    1,
    //15
    1,
    //16
    3,
    //17
    1,
    //18
    1,
    //19
    1,
    //20
    1,
    //21
    1,
    //22
    1,
    //23
    1,
    //24
    1,
    //25
    1,
    //26
    1,
    //27
    1,
    //28
    1,
    //29
    1,
    //30
    1,
    //31
    1,
    //32
    1,
    //33
    1,
    //34
    1,
    //35
    1,
    //36
    1,
    //37
    1,
    //38
    1,
    //39
    1,
    //40
    1,
    //41
    1,
    //42
    1,
    //43
    1,
    //44
    1,
    //45
    1,
    //46
    1,
    //47
    1,
    //48
    1,
    //49
    1,
    //50
    1,
    //51
    1,
    //52
    1,
    //53
    1,
    //54
    1,
    //55
    1,
    //56
    1,
    //57
    1,
    //58
    1,
    //59
    1,
    //60
    1,
    //61
    1,
    //62
    1,
    //63
    1,
    //64
    1,
    //65
    1,
    //66
    1,
    //67
    1,
    //68
    2,
    //69
    1,
    //70
    1,
    //71
    1,
    //72
    1,
    //73
    1,
    //74
    1,
    //75
    1,
    //76
    1,
    //77
    2,
    //78
    2,
    //79
    1,
    //80
    1
  };
  int* expected_next_cell_index_out = new int[ncells] {
     //1
    -2,
    //2
    -1,
    //3
     2,
    //4
    16,
    //5
    -2,
    //6
    -2,
    //7
    -2,
    //8
    -2,
    //9
    -2,
    //10
    -2,
    //11
    27,
    //12
    29,
    //13
    11,
    //14
    29,
    //15
    33,
    //16
    15,
    //17
    18,
    //18
    19,
    //19
    -1,
    //20
    -2,
    //21
    -2,
    //22
    -2,
    //23
    -2,
    //24
    -2,
    //25
    -2,
    //26
    -2,
    //27
    -1,
    //28
    27,
    //29
    -1,
    //30
    -1,
    //31
    30,
    //32
    30,
    //33
    32,
    //34
    54,
    //35
    37,
    //36
    37,
    //37
    -1,
    //38
    -1,
    //39
    -2,
    //40
    -2,
    //41
    -2,
    //42
    -2,
    //43
    -2,
    //44
    -2,
    //45
    -2,
    //46
    -1,
    //47
    -1,
    //48
    -1,
    //49
    -1,
    //50
    49,
    //51
    49,
    //52
    53,
    //53
    54,
    //54
    -1,
    //55
    38,
    //56
    37,
    //57
    -1,
    //58
    -2,
    //59
    -2,
    //60
    -2,
    //61
    -2,
    //62
    -2,
    //63
    -2,
    //64
    -1,
    //65
    46,
    //66
    48,
    //67
    68,
    //68
    69,
    //69
    52,
    //70
    56,
    //71
    72,
    //72
    -1,
    //73
    -2,
    //74
    -2,
    //75
    -2,
    //76
    -2,
    //77
    78,
    //78
    68,
    //79
    72,
    //80
    -2
  };
  int* expected_bifurcations_next_cell_index_slice_one = new int[ncells] {
    //1
    -9,
    //2
    -9,
    //3
    -9,
    //4
     2,
    //5
    -9,
    //6
    -9,
    //7
    -9,
    //8
    -9,
    //9
    -9,
    //10
    -9,
    //11
    -9,
    //12
    -9,
    //13
    -9,
    //14
    -9,
    //15
    -9,
    //16
    14,
    //17
    -9,
    //18
    -9,
    //19
    -9,
    //20
    -9,
    //21
    -9,
    //22
    -9,
    //23
    -9,
    //24
    -9,
    //25
    -9,
    //26
    -9,
    //27
    -9,
    //28
    -9,
    //29
    -9,
    //30
    -9,
    //31
    -9,
    //32
    -9,
    //33
    -9,
    //34
    -9,
    //35
    -9,
    //36
    -9,
    //37
    -9,
    //38
    -9,
    //39
    -9,
    //40
    -9,
    //41
    -9,
    //42
    -9,
    //43
    -9,
    //44
    -9,
    //45
    -9,
    //46
    -9,
    //47
    -9,
    //48
    -9,
    //49
    -9,
    //50
    -9,
    //51
    -9,
    //52
    -9,
    //53
    -9,
    //54
    -9,
    //55
    -9,
    //56
    -9,
    //57
    -9,
    //58
    -9,
    //59
    -9,
    //60
    -9,
    //61
    -9,
    //62
    -9,
    //63
    -9,
    //64
    -9,
    //65
    -9,
    //66
    -9,
    //67
    -9,
    //68
    71,
    //69
    -9,
    //70
    -9,
    //71
    -9,
    //72
    -9,
    //73
    -9,
    //74
    -9,
    //75
    -9,
    //76
    -9,
    //77
    64,
    //78
    70,
    //79
    -9,
    //80
    -9,
  };
  int* expected_bifurcations_next_cell_index_slice_two = new int[ncells] {
     //1
    -9,
    //2
    -9,
    //3
    -9,
    //4
    -9,
    //5
    -9,
    //6
    -9,
    //7
    -9,
    //8
    -9,
    //9
    -9,
    //10
    -9,
    //11
    -9,
    //12
    -9,
    //13
    -9,
    //14
    -9,
    //15
    -9,
    //16
    13,
    //17
    -9,
    //18
    -9,
    //19
    -9,
    //20
    -9,
    //21
    -9,
    //22
    -9,
    //23
    -9,
    //24
    -9,
    //25
    -9,
    //26
    -9,
    //27
    -9,
    //28
    -9,
    //29
    -9,
    //30
    -9,
    //31
    -9,
    //32
    -9,
    //33
    -9,
    //34
    -9,
    //35
    -9,
    //36
    -9,
    //37
    -9,
    //38
    -9,
    //39
    -9,
    //40
    -9,
    //41
    -9,
    //42
    -9,
    //43
    -9,
    //44
    -9,
    //45
    -9,
    //46
    -9,
    //47
    -9,
    //48
    -9,
    //49
    -9,
    //50
    -9,
    //51
    -9,
    //52
    -9,
    //53
    -9,
    //54
    -9,
    //55
    -9,
    //56
    -9,
    //57
    -9,
    //58
    -9,
    //59
    -9,
    //60
    -9,
    //61
    -9,
    //62
    -9,
    //63
    -9,
    //64
    -9,
    //65
    -9,
    //66
    -9,
    //67
    -9,
    //68
    -9,
    //69
    -9,
    //70
    -9,
    //71
    -9,
    //72
    -9,
    //73
    -9,
    //74
    -9,
    //75
    -9,
    //76
    -9,
    //77
    -9,
    //78
    -9,
    //79
    -9,
    //80
    -9
  };
  int* expected_bifurcations_next_cell_index_other_slices = new int[ncells] {
      //1
    -9,
    //2
    -9,
    //3
    -9,
    //4
    -9,
    //5
    -9,
    //6
    -9,
    //7
    -9,
    //8
    -9,
    //9
    -9,
    //10
    -9,
    //11
    -9,
    //12
    -9,
    //13
    -9,
    //14
    -9,
    //15
    -9,
    //16
    -9,
    //17
    -9,
    //18
    -9,
    //19
    -9,
    //20
    -9,
    //21
    -9,
    //22
    -9,
    //23
    -9,
    //24
    -9,
    //25
    -9,
    //26
    -9,
    //27
    -9,
    //28
    -9,
    //29
    -9,
    //30
    -9,
    //31
    -9,
    //32
    -9,
    //33
    -9,
    //34
    -9,
    //35
    -9,
    //36
    -9,
    //37
    -9,
    //38
    -9,
    //39
    -9,
    //40
    -9,
    //41
    -9,
    //42
    -9,
    //43
    -9,
    //44
    -9,
    //45
    -9,
    //46
    -9,
    //47
    -9,
    //48
    -9,
    //49
    -9,
    //50
    -9,
    //51
    -9,
    //52
    -9,
    //53
    -9,
    //54
    -9,
    //55
    -9,
    //56
    -9,
    //57
    -9,
    //58
    -9,
    //59
    -9,
    //60
    -9,
    //61
    -9,
    //62
    -9,
    //63
    -9,
    //64
    -9,
    //65
    -9,
    //66
    -9,
    //67
    -9,
    //68
    -9,
    //69
    -9,
    //70
    -9,
    //71
    -9,
    //72
    -9,
    //73
    -9,
    //74
    -9,
    //75
    -9,
    //76
    -9,
    //77
    -9,
    //78
    -9,
    //79
    -9,
    //80
    -9
  };
  int* secondary_neighboring_cell_indices_in = new int[ncells*9];
  auto grid_params_in = new icon_single_index_grid_params(ncells,
                                                          cell_neighbors,true,
                                                          secondary_neighboring_cell_indices_in);
  map<int,vector<int>> river_mouths_in;
  vector<int> additional_river_mouths;
  vector<int> more_additional_river_mouths;
  additional_river_mouths.push_back(29);
  additional_river_mouths.push_back(27);
  additional_river_mouths.push_back(2);
  more_additional_river_mouths.push_back(37);
  more_additional_river_mouths.push_back(72);
  more_additional_river_mouths.push_back(64);
  river_mouths_in[30] = additional_river_mouths;
  river_mouths_in[54] = more_additional_river_mouths;
  int* bifurcations_next_cell_index_in = new int[80*11];
  fill_n(bifurcations_next_cell_index_in,80*11,-9.0);
  int* number_of_outflows_in = new int[80];
  fill_n(number_of_outflows_in,80,1);
  double cumulative_flow_threshold_fraction_in = 0.1;
  int minimum_cells_from_split_to_main_mouth_in = 3;
  int maximum_cells_from_split_to_main_mouth_in = INT_MAX;
  icon_single_index_bifurcate_rivers_basic(river_mouths_in,
                                           original_next_cell_index_in,
                                           bifurcations_next_cell_index_in,
                                           cumulative_flow_in,
                                           number_of_outflows_in,
                                           landsea_mask_in,
                                           cumulative_flow_threshold_fraction_in,
                                           minimum_cells_from_split_to_main_mouth_in,
                                           maximum_cells_from_split_to_main_mouth_in,
                                           ncells,
                                           cell_neighbors);
  EXPECT_TRUE(field<int>(number_of_outflows_in,grid_params_in) ==
              field<int>(expected_number_of_outflows_out,grid_params_in));
  EXPECT_TRUE(field<int>(original_next_cell_index_in,grid_params_in) ==
              field<int>(expected_next_cell_index_out,grid_params_in));
  for (int i = 0; i < ncells;i++){
    EXPECT_TRUE(bifurcations_next_cell_index_in[i] ==
                expected_bifurcations_next_cell_index_slice_one[i]);
  }
  for (int i = 0; i < ncells;i++){
    EXPECT_TRUE(bifurcations_next_cell_index_in[i+(ncells*1)] ==
                expected_bifurcations_next_cell_index_slice_two[i]);
  }
  for (int j = 2; j < 11;j++){
    for (int i = 0; i < ncells;i++){
      EXPECT_TRUE(bifurcations_next_cell_index_in[i+(ncells*j)] ==
                  expected_bifurcations_next_cell_index_other_slices[i]);
    }
  }
  delete[] cell_neighbors;
  delete[] landsea_mask_in;
  delete[] original_next_cell_index_in;
  delete[] bifurcations_next_cell_index_in;
  delete[] expected_bifurcations_next_cell_index_slice_one;
  delete[] expected_bifurcations_next_cell_index_slice_two;
  delete[] expected_bifurcations_next_cell_index_other_slices;
  delete[]  secondary_neighboring_cell_indices_in;
  delete grid_params_in;
}

}
