/*
 * test_orography_creation.cpp
 *
 * Unit test for the orography creation C++ code using Google's
 * Google Test Framework
 *
 *  Created on: May 22, 2020
 *      Author: thomasriddick using Google's recommended template code
 */

#include "drivers/create_orography.hpp"
#include "base/field.hpp"
#include "gtest/gtest.h"

using namespace std;

namespace {

class OrographyCreationTest : public ::testing::Test {
 protected:

  OrographyCreationTest() {
  };

  virtual ~OrographyCreationTest() {
  };

  // Common object can go here

};

TEST_F(OrographyCreationTest, OrographyCreationTestSmallGrid){
  int nlat = 20;
  int nlon = 20;
  auto grid_params_in = new latlon_grid_params(nlat,nlon,false);
  bool* landsea_in = new bool[nlat*nlon]{
    true, true,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true, true, true,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false, true,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,

    true, true, true,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true, true, true, true,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true, true, true, true,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true, true,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true, true, true, true,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,

    true, true,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true, true, true, true, true, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true, true, true, true,false, false,false,false,false,false, false,false,false,false,false, false, true,false,false,true,
    true, true,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,

    true, true, true,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true, true, true,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true, true,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,true,true,
    true, true,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true
  };
  double* inclines_in = new double[nlat*nlon] {
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,

    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,

    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,10,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,10,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,10,10,10,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,10,10,10,1, 1,1,1,1,1, 1,1,1,1,1,

    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1 };
  double* orography_out = new double[nlat*nlon];
  double* expected_orography_out = new double[nlat*nlon] {
     -1, -1,  1,  2,  2,  3,  4,  5,  6,  6,  7,  8,  7,  6,  5,  4,  3,  2,  1, -1,
     -1,  1,  1,  1,  2,  3,  4,  5,  5,  6,  7,  8,  7,  6,  5,  4,  3,  2,  1, -1,
     -1, -1, -1,  1,  2,  3,  4,  4,  5,  6,  7,  8,  7,  6,  5,  4,  3,  2,  1, -1,
     -1,  1, -1,  1,  2,  3,  3,  4,  5,  6,  7,  8,  7,  6,  5,  4,  3,  2,  1, -1,
     -1,  1,  1,  1,  2,  2,  3,  4,  5,  6,  7,  7,  7,  6,  5,  4,  3,  2,  1, -1,
     -1, -1, -1,  1,  1,  2,  3,  4,  5,  6,  6,  7,  7,  6,  5,  4,  3,  2,  1, -1,
     -2, -2, -1, -1,  1,  2,  3,  4,  5,  5,  6,  7,  7,  6,  5,  4,  3,  2,  1, -1,
     -2, -1, -1, -1,  1,  2,  3,  4,  4,  5,  6,  6,  6,  6,  5,  4,  3,  2,  1, -1,
     -2, -1,  1,  1,  1,  2,  3,  3,  4,  5,  6,  5,  5,  5,  5,  4,  3,  2,  1, -1,
     -2, -1, -1, -1,  1,  2,  2,  3,  4,  5,  6,  5,  4,  4,  4,  4,  3,  2,  1, -1,
     -2, -1,  1,  1,  1,  1,  2,  3,  4,  5,  6,  5,  4,  3,  3,  3,  3,  2,  1, -1,
     -1, -1, -1, -1, -1,  1, 11,  3,  4,  5,  6,  5,  4,  3,  2,  2,  2,  2,  1, -1,
     -1,  1,  1,  1,  1,  1, 11,  4,  4,  5,  6,  5,  4,  3,  2,  1,  1,  1,  1, -1,
     -1, -1, -1, -1,  1,  2, 11, 14, 14,  5,  6,  5,  4,  3,  2,  1, -1,  1,  1, -1,
     -2, -1,  1,  1,  1,  2, 12, 13, 14,  6,  6,  5,  4,  3,  2,  1,  1,  1,  1, -1,
     -1, -1, -1,  1,  2,  2,  3,  4,  5,  6,  6,  5,  4,  3,  2,  2,  2,  2,  1, -1,
     -1,  1,  1,  1,  2,  3,  3,  4,  5,  6,  6,  5,  4,  3,  3,  3,  2,  2,  1, -1,
     -1, -1, -1,  1,  2,  3,  4,  4,  5,  6,  6,  5,  4,  4,  4,  3,  2,  1,  1, -1,
     -2, -1,  1,  1,  2,  3,  4,  5,  5,  6,  6,  5,  5,  5,  4,  3,  2,  1, -1, -1,
     -2, -1,  1,  2,  2,  3,  4,  5,  6,  6,  6,  6,  6,  5,  4,  3,  2,  1,  1, -1 };
  fill_n(orography_out,nlat*nlon,0);
  create_orography(landsea_in,inclines_in,orography_out,
                   0.0,nlat,nlon);
  EXPECT_TRUE(field<double>(expected_orography_out,grid_params_in)
              == field<double>(orography_out,grid_params_in));
  delete grid_params_in;
  delete[] expected_orography_out;
  delete[] orography_out;
  delete[] inclines_in;
  delete[] landsea_in;
}

}
