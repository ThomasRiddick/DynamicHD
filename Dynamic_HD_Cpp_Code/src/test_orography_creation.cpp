/*
 * test_orography_creation.cpp
 *
 * Unit test for the orography creation C++ code using Google's
 * Google Test Framework
 *
 *  Created on: May 22, 2020
 *      Author: thomasriddick using Google's recommended template code
 */

#include "create_orography.hpp"
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
  fill_n(orography_out,nlat*nlon,0);
  create_orography(landsea_in,inclines_in,orography_out,
                   0.0,nlat,nlon);
  for (int i = 0; i < nlat; i++){
    for (int j = 0; j < nlon; j++){
      cout << setw(3) << orography_out[nlat*i+j] << " ";
    }
    cout << endl;
  }
  EXPECT_TRUE(false);
}

}
