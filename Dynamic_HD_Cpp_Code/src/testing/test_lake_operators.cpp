/*
 * test_lake_operators.cpp
 *
 * Unit test for the various lake related C++ code using Google's
 * Google Test Framework
 *
 *  Created on: Mar 11, 2018
 *      Author: thomasriddick using Google's recommended template code
 */

#include "drivers/reduce_connected_areas_to_points.hpp"
#include "drivers/burn_carved_rivers.hpp"
#include "drivers/fill_lakes.hpp"
#include "drivers/fill_sinks.hpp"
#include "drivers/redistribute_water.hpp"
#include "drivers/filter_out_shallow_lakes.hpp"
#include "gtest/gtest.h"
using namespace std;

namespace lake_operator_unittests {

class BurnCarvedRiversTest : public ::testing::Test {
 protected:

	BurnCarvedRiversTest() {
  }

  virtual ~BurnCarvedRiversTest() {
    // Don't include exceptions here!
  }

// Common objects can go here

};

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestOne) {
	int nlat = 8;
	int nlon = 8;
	double* orography = new double[8*8] {
		1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,
		1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0,
		1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0,
		1.0,1.5,2.5,1.2, 1.3,2.5,1.5,1.0,
		1.0,1.5,2.5,1.3, 1.3,2.3,1.4,1.0,
		1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0,
		1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0,
		1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0
	};
	double* rdirs = new double[8*8] {
    7.0,8.0,8.0,8.0, 8.0,8.0,8.0,9.0,
    4.0,7.0,8.0,8.0, 7.0,9.0,7.0,6.0,
    4.0,7.0,3.0,2.0, 1.0,9.0,7.0,6.0,
    4.0,1.0,4.0,3.0, 3.0,3.0,6.0,6.0,
    7.0,7.0,4.0,6.0, 6.0,6.0,9.0,6.0,
    4.0,1.0,1.0,2.0, 1.0,6.0,6.0,9.0,
    4.0,2.0,2.0,3.0, 1.0,2.0,3.0,3.0,
    1.0,2.0,2.0,2.0, 3.0,2.0,3.0,3.0
  };
	bool* minima = new bool[8*8] {
		false,false,false,false, false,false,false,false,
		false,false,false,false, false,false,false,false,
		false,false,false,false, false,false,false,false,
		false,false,false,true,  false,false,false,false,
		false,false,false,false, false,false,false,false,
		false,false,false,false, false,false,false,false,
		false,false,false,false, false,false,false,false,
		false,false,false,false, false,false,false,false
	};
  double* expected_orography_out = new double[8*8] {
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,
    1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0,
    1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0,
    1.0,1.5,2.5,1.2, 1.3,2.5,1.5,1.0,
    1.0,1.5,2.5,1.3, 1.2,1.2,1.2,1.0,
    1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0,
    1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0
  };
	bool* lakemask = new bool[8*8] {false};
	latlon_burn_carved_rivers(orography,rdirs,minima,lakemask,nlat,nlon);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestTwo) {
  int nlat = 8;
  int nlon = 8;
  double* orography = new double[8*8] {
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,
    1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0,
    1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0,
    1.0,1.5,2.5,1.2, 1.3,2.5,1.5,1.0,
    1.0,1.5,2.5,1.3, 1.4,2.5,1.4,1.0,
    1.0,1.5,2.5,1.4, 2.5,2.5,1.5,1.0,
    1.0,1.5,1.5,1.4, 1.5,1.5,1.5,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0
  };
  double* rdirs = new double[8*8] {
    7.0,8.0,8.0,8.0, 8.0,8.0,8.0,9.0,
    4.0,7.0,8.0,8.0, 7.0,9.0,7.0,6.0,
    4.0,7.0,3.0,2.0, 1.0,1.0,7.0,6.0,
    4.0,1.0,3.0,2.0, 3.0,4.0,6.0,6.0,
    7.0,7.0,3.0,2.0, 1.0,7.0,9.0,6.0,
    4.0,1.0,3.0,2.0, 1.0,7.0,6.0,9.0,
    4.0,2.0,2.0,1.0, 1.0,2.0,3.0,3.0,
    1.0,2.0,2.0,2.0, 3.0,2.0,3.0,3.0
  };
  bool* minima = new bool[8*8] {
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,true,  false,false,false,false,
    false,false,false,false,  false,false,false,false,
    false,false,false,false,  false,false,false,false,
    false,false,false,false,  false,false,false,false,
    false,false,false,false,  false,false,false,false
  };
  double* expected_orography_out = new double[8*8] {
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,
    1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0,
    1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0,
    1.0,1.5,2.5,1.2, 1.3,2.5,1.5,1.0,
    1.0,1.5,2.5,1.2, 1.4,2.5,1.4,1.0,
    1.0,1.5,2.5,1.2, 2.5,2.5,1.5,1.0,
    1.0,1.5,1.5,1.2, 1.5,1.5,1.5,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0
  };
  bool* lakemask = new bool[8*8] {false};
  latlon_burn_carved_rivers(orography,rdirs,minima,lakemask,nlat,nlon);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestThree) {
  int nlat = 8;
  int nlon = 8;
  double* orography = new double[8*8] {
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,
    1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0,
    1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0,
    0.5,1.5,1.3,1.2, 1.3,2.5,1.5,1.0,
    1.0,1.1,2.5,0.8, 0.7,2.5,1.4,1.0,
    1.0,1.5,2.5,1.8, 2.5,2.5,1.5,1.0,
    1.0,1.5,1.5,1.4, 1.5,1.5,1.5,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0
  };
  double* rdirs = new double[8*8] {
    7.0,8.0,8.0,8.0, 8.0,8.0,8.0,9.0,
    4.0,7.0,8.0,8.0, 7.0,9.0,7.0,6.0,
    4.0,1.0,2.0,1.0, 1.0,1.0,7.0,6.0,
    4.0,4.0,1.0,4.0, 3.0,1.0,6.0,6.0,
    7.0,7.0,4.0,7.0, 4.0,4.0,9.0,6.0,
    4.0,1.0,3.0,2.0, 1.0,7.0,6.0,9.0,
    4.0,2.0,2.0,1.0, 1.0,2.0,3.0,3.0,
    1.0,2.0,2.0,2.0, 3.0,2.0,3.0,3.0
  };
  bool* minima = new bool[8*8] {
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false,  false,false,false,false,
    false,false,false,false,  true,false,false,false,
    false,false,false,false,  false,false,false,false,
    false,false,false,false,  false,false,false,false,
    false,false,false,false,  false,false,false,false
  };
  double* expected_orography_out = new double[8*8] {
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,
    1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0,
    1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0,
    0.5,1.5,0.7,1.2, 1.3,2.5,1.5,1.0,
    1.0,0.7,2.5,0.7, 0.7,2.5,1.4,1.0,
    1.0,1.5,2.5,1.8, 2.5,2.5,1.5,1.0,
    1.0,1.5,1.5,1.4, 1.5,1.5,1.5,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0
  };
  bool* lakemask = new bool[8*8] {false};
  latlon_burn_carved_rivers(orography,rdirs,minima,lakemask,nlat,nlon);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestFour) {
  int nlat = 8;
  int nlon = 8;
  double* orography = new double[8*8] {
    1.0,1.0,1.0,1.0, 0.7,1.0,1.0,1.0,
    1.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.0,
    1.0,1.5,2.5,2.5, 2.5,1.2,1.5,1.0,
    1.0,1.5,1.4,0.7, 2.5,1.3,1.5,1.0,
    1.0,1.5,2.5,0.9, 1.0,2.5,1.4,1.0,
    1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0,
    1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0
  };
  double* rdirs = new double[8*8] {
    7.0,8.0,8.0,8.0, 8.0,8.0,8.0,9.0,
    4.0,7.0,8.0,9.0, 8.0,8.0,7.0,6.0,
    4.0,1.0,2.0,9.0, 8.0,7.0,9.0,6.0,
    4.0,4.0,6.0,3.0, 9.0,8.0,6.0,6.0,
    7.0,7.0,4.0,6.0, 9.0,8.0,9.0,6.0,
    4.0,1.0,3.0,9.0, 8.0,7.0,6.0,9.0,
    4.0,2.0,2.0,1.0, 1.0,2.0,3.0,3.0,
    1.0,2.0,2.0,2.0, 3.0,2.0,3.0,3.0
  };
  bool* minima = new bool[8*8] {
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false, true, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[8*8] {
    1.0,1.0,1.0,1.0, 0.7,1.0,1.0,1.0,
    1.0,1.5,1.5,1.5, 0.7,1.5,1.5,1.0,
    1.0,1.5,2.5,2.5, 2.5,0.7,1.5,1.0,
    1.0,1.5,1.4,0.7, 2.5,0.7,1.5,1.0,
    1.0,1.5,2.5,0.9, 0.7,2.5,1.4,1.0,
    1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0,
    1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0
  };
  bool* lakemask = new bool[8*8] {false};
  latlon_burn_carved_rivers(orography,rdirs,minima,lakemask,nlat,nlon);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestFive) {
  int nlat = 8;
  int nlon = 8;
  double* orography = new double[8*8] {
    1.0,1.0,1.0,1.0, 0.7,1.0,1.0,1.0,
    1.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.0,
    1.0,1.5,2.5,2.5, 2.5,1.2,1.5,1.0,
    1.0,1.5,1.4,0.7, 2.5,1.3,1.5,1.0,
    1.0,1.5,2.5,0.9, 1.0,2.5,1.4,1.0,
    1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0,
    1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0
  };
  double* rdirs = new double[8*8] {
    7.0,8.0,8.0,8.0, 8.0,8.0,8.0,9.0,
    4.0,7.0,8.0,9.0, 8.0,8.0,7.0,6.0,
    4.0,1.0,2.0,9.0, 8.0,7.0,9.0,6.0,
    4.0,4.0,6.0,3.0, 9.0,8.0,6.0,6.0,
    7.0,7.0,4.0,6.0, 9.0,8.0,9.0,6.0,
    4.0,1.0,3.0,9.0, 8.0,7.0,6.0,9.0,
    4.0,2.0,2.0,1.0, 1.0,2.0,3.0,3.0,
    1.0,2.0,2.0,2.0, 3.0,2.0,3.0,3.0
  };
  bool* minima = new bool[8*8] {
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false, true, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[8*8] {
    1.0,1.0,1.0,1.0, 0.7,1.0,1.0,1.0,
    1.0,1.5,1.5,1.5, 0.7,1.5,1.5,1.0,
    1.0,1.5,2.5,2.5, 2.5,1.2,1.5,1.0,
    1.0,1.5,1.4,0.7, 2.5,0.7,1.5,1.0,
    1.0,1.5,2.5,0.9, 1.0,2.5,1.4,1.0,
    1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0,
    1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0
  };
  bool* lakemask = new bool[8*8] {
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false, true,false,false,
    false,false,false, true, false,false,false,false,
    false,false,false, true, true, false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false
  };
  latlon_burn_carved_rivers(orography,rdirs,minima,lakemask,nlat,nlon);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestSix) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,0.4,0.8, 2.0,2.0,1.0,2.0, 1.0,1.0,0.3,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.8,1.0,2.0,0.8, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,0.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,0.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
//
    0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.2,0.2,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,0.1,1.0, 1.0,0.5,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,1.0,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,1.0,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,1.0,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,1.5,1.5, 1.5,1.5,1.2,1.5, 1.5,1.1,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,

  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false, true,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, true, false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false, true,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false, true,false, false,false,false,false, false,false, true,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false, true,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,0.4,0.4, 2.0,2.0,0.2,2.0, 0.2,0.2,0.2,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.4,0.2,2.0,0.8, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,0.2,2.0,2.0, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,0.2,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,0.2,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
//
    0.0,1.5,2.0,2.0, 2.0,0.2,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,0.2,0.2,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,0.1,0.5, 0.5,0.5,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,0.2,2.0, 2.0,0.1,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,0.2,2.0, 2.0,0.1,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,0.2,2.0, 2.0,0.1,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,1.5,1.5, 1.5,1.5,0.2,1.5, 1.5,0.1,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false, true, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false, true,false,false, false,false,false,false, false,false,false,false,
    false,false, true,false, false, true,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,true,true,true,    true,true,true,true,     true,true,true,true,     true,true,true,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,true,true,true, true,true,true,true, true,true,true,true, true,true,true,true
  };

  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] landsea;
  delete[] truesinks;
  delete[] minima;
  delete[] lakemask;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestSeven) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,0.4,0.8, 2.0,2.0,1.0,2.0, 1.0,1.0,0.3,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.8,1.0,2.0,0.8, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,0.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,0.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
//
    0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.2,0.2,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,0.1,1.0, 1.0,0.5,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,1.0,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,1.0,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,1.0,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,1.5,1.5, 1.5,1.5,1.2,1.5, 1.5,1.1,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,

  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false, true,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, true, false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false, true,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false, true,false, false,false,false,false, false,false, true,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false, true,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,0.4,0.4, 2.0,2.0,1.0,2.0, 1.0,1.0,0.2,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.4,1.0,2.0,0.8, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,0.4,2.0,2.0, 2.0,2.0,2.0,0.2, 0.2,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,0.4,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,0.4,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
//
    0.0,1.5,2.0,2.0, 2.0,0.4,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,0.2,0.2,2.0, 2.0,1.0,2.0,2.0, 2.0,2.0,0.1,0.5, 0.5,0.5,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,0.4,2.0, 2.0,0.1,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,0.4,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,0.4,2.0, 2.0,0.8,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,1.5,1.5, 1.5,1.5,1.2,1.5, 1.5,1.1,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false, true, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false, true,false,false, false,false,false,false, false,false,false,false,
    false,false, true,false, false, true,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,true,true,true,    true,true,true,true,     true,true,true,true,     true,true,true,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true,true,true,true,    true,true,true,true,     true,true,true,true,     true,true,true,true
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  rdirs[nlat*13+9] = 5;
  rdirs[nlat*13+6] = 0;
  rdirs[nlat*3+10] = -1;
  rdirs[nlat*10+11] = 5;
  rdirs[nlat*11+9] = -1;
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestEight) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,4.0,5.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,4.0,4.0, 3.0,3.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5, 3.0,3.0,3.0,3.0, 4.0,4.0,4.0,4.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false, true,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,2.65,2.6, 2.55,2.5,2.45,2.4, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.7,6.0,6.0, 6.0,6.0,6.0,6.0, 2.35,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.75,6.0,6.0, 6.0,6.0,6.0,6.0, 2.3,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.8,6.0,6.0, 6.0,6.0,6.0,6.0, 2.25,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,2.85,2.90, 2.95,3.0,6.0,6.0, 2.2,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.15,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.1,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.05,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.95,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5, 1.55,1.6,1.65,1.7, 1.75,1.8,1.85,1.9, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,4,0.1);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestNine) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,4.0,5.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,4.0,4.0, 3.0,3.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5, 2.5,2.5,2.5,2.5, 4.0,4.0,4.0,4.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false, true,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,2.65,2.6, 2.55,2.5,2.45,2.4, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.7,6.0,6.0, 6.0,6.0,6.0,6.0, 2.35,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.75,6.0,6.0, 6.0,6.0,6.0,6.0, 2.3,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.8,6.0,6.0, 6.0,6.0,6.0,6.0, 2.25,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,2.85,2.90, 2.95,3.0,6.0,6.0, 2.2,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.15,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.1,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.05,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.95,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5, 1.55,1.6,1.65,1.7, 1.75,1.8,1.85,1.9, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,4,0.6);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestTen) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,4.0,5.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5, 2.5,2.5,2.5,2.5, 4.0,4.0,4.0,4.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false, true,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,2.94,2.92, 2.9,2.88,2.86,2.84, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.96,6.0,6.0, 6.0,6.0,6.0,6.0, 2.82,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.98,6.0,6.0, 6.0,6.0,6.0,6.0, 2.8,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.78,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 2.76,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.74,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.72,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.7,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.68,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.66,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5, 2.5,2.52,2.54,2.56, 2.58,2.6,2.62,2.64, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,3,0.6);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestEleven) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,5.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,0.5,0.5,0.5, 0.5,0.5,1.0,2.5, 2.5,2.5,2.5,4.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false, true,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,2.9, 2.8,2.7,2.6,2.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.4,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.3,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.2,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 2.1,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 2.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 1.9,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 1.8,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.7,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.6,6.0,6.0,9.0,
    0.0,0.5,0.5,0.5, 0.5,0.5,1.0,1.1, 1.2,1.3,1.4,1.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,9,0.6);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestTwelve) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,0.5,0.5,0.5, 0.5,0.5,1.0,2.5, 2.5,2.5,2.5,4.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false, true, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 2.875,2.75,2.625,2.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.375,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.25,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.125,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 2.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 1.875,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 1.75,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 1.625,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.5,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.375,6.0,6.0,9.0,
    0.0,0.5,0.5,0.5, 0.5,0.5,0.625,0.75, 0.875,1.0,1.125,1.25, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,9,2.1);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestThirteen) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,0.2,0.2,0.2, 0.2,0.2,0.2,0.5, 4.0,4.0,4.0,4.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false, true,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,2.875,2.75, 2.625,2.5,2.375,2.25, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.125,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.875,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 1.75,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 1.625,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 1.5,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 1.375,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.25,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.125,6.0,6.0,9.0,
    0.0,0.2,0.2,0.2, 0.2,0.2,0.2,0.5, 0.625,0.75,0.875, 1.0,5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,5,2.1);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestFourteen) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,0.2,0.2,0.2, 0.2,0.2,0.2,0.5, 4.0,4.0,4.0,4.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false, true,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,2.875,2.75, 2.625,2.5,2.375,2.25, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.125,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.875,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 1.75,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 1.625,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 1.5,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 1.375,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.25,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.125,6.0,6.0,9.0,
    0.0,0.2,0.2,0.2, 0.2,0.2,0.2,0.5, 0.625,0.75,0.875, 1.0,5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,0,5.1);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}


TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestFifteen) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,0.2,0.2,0.2, 0.2,0.2,0.2,0.5, 4.0,4.0,4.0,4.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false, true,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,2.888,2.776, 2.664,2.552,2.44,2.328, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.216,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.104,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.992,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 1.88,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 1.768,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 1.656,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,6.0, 1.544,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.432,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.32,6.0,6.0,9.0,
    0.0,0.2,0.2,0.312, 0.424,0.536,0.648,0.76, 0.872,0.984,1.096, 1.208,5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,5,5.1);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_FLOAT_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestSixteen) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,4.0,5.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,4.0,4.0, 3.0,3.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5, 3.0,3.0,3.0,3.0, 4.0,4.0,4.0,4.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false, true,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,2.65,2.6, 5.0,2.5,2.45,2.4, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.7,6.0,6.0, 6.0,6.0,6.0,6.0, 2.35,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.75,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.8,6.0,6.0, 6.0,6.0,6.0,6.0, 2.25,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,2.85,2.90, 2.95,3.0,6.0,6.0, 2.2,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.15,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.05,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.95,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5, 1.55,1.6,1.65,1.7, 1.75,1.8,1.85,1.9, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,  true,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false,  true,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false,  true,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,4,0.1);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestSeventeen) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,4.0,5.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5, 2.0,2.0,2.0,2.0, 2.0,4.0,4.0,4.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false, true,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,2.72,2.68, 2.64,2.6,2.56,2.52, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.76,6.0,6.0, 6.0,6.0,6.0,6.0, 2.48,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.8,6.0,6.0, 6.0,6.0,6.0,6.0, 2.44,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.84,6.0,6.0, 6.0,6.0,6.0,6.0, 2.4,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,2.88,2.92, 2.96,3.0,6.0,6.0, 2.36,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.32,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.28,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.24,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.2,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.16,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5,   2,2,2,2, 2,2.04,2.08,2.12, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false, true,false, true, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,8,4.1);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestEighteen) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 5.5,5.5,5.0,5.0, 5.0,4.0,4.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,6.0,4.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,6.0,4.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,6.0,4.0,6.0, 6.0,6.0,6.0,0.0,
//
    9.0,6.0,6.0,5.0, 6.0,6.0,3.0,3.0, 4.0,4.0,4.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 4.0,4.0,4.0,4.0, 2.5,2.5,2.5,2.5, 1.5,1.5,1.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false, true,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 2.4,2.45,2.5,2.55, 2.6,2.65,4.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,2.35, 6.0,6.0,6.0,6.0, 6.0,6.0,2.7,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,2.3, 6.0,6.0,6.0,6.0, 6.0,6.0,2.75,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,2.25, 6.0,6.0,6.0,6.0, 6.0,6.0,2.8,6.0, 6.0,6.0,6.0,0.0,
//
    9.0,6.0,6.0,2.2, 6.0,6.0,3.0,2.95, 2.90,2.85,4.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,2.15, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,2.1, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,2.05, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,2.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,1.95, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 1.9,1.85,1.8,1.75, 1.7,1.65,1.6,1.55, 1.5,1.5,1.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,4,0.6);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestNineteen) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,5.0,5.0,5.0, 5.0,5.0,5.0,5.0, 5.0,5.0,5.0,5.0, 6.0,6.0,6.0,6.0,
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,5.5, 6.0,6.0,6.0,6.0,
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,5.5, 6.0,6.0,6.0,6.0,
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,3.0, 6.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0,
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,3.0, 6.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0,
//
    6.0,2.5,6.0,6.0, 6.0,6.0,6.0,4.0, 6.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0,
    6.0,2.5,6.0,6.0, 6.0,6.0,6.0,4.0, 6.0,6.0,6.0,4.0, 6.0,6.0,6.0,6.0,
    6.0,2.5,6.0,6.0, 6.0,6.0,6.0,4.0, 4.0,4.0,4.0,4.0, 6.0,6.0,6.0,6.0,
    6.0,2.5,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,1.5,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,1.5,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,1.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false, true, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,5.0,1.95,2.0, 2.05,2.1,2.15,2.2, 2.25,2.3,2.35,5.0, 6.0,6.0,6.0,6.0,
    6.0,1.9,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,2.4, 6.0,6.0,6.0,6.0,
    6.0,1.85,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,2.45, 6.0,6.0,6.0,6.0,
    6.0,1.8,6.0,6.0, 6.0,6.0,6.0,3.0, 6.0,6.0,6.0,2.5, 6.0,6.0,6.0,6.0,
    6.0,1.75,6.0,6.0, 6.0,6.0,6.0,2.95, 6.0,6.0,6.0,2.55, 6.0,6.0,6.0,6.0,
//
    6.0,1.7,6.0,6.0, 6.0,6.0,6.0,2.9, 6.0,6.0,6.0, 2.6, 6.0,6.0,6.0,6.0,
    6.0,1.65,6.0,6.0, 6.0,6.0,6.0,2.85, 6.0,6.0,6.0,2.65, 6.0,6.0,6.0,6.0,
    6.0,1.6,6.0,6.0, 6.0,6.0,6.0,4.0, 2.8,2.75,2.7,4.0, 6.0,6.0,6.0,6.0,
    6.0,1.55,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,1.5,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,1.5,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,1.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
     true, true, true, true,  true, true, true, true,  true, true, true, true,  true, true, true, true,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,4,0.6);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestTwenty) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,1.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,1.5,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,1.5,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,2.5,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,2.5,6.0,6.0, 6.0,6.0,6.0,4.0, 4.0,4.0,4.0,4.0, 6.0,6.0,6.0,6.0,
    6.0,2.5,6.0,6.0, 6.0,6.0,6.0,4.0, 6.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0,
    6.0,2.5,6.0,6.0, 6.0,6.0,6.0,4.0, 6.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0,
//
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,3.0, 6.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0,
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,3.0, 6.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0,
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,5.5, 6.0,6.0,6.0,6.0,
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,5.5, 6.0,6.0,6.0,6.0,
    6.0,5.0,5.0,5.0, 5.0,5.0,5.0,5.0, 5.0,5.0,5.0,5.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false, true, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    6.0,1.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,1.5,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,1.5,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,1.55,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,1.6,6.0,6.0, 6.0,6.0,6.0,4.0, 2.8,2.75,2.7,4.0, 6.0,6.0,6.0,6.0,
    6.0,1.65,6.0,6.0, 6.0,6.0,6.0,2.85, 6.0,6.0,6.0,2.65, 6.0,6.0,6.0,6.0,
    6.0,1.7,6.0,6.0, 6.0,6.0,6.0,2.9, 6.0,6.0,6.0,2.6, 6.0,6.0,6.0,6.0,
//
    6.0,1.75,6.0,6.0, 6.0,6.0,6.0,2.95, 6.0,6.0,6.0,2.55, 6.0,6.0,6.0,6.0,
    6.0,1.8,6.0,6.0, 6.0,6.0,6.0,3.0, 6.0,6.0,6.0,2.5, 6.0,6.0,6.0,6.0,
    6.0,1.85,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,2.45, 6.0,6.0,6.0,6.0,
    6.0,1.9,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,2.4, 6.0,6.0,6.0,6.0,
    6.0,5.0,1.95,2.0, 2.05,2.1,2.15,2.2, 2.25,2.3,2.35,5.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
     true, true, true, true,  true, true, true, true,  true, true, true, true,  true, true, true, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,4,0.6);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestTwentyOne) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,4.0,5.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
//
    0.0,1.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,1.5,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,1.5, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 2.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,2.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,2.0, 2.0,4.0,4.0,4.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false, true,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,2.72,2.68, 2.64,2.6,2.56,2.52, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.76,6.0,6.0, 6.0,6.0,6.0,6.0, 2.48,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.8,6.0,6.0, 6.0,6.0,6.0,6.0, 2.44,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.84,6.0,6.0, 6.0,6.0,6.0,6.0, 2.4,6.0,6.0,9.0,
//
    0.0,1.0,6.0,6.0, 6.0,4.0,2.88,2.92, 2.96,3.0,6.0,6.0, 2.36,6.0,6.0,9.0,
    0.0,6.0,1.5,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.32,6.0,6.0,9.0,
    0.0,6.0,6.0,1.5, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.28,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 2.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.24,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.2,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,2.0,6.0, 6.0,6.0,6.0,6.0, 2.16,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,2.0, 2.0,2.04,2.08,2.12, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false, true,false, true, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,8,4.1);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestTwentyTwo) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 5.5,5.5,5.0,5.0, 5.0,4.0,4.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,6.0,4.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,6.0,4.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,6.0,4.0,6.0, 6.0,6.0,6.0,0.0,
//
    9.0,6.0,6.0,5.0, 6.0,6.0,3.0,3.0, 4.0,4.0,4.0,6.0, 6.0,6.0,1.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,1.5,6.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.5,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,2.5, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,6.0,2.5,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0, 6.0,2.5,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 4.0,4.0,4.0,4.0, 2.5,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false, true,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 2.4,2.45,2.5,2.55, 2.6,2.65,4.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,2.35, 6.0,6.0,6.0,6.0, 6.0,6.0,2.7,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,2.3, 6.0,6.0,6.0,6.0, 6.0,6.0,2.75,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,2.25, 6.0,6.0,6.0,6.0, 6.0,6.0,2.8,6.0, 6.0,6.0,6.0,0.0,
//
    9.0,6.0,6.0,2.2, 6.0,6.0,3.0,2.95, 2.90,2.85,4.0,6.0, 6.0,6.0,1.0,0.0,
    9.0,6.0,6.0,2.15, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,1.5,6.0,0.0,
    9.0,6.0,6.0,2.1, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.5,6.0,6.0,0.0,
    9.0,6.0,6.0,2.05, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,1.55, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,2.0, 6.0,6.0,6.0,6.0, 6.0,6.0,1.6,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,1.95, 6.0,6.0,6.0,6.0, 6.0,1.65,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,5.0, 1.9,1.85,1.8,1.75, 1.7,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,4,0.6);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestTwentyThree) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,5.0,5.0,5.0, 5.0,5.0,5.0,5.0, 5.0,5.0,5.0,5.0, 6.0,6.0,6.0,6.0,
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,5.5, 6.0,6.0,6.0,6.0,
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,5.5, 6.0,6.0,6.0,6.0,
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,3.0, 6.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0,
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,3.0, 6.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0,
//
    6.0,6.0,2.5,6.0, 6.0,6.0,6.0,4.0, 6.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,2.5, 6.0,6.0,6.0,4.0, 6.0,6.0,6.0,4.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 2.5,6.0,6.0,4.0, 4.0,4.0,4.0,4.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,2.5,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,1.5,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,1.5, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false, true, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,5.0,1.95,2.0, 2.05,2.1,2.15,2.2, 2.25,2.3,2.35,5.0, 6.0,6.0,6.0,6.0,
    6.0,1.9,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,2.4, 6.0,6.0,6.0,6.0,
    6.0,1.85,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,2.45, 6.0,6.0,6.0,6.0,
    6.0,1.8,6.0,6.0, 6.0,6.0,6.0,3.0, 6.0,6.0,6.0,2.5, 6.0,6.0,6.0,6.0,
    6.0,1.75,6.0,6.0, 6.0,6.0,6.0,2.95, 6.0,6.0,6.0,2.55, 6.0,6.0,6.0,6.0,
//
    6.0,6.0,1.7,6.0, 6.0,6.0,6.0,2.9, 6.0,6.0,6.0, 2.6, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,1.65, 6.0,6.0,6.0,2.85, 6.0,6.0,6.0,2.65, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 1.6,6.0,6.0,4.0, 2.8,2.75,2.7,4.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,1.55,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,1.5,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,1.5, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 1.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
     true, true, true, true,  true, true, true, true,  true, true, true, true,  true, true, true, true,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,4,0.6);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestTwentyFour) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,5.0,5.0,5.0, 5.0,5.0,5.0,5.0, 5.0,5.0,5.0,5.0, 6.0,6.0,6.0,6.0,
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,5.5, 6.0,6.0,6.0,6.0,
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,5.5, 6.0,6.0,6.0,6.0,
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,3.0, 6.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0,
    6.0,4.0,6.0,6.0, 6.0,6.0,6.0,3.0, 6.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0,
//
    6.0,6.0,2.5,6.0, 6.0,6.0,6.0,4.0, 6.0,6.0,6.0,5.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,2.5, 6.0,6.0,6.0,4.0, 6.0,6.0,6.0,4.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,2.5,6.0, 6.0,6.0,6.0,4.0, 4.0,4.0,4.0,4.0, 6.0,6.0,6.0,6.0,
    6.0,2.5,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    1.5,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,1.5,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,1.0,6.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false, true, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,5.0,1.95,2.0, 2.05,2.1,2.15,2.2, 2.25,2.3,2.35,5.0, 6.0,6.0,6.0,6.0,
    6.0,1.9,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,2.4, 6.0,6.0,6.0,6.0,
    6.0,1.85,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,2.45, 6.0,6.0,6.0,6.0,
    6.0,1.8,6.0,6.0, 6.0,6.0,6.0,3.0, 6.0,6.0,6.0,2.5, 6.0,6.0,6.0,6.0,
    6.0,1.75,6.0,6.0, 6.0,6.0,6.0,2.95, 6.0,6.0,6.0,2.55, 6.0,6.0,6.0,6.0,
//
    6.0,6.0,1.7,6.0, 6.0,6.0,6.0,2.9, 6.0,6.0,6.0, 2.6, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,1.65, 6.0,6.0,6.0,2.85, 6.0,6.0,6.0,2.65, 6.0,6.0,6.0,6.0,
    6.0,6.0,1.6,6.0, 6.0,6.0,6.0,4.0, 2.8,2.75,2.7,4.0, 6.0,6.0,6.0,6.0,
    6.0,1.55,6.0,6.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    1.5,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,1.5,
    6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,1.0,6.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0
  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
     true, true, true, true,  true, true, true, true,  true, true, true, true,  true, true, true, true,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,4,0.6);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestTwentyFive) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,4.0,5.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5, 2.0,2.0,2.0,2.0, 2.0,4.0,4.0,4.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false, true,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,2.72,2.68, 5.0,2.6,2.56,2.52, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.76,6.0,6.0, 6.0,6.0,6.0,6.0, 2.48,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.8,6.0,6.0, 6.0,6.0,6.0,6.0, 2.44,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.84,6.0,6.0, 6.0,6.0,6.0,6.0, 2.4,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,2.88,2.92, 2.96,3.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.32,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.28,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.24,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.2,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.16,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5,   2,2,2,2, 2,2.04,2.08,2.12, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,  true,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, true,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false, true,false, true, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,8,4.1);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestTwentySix) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,4.0,5.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 4.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 4.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5, 2.0,2.0,2.0,2.0, 2.0,4.0,4.0,3.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false, true,false,false,  true,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false, true, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,2.72,2.68, 2.64,2.6,2.56,2.52, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.76,6.0,6.0, 6.0,6.0,6.0,6.0, 2.48,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.8,6.0,6.0, 6.0,6.0,6.0,6.0, 2.44,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.84,6.0,6.0, 6.0,6.0,6.0,6.0, 2.4,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,2.88,2.92, 2.96,3.0,6.0,6.0, 2.36,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.32,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.28,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.24,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.2,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.16,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5,   2,2,2,2, 2,2.04,2.08,2.12, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false, true,false, true, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,8,4.1);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestTwentySeven) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,3.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 5.0,5.0,5.5,5.5, 5.0,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,9.0,
    0.0,1.0,1.5,2.0, 2.0,2.0,2.5,2.5, 2.5,2.5,2.5,5.0, 5.0,3.0,3.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false, true,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,3.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.1,2.2,2.3,2.4, 2.5,2.6,2.7,2.8, 2.9,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.1,2.2,2.3,2.4, 2.5,2.6,2.7,2.8, 2.9,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,2.9,9.0,
    0.0,1.0,1.5,2.0, 2.0,2.0,2.1,2.2, 2.3,2.4,2.5,2.6, 2.7,2.8,3.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,8,0.6);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestTwentyEight) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,3.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 5.0,5.0,5.5,5.5, 5.0,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.25,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.25,9.0,
    0.0,1.0,1.0,2.0, 2.0,2.0,2.6,2.7, 2.5,2.5,2.5,5.0, 5.0,3.25,3.25,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false, true,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, true,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,3.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.1,2.2,2.3,2.4, 2.5,2.6,2.7,2.8, 2.9,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.1,2.2,2.3,2.4, 2.5,2.6,2.7,2.8, 2.9,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.25,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.125,9.0,
    0.0,1.0,1.0,1.25, 1.5,1.75,2.0,2.25, 2.375,2.5,2.625, 2.75,2.875,3,3.25,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,10,0.9);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestTwentyNine) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,3.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 5.0,5.0,5.5,5.5, 5.0,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,
    9.0,3.25,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,3.25,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,3.25,3.25,5.0, 5.0,2.5,2.5,2.5, 2.7,2.6,2.0,2.0, 2.0,1.0,1.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false, true,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false, true, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,3.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.1,2.2,2.3,2.4, 2.5,2.6,2.7,2.8, 2.9,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.1,2.2,2.3,2.4, 2.5,2.6,2.7,2.8, 2.9,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,
    9.0,3.25,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,3.125,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,3.25,3,2.875, 2.75,2.625,2.5,2.375, 2.25,2.0,1.75,1.5, 1.25,1.0,1.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,

     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
//
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,10,0.9);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestThirty) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,3.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 5.0,5.0,5.5,5.5, 5.0,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,
    9.0,3.25,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,3.25,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,3.25,3.25,5.0, 5.0,2.75,2.5,2.5, 2.7,2.6,2.0,2.0, 2.0,1.0,1.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false, true,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false, true, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,3.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.1,2.2,2.3,2.4, 2.5,2.6,2.7,2.8, 2.9,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.1,2.2,2.3,2.4, 2.5,2.6,2.7,2.8, 2.9,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,
    9.0,3.25,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,3.125,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0,
    9.0,3.25,3,2.875, 2.75,2.625,2.5,2.375, 2.25,2.0,1.75,1.5, 1.25,1.0,1.0,0.0,
    9.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,0.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,

     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
//
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,10,0.9);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestThirtyOne) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,3.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 5.0,5.0,5.5,5.5, 5.0,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.25,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.25,9.0,
    0.0,1.0,1.0,2.0, 2.0,2.0,2.6,2.7, 2.5,2.5,2.75,5.0, 5.0,3.25,3.25,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false, true,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, true,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,3.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.1,2.2,2.3,2.4, 2.5,2.6,2.7,2.8, 2.9,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.1,2.2,2.3,2.4, 2.5,2.6,2.7,2.8, 2.9,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.25,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,3.125,9.0,
    0.0,1.0,1.0,1.25, 1.5,1.75,2.0,2.25, 2.375,2.5,2.625, 2.75,2.875,3,3.25,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,10,0.9);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestThirtyTwo) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,4.0,5.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5, 3.0,2.75,2.75,3.0, 4.0,4.0,4.0,4.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false, true, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,2.95,2.94, 2.93,2.92,2.91,2.90, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.96,6.0,6.0, 6.0,6.0,6.0,6.0, 2.89,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.97,6.0,6.0, 6.0,6.0,6.0,6.0, 2.88,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.98,6.0,6.0, 6.0,6.0,6.0,6.0, 2.87,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,2.99,3.0, 3.0,3.0,6.0,6.0, 2.86,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.85,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.84,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.83,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.82,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.81,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5, 3.0,2.75,2.75,2.76, 2.77,2.78,2.79,2.80, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false, true,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,4,0.6);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestThirtyThree) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,4.0,5.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,3.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,3.0,3.0, 3.0,3.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5, 3.0,2.75,2.75,3.0, 4.0,4.0,4.0,4.0, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false, true, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,4.0,2.95,2.94, 2.93,2.92,2.91,2.90, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.96,6.0,6.0, 6.0,6.0,6.0,6.0, 2.89,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.97,6.0,6.0, 6.0,6.0,6.0,6.0, 2.88,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,2.98,6.0,6.0, 6.0,6.0,6.0,6.0, 2.87,6.0,6.0,9.0,
//
    0.0,6.0,6.0,6.0, 6.0,4.0,2.99,3.0, 3.0,3.0,6.0,6.0, 2.86,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.85,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.84,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.83,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.82,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 2.81,6.0,6.0,9.0,
    0.0,1.0,1.5,1.5, 3.0,2.75,2.75,2.76, 2.77,2.78,2.79,2.80, 5.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,true,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,4,0.6);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestThirtyFour) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,6.0,
    6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,6.0,
    6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,6.0,
    6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,6.0,
    4.0,5.0, 5.0,5.0,5.5,5.5, 5.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,4.0,
    6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,4.0,
    6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,4.0,
    6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,4.0,
//
    4.0,4.0, 3.0,3.0,6.0,6.0, 5.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,4.0,
    6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,6.0,
    6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,6.0,
    6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,6.0,
    6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,6.0,
    6.0,6.0, 6.0,6.0,6.0,6.0, 5.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,6.0,
    3.0,3.0, 4.0,4.0,4.0,4.0, 5.0,6.0,6.0,9.0, 0.0,1.0,1.5,1.5, 3.0,3.0,
    6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,6.0,
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false, true, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
     6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,    0.0,6.0,6.0,6.0, 6.0,6.0,
     6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,    0.0,6.0,6.0,6.0, 6.0,6.0,
     6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,    0.0,6.0,6.0,6.0, 6.0,6.0,
     6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,    0.0,6.0,6.0,6.0, 6.0,6.0,
     2.65,2.6, 2.55,2.5,2.45,2.4, 5.0,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,4.0,
     6.0,6.0, 6.0,6.0,6.0,6.0, 2.35,6.0,6.0,9.0,   0.0,6.0,6.0,6.0, 6.0,2.7,
    6.0,6.0, 6.0,6.0,6.0,6.0, 2.3,6.0,6.0,9.0,   0.0,6.0,6.0,6.0, 6.0,2.75,
     6.0,6.0, 6.0,6.0,6.0,6.0, 2.25,6.0,6.0,9.0,   0.0,6.0,6.0,6.0, 6.0,2.8,
//
     2.85,2.90, 2.95,3.0,6.0,6.0, 2.2,6.0,6.0,9.0, 0.0,6.0,6.0,6.0, 6.0,4.0,
     6.0,6.0, 6.0,6.0,6.0,6.0, 2.15,6.0,6.0,9.0,   0.0,6.0,6.0,6.0, 6.0,6.0,
     6.0,6.0, 6.0,6.0,6.0,6.0, 2.1,6.0,6.0,9.0,    0.0,6.0,6.0,6.0, 6.0,6.0,
     6.0,6.0, 6.0,6.0,6.0,6.0, 2.05,6.0,6.0,9.0,   0.0,6.0,6.0,6.0, 6.0,6.0,
     6.0,6.0, 6.0,6.0,6.0,6.0, 2.0,6.0,6.0,9.0,    0.0,6.0,6.0,6.0, 6.0,6.0,
     6.0,6.0, 6.0,6.0,6.0,6.0, 1.95,6.0,6.0,9.0,   0.0,6.0,6.0,6.0, 6.0,6.0,
    1.65,1.7, 1.75,1.8,1.85,1.9, 5.0,6.0,6.0,9.0,  0.0,1.0,1.5,1.5, 1.55,1.6,
     6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,     0.0,6.0,6.0,6.0, 6.0,6.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,

    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,

    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,4,0.1);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestThirtyFive) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 2.5,5.0,5.0,5.0, 5.0,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 2.5,2.5,2.5,2.5, 5.0,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,true, false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 2.5,2.6,2.7,2.8, 2.9,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 2.5,2.5,2.5,2.5, 2.75,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,8,0.6,4,0.1);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestThirtySix) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 2.5,5.0,5.0,5.0, 5.0,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 2.5,2.5,2.5,2.5, 5.0,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,true, false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.1,2.2,2.3,2.4, 2.5,2.6,2.7,2.8, 2.9,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 2.5,2.5,2.5,2.5, 2.75,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,8,0.6,3,0.1);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestThirtySeven) {
  int nlat = 16;
  int nlon = 16;
  double* orography = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 2.5,5.0,5.0,5.0, 5.0,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.5,2.5,2.5,2.5, 2.5,5.0,5.0,5.0, 5.0,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0
  };

  short* rdirs = new short[16*16] {0};

  bool* minima = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  //
    false,false,false,false, false,false,false,false, false,false,false,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,true, false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* expected_orography_out = new double[16*16] {
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
//
    0.0,2.0,2.0,2.0, 2.1,2.2,2.3,2.4, 2.5,2.6,2.7,2.8, 2.9,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,
    0.0,2.0,2.0,2.0, 2.1,2.2,2.3,2.4, 2.5,2.6,2.7,2.8, 2.9,3.0,6.0,9.0,
    0.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0

  };
  bool* lakemask = new bool[16*16] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* truesinks = new bool[16*16] {false};
  bool* landsea = new bool[16*16] {
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  int* empty = new int[16*16] {0};
  latlon_fill_sinks(orography,nlat,nlon,4,landsea,false,truesinks,false,0.0,empty,empty,rdirs,
                    empty,false,false);
  double* rdirs_double = new double[16*16];
  for (auto i=0; i < nlat*nlon; i++){
    rdirs_double[i] = rdirs[i];
  }
  latlon_burn_carved_rivers(orography,rdirs_double,minima,lakemask,nlat,nlon,true,8,0.6,3,0.1);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(expected_orography_out[i],orography[i]);
  }
  delete[] minima;
  delete[] lakemask;
  delete[] landsea;
  delete[] truesinks;
  delete[] empty;
  delete[] expected_orography_out;
  delete[] orography;
  delete[] rdirs;
  delete[] rdirs_double;
}

class ConnectedAreaReductionTest : public ::testing::Test {
 protected:

  ConnectedAreaReductionTest() {
  }

  virtual ~ConnectedAreaReductionTest() {
    // Don't include exceptions here!
  }

// Common objects can go here

};

TEST_F(ConnectedAreaReductionTest, ConnectedAreaReductionTestOne){
  int nlat = 16;
  int nlon = 16;
  bool* areas = new bool[nlat*nlon] {
    true, false, true,false, false,false,false,false, false,true, true, false, false,true, false,false,
    true,  true,false,false, false,false,false,false, false,false,false,false, true, false, true,false,
    false,false, true,false, false,false,false,false, false,false,false,false, false, true,false,false,
    false, true,false, true, false,false,false,false, false,false,false,false, false,false, true,false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false, true, false,false,false,false, false,false,false,false,
    true, false,false,false, true,  true, true,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false, true, true,false, false,false,false,false, false,false,false, true,
//
    false, true,false,false, false,false,false,false,  true,false,false,true, false,false,false,false,
     true,false,true, false, false,false,false,false, false, true,false,true, false,false,false,false,
     true,false,false, true, false,false,false,false, false, true,false,true, false,false,true ,false,
    false, true,false, true, false,false,false,false, false,false,true,false, false,false,false,false,

    false,false,false,false, false,true, false,false, false,false,false,false, true, false,false,true,
    true, false,false,false, false,false, true,false, false,false,false,false, true, false,false,true,
    false,false,false,false, false,false,false, true, false,false,false,false, true, false,false,true,
    true,false,false,false, false, true, true,false, false,true, true ,false, true, true, true, false,
  };
  bool* expected_areas_out = new bool[nlat*nlon] {
    true, false,false,false, false,false,false,false, false,true, false,false, false,true, false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false, true, false,false,false,false, false,false,false,false,
    true, false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false, true,false,false, false,false,false,false,  true,false,false,false,  false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,true ,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,

    false,false,false,false, false,true, false,false, false,false,false,false, true, false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,true, false,false, false,false,false,false,
  };
  latlon_reduce_connected_areas_to_points(areas,nlat,nlon,true);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(areas[i],expected_areas_out[i]);
  }
  delete[] areas;
  delete[] expected_areas_out;
}

TEST_F(ConnectedAreaReductionTest, ConnectedAreaReductionTestTwo){
  int nlat = 16;
  int nlon = 16;
  bool* areas = new bool[nlat*nlon] {
    false,false,false,false, false,false,false,false, false,false,true, false, false,false,false,false,
    false,true, true, true,  false, true,false,false, false,true, false,true,  false,false,false,false,
    false,true, false,false, false,false,true, false, false,true, false,true,  false,false,false,false,
    false,true, false,false, false,false,true, false, false,false,true, false, false,false,false,false,
//
    false,true, false,false, false,false,true, false, false,false,false,false, false,true, true, true,
    false,true, false,false, false,false,true, false, false,false,false,false, false,true, false,true,
    false,true, true, true,  true, true, false,false, false,false,false,false, false,true, false,true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,true, true, true,
//
    false,false,false,false, false,false,false,false, false,true, true, true,  false,false,false,false,
    false,false,false,false, false,false,false,false, true, false,false,true,  false,false,false,false,
    false,false,false,false, false,false,false,false, true, false,false,true,  false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,true,  false,false,false,false,
//
    true, true, true, false, true, true, true, true,  false,false,false,false, false,false,true, false,
    true, false,false,false, false,false,false,true,  false,false,false,false, true, false,true, false,
    true, false,false,false, false,false,false,true,  false,false,false,false, true, false,true, false,
    false,true, true, false, true, true, true, true,  false,false,false,false, false,true, false,false
  };
  bool* expected_areas_out = new bool[nlat*nlon] {
    false,false,false,false, false,false,false,false, false,false,true, false, false,false,false,false,
    false,true, false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,true, false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,true, false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    true, false,false,false, true, false,false,false, false,false,false,false, false,false,true, false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  latlon_reduce_connected_areas_to_points(areas,nlat,nlon,true);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(areas[i],expected_areas_out[i]);
  }
  delete[] areas;
  delete[] expected_areas_out;
}

TEST_F(ConnectedAreaReductionTest, ConnectedAreaReductionTestThree){
  int nlat = 16;
  int nlon = 16;
  bool* areas = new bool[nlat*nlon] {
    false,false,false,false, false,true, true, false, false,false,false,false, false,false,false,false,
    false,false,false,false, true, false,false,true,  false,false,false,false, false,false,false,false,
    false,false,false,false, true, false,false,true,  false,false,false,false, false,false,false,false,
    false,false,false,false, false,true, true, false, false,false,false,false, false,false,false,false,
//
    true, false,false,false, false,false,false,false, true, true, true, true,  false,false,false,false,
    false,false,false,false, false,false,false,false, true, false,false,true,  false,false,false,true,
    true, false,false,false, false,false,false,false, true, false,false,true,  false,false,false,false,
    false,false,false,false, false,false,false,false, true, true, true, true,  false,false,false,false,
//
    false,false,false,false, false,true, false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, true, true, true, false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,true, false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, true, false,true, false, false,false,false,false,
    true, false,false,false, false,false,false,false, false,true, false,false, false,false,false,true,
    false,false,false,false, false,false,false,false, true, false,true, false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* expected_areas_out = new bool[nlat*nlon] {
    false,false,false,false, false,true, false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, true, false,false,true,  false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,true, false,false, false,false,false,false, false,false,false,false,
//
    true, false,false,false, false,false,false,false, true, false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    true, false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,true, false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, true, false,true, false, false,false,false,false,
    true, false,false,false, false,false,false,false, false,true, false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, true, false,true, false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
  };
  latlon_reduce_connected_areas_to_points(areas,nlat,nlon,false);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(areas[i],expected_areas_out[i]);
  }
  delete[] areas;
  delete[] expected_areas_out;
}

TEST_F(ConnectedAreaReductionTest, ConnectedAreaReductionTestFour){
  int nlat = 16;
  int nlon = 16;
  bool* areas = new bool[nlat*nlon] {
    false,false,false,false, false,true, true, false, false,false,false,false, false,false,false,false,
    false,false,false,false, true, false,false,true,  false,false,false,false, false, true, true,false,
    false,false,false,false, true, false,false,true,  false,false,false,false, false, true, true,false,
    false,false,false,false, false,true, true, false, false,false,false,false, false,false,false,false,
//
    true, false,false,false, false,false,false,false, true, true, true, true,  false,false,false,false,
    false,false,false,false, false,false,false,false, true, false,false,true,  false,false,false,true,
    true, false,false,false, false,false,false,false, true, false,false,true,  false,false,false,false,
    false,false,false,false, false,false,false,false, true, true, true, true,  false,false,false,false,
//
    false,false,false,false, false,true, false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, true, true, true, false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,true, false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, true, false,true, false, false,false,false,false,
    true, false,false,false, false,false,false,false, false,true, false,false, false,false,false,true,
    false,false,false,false, false,false,false,false, true, false,true, false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* expected_areas_out = new bool[nlat*nlon] {
    false,false,false,false, false, true,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* orography = new double[nlat*nlon] {
    1.0,1.0,1.0,1.0, 1.0,0.1,0.1,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,
    1.0,1.0,1.0,1.0, 0.1,1.0,1.0,0.1, 1.0,1.0,1.0,1.0, 1.0,0.5,0.5,1.0,
    1.0,1.0,1.0,1.0, 0.1,1.0,1.0,0.1, 1.0,1.0,1.0,1.0, 1.0,0.5,0.5,1.0,
    1.0,1.0,1.0,1.0, 1.0,0.1,0.1,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,

    0.1,0.1,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,0.2,0.2,0.2, 1.0,1.0,1.0,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,1.0,1.0,0.2, 1.0,1.0,1.0,0.1,
    0.1,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,1.0,1.0,0.2, 1.0,1.0,1.0,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,0.2,0.2,0.2, 1.0,1.0,1.0,1.0,

    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,0.1,1.0,1.0, 1.0,1.0,1.0,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,

    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.1,1.0,0.1,1.0, 1.0,1.0,1.0,1.0,
    0.5,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,0.1,1.0,1.0, 1.0,1.0,1.0,0.5,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.1,1.0,0.1,1.0, 1.0,1.0,1.0,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,0.1,1.0, 1.0,1.0,1.0,1.0
  };
  latlon_reduce_connected_areas_to_points(areas,nlat,nlon,true,orography,true);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(areas[i],expected_areas_out[i]);
  }
  delete[] areas;
  delete[] expected_areas_out;
  delete[] orography;
}

TEST_F(ConnectedAreaReductionTest, ConnectedAreaReductionTestFive){
  int nlat = 16;
  int nlon = 16;
  bool* areas = new bool[nlat*nlon] {
    false,false,false,false, false,true, true, false, false,false,false,false, false,false,false,false,
    false,false,false,false, true, false,false,true,  false,false,false,false, false, true, true,false,
    false,false,false,false, true, false,false,true,  false,false,false,false, false, true, true,false,
    false,false,false,false, false,true, true, false, false,false,false,false, false,false,false,false,
//
    true, false,false,false, false,false,false,false, true, true, true, true,  false,false,false,false,
    false,false,false,false, false,false,false,false, true, false,false,true,  false,false,false,true,
    true, false,false,false, false,false,false,false, true, false,false,true,  false,false,false,false,
    false,false,false,false, false,false,false,false, true, true, true, true,  false,false,false,false,
//
    false,false,false,false, false,true, false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, true, true, true, false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,true, false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, true, false,true, false, false,false,false,false,
    true, false,false,false, false,false,false,false, false,true, false,false, false,false,false,true,
    false,false,false,false, false,false,false,false, true, false,true, false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* expected_areas_out = new bool[nlat*nlon] {
    false,false,false,false, false, true,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false,  true,false,false, true, false,false,false,false, false, true,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false, true,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
     true,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false,  true,false, true,false, false,false,false,false,
     true,false,false,false, false,false,false,false, false, true,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,  true,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* orography = new double[nlat*nlon] {
    1.0,1.0,1.0,1.0, 1.0,0.1,0.1,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,
    1.0,1.0,1.0,1.0, 0.1,1.0,1.0,0.1, 1.0,1.0,1.0,1.0, 1.0,0.5,0.5,1.0,
    1.0,1.0,1.0,1.0, 0.1,1.0,1.0,0.1, 1.0,1.0,1.0,1.0, 1.0,0.5,0.5,1.0,
    1.0,1.0,1.0,1.0, 1.0,0.1,0.1,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,

    0.1,0.1,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,0.2,0.2,0.2, 1.0,1.0,1.0,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,1.0,1.0,0.2, 1.0,1.0,1.0,0.1,
    0.1,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,1.0,1.0,0.2, 1.0,1.0,1.0,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.2,0.2,0.2,0.2, 1.0,1.0,1.0,1.0,

    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,0.1,1.0,1.0, 1.0,1.0,1.0,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,

    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.1,1.0,0.1,1.0, 1.0,1.0,1.0,1.0,
    0.5,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,0.1,1.0,1.0, 1.0,1.0,1.0,0.5,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 0.1,1.0,0.1,1.0, 1.0,1.0,1.0,1.0,
    1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, 1.0,1.0,0.1,1.0, 1.0,1.0,1.0,1.0
  };
  latlon_reduce_connected_areas_to_points(areas,nlat,nlon,false,orography,true);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(areas[i],expected_areas_out[i]);
  }
  delete[] areas;
  delete[] expected_areas_out;
  delete[] orography;
}

class LakeFillingTest : public ::testing::Test {
 protected:

  LakeFillingTest() {
  }

  virtual ~LakeFillingTest() {
    // Don't include exceptions here!
  }

// Common objects can go here

};

TEST_F(LakeFillingTest, LakeFillingTestOne){
  int nlat = 16;
  int nlon = 16;
  bool* lake_minima = new bool[nlat*nlon] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,true, false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* lake_mask = new bool[nlat*nlon] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
//
    false,false,false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
    false,false,false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
    false,false,false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
    false,false,false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
//
    false,false,false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* orography = new double[16*16] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.5, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.6,0.2,0.2,0.3, 0.4,1.8,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.5,0.2,0.1,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.5,0.2,0.1,0.2, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.4,0.2,0.2,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
//
    0.0,1.5,2.0,2.0, 0.4,0.4,0.4,0.4, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,1.5,1.5, 1.5,1.5,1.2,1.5, 1.5,1.1,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
  };
  double* expected_orography_out = new double[16*16] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,1.8,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
//
    0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,1.5,1.5, 1.5,1.5,1.2,1.5, 1.5,1.1,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
  };
  latlon_fill_lakes(lake_minima,lake_mask,orography,nlat,nlon,false);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(orography[i],expected_orography_out[i]);
  }
  delete[] expected_orography_out;
  delete[] orography;
  delete[] lake_mask;
  delete[] lake_minima;
}

TEST_F(LakeFillingTest, LakeFillingTestTwo){
  int nlat = 16;
  int nlon = 16;
  bool* lake_minima = new bool[nlat*nlon] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,true, false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* lake_mask = new bool[nlat*nlon] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
//
    false,false,false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
    false,false,false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
    false,false,false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
    false,false,false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
//
    false,false,false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* orography = new double[16*16] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.6,0.6,0.6,0.5, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.6,0.2,0.2,0.3, 0.4,1.8,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.5,0.2,0.1,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.5,0.2,0.1,0.2, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 0.4,0.2,0.2,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
//
    0.0,1.5,2.0,2.0, 0.4,0.4,0.4,0.4, 0.4,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,1.5,1.5, 1.5,1.5,1.2,1.5, 1.5,1.1,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
  };
  double* expected_orography_out = new double[16*16] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    0.0,1.5,1.5,1.5, 1.1,1.5,1.5,1.5, 1.5,1.5,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,1.8,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
//
    0.0,1.5,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,1.5,0.0,
    0.0,1.5,1.5,1.5, 1.5,1.5,1.2,1.5, 1.5,1.1,1.5,1.5, 1.5,1.5,1.5,0.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
  };
  latlon_fill_lakes(lake_minima,lake_mask,orography,nlat,nlon,true);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(orography[i],expected_orography_out[i]);
  }
  delete[] expected_orography_out;
  delete[] orography;
  delete[] lake_mask;
  delete[] lake_minima;
}

TEST_F(LakeFillingTest, LakeFillingTestThree){
  int nlat = 16;
  int nlon = 16;
  bool* lake_minima = new bool[nlat*nlon] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,true, false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* lake_mask = new bool[nlat*nlon] {
    true, true, true, false, false,false,false,false, false,false,false,false, false,false,false,true,
    true, true, true, true,  false,false,false,false, false,false,false,false, false,false,false,true,
    true, true, true, false, false,false,false,false, false,false,false,false, false,false,false,true,
    true, true, false,false, true, true, true, true,  true, false,false,false, false,false,false,true,
//
    true, false,false,false, true, true, true, true,  true, false,false, true, false,false,false,true,
    true, false,false,false, true, true, true, true,  true, false, true,false, true, true, false,true,
    true, false,false,false, true, true, true, true,  true, false, true,false, false,true, true, true,
    true, false,false,false, true, true, true, true,  true, false,false,false, false,false,false,true,
//
    true, true, false,false, true, true, true, true,  true, false,false,false, false,false,false,true,
    false,true, true, false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,true, false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,true, true,  false,false,false,false, false,false,false,false, false,false,false,true,
  };
  double* orography = new double[16*16] {
   -0.1,-0.0,0.3,2.0,2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1,
    0.2,0.1,0.2,0.5, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    0.2,0.5,0.3,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    0.2,0.2,2.0,2.0, 0.6,0.6,0.6,0.5, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    0.3,2.0,2.0,2.0, 0.6,0.2,0.2,0.3, 0.4,1.8,2.0,0.1, 2.0,2.0,2.0,0.2,
    0.3,2.0,2.0,2.0, 0.5,0.2,0.1,0.3, 0.4,2.0,0.2,2.0, 0.2,0.1,2.0,0.3,
    0.2,2.0,2.0,2.0, 0.5,0.2,0.1,0.2, 0.4,2.0,0.3,2.0, 2.0,0.2,0.1,0.2,
    0.1,2.0,2.0,2.0, 0.4,0.2,0.2,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.5,
//
    0.2,0.3,2.0,2.0, 0.4,0.4,0.4,0.4, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.3,
    2.0,0.1,0.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    2.0,2.0,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1,
    2.0,2.0,0.1,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    2.0,2.0,0.2,0.3, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3
  };
  double* expected_orography_out = new double[16*16] {
    1.7,1.7,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    1.7,1.7,1.7,1.7, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    1.7,1.7,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    1.7,1.7,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,1.8,2.0,1.7, 2.0,2.0,2.0,1.7,
    1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,1.7,2.0, 1.7,1.7,2.0,1.7,
    1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,1.7,2.0, 2.0,1.7,1.7,1.7,
    1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
//
    1.7,1.7,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    2.0,1.7,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    2.0,2.0,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    2.0,2.0,0.1,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    2.0,2.0,0.2,0.3, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7
  };
  latlon_fill_lakes(lake_minima,lake_mask,orography,nlat,nlon,true);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(orography[i],expected_orography_out[i]);
  }
  delete[] expected_orography_out;
  delete[] orography;
  delete[] lake_mask;
  delete[] lake_minima;
}

TEST_F(LakeFillingTest, LakeFillingTestFour){
  int nlat = 16;
  int nlon = 16;
  bool* lake_minima = new bool[nlat*nlon] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,true, false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,true, false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* lake_mask = new bool[nlat*nlon] {
    true, true, true, false, false,false,false,false, false,false,false,false, false,false,false,true,
    true, true, true, true,  false,false,false,false, false,false,false,false, false,false,false,true,
    true, true, true, false, false,false,false,false, false,false,false,false, false,false,false,true,
    true, true, false,false, true, true, true, true,  true, false,false,false, false,false,false,true,
//
    true, false,false,false, true, true, true, true,  true, false,false, true, false,false,false,true,
    true, false,false,false, true, true, true, true,  true, false, true,false, true, true, false,true,
    true, false,false,false, true, true, true, true,  true, false, true,false, false,true, true, true,
    true, false,false,false, true, true, true, true,  true, false,false,false, false,false,false,true,
//
    true, true, false,false, true, true, true, true,  true, false,false,false, false,false,false,true,
    false,true, true, false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,true, false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,true, false,  false,false,false,false, false,false,false,false, false,false,false,true,
  };
  double* orography = new double[16*16] {
   -0.1,-0.0,0.3,2.0,2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1,
    0.2,0.1,0.2,0.5, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    0.2,0.5,0.3,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    0.2,0.2,2.0,2.0, 0.6,0.6,0.6,0.5, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    0.3,2.0,2.0,2.0, 0.6,0.2,0.2,0.3, 0.4,1.8,2.0,0.1, 2.0,2.0,2.0,0.2,
    0.3,2.0,2.0,2.0, 0.5,0.2,0.1,0.3, 0.4,2.0,0.2,2.0, 0.2,0.1,2.0,0.3,
    0.2,2.0,2.0,2.0, 0.5,0.2,0.1,0.2, 0.4,2.0,0.3,2.0, 2.0,0.2,0.1,0.2,
    0.1,2.0,2.0,2.0, 0.4,0.2,0.2,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.5,
//
    0.2,0.3,2.0,2.0, 0.4,0.4,0.4,0.4, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.3,
    2.0,0.1,0.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    2.0,2.0,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1,
    2.0,2.0,0.1,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    2.0,2.0,0.2,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3
  };
  double* expected_orography_out = new double[16*16] {
    1.7,1.7,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    1.7,1.7,1.7,1.7, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    1.7,1.7,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    1.7,1.7,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,1.8,2.0,1.7, 2.0,2.0,2.0,1.7,
    1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,1.7,2.0, 1.7,1.7,2.0,1.7,
    1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,1.7,2.0, 2.0,1.7,1.7,1.7,
    1.7,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
//
    1.7,1.7,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    2.0,1.7,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    2.0,2.0,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    2.0,2.0,0.3,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7,
    2.0,2.0,0.3,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.7
  };
  latlon_fill_lakes(lake_minima,lake_mask,orography,nlat,nlon,true);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(orography[i],expected_orography_out[i]);
  }
  delete[] expected_orography_out;
  delete[] orography;
  delete[] lake_mask;
  delete[] lake_minima;
}

TEST_F(LakeFillingTest, LakeFillingTestSix){
  int nlat = 16;
  int nlon = 16;
  bool* lake_minima = new bool[nlat*nlon] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,true, false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,true, false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  bool* lake_mask = new bool[nlat*nlon] {
    true, true, true, false, false,false,false,false, false,false,false,false, false,false,false,true,
    true, true, true, true,  false,false,false,false, false,false,false,false, false,false,false,true,
    true, true, true, false, false,false,false,false, false,false,false,false, false,false,false,true,
    true, true, false,false, true, true, true, true,  true, false,false,false, false,false,false,true,
//
    true, false,false,false, true, true, true, true,  true, false,false, true, false,false,false,true,
    true, false,false,false, true, true, true, true,  true, false, true,false, true, true, false,true,
    true, false,false,false, true, true, true, true,  true, false, true,false, false,true, true, true,
    true, false,false,false, true, true, true, true,  true, false,false,false, false,false,false,true,
//
    true, true, false,false, true, true, true, true,  true, false,false,false, false,false,false,true,
    false,true, true, false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,true, false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,true, false,  false,false,false,false, false,false,false,false, false,false,false,true,
  };
  double* orography = new double[16*16] {
   -0.1,-0.0,0.3,2.0,2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1,
    0.2,0.1,0.2,0.5, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    0.2,0.5,0.3,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    0.2,0.2,2.0,2.0, 0.6,0.6,0.6,0.5, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    0.3,2.0,2.0,2.0, 0.6,0.2,0.2,0.3, 0.4,1.8,2.0,0.1, 2.0,2.0,2.0,0.2,
    0.3,2.0,2.0,2.0, 0.5,0.2,0.1,0.3, 0.4,2.0,0.2,2.0, 0.2,0.1,2.0,0.3,
    0.2,2.0,2.0,2.0, 0.5,0.2,0.1,0.2, 0.4,2.0,0.3,2.0, 2.0,0.2,0.1,0.2,
    0.1,2.0,2.0,2.0, 0.4,0.2,0.2,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.5,
//
    0.2,0.3,2.0,2.0, 0.4,0.4,0.4,0.4, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,0.3,
    2.0,0.1,0.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    2.0,2.0,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.1,
    2.0,2.0,0.1,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    2.0,2.0,0.2,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3
  };
  double* expected_orography_out = new double[16*16] {
    0.7,0.7,0.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7,
    0.7,0.7,0.7,0.7, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7,
    0.7,0.7,0.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7,
    0.7,0.7,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,2.0,0.7,
    0.7,2.0,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,1.8,2.0,0.7, 2.0,2.0,2.0,0.7,
    0.7,2.0,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,0.7,2.0, 0.7,0.7,2.0,0.7,
    0.7,2.0,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,0.7,2.0, 2.0,0.7,0.7,0.7,
    0.7,2.0,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,2.0,0.7,
//
    0.7,0.7,2.0,2.0, 0.6,0.6,0.6,0.6, 0.6,2.0,2.0,2.0, 2.0,2.0,2.0,0.7,
    2.0,0.7,0.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7,
    2.0,2.0,1.7,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7,
    2.0,2.0,0.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7,
    2.0,2.0,0.2,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7
  };
  latlon_fill_lakes(lake_minima,lake_mask,orography,nlat,nlon,false);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(orography[i],expected_orography_out[i]);
  }
  delete[] expected_orography_out;
  delete[] orography;
  delete[] lake_mask;
  delete[] lake_minima;
}

TEST_F(LakeFillingTest, LakeFillingTestSeven){
  int nlat = 16;
  int nlon = 16;
  bool* lake_minima = new bool[nlat*nlon] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,true, false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,true, false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,true, false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,true, false, false,false,false,false, false,false,false,false, false,false,false,true
  };
  bool* lake_mask = new bool[nlat*nlon] {
    true, true, true, false, false,false,false,false, false,false,false,false, false,false,false,false,
    true, true, true, true,  false,false,true,false,  false,false,false,false, false,false,false,false,
    true, true, true, false, false,false,true,false,  false,false,false,false, false,false,false,false,
    true, true, false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
//
    true, false,false,false, true, true, true, true,  true, false,false, true, false,false,false,false,
    true, false,false,false, true, true, true, true,  true, false, true,false, false,false,false,false,
    true, false,false,false, true, true, true, true,  true, false, true,false, false,false,false,false,
    true, false,false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
//
    true, true, false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
    false,true, true, false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,true, false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,true, false, false,false,false,false, false,false,false,false, false,false,false,true,
  };
  double* orography = new double[16*16] {
   -0.1,-0.0,0.3,2.0,2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    0.2,0.1,0.2,0.5, 2.0,2.0,1.9,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    0.2,0.5,0.3,2.0, 2.0,2.0,1.9,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.4,
    0.2,0.2,2.0,2.0, 0.6,0.6,0.6,0.5, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    0.3,2.0,2.0,2.0, 0.6,0.2,0.2,0.3, 0.4,1.8,2.0,0.1, 2.0,2.0,2.0,2.0,
    0.3,2.0,2.0,2.0, 0.5,0.2,0.1,0.3, 0.4,2.0,0.2,2.0, 0.2,0.1,2.0,2.0,
    0.2,2.0,2.0,2.0, 0.5,0.2,0.1,0.2, 0.4,2.0,0.3,2.0, 2.0,0.2,0.1,2.0,
    0.1,2.0,2.0,2.0, 0.4,0.2,0.2,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
//
    0.2,0.3,2.0,2.0, 0.4,0.4,0.4,0.4, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,0.1,0.7,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,1.7,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,0.1,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.2,
    0.7,2.0,0.2,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3
  };
  double* expected_orography_out = new double[16*16] {
    1.4,1.4,1.4,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    1.4,1.4,1.4,1.4, 2.0,2.0,1.9,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    1.4,1.4,1.4,2.0, 2.0,2.0,1.9,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.4,
    1.4,1.4,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,1.8,2.0,0.1, 2.0,2.0,2.0,2.0,
    1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,0.2,2.0, 0.2,0.1,2.0,2.0,
    1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,0.3,2.0, 2.0,0.2,0.1,2.0,
    1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
//
    1.4,1.4,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,1.4,1.4,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,1.7,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,0.3,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7,
    0.7,2.0,0.3,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7
  };
  latlon_fill_lakes(lake_minima,lake_mask,orography,nlat,nlon,true);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(orography[i],expected_orography_out[i]);
  }
  delete[] expected_orography_out;
  delete[] orography;
  delete[] lake_mask;
  delete[] lake_minima;
}

TEST_F(LakeFillingTest, LakeFillingTestEight){
  int nlat = 16;
  int nlon = 16;
  bool* lake_minima = new bool[nlat*nlon] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,true, false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,true, false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,true, false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,true, false, false,false,false,false, false,false,false,false,
    false,false,false,true,  false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,true, true,  false,true, false,false, false,false,false,false,
    false,false,false,false, false,false,true, true,  false,false,false,false, false,false,false,false,
    false,false,true, false, false,false,false,false, false,false,false,false, false,false,false,true
  };
  bool* lake_mask = new bool[nlat*nlon] {
    true, true, true, false, false,false,false,false, false,false,false,false, false,false,false,false,
    true, true, true, true,  false,false,true,false,  false,false,false,false, false,false,false,false,
    true, true, true, false, false,false,true,false,  false,false,false,false, false,false,false,false,
    true, true, false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
//
    true, false,false,false, true, true, true, true,  true, false,false, true, false,false,false,false,
    true, false,false,false, true, true, true, true,  true, false, true,false, false,false,false,false,
    true, false,false,false, true, true, true, true,  true, false, true,false, false,false,false,false,
    true, false,false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
//
    true, true, false,false, true, true, true, true,  true, false,false,false, false,false,false,false,
    false,true, true, false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,true, false, false,false,false,false, false,false,false,false, false,false,false,true,
    false,false,true, false, false,false,false,false, false,false,false,false, false,false,false,true,
  };
  double* orography = new double[16*16] {
   -0.1,-0.0,0.3,2.0,2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    0.2,0.1,0.2,0.5, 2.0,2.0,1.9,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    0.2,0.5,0.3,2.0, 2.0,2.0,1.9,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.4,
    0.2,0.2,2.0,2.0, 0.6,0.6,0.6,0.5, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    0.3,2.0,2.0,2.0, 0.6,0.2,0.2,0.3, 0.4,1.8,2.0,0.1, 2.0,2.0,2.0,2.0,
    0.3,2.0,2.0,2.0, 0.5,0.2,0.1,0.3, 0.4,2.0,0.2,2.0, 0.2,0.1,2.0,2.0,
    0.2,2.0,2.0,2.0, 0.5,0.2,0.1,0.2, 0.4,2.0,0.3,2.0, 2.0,0.2,0.1,2.0,
    0.1,2.0,2.0,2.0, 0.4,0.2,0.2,0.3, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
//
    0.2,0.3,2.0,2.0, 0.4,0.4,0.4,0.4, 0.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,0.1,0.7,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,1.7,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0, 2.0,2.0,0.2,0.1, 2.0,0.1,0.1,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,0.1,2.0, 2.0,2.0,0.3,0.2, 2.0,0.2,0.2,2.0, 2.0,2.0,2.0,0.2,
    0.7,2.0,0.2,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.3
  };
  double* expected_orography_out = new double[16*16] {
    1.4,1.4,1.4,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    1.4,1.4,1.4,1.4, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    1.4,1.4,1.4,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,1.4,
    1.4,1.4,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,1.8,2.0,0.1, 2.0,2.0,2.0,2.0,
    1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,0.2,2.0, 0.2,0.1,2.0,2.0,
    1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,0.3,2.0, 2.0,0.2,0.1,2.0,
    1.4,2.0,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
//
    1.4,1.4,2.0,2.0, 1.8,1.8,1.8,1.8, 1.8,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,1.4,1.4,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,1.7,2.0, 2.0,2.0,2.2,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,2.0,2.0, 2.0,2.0,0.2,0.1, 2.0,0.1,0.1,2.0, 2.0,2.0,2.0,2.0,
    2.0,2.0,0.3,2.0, 2.0,2.0,0.3,0.2, 2.0,0.2,0.2,2.0, 2.0,2.0,2.0,0.7,
    0.7,2.0,0.3,0.3, 1.4,2.0,2.0,2.0, 2.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.7
  };
  latlon_fill_lakes(lake_minima,lake_mask,orography,nlat,nlon,true);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(orography[i],expected_orography_out[i]);
  }
  delete[] expected_orography_out;
  delete[] orography;
  delete[] lake_mask;
  delete[] lake_minima;
}

class WaterRedistributionTest : public ::testing::Test {
 protected:

  WaterRedistributionTest() {
  }

  virtual ~WaterRedistributionTest() {
    // Don't include exceptions here!
  }

// Common objects can go here

};


TEST_F(WaterRedistributionTest, WaterRedistributionTestOne){
  int nlat = 16;
  int nlon = 16;
  int coarse_nlat = 4;
  int coarse_nlon = 4;
  int* lake_numbers_in = new int[nlat*nlon] {
    0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,9, 9,9,9,9, 9,9,9,0,
    0,0,0,0, 3,3,0,0, 0,9,9,0, 0,0,9,0,
    0,0,0,0, 3,0,3,0, 0,9,9,0, 0,0,0,0,
//
    0,0,0,3, 1,0,0,0, 0,0,0,0, 0,0,0,1,
    0,0,15,0, 0,0,0,0, 0,0,4,0, 0,0,1,0,
    15,15,0,0, 0,0,0,0, 0,4,0,0, 0,0,0,0,
    15,15,0,0, 0,0,0,0, 0,4,4,0, 0,0,0,0,
//
    0,0,2,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
    2,2,2,0, 0,0,0,0, 0,0,0,0, 0,0,0,2,
    0,0,2,2, 2,0,0,0, 0,0,0,0, 0,2,2,2,
    0,0,0,0, 0,0,0,0, 0,0,7,0, 0,0,0,0,
//
    0,0,0,0, 0,0,0,0, 0,0,7,0, 0,0,0,1,
    0,0,0,12, 0,0,0,0, 0,0,7,0, 0,1,1,0,
    0,0,0,12, 12,0,0,0, 0,0,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0
  };
  bool* lake_centers_in = new bool[nlat*nlon] {
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false, true, false,false,false,false, false,false,false,false, false,false,false, true,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false, true,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false, true,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false, true,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
//
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false, true,false, false,false,false,false,
    false,false,false, true, false,false,false,false, false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false, false,false,false,false, false,false,false,false
  };
  double* water_to_redistribute_in = new double[nlat*nlon]{
      0,0,  1,0, 0,0,0,0, 0,  0,21.0,0,   0,0,  0,0,
    1.5,0,  0,0, 0,0,0,0, 0,7.0,   0,0, 8.0,0,7.5,0,
      0,0,1.6,0, 0,0,0,0, 0,  0,   0,0,   0,0,  0,0,
      0,0,  0,0, 0,0,0,0, 0,  0,   0,0,   0,0,  0,0,
//
    3.0,0,  0,2.0, 0,  0,0,0, 0,0,  0,0,   0,0,0,7.0,
      0,0,6.0,  0, 0,2.2,0,0, 0,0,  0,0,   0,0,0,  0,
      0,0,  0,  0, 0,  0,0,0, 0,0,6.0,0,   0,0,0,  0,
    1.5,0,  0,  0, 0,  0,0,0, 0,0,  0,0, 4.0,0,0,  0,
//
      0,  0,0,0, 0,0,0,0, 0,0,   0,0, 0,0,0,  0,
    2.8,  0,0,0, 0,0,0,0, 0,0,   0,0, 0,0,0,9.0,
      0,  0,0,0, 0,0,0,0, 0,0,13.0,0, 0,0,0,  0,
      0,1.0,0,0, 0,0,0,0, 0,0,   0,0, 0,0,0,  0,
//
      0,0,0,  0, 0,0,0,8.0, 0,0,  0,0,    0,  0,  0,0,
      0,0,0,  0, 0,0,0,  0, 0,0,2.8,0, 13.0,2.1,3.8,0,
    2.5,0,0,1.1, 0,0,0,  0, 0,0,  0,0,  5.0,  0,  0,0,
      0,0,0,  0, 0,0,0,  0, 0,0,  0,0,    0,  0,  0,0
  };
  double* water_redistributed_to_lakes_out = new double[nlat*nlon];
  fill_n(water_redistributed_to_lakes_out,nlat*nlon,0.0);
  double* water_redistributed_to_rivers_out = new double[coarse_nlat*coarse_nlon];
  fill_n(water_redistributed_to_rivers_out,coarse_nlat*coarse_nlon,0.0);
  double* water_redistributed_to_lakes_expected_out = new double[nlat*nlon]{
        0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,22.5,0.0, 0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
//
        0.0,0.0,0.0,2.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,12.9,
        0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
        0.0,7.5,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
//
        0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
        0.0,0.0,11.8,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
//
        0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,2.8,0.0, 0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,1.1, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0
  };
  double* water_redistributed_to_rivers_expected_out =
    new double[coarse_nlat*coarse_nlon]{
     4.1,0.0,21.0,0.0,
     3.0,2.2,6.0,4.0,
     1.0,0.0,13.0,0.0,
     2.5,8.0,0.0,18.0
    };
  latlon_redistribute_water(lake_numbers_in,
                            lake_centers_in,
                            water_to_redistribute_in,
                            water_redistributed_to_lakes_out,
                            water_redistributed_to_rivers_out,
                            nlat,nlon,coarse_nlat,coarse_nlon);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_DOUBLE_EQ(water_redistributed_to_lakes_expected_out[i],water_redistributed_to_lakes_out[i]);
  }
  for (auto i =0; i < coarse_nlat*coarse_nlon; i++){
    EXPECT_DOUBLE_EQ(water_redistributed_to_rivers_expected_out[i],water_redistributed_to_rivers_out[i]);
  }
  delete[] water_redistributed_to_rivers_expected_out;
  delete[] water_redistributed_to_lakes_expected_out;
  delete[] water_redistributed_to_rivers_out;
  delete[] water_redistributed_to_lakes_out;
  delete[] water_to_redistribute_in;
  delete[] lake_centers_in;
  delete[] lake_numbers_in;
}

class FilterOutShallowLakeTest : public ::testing::Test {
 protected:

  FilterOutShallowLakeTest() {
  }

  virtual ~FilterOutShallowLakeTest() {
    // Don't include exceptions here!
  }

// Common objects can go here

};

TEST_F(FilterOutShallowLakeTest, FilterOutShallowLakeTestOne){
  int nlat = 20;
  int nlon = 20;
  double* filled_orography_in = new double[nlat*nlon]{
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     9.9,8.9,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.3,8.3,8.8, 8.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     8.1,8.1,8.1,8.1,10.7, 8.8,8.6,8.3,8.3,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,8.1, 8.1,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,8.1,9.9, 8.1,8.5,  8.9,8.4,9.5,8.1,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.1,9.5,9.9,10.2,4.2,  8.9,8.4,8.1,8.8,10.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9, 8.1,  9.9,9.2,9.3, 9.1,3.5,  8.7,8.2,8.1,8.1, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8, 8.1,  8.8,8.6,8.1,9.8, 4.9,  8.8,8.6,8.1,8.1,10.7,
     8.8,8.6,8.8,9.8,10.7, 8.8,8.6,8.1,9.8, 8.1,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
    10.1,8.9,8.6,9.8,10.0, 8.8,8.1,8.1,8.8, 8.1,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     8.2,8.2,9.5,9.8,10.0, 8.8,8.1,8.1,8.1,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.2, 8.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,8.1,8.1,8.1,10.7, 8.8,8.1,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
    10.1,8.1,8.1,8.1,10.0, 8.1,8.9,8.1,8.1, 8.1,  8.7,9.5,8.1, 8.1,8.1,  8.9,8.4,9.5,8.8,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.1,10.3,  8.7,8.1,9.9,10.2,8.5,  8.9,8.4,8.1,8.8,10.2,
     8.2,8.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.1,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 8.2
  };

  double* unfilled_orography_in = new double[nlat*nlon]{
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     9.9,1.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,6.9,6.3,8.8, 6.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     7.8,7.6,8.1,6.8,10.7, 8.8,8.6,8.3,6.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,1.8, 6.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,7.2,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,7.1,9.8,10.7,  8.8,8.6,7.1,9.8,10.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,7.0,9.9, 5.4,8.5,  8.9,8.4,9.5,5.2,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  6.3,9.5,9.9,10.2,4.2,  8.9,8.4,3.5,8.8,10.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9, 5.3,  9.9,9.2,9.3, 9.1,3.5,  8.7,8.2,8.1,4.6, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8, 5.2,  8.8,8.6,8.1,9.8, 4.9,  8.8,8.6,8.1,5.2,10.7,
     8.8,8.6,8.8,9.8,10.7, 8.8,8.6,8.1,9.8, 5.0,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,5.9,9.8,10.7,
    10.1,8.9,4.5,9.8,10.0, 8.8,2.3,9.3,8.8,-0.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     7.1,8.2,9.5,9.8,10.0, 8.8,2.1,1.0,0.9,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,7.8, 7.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,6.6,1.1,6.8,10.7, 8.8,6.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,5.3,9.8,10.7,
    10.1,7.9,1.5,7.8,10.0, 1.8,8.9,6.3,5.8, 1.3,  8.7,9.5,7.9, 7.2,0.5,  8.9,8.4,9.5,8.8,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,6.8,10.3,  8.7,1.5,9.9,10.2,8.5,  8.9,8.4,7.5,8.8,10.2,
     5.9,6.2,9.3,9.1, 8.8, 8.2,8.3,8.7,1.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 5.7
  };

  double* unfilled_orography_expected_out = new double[nlat*nlon]{
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     9.9,1.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.3,8.3,8.8, 8.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     7.8,7.6,8.1,8.1,10.7, 8.8,8.6,8.3,8.3,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,1.8, 6.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,7.2,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,7.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,7.0,9.9, 5.4,8.5,  8.9,8.4,9.5,8.1,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  6.3,9.5,9.9,10.2,4.2,  8.9,8.4,8.1,8.8,10.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9, 5.3,  9.9,9.2,9.3, 9.1,3.5,  8.7,8.2,8.1,8.1, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8, 5.2,  8.8,8.6,8.1,9.8, 4.9,  8.8,8.6,8.1,8.1,10.7,
     8.8,8.6,8.8,9.8,10.7, 8.8,8.6,8.1,9.8, 5.0,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
    10.1,8.9,8.6,9.8,10.0, 8.8,2.3,9.3,8.8,-0.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     8.2,8.2,9.5,9.8,10.0, 8.8,2.1,1.0,0.9,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.2, 8.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,6.6,1.1,6.8,10.7, 8.8,6.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
    10.1,7.9,1.5,7.8,10.0, 1.8,8.9,6.3,5.8, 1.3,  8.7,9.5,7.9, 7.2,0.5,  8.9,8.4,9.5,8.8,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,6.8,10.3,  8.7,1.5,9.9,10.2,8.5,  8.9,8.4,8.1,8.8,10.2,
     8.2,8.2,9.3,9.1, 8.8, 8.2,8.3,8.7,1.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 8.2
  };
  latlon_filter_out_shallow_lakes(unfilled_orography_in,filled_orography_in,
                                  5.0,nlat,nlon);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(unfilled_orography_in[i],unfilled_orography_expected_out[i]);
  }
  delete[] unfilled_orography_expected_out;
  delete[] unfilled_orography_in;
  delete[] filled_orography_in;
};

TEST_F(FilterOutShallowLakeTest, FilterOutShallowLakeTestTwo){
  int nlat = 20;
  int nlon = 20;
  double* filled_orography_in = new double[nlat*nlon]{
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     9.9,8.9,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.3,8.3,8.8, 8.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     8.1,8.1,8.1,8.1,10.7, 8.8,8.6,8.3,8.3,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,8.1, 8.1,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,8.1,9.9, 8.1,8.5,  8.9,8.4,9.5,8.1,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.1,9.5,9.9,10.2,4.2,  8.9,8.4,8.1,8.8,10.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9, 8.1,  9.9,9.2,9.3, 9.1,3.5,  8.7,8.2,8.1,8.1, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8, 8.1,  8.8,8.6,8.1,9.8, 4.9,  8.8,8.6,8.1,8.1,10.7,
     8.8,8.6,8.8,9.8,10.7, 8.8,8.6,8.1,9.8, 8.1,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
    10.1,8.9,8.6,9.8,10.0, 8.8,8.1,8.1,8.8, 8.1,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     8.2,8.2,9.5,9.8,10.0, 8.8,8.1,8.1,8.1,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.2, 8.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,8.1,8.1,8.1,10.7, 8.8,8.1,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
    10.1,8.1,8.1,8.1,10.0, 8.1,8.9,8.1,8.1, 8.1,  8.7,9.5,8.1, 8.1,8.1,  8.9,8.4,9.5,8.8,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.1,10.3,  8.7,8.1,9.9,10.2,8.5,  8.9,8.4,8.1,8.8,10.2,
     8.2,8.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.1,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 8.2
  };

  double* unfilled_orography_in = new double[nlat*nlon]{
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     9.9,1.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,6.9,6.3,8.8, 6.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     7.8,7.6,8.1,6.8,10.7, 8.8,8.6,8.3,6.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,1.8, 6.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,7.2,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,7.1,9.8,10.7,  8.8,8.6,7.1,9.8,10.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,7.0,9.9, 5.4,8.5,  8.9,8.4,9.5,5.2,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  6.3,9.5,9.9,10.2,4.2,  8.9,8.4,3.5,8.8,10.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9, 5.3,  9.9,9.2,9.3, 9.1,3.5,  8.7,8.2,8.1,4.6, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8, 5.2,  8.8,8.6,8.1,9.8, 4.9,  8.8,8.6,8.1,5.2,10.7,
     8.8,8.6,8.8,9.8,10.7, 8.8,8.6,8.1,9.8, 5.0,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,5.9,9.8,10.7,
    10.1,8.9,4.5,9.8,10.0, 8.8,2.3,9.3,8.8,-0.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     7.1,8.2,9.5,9.8,10.0, 8.8,2.1,1.0,0.9,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,7.8, 7.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,6.6,1.1,6.8,10.7, 8.8,6.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,5.3,9.8,10.7,
    10.1,7.9,1.5,7.8,10.0, 1.8,8.9,6.3,5.8, 1.3,  8.7,9.5,7.9, 7.2,0.5,  8.9,8.4,9.5,8.8,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,6.8,10.3,  8.7,1.5,9.9,10.2,8.5,  8.9,8.4,7.5,8.8,10.2,
     5.9,6.2,9.3,9.1, 8.8, 8.2,8.3,8.7,1.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 5.7
  };

  double* unfilled_orography_expected_out = new double[nlat*nlon]{
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     9.9,1.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,6.9,6.3,8.8, 6.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     7.8,7.6,8.1,6.8,10.7, 8.8,8.6,8.3,6.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,1.8, 6.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,7.2,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,7.1,9.8,10.7,  8.8,8.6,7.1,9.8,10.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,7.0,9.9, 5.4,8.5,  8.9,8.4,9.5,5.2,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  6.3,9.5,9.9,10.2,4.2,  8.9,8.4,3.5,8.8,10.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9, 5.3,  9.9,9.2,9.3, 9.1,3.5,  8.7,8.2,8.1,4.6, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8, 5.2,  8.8,8.6,8.1,9.8, 4.9,  8.8,8.6,8.1,5.2,10.7,
     8.8,8.6,8.8,9.8,10.7, 8.8,8.6,8.1,9.8, 5.0,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,5.9,9.8,10.7,
    10.1,8.9,4.5,9.8,10.0, 8.8,2.3,9.3,8.8,-0.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     7.1,8.2,9.5,9.8,10.0, 8.8,2.1,1.0,0.9,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,7.8, 7.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,6.6,1.1,6.8,10.7, 8.8,6.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,5.3,9.8,10.7,
    10.1,7.9,1.5,7.8,10.0, 1.8,8.9,6.3,5.8, 1.3,  8.7,9.5,7.9, 7.2,0.5,  8.9,8.4,9.5,8.8,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,6.8,10.3,  8.7,1.5,9.9,10.2,8.5,  8.9,8.4,7.5,8.8,10.2,
     5.9,6.2,9.3,9.1, 8.8, 8.2,8.3,8.7,1.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 5.7
  };
  latlon_filter_out_shallow_lakes(unfilled_orography_in,filled_orography_in,
                                  0.0,nlat,nlon);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(unfilled_orography_in[i],unfilled_orography_expected_out[i]);
  }
  delete[] unfilled_orography_expected_out;
  delete[] unfilled_orography_in;
  delete[] filled_orography_in;
};

TEST_F(FilterOutShallowLakeTest, FilterOutShallowLakeTestThree){
  int nlat = 20;
  int nlon = 20;
  double* filled_orography_in = new double[nlat*nlon]{
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     9.9,8.9,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.3,8.3,8.8, 8.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     8.1,8.1,8.1,8.1,10.7, 8.8,8.6,8.3,8.3,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,8.1, 8.1,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,8.1,9.9, 8.1,8.5,  8.9,8.4,9.5,8.1,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.1,9.5,9.9,10.2,4.2,  8.9,8.4,8.1,8.8,10.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9, 8.1,  9.9,9.2,9.3, 9.1,3.5,  8.7,8.2,8.1,8.1, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8, 8.1,  8.8,8.6,8.1,9.8, 4.9,  8.8,8.6,8.1,8.1,10.7,
     8.8,8.6,8.8,9.8,10.7, 8.8,8.6,8.1,9.8, 8.1,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
    10.1,8.9,8.6,9.8,10.0, 8.8,8.1,9.3,8.8, 8.1,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     8.2,8.2,9.5,9.8,10.0, 8.8,8.1,8.1,8.1,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.2, 8.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,8.1,8.1,8.1,10.7, 8.8,8.1,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
    10.1,8.1,8.1,8.1,10.0, 8.1,8.9,8.1,8.1, 8.1,  8.7,9.5,8.1, 8.1,8.1,  8.9,8.4,9.5,8.8,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.1,10.3,  8.7,8.1,9.9,10.2,8.5,  8.9,8.4,8.1,8.8,10.2,
     8.2,8.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.1,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 8.2
  };

  double* unfilled_orography_in = new double[nlat*nlon]{
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     9.9,1.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,6.9,6.3,8.8, 6.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     7.8,7.6,8.1,6.8,10.7, 8.8,8.6,8.3,6.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,1.8, 6.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,7.2,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,7.1,9.8,10.7,  8.8,8.6,7.1,9.8,10.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,7.0,9.9, 5.4,8.5,  8.9,8.4,9.5,5.2,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  6.3,9.5,9.9,10.2,4.2,  8.9,8.4,3.5,8.8,10.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9, 5.3,  9.9,9.2,9.3, 9.1,3.5,  8.7,8.2,8.1,4.6, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8, 5.2,  8.8,8.6,8.1,9.8, 4.9,  8.8,8.6,8.1,5.2,10.7,
     8.8,8.6,8.8,9.8,10.7, 8.8,8.6,8.1,9.8, 5.0,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,5.9,9.8,10.7,
    10.1,8.9,4.5,9.8,10.0, 8.8,2.3,9.3,8.8,-0.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     7.1,8.2,9.5,9.8,10.0, 8.8,2.1,1.0,0.9,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,7.8, 7.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,6.6,1.1,6.8,10.7, 8.8,6.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,5.3,9.8,10.7,
    10.1,7.9,1.5,7.8,10.0, 1.8,8.9,6.3,5.8, 1.3,  8.7,9.5,7.9, 7.2,0.5,  8.9,8.4,9.5,8.8,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,6.8,10.3,  8.7,1.5,9.9,10.2,8.5,  8.9,8.4,7.5,8.8,10.2,
     5.9,6.2,9.3,9.1, 8.8, 8.2,8.3,8.7,1.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 5.7
  };

  double* unfilled_orography_expected_out = new double[nlat*nlon]{
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     9.9,8.9,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.3,8.3,8.8, 8.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     8.1,8.1,8.1,8.1,10.7, 8.8,8.6,8.3,8.3,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,8.1, 8.1,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.7,8.1,9.9, 8.1,8.5,  8.9,8.4,9.5,8.1,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.8,10.3,  8.1,9.5,9.9,10.2,4.2,  8.9,8.4,8.1,8.8,10.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9, 8.1,  9.9,9.2,9.3, 9.1,3.5,  8.7,8.2,8.1,8.1, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8, 8.1,  8.8,8.6,8.1,9.8, 4.9,  8.8,8.6,8.1,8.1,10.7,
     8.8,8.6,8.8,9.8,10.7, 8.8,8.6,8.1,9.8, 8.1,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
    10.1,8.9,8.6,9.8,10.0, 8.8,8.1,9.3,8.8, 8.1,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.8,10.2,
     8.2,8.2,9.5,9.8,10.0, 8.8,8.1,8.1,8.1,10.3,  8.7,9.5,9.9,10.2,8.5,  8.9,8.4,9.5,8.2, 8.2,
     9.9,9.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.9,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 9.7,
//
     8.8,8.6,8.1,9.8,10.7, 8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
     8.8,8.1,8.1,8.1,10.7, 8.8,8.1,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,  8.8,8.6,8.1,9.8,10.7,
    10.1,8.1,8.1,8.1,10.0, 8.1,8.9,8.1,8.1, 8.1,  8.7,9.5,8.1, 8.1,8.1,  8.9,8.4,9.5,8.8,10.2,
    10.1,8.9,9.5,9.8,10.0, 8.8,8.9,9.3,8.1,10.3,  8.7,8.1,9.9,10.2,8.5,  8.9,8.4,8.1,8.8,10.2,
     8.2,8.2,9.3,9.1, 8.8, 8.2,8.3,8.7,8.1,10.2,  9.9,9.2,9.3, 9.1,8.3,  8.7,8.2,8.1,8.2, 8.2
  };
  latlon_filter_out_shallow_lakes(unfilled_orography_in,filled_orography_in,
                                  11.0,nlat,nlon);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(unfilled_orography_in[i],unfilled_orography_expected_out[i]);
  }
  delete[] unfilled_orography_expected_out;
  delete[] unfilled_orography_in;
  delete[] filled_orography_in;
};

}  // namespace
