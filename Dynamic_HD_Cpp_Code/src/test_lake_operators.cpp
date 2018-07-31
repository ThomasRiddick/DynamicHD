/*
 * test_lake_operators.cpp
 *
 * Unit test for the various lake related C++ code using Google's
 * Google Test Framework
 *
 *  Created on: Mar 11, 2018
 *      Author: thomasriddick using Google's recommended template code
 */

#include "reduce_connected_areas_to_points.hpp"
#include "burn_carved_rivers.hpp"
#include "fill_lakes.hpp"
#include "fill_sinks.hpp"
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

}  // namespace
