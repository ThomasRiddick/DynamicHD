/*
 * test_lake_operators.cpp
 *
 * Unit test for the various lake related C++ code using Google's
 * Google Test Framework
 *
 *  Created on: Mar 11, 2018
 *      Author: thomasriddick using Google's recommended template code
 */

#include "burn_carved_rivers.hpp"
#include "fill_lakes.hpp"
#include "reduce_connected_areas_to_points.hpp"
#include "gtest/gtest.h"
using namespace std;

namespace lake_operator_unittests {

// The fixture for testing class Foo.
class BurnCarvedRiversTest : public ::testing::Test {
 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

	BurnCarvedRiversTest() {
    // You can do set-up work for each test here.
  }

  virtual ~BurnCarvedRiversTest() {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  // Objects declared here can be used by all tests in the test case for Foo.
};

TEST_F(BurnCarvedRiversTest, BurnCarvedRiversTestOne) {
	int nlat = 10;
	int nlon = 10;
	double* orography = new double[8*8] {
		1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,
		1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0,
		1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0,
		1.0,1.5,2.5,1.2, 1.3,2.5,1.5,1.0,
		1.0,1.5,2.5,1.3, 1.3,2.3,1.4,1.0,
		1.0,1.5,2.5,2.5, 2.5,2.5,1.5,1.0,
		1.0,1.5,1.5,1.5, 1.5,1.5,1.5,1.0,
		1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,
	};
	double* rdirs = new double[8*8] {0.0};
	bool* minima = new bool[8*8] {
		false,false,false,false, false,false,false,false,
		false,false,false,false, false,false,false,false,
		false,false,false,false, false,false,false,false,
		false,false,false,true,  false,false,false,false,
		false,false,false,false,  false,false,false,false,
		false,false,false,false,  false,false,false,false,
		false,false,false,false,  false,false,false,false,
		false,false,false,false,  false,false,false,false,
	};
	bool* lakemask = new bool[8*8] {false};
	latlon_burn_carved_rivers(orography,rdirs,minima,lakemask,nlat,nlon);
}

}  // namespace
