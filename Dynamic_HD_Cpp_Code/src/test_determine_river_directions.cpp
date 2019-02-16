/*
 * test_determine_river_directions.cpp
 *
 * Unit test for the various lake related C++ code using Google's
 * Google Test Framework
 *
 *  Created on: Mar 16, 2019
 *      Author: thomasriddick using Google's recommended template code
 */

#include "determine_river_directions.hpp"
#include "gtest/gtest.h"
using namespace std;

namespace determine_river_directions_unittests {

class DetermineRiverDirectionsTest : public ::testing::Test {
 protected:

  DetermineRiverDirectionsTest() {
  }

  virtual ~DetermineRiverDirectionsTest() {
    // Don't include exceptions here!
  }

// Common objects can go here

};


TEST_F(DetermineRiverDirectionsTest,DetermineRiverDirectionsTestOne) {
  int nlat = 8;
  int nlon = 8;
  double* orography = new double[8*8] {
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,
    10.0,10.0, 9.8,10.0,  0.0, 3.0,10.0,10.0,
    10.0, 9.0,10.0,10.0, 10.0,10.0, 4.0,10.0,
    10.0, 8.3,10.0,10.0, 10.0,10.0, 4.1,10.0,
    10.0, 7.3,10.0,10.0, 10.0,10.0, 4.8,10.0,
    10.0,10.0, 7.1, 6.0,  5.0, 4.9,10.0,10.0,
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0
  };

  bool* lsmask = new bool[8*8] {
    false,false,false,false, false,false,false,false,
    false,false,false,false,  true,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false
  };

  bool* truesinks = new bool[8*8] {
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false
  };

  double* expected_rdirs_out = new double[8*8] {
    10.0,   3,   2,   3, 10.0,10.0,10.0,10.0,
       3,   2, 9.8,   6,  0.0, 3.0,10.0,10.0,
       3, 9.0,10.0,10.0, 10.0,10.0, 4.0,10.0,
       3, 8.3,10.0,10.0, 10.0,10.0, 4.1,10.0,
       6, 7.3,10.0,10.0, 10.0,10.0, 4.8,10.0,
    10.0,10.0, 7.1, 6.0,  5.0, 4.9,10.0,10.0,
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0
  };

  double* rdirs = new double[8*8];
  bool always_flow_to_sea_in = true;
  bool use_diagonal_nbrs_in = true;
  bool mark_pits_as_true_sinks_in = false;
  latlon_determine_river_directions(rdirs,
                                    orography,
                                    lsmask,
                                    truesinks,
                                    nlat,nlon,
                                    always_flow_to_sea_in,
                                    use_diagonal_nbrs_in,
                                    mark_pits_as_true_sinks_in);
  for (int i = 0; i < nlat; i++){
    for (int j = 0; j < nlon; j++){
      cout << setw(3) << rdirs[nlat*i+j] << " ";
    }
    cout << endl;
  }
  ASSERT_TRUE(false);
};


} //namespace
