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
       6,   3,   2,   3,    2,   1,   1,   6,
       3,   2,   1,   6,    0,   4,   4,   1,
       3,   2,   1,   9,    8,   7,   7,   4,
       3,   2,   1,   6,    6,   9,   8,   7,
       6,   3,   3,   3,    3,   9,   8,   7,
       9,   6,   6,   6,    6,   9,   8,   7,
       4,   9,   9,   9,    9,   8,   7,   1,
       7,   6,   6,   6,    9,   8,   7,   4
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
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(rdirs[i],expected_rdirs_out[i]);
  }
  delete[] rdirs;
  delete[] expected_rdirs_out;
  delete[] truesinks;
  delete[] lsmask;
  delete[] orography;
};

TEST_F(DetermineRiverDirectionsTest,DetermineRiverDirectionsTestTwo) {
  int nlat = 8;
  int nlon = 8;
  double* orography = new double[8*8] {
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,
    10.0,10.0, 9.8,10.0, -1.0, 3.0,10.0,10.0,
    10.0, 9.0,10.0,10.0, 10.0,10.0, 4.0,10.0,
    10.0, 8.3,10.0,10.0, 10.0,10.0, 4.1,10.0,
    10.0, 7.3,10.0,10.0, 10.0,10.0, 4.8,10.0,
    10.0,10.0, 7.1, 6.0,  5.0, 4.9,10.0,10.0,
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0
  };

  bool* lsmask = new bool[8*8] {
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false
  };

  bool* truesinks = new bool[8*8] {
    false,false,false,false, false,false,false,false,
    false,false,false,false,  true,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false
  };

  double* expected_rdirs_out = new double[8*8] {
       6,   3,   2,   3,    2,   1,   1,   6,
       3,   2,   1,   6,    5,   4,   4,   1,
       3,   2,   1,   9,    8,   7,   7,   4,
       3,   2,   1,   6,    9,   9,   8,   7,
       6,   3,   3,   3,    3,   9,   8,   7,
       9,   6,   6,   6,    6,   9,   8,   7,
       6,   9,   9,   9,    9,   8,   7,   6,
       9,   8,   7,   4,    4,   6,   9,   9
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
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(rdirs[i],expected_rdirs_out[i]);
  }
  delete[] rdirs;
  delete[] expected_rdirs_out;
  delete[] truesinks;
  delete[] lsmask;
  delete[] orography;
};

TEST_F(DetermineRiverDirectionsTest,DetermineRiverDirectionsTestThree) {
  int nlat = 8;
  int nlon = 8;
  double* orography = new double[8*8] {
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,
    0.0,  3.0, 10.0,10.0, 10.0,10.0, 9.8,10.0,
    10.0,10.0, 4.0,10.0, 10.0, 9.0,10.0,10.0,
    10.0,10.0, 4.1,10.0, 10.0, 8.3,10.0,10.0,
    10.0,10.0, 4.8,10.0, 10.0, 7.3,10.0,10.0,
     5.0, 4.9,10.0,10.0, 10.0,10.0, 7.1, 6.0,
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0
  };

  bool* lsmask = new bool[8*8] {
    false,false,false,false, false,false,false,false,
     true,false,false,false, false,false,false,false,
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
       2,   1,   1,   6,    6,   3,   2,   3,
       0,   4,   4,   1,    3,   2,   1,   6,
       8,   7,   7,   4,    3,   2,   1,   9,
       1,   9,   8,   7,    3,   2,   1,   2,
       3,   9,   8,   7,    6,   3,   3,   3,
       6,   9,   8,   7,    9,   6,   6,   6,
       9,   8,   7,   6,    3,   9,   9,   9,
       4,   4,   9,   9,    6,   9,   8,   7
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
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(rdirs[i],expected_rdirs_out[i]);
  }
  delete[] rdirs;
  delete[] expected_rdirs_out;
  delete[] truesinks;
  delete[] lsmask;
  delete[] orography;
};

TEST_F(DetermineRiverDirectionsTest,DetermineRiverDirectionsTestFour) {
  int nlat = 8;
  int nlon = 8;
  double* orography = new double[8*8] {
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,
    2.0,  3.0, 10.0,10.0, 10.0,10.0, 9.8,10.0,
    10.0,10.0, 4.0,10.0, 10.0, 9.0,10.0,10.0,
    10.0,10.0, 4.1,10.0, 10.0, 8.3,10.0,10.0,
    10.0,10.0, 4.8,10.0, 10.0, 7.3,10.0,10.0,
     5.0, 4.9,10.0,10.0, 10.0,10.0, 7.1, 6.0,
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0
  };

  bool* lsmask = new bool[8*8] {
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false
  };

  bool* truesinks = new bool[8*8] {
    false,false,false,false, false,false,false,false,
     true,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false
  };

  double* expected_rdirs_out = new double[8*8] {
       2,   1,   1,   6,    6,   3,   2,   3,
       5,   4,   4,   1,    3,   2,   1,   6,
       8,   7,   7,   4,    3,   2,   1,   9,
       4,   9,   8,   7,    3,   2,   1,   7,
       3,   9,   8,   7,    6,   3,   3,   3,
       6,   9,   8,   7,    9,   6,   6,   6,
       9,   8,   7,   1,    3,   9,   9,   9,
       8,   7,   4,   4,    6,   6,   6,   9
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
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(rdirs[i],expected_rdirs_out[i]);
  }
  delete[] rdirs;
  delete[] expected_rdirs_out;
  delete[] truesinks;
  delete[] lsmask;
  delete[] orography;
};

TEST_F(DetermineRiverDirectionsTest,DetermineRiverDirectionsTestFive) {
  int nlat = 8;
  int nlon = 8;
  double* orography = new double[8*8] {
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,
    2.0,  3.0, 10.0,10.0, 10.0,10.0, 9.8,10.0,
    10.0,10.0, 4.0,10.0, 10.0, 9.0,10.0,10.0,
    10.0,10.0, 4.1,10.0, 10.0, 8.3,10.0, 1.0,
    10.0,10.0, 4.8,10.0, 10.0, 7.3,10.0,10.0,
     5.0, 4.9,10.0,10.0, 10.0,10.0, 7.1, 6.0,
    10.0,10.0,10.0,10.0, 10.0,10.0,10.0,10.0,
    10.0,-1.0,10.0, 1.2,  1.3,10.0,10.0,10.0
  };

  bool* lsmask = new bool[8*8] {
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false
  };

  bool* truesinks = new bool[8*8] {
    false,false,false,false, false,false,false,false,
     true,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false,
    false,false,false,false, false,false,false,false
  };

  double* expected_rdirs_out = new double[8*8] {
       2,   1,   1,   6,    6,   3,   2,   3,
       5,   4,   4,   1,    3,   2,   1,   6,
       1,   7,   7,   4,    3,   2,   3,   2,
       4,   9,   8,   7,    3,   2,   6,   5,
       7,   9,   8,   7,    6,   3,   9,   8,
       6,   9,   8,   7,    9,   6,   6,   6,
       3,   2,   1,   2,    1,   1,   9,   9,
       6,   5,   4,   5,    4,   4,   6,   9
  };

  double* rdirs = new double[8*8];
  bool always_flow_to_sea_in = true;
  bool use_diagonal_nbrs_in = true;
  bool mark_pits_as_true_sinks_in = true;
  latlon_determine_river_directions(rdirs,
                                    orography,
                                    lsmask,
                                    truesinks,
                                    nlat,nlon,
                                    always_flow_to_sea_in,
                                    use_diagonal_nbrs_in,
                                    mark_pits_as_true_sinks_in);
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(rdirs[i],expected_rdirs_out[i]);
  }
  delete[] rdirs;
  delete[] expected_rdirs_out;
  delete[] truesinks;
  delete[] lsmask;
  delete[] orography;
};


} //namespace
