/*
 * test_post_process_basins.cpp
 *
 * Unit test for lake post processing tool C++ code using Google's
 * Google Test Framework
 *
 *  Created on: Sep 3, 2019
 *      Author: thomasriddick using Google's recommended template code
 */

#include
using namespace std;

class PostProcessBasinsTest : public ::testing::Test {
 protected:

  PostProcessBasinsTest() {
  }

  virtual ~PostProcessBasinsTest() {
    // Don't include exceptions here!
  }

// Common objects can go here

};

TEST_F(PostProcessBasinsTest, PostProcessBasinsTestOne) {
   int* flood_next_cell_lat_index_expected_out = new int[20*20]{
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,   3,
  2,  3,   5,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,   2,   2,   4,   5,  -1,  -1, -1,   1,
 -1,  4,   2,  -1,  -1,  8,  2,  -1,  -1,  -1, -1,  2,   5,   3,   9,   2,  -1,  -1, -1,  -1,
 -1,  3,   4,  -1,  -1,  6,  4,   6,   3,  -1, -1, -1,   7,   3,   4,   3,   6,  -1, -1,  -1,
 -1,  6,   5,   3,   6,  7, -1,   4,   4,  -1,  5,  7,   6,   6,   4,   8,   4,   3, -1,  -1,
  7,  6,   6,  -1,   5,  5,  6,   5,   5,  -1,  5,  5,   4,   8,   6,   6,   5,   5, -1,  -1,
 -1,  6,   7,  -1,  -1,  6, -1,   4,  -1,  -1, -1,  5,   6,   6,   8,   7,   5,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1,  8,   7,   7,   8,   6,   7,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,   8,   9,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  14,  14,  13, 16,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  13,  14,  13, -1,  -1,
 -1, -1,  -1,  17,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1 };
  int* flood_next_cell_lon_index_expected_out = new int[20*20]{
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,   1,
 19,  5,   2,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  15,  12,  16,   3,  -1,  -1, -1,  19,
 -1,  2,   2,  -1,  -1, 18,  1,  -1,  -1,  -1, -1, 13,  15,  12,  13,  14,  -1,  -1, -1,  -1,
 -1,  2,   1,  -1,  -1,  8,  5,  10,   6,  -1, -1, -1,  12,  13,  13,  14,  17,  -1, -1,  -1,
 -1,  2,   1,   5,   7,  5, -1,   6,   8,  -1, 14, 14,  10,  12,  14,  15,  15,  11, -1,  -1,
  2,  0,   1,  -1,   4,  5,  4,   7,   8,  -1, 10, 11,  12,  12,  13,  14,  16,  17, -1,  -1,
 -1,  4,   1,  -1,  -1,  6, -1,   7,  -1,  -1, -1, 12,  11,  15,  14,  13,   9,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, 16,  11,  15,  13,  16,  16,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  11,  12,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  15,  16,  18, 15,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  16,  17,  17, -1,  -1,
 -1, -1,  -1,   3,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1, -1,  -1 };
  int* connect_next_cell_lat_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,   3,       7,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1,  3,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* connect_next_cell_lon_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,   6,       7,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, 15,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1  };
  int* flood_redirect_lat_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,   1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,   1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1,  7, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1,  7, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1,  6, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1,  5,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1,  3, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1,  3, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* flood_redirect_lon_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,   0, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,   3, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, 14, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, 14, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  0, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1,  5, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, 13,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1,  3, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1,  0, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* connect_redirect_lat_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* connect_redirect_lon_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  bool* flood_local_redirect_expected_out = new bool[20*20]{
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false, false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,true,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false, true,false,false, false,false,
    false,false,false,false,
   false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false, false,
    true,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false, false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false};
  bool* connect_local_redirect_expected_out = new bool[20*20]{
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false, false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false, false,false,
    false,false,false,false,
   false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false, false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false, false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false};
  merge_types* merge_points_expected_out = new merge_types[20*20]{
    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, connection_merge_not_set_flood_merge_as_primary,no_merge,no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge,no_merge,no_merge, connection_merge_not_set_flood_merge_as_secondary,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,connection_merge_not_set_flood_merge_as_secondary,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,

    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,connection_merge_not_set_flood_merge_as_secondary,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
    no_merge,connection_merge_not_set_flood_merge_as_secondary,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,connection_merge_not_set_flood_merge_as_primary,no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,connection_merge_not_set_flood_merge_as_primary, no_merge,no_merge,no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,

    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,connection_merge_not_set_flood_merge_as_secondary,no_merge,
    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,

    no_merge,no_merge,no_merge,connection_merge_not_set_flood_merge_as_secondary,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge,
      no_merge,no_merge,no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,no_merge,no_merge };

  int* flood_force_merge_lat_index_expected_out = new int[20*20] {

    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,   6, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1,  6, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1,  7,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* flood_force_merge_lon_index_expected_out = new int[20*20] {
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,   2, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1,  8, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, 12,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* connect_force_merge_lat_index_expected_out = new int[20*20] {
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* connect_force_merge_lon_index_expected_out = new int[20*20] {
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };

}
