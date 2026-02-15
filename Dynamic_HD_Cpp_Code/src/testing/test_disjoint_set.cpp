/*
 * test_disjoint_set.cpp
 *
 * Unit test for the basin evaluation C++ code using Google's
 * Google Test Framework
 *
 *  Created on: Sept 28, 2018
 *      Author: thomasriddick using Google's recommended template
 */

#define GTEST_HAS_TR1_TUPLE 0

#include "gtest/gtest.h"
#include "base/disjoint_set.hpp"
#include <iostream>
using namespace std;

namespace unittests {

class DisjointSetTest : public ::testing::Test {};

TEST_F(DisjointSetTest,TestDisjointSets){
  auto dsets = disjoint_set_forest();
  dsets.add_set(1);
  dsets.add_set(2);
  dsets.add_set(5);
  dsets.add_set(6);
  dsets.add_set(9);
  dsets.add_set(10);
  dsets.add_set(13);
  dsets.add_set(14);
  dsets.add_set(17);
  dsets.add_set(18);
  dsets.add_set(19);
  dsets.add_set(20);
  dsets.make_new_link(2,5);
  dsets.make_new_link(5,9);
  dsets.make_new_link(5,10);
  dsets.make_new_link(13,1);
  dsets.make_new_link(1,14);
  dsets.make_new_link(14,17);
  dsets.make_new_link(14,18);
  dsets.make_new_link(13,19);
  dsets.make_new_link(20,18);
  EXPECT_TRUE(dsets.check_subset_has_elements(10,vector<int>{2, 5, 9, 10}));
  EXPECT_TRUE(dsets.check_subset_has_elements(6,vector<int>{6}));
  EXPECT_TRUE(dsets.check_subset_has_elements(18,vector<int>{20, 13, 1, 14, 17, 18, 19}));
  EXPECT_TRUE(dsets.make_new_link(2,18));
  EXPECT_TRUE(dsets.check_subset_has_elements(6,vector<int>{6}));
  EXPECT_TRUE(dsets.check_subset_has_elements(2,vector<int>{2, 5, 9, 10, 20, 13, 1, 14, 17, 18, 19}));
  EXPECT_FALSE(dsets.make_new_link(17,10));
}

}
