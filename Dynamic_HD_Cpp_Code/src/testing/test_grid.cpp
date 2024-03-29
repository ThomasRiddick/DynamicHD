/*
 * test_grid.cpp
 *
 * Unit test for the fill sinks C++ code using Google's
 * Google Test Framework
 *
 *  Created on: Sept 28, 2018
 *      Author: thomasriddick using Google's recommended template
 */

#define GTEST_HAS_TR1_TUPLE 0

#include "gtest/gtest.h"
#include "base/grid.hpp"
#include <iostream>
using namespace std;

namespace unittests {

class GridParamsTest : public ::testing::Test {};

TEST_F(GridParamsTest,TestSecondaryNeighborGeneration){
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
  int* secondary_neighbors_expected_out = new int[80*9]{
      4, 19, -1, 6, 8, 20, 10, 3, 9,
      5, 7, -1, 9, 11, 8, 13, 4, 12,
      1, 10, -1, 12, 14, 11, 16, 5, 15,
      2, 13, -1, 15, 17, 14, 19, 1, 18,
      3, 16, -1, 18, 20, 17, 7, 2, 6,
//6
      19, 39, 5, 40, 22, -1, 1, 8, 23,
      5, 2, 19, 20, 21, 10, 23, 9, 22,
      1, 6, 2, 22, 24, 21, 25, 10, -1,
      7, 23, 1, 24, 26, -1, 2, 11, 27,
      1, 3, 7, 8, 25, 13, 27, 12, 26,
//11
      2, 9, 3, 26, 28, 25, 29, 13, -1,
      10, 27, 2, 28, 30, -1, 3, 14, 31,
      2, 4, 10, 11, 29, 16, 31, 15, 30,
      3, 12, 4, 30, 32, 29, 33, 16, -1,
      13, 31, 3, 32, 34, -1, 4, 17, 35,
//16
      3, 5, 13, 14, 33, 19, 35, 18, 34,
      4, 15, 5, 34, 36, 33, 37, 19, -1,
      16, 35, 4, 36, 38, -1, 5, 20, 39,
      4, 1, 16, 17, 37, 7, 39, 6, 38,
      5, 18, 1, 38, 40, 37, 21, 7, -1,
//21
      20, 7, -1, 39, 59, 8, 41, 23, 60,
      6, 40, 7, 60, 42, 59, 8, 24, 43,
      7, 9, 6, 21, 41, -1, 43, 25, 42,
      8, 22, -1, 42, 44, 41, 26, 9, 45,
      23, 43, -1, 45, 27, 44, 8, 10, 11,
//26
      24, 9, 43, 44, 46, 10, 11, 28, 47,
      10, 12, 9, 25, 45, -1, 47, 29, 46,
      11, 26, -1, 46, 48, 45, 12, 30, 49,
      11, 13, -1, 27, 47, 14, 49, 31, 48,
      12, 28, 13, 48, 50, 47, 14, 32, 51,
//31
      13, 15, 12, 29, 49, -1, 51, 33, 50,
      14, 30, -1, 50, 52, 49, 15, 34, 53,
      14, 16, -1, 31, 51, 17, 53, 35, 52,
      15, 32, 16, 52, 54, 51, 17, 36, 55,
      16, 18, 15, 33, 53, -1, 55, 37, 54,
//36
      17, 34, -1, 54, 56, 53, 18, 38, 57,
      17, 19, -1, 35, 55, 20, 57, 39, 56,
      18, 36, 19, 56, 58, 55, 20, 40, 59,
      19, 6, 18, 37, 57, -1, 59, 21, 58,
      20, 38, -1, 58, 60, 57, 6, 22, 41,
//41
      21, 23, 40, 59, 75, 24, 61, 43, -1,
      22, 60, 23, 75, 62, -1, 24, 44, 63,
      23, 25, 22, 41, 61, 26, 63, 45, 62,
      24, 42, 25, 62, 64, 61, 26, 46, -1,
      25, 27, 24, 43, 63, 28, 64, 47, -1,
//46
      26, 44, 27, 63, 65, -1, 28, 48, 66,
      27, 29, 26, 45, 64, 30, 66, 49, 65,
      28, 46, 29, 65, 67, 64, 30, 50, -1,
      29, 31, 28, 47, 66, 32, 67, 51, -1,
      30, 48, 31, 66, 68, -1, 32, 52, 69,
//51
      31, 33, 30, 49, 67, 34, 69, 53, 68,
      32, 50, 33, 68, 70, 67, 34, 54, -1,
      33, 35, 32, 51, 69, 36, 70, 55, -1,
      34, 52, 35, 69, 71, -1, 36, 56, 72,
      35, 37, 34, 53, 70, 38, 72, 57, 71,
//56
      36, 54, 37, 71, 73, 70, 38, 58, -1,
      37, 39, 36, 55, 72, 40, 73, 59, -1,
      38, 56, 39, 72, 74, -1, 40, 60, 75,
      39, 21, 38, 57, 73, 22, 75, 41, 74,
      40, 58, 21, 74, 61, 73, 22, 42, -1,
//61
      41, 43, -1, 60, 74, 44, 76, 63, 80,
      42, 75, 43, 80, 77, 74, 44, 64, 65,
      43, 45, 42, 61, 76, -1, 46, 65, 77,
      45, 47, -1, 44, 62, 48, 77, 66, 76,
      46, 63, 47, 76, 78, 62, 48, 67, 68,
//66
      47, 49, 46, 64, 77, -1, 50, 68, 78,
      49, 51, -1, 48, 65, 52, 78, 69, 77,
      50, 66, 51, 77, 79, 65, 52, 70, 71,
      51, 53, 50, 67, 78, -1, 54, 71, 79,
      53, 55, -1, 52, 68, 56, 79, 72, 78,
//71
      54, 69, 55, 78, 80, 68, 56, 73, 74,
      55, 57, 54, 70, 79, -1, 58, 74, 80,
      57, 59, -1, 56, 71, 60, 80, 75, 79,
      58, 72, 59, 79, 76, 71, 60, 61, 62,
      59, 41, 58, 73, 80, -1, 42, 62, 76,
//76
      61, 63, 75, 74, 79, 64, 65, 78, -1,
      64, 66, 63, 62, 80, 67, 68, 79, -1,
      67, 69, 66, 65, 76, 70, 71, 80, -1,
      70, 72, 69, 68, 77, 73, 74, 76, -1,
      73, 75, 72, 71, 78, 61, 62, 77, -1,
  };
  int* secondary_neighbors = nullptr;
  auto grid_pars_obj = icon_single_index_grid_params(80,cell_neighbors,true,secondary_neighbors);
  grid_pars_obj.icon_single_index_grid_calculate_secondary_neighbors();
  secondary_neighbors = grid_pars_obj.get_secondary_neighboring_cell_indices();
  for (auto i =0; i < 80*9; i++){
    EXPECT_EQ(secondary_neighbors[i],secondary_neighbors_expected_out[i]);
  }
  delete[] cell_neighbors;
  delete[] secondary_neighbors_expected_out;
}

} // namespacek
