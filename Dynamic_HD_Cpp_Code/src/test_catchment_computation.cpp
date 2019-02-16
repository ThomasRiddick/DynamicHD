/*
 * test_catchment_computation.cpp
 *
 * Unit test for the catchment computation C++ code using Google's
 * Google Test Framework
 *
 *  Created on: Mar 11, 2018
 *      Author: thomasriddick using Google's recommended template code
 */

#include "compute_catchments.hpp"
#include "catchment_computation_algorithm.hpp"
#include "gtest/gtest.h"
#include "cell.hpp"
#include "context.hpp"
#include <string>

using namespace std;

namespace {

class CatchmentComputationTest : public ::testing::Test {
 protected:

  CatchmentComputationTest() {
  };

  virtual ~CatchmentComputationTest() {
  };

  // Common object can go here

};

TEST_F(CatchmentComputationTest, CatchmentComputationSingleCatchmentTest){
  int nlat = 8;
  int nlon = 8;
  int* catchment_numbers = new int[8*8] {0};
  double* rdirs = new double[8*8] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    0.0,7.0,8.0,8.0, 7.0,9.0,7.0,0.0,
    0.0,7.0,3.0,2.0, 1.0,9.0,7.0,0.0,
    0.0,1.0,4.0,3.0, 3.0,3.0,6.0,0.0,
    0.0,7.0,4.0,6.0, 6.0,6.0,9.0,0.0,
    0.0,1.0,1.0,2.0, 1.0,6.0,6.0,0.0,
    0.0,2.0,2.0,3.0, 1.0,2.0,3.0,0.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0
  };
  int* expected_catchments_number_out = new int[8*8] {
    207,0,0,0, 0,0,0,0,
    0,207,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0,
    0,0,0,0, 0,0,0,0,
  };
  auto alg = catchment_computation_algorithm_latlon();
  auto grid_params_in = new latlon_grid_params(nlat,nlon);
  landsea_cell* outflow_in = new landsea_cell(new latlon_coords(0,0));
  alg.setup_fields(catchment_numbers,
                   rdirs,grid_params_in);
  alg.test_compute_catchment(outflow_in,207);
  delete grid_params_in;
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(expected_catchments_number_out[i],catchment_numbers[i]);
  }
  delete[] expected_catchments_number_out;
  delete[] catchment_numbers;
  delete[] rdirs;
}

TEST_F(CatchmentComputationTest, CatchmentComputationGeneralTestOne) {
  int nlat = 8;
  int nlon = 8;
  int* catchment_numbers = new int[8*8] {0};
  double* rdirs = new double[8*8] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    0.0,7.0,8.0,8.0, 7.0,9.0,7.0,0.0,
    0.0,7.0,3.0,2.0, 1.0,9.0,7.0,0.0,
    0.0,1.0,4.0,3.0, 3.0,3.0,6.0,0.0,
    0.0,7.0,4.0,6.0, 6.0,6.0,9.0,0.0,
    0.0,1.0,1.0,2.0, 1.0,6.0,6.0,0.0,
    0.0,2.0,2.0,3.0, 1.0,2.0,3.0,0.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0
  };
  int* expected_catchments_number_out = new int[8*8] {
    1, 2, 3, 4,  5, 6, 7, 8,
    9, 1, 3, 4,  4, 7, 6,10,
   11, 9,14,14, 14, 6, 7,12,
   13,15,15,14, 14,14,14,14,
   15,13,13,14, 14,14,14,16,
   17,19,22,25, 25,18,18,18,
   19,22,23,25, 24,26,28,20,
   21,22,23,24, 25,26,27,28
  };
  string log_file_path = datadir + "/temp/loop_log_cpp_null.txt";
  latlon_compute_catchments(catchment_numbers,rdirs,
                            log_file_path,
                            nlat,nlon);
  int count = 0;
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(expected_catchments_number_out[i],catchment_numbers[i]);
    if(expected_catchments_number_out[i] != catchment_numbers[i]) count++;
  }
  delete[] expected_catchments_number_out;
  delete[] catchment_numbers;
  delete[] rdirs;
}

TEST_F(CatchmentComputationTest, CatchmentComputationGeneralWithComplexLoop) {
  int nlat = 8;
  int nlon = 8;
  int* catchment_numbers = new int[8*8] {0};
  double* rdirs = new double[8*8] {
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0,
    0.0,2.0,2.0,2.0, 2.0,2.0,2.0,0.0,
    0.0,2.0,2.0,2.0, 2.0,2.0,8.0,0.0,
    0.0,6.0,6.0,6.0, 2.0,4.0,4.0,0.0,
    0.0,8.0,6.0,8.0, 4.0,8.0,4.0,0.0,
    0.0,8.0,8.0,8.0, 8.0,8.0,4.0,0.0,
    0.0,8.0,8.0,8.0, 8.0,4.0,4.0,0.0,
    0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0
  };
  int* expected_catchments_number_out = new int[8*8] {
    1, 2, 3, 4,  5, 6, 7, 8,
    9,29,29,29, 29,29,30,10,
   11,29,29,29, 29,29,30,12,
   13,29,29,29, 29,29,29,14,
   15,29,29,29, 29,29,29,16,
   17,29,29,29, 29,29,29,18,
   19,29,29,29, 29,29,29,20,
   21,22,23,24, 25,26,27,28
  };
  string log_file_path = datadir + "/temp/loop_log_cpp_cloops.txt";
  latlon_compute_catchments(catchment_numbers,rdirs,
                            log_file_path,
                            nlat,nlon);
  int count = 0;
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(expected_catchments_number_out[i],catchment_numbers[i]);
    if(expected_catchments_number_out[i] != catchment_numbers[i]) count++;
  }
  ifstream logfile;
  string output_line_one;
  string output_line_two;
  string output_line_three;
  logfile.open(log_file_path);
  getline(logfile,output_line_one);
  logfile >> output_line_two;
  logfile >> output_line_three;
  EXPECT_TRUE(output_line_one == "Loops found in catchments:");
  int loop_catchment_out_one = stoi(output_line_two);
  int loop_catchment_out_two = stoi(output_line_three);
  logfile.close();
  EXPECT_EQ(loop_catchment_out_one,29);
  EXPECT_EQ(loop_catchment_out_two,30);
  delete[] expected_catchments_number_out;
  delete[] catchment_numbers;
  delete[] rdirs;
}

TEST_F(CatchmentComputationTest, CatchmentComputationLoopFindingTestOne) {
  int nlat = 3;
  int nlon = 3;
  int* catchment_numbers = new int[3*3] {0};
  double* rdirs = new double[3*3] {
    6,6,2,
    8,5,2,
    8,4,4
  };
  int* expected_catchments_number_out = new int[3*3] {
    2,2,2,
    2,1,2,
    2,2,2
  };
  string log_file_path = datadir + "/temp/loop_log_cpp_loop_test.txt";
  remove(log_file_path.c_str());
  latlon_compute_catchments(catchment_numbers,rdirs,
                            log_file_path,
                            nlat,nlon);
  int count = 0;
  for (auto i =0; i < nlat*nlon; i++){
    EXPECT_EQ(expected_catchments_number_out[i],catchment_numbers[i]);
    if(expected_catchments_number_out[i] != catchment_numbers[i]) count++;
  }
  ifstream logfile;
  string output_line_one;
  string output_line_two;
  logfile.open(log_file_path);
  getline(logfile,output_line_one);
  logfile >> output_line_two;
  int loop_catchment_out = stoi(output_line_two);
  logfile.close();
  EXPECT_EQ(loop_catchment_out,2);
  delete[] expected_catchments_number_out;
  delete[] catchment_numbers;
  delete[] rdirs;
}

TEST_F(CatchmentComputationTest, CatchmentComputationICONTestOne) {
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
  int* next_cell_index_in = new int[80*3] {
    //1
    7,
    //2
    10,
    //3
    4,
    //4
    -1,
    //5
    4,
    //6
    20,
    //7
    6,
    //8
    6,
    //9
    8,
    //10
    26,
    //11
    12,
    //12
    13,
    //13
    3,
    //14
    13,
    //15
    16,
    //16
    19,
    //17
    16,
    //18
    20,
    //19
    20,
    //20
    -5,
    //21
    39,
    //22
    21,
    //23
    6,
    //24
    25,
    //25
    26,
    //26
    -5,
    //27
    26,
    //28
    27,
    //29
    28,
    //30
    51,
    //31
    32,
    //32
    51,
    //33
    15,
    //34
    16,
    //35
    17,
    //36
    54,
    //37
    36,
    //38
    37,
    //39
    20,
    //40
    21,
    //41
    60,
    //42
    43,
    //43
    26,
    //44
    45,
    //45
    26,
    //46
    26,
    //47
    65,
    //48
    65,
    //49
    50,
    //50
    51,
    //51
    52,
    //52
    53,
    //53
    54,
    //54
    -1,
    //55
    34,
    //56
    55,
    //57
    36,
    //58
    39,
    //59
    58,
    //60
    59,
    //61
    62,
    //62
    63,
    //63
    64,
    //64
    65,
    //65
    -5,
    //66
    65,
    //67
    52,
    //68
    66,
    //69
    68,
    //70
    54,
    //71
    70,
    //72
    54,
    //73
    58,
    //74
    75,
    //75
    42,
    //76
    62,
    //77
    78,
    //78
    79,
    //79
    80,
    //80
    73
  };
  int* expected_catchment_number_out= new int[80*3] {
    //1
    2,
    //2
    3,
    //3
    1,
    //4
    1,
    //5
    1,
    //6
    2,
    //7
    2,
    //8
    2,
    //9
    2,
    //10
    3,
    //11
    1,
    //12
    1,
    //13
    1,
    //14
    1,
    //15
    2,
    //16
    2,
    //17
    2,
    //18
    2,
    //19
    2,
    //20
    2,
    //21
    2,
    //22
    2,
    //23
    2,
    //24
    3,
    //25
    3,
    //26
    3,
    //27
    3,
    //28
    3,
    //29
    3,
    //30
    4,
    //31
    4,
    //32
    4,
    //33
    2,
    //34
    2,
    //35
    2,
    //36
    4,
    //37
    4,
    //38
    4,
    //39
    2,
    //40
    2,
    //41
    2,
    //42
    3,
    //43
    3,
    //44
    3,
    //45
    3,
    //46
    3,
    //47
    5,
    //48
    5,
    //49
    4,
    //50
    4,
    //51
    4,
    //52
    4,
    //53
    4,
    //54
    4,
    //55
    2,
    //56
    2,
    //57
    4,
    //58
    2,
    //59
    2,
    //60
    2,
    //61
    5,
    //62
    5,
    //63
    5,
    //64
    5,
    //65
    5,
    //66
    5,
    //67
    4,
    //68
    5,
    //69
    5,
    //70
    4,
    //71
    4,
    //72
    4,
    //73
    2,
    //74
    3,
    //75
    3,
    //76
    5,
    //77
    2,
    //78
    2,
    //79
    2,
    //80
    2
  };
  int* secondary_neighboring_cell_indices_in = new int[80*9];
  int* catchment_numbers_in = new int[80];
  icon_single_index_grid_params* grid_params_in =
      new icon_single_index_grid_params(80,cell_neighbors,
                                        true,secondary_neighboring_cell_indices_in);
  auto alg = catchment_computation_algorithm_icon_single_index();
  grid_params_in->icon_single_index_grid_calculate_secondary_neighbors();
  alg.setup_fields(catchment_numbers_in,
                   next_cell_index_in,
                   grid_params_in);
  alg.compute_catchments();
  EXPECT_TRUE(field<int>(catchment_numbers_in,grid_params_in) ==
              field<int>(expected_catchment_number_out,grid_params_in));
  delete[] secondary_neighboring_cell_indices_in;
  delete[] catchment_numbers_in; delete[] cell_neighbors;
  delete[] next_cell_index_in; delete grid_params_in;
  delete[] expected_catchment_number_out;
}

TEST_F(CatchmentComputationTest, CatchmentComputationICONTestTwo) {
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
  int* next_cell_index_in = new int[80*3] {
    //1
    7,
    //2
    10,
    //3
    4,
    //4
    -1,
    //5
    4,
    //6
    20,
    //7
    6,
    //8
    6,
    //9
    8,
    //10
    26,
    //11
    12,
    //12
    13,
    //13
    3,
    //14
    13,
    //15
    16,
    //16
    19,
    //17
    16,
    //18
    20,
    //19
    20,
    //20
    -5,
    //21
    39,
    //22
    21,
    //23
    6,
    //24
    25,
    //25
    26,
    //26
    -5,
    //27
    26,
    //28
    27,
    //29
    28,
    //30
    51,
    //31
    32,
    //32
    51,
    //33
    15,
    //34
    16,
    //35
    17,
    //36
    54,
    //37
    36,
    //38
    37,
    //39
    20,
    //40
    21,
    //41
    60,
    //42
    43,
    //43
    26,
    //44
    45,
    //45
    26,
    //46
    26,
    //47
    65,
    //48
    65,
    //49
    50,
    //50
    51,
    //51
    52,
    //52
    53,
    //53
    54,
    //54
    -1,
    //55
    34,
    //56
    55,
    //57
    36,
    //58
    39,
    //59
    58,
    //60
    59,
    //61
    63,
    //62
    76,
    //63
    62,
    //64
    65,
    //65
    -5,
    //66
    65,
    //67
    52,
    //68
    66,
    //69
    68,
    //70
    54,
    //71
    70,
    //72
    54,
    //73
    58,
    //74
    75,
    //75
    42,
    //76
    61,
    //77
    78,
    //78
    79,
    //79
    80,
    //80
    73
  };
  int* expected_catchment_number_out= new int[80*3] {
    //1
    2,
    //2
    3,
    //3
    1,
    //4
    1,
    //5
    1,
    //6
    2,
    //7
    2,
    //8
    2,
    //9
    2,
    //10
    3,
    //11
    1,
    //12
    1,
    //13
    1,
    //14
    1,
    //15
    2,
    //16
    2,
    //17
    2,
    //18
    2,
    //19
    2,
    //20
    2,
    //21
    2,
    //22
    2,
    //23
    2,
    //24
    3,
    //25
    3,
    //26
    3,
    //27
    3,
    //28
    3,
    //29
    3,
    //30
    4,
    //31
    4,
    //32
    4,
    //33
    2,
    //34
    2,
    //35
    2,
    //36
    4,
    //37
    4,
    //38
    4,
    //39
    2,
    //40
    2,
    //41
    2,
    //42
    3,
    //43
    3,
    //44
    3,
    //45
    3,
    //46
    3,
    //47
    5,
    //48
    5,
    //49
    4,
    //50
    4,
    //51
    4,
    //52
    4,
    //53
    4,
    //54
    4,
    //55
    2,
    //56
    2,
    //57
    4,
    //58
    2,
    //59
    2,
    //60
    2,
    //61
    0,
    //62
    0,
    //63
    0,
    //64
    5,
    //65
    5,
    //66
    5,
    //67
    4,
    //68
    5,
    //69
    5,
    //70
    4,
    //71
    4,
    //72
    4,
    //73
    2,
    //74
    3,
    //75
    3,
    //76
    0,
    //77
    2,
    //78
    2,
    //79
    2,
    //80
    2
  };
  int* expected_catchment_numbers_out_two = new int[80*3] {
    //1
    2,
    //2
    3,
    //3
    1,
    //4
    1,
    //5
    1,
    //6
    2,
    //7
    2,
    //8
    2,
    //9
    2,
    //10
    3,
    //11
    1,
    //12
    1,
    //13
    1,
    //14
    1,
    //15
    2,
    //16
    2,
    //17
    2,
    //18
    2,
    //19
    2,
    //20
    2,
    //21
    2,
    //22
    2,
    //23
    2,
    //24
    3,
    //25
    3,
    //26
    3,
    //27
    3,
    //28
    3,
    //29
    3,
    //30
    4,
    //31
    4,
    //32
    4,
    //33
    2,
    //34
    2,
    //35
    2,
    //36
    4,
    //37
    4,
    //38
    4,
    //39
    2,
    //40
    2,
    //41
    2,
    //42
    3,
    //43
    3,
    //44
    3,
    //45
    3,
    //46
    3,
    //47
    5,
    //48
    5,
    //49
    4,
    //50
    4,
    //51
    4,
    //52
    4,
    //53
    4,
    //54
    4,
    //55
    2,
    //56
    2,
    //57
    4,
    //58
    2,
    //59
    2,
    //60
    2,
    //61
    6,
    //62
    6,
    //63
    6,
    //64
    5,
    //65
    5,
    //66
    5,
    //67
    4,
    //68
    5,
    //69
    5,
    //70
    4,
    //71
    4,
    //72
    4,
    //73
    2,
    //74
    3,
    //75
    3,
    //76
    6,
    //77
    2,
    //78
    2,
    //79
    2,
    //80
    2
  };
  int* secondary_neighboring_cell_indices_in = new int[80*9];
  int* catchment_numbers_in = new int[80];
  icon_single_index_grid_params* grid_params_in =
      new icon_single_index_grid_params(80,cell_neighbors,
                                        true,secondary_neighboring_cell_indices_in);
  auto alg = catchment_computation_algorithm_icon_single_index();
  grid_params_in->icon_single_index_grid_calculate_secondary_neighbors();
  alg.setup_fields(catchment_numbers_in,
                   next_cell_index_in,
                   grid_params_in);
  alg.compute_catchments();
  EXPECT_TRUE(field<int>(catchment_numbers_in,grid_params_in) ==
              field<int>(expected_catchment_number_out,grid_params_in));
  vector<int>* loop_numbers = alg.identify_loops();
  EXPECT_TRUE(loop_numbers->size() == 1);
  EXPECT_TRUE((*loop_numbers)[0] == 6);
  EXPECT_TRUE(field<int>(catchment_numbers_in,grid_params_in) ==
              field<int>(expected_catchment_numbers_out_two,grid_params_in));
  delete[] secondary_neighboring_cell_indices_in;
  delete[] catchment_numbers_in; delete[] cell_neighbors;
  delete[] next_cell_index_in; delete grid_params_in;
  delete[] expected_catchment_number_out;
  delete[] expected_catchment_numbers_out_two;
  delete loop_numbers;
}

}  // namespace
