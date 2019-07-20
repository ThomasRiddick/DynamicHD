/*
 * test_evaluate_basins.cpp
 *
 * Unit test for the basin evaluation C++ code using Google's
 * Google Test Framework
 *
 *  Created on: Mar 11, 2018
 *      Author: thomasriddick using Google's recommended template code
 */

#include "basin_evaluation_algorithm.hpp"
#include "gtest/gtest.h"
#include "enums.hpp"
#include <cmath>

namespace basin_evaluation_unittests {

class BasinEvaluationTest : public ::testing::Test {
 protected:

	BasinEvaluationTest() {
  }
  virtual ~BasinEvaluationTest() {
    // Don't include exceptions here!
  }

// Common objects can go here

};

TEST_F(BasinEvaluationTest, TestAddingMinimaToQueueOne) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  double* raw_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  double* corrected_orography_in = new double[9*9] {0.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  bool* minima_in = new bool[9*9] {false,false,false, false,false,false,  true,false,false,
                                   false,false,false, false,false,false, false,false,false,
                                   false,false,false, false,false,false, false,false,false,
//
                                   false,false,false, false,false,false, false,false,false,
                                   false,false,false, false, true,false, false,false,false,
                                   false,false,false, false,false,false, false,false,false,
//
                                   false,false,false, false,false,false, false,false,false,
                                   false,false,false, false,false,false, false, true,false,
                                   false,false,false, false,false,false, false,false,false};
  height_types* expected_cells_in_queue = new height_types[9*9]
    {null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, flood_height,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,flood_height,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,flood_height,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype };
  bool* landsea_in = new bool[9*9];
  fill_n(landsea_in,9*9,false);
  landsea_in[0] = true;
  bool* true_sinks_in = new bool[9*9];
  fill_n(true_sinks_in,9*9,false);
  int* next_cell_lat_index_in = new int[9*9];
  int* next_cell_lon_index_in = new int[9*9];
  short* rdirs_in = new short[9*9];
  int* catchment_nums_in = new int[9*9];
  queue<cell*> q;
  auto alg4 = new sink_filling_algorithm_4_latlon();
  alg4->setup_flags(false,true,false,false);
  alg4->setup_fields(corrected_orography_in,
                     landsea_in,
                     true_sinks_in,
                     next_cell_lat_index_in,
                     next_cell_lon_index_in,
                     grid_params_in,
                     rdirs_in,
                     catchment_nums_in);
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_add_minima_to_queue(raw_orography_in,
                                          corrected_orography_in,
                                          minima_in,
                                          alg4,
                                          grid_params_in,
                                          coarse_grid_params_in);
  auto count = 0;
  field<height_types> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(null_htype);
  while (! q.empty()) {
    auto cell = static_cast<basin_cell*>(q.front());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = cell->get_height_type();
    count++;
    q.pop();
    delete cell;
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
  delete[] raw_orography_in; delete[] corrected_orography_in; delete[] minima_in;
  delete[] expected_cells_in_queue;
  delete grid_params_in; delete coarse_grid_params_in;
  delete alg4;
  delete[] landsea_in; delete[] true_sinks_in;
  delete[] next_cell_lat_index_in; delete[] next_cell_lon_index_in;
  delete[] rdirs_in; delete[] catchment_nums_in;
}

TEST_F(BasinEvaluationTest, TestAddingMinimaToQueueTwo) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  double* raw_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  double* corrected_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 1.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,3.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  bool* minima_in = new bool[9*9] {false,false,false, false,false,false,  true,false,false,
                                   false,false,false, false,false,false, false,false,false,
                                   false,false,false, false,false,false, false,false,false,
//
                                   false,false,false, false,false,false, false,false,false,
                                   false,false,false, false, true,false, false,false,false,
                                   false,false,false, false,false,false, false,false,false,
//
                                   false,false,false, false,false,false, false,false,false,
                                   false,false,false, false,false,false, false, true,false,
                                   false,false,false, false,false,false, false,false,false};
  height_types* expected_cells_in_queue = new height_types[9*9]
    {null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, connection_height,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,flood_height,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,flood_height,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype };
  bool* landsea_in = new bool[9*9];
  fill_n(landsea_in,9*9,false);
  landsea_in[0] = true;
  bool* true_sinks_in = new bool[9*9];
  fill_n(true_sinks_in,9*9,false);
  int* next_cell_lat_index_in = new int[9*9];
  int* next_cell_lon_index_in = new int[9*9];
  short* rdirs_in = new short[9*9];
  int* catchment_nums_in = new int[9*9];
  queue<cell*> q;
  auto alg4 = new sink_filling_algorithm_4_latlon();
  alg4->setup_flags(false,true,false,false);
  alg4->setup_fields(corrected_orography_in,
                     landsea_in,
                     true_sinks_in,
                     next_cell_lat_index_in,
                     next_cell_lon_index_in,
                     grid_params_in,
                     rdirs_in,
                     catchment_nums_in);
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_add_minima_to_queue(raw_orography_in,
                                          corrected_orography_in,
                                          minima_in,
                                          alg4,
                                          grid_params_in,
                                          coarse_grid_params_in);
  auto count = 0;
  field<height_types> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(null_htype);
  while (! q.empty()) {
    auto cell = static_cast<basin_cell*>(q.front());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = cell->get_height_type();
    count++;
    q.pop();
    delete cell;
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
  delete[] raw_orography_in; delete[] corrected_orography_in; delete[] minima_in;
  delete[] expected_cells_in_queue;
  delete grid_params_in; delete coarse_grid_params_in;
  delete alg4;
  delete[] landsea_in; delete[] true_sinks_in;
  delete[] next_cell_lat_index_in; delete[] next_cell_lon_index_in;
  delete[] rdirs_in; delete[] catchment_nums_in;
}

TEST_F(BasinEvaluationTest, TestAddingMinimaToQueueThree) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  double* raw_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  double* corrected_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 1.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,3.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  bool* minima_in = new bool[9*9] {false,false,false, false,false,false,  true,false,false,
                                   false,false,false, false,false,false, false,false,false,
                                   false,false,false, false,false,false, false,false,false,
//
                                   false,false,false, false,false,false, false,false,false,
                                   false,false,false, false, true,false, false,false,false,
                                   false,false,false, false,false,false, false,false,false,
//
                                    true,false,false, false,false,false, false,false,false,
                                   false, true,false, false,false,false, false, true,false,
                                   false,false,false, false,false,false, false,false,false};
  height_types* expected_cells_in_queue = new height_types[9*9]
    {null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, connection_height,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,flood_height,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
//
     flood_height,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,flood_height,null_htype, null_htype,null_htype,null_htype, null_htype,flood_height,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype };
  bool* landsea_in = new bool[9*9];
  fill_n(landsea_in,9*9,false);
  landsea_in[0] = true;
  bool* true_sinks_in = new bool[9*9];
  fill_n(true_sinks_in,9*9,false);
  int* next_cell_lat_index_in = new int[9*9];
  int* next_cell_lon_index_in = new int[9*9];
  short* rdirs_in = new short[9*9];
  int* catchment_nums_in = new int[9*9];
  queue<cell*> q;
  auto alg4 = new sink_filling_algorithm_4_latlon();
  alg4->setup_flags(false,true,false,false);
  alg4->setup_fields(corrected_orography_in,
                     landsea_in,
                     true_sinks_in,
                     next_cell_lat_index_in,
                     next_cell_lon_index_in,
                     grid_params_in,
                     rdirs_in,
                     catchment_nums_in);
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_add_minima_to_queue(raw_orography_in,
                                          corrected_orography_in,
                                          minima_in,
                                          alg4,
                                          grid_params_in,
                                          coarse_grid_params_in);
  auto count = 0;
  field<height_types> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(null_htype);
  while (! q.empty()) {
    auto cell = static_cast<basin_cell*>(q.front());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = cell->get_height_type();
    count++;
    q.pop();
    delete cell;
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
  delete[] raw_orography_in; delete[] corrected_orography_in; delete[] minima_in;
  delete[] expected_cells_in_queue;
  delete grid_params_in; delete coarse_grid_params_in;
  delete alg4;
  delete[] landsea_in; delete[] true_sinks_in;
  delete[] next_cell_lat_index_in; delete[] next_cell_lon_index_in;
  delete[] rdirs_in; delete[] catchment_nums_in;
}

TEST_F(BasinEvaluationTest, TestAddingMinimaToQueueFour) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  double* raw_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  double* corrected_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  bool* minima_in = new bool[9*9] { true,false,false, false,false,false, false,false,false,
                                   false,false,false, false,false,false, false,false,false,
                                   false,false,false, false,false,false, false, true,false,
//
                                   false,false,false, false,false,false, false,false,false,
                                   false,false,false, false, true,false, false,false,false,
                                   false,false,false, false,false,false, false,false,false,
//
                                   false,false,false, false,false,false, false,false,false,
                                    true,false,false, false,false,false, false,false,false,
                                   false,false,false, false,false,false,  true,false,false};
  height_types* expected_cells_in_queue = new height_types[9*9]
    {flood_height,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,flood_height,null_htype,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,flood_height,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     flood_height,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, flood_height,null_htype,null_htype };
  bool* landsea_in = new bool[9*9];
  fill_n(landsea_in,9*9,false);
  landsea_in[9*9-1] = true;
  bool* true_sinks_in = new bool[9*9];
  fill_n(true_sinks_in,9*9,false);
  int* next_cell_lat_index_in = new int[9*9];
  int* next_cell_lon_index_in = new int[9*9];
  short* rdirs_in = new short[9*9];
  int* catchment_nums_in = new int[9*9];
  queue<cell*> q;
  auto alg4 = new sink_filling_algorithm_4_latlon();
  alg4->setup_flags(false,true,false,false);
  alg4->setup_fields(corrected_orography_in,
                     landsea_in,
                     true_sinks_in,
                     next_cell_lat_index_in,
                     next_cell_lon_index_in,
                     grid_params_in,
                     rdirs_in,
                     catchment_nums_in);
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_add_minima_to_queue(raw_orography_in,
                                          corrected_orography_in,
                                          minima_in,
                                          alg4,
                                          grid_params_in,
                                          coarse_grid_params_in);
  auto count = 0;
  field<height_types> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(null_htype);
  while (! q.empty()) {
    auto cell = static_cast<basin_cell*>(q.front());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = cell->get_height_type();
    count++;
    q.pop();
    delete cell;
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete alg4;
  delete[] expected_cells_in_queue;
  delete[] raw_orography_in; delete[] corrected_orography_in; delete[] minima_in;
  delete[] landsea_in; delete[] true_sinks_in;
  delete[] next_cell_lat_index_in; delete[] next_cell_lon_index_in;
  delete[] rdirs_in; delete[] catchment_nums_in;
}

TEST_F(BasinEvaluationTest, TestAddingMinimaToQueueFive) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  double* raw_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  double* corrected_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  bool* minima_in = new bool[9*9] {false,false,false, false,false,false, false,false,false,
                                   false,false,true, false,false,false, false,false,false,
                                   false,false,false, false,false,false, false,false, true,
//
                                   false,false,false, false,false,false, false,false,false,
                                   false,false,false, false, true,false, false,false,false,
                                   false,false,false, false,false,false, false,false,false,
//
                                   false,false, true, false,false,false, false, true,false,
                                   false,false,false, false,false,false, false,false,false,
                                   false,false,false, false,false,false, false,false,false};
  height_types* expected_cells_in_queue = new height_types[9*9]
    {null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,flood_height, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,flood_height,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,flood_height,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
//
     null_htype,null_htype,flood_height, null_htype,null_htype,null_htype, null_htype,flood_height,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype };
  bool* landsea_in = new bool[9*9];
  fill_n(landsea_in,9*9,false);
  landsea_in[0] = true;
  bool* true_sinks_in = new bool[9*9];
  fill_n(true_sinks_in,9*9,false);
  int* next_cell_lat_index_in = new int[9*9];
  int* next_cell_lon_index_in = new int[9*9];
  short* rdirs_in = new short[9*9];
  int* catchment_nums_in = new int[9*9];
  queue<cell*> q;
  auto alg4 = new sink_filling_algorithm_4_latlon();
  alg4->setup_flags(false,true,false,false);
  alg4->setup_fields(corrected_orography_in,
                     landsea_in,
                     true_sinks_in,
                     next_cell_lat_index_in,
                     next_cell_lon_index_in,
                     grid_params_in,
                     rdirs_in,
                     catchment_nums_in);
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_add_minima_to_queue(raw_orography_in,
                                          corrected_orography_in,
                                          minima_in,
                                          alg4,
                                          grid_params_in,
                                          coarse_grid_params_in);
  auto count = 0;
  field<height_types> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(null_htype);
  while (! q.empty()) {
    auto cell = static_cast<basin_cell*>(q.front());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = cell->get_height_type();
    count++;
    q.pop();
    delete cell;
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete alg4;
  delete[] expected_cells_in_queue;
  delete[] raw_orography_in; delete[] corrected_orography_in; delete[] minima_in;
  delete[] landsea_in; delete[] true_sinks_in;
  delete[] next_cell_lat_index_in; delete[] next_cell_lon_index_in;
  delete[] rdirs_in; delete[] catchment_nums_in;
}

TEST_F(BasinEvaluationTest, TestAddingMinimaToQueueSix) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  double* raw_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  double* corrected_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  bool* minima_in = new bool[9*9] {false,false,false, false,false,false, false,false,false,
                                   false,false,false, false,false,false, false,false,false,
                                   false,false,false, false,false,false, false,false, true,
//
                                   false,false,false, false,false,false, false,false,false,
                                   false,false,false, false, true,false, false,false,false,
                                   false,false,false, false,false,false, false,false,false,
//
                                   false,false,false, false,false,false, false, true,false,
                                   false, true,false, false,false,false, false,false,false,
                                   false,false,false, false,false,false, false,false,false};
  height_types* expected_cells_in_queue = new height_types[9*9]
    {null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,flood_height,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,flood_height,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,flood_height,null_htype,
     null_htype,flood_height,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype };
  bool* landsea_in = new bool[9*9];
  fill_n(landsea_in,9*9,false);
  landsea_in[0] = true;
  bool* true_sinks_in = new bool[9*9];
  fill_n(true_sinks_in,9*9,false);
  int* next_cell_lat_index_in = new int[9*9];
  int* next_cell_lon_index_in = new int[9*9];
  short* rdirs_in = new short[9*9];
  int* catchment_nums_in = new int[9*9];
  queue<cell*> q;
  auto alg4 = new sink_filling_algorithm_4_latlon();
  alg4->setup_flags(false,true,false,false);
  alg4->setup_fields(corrected_orography_in,
                     landsea_in,
                     true_sinks_in,
                     next_cell_lat_index_in,
                     next_cell_lon_index_in,
                     grid_params_in,
                     rdirs_in,
                     catchment_nums_in);
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_add_minima_to_queue(raw_orography_in,
                                          corrected_orography_in,
                                          minima_in,
                                          alg4,
                                          grid_params_in,
                                          coarse_grid_params_in);
  auto count = 0;
  field<height_types> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(null_htype);
  while (! q.empty()) {
    auto cell = static_cast<basin_cell*>(q.front());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = cell->get_height_type();
    count++;
    q.pop();
    delete cell;
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
  EXPECT_EQ(4,count);
  delete grid_params_in; delete coarse_grid_params_in;
  delete alg4;
  delete[] expected_cells_in_queue;
  delete[] raw_orography_in; delete[] corrected_orography_in; delete[] minima_in;
  delete[] landsea_in; delete[] true_sinks_in;
  delete[] next_cell_lat_index_in; delete[] next_cell_lon_index_in;
  delete[] rdirs_in; delete[] catchment_nums_in;
}

TEST_F(BasinEvaluationTest, TestingProcessingNeighbors) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  double* raw_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 3.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,1.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 3.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  double* corrected_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 1.0,2.0,3.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,1.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  bool* completed_cells_in = new bool[9*9] {false,false,false, false,false,false,  true,false,false,
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
//
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false, true,false, false,false,false,
                                            false,false,false, false, true,false, false,false,false,
//
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false, true,false,
                                            false,false,false, false,false,false, false,false,false};
  bool* completed_cells_expected_out = new bool[9*9]
                                            {false,false,false, false,false,false,  true,false,false,
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
//
                                            false,false,false,  true, true,true, false,false,false,
                                            false,false,false,  true, true,true, false,false,false,
                                            false,false,false,  true, true,true, false,false,false,
//
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false, true,false,
                                            false,false,false, false,false,false, false,false,false};
  height_types* expected_cells_in_queue = new height_types[9*9]
    {null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
//
     null_htype,null_htype,null_htype, connection_height,flood_height,flood_height, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, flood_height,null_htype,flood_height, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, connection_height,null_htype,connection_height, null_htype,null_htype,null_htype,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype };
  priority_cell_queue q;
  coords* center_coords_in = new latlon_coords(4,4);
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_neighbors(center_coords_in,
                                        completed_cells_in,
                                        raw_orography_in,
                                        corrected_orography_in,
                                        grid_params_in);
  auto count = 0;
  field<height_types> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(null_htype);
  while (! q.empty()) {
    auto cell = static_cast<basin_cell*>(q.top());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = cell->get_height_type();
    count++;
    q.pop();
    delete cell;
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
  EXPECT_TRUE(field<bool>(completed_cells_in,grid_params_in) == field<bool>(completed_cells_expected_out,grid_params_in));
  EXPECT_EQ(7,count);
  delete grid_params_in;
  delete[] raw_orography_in; delete[] corrected_orography_in; delete[] completed_cells_in;
  delete[] completed_cells_expected_out; delete[] expected_cells_in_queue;
  delete center_coords_in;
}

TEST_F(BasinEvaluationTest, TestingProcessingNeighborsTwo) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  double* raw_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,3.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,1.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 3.0,2.0,2.0, 2.0,-2.0,-2.0,
                                              2.0,2.0,2.0, 2.0,2.0,1.0, 2.0,-3.0,2.0,
                                              2.0,2.0,2.0, 3.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, -3.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, -2.0,2.0,2.0, 2.0,2.0,2.0};
  double* corrected_orography_in = new double[9*9] {2.0,1.0,1.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,3.0,3.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 1.0,2.0,3.0, 2.0,-1.0,-3.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,-3.0,2.0,
                                                    2.0,1.0,2.0, 2.0,2.0,1.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, -2.0,-1.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,3.0, 2.0,2.0,2.0,
                                                    2.0,2.0,1.0, -3.0,-1.0,2.0, 2.0,2.0,2.0};
  bool* completed_cells_in = new bool[9*9] {false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
//
                                            false,false,false,  true, true, true, false, false,false,
                                            false,false,false,  true, true, true, false, false,false,
                                            false,false,false,  true, true, true, false, false,false,
//
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false};
  bool* completed_cells_expected_out = new bool[9*9]
                                            {false,false,true, false, true,false,false,false,false,
                                             false,false,true,  true, true,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
//
                                            false,false,false,  true, true,true, false,true,true,
                                            true, true, true,  true, true,true, false, true,false,
                                            true,false, true,  true, true,true, false, true,true,
//
                                            true, true, true,  true, true, true, false,false,false,
                                            false,false,false,  true,false, true, false,false,false,
                                            false,false,false,  true,true, true, false,false,false};
  height_types* expected_cells_in_queue = new height_types[9*9]
    {null_htype,null_htype,connection_height, null_htype,connection_height,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,flood_height, flood_height,flood_height,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,flood_height,connection_height,
     flood_height,flood_height,flood_height, null_htype,null_htype,null_htype, null_htype,flood_height,null_htype,
     flood_height,null_htype,flood_height, null_htype,null_htype,null_htype, null_htype,flood_height,flood_height,
//
     flood_height,flood_height,flood_height, flood_height,connection_height,flood_height, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, flood_height,null_htype,flood_height, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, connection_height,connection_height,flood_height, null_htype,null_htype,null_htype };
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  coords* center_coords_in = new latlon_coords(0,3);
  q = basin_eval.test_process_neighbors(center_coords_in,
                                        completed_cells_in,
                                        raw_orography_in,
                                        corrected_orography_in,
                                        grid_params_in);
  delete center_coords_in;
  center_coords_in = new latlon_coords(4,8);
  q = basin_eval.test_process_neighbors(center_coords_in,
                                        completed_cells_in,
                                        raw_orography_in,
                                        corrected_orography_in,
                                        grid_params_in);
  delete center_coords_in;
  center_coords_in = new latlon_coords(7,4);
  q = basin_eval.test_process_neighbors(center_coords_in,
                                        completed_cells_in,
                                        raw_orography_in,
                                        corrected_orography_in,
                                        grid_params_in);
  delete center_coords_in;
  center_coords_in = new latlon_coords(5,1);
  q = basin_eval.test_process_neighbors(center_coords_in,
                                        completed_cells_in,
                                        raw_orography_in,
                                        corrected_orography_in,
                                        grid_params_in);
  delete center_coords_in;
  auto count = 0;
  field<height_types> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(null_htype);
  while (! q.empty()) {
    auto cell = static_cast<basin_cell*>(q.top());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = cell->get_height_type();
    count++;
    q.pop();
    delete cell;
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
  EXPECT_TRUE(field<bool>(completed_cells_in,grid_params_in) == field<bool>(completed_cells_expected_out,grid_params_in));
  EXPECT_EQ(26,count);
  delete grid_params_in;
  delete[] raw_orography_in; delete[] corrected_orography_in; delete[] completed_cells_in;
  delete[] completed_cells_expected_out; delete[] expected_cells_in_queue;
}

TEST_F(BasinEvaluationTest, TestingProcessingNeighborsThree) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* raw_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  double* corrected_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,1.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              1.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              3.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              3.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              1.0,2.0,2.0, 1.0,2.0,3.0, 2.0,2.0,2.0};
  bool* completed_cells_in = new bool[9*9] {false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
//
                                            false,false,false,  true, true, true, false, false,false,
                                            false,false,false,  true, true, true, false, false,false,
                                            false,false,false,  true, true, true, false, false,false,
//
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false};
  bool* completed_cells_expected_out = new bool[9*9]
                                           {false, true,false, false,false,false, false,false, true,
                                             true, true,false, false,false,false, false,false, true,
                                            false,false,false, false,false,false, false,false,false,
//
                                             true,false,false,  true, true, true, false,  true, true,
                                             true,false,false,  true, true, true, false,  true,false,
                                             true,false,false,  true, true, true, false,  true, true,
//
                                            false,false,false, false,false,false, false,false,false,
                                             true,false,false,  true, true, true, false, true, true,
                                             true,false,false,  true,false, true, false, true,false};
  height_types* expected_cells_in_queue = new height_types[9*9]
    {null_htype,flood_height,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,flood_height,
     flood_height,flood_height,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,connection_height,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
//
     flood_height,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,flood_height,flood_height,
     connection_height,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,flood_height,null_htype,
     flood_height,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,flood_height,flood_height,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     flood_height,null_htype,null_htype, flood_height,flood_height,flood_height, null_htype,flood_height,flood_height,
     connection_height,null_htype,null_htype, connection_height,null_htype,flood_height, null_htype,flood_height,null_htype };
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  coords* center_coords_in = new latlon_coords(0,0);
  q = basin_eval.test_process_neighbors(center_coords_in,
                                        completed_cells_in,
                                        raw_orography_in,
                                        corrected_orography_in,
                                        grid_params_in);
  delete center_coords_in;
  center_coords_in = new latlon_coords(8,4);
  q = basin_eval.test_process_neighbors(center_coords_in,
                                        completed_cells_in,
                                        raw_orography_in,
                                        corrected_orography_in,
                                        grid_params_in);
  delete center_coords_in;
  center_coords_in = new latlon_coords(4,8);
  q = basin_eval.test_process_neighbors(center_coords_in,
                                        completed_cells_in,
                                        raw_orography_in,
                                        corrected_orography_in,
                                        grid_params_in);
  delete center_coords_in;
  center_coords_in = new latlon_coords(8,8);
  q = basin_eval.test_process_neighbors(center_coords_in,
                                        completed_cells_in,
                                        raw_orography_in,
                                        corrected_orography_in,
                                        grid_params_in);
  delete center_coords_in;
  auto count = 0;
  field<height_types> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(null_htype);
  while (! q.empty()) {
    auto cell = static_cast<basin_cell*>(q.top());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = cell->get_height_type();
    count++;
    q.pop();
    delete cell;
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
  EXPECT_TRUE(field<bool>(completed_cells_in,grid_params_in) == field<bool>(completed_cells_expected_out,grid_params_in));
  EXPECT_EQ(23,count);
  delete grid_params_in;
  delete[] raw_orography_in; delete[] corrected_orography_in; delete[] completed_cells_in;
  delete[] completed_cells_expected_out; delete[] expected_cells_in_queue;
}

TEST_F(BasinEvaluationTest, TestProcessingSearchNeighbors) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  bool* completed_cells_in = new bool[9*9] {false,false,false, false,false,false,  true,false,false,
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
//
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false, true,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
//
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false, true,false,
                                            false,false,false, false,false,false, false,false,false};
  bool* completed_cells_expected_out = new bool[9*9]
                                           {false,false,false,  true,false, true,  true,false,false,
                                            false,false,false,  true, true, true, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
//
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false, true, true,  true, true, true,
                                            false,false,false, false,false, true, false, true,false,
//
                                             true, true, true, false,false, true,  true, true, true,
                                             true,false, true, false,false,false, false, true,false,
                                             true, true, true, false,false,false, false,false,false};
  bool* expected_cells_in_queue = new bool[9*9]
                                           {false,false,false,  true,false, true, false,false,false,
                                            false,false,false,  true, true, true, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
//
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false, true,  true, true, true,
                                            false,false,false, false,false, true, false, true,false,
//
                                             true, true, true, false,false, true,  true, true, true,
                                             true,false, true, false,false,false, false,false,false,
                                             true, true, true, false,false,false, false,false,false};
  queue<landsea_cell*> q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  coords* search_coords_in = new latlon_coords(5,6);
  q = basin_eval.test_search_process_neighbors(search_coords_in,
                                               completed_cells_in,
                                               grid_params_in);
  delete search_coords_in;
  search_coords_in = new latlon_coords(7,1);
  q = basin_eval.test_search_process_neighbors(search_coords_in,
                                               completed_cells_in,
                                               grid_params_in);
  delete search_coords_in;
  search_coords_in = new latlon_coords(0,4);
  q = basin_eval.test_search_process_neighbors(search_coords_in,
                                               completed_cells_in,
                                               grid_params_in);
  delete search_coords_in;
  search_coords_in = new latlon_coords(5,8);
  q = basin_eval.test_search_process_neighbors(search_coords_in,
                                               completed_cells_in,
                                               grid_params_in);
  delete search_coords_in;
  auto count = 0;
  field<bool> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(false);
  while (! q.empty()) {
    auto cell = static_cast<landsea_cell*>(q.front());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = true;
    count++;
    q.pop();
    delete cell;
  }
  EXPECT_TRUE(cells_in_queue == field<bool>(expected_cells_in_queue,grid_params_in));
  EXPECT_TRUE(field<bool>(completed_cells_in,grid_params_in) == field<bool>(completed_cells_expected_out,grid_params_in));
  EXPECT_EQ(23,count);
  delete grid_params_in;
  delete[] completed_cells_in; delete[] completed_cells_expected_out; delete[] expected_cells_in_queue;
}

TEST_F(BasinEvaluationTest, TestProcessingSearchNeighborsTwo) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  bool* completed_cells_in = new bool[9*9] {false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
//
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
//
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
                                            false,false,false, false,false,false, false,false,false};
  bool* completed_cells_expected_out = new bool[9*9]
                                           {false,false,false,  true,false, true, false,false,false,
                                            false,false,false,  true, true, true, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
//
                                             true, true,false, false, true, true,  true,false, true,
                                            false, true,false, false, true,false,  true,false, true,
                                            true,  true,false, false, true, true,  true,false, true,
//
                                            false,false,false, false,false,false, false,false,false,
                                             true,false,false, false,false,false, false, true, true,
                                             true,false,false, false,false,false, false, true,false};
  bool* expected_cells_in_queue = new bool[9*9]
                                           {false,false,false,  true,false, true, false,false,false,
                                            false,false,false,  true, true, true, false,false,false,
                                            false,false,false, false,false,false, false,false,false,
//
                                             true, true,false, false, true, true,  true,false, true,
                                            false, true,false, false, true,false,  true,false, true,
                                            true,  true,false, false, true, true,  true,false, true,
//
                                            false,false,false, false,false,false, false,false,false,
                                             true,false,false, false,false,false, false, true, true,
                                             true,false,false, false,false,false, false, true,false};
  queue<landsea_cell*> q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  coords* search_coords_in = new latlon_coords(4,5);
  q = basin_eval.test_search_process_neighbors(search_coords_in,
                                               completed_cells_in,
                                               grid_params_in);
  delete search_coords_in;
  search_coords_in = new latlon_coords(4,0);
  q = basin_eval.test_search_process_neighbors(search_coords_in,
                                               completed_cells_in,
                                               grid_params_in);
  delete search_coords_in;
  search_coords_in = new latlon_coords(0,4);
  q = basin_eval.test_search_process_neighbors(search_coords_in,
                                               completed_cells_in,
                                               grid_params_in);
  delete search_coords_in;
  search_coords_in = new latlon_coords(8,8);
  q = basin_eval.test_search_process_neighbors(search_coords_in,
                                               completed_cells_in,
                                               grid_params_in);
  delete search_coords_in;
  auto count = 0;
  field<bool> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(false);
  while (! q.empty()) {
    auto cell = static_cast<landsea_cell*>(q.front());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = true;
    count++;
    q.pop();
    delete cell;
  }
  EXPECT_TRUE(cells_in_queue == field<bool>(expected_cells_in_queue,grid_params_in));
  EXPECT_TRUE(field<bool>(completed_cells_in,grid_params_in) == field<bool>(completed_cells_expected_out,grid_params_in));
  EXPECT_EQ(26,count);
  delete grid_params_in;
  delete[] completed_cells_in; delete[] completed_cells_expected_out; delete[] expected_cells_in_queue;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCell) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,-1.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,7,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,2,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,9.1,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,67,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0};
  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false, true, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 67;
  double lake_area_in = 6.0;
  double center_cell_height_in = 1.5;
  double center_cell_volume_threshold_in = 8.5;
  double previous_filled_cell_height_in = 1.4;
  coords* previous_filled_cell_coords_in = new latlon_coords(7,1);
  coords* center_coords_in = new latlon_coords(7,2);
  basin_cell* center_cell_in = new basin_cell(1.4,connection_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = connection_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  basin_cell* qcell = static_cast<basin_cell*>(q.top());
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lat(),7);
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lon(),1);
  EXPECT_EQ(static_cast<basin_cell*>(qcell)->get_height_type(),flood_height);
  q.pop();
  delete qcell;
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 9.1)<1E-7);
  EXPECT_EQ(lake_area_in,6.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.4);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellTwo) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,-1.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,7,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,2,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,9.1,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,67,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0};
  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false, true, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 67;
  double lake_area_in = 6.0;
  double center_cell_height_in = 1.5;
  double center_cell_volume_threshold_in = 8.5;
  double previous_filled_cell_height_in = 1.4;
  coords* previous_filled_cell_coords_in = new latlon_coords(7,1);
  coords* center_coords_in = new latlon_coords(7,2);
  basin_cell* center_cell_in = new basin_cell(1.4,flood_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = flood_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 9.1)<1E-7);
  EXPECT_EQ(lake_area_in,7.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.4);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete previous_filled_cell_coords_in; delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellThree) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,-1.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1, 7, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,2, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,9.1, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,67, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0};
  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false, true, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 67;
  double lake_area_in = 6.0;
  double center_cell_height_in = 1.5;
  double center_cell_volume_threshold_in = 8.5;
  double previous_filled_cell_height_in = 1.4;
  coords* previous_filled_cell_coords_in = new latlon_coords(7,2);
  coords* center_coords_in = new latlon_coords(7,2);
  basin_cell* center_cell_in = new basin_cell(1.4,flood_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = flood_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 9.1)<1E-7);
  EXPECT_EQ(lake_area_in,7.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.4);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete previous_filled_cell_coords_in; delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellFour) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,-1.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                               0,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                               3,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             2.1,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            31,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0};
  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false,  true,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 31;
  double lake_area_in = 2.0;
  double center_cell_height_in = 2.1;
  double center_cell_volume_threshold_in = 1.5;
  double previous_filled_cell_height_in = 1.8;
  coords* previous_filled_cell_coords_in = new latlon_coords(3,0);
  coords* center_coords_in = new latlon_coords(0,3);
  basin_cell* center_cell_in = new basin_cell(1.8,connection_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = connection_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  basin_cell* qcell = static_cast<basin_cell*>(q.top());
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lat(),3);
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lon(),0);
  EXPECT_EQ(static_cast<basin_cell*>(qcell)->get_height_type(),flood_height);
  q.pop();
  delete qcell;
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 2.1)<1E-7);
  EXPECT_EQ(lake_area_in,2.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.8);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellFive) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,-1.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                         0,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                         3,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};

  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             2.1,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            31,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0};
  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false,  true,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 31;
  double lake_area_in = 2.0;
  double center_cell_height_in = 2.1;
  double center_cell_volume_threshold_in = 1.5;
  double previous_filled_cell_height_in = 1.8;
  coords* previous_filled_cell_coords_in = new latlon_coords(3,0);
  coords* center_coords_in = new latlon_coords(0,3);
  basin_cell* center_cell_in = new basin_cell(1.8,flood_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = flood_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 2.1)<1E-7);
  EXPECT_EQ(lake_area_in,3.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.8);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete previous_filled_cell_coords_in; delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellSix) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,-1.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                         3,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                         0,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             2.1,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            31,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0};
  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                       true,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 31;
  double lake_area_in = 2.0;
  double center_cell_height_in = 2.1;
  double center_cell_volume_threshold_in = 1.5;
  double previous_filled_cell_height_in = 1.8;
  coords* previous_filled_cell_coords_in = new latlon_coords(3,0);
  coords* center_coords_in = new latlon_coords(3,0);
  basin_cell* center_cell_in = new basin_cell(1.8,flood_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = flood_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 2.1)<1E-7);
  EXPECT_EQ(lake_area_in,3.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.8);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete previous_filled_cell_coords_in; delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellSeven) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,-1.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1, 8, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1, 2, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0, 0.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,67, 0,0,0, 0,0,0};
  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false, true, false,false,false, false,false,false};
  int basin_catchment_number_in = 67;
  double lake_area_in = 0.0;
  double center_cell_height_in = 1.1;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 0.0;
  coords* previous_filled_cell_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,2);
  basin_cell* center_cell_in = new basin_cell(1.1,connection_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = connection_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  basin_cell* qcell = static_cast<basin_cell*>(q.top());
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lat(),8);
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lon(),2);
  EXPECT_EQ(static_cast<basin_cell*>(qcell)->get_height_type(),flood_height);
  q.pop();
  delete qcell;
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.0)<1E-7);
  EXPECT_EQ(lake_area_in,0.0);
  EXPECT_EQ(previous_filled_cell_height_in,0.0);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellEight) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,-1.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,8, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,2, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0, 0.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,67, 0,0,0, 0,0,0};
  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false, true, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 67;
  double lake_area_in = 0.0;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_filled_cell_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,2);
  basin_cell* center_cell_in = new basin_cell(1.1,flood_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = flood_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.0)<1E-7);
  EXPECT_EQ(lake_area_in,1.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.1);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete previous_filled_cell_coords_in; delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellNine) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,-1.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,8, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,0.2, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,67, 0,0,0, 0,0,0};
  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false, true,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 67;
  double lake_area_in = 1.0;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_filled_cell_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,flood_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = flood_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(lake_area_in,2.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.1);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete previous_filled_cell_coords_in; delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellTen) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,-1.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1, 8, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1, 1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,0.2, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,67, 0,0,0, 0,0,0};

  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false, true,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 67;
  double lake_area_in = 1.0;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_filled_cell_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,connection_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = connection_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  basin_cell* qcell = static_cast<basin_cell*>(q.top());
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lat(),8);
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lon(),2);
  EXPECT_EQ(static_cast<basin_cell*>(qcell)->get_height_type(),flood_height);
  q.pop();
  delete qcell;
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(lake_area_in,1.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.1);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellEleven) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,0.5, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,10, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1, 8, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1, 1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0, 0.2, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,0.5, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,10, 0,0,0, 0,0,0};

  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false, true,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 67;
  double lake_area_in = 1.0;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_filled_cell_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,connection_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = connection_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  EXPECT_TRUE(q.size() == 1);
  basin_cell* qcell = static_cast<basin_cell*>(q.top());
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lat(),8);
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lon(),2);
  EXPECT_EQ(static_cast<basin_cell*>(qcell)->get_height_type(),flood_height);
  q.pop();
  delete qcell;
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(lake_area_in,1.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.1);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellTwelve) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,0.5, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,10, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1, 8, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1, 1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,0.2, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,10, 0,0,0, 0,0,0};

  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false, true,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 67;
  double lake_area_in = 1.0;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_filled_cell_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,flood_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = flood_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(lake_area_in,2.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.1);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete previous_filled_cell_coords_in; delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellThirteen) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,0.5, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1, 8, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1, 1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,0.2, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,0.5, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,67, 0,0,0, 0,0,0};

  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false, true,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 67;
  double lake_area_in = 1.0;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_filled_cell_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,connection_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = connection_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  basin_cell* qcell = static_cast<basin_cell*>(q.top());
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lat(),8);
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lon(),2);
  EXPECT_EQ(static_cast<basin_cell*>(qcell)->get_height_type(),flood_height);
  q.pop();
  delete qcell;
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(lake_area_in,1.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.1);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellFourteen) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,0.5, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1, 8, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,0.2, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,67, 0,0,0, 0,0,0};

  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false, true,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 67;
  double lake_area_in = 1.0;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_filled_cell_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,flood_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = flood_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(lake_area_in,2.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.1);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete previous_filled_cell_coords_in; delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellFifteen) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,10, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1, 8, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1, 1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,0.2, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,10, 0,0,0, 0,0,0};

  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false, true,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 67;
  double lake_area_in = 1.0;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_filled_cell_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,connection_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = connection_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  basin_cell* qcell = static_cast<basin_cell*>(q.top());
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lat(),8);
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lon(),2);
  EXPECT_EQ(static_cast<basin_cell*>(qcell)->get_height_type(),flood_height);
  q.pop();
  delete qcell;
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(lake_area_in,1.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.1);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellSixteen) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,10, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1, 8, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,0.2, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,10, 0,0,0, 0,0,0};

  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false, true,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 67;
  double lake_area_in = 1.0;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_filled_cell_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,flood_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = flood_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(lake_area_in,2.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.1);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete previous_filled_cell_coords_in; delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellSeventeen) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,-1.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,8, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,0.2, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,67, 0,0,0, 0,0,0};
  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false, true,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 67;
  double lake_area_in = 1.0;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_filled_cell_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,connection_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = flood_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(lake_area_in,1.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.1);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete previous_filled_cell_coords_in; delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellEighteen) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,-1.0);
  double* raw_orography_in = new double[9*9] {1.2,1.3,1.4, 2.1,2.3,2.3, 2.4,0.3,0.2,
                                              1.3,1.5,1.5, 1.9,2.1,2.2, 0.5,0.4,0.4,
                                              1.5,1.6,1.7, 2.0,2.0,0.9, 0.6,0.5,0.5,
//
                                              1.8,1.6,1.5, 1.4,1.2,0.9, 0.8,2.0,0.6,
                                              2.1,1.8,1.6, 1.5,1.3,1.2, 2.7,2.3,2.3,
                                              2.1,1.9,2.2, 1.8,1.4,2.8, 2.7,2.3,2.1,
//
                                              2.3,2.4,1.8, 1.5,1.8,2.9, 2.5,1.9,1.6,
                                              2.5,1.4,1.5, 1.4,2.2,2.9, 1.4,1.2,1.1,
                                              1.5,1.3,1.1, 1.1,2.4,2.2, 1.9,1.8,1.2};
  double* cell_areas_in = new double[9*9];
  std::fill_n(cell_areas_in,9*9,1.0);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  bool* flooded_cells_in = new bool[9*9];
  std::fill_n(flooded_cells_in,9*9,false);
  bool* connected_cells_in = new bool[9*9];
  std::fill_n(connected_cells_in,9*9,false);
  int* basin_catchment_numbers_in = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
//
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,0, 0,0,0, 0,0,0,
                                                  0,0,10, 0,0,0, 0,0,0};
  int* flood_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                        -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1, 8, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                              -1,-1, 1, -1,-1,-1, -1,-1,-1};
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,0.2, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                                             -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  int* basin_catchment_numbers_expected_out = new int[9*9] {0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
//
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,0, 0,0,0, 0,0,0,
                                                            0,0,10, 0,0,0, 0,0,0};

  bool* flooded_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false, true,false, false,false,false, false,false,false};
  bool* connected_cells_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  int basin_catchment_number_in = 67;
  double lake_area_in = 1.0;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_filled_cell_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,flood_height,center_coords_in);
  height_types previous_filled_cell_height_type_in = connection_height;
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_filled_cell_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          cell_areas_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          flooded_cells_in,
                                          connected_cells_in,
                                          center_cell_volume_threshold_in,
                                          lake_area_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          previous_filled_cell_height_type_in,
                                          grid_params_in);
  basin_cell* qcell = static_cast<basin_cell*>(q.top());
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lat(),8);
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lon(),2);
  EXPECT_EQ(static_cast<basin_cell*>(qcell)->get_height_type(),flood_height);
  q.pop();
  delete qcell;
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in) == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in) == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(lake_area_in,2.0);
  EXPECT_EQ(previous_filled_cell_height_in,1.1);
  EXPECT_TRUE(field<bool>(flooded_cells_in,grid_params_in) == field<bool>(flooded_cells_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connected_cells_in,grid_params_in) == field<bool>(connected_cells_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_volume_thresholds_in; delete[] connection_volume_thresholds_in; delete[] raw_orography_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flooded_cells_in; delete[] connected_cells_in; delete[] basin_catchment_numbers_in;
  delete[] flood_next_cell_lat_index_expected_out; delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out; delete[] connect_next_cell_lon_index_expected_out;
  delete[] connection_volume_thresholds_expected_out; delete[] flood_volume_thresholds_expected_out;
  delete[] basin_catchment_numbers_expected_out; delete[] flooded_cells_expected_out;
  delete[] connected_cells_expected_out;
  delete[] cell_areas_in;
  delete center_cell_in;
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectOne) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,false);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  3,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, connection_merge_not_set_flood_merge_as_primary,no_merge,no_merge,
    no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  5,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, 2,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  2,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(5,2);
  coords* center_coords_in = new latlon_coords(6,3);
  coords* previous_filled_cell_coords_in = new latlon_coords(8,3);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,6));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectTwo) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,false);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  3,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,connection_merge_not_set_flood_merge_as_primary,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,0,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1, 5,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1, 1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1, 5,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false, true,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(0,5);
  coords* center_coords_in = new latlon_coords(1,3);
  coords* previous_filled_cell_coords_in = new latlon_coords(0,7);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,5));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectThree) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,false);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  3,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,connection_merge_not_set_flood_merge_as_primary,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1, 3,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1, 7,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1, 4,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1, 8,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false, true,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(3,7);
  coords* center_coords_in = new latlon_coords(3,6);
  coords* previous_filled_cell_coords_in = new latlon_coords(4,1);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,5));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectFour) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,false);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  3,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    connection_merge_not_set_flood_merge_as_primary,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1,  3,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, 8,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, 4,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, 8,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false,  true,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(3,8);
  coords* center_coords_in = new latlon_coords(5,6);
  coords* previous_filled_cell_coords_in = new latlon_coords(0,6);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,5));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectFive) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,false);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  8,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,connection_merge_not_set_flood_merge_as_primary,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1, 2,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1, 2,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1, 1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1, 1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false, true,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(2,2);
  coords* center_coords_in = new latlon_coords(2,0);
  coords* previous_filled_cell_coords_in = new latlon_coords(3,1);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,5));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectSix) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,false);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  3,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,connection_merge_not_set_flood_merge_as_primary,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1, 5,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1, 1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1, 2,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1, 1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(5,1);
  coords* center_coords_in = new latlon_coords(6,0);
  coords* previous_filled_cell_coords_in = new latlon_coords(8,1);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,5));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectSeven) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,false);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  3,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    no_merge,no_merge,no_merge, connection_merge_not_set_flood_merge_as_primary,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1,  5,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1,  1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1,  2,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1,  2,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(5,1);
  coords* center_coords_in = new latlon_coords(0,4);
  coords* previous_filled_cell_coords_in = new latlon_coords(0,3);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,5));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                           flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectEight) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,false);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  3,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,connection_merge_not_set_flood_merge_as_primary,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1, 0,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1, 7,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1, 0,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1, 1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(0,7);
  coords* center_coords_in = new latlon_coords(8,8);
  coords* previous_filled_cell_coords_in = new latlon_coords(6,7);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,5));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectNine) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  3,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,connection_merge_not_set_flood_merge_as_primary, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1, 5,  -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1, 2,  -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1, 2,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1, 1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(5,2);
  coords* center_coords_in = new latlon_coords(6,3);
  coords* previous_filled_cell_coords_in = new latlon_coords(8,2);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,6));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectTen) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  3,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    connection_merge_not_set_flood_merge_as_primary,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              { 0,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              { 5,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              { 1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              { 5,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     { true,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(0,5);
  coords* center_coords_in = new latlon_coords(1,3);
  coords* previous_filled_cell_coords_in = new latlon_coords(0,0);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,5));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectEleven) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  3,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,connection_merge_not_set_flood_merge_as_primary,no_merge,
    no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1, 3,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1, 7,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1, 4,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1, 8,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false, true,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(3,7);
  coords* center_coords_in = new latlon_coords(3,6);
  coords* previous_filled_cell_coords_in = new latlon_coords(4,4);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,5));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectTwelve) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  3,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    connection_merge_not_set_flood_merge_as_primary,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1,  3,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1,  8,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1,  4,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1,  8,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false,  true,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(3,8);
  coords* center_coords_in = new latlon_coords(5,6);
  coords* previous_filled_cell_coords_in = new latlon_coords(0,6);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,5));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectThirteen) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  3,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    connection_merge_not_set_flood_merge_as_primary,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1,  2,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1,  2,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1,  1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1,  0,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(2,2);
  coords* center_coords_in = new latlon_coords(0,5);
  coords* previous_filled_cell_coords_in = new latlon_coords(5,6);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,5));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectFourteen) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  3,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    connection_merge_not_set_flood_merge_as_primary,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                5,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                2,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                                1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(5,1);
  coords* center_coords_in = new latlon_coords(6,0);
  coords* previous_filled_cell_coords_in = new latlon_coords(7,0);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,5));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectFifteen) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  3,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, connection_merge_not_set_flood_merge_as_primary,no_merge,no_merge,
    no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  5,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  2,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  2,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(5,1);
  coords* center_coords_in = new latlon_coords(0,4);
  coords* previous_filled_cell_coords_in = new latlon_coords(1,3);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,5));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectSixteen) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,8,9,
                                                  3,3,3, 3,8,8, 8,9,9,
                                                  3,3,3, 8,8,8, 9,9,9,
//
                                                  3,8,8, 8,8,8, 8,9,9,
                                                  3,8,6, 6,8,8, 8,8,9,
                                                  3,6,6, 6,6,6, 8,9,9,
//
                                                  3,3,5, 5,6,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,9,9,
                                                  3,3,5, 5,5,6, 6,6,9};
  int* coarse_catchment_nums_in = new int[3*3] {71,55,55,
                                                71,42,42,
                                                86,38,38};
  int* flood_force_merge_lat_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lat_index_in,9*9,-1);
  int* flood_force_merge_lon_index_in = new int[9*9];
  std::fill_n(flood_force_merge_lon_index_in,9*9,-1);
  int* connect_force_merge_lat_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_in,9*9,-1);
  int* connect_force_merge_lon_index_in = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[9*9];
  std::fill_n(flood_local_redirect_in,9*9,false);
  bool* connect_local_redirect_in = new bool[9*9];
  std::fill_n(connect_local_redirect_in,9*9,false);
  merge_types* merge_points_in = new merge_types[9*9];
  std::fill_n(merge_points_in,9*9,no_merge);
  merge_types* merge_points_expected_out = new merge_types[9*9]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,connection_merge_not_set_flood_merge_as_primary,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,0,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,7,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lat_index_expected_out,9*9,-1);
  int* connect_force_merge_lon_index_expected_out = new int[9*9];
  std::fill_n(connect_force_merge_lon_index_expected_out,9*9,-1);
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1, 0,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1, 2,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(0,7);
  coords* center_coords_in = new latlon_coords(8,8);
  coords* previous_filled_cell_coords_in = new latlon_coords(6,8);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(1,5));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest,TestSettingSecondaryRedirect) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  bool* requires_flood_redirect_indices_in = new bool[9*9];
  std::fill_n(requires_flood_redirect_indices_in,9*9,false);
  bool* requires_connect_redirect_indices_in = new bool[9*9];
  std::fill_n(requires_connect_redirect_indices_in,9*9,false);
  double* raw_orography_in = new double[9*9];
  std::fill_n(raw_orography_in,9*9,10.0);
  double* corrected_orography_in = new double[9*9];
  std::fill_n(corrected_orography_in,9*9, 9.0);
  int* flood_next_cell_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,2,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1, 7,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,2,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,7,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  bool* requires_flood_redirect_indices_expected_out = new bool[9*9]
                                     {false,false,false, false, true,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* requires_connect_redirect_indices_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(2,7);
  coords* center_coords_in = new latlon_coords(5,1);
  coords* previous_filled_cell_coords_in = new latlon_coords(0,4);
  height_types previous_filled_cell_height_type_in = flood_height;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_secondary_redirect(flood_next_cell_lat_index_in,
                                         flood_next_cell_lon_index_in,
                                         connect_next_cell_lat_index_in,
                                         connect_next_cell_lon_index_in,
                                         flood_redirect_lat_index_in,
                                         flood_redirect_lon_index_in,
                                         connect_redirect_lat_index_in,
                                         connect_redirect_lon_index_in,
                                         requires_flood_redirect_indices_in,
                                         requires_connect_redirect_indices_in,
                                         raw_orography_in,
                                         corrected_orography_in,new_center_coords_in,
                                         center_coords_in,previous_filled_cell_coords_in,
                                         previous_filled_cell_height_type_in,
                                         grid_params_in);
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in)
              == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in)
              == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(requires_flood_redirect_indices_in,grid_params_in)
              == field<bool>(requires_flood_redirect_indices_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(requires_connect_redirect_indices_in,grid_params_in)
              == field<bool>(requires_connect_redirect_indices_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] requires_flood_redirect_indices_in;
  delete[] requires_connect_redirect_indices_in;
  delete[] raw_orography_in; delete[] corrected_orography_in;
  delete[] flood_next_cell_lat_index_expected_out;
  delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out;
  delete[] connect_next_cell_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] requires_flood_redirect_indices_expected_out;
  delete[] requires_connect_redirect_indices_expected_out;
  delete new_center_coords_in; delete center_coords_in;
  delete previous_filled_cell_coords_in;
}

TEST_F(BasinEvaluationTest,TestSettingSecondaryRedirectTwo) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  bool* requires_flood_redirect_indices_in = new bool[9*9];
  std::fill_n(requires_flood_redirect_indices_in,9*9,false);
  bool* requires_connect_redirect_indices_in = new bool[9*9];
  std::fill_n(requires_connect_redirect_indices_in,9*9,false);
  double* raw_orography_in = new double[9*9];
  std::fill_n(raw_orography_in,9*9,10.0);
  double* corrected_orography_in = new double[9*9];
  std::fill_n(corrected_orography_in,9*9, 9.0);
  int* flood_next_cell_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1, 5,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,8,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,5,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,8,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  bool* requires_flood_redirect_indices_expected_out = new bool[9*9]
                                     {false,false,false, false, true,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* requires_connect_redirect_indices_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(5,8);
  coords* center_coords_in = new latlon_coords(5,1);
  coords* previous_filled_cell_coords_in = new latlon_coords(0,4);
  height_types previous_filled_cell_height_type_in = flood_height;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_secondary_redirect(flood_next_cell_lat_index_in,flood_next_cell_lon_index_in,
                                         connect_next_cell_lat_index_in,connect_next_cell_lon_index_in,
                                         flood_redirect_lat_index_in,
                                         flood_redirect_lon_index_in,
                                         connect_redirect_lat_index_in,
                                         connect_redirect_lon_index_in,
                                         requires_flood_redirect_indices_in,
                                         requires_connect_redirect_indices_in,
                                         raw_orography_in,
                                         corrected_orography_in,new_center_coords_in,
                                         center_coords_in,previous_filled_cell_coords_in,
                                         previous_filled_cell_height_type_in,
                                         grid_params_in);
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in)
              == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in)
              == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(requires_flood_redirect_indices_in,grid_params_in)
              == field<bool>(requires_flood_redirect_indices_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(requires_connect_redirect_indices_in,grid_params_in)
              == field<bool>(requires_connect_redirect_indices_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] requires_flood_redirect_indices_in;
  delete[] requires_connect_redirect_indices_in;
  delete[] raw_orography_in; delete[] corrected_orography_in;
  delete[] flood_next_cell_lat_index_expected_out;
  delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out;
  delete[] connect_next_cell_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] requires_flood_redirect_indices_expected_out;
  delete[] requires_connect_redirect_indices_expected_out;
  delete new_center_coords_in; delete center_coords_in;
  delete previous_filled_cell_coords_in;
}

TEST_F(BasinEvaluationTest,TestSettingSecondaryRedirectEqualHeights) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  bool* requires_flood_redirect_indices_in = new bool[9*9];
  std::fill_n(requires_flood_redirect_indices_in,9*9,false);
  bool* requires_connect_redirect_indices_in = new bool[9*9];
  std::fill_n(requires_connect_redirect_indices_in,9*9,false);
  double* raw_orography_in = new double[9*9];
  std::fill_n(raw_orography_in,9*9,10.0);
  double* corrected_orography_in = new double[9*9];
  std::fill_n(corrected_orography_in,9*9,10.0);
  int* flood_next_cell_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,7,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,5,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,7,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,5,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  bool* requires_flood_redirect_indices_expected_out = new bool[9*9]
                                     {false,false,false, false, true,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* requires_connect_redirect_indices_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(7,5);
  coords* center_coords_in = new latlon_coords(5,1);
  coords* previous_filled_cell_coords_in = new latlon_coords(0,4);
  height_types previous_filled_cell_height_type_in = flood_height;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_secondary_redirect(flood_next_cell_lat_index_in,flood_next_cell_lon_index_in,
                                         connect_next_cell_lat_index_in,connect_next_cell_lon_index_in,
                                         flood_redirect_lat_index_in,
                                         flood_redirect_lon_index_in,
                                         connect_redirect_lat_index_in,
                                         connect_redirect_lon_index_in,
                                         requires_flood_redirect_indices_in,
                                         requires_connect_redirect_indices_in,
                                         raw_orography_in,
                                         corrected_orography_in,new_center_coords_in,
                                         center_coords_in,previous_filled_cell_coords_in,
                                         previous_filled_cell_height_type_in,
                                         grid_params_in);
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in)
              == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in)
              == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(requires_flood_redirect_indices_in,grid_params_in)
              == field<bool>(requires_flood_redirect_indices_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(requires_connect_redirect_indices_in,grid_params_in)
              == field<bool>(requires_connect_redirect_indices_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] requires_flood_redirect_indices_in;
  delete[] requires_connect_redirect_indices_in;
  delete[] raw_orography_in; delete[] corrected_orography_in;
  delete[] flood_next_cell_lat_index_expected_out;
  delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out;
  delete[] connect_next_cell_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] requires_flood_redirect_indices_expected_out;
  delete[] requires_connect_redirect_indices_expected_out;
  delete new_center_coords_in; delete center_coords_in;
  delete previous_filled_cell_coords_in;
}

TEST_F(BasinEvaluationTest,TestSettingSecondaryRedirectCorrectedHigher) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  int* flood_next_cell_lat_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lat_index_in,9*9,-1);
  int* flood_next_cell_lon_index_in = new int[9*9];
  std::fill_n(flood_next_cell_lon_index_in,9*9,-1);
  int* connect_next_cell_lat_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lat_index_in,9*9,-1);
  int* connect_next_cell_lon_index_in = new int[9*9];
  std::fill_n(connect_next_cell_lon_index_in,9*9,-1);
  int* flood_redirect_lat_index_in = new int[9*9];
  std::fill_n(flood_redirect_lat_index_in,9*9,-1);
  int* flood_redirect_lon_index_in = new int[9*9];
  std::fill_n(flood_redirect_lon_index_in,9*9,-1);
  int* connect_redirect_lat_index_in = new int[9*9];
  std::fill_n(connect_redirect_lat_index_in,9*9,-1);
  int* connect_redirect_lon_index_in = new int[9*9];
  std::fill_n(connect_redirect_lon_index_in,9*9,-1);
  bool* requires_flood_redirect_indices_in = new bool[9*9];
  std::fill_n(requires_flood_redirect_indices_in,9*9,false);
  bool* requires_connect_redirect_indices_in = new bool[9*9];
  std::fill_n(requires_connect_redirect_indices_in,9*9,false);
  double* raw_orography_in = new double[9*9];
  std::fill_n(raw_orography_in,9*9,10.0);
  double* corrected_orography_in = new double[9*9];
  std::fill_n(corrected_orography_in,9*9,11.0);
  int* flood_next_cell_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1, 3,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1, 2,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,3,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,2,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  bool* requires_flood_redirect_indices_expected_out = new bool[9*9]
                                     {false,false,false, false, true,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  bool* requires_connect_redirect_indices_expected_out = new bool[9*9]
                                     {false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(3,2);
  coords* center_coords_in = new latlon_coords(5,1);
  coords* previous_filled_cell_coords_in = new latlon_coords(0,4);
  height_types previous_filled_cell_height_type_in = flood_height;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_secondary_redirect(flood_next_cell_lat_index_in,flood_next_cell_lon_index_in,
                                         connect_next_cell_lat_index_in,connect_next_cell_lon_index_in,
                                         flood_redirect_lat_index_in,
                                         flood_redirect_lon_index_in,
                                         connect_redirect_lat_index_in,
                                         connect_redirect_lon_index_in,
                                         requires_flood_redirect_indices_in,
                                         requires_connect_redirect_indices_in,
                                         raw_orography_in,
                                         corrected_orography_in,new_center_coords_in,
                                         center_coords_in,previous_filled_cell_coords_in,
                                         previous_filled_cell_height_type_in,
                                         grid_params_in);
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in)
              == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in)
              == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(requires_flood_redirect_indices_in,grid_params_in)
              == field<bool>(requires_flood_redirect_indices_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(requires_connect_redirect_indices_in,grid_params_in)
              == field<bool>(requires_connect_redirect_indices_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  delete grid_params_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] requires_flood_redirect_indices_in;
  delete[] requires_connect_redirect_indices_in;
  delete[] raw_orography_in; delete[] corrected_orography_in;
  delete[] flood_next_cell_lat_index_expected_out;
  delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out;
  delete[] connect_next_cell_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] requires_flood_redirect_indices_expected_out;
  delete[] requires_connect_redirect_indices_expected_out;
  delete new_center_coords_in; delete center_coords_in;
  delete previous_filled_cell_coords_in;
}

TEST_F(BasinEvaluationTest,TestSettingRemainingRedirects) {
  auto grid_params_in = new latlon_grid_params(12,12,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* flood_redirect_lat_index_in = new int[12*12]
                                     {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1,  2,-1,-1,
                                     -1,-1,-1, -1, 2,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1, 2,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  5,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 9,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_in = new int[12*12]
                                     {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, 11,-1,-1,
                                     -1,-1,-1, -1, 3,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1, 1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  2,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,11,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_in = new int[12*12];
  std::fill_n(connect_redirect_lat_index_in,12*12,-1);
  int* connect_redirect_lon_index_in = new int[12*12];
  std::fill_n(connect_redirect_lon_index_in,12*12,-1);
  bool* flood_local_redirect_in = new bool[12*12];
  std::fill_n(flood_local_redirect_in,12*12,false);
  bool* connect_local_redirect_in = new bool[12*12];
  std::fill_n(connect_local_redirect_in,12*12,false);
  int* prior_fine_catchments_in = new int[12*12]
                                    {21,21,21, 21,21,21, 24,24,24, 24,24,24,
                                     21,21,21, 21,21,24, 24,24,24, 24,24,24,
                                     22,21,21, 22,21,24, 24,24,23, 23,23,23,
//
                                     22,22,22, 21,21,21, 24,24,24, 23,23,23,
                                     22,22,22, 21,21,21, 24,24,23, 23,23,23,
                                     22,22,22, 26,21,26, 26,24,23, 23,23,23,
//
                                     22,22,22, 26,26,26, 26,22,23, 23,23,23,
                                     22,22,22, 22,26,26, 26,23,23, 22,23,23,
                                     22,22,22, 22,22,22, 22,23,22, 22,25,23,

                                     22,22,22, 22,22,22, 22,22,22, 23,25,23,
                                     22,22,22, 22,22,23, 22,23,23, 22,23,23,
                                     22,22,22, 22,22,23, 23,23,23, 23,23,23};
  int* basin_catchment_numbers_in = new int[12*12]
                                    {1,1,1, 1,1,1, 4,4,4, 4,4,4,
                                     1,1,1, 1,1,4, 4,4,4, 4,4,4,
                                    0,1,1,0,1,4, 4,4,0,0,0,0,
//
                                     7,7,0, 1,1,1, 4,4,4,  0,0,0,
                                     0,0,0, 1,1,1, 4,4,0, 0,0,0,
                                     0,0,0, 6,1,6, 6,4,0, 0,0,0,
//
                                     0,0,0,  6,6,6, 6,0,0, 0,0,0,
                                     0,0,0, 0,6,6, 6,0,0, 0,0,0,
                                      0, 0, 0,  0,0,0, 0,0, 0,  0, 5,0,
//
                                     0,0,0, 0,0, 0,  0, 0, 0, 0, 5,0,
                                     0,0,0, 0,0,0,  0,0,0,  0,0,0,
                                     0,0,0, 0,0,0, 0,0,0, 0,0,0};
  double*  prior_fine_rdirs_in = new double[12*12]
                                    {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0,  1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0, 2.0,
//
                                     -1.0, 8.0, 2.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0, 2.0,
                                      2.0, 4.0, 2.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0, 1.0,
                                      2.0, 8.0, 4.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0, 3.0,-1.0,
//
                                      2.0, 6.0, 2.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,  3.0, 4.0, 2.0,
                                      2.0, 8.0, 2.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0, 3.0, 7.0,
                                      9.0,-1.0, 2.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0, 2.0,
//
                                     -1.0,-1.0, 2.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0, 1.0,
                                     -1.0,-1.0, 2.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0, 1.0,-1.0,
                                     -1.0,-1.0, 0.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,  5.0,-1.0,-1.0};
  bool* requires_flood_redirect_indices_in = new bool[12*12]
                                    {false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false,  true,false,false,
                                     false,false,false, false, true,false, false,false,false, false,false,false,
//
                                     false, true,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false,  true,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false, true,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false};
  bool* requires_connect_redirect_indices_in = new bool[12*12]
                                    {false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false};
  int* flood_next_cell_lat_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1,  5,-1,-1,
                                     -1,-1,-1, -1, 7,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1, 6,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  2,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 2,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1,  1,-1,-1,
                                     -1,-1,-1, -1, 4,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1, 2,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  7,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,14,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lat_index_expected_out = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, 1,-1,-1,
                                     -1,-1,-1, -1, 1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1, 1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 3,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1,  3,-1,-1,
                                     -1,-1,-1, -1, 0,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1, 1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  0,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 3,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out  = new bool[12*12]
                                    {false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false, true,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[12*12]
                                    {false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false};
  int* coarse_catchment_nums_in = new int[4*4]
                                     {11,11,14,14,
                                      12,11,14,13,
                                      12,12,12,13,
                                      12,12,13,13};
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,10));
  basin_catchment_centers_in.push_back(new latlon_coords(0,11));
  basin_catchment_centers_in.push_back(new latlon_coords(4,1));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_remaining_redirects(basin_catchment_centers_in,
                                          prior_fine_rdirs_in,
                                          requires_flood_redirect_indices_in,
                                          requires_connect_redirect_indices_in,
                                          basin_catchment_numbers_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          flood_redirect_lat_index_in,
                                          flood_redirect_lon_index_in,
                                          connect_redirect_lat_index_in,
                                          connect_redirect_lon_index_in,
                                          flood_local_redirect_in,
                                          connect_local_redirect_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_in; delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_in; delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] flood_local_redirect_expected_out; delete[] connect_local_redirect_expected_out;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] coarse_catchment_nums_in; delete[] prior_fine_catchments_in;
  delete[] basin_catchment_numbers_in; delete[] prior_fine_rdirs_in;
  delete[] requires_flood_redirect_indices_in; delete[] requires_connect_redirect_indices_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest,TestSettingRemainingRedirectsTwo) {
  auto grid_params_in = new latlon_grid_params(12,12,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* flood_redirect_lat_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1,  2,-1,-1,
                                     -1,-1,-1, -1, 4,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  5,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 8,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, 11,-1,-1,
                                     -1,-1,-1, -1, 2,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  2,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 8,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1,  2,-1,-1,
                                     -1,-1,-1, -1, 4,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  5,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 8,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, 11,-1,-1,
                                     -1,-1,-1, -1, 2,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  2,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 8,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_in = new bool[12*12];
  std::fill_n(flood_local_redirect_in,12*12,false);
  bool* connect_local_redirect_in = new bool[12*12];
  std::fill_n(connect_local_redirect_in,12*12,false);
  int* prior_fine_catchments_in = new int[12*12]
                                    {21,21,21, 21,21,21, 24,24,24, 24,24,24,
                                     21,21,21, 21,21,24, 24,24,24, 24,24,24,
                                     22,21,21, 22,21,24, 24,24,23, 23,23,23,
//
                                     22,22,22, 21,21,21, 24,24,24, 23,23,23,
                                     22,22,22, 21,21,21, 24,24,23, 23,23,23,
                                     22,22,22, 26,21,26, 26,24,23, 23,23,23,
//
                                     22,22,22, 26,26,26, 26,22,23, 23,23,23,
                                     22,22,22, 22,26,26, 26,23,23, 22,23,23,
                                     22,22,22, 22,22,22, 22,23,22, 22,25,23,

                                     22,22,22, 22,22,22, 22,22,22, 23,25,23,
                                     22,22,22, 22,22,23, 22,23,23, 22,23,23,
                                     22,22,22, 22,22,23, 23,23,23, 23,23,23};
  int* basin_catchment_numbers_in = new int[12*12]
                                    {1,1,1, 1,1,1, 4,4,4, 4,4,4,
                                     1,1,1, 1,1,4, 4,4,4, 4,4,4,
                                     2,1,1, 2,1,4, 4,4,3, 3,3,3,
//
                                     2,2,2, 1,1,1, 4,4,4, 3,3,3,
                                     2,2,2, 1,1,1, 4,4,3, 3,3,3,
                                     2,2,2, 6,1,6, 6,4,3, 3,3,3,
//
                                     2,2,2, 6,6,6, 6,2,3, 3,3,3,
                                     2,2,2, 2,6,6, 6,3,3, 2,3,3,
                                     2,2,2, 2,2,2, 2,3,2, 2,5,3,

                                     2,2,2, 2,2,2, 2,2,2, 3,5,3,
                                     2,2,2, 2,2,3, 2,3,3, 2,3,3,
                                     2,2,2, 2,2,3, 3,3,3, 3,3,3};
  double*  prior_fine_rdirs_in = new double[12*12]
                                    {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0, 0.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,  5.0,-1.0,-1.0};
  bool* requires_flood_redirect_indices_in = new bool[12*12]
                                    {false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false,  true,false,false,
                                     false,false,false, false, true,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false,  true,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false, true,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false};
  bool* requires_connect_redirect_indices_in = new bool[12*12]
                                    {false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false};
  int* flood_next_cell_lat_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1,  4,-1,-1,
                                     -1,-1,-1, -1, 8,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  2,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 3,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, 17,-1,-1,
                                     -1,-1,-1, -1, 1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  7,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,14,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lat_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1,  3,-1,-1,
                                     -1,-1,-1, -1,18,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  9,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,12,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_next_cell_lon_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, 10,-1,-1,
                                     -1,-1,-1, -1, 5,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, 12,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,18,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lat_index_expected_out = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1,  1,-1,-1,
                                     -1,-1,-1, -1, 3,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  3,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 2,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1,  3,-1,-1,
                                     -1,-1,-1, -1, 0,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  0,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 2,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1,  2,-1,-1,
                                     -1,-1,-1, -1, 4,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  5,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 8,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, 11,-1,-1,
                                     -1,-1,-1, -1, 2,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  2,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 8,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out  = new bool[12*12]
                                    {false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false, true,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false,  true,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[12*12]
                                    {false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false};
  int* coarse_catchment_nums_in = new int[4*4]
                                     {11,11,14,14,
                                      12,11,14,13,
                                      12,12,12,13,
                                      12,12,13,13};
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(3,0));
  basin_catchment_centers_in.push_back(new latlon_coords(10,7));
  basin_catchment_centers_in.push_back(new latlon_coords(8,10));
  basin_catchment_centers_in.push_back(new latlon_coords(0,11));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_remaining_redirects(basin_catchment_centers_in,
                                          prior_fine_rdirs_in,
                                          requires_flood_redirect_indices_in,
                                          requires_connect_redirect_indices_in,
                                          basin_catchment_numbers_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          flood_redirect_lat_index_in,
                                          flood_redirect_lon_index_in,
                                          connect_redirect_lat_index_in,
                                          connect_redirect_lon_index_in,
                                          flood_local_redirect_in,
                                          connect_local_redirect_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_in; delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_in; delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] flood_local_redirect_expected_out; delete[] connect_local_redirect_expected_out;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] coarse_catchment_nums_in; delete[] prior_fine_catchments_in;
  delete[] basin_catchment_numbers_in; delete[] prior_fine_rdirs_in;
  delete[] requires_flood_redirect_indices_in; delete[] requires_connect_redirect_indices_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectWrap) {
  auto grid_params_in = new latlon_grid_params(12,12,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* basin_catchment_numbers_in = new int[12*12] {3,3,3, 3,3,8, 8,8,8, 8,1,1,
                                                  3,3,3, 3,8,8, 8,9,9, 9,1,1,
                                                  3,3,3, 8,8,8, 9,9,9, 1,1,1,
//
                                                  3,8,8, 8,8,8, 8,9,9, 1,1,1,
                                                  3,8,6, 6,8,8, 8,8,9, 1,1,1,
                                                  3,6,8, 6,6,6, 8,9,9, 1,1,1,
//
                                                  3,3,8, 5,6,6, 6,9,9, 1,1,1,
                                                  3,3,5, 5,5,6, 6,9,9, 1,1,1,
                                                  3,3,5, 5,5,6, 6,6,9, 1,1,1,

                                                  3,3,5, 5,5,6, 6,6,9, 1,1,1,
                                                  3,3,5, 5,5,6, 6,6,9, 1,1,1,
                                                  3,3,5, 5,5,6, 6,6,9, 1,1,1};
  int* coarse_catchment_nums_in = new int[4*4] {86,55,55,11,
                                                71,42,42,11,
                                                86,38,86,11,
                                                11,11,11,11};
  int* flood_force_merge_lat_index_in = new int[12*12];
  std::fill_n(flood_force_merge_lat_index_in,12*12,-1);
  int* flood_force_merge_lon_index_in = new int[12*12];
  std::fill_n(flood_force_merge_lon_index_in,12*12,-1);
  int* connect_force_merge_lat_index_in = new int[12*12];
  std::fill_n(connect_force_merge_lat_index_in,12*12,-1);
  int* connect_force_merge_lon_index_in = new int[12*12];
  std::fill_n(connect_force_merge_lon_index_in,12*12,-1);
  int* flood_redirect_lat_index_in = new int[12*12];
  std::fill_n(flood_redirect_lat_index_in,12*12,-1);
  int* flood_redirect_lon_index_in = new int[12*12];
  std::fill_n(flood_redirect_lon_index_in,12*12,-1);
  int* connect_redirect_lat_index_in = new int[12*12];
  std::fill_n(connect_redirect_lat_index_in,12*12,-1);
  int* connect_redirect_lon_index_in = new int[12*12];
  std::fill_n(connect_redirect_lon_index_in,12*12,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[12*12];
  std::fill_n(flood_local_redirect_in,12*12,false);
  bool* connect_local_redirect_in = new bool[12*12];
  std::fill_n(connect_local_redirect_in,12*12,false);
  merge_types* merge_points_in = new merge_types[12*12];
  std::fill_n(merge_points_in,12*12,no_merge);
  merge_types* merge_points_expected_out = new merge_types[12*12]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,connection_merge_not_set_flood_merge_as_primary, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge};
  int* flood_force_merge_lat_index_expected_out = new int[12*12]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,-1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,-1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,-1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1, 0,-1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1,-1,-1,-1,-1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[12*12]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1, 9, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[12*12];
  std::fill_n(connect_force_merge_lat_index_expected_out,12*12,-1);
  int* connect_force_merge_lon_index_expected_out = new int[12*12];
  std::fill_n(connect_force_merge_lon_index_expected_out,12*12,-1);
  int* flood_redirect_lat_index_expected_out = new int[12*12]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1, 0, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[12*12]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1, 0, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[12*12]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[12*12]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[12*12]
                                     {false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[12*12]
                                     {false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(0,9);
  coords* center_coords_in = new latlon_coords(1,9);
  coords* previous_filled_cell_coords_in = new latlon_coords(7,8);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(6,2));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectNoWrap) {
  auto grid_params_in = new latlon_grid_params(12,12,true);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,true);
  int* basin_catchment_numbers_in = new int[12*12] {3,3,3, 3,3,8, 8,8,8, 8,1,1,
                                                  3,3,3, 3,8,8, 8,9,9, 9,1,1,
                                                  3,3,3, 8,8,8, 9,9,9, 1,1,1,
//
                                                  3,8,8, 8,8,8, 8,9,9, 1,1,1,
                                                  3,8,6, 6,8,8, 8,8,9, 1,1,1,
                                                  3,6,8, 6,6,6, 8,9,9, 1,1,1,
//
                                                  3,3,8, 5,6,6, 6,9,9, 1,1,1,
                                                  3,3,5, 5,5,6, 6,9,9, 1,1,1,
                                                  3,3,5, 5,5,6, 6,6,9, 1,1,1,

                                                  3,3,5, 5,5,6, 6,6,9, 1,1,1,
                                                  3,3,5, 5,5,6, 6,6,9, 1,1,1,
                                                  3,3,5, 5,5,6, 6,6,9, 1,1,1};
  int* coarse_catchment_nums_in = new int[4*4] {86,55,55,11,
                                                71,42,42,11,
                                                86,38,86,11,
                                                11,11,11,11};
  int* flood_force_merge_lat_index_in = new int[12*12];
  std::fill_n(flood_force_merge_lat_index_in,12*12,-1);
  int* flood_force_merge_lon_index_in = new int[12*12];
  std::fill_n(flood_force_merge_lon_index_in,12*12,-1);
  int* connect_force_merge_lat_index_in = new int[12*12];
  std::fill_n(connect_force_merge_lat_index_in,12*12,-1);
  int* connect_force_merge_lon_index_in = new int[12*12];
  std::fill_n(connect_force_merge_lon_index_in,12*12,-1);
  int* flood_redirect_lat_index_in = new int[12*12];
  std::fill_n(flood_redirect_lat_index_in,12*12,-1);
  int* flood_redirect_lon_index_in = new int[12*12];
  std::fill_n(flood_redirect_lon_index_in,12*12,-1);
  int* connect_redirect_lat_index_in = new int[12*12];
  std::fill_n(connect_redirect_lat_index_in,12*12,-1);
  int* connect_redirect_lon_index_in = new int[12*12];
  std::fill_n(connect_redirect_lon_index_in,12*12,-1);
  height_types previous_filled_cell_height_type_in = flood_height;
  bool* flood_local_redirect_in = new bool[12*12];
  std::fill_n(flood_local_redirect_in,12*12,false);
  bool* connect_local_redirect_in = new bool[12*12];
  std::fill_n(connect_local_redirect_in,12*12,false);
  merge_types* merge_points_in = new merge_types[12*12];
  std::fill_n(merge_points_in,12*12,no_merge);
  merge_types* merge_points_expected_out = new merge_types[12*12]{
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,connection_merge_not_set_flood_merge_as_primary, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
//
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge,
    no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge, no_merge,no_merge,no_merge
  };
  int* flood_force_merge_lat_index_expected_out = new int[12*12]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,-1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,-1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,-1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1, 0,-1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1,-1,-1,-1,-1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_force_merge_lon_index_expected_out = new int[12*12]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1, 9, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_force_merge_lat_index_expected_out = new int[12*12];
  std::fill_n(connect_force_merge_lat_index_expected_out,12*12,-1);
  int* connect_force_merge_lon_index_expected_out = new int[12*12];
  std::fill_n(connect_force_merge_lon_index_expected_out,12*12,-1);
  int* flood_redirect_lat_index_expected_out = new int[12*12]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1, 2, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_redirect_lon_index_expected_out = new int[12*12]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1, 2, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lat_index_expected_out = new int[12*12]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* connect_redirect_lon_index_expected_out = new int[12*12]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  bool* flood_local_redirect_expected_out = new bool[12*12]
                                     {false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false};
  bool* connect_local_redirect_expected_out = new bool[12*12]
                                     {false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
//
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false,
                                      false,false,false, false,false,false, false,false,false, false,false,false};
  coords* new_center_coords_in = new latlon_coords(0,9);
  coords* center_coords_in = new latlon_coords(1,9);
  coords* previous_filled_cell_coords_in = new latlon_coords(7,8);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(8,4));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(8,6));
  basin_catchment_centers_in.push_back(new latlon_coords(6,2));
  basin_catchment_centers_in.push_back(new latlon_coords(4,8));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 flood_force_merge_lat_index_in,
                                                 flood_force_merge_lon_index_in,
                                                 connect_force_merge_lat_index_in,
                                                 connect_force_merge_lon_index_in,
                                                 flood_redirect_lat_index_in,
                                                 flood_redirect_lon_index_in,
                                                 connect_redirect_lat_index_in,
                                                 connect_redirect_lon_index_in,
                                                 flood_local_redirect_in,
                                                 connect_local_redirect_in,
                                                 merge_points_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] merge_points_in; delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestEvaluateBasinsOne) {
  auto grid_params_in = new latlon_grid_params(20,20,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* coarse_catchment_nums_in = new int[4*4] {3,3,2,2,
                                                3,3,2,2,
                                                1,1,1,2,
                                                1,1,1,1};
  double* corrected_orography_in = new double[20*20]
  {10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,
   10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0,
    1.0, 8.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 6.0, 6.0, 5.0, 6.0,10.0,10.0,10.0, 1.0,
   10.0, 3.0, 3.0,10.0,10.0, 7.0, 3.0,10.0,10.0,10.0,10.0, 6.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0,10.0,10.0,
   10.0, 3.0, 3.0,10.0,10.0, 3.0, 3.0, 4.0, 3.0,10.0,10.0,10.0, 2.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0,10.0,
   10.0, 3.0, 3.0, 6.0, 2.0, 1.0,10.0, 2.0, 3.0, 5.0, 3.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0,
    4.0, 4.0, 3.0,10.0, 2.0, 1.0, 2.0, 2.0, 3.0,10.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0,
   10.0, 4.0, 4.0,10.0,10.0, 2.0,10.0, 4.0,10.0,10.0,10.0, 3.0, 2.0, 3.0, 2.0, 3.0, 5.0, 9.0,10.0,10.0,
   10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 5.0, 3.0, 3.0, 3.0, 4.0, 5.0,10.0, 8.0,10.0,
   10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 5.0, 5.0,10.0,10.0,10.0,10.0, 7.0,10.0,
   10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 6.0,10.0,
   10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 5.0,10.0,
   10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 4.0,10.0,
   10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0, 2.0, 3.0, 3.0,10.0,
   10.0,10.0,10.0, 3.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0, 3.0, 3.0,10.0,10.0,
   10.0,10.0,10.0, 2.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 4.0,10.0,10.0,10.0,10.0,
   10.0,10.0,10.0, 3.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 3.0,10.0,10.0,10.0,10.0,
   10.0,10.0,10.0, 2.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0,10.0,10.0,10.0,10.0,
   10.0,10.0,10.0, 2.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 1.0,10.0,10.0,10.0,10.0,
   10.0,10.0,10.0, 1.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 0.0,10.0,10.0,10.0,10.0};

  double* raw_orography_in = new double[20*20]
  {10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,
   10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0,
    1.0, 8.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 6.0, 6.0, 5.0, 6.0,10.0,10.0,10.0, 1.0,
   10.0, 3.0, 3.0,10.0,10.0, 8.0, 7.0,10.0,10.0,10.0,10.0, 6.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0,10.0,10.0,
   10.0, 3.0, 3.0,10.0,10.0, 3.0, 3.0, 4.0, 3.0,10.0,10.0,10.0, 2.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0,10.0,
   10.0, 3.0, 3.0, 6.0, 2.0, 1.0,10.0, 2.0, 3.0,10.0, 3.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0,
    4.0, 4.0, 3.0,10.0, 2.0, 1.0, 2.0, 2.0, 3.0,10.0, 3.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0,10.0,10.0,
   10.0, 4.0, 4.0,10.0,10.0, 2.0,10.0, 4.0,10.0,10.0,10.0, 3.0, 2.0, 3.0, 2.0, 3.0, 5.0, 9.0,10.0,10.0,
   10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 5.0, 3.0, 3.0, 3.0, 4.0, 5.0,10.0, 8.0,10.0,
   10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 5.0, 5.0,10.0,10.0,10.0,10.0, 7.0,10.0,
   10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 6.0,10.0,
   10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 5.0,10.0,
   10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 4.0,10.0,
   10.0,10.0,10.0,10.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0, 2.0, 3.0, 3.0,10.0,
   10.0,10.0,10.0, 3.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0, 3.0, 3.0,10.0,10.0,
   10.0,10.0,10.0, 2.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 4.0,10.0,10.0,10.0,10.0,
   10.0,10.0,10.0, 3.0, 3.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 3.0,10.0,10.0,10.0,10.0,
   10.0,10.0,10.0, 2.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 2.0,10.0,10.0,10.0,10.0,
   10.0,10.0,10.0, 2.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 1.0,10.0,10.0,10.0,10.0,
   10.0,10.0,10.0, 1.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0, 0.0,10.0,10.0,10.0,10.0};

  bool* minima_in = new bool[20*20]
  {false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   true, false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,true, false,false,
    false,false,false,false,
   false,false,false,false,false,true, false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,true, false,
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
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,true, false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false};
  double* prior_fine_rdirs_in = new double[20*20] {
  1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,3.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,3.0,3.0,2.0,
  2.0,1.0,2.0,1.0,1.0,2.0,1.0,1.0,1.0,3.0,3.0,3.0,2.0,3.0,2.0,1.0,1.0,3.0,3.0,3.0,
  5.0,4.0,1.0,1.0,3.0,2.0,1.0,1.0,1.0,1.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,6.0,6.0,6.0,
  8.0,7.0,4.0,4.0,3.0,2.0,1.0,1.0,2.0,1.0,6.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,
  9.0,8.0,7.0,3.0,3.0,2.0,1.0,2.0,1.0,1.0,3.0,6.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0,
  9.0,8.0,7.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,3.0,3.0,6.0,5.0,4.0,4.0,1.0,1.0,1.0,3.0,
  9.0,8.0,7.0,6.0,6.0,5.0,4.0,4.0,4.0,4.0,6.0,6.0,9.0,8.0,7.0,1.0,1.0,4.0,4.0,6.0,
  9.0,9.0,8.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0,9.0,9.0,8.0,7.0,5.0,4.0,4.0,7.0,7.0,9.0,
  9.0,9.0,8.0,7.0,9.0,8.0,7.0,8.0,7.0,7.0,9.0,9.0,8.0,7.0,8.0,7.0,7.0,7.0,2.0,1.0,
  9.0,9.0,8.0,9.0,9.0,8.0,7.0,7.0,7.0,9.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0,7.0,2.0,1.0,
  1.0,1.0,9.0,9.0,9.0,8.0,7.0,7.0,7.0,9.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0,3.0,2.0,1.0,
  1.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,9.0,8.0,3.0,3.0,2.0,1.0,3.0,2.0,1.0,
  1.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,3.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,
  4.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,3.0,6.0,6.0,6.0,5.0,4.0,4.0,4.0,4.0,
  7.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,3.0,3.0,3.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0,
  7.0,3.0,6.0,5.0,4.0,1.0,1.0,1.0,1.0,1.0,3.0,3.0,3.0,3.0,9.0,8.0,7.0,7.0,7.0,7.0,
  3.0,3.0,3.0,2.0,1.0,4.0,4.0,4.0,4.0,4.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,7.0,7.0,
  3.0,3.0,3.0,2.0,1.0,7.0,7.0,7.0,7.0,7.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,1.0,7.0,
  3.0,3.0,3.0,2.0,1.0,1.0,7.0,7.0,7.0,7.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,1.0,3.0,
  6.0,6.0,6.0,0.0,4.0,4.0,4.0,7.0,7.0,7.0,6.0,6.0,6.0,6.0,6.0,0.0,4.0,4.0,4.0,6.0};
  int* prior_fine_catchments_in = new int[20*20] {
  11,11,11,11,11,11,13,13,12,12,12,12,12,12,12,12,12,11,11,11,
  11,11,11,11,11,13,13,13,13,12,12,12,12,12,12,12,12,11,11,11,
  11,11,11,11,13,13,13,13,13,13,12,12,12,12,12,12,12,11,11,11,
  11,11,11,11,13,13,13,13,13,13,12,12,12,12,12,12,12,12,11,11,
  11,11,11,13,13,13,13,13,13,13,12,12,12,12,12,12,12,14,14,11,
  11,11,11,13,13,13,13,13,13,13,12,12,12,12,12,12,14,14,14,11,
  11,11,11,13,13,13,13,13,13,13,12,12,12,12,12,14,14,14,14,11,
  11,11,11,13,13,13,13,13,13,13,12,12,12,12,14,14,14,14,14,11,
  11,11,11,11,13,13,13,13,13,13,12,12,12,12,14,14,14,14,15,15,
  11,11,11,13,13,13,13,13,13,12,12,12,12,12,12,14,14,14,15,15,
  15,15,13,13,13,13,13,13,13,12,12,12,12,12,12,12,14,15,15,15,
  15,16,16,16,16,16,16,16,12,12,12,12,12,15,15,15,15,15,15,15,
  15,16,16,16,16,16,16,16, 4,12,12,12,15,15,15,15,15,15,15,15,
  15,16,16,16,16,16,16, 4, 4,12,12, 9,15,15,15,15,15,15,15,15,
  15,16,16,16,16,16, 4, 4, 4, 4,10, 9, 9,15,15,15,15,15,15,15,
  15, 4,16,16,16, 4, 4, 4, 4, 4, 7,10, 9, 9,15,15,15,15,15,15,
   5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7,10, 9, 9, 9, 9, 9,15,15,
   2, 5, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7,10, 9, 9, 9, 8, 6,15,
   2, 2, 5, 4, 3, 1, 4, 4, 4, 4, 7, 7, 7, 7,10, 9, 8, 6, 6, 2,
   2, 2, 2, 0, 1, 1, 1, 4, 4, 4, 7, 7, 7, 7, 7, 0, 6, 6, 6, 2 };
  double* cell_areas_in = new double[20*20];
  std::fill_n(cell_areas_in,20*20,1.0);
  double* connection_volume_thresholds_in = new double[20*20];
  std::fill_n(connection_volume_thresholds_in,20*20,0.0);
  double* flood_volume_thresholds_in = new double[20*20];
  std::fill_n(flood_volume_thresholds_in,20*20,0.0);
  int* flood_next_cell_lat_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lat_index_in,20*20,-1);
  int* flood_next_cell_lon_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lon_index_in,20*20,-1);
  int* connect_next_cell_lat_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lat_index_in,20*20,-1);
  int* connect_next_cell_lon_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lon_index_in,20*20,-1);
  int* flood_force_merge_lat_index_in = new int[20*20];
  std::fill_n(flood_force_merge_lat_index_in,20*20,-1);
  int* flood_force_merge_lon_index_in = new int[20*20];
  std::fill_n(flood_force_merge_lon_index_in,20*20,-1);
  int* connect_force_merge_lat_index_in = new int[20*20];
  std::fill_n(connect_force_merge_lat_index_in,20*20,-1);
  int* connect_force_merge_lon_index_in = new int[20*20];
  std::fill_n(connect_force_merge_lon_index_in,20*20,-1);
  int* flood_redirect_lat_index_in = new int[20*20];
  std::fill_n(flood_redirect_lat_index_in,20*20,-1);
  int* flood_redirect_lon_index_in = new int[20*20];
  std::fill_n(flood_redirect_lon_index_in,20*20,-1);
  int* connect_redirect_lat_index_in = new int[20*20];
  std::fill_n(connect_redirect_lat_index_in,20*20,-1);
  int* connect_redirect_lon_index_in = new int[20*20];
  std::fill_n(connect_redirect_lon_index_in,20*20,-1);
  bool* flood_local_redirect_in = new bool[20*20];
  std::fill_n(flood_local_redirect_in,20*20,false);
  bool* connect_local_redirect_in = new bool[20*20];
  std::fill_n(connect_local_redirect_in,20*20,false);
  int* additional_flood_redirect_lat_index_in = new int[20*20];
  std::fill_n(additional_flood_redirect_lat_index_in,20*20,-1);
  int* additional_flood_redirect_lon_index_in = new int[20*20];
  std::fill_n(additional_flood_redirect_lon_index_in,20*20,-1);
  int* additional_connect_redirect_lat_index_in = new int[20*20];
  std::fill_n(additional_connect_redirect_lat_index_in,20*20,-1);
  int* additional_connect_redirect_lon_index_in = new int[20*20];
  std::fill_n(additional_connect_redirect_lon_index_in,20*20,-1);
  bool* additional_flood_local_redirect_in = new bool[20*20];
  std::fill_n(additional_flood_local_redirect_in,20*20,false);
  bool* additional_connect_local_redirect_in = new bool[20*20];
  std::fill_n(additional_connect_local_redirect_in,20*20,false);
  merge_types* merge_points_in = new merge_types[20*20];
  std::fill_n(merge_points_in,20*20,no_merge);
  double* flood_volume_thresholds_expected_out = new double[20*20] {
 -1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1,
 -1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   5.0,
  0.0, 262.0,5.0, -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,    111.0,111.0,56.0,111.0,-1,  -1,     -1,   2.0,
 -1,   5.0,  5.0, -1,   -1,  340.0,262.0,-1, -1,  -1, -1,  111.0,1.0,  1.0,  56.0,56.0,-1,   -1,     -1,   -1,
 -1,   5.0,  5.0, -1,   -1,  10.0, 10.0, 38.0,10.0,-1,-1, -1,    0.0,  1.0,  1.0, 26.0, 56.0,-1,     -1,   -1,
 -1,   5.0,  5.0, 186.0, 2.0,2.0,  -1,    10.0,10.0,-1, 1.0,6.0,  1.0,  0.0,  1.0, 26.0, 26.0,111.0, -1,   -1,
  16.0,16.0, 16.0,-1,    2.0,0.0,  2.0,  2.0, 10.0,-1, 1.0,0.0,  0.0,  1.0,  1.0, 1.0,  26.0,56.0,   -1,   -1,
 -1,   46.0, 16.0,-1,   -1,  2.0, -1,    23.0,-1,  -1, -1,  1.0,  0.0,  1.0,  1.0, 1.0,  56.0,-1,    -1,   -1,
 -1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1,  56.0, 1.0,  1.0,  1.0, 26.0, 56.0,-1,    -1,   -1,
 -1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,    56.0, 56.0,-1,  -1,   -1,  -1,     -1,   -1,
 -1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1,
 -1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1,
 -1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1,
 -1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,   0.0,  3.0, 3.0,    10.0,-1,
 -1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,   0.0,  3.0, 3.0,   -1,   -1,
 -1,  -1,   -1,    1.0, -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1,
 -1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1,
 -1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1,
 -1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1,
 -1,  -1,   -1,   -1,   -1, -1,   -1,    -1,  -1,  -1, -1, -1,   -1,   -1,   -1,  -1,   -1,  -1,     -1,   -1 };
  double* connection_volume_thresholds_expected_out = new double[20*20]{
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  186.0, 23.0,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, 56.0,-1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
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
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.setup_fields(minima_in,
                          raw_orography_in,
                          corrected_orography_in,
                          cell_areas_in,
                          connection_volume_thresholds_in,
                          flood_volume_thresholds_in,
                          prior_fine_rdirs_in,
                          prior_fine_catchments_in,
                          coarse_catchment_nums_in,
                          flood_next_cell_lat_index_in,
                          flood_next_cell_lon_index_in,
                          connect_next_cell_lat_index_in,
                          connect_next_cell_lon_index_in,
                          flood_force_merge_lat_index_in,
                          flood_force_merge_lon_index_in,
                          connect_force_merge_lat_index_in,
                          connect_force_merge_lon_index_in,
                          flood_redirect_lat_index_in,
                          flood_redirect_lon_index_in,
                          connect_redirect_lat_index_in,
                          connect_redirect_lon_index_in,
                          additional_flood_redirect_lat_index_in,
                          additional_flood_redirect_lon_index_in,
                          additional_connect_redirect_lat_index_in,
                          additional_connect_redirect_lon_index_in,
                          flood_local_redirect_in,
                          connect_local_redirect_in,
                          additional_flood_local_redirect_in,
                          additional_connect_local_redirect_in,
                          merge_points_in,
                          grid_params_in,
                          coarse_grid_params_in);
  bool* landsea_in = new bool[20*20] {
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
   false,false,false, true,false,false,false,false,false,false,false,false,false,false,false, true,
    false,false,false,false};
  bool* true_sinks_in = new bool[20*20];
  fill_n(true_sinks_in,20*20,false);
  int* next_cell_lat_index_in = new int[20*20];
  int* next_cell_lon_index_in = new int[20*20];
  short* sinkless_rdirs_out = new short[20*20];
  int* catchment_nums_in = new int[20*20];
  reverse_priority_cell_queue q;
  auto alg4 = new sink_filling_algorithm_4_latlon();
  alg4->setup_flags(false,true,false,false);
  alg4->setup_fields(corrected_orography_in,
                     landsea_in,
                     true_sinks_in,
                     next_cell_lat_index_in,
                     next_cell_lon_index_in,
                     grid_params_in,
                     sinkless_rdirs_out,
                     catchment_nums_in);
  basin_eval.setup_sink_filling_algorithm(alg4);
  basin_eval.evaluate_basins();
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in)
              == field<double>(flood_volume_thresholds_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in)
              == field<double>(connection_volume_thresholds_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in)
              == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in)
              == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in; delete alg4;
  delete[] coarse_catchment_nums_in; delete[] corrected_orography_in;
  delete[] raw_orography_in; delete[] minima_in;
  delete[] prior_fine_rdirs_in; delete[] prior_fine_catchments_in;
  delete[] connection_volume_thresholds_in; delete[] flood_volume_thresholds_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] additional_flood_redirect_lat_index_in;
  delete[] additional_flood_redirect_lon_index_in;
  delete[] additional_connect_redirect_lat_index_in;
  delete[] additional_connect_redirect_lon_index_in;
  delete[] additional_flood_local_redirect_in;
  delete[] additional_connect_local_redirect_in;
  delete[] merge_points_in; delete[] flood_volume_thresholds_expected_out;
  delete[] connection_volume_thresholds_expected_out;
  delete[] flood_next_cell_lat_index_expected_out;
  delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out;
  delete[] connect_next_cell_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] landsea_in;
  delete[] true_sinks_in;
  delete[] next_cell_lat_index_in;
  delete[] next_cell_lon_index_in;
  delete[] sinkless_rdirs_out;
  delete[] catchment_nums_in;
  delete[] cell_areas_in;

}


TEST_F(BasinEvaluationTest, TestEvaluateBasinsTwo) {
  auto grid_params_in = new latlon_grid_params(20,20,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* coarse_catchment_nums_in = new int[4*4] {3,3,2,2,
                                                3,3,2,2,
                                                1,1,1,2,
                                                1,1,1,1};
  double* corrected_orography_in = new double[20*20]
  {20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 1.0,10.0,20.0, 9.0, 20.0,20.0,20.0, 1.0,16.0, 20.0,20.0, 1.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 2.0,11.0,20.0, 8.0, 20.0,20.0,20.0, 2.0,15.0, 20.0,20.0, 2.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   18.0, 3.0,12.0,20.0, 7.0, 20.0,20.0,20.0, 3.0,14.0, 20.0,20.0, 3.0,18.0, 0.9,  0.8, 0.0, 0.7, 0.8, 0.9,
   20.0, 4.0,13.0, 6.0, 5.0, 20.0,20.0,20.0, 4.0,13.0,  5.0, 5.0, 4.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
//
   20.0, 5.0,14.0,20.0, 4.0, 20.0,20.0,20.0, 5.0,12.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 6.0,15.0,20.0, 3.0, 20.0,20.0,20.0, 6.0,11.0, 20.0,20.0, 7.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 8.0,16.0,20.0, 2.0, 20.0,20.0,20.0, 7.0,10.0, 20.0,20.0, 8.0,20.0,20.0, 20.0,20.0, 5.0,20.0,20.0,
   20.0, 9.0,17.0,20.0, 1.0, 20.0,20.0,20.0, 8.0, 9.0, 20.0,20.0, 9.0,20.0,20.0, 20.0, 4.0, 6.0,20.0,20.0,
   20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,  3.0, 7.0,20.0,20.0,20.0,
//
   20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0, 2.0,  8.0,20.0,20.0,20.0,20.0,
   20.0, 1.0,16.0,20.0, 2.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 1.0, 9.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 2.0,15.0,20.0, 3.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,10.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 3.0,14.0,20.0, 4.0, 20.0,20.0, 1.0,20.0, 5.0, 20.0,20.0,20.0, 0.9,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 4.0,13.0, 6.0, 5.0, 20.0,20.0, 2.0, 4.0, 3.0, 20.0,20.0, 0.8,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
//
   20.0, 5.0,12.0,20.0, 7.0, 20.0,20.0, 3.0,20.0, 2.0, 20.0,20.0, 0.7,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 6.0,11.0,20.0, 8.0, 20.0,20.0, 4.0,20.0, 1.0, 20.0,20.0, 0.6,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 7.0,10.0,20.0, 9.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0, 0.5,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 8.0, 9.0,20.0,10.0, 18.0, 0.9, 0.8,20.0,20.0, 20.0, 0.4,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 0.0, 0.2,  0.3,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0};

  double* raw_orography_in = new double[20*20]
  {20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 1.0,10.0,20.0, 9.0, 20.0,20.0,20.0, 1.0,16.0, 20.0,20.0, 1.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 2.0,11.0,20.0, 8.0, 20.0,20.0,20.0, 2.0,15.0, 20.0,20.0, 2.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   18.0, 3.0,12.0,20.0, 7.0, 20.0,20.0,20.0, 3.0,14.0, 20.0,20.0, 3.0,18.0, 0.9,  0.8, 0.0, 0.7, 0.8, 0.9,
   20.0, 4.0,13.0,20.0, 5.0, 20.0,20.0,20.0, 4.0,13.0,  5.0, 5.0, 4.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
//
   20.0, 5.0,14.0,20.0, 4.0, 20.0,20.0,20.0, 5.0,12.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 6.0,15.0,20.0, 3.0, 20.0,20.0,20.0, 6.0,11.0, 20.0,20.0, 7.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 8.0,16.0,20.0, 2.0, 20.0,20.0,20.0, 7.0,10.0, 20.0,20.0, 8.0,20.0,20.0, 20.0,20.0, 5.0,20.0,20.0,
   20.0, 9.0,17.0,20.0, 1.0, 20.0,20.0,20.0, 8.0, 9.0, 20.0,20.0, 9.0,20.0,20.0, 20.0, 4.0, 6.0,20.0,20.0,
   20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,  3.0, 7.0,20.0,20.0,20.0,
//
   20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0, 2.0,  8.0,20.0,20.0,20.0,20.0,
   20.0, 1.0,16.0,20.0, 2.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 1.0, 9.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 2.0,15.0,20.0, 3.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,10.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 3.0,14.0,20.0, 4.0, 20.0,20.0, 1.0,20.0, 5.0, 20.0,20.0,20.0, 0.9,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 4.0,13.0,20.0, 5.0, 20.0,20.0, 2.0, 4.0, 3.0, 20.0,20.0, 0.8,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
//
   20.0, 5.0,12.0,20.0, 7.0, 20.0,20.0, 3.0,20.0, 2.0, 20.0,20.0, 0.7,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 6.0,11.0,20.0, 8.0, 20.0,20.0, 4.0,20.0, 1.0, 20.0,20.0, 0.6,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 7.0,10.0,20.0, 9.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0, 0.5,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 8.0, 9.0,20.0,10.0, 18.0, 0.9, 0.8,20.0,20.0, 20.0, 0.4,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 0.0, 0.2,  0.3,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0};

  bool* minima_in = new bool[20*20]
  {false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false, true,false,false,false,false,false,false, true,false,false,false, true,false,false,false,
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
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false, true,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false, true,false,false, true,false,false,false,false,false,false,false,false, true,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false, true,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false, true,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false};
  double* prior_fine_rdirs_in = new double[20*20] {
  1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,3.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,3.0,3.0,2.0,
  2.0,1.0,2.0,1.0,1.0,2.0,1.0,1.0,1.0,3.0,3.0,3.0,2.0,3.0,2.0,1.0,1.0,3.0,3.0,3.0,
  5.0,4.0,1.0,1.0,3.0,2.0,1.0,1.0,1.0,1.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,6.0,6.0,6.0,
  8.0,7.0,4.0,4.0,3.0,2.0,1.0,1.0,2.0,1.0,6.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,
  9.0,8.0,7.0,3.0,3.0,2.0,1.0,2.0,1.0,1.0,3.0,6.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0,
  9.0,8.0,7.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,3.0,3.0,6.0,5.0,4.0,4.0,1.0,1.0,1.0,3.0,
  9.0,8.0,7.0,6.0,6.0,5.0,4.0,4.0,4.0,4.0,6.0,6.0,9.0,8.0,7.0,1.0,1.0,4.0,4.0,6.0,
  9.0,9.0,8.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0,9.0,9.0,8.0,7.0,5.0,4.0,4.0,7.0,7.0,9.0,
  9.0,9.0,8.0,7.0,9.0,8.0,7.0,8.0,7.0,7.0,9.0,9.0,8.0,7.0,8.0,7.0,7.0,7.0,2.0,1.0,
  9.0,9.0,8.0,9.0,9.0,8.0,7.0,7.0,7.0,9.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0,7.0,2.0,1.0,
  1.0,1.0,9.0,9.0,9.0,8.0,7.0,7.0,7.0,9.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0,3.0,2.0,1.0,
  1.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,9.0,8.0,3.0,3.0,2.0,1.0,3.0,2.0,1.0,
  1.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,3.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,
  4.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,3.0,6.0,6.0,6.0,5.0,4.0,4.0,4.0,4.0,
  7.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,3.0,3.0,3.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0,
  7.0,3.0,6.0,5.0,4.0,1.0,1.0,1.0,1.0,1.0,3.0,3.0,3.0,3.0,9.0,8.0,7.0,7.0,7.0,7.0,
  3.0,3.0,3.0,2.0,1.0,4.0,4.0,4.0,4.0,4.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,7.0,7.0,
  3.0,3.0,3.0,2.0,1.0,7.0,7.0,7.0,7.0,7.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,1.0,7.0,
  3.0,3.0,3.0,2.0,1.0,1.0,7.0,7.0,7.0,7.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,1.0,3.0,
  6.0,6.0,6.0,0.0,4.0,4.0,4.0,7.0,7.0,7.0,6.0,6.0,6.0,6.0,6.0,0.0,4.0,4.0,4.0,6.0};
  int* prior_fine_catchments_in = new int[20*20] {
  11,11,11,11,11,11,13,13,12,12,12,12,12,12,12,12,12,11,11,11,
  11,11,11,11,11,13,13,13,13,12,12,12,12,12,12,12,12,11,11,11,
  11,11,11,11,13,13,13,13,13,13,12,12,12,12,12,12,12,11,11,11,
  11,11,11,11,13,13,13,13,13,13,12,12,12,12,12,12,12,12,11,11,
  11,11,11,13,13,13,13,13,13,13,12,12,12,12,12,12,12,14,14,11,
  11,11,11,13,13,13,13,13,13,13,12,12,12,12,12,12,14,14,14,11,
  11,11,11,13,13,13,13,13,13,13,12,12,12,12,12,14,14,14,14,11,
  11,11,11,13,13,13,13,13,13,13,12,12,12,12,14,14,14,14,14,11,
  11,11,11,11,13,13,13,13,13,13,12,12,12,12,14,14,14,14,15,15,
  11,11,11,13,13,13,13,13,13,12,12,12,12,12,12,14,14,14,15,15,
  15,15,13,13,13,13,13,13,13,12,12,12,12,12,12,12,14,15,15,15,
  15,16,16,16,16,16,16,16,12,12,12,12,12,15,15,15,15,15,15,15,
  15,16,16,16,16,16,16,16, 4,12,12,12,15,15,15,15,15,15,15,15,
  15,16,16,16,16,16,16, 4, 4,12,12, 9,15,15,15,15,15,15,15,15,
  15,16,16,16,16,16, 4, 4, 4, 4,10, 9, 9,15,15,15,15,15,15,15,
  15, 4,16,16,16, 4, 4, 4, 4, 4, 7,10, 9, 9,15,15,15,15,15,15,
   5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7,10, 9, 9, 9, 9, 9,15,15,
   2, 5, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7,10, 9, 9, 9, 8, 6,15,
   2, 2, 5, 4, 3, 1, 4, 4, 4, 4, 7, 7, 7, 7,10, 9, 8, 6, 6, 2,
   2, 2, 2, 0, 1, 1, 1, 4, 4, 4, 7, 7, 7, 7, 7, 0, 6, 6, 6, 2 };
  double* cell_areas_in = new double[20*20];
  std::fill_n(cell_areas_in,20*20,1.0);
  double* connection_volume_thresholds_in = new double[20*20];
  std::fill_n(connection_volume_thresholds_in,20*20,0.0);
  double* flood_volume_thresholds_in = new double[20*20];
  std::fill_n(flood_volume_thresholds_in,20*20,0.0);
  int* flood_next_cell_lat_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lat_index_in,20*20,-1);
  int* flood_next_cell_lon_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lon_index_in,20*20,-1);
  int* connect_next_cell_lat_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lat_index_in,20*20,-1);
  int* connect_next_cell_lon_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lon_index_in,20*20,-1);
  int* flood_force_merge_lat_index_in = new int[20*20];
  std::fill_n(flood_force_merge_lat_index_in,20*20,-1);
  int* flood_force_merge_lon_index_in = new int[20*20];
  std::fill_n(flood_force_merge_lon_index_in,20*20,-1);
  int* connect_force_merge_lat_index_in = new int[20*20];
  std::fill_n(connect_force_merge_lat_index_in,20*20,-1);
  int* connect_force_merge_lon_index_in = new int[20*20];
  std::fill_n(connect_force_merge_lon_index_in,20*20,-1);
  int* flood_redirect_lat_index_in = new int[20*20];
  std::fill_n(flood_redirect_lat_index_in,20*20,-1);
  int* flood_redirect_lon_index_in = new int[20*20];
  std::fill_n(flood_redirect_lon_index_in,20*20,-1);
  int* connect_redirect_lat_index_in = new int[20*20];
  std::fill_n(connect_redirect_lat_index_in,20*20,-1);
  int* connect_redirect_lon_index_in = new int[20*20];
  std::fill_n(connect_redirect_lon_index_in,20*20,-1);
  bool* flood_local_redirect_in = new bool[20*20];
  std::fill_n(flood_local_redirect_in,20*20,false);
  bool* connect_local_redirect_in = new bool[20*20];
  std::fill_n(connect_local_redirect_in,20*20,false);
  int* additional_flood_redirect_lat_index_in = new int[20*20];
  std::fill_n(additional_flood_redirect_lat_index_in,20*20,-1);
  int* additional_flood_redirect_lon_index_in = new int[20*20];
  std::fill_n(additional_flood_redirect_lon_index_in,20*20,-1);
  int* additional_connect_redirect_lat_index_in = new int[20*20];
  std::fill_n(additional_connect_redirect_lat_index_in,20*20,-1);
  int* additional_connect_redirect_lon_index_in = new int[20*20];
  std::fill_n(additional_connect_redirect_lon_index_in,20*20,-1);
  bool* additional_flood_local_redirect_in = new bool[20*20];
  std::fill_n(additional_flood_local_redirect_in,20*20,false);
  bool* additional_connect_local_redirect_in = new bool[20*20];
  std::fill_n(additional_connect_local_redirect_in,20*20,false);
  merge_types* merge_points_in = new merge_types[20*20];
  std::fill_n(merge_points_in,20*20,no_merge);
  double* flood_volume_thresholds_expected_out = new double[20*20] {
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0, 1.0,51.0,-1.0,57.0, -1.0,-1.0,-1.0, 1.0,216.0,   -1.0,-1.0, 1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0, 3.0,61.0,-1.0,33.0, -1.0,-1.0,-1.0, 3.0,164.0,   -1.0,-1.0, 3.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0, 6.0,80.0,-1.0,26.0, -1.0,-1.0,-1.0, 6.0,139.0,   -1.0,-1.0, 6.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,10.0,100.0,-1.0,15.0, -1.0,-1.0,-1.0, 10.0,115.0, 16.0,10.0, 10.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
//
   -1.0,15.0,121.0,-1.0,10.0, -1.0,-1.0,-1.0, 15.0,92.0, -1.0,-1.0,23.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,27.0,143.0,-1.0, 6.0, -1.0,-1.0,-1.0, 21.0,66.0, -1.0,-1.0,31.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,34.0,166.0,-1.0, 3.0, -1.0,-1.0,-1.0, 28.0,55.0, -1.0,-1.0,40.0,-1.0,-1.0, -1.0,-1.0,15.0,-1.0,-1.0,
   -1.0,42.0,190.0,-1.0, 1.0, -1.0,-1.0,-1.0, 36.0, 45.0, -1.0,-1.0,70.0,-1.0,-1.0, -1.0,10.0,21.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,  6.0,28.0,-1.0,-1.0,-1.0,
//
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0, 3.0,  36.0,-1.0,-1.0,-1.0,-1.0,
   -1.0, 1.0,182.0,-1.0, 1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, 1.0,45.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0, 3.0,134.0,-1.0, 3.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0, 6.0,111.0,-1.0, 6.0, -1.0,-1.0, 1.0,-1.0,23.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,10.0,89.0,-1.0,10.0, -1.0,-1.0,  3.0, 6.0, 6.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
//
   -1.0,15.0,68.0,-1.0,19.0, -1.0,-1.0, 6.0,-1.0, 3.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,21.0,66.0,-1.0,25.0, -1.0,-1.0,14.0,-1.0, 1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,28.0,55.0,-1.0,32.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,36.0,45.0,-1.0,48.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0 };
  double* connection_volume_thresholds_expected_out = new double[20*20]{
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1,20.0,-1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1,14.0,-1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* flood_next_cell_lat_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  2,  2, -1,  2,  -1, -1, -1,  2,  3,  -1, -1,  2, -1, -1,  -1, -1, -1, -1, -1,
    -1,  3,  3, -1,  1,  -1, -1, -1,  3,  1,  -1, -1,  3, -1, -1,  -1, -1, -1, -1, -1,
    -1,  4,  4, -1,  2,  -1, -1, -1,  4,  2,  -1, -1,  4, -1, -1,  -1, -1, -1, -1, -1,
    -1,  5,  5, -1,  4,  -1, -1, -1,  5,  3,   5,  4,  4, -1, -1,  -1, -1, -1, -1, -1,
    //
    -1,  6,  6, -1,  4,  -1, -1, -1,  6,  4,  -1, -1,  6, -1, -1,  -1, -1, -1, -1, -1,
    -1,  7,  7, -1,  5,  -1, -1, -1,  7,  4,  -1, -1,  7, -1, -1,  -1, -1, -1, -1, -1,
    -1,  8,  8, -1,  6,  -1, -1, -1,  8,  6,  -1, -1,  8, -1, -1,  -1, -1,  8, -1, -1,
    -1,  1,  3, -1,  7,  -1, -1, -1,  8,  7,  -1, -1,  5, -1, -1,  -1,  7,  9, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,   8, 10, -1, -1, -1,
    //
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1,  9,  11, -1, -1, -1, -1,
    -1, 12, 18, -1, 12,  -1, -1, -1, -1, -1,  -1, -1, -1, 10, 13,  -1, -1, -1, -1, -1,
    -1, 13, 11, -1, 13,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, 14, 12, -1, 14,  -1, -1, 14, -1, 18,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, 15, 13, -1, 14,  -1, -1, 15, 16, 13,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    //
    -1, 16, 14, -1, 16,  -1, -1, 14, -1, 14,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, 17, 14, -1, 17,  -1, -1, 13, -1, 15,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, 18, 16, -1, 18,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, 18, 17, -1, 15,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* flood_next_cell_lon_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  1,  -1, -1, -1,  8, 14,  -1, -1, 12, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  4,  -1, -1, -1,  8,  9,  -1, -1, 12, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  4,  -1, -1, -1,  8,  9,  -1, -1, 12, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  3,  -1, -1, -1,  8,  9,  12, 10, 11, -1, -1,  -1, -1, -1, -1, -1,
    //
    -1,  1,  2, -1,  4,  -1, -1, -1,  8,  9,  -1, -1, 12, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  4,  -1, -1, -1,  8, 10,  -1, -1, 12, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  4,  -1, -1, -1,  8,  9,  -1, -1, 12, -1, -1,  -1, -1, 17, -1, -1,
    -1,  2, 19, -1,  4,  -1, -1, -1,  9,  9,  -1, -1,  9, -1, -1,  -1, 17, 16, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  16, 15, -1, -1, -1,
    //
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, 15,  14, -1, -1, -1, -1,
    -1,  1,  6, -1,  4,  -1, -1, -1, -1, -1,  -1, -1, -1, 14, 13,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  4,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  4,  -1, -1,  7, -1,  7,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  3,  -1, -1,  7,  7,  7,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    //
    -1,  1,  2, -1,  4,  -1, -1,  8, -1,  9,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  3, -1,  4,  -1, -1,  9, -1,  9,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  4,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  2,  2, -1,  2,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* connect_next_cell_lat_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1,  3, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
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
    -1, -1, -1,  15, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1,-1,
    //
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* connect_next_cell_lon_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1,  4, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
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
    -1, -1, -1,  4, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    //
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* flood_redirect_lat_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1,  1, -1, -1, -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  2, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  int* flood_redirect_lon_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1,  1, -1, -1, -1, -1,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  3, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
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
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false,  true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false,  true, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false,  true, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false,  true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false };
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
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,  no_merge,   no_merge,   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   connection_merge_not_set_flood_merge_as_primary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   connection_merge_not_set_flood_merge_as_primary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   connection_merge_not_set_flood_merge_as_primary,  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   connection_merge_not_set_flood_merge_as_primary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,  };

  int* flood_force_merge_lat_index_expected_out = new int[20*20] {

    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1,  4, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,  4, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, 15, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, 14,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* flood_force_merge_lon_index_expected_out = new int[20*20] {
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1,  3, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,  8, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1,  9, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1,  1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
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
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.setup_fields(minima_in,
                          raw_orography_in,
                          corrected_orography_in,
                          cell_areas_in,
                          connection_volume_thresholds_in,
                          flood_volume_thresholds_in,
                          prior_fine_rdirs_in,
                          prior_fine_catchments_in,
                          coarse_catchment_nums_in,
                          flood_next_cell_lat_index_in,
                          flood_next_cell_lon_index_in,
                          connect_next_cell_lat_index_in,
                          connect_next_cell_lon_index_in,
                          flood_force_merge_lat_index_in,
                          flood_force_merge_lon_index_in,
                          connect_force_merge_lat_index_in,
                          connect_force_merge_lon_index_in,
                          flood_redirect_lat_index_in,
                          flood_redirect_lon_index_in,
                          connect_redirect_lat_index_in,
                          connect_redirect_lon_index_in,
                          additional_flood_redirect_lat_index_in,
                          additional_flood_redirect_lon_index_in,
                          additional_connect_redirect_lat_index_in,
                          additional_connect_redirect_lon_index_in,
                          flood_local_redirect_in,
                          connect_local_redirect_in,
                          additional_flood_local_redirect_in,
                          additional_connect_local_redirect_in,
                          merge_points_in,
                          grid_params_in,
                          coarse_grid_params_in);
  bool* landsea_in = new bool[20*20] {
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false, false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
     true,false,false,false,
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
   false,false,false,false,false,false,false,false, true,false,false,false,false,false,false,false,
    false,false,false,false};
  bool* true_sinks_in = new bool[20*20];
  fill_n(true_sinks_in,20*20,false);
  int* next_cell_lat_index_in = new int[20*20];
  int* next_cell_lon_index_in = new int[20*20];
  short* sinkless_rdirs_out = new short[20*20];
  int* catchment_nums_in = new int[20*20];
  reverse_priority_cell_queue q;
  auto alg4 = new sink_filling_algorithm_4_latlon();
  alg4->setup_flags(false,true,false,false);
  alg4->setup_fields(corrected_orography_in,
                     landsea_in,
                     true_sinks_in,
                     next_cell_lat_index_in,
                     next_cell_lon_index_in,
                     grid_params_in,
                     sinkless_rdirs_out,
                     catchment_nums_in);
  basin_eval.setup_sink_filling_algorithm(alg4);
  basin_eval.evaluate_basins();
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in)
              == field<double>(flood_volume_thresholds_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in)
              == field<double>(connection_volume_thresholds_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in)
              == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in)
              == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in; delete alg4;
  delete[] coarse_catchment_nums_in; delete[] corrected_orography_in;
  delete[] raw_orography_in; delete[] minima_in;
  delete[] prior_fine_rdirs_in; delete[] prior_fine_catchments_in;
  delete[] connection_volume_thresholds_in; delete[] flood_volume_thresholds_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] additional_flood_redirect_lat_index_in;
  delete[] additional_flood_redirect_lon_index_in;
  delete[] additional_connect_redirect_lat_index_in;
  delete[] additional_connect_redirect_lon_index_in;
  delete[] additional_flood_local_redirect_in;
  delete[] additional_connect_local_redirect_in;
  delete[] merge_points_in; delete[] flood_volume_thresholds_expected_out;
  delete[] connection_volume_thresholds_expected_out;
  delete[] flood_next_cell_lat_index_expected_out;
  delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out;
  delete[] connect_next_cell_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] landsea_in;
  delete[] true_sinks_in;
  delete[] next_cell_lat_index_in;
  delete[] next_cell_lon_index_in;
  delete[] sinkless_rdirs_out;
  delete[] catchment_nums_in;
  delete[] cell_areas_in;
}

TEST_F(BasinEvaluationTest, TestEvaluateBasinsThree) {
  auto grid_params_in = new latlon_grid_params(20,20,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* coarse_catchment_nums_in = new int[4*4] {3,3,2,2,
                                                3,3,2,2,
                                                1,1,1,2,
                                                1,1,1,1};
  double* corrected_orography_in = new double[20*20]
  {20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 1.0,10.0,20.0, 9.0, 20.0,20.0,20.0, 1.0,16.0, 20.0,20.0, 1.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 2.0,11.0,20.0, 8.0, 20.0,20.0,20.0, 2.0,15.0, 20.0,20.0, 2.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   18.0, 3.0,12.0,20.0, 7.0, 20.0,20.0,20.0, 3.0,14.0, 20.0,20.0, 3.0,18.0, 0.9,  0.8, 0.0, 0.7, 0.8, 0.9,
   20.0, 4.0,13.0, 6.0, 5.0, 20.0,20.0,20.0, 4.0,13.0,  5.0, 5.0, 4.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
//
   20.0, 5.0,14.0,20.0, 4.0, 20.0,20.0,20.0, 5.0,12.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 6.0,15.0,20.0, 3.0, 20.0,20.0,20.0, 6.0,11.0, 20.0,20.0, 7.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 8.0,16.0,20.0, 2.0, 20.0,20.0,20.0, 7.0,10.0, 20.0,20.0, 8.0,20.0,20.0, 20.0,20.0, 5.0,20.0,20.0,
   20.0, 9.0,17.0,20.0, 1.0, 20.0,20.0,20.0, 8.0, 9.0, 20.0,20.0, 9.0,20.0,20.0, 20.0, 4.0, 6.0,20.0,20.0,
   20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,  3.0, 7.0,20.0,20.0,20.0,
//
   20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0, 2.0,  8.0,20.0,20.0,20.0,20.0,
   20.0, 1.0,16.0,20.0, 2.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 1.0, 9.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 2.0,15.0,20.0, 3.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,10.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 3.0,14.0,20.0, 4.0, 20.0,20.0, 1.0,20.0, 5.0, 20.0,20.0,20.0, 0.9,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 4.0,13.0, 6.0, 5.0, 20.0,20.0, 2.0, 4.0, 3.0, 20.0,20.0, 0.8,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
//
   20.0, 5.0,12.0,20.0, 7.0, 20.0,20.0, 3.0,20.0, 2.0, 20.0,20.0, 0.7,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 6.0,11.0,20.0, 8.0, 20.0,20.0, 4.0,20.0, 1.0, 20.0,20.0, 0.6,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 7.0,10.0,20.0, 9.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0, 0.5,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 8.0, 9.0,20.0,10.0, 18.0, 0.9, 0.8,20.0,20.0, 20.0, 0.4,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 0.0, 0.2,  0.3,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0};

  double* raw_orography_in = new double[20*20]
  {20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 1.0,10.0,20.0, 9.0, 20.0,20.0,20.0, 1.0,16.0, 20.0,20.0, 1.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 2.0,11.0,20.0, 8.0, 20.0,20.0,20.0, 2.0,15.0, 20.0,20.0, 2.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   18.0, 3.0,12.0,20.0, 7.0, 20.0,20.0,20.0, 3.0,14.0, 20.0,20.0, 3.0,18.0, 0.9,  0.8, 0.0, 0.7, 0.8, 0.9,
   20.0, 4.0,13.0,20.0, 5.0, 20.0,20.0,20.0, 4.0,13.0,  5.0, 5.0, 4.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
//
   20.0, 5.0,14.0,20.0, 4.0, 20.0,20.0,20.0, 5.0,12.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 6.0,15.0,20.0, 3.0, 20.0,20.0,20.0, 6.0,11.0, 20.0,20.0, 7.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 8.0,16.0,20.0, 2.0, 20.0,20.0,20.0, 7.0,10.0, 20.0,20.0, 8.0,20.0,20.0, 20.0,20.0, 5.0,20.0,20.0,
   20.0, 9.0,17.0,20.0, 1.0, 20.0,20.0,20.0, 8.0, 9.0, 20.0,20.0, 9.0,20.0,20.0, 20.0, 4.0, 6.0,20.0,20.0,
   20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,  3.0, 7.0,20.0,20.0,20.0,
//
   20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0, 2.0,  8.0,20.0,20.0,20.0,20.0,
   20.0, 1.0,16.0,20.0, 2.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 1.0, 9.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 2.0,15.0,20.0, 3.0, 20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,10.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 3.0,14.0,20.0, 4.0, 20.0,20.0, 1.0,20.0, 5.0, 20.0,20.0,20.0, 0.9,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 4.0,13.0,20.0, 5.0, 20.0,20.0, 2.0, 4.0, 3.0, 20.0,20.0, 0.8,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
//
   20.0, 5.0,12.0,20.0, 7.0, 20.0,20.0, 3.0,20.0, 2.0, 20.0,20.0, 0.7,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 6.0,11.0,20.0, 8.0, 20.0,20.0, 4.0,20.0, 1.0, 20.0,20.0, 0.6,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 7.0,10.0,20.0, 9.0, 20.0,20.0, 6.0,20.0,20.0, 20.0,20.0, 0.5,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0, 8.0, 9.0,20.0,10.0, 18.0, 0.9, 0.8,20.0,20.0, 20.0, 0.4,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0,
   20.0,20.0,20.0,20.0,20.0, 20.0,20.0,20.0, 0.0, 0.2,  0.3,20.0,20.0,20.0,20.0, 20.0,20.0,20.0,20.0,20.0};

  bool* minima_in = new bool[20*20]
  {false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false, true,false,false,false,false,false,false, true,false,false,false, true,false,false,false,
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
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false, true,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false, true,false,false, true,false,false,false,false,false,false,false,false, true,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false, true,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false, true,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false};
  double* prior_fine_rdirs_in = new double[20*20] {
  1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,3.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,3.0,3.0,2.0,
  2.0,1.0,2.0,1.0,1.0,2.0,1.0,1.0,1.0,3.0,3.0,3.0,2.0,3.0,2.0,1.0,1.0,3.0,3.0,3.0,
  5.0,4.0,1.0,1.0,3.0,2.0,1.0,1.0,1.0,1.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,6.0,6.0,6.0,
  8.0,7.0,4.0,4.0,3.0,2.0,1.0,1.0,2.0,1.0,6.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,
  9.0,8.0,7.0,3.0,3.0,2.0,1.0,2.0,1.0,1.0,3.0,6.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0,
  9.0,8.0,7.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,3.0,3.0,6.0,5.0,4.0,4.0,1.0,1.0,1.0,3.0,
  9.0,8.0,7.0,6.0,6.0,5.0,4.0,4.0,4.0,4.0,6.0,6.0,9.0,8.0,7.0,1.0,1.0,4.0,4.0,6.0,
  9.0,9.0,8.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0,9.0,9.0,8.0,7.0,5.0,4.0,4.0,7.0,7.0,9.0,
  9.0,9.0,8.0,7.0,9.0,8.0,7.0,8.0,7.0,7.0,9.0,9.0,8.0,7.0,8.0,7.0,7.0,7.0,2.0,1.0,
  9.0,9.0,8.0,9.0,9.0,8.0,7.0,7.0,7.0,9.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0,7.0,2.0,1.0,
  1.0,1.0,9.0,9.0,9.0,8.0,7.0,7.0,7.0,9.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0,3.0,2.0,1.0,
  1.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,9.0,8.0,3.0,3.0,2.0,1.0,3.0,2.0,1.0,
  1.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,9.0,3.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,
  4.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,9.0,9.0,3.0,6.0,6.0,6.0,5.0,4.0,4.0,4.0,4.0,
  7.0,3.0,3.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,3.0,3.0,3.0,9.0,9.0,8.0,7.0,7.0,7.0,7.0,
  7.0,3.0,6.0,5.0,4.0,1.0,1.0,1.0,1.0,1.0,3.0,3.0,3.0,3.0,9.0,8.0,7.0,7.0,7.0,7.0,
  3.0,3.0,3.0,2.0,1.0,4.0,4.0,4.0,4.0,4.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,7.0,7.0,
  3.0,3.0,3.0,2.0,1.0,7.0,7.0,7.0,7.0,7.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,1.0,7.0,
  3.0,3.0,3.0,2.0,1.0,1.0,7.0,7.0,7.0,7.0,3.0,3.0,3.0,3.0,3.0,2.0,1.0,1.0,1.0,3.0,
  6.0,6.0,6.0,0.0,4.0,4.0,4.0,7.0,7.0,7.0,6.0,6.0,6.0,6.0,6.0,0.0,4.0,4.0,4.0,6.0};
  int* prior_fine_catchments_in = new int[20*20] {
  11,11,11,11,11,11,13,13,12,12,12,12,12,12,12,12,12,11,11,11,
  11,11,11,11,11,13,13,13,13,12,12,12,12,12,12,12,12,11,11,11,
  11,11,11,11,13,13,13,13,13,13,12,12,12,12,12,12,12,11,11,11,
  11,11,11,11,13,13,13,13,13,13,12,12,12,12,12,12,12,12,11,11,
  11,11,11,13,13,13,13,13,13,13,12,12,12,12,12,12,12,14,14,11,
  11,11,11,13,13,13,13,13,13,13,12,12,12,12,12,12,14,14,14,11,
  11,11,11,13,13,13,13,13,13,13,12,12,12,12,12,14,14,14,14,11,
  11,11,11,13,13,13,13,13,13,13,12,12,12,12,14,14,14,14,14,11,
  11,11,11,11,13,13,13,13,13,13,12,12,12,12,14,14,14,14,15,15,
  11,11,11,13,13,13,13,13,13,12,12,12,12,12,12,14,14,14,15,15,
  15,15,13,13,13,13,13,13,13,12,12,12,12,12,12,12,14,15,15,15,
  15,16,16,16,16,16,16,16,12,12,12,12,12,15,15,15,15,15,15,15,
  15,16,16,16,16,16,16,16, 4,12,12,12,15,15,15,15,15,15,15,15,
  15,16,16,16,16,16,16, 4, 4,12,12, 9,15,15,15,15,15,15,15,15,
  15,16,16,16,16,16, 4, 4, 4, 4,10, 9, 9,15,15,15,15,15,15,15,
  15, 4,16,16,16, 4, 4, 4, 4, 4, 7,10, 9, 9,15,15,15,15,15,15,
   5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7,10, 9, 9, 9, 9, 9,15,15,
   2, 5, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7,10, 9, 9, 9, 8, 6,15,
   2, 2, 5, 4, 3, 1, 4, 4, 4, 4, 7, 7, 7, 7,10, 9, 8, 6, 6, 2,
   2, 2, 2, 0, 1, 1, 1, 4, 4, 4, 7, 7, 7, 7, 7, 0, 6, 6, 6, 2 };
  double* cell_areas_in = new double[20*20]{
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
    //
    6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
    9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
    10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,
    //
    10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,
    9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
    8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
    //
    5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
  };
  double* connection_volume_thresholds_in = new double[20*20];
  std::fill_n(connection_volume_thresholds_in,20*20,0.0);
  double* flood_volume_thresholds_in = new double[20*20];
  std::fill_n(flood_volume_thresholds_in,20*20,0.0);
  int* flood_next_cell_lat_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lat_index_in,20*20,-1);
  int* flood_next_cell_lon_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lon_index_in,20*20,-1);
  int* connect_next_cell_lat_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lat_index_in,20*20,-1);
  int* connect_next_cell_lon_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lon_index_in,20*20,-1);
  int* flood_force_merge_lat_index_in = new int[20*20];
  std::fill_n(flood_force_merge_lat_index_in,20*20,-1);
  int* flood_force_merge_lon_index_in = new int[20*20];
  std::fill_n(flood_force_merge_lon_index_in,20*20,-1);
  int* connect_force_merge_lat_index_in = new int[20*20];
  std::fill_n(connect_force_merge_lat_index_in,20*20,-1);
  int* connect_force_merge_lon_index_in = new int[20*20];
  std::fill_n(connect_force_merge_lon_index_in,20*20,-1);
  int* flood_redirect_lat_index_in = new int[20*20];
  std::fill_n(flood_redirect_lat_index_in,20*20,-1);
  int* flood_redirect_lon_index_in = new int[20*20];
  std::fill_n(flood_redirect_lon_index_in,20*20,-1);
  int* connect_redirect_lat_index_in = new int[20*20];
  std::fill_n(connect_redirect_lat_index_in,20*20,-1);
  int* connect_redirect_lon_index_in = new int[20*20];
  std::fill_n(connect_redirect_lon_index_in,20*20,-1);
  bool* flood_local_redirect_in = new bool[20*20];
  std::fill_n(flood_local_redirect_in,20*20,false);
  bool* connect_local_redirect_in = new bool[20*20];
  std::fill_n(connect_local_redirect_in,20*20,false);
  int* additional_flood_redirect_lat_index_in = new int[20*20];
  std::fill_n(additional_flood_redirect_lat_index_in,20*20,-1);
  int* additional_flood_redirect_lon_index_in = new int[20*20];
  std::fill_n(additional_flood_redirect_lon_index_in,20*20,-1);
  int* additional_connect_redirect_lat_index_in = new int[20*20];
  std::fill_n(additional_connect_redirect_lat_index_in,20*20,-1);
  int* additional_connect_redirect_lon_index_in = new int[20*20];
  std::fill_n(additional_connect_redirect_lon_index_in,20*20,-1);
  bool* additional_flood_local_redirect_in = new bool[20*20];
  std::fill_n(additional_flood_local_redirect_in,20*20,false);
  bool* additional_connect_local_redirect_in = new bool[20*20];
  std::fill_n(additional_connect_local_redirect_in,20*20,false);
  merge_types* merge_points_in = new merge_types[20*20];
  std::fill_n(merge_points_in,20*20,no_merge);
  double* flood_volume_thresholds_expected_out = new double[20*20] {
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0, 2.0,229.0,-1.0,363.0, -1.0,-1.0,-1.0, 2.0,1150.0,   -1.0,-1.0, 2.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0, 7.0,278.0,-1.0,231.0, -1.0,-1.0,-1.0, 7.0,866.0,   -1.0,-1.0, 7.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,16.0,375.0,-1.0,189.0, -1.0,-1.0,-1.0, 16.0,726.0,   -1.0,-1.0, 16.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,30.0,477.0,-1.0,115.0, -1.0,-1.0,-1.0, 30.0,589.0, 54.0,30.0, 30.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
//
   -1.0,50.0,585.0,-1.0,80.0, -1.0,-1.0,-1.0, 50.0,456.0, -1.0,-1.0,84.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,104.0,700.0,-1.0,50.0, -1.0,-1.0,-1.0, 77.0,338.0, -1.0,-1.0,121.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,139.0,823.0,-1.0,26.0, -1.0,-1.0,-1.0, 112.0,270.0, -1.0,-1.0,166.0,-1.0,-1.0, -1.0,-1.0,141.0,-1.0,-1.0,
   -1.0,183.0,955.0,-1.0, 9.0, -1.0,-1.0,-1.0, 156.0,209.0, -1.0,-1.0,328.0,-1.0,-1.0, -1.0,95.0,196.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, 57.0, 261.0,-1.0,-1.0,-1.0,
//
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0, 28.0,  336.0,-1.0,-1.0,-1.0,-1.0,
   -1.0, 9.0,1026.0,-1.0,9.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, 9.0,420.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,26.0,762.0,-1.0,26.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,50.0,639.0,-1.0,50.0, -1.0,-1.0, 7.0,-1.0,131.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,80.0,524.0,-1.0,80.0, -1.0,-1.0, 20.0, 38.0,28.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
//
   -1.0,115.0,416.0,-1.0,145.0, -1.0,-1.0,38.0,-1.0,13.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,154.0,388.0,-1.0,184.0, -1.0,-1.0,81.0,-1.0, 4.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,196.0,335.0,-1.0,226.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,240.0,286.0,-1.0,314.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0 };
  double* connection_volume_thresholds_expected_out = new double[20*20]{
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1,150.0,-1,  -1,      -1,  -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1,110.0,-1,  -1,      -1,  -1,  -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, -1,   -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* flood_next_cell_lat_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  2,  2, -1,  2,  -1, -1, -1,  2,  3,  -1, -1,  2, -1, -1,  -1, -1, -1, -1, -1,
    -1,  3,  3, -1,  1,  -1, -1, -1,  3,  1,  -1, -1,  3, -1, -1,  -1, -1, -1, -1, -1,
    -1,  4,  4, -1,  2,  -1, -1, -1,  4,  2,  -1, -1,  4, -1, -1,  -1, -1, -1, -1, -1,
    -1,  5,  5, -1,  4,  -1, -1, -1,  5,  3,   5,  4,  4, -1, -1,  -1, -1, -1, -1, -1,
    //
    -1,  6,  6, -1,  4,  -1, -1, -1,  6,  4,  -1, -1,  6, -1, -1,  -1, -1, -1, -1, -1,
    -1,  7,  7, -1,  5,  -1, -1, -1,  7,  4,  -1, -1,  7, -1, -1,  -1, -1, -1, -1, -1,
    -1,  8,  8, -1,  6,  -1, -1, -1,  8,  6,  -1, -1,  8, -1, -1,  -1, -1,  8, -1, -1,
    -1,  1,  3, -1,  7,  -1, -1, -1,  8,  7,  -1, -1,  5, -1, -1,  -1,  7,  9, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,   8, 10, -1, -1, -1,
    //
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1,  9,  11, -1, -1, -1, -1,
    -1, 12, 18, -1, 12,  -1, -1, -1, -1, -1,  -1, -1, -1, 10, 13,  -1, -1, -1, -1, -1,
    -1, 13, 11, -1, 13,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, 14, 12, -1, 14,  -1, -1, 14, -1, 18,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, 15, 13, -1, 14,  -1, -1, 15, 16, 13,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    //
    -1, 16, 14, -1, 16,  -1, -1, 14, -1, 14,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, 17, 14, -1, 17,  -1, -1, 13, -1, 15,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, 18, 16, -1, 18,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, 18, 17, -1, 15,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* flood_next_cell_lon_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  1,  -1, -1, -1,  8, 14,  -1, -1, 12, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  4,  -1, -1, -1,  8,  9,  -1, -1, 12, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  4,  -1, -1, -1,  8,  9,  -1, -1, 12, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  3,  -1, -1, -1,  8,  9,  12, 10, 11, -1, -1,  -1, -1, -1, -1, -1,
    //
    -1,  1,  2, -1,  4,  -1, -1, -1,  8,  9,  -1, -1, 12, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  4,  -1, -1, -1,  8, 10,  -1, -1, 12, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  4,  -1, -1, -1,  8,  9,  -1, -1, 12, -1, -1,  -1, -1, 17, -1, -1,
    -1,  2, 19, -1,  4,  -1, -1, -1,  9,  9,  -1, -1,  9, -1, -1,  -1, 17, 16, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  16, 15, -1, -1, -1,
    //
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, 15,  14, -1, -1, -1, -1,
    -1,  1,  6, -1,  4,  -1, -1, -1, -1, -1,  -1, -1, -1, 14, 13,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  4,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  4,  -1, -1,  7, -1,  7,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  3,  -1, -1,  7,  7,  7,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    //
    -1,  1,  2, -1,  4,  -1, -1,  8, -1,  9,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  3, -1,  4,  -1, -1,  9, -1,  9,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  1,  2, -1,  4,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1,  2,  2, -1,  2,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* connect_next_cell_lat_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1,  3, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
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
    -1, -1, -1,  15, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1,-1,
    //
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* connect_next_cell_lon_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1,  4, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
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
    -1, -1, -1,  4, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    //
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* flood_redirect_lat_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1,  1, -1, -1, -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  2, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  int* flood_redirect_lon_index_expected_out = new int[20*20]{
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1,  1, -1, -1, -1, -1,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  3, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
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
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false,  true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false,  true, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false,  true, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false,  true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false };
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
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,  no_merge,   no_merge,   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   connection_merge_not_set_flood_merge_as_primary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   connection_merge_not_set_flood_merge_as_primary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   connection_merge_not_set_flood_merge_as_primary,  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   connection_merge_not_set_flood_merge_as_primary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,  };

  int* flood_force_merge_lat_index_expected_out = new int[20*20] {

    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1,  4, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,  4, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, 15, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, 14,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1 };
  int* flood_force_merge_lon_index_expected_out = new int[20*20] {
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1,  3, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,  8, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
//
    -1, -1, -1, -1, -1,  -1, -1,  9, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1,  1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
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
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.setup_fields(minima_in,
                          raw_orography_in,
                          corrected_orography_in,
                          cell_areas_in,
                          connection_volume_thresholds_in,
                          flood_volume_thresholds_in,
                          prior_fine_rdirs_in,
                          prior_fine_catchments_in,
                          coarse_catchment_nums_in,
                          flood_next_cell_lat_index_in,
                          flood_next_cell_lon_index_in,
                          connect_next_cell_lat_index_in,
                          connect_next_cell_lon_index_in,
                          flood_force_merge_lat_index_in,
                          flood_force_merge_lon_index_in,
                          connect_force_merge_lat_index_in,
                          connect_force_merge_lon_index_in,
                          flood_redirect_lat_index_in,
                          flood_redirect_lon_index_in,
                          connect_redirect_lat_index_in,
                          connect_redirect_lon_index_in,
                          additional_flood_redirect_lat_index_in,
                          additional_flood_redirect_lon_index_in,
                          additional_connect_redirect_lat_index_in,
                          additional_connect_redirect_lon_index_in,
                          flood_local_redirect_in,
                          connect_local_redirect_in,
                          additional_flood_local_redirect_in,
                          additional_connect_local_redirect_in,
                          merge_points_in,
                          grid_params_in,
                          coarse_grid_params_in);
  bool* landsea_in = new bool[20*20] {
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false, false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
    false,false,false,false,
   false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,
     true,false,false,false,
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
   false,false,false,false,false,false,false,false, true,false,false,false,false,false,false,false,
    false,false,false,false};
  bool* true_sinks_in = new bool[20*20];
  fill_n(true_sinks_in,20*20,false);
  int* next_cell_lat_index_in = new int[20*20];
  int* next_cell_lon_index_in = new int[20*20];
  short* sinkless_rdirs_out = new short[20*20];
  int* catchment_nums_in = new int[20*20];
  reverse_priority_cell_queue q;
  auto alg4 = new sink_filling_algorithm_4_latlon();
  alg4->setup_flags(false,true,false,false);
  alg4->setup_fields(corrected_orography_in,
                     landsea_in,
                     true_sinks_in,
                     next_cell_lat_index_in,
                     next_cell_lon_index_in,
                     grid_params_in,
                     sinkless_rdirs_out,
                     catchment_nums_in);
  basin_eval.setup_sink_filling_algorithm(alg4);
  basin_eval.evaluate_basins();
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in)
              == field<double>(flood_volume_thresholds_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in)
              == field<double>(connection_volume_thresholds_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in)
              == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in)
              == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
              == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
              == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
              == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
              == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
              == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
              == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
              == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
              == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
              == field<merge_types>(merge_points_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
              == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
              == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
              == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
              == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in; delete alg4;
  delete[] coarse_catchment_nums_in; delete[] corrected_orography_in;
  delete[] raw_orography_in; delete[] minima_in;
  delete[] prior_fine_rdirs_in; delete[] prior_fine_catchments_in;
  delete[] connection_volume_thresholds_in; delete[] flood_volume_thresholds_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  delete[] additional_flood_redirect_lat_index_in;
  delete[] additional_flood_redirect_lon_index_in;
  delete[] additional_connect_redirect_lat_index_in;
  delete[] additional_connect_redirect_lon_index_in;
  delete[] additional_flood_local_redirect_in;
  delete[] additional_connect_local_redirect_in;
  delete[] merge_points_in; delete[] flood_volume_thresholds_expected_out;
  delete[] connection_volume_thresholds_expected_out;
  delete[] flood_next_cell_lat_index_expected_out;
  delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out;
  delete[] connect_next_cell_lon_index_expected_out;
  delete[] flood_redirect_lat_index_expected_out;
  delete[] flood_redirect_lon_index_expected_out;
  delete[] connect_redirect_lat_index_expected_out;
  delete[] connect_redirect_lon_index_expected_out;
  delete[] flood_local_redirect_expected_out;
  delete[] connect_local_redirect_expected_out;
  delete[] merge_points_expected_out;
  delete[] flood_force_merge_lat_index_expected_out;
  delete[] flood_force_merge_lon_index_expected_out;
  delete[] connect_force_merge_lat_index_expected_out;
  delete[] connect_force_merge_lon_index_expected_out;
  delete[] landsea_in;
  delete[] true_sinks_in;
  delete[] next_cell_lat_index_in;
  delete[] next_cell_lon_index_in;
  delete[] sinkless_rdirs_out;
  delete[] catchment_nums_in;
  delete[] cell_areas_in;
}

TEST_F(BasinEvaluationTest, TestEvaluateBasinsFour) {
  auto grid_params_in = new latlon_grid_params(20,20,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* coarse_catchment_nums_in = new int[4*4] {0,0,0,0,
                                                0,7,9,0,
                                                0,1,3,0,
                                                0,0,0,0};
  double* corrected_orography_in = new double[20*20]
  {2833.89, 2408.27, 2605.78, 2672.46, 2670.99, 2535.43, 2573.21, 2600.76, 2558.59, 2706.72, 2506.95, 3109.13, 2567.04, 2453.29, 2433.97, 2471.12, 2023.5, 2203.63, 2300.46, 2169.43,
2724.82, 2821.9, 2549.44, 2472.69, 2688.66, 2683.13, 2719.41, 2683.89, 2244.59, 2483.89, 2689.2, 2432.77, 2797.78, 2544.55, 2494.41, 2536.93, 2465.66, 2440.65, 2225.69, 2288.92,
2692.48, 2748.9, 2447.34, 2755.49, 2874.7, 2346.54, 2536.99, 2721.65, 2468.33, 2546.6, 2963.46, 2564.8, 2937.53, 3036.52, 2731.27, 2529.42, 2821.21, 2742.64, 2499.66, 2405.14,
2627.23, 2395.91, 2848.61, 2678.34, 3011.72, 2613.43, 2745.54, 2989.27, 2920.23, 2291.84, 2545.44, 2789.72, 2346.59, 2786.36, 2684.83, 2438.92, 2610.69, 2689.3, 2602.33, 2525.55,
2423.28, 2483.7, 2735.31, 2976.6, 3233.42, 2668.57, 2292.6, 2740.1, 2857.81, 2743.6, 2444.06, 2573.87, 2855.08, 2613.29, 2701.57, 3352.55, 2564, 3025.03, 2427.56, 2469.08,
2981.94, 2267.33, 2782.81, 2527.5, 2766.48, 2957.41, 3343.05, 3141.05, 2566.35, 2650.34, 2742.92, 2280.02, 2626.72, 2881.62, 3167.12, 3115.05, 2838.05, 2636.08, 2783.24, 3126.29,
2562.15, 2820.65, 2911.99, 2642.95, 3150.44, 2533.18, 3067.33, 3084.39, 2840.64, 2760.65, 2403.02, 2529.29, 3511.31, 2271.61, 2227.3, 2508.7, 2858.88, 3293.69, 3142.58, 2680.7,
2190.85, 3114.9, 3131.08, 2820.27, 3287.25, 3384.07, 3141.46, 3457.63, 2889.2, 2867.08, 2273.87, 3345.01, 3061.76, 3106.28, 2781.76, 3295.93, 3217.44, 2903.97, 2791.47, 3121.01,
2919.65, 2906.96, 2959.25, 2909.48, 2775.41, 2819.85, 2863.38, 3402.57, 3294.53, 3408.99, 3257.53, 2952.45, 2855.42, 2938.21, 2984.79, 2621.18, 3244.9, 3160.94, 2213.43, 2917.12,
2702.87, 2762.45, 2828.82, 2774.3, 2822.31, 3045.13, 2921.17, 2639.54, 3239.07, 3116.82, 2887.34, 2887.21, 3021.7, 2964.61, 2807.67, 2814.2, 2900.88, 2604.8, 3330.18, 2857.24,
2805.18, 2773.8, 2415.13, 2698.35, 2815.15, 2832.7, 2767.04, 2951.54, 3424.48, 2821.08, 3117.57, 3071.27, 3405.28, 3236.87, 2979.43, 2855.74, 2865.74, 2706.25, 2816.8, 2911.33,
2711.15, 2488.32, 2026.33, 2827.07, 3217.7, 2745.57, 3005.12, 2625.65, 2892.49, 2968.37, 3117.16, 2917.88, 2897.28, 3336.53, 2734.44, 3487.47, 2808.68, 2663.77, 3230.12, 2849.32,
2965.66, 2752.69, 2822.87, 3190.11, 2833.7, 2757.82, 2936.43, 3039.44, 2797.05, 2715.93, 2975.38, 2853.35, 2857.33, 3466.81, 3222.28, 2746.64, 2664.91, 2942.43, 3019.91, 2931.11,
2721.32, 2739.07, 2827.06, 2792.29, 2271.22, 2805.07, 2486.66, 2276.78, 2765.03, 2963.27, 2653.15, 2579.8, 3302.41, 3137.43, 3141.73, 2825.02, 3057.72, 2786.84, 2690.5, 2983.38,
2984.65, 2888.7, 2546.79, 2908.24, 2780.61, 2906.86, 3314.75, 3106.97, 3033.48, 3041.77, 2841.53, 2667.07, 2880.55, 2972.32, 3179.55, 3117.97, 2951.07, 3388.83, 3449.22, 2587.47,
2611.37, 2768.02, 2303.46, 2803.46, 2622.72, 2292.72, 2667.95, 2582.67, 2951.93, 2800.26, 3151.58, 2882.46, 3030.87, 3141.97, 3126.41, 3341.3, 2686.55, 2545.24, 3390.11, 2184.01,
1985.21, 2414.69, 2470.74, 2611.94, 2647.45, 2423.39, 2608.3, 2276.45, 2485.98, 2584.66, 3133.45, 3426.55, 2667.1, 2962.01, 2948.8, 2967.54, 3158.43, 2586.37, 2798.46, 2669.57,
2225.67, 2770.73, 2372, 1766.48, 2738.81, 3142.5, 2609.9, 2975.05, 2681.23, 2820.64, 2538.64, 2740.82, 2776.82, 2492.22, 3088.22, 2828.46, 3058.16, 3223.74, 2993.81, 2788.68,
2112.81, 2293.92, 1545.11, 2160.44, 2771.72, 2602.59, 2410.17, 2656.02, 2711.45, 2800.98, 2867.34, 2683.17, 2811.27, 2497.78, 2911.4, 2693.63, 3010.6, 2708.76, 2661.58, 3216.83,
2363.68, 1740.12, 2561.46, 2363.19, 2825.42, 2054.6, 1979.61, 2261.22, 2079.41, 2659.03, 2567.5, 3169.21, 2685.55, 2924.68, 3025.55, 3043.83, 2845.74, 3167.29, 2906.02, 3262.75
};

  double* raw_orography_in = new double[20*20]
  {2833.89, 2408.27, 2605.78, 2672.46, 2670.99, 2535.43, 2573.21, 2600.76, 2558.59, 2706.72, 2506.95, 3109.13, 2567.04, 2453.29, 2433.97, 2471.12, 2023.5, 2203.63, 2300.46, 2169.43,
2724.82, 2821.9, 2549.44, 2472.69, 2688.66, 2683.13, 2719.41, 2683.89, 2244.59, 2483.89, 2689.2, 2432.77, 2797.78, 2544.55, 2494.41, 2536.93, 2465.66, 2440.65, 2225.69, 2288.92,
2692.48, 2748.9, 2447.34, 2755.49, 2874.7, 2346.54, 2536.99, 2721.65, 2468.33, 2546.6, 2963.46, 2564.8, 2937.53, 3036.52, 2731.27, 2529.42, 2821.21, 2742.64, 2499.66, 2405.14,
2627.23, 2395.91, 2848.61, 2678.34, 3011.72, 2613.43, 2745.54, 2989.27, 2920.23, 2291.84, 2545.44, 2789.72, 2346.59, 2786.36, 2684.83, 2438.92, 2610.69, 2689.3, 2602.33, 2525.55,
2423.28, 2483.7, 2735.31, 2976.6, 3233.42, 2668.57, 2292.6, 2740.1, 2857.81, 2743.6, 2444.06, 2573.87, 2855.08, 2613.29, 2701.57, 3352.55, 2564, 3025.03, 2427.56, 2469.08,
2981.94, 2267.33, 2782.81, 2527.5, 2766.48, 2957.41, 3343.05, 3141.05, 2566.35, 2650.34, 2742.92, 2280.02, 2626.72, 2881.62, 3167.12, 3115.05, 2838.05, 2636.08, 2783.24, 3126.29,
2562.15, 2820.65, 2911.99, 2642.95, 3150.44, 2533.18, 3067.33, 3084.39, 2840.64, 2760.65, 2403.02, 2529.29, 3511.31, 2271.61, 2227.3, 2508.7, 2858.88, 3293.69, 3142.58, 2680.7,
2190.85, 3114.9, 3131.08, 2820.27, 3287.25, 3384.07, 3141.46, 3457.63, 2889.2, 2867.08, 2273.87, 3345.01, 3061.76, 3106.28, 2781.76, 3295.93, 3217.44, 2903.97, 2791.47, 3121.01,
2919.65, 2906.96, 2959.25, 2909.48, 2775.41, 2819.85, 2863.38, 3402.57, 3294.53, 3408.99, 3257.53, 2952.45, 2855.42, 2938.21, 2984.79, 2621.18, 3244.9, 3160.94, 2213.43, 2917.12,
2702.87, 2762.45, 2828.82, 2774.3, 2822.31, 3045.13, 2921.17, 2639.54, 3239.07, 3116.82, 2887.34, 2887.21, 3021.7, 2964.61, 2807.67, 2814.2, 2900.88, 2604.8, 3330.18, 2857.24,
2805.18, 2773.8, 2415.13, 2698.35, 2815.15, 2832.7, 2767.04, 2951.54, 3424.48, 2821.08, 3117.57, 3071.27, 3405.28, 3236.87, 2979.43, 2855.74, 2865.74, 2706.25, 2816.8, 2911.33,
2711.15, 2488.32, 2026.33, 2827.07, 3217.7, 2745.57, 3005.12, 2625.65, 2892.49, 2968.37, 3117.16, 2917.88, 2897.28, 3336.53, 2734.44, 3487.47, 2808.68, 2663.77, 3230.12, 2849.32,
2965.66, 2752.69, 2822.87, 3190.11, 2833.7, 2757.82, 2936.43, 3039.44, 2797.05, 2715.93, 2975.38, 2853.35, 2857.33, 3466.81, 3222.28, 2746.64, 2664.91, 2942.43, 3019.91, 2931.11,
2721.32, 2739.07, 2827.06, 2792.29, 2271.22, 2805.07, 2486.66, 2276.78, 2765.03, 2963.27, 2653.15, 2579.8, 3302.41, 3137.43, 3141.73, 2825.02, 3057.72, 2786.84, 2690.5, 2983.38,
2984.65, 2888.7, 2546.79, 2908.24, 2780.61, 2906.86, 3314.75, 3106.97, 3033.48, 3041.77, 2841.53, 2667.07, 2880.55, 2972.32, 3179.55, 3117.97, 2951.07, 3388.83, 3449.22, 2587.47,
2611.37, 2768.02, 2303.46, 2803.46, 2622.72, 2292.72, 2667.95, 2582.67, 2951.93, 2800.26, 3151.58, 2882.46, 3030.87, 3141.97, 3126.41, 3341.3, 2686.55, 2545.24, 3390.11, 2184.01,
1985.21, 2414.69, 2470.74, 2611.94, 2647.45, 2423.39, 2608.3, 2276.45, 2485.98, 2584.66, 3133.45, 3426.55, 2667.1, 2962.01, 2948.8, 2967.54, 3158.43, 2586.37, 2798.46, 2669.57,
2225.67, 2770.73, 2372, 1766.48, 2738.81, 3142.5, 2609.9, 2975.05, 2681.23, 2820.64, 2538.64, 2740.82, 2776.82, 2492.22, 3088.22, 2828.46, 3058.16, 3223.74, 2993.81, 2788.68,
2112.81, 2293.92, 1545.11, 2160.44, 2771.72, 2602.59, 2410.17, 2656.02, 2711.45, 2800.98, 2867.34, 2683.17, 2811.27, 2497.78, 2911.4, 2693.63, 3010.6, 2708.76, 2661.58, 3216.83,
2363.68, 1740.12, 2561.46, 2363.19, 2825.42, 2054.6, 1979.61, 2261.22, 2079.41, 2659.03, 2567.5, 3169.21, 2685.55, 2924.68, 3025.55, 3043.83, 2845.74, 3167.29, 2906.02, 3262.75 };

  bool* minima_in = new bool[20*20]
  {false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
   false, false, false, false, false, false, false, false, true, false, false, true, false, false, false, false, false, false, false, false,
   false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
   false, true, false, false, false, false, false, false, false, true, false, false, true, false, false, true, false, false, false, false,
   false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, true, false,
   false, true, false, true, false, false, false, false, true, false, false, true, false, false, false, false, false, false, false, false,
   false, false, false, false, false, true, false, false, false, false, false, false, false, false, true, false, false, false, false, false,
   false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false,
   false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, true, false, false, true, false,
   false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false,
   false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false,
   false, false, true, false, false, true, false, true, false, false, false, false, false, false, true, false, false, true, false, false,
   false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
   false, false, false, false, true, false, false, true, false, false, false, true, false, false, false, false, false, false, false, false,
   false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
   false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, true, false, false,
   false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false,
   false, false, false, false, false, false, false, false, false, false, true, false, false, true, false, false, false, false, false, false,
   false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, true, false,
   false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false };
  double* prior_fine_rdirs_in = new double[20*20] {
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   8,   7,   1,   3,   2,   1,   6,   5,   4,   6,   5,   4,   9,   8,   9,   8,   7,   9,   0,
    0,   2,   1,   4,   6,   5,   4,   9,   8,   7,   1,   3,   2,   1,   3,   2,   1,   9,   8,   0,
    0,   5,   4,   7,   9,   3,   2,   1,   6,   5,   4,   6,   5,   4,   6,   5,   4,   3,   9,   0,
    0,   2,   1,   2,   1,   6,   5,   4,   9,   8,   3,   2,   1,   7,   9,   8,   7,   6,   5,   0,
    0,   5,   4,   5,   4,   9,   8,   7,   5,   3,   6,   5,   3,   3,   2,   1,   1,   9,   8,   0,
    0,   1,   7,   8,   7,   5,   4,   9,   8,   3,   2,   1,   6,   6,   5,   4,   4,   8,   7,   0,
    0,   4,   9,   8,   9,   8,   7,   9,   9,   6,   5,   4,   9,   9,   8,   7,   7,   3,   2,   0,
    0,   7,   1,   2,   1,   4,   3,   2,   1,   9,   8,   7,   5,   9,   6,   5,   3,   6,   5,   0,
    0,   3,   2,   1,   1,   3,   6,   5,   4,   2,   1,   9,   8,   6,   9,   8,   6,   9,   8,   0,
    0,   3,   2,   1,   4,   2,   3,   2,   1,   5,   4,   8,   7,   3,   2,   1,   9,   8,   7,   0,
    0,   6,   5,   4,   7,   5,   6,   5,   4,   2,   1,   2,   1,   6,   5,   3,   6,   5,   4,   0,
    0,   9,   8,   7,   2,   1,   3,   2,   1,   3,   3,   2,   1,   9,   8,   6,   9,   8,   7,   0,
    0,   3,   2,   6,   5,   4,   6,   5,   4,   6,   6,   5,   4,   7,   9,   9,   8,   7,   3,   0,
    0,   3,   2,   9,   8,   7,   9,   8,   7,   9,   9,   8,   7,   4,   9,   3,   3,   2,   3,   0,
    0,   1,   5,   4,   6,   5,   3,   2,   1,   1,   1,   8,   7,   1,   2,   6,   6,   5,   6,   0,
    0,   4,   3,   2,   1,   8,   6,   5,   4,   4,   2,   1,   3,   2,   1,   9,   9,   8,   9,   0,
    0,   3,   2,   1,   4,   3,   9,   8,   7,   7,   5,   4,   6,   5,   4,   2,   9,   8,   7,   0,
    0,   6,   5,   4,   7,   3,   2,   1,   2,   1,   8,   7,   9,   8,   7,   5,   4,   6,   5,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  };
  int* prior_fine_catchments_in = new int[20*20] {
   13597, 13597, 13597, 4948, 4948, 9382, 9382, 8111, 8111, 8111, 7617, 7617, 5129, 5129, 5129, 7623, 7623, 7623, 7185, 7185,
   13597, 13597, 13597, 4948, 8682, 8682, 8682, 8111, 8111, 8111, 7617, 7617, 7617, 5129, 5129, 7623, 7623, 7623, 7185, 7185,
   4948, 4948, 4948, 4948, 8682, 8682, 8682, 8111, 8111, 8111, 6765, 9368, 9368, 9368, 8678, 8678, 8678, 7185, 7185, 8681,
   4948, 4948, 4948, 4948, 8682, 4314, 4314, 4314, 6765, 6765, 6765, 9368, 9368, 9368, 8678, 8678, 8678, 7607, 8681, 8681,
   2237, 2237, 2237, 7168, 7168, 4314, 4314, 4314, 6765, 6765, 11616, 11616, 11616, 9368, 8678, 8678, 8678, 7607, 7607, 7607,
   2237, 2237, 2237, 7168, 7168, 4314, 4314, 4314, 8105, 6425, 11616, 11616, 10268, 10268, 10268, 10268, 10268, 7607, 7607, 7607,
   9363, 9363, 2237, 7168, 7168, 10269, 10269, 8105, 8105, 6425, 6425, 6425, 10268, 10268, 10268, 10268, 10268, 7607, 7607, 5306,
   9363, 9363, 7168, 7168, 10269, 10269, 10269, 8105, 6425, 6425, 6425, 6425, 10268, 10268, 10268, 10268, 10268, 10260, 10260, 10260,
   9363, 9363, 6756, 6756, 6756, 6756, 6423, 6423, 6423, 6425, 6425, 6425, 10262, 10268, 10261, 10261, 10260, 10260, 10260, 10260,
   9363, 6756, 6756, 6756, 6756, 17883, 6423, 6423, 6423, 4762, 4762, 10262, 10262, 10261, 10261, 10261, 10260, 10260, 10260, 10260,
   6756, 6756, 6756, 6756, 6756, 4185, 17883, 17883, 17883, 4762, 4762, 10262, 10262, 8665, 8665, 8665, 10260, 10260, 10260, 10260,
   6756, 6756, 6756, 6756, 6756, 4185, 17883, 17883, 17883, 7595, 7595, 7595, 7595, 8665, 8665, 8664, 8664, 8664, 8664, 10260,
   6756, 6756, 6756, 6756, 10253, 10253, 8095, 8095, 8095, 7595, 7595, 7595, 7595, 8665, 8665, 8664, 8664, 8664, 8664, 6417,
   2551, 9351, 9351, 10253, 10253, 10253, 8095, 8095, 8095, 7595, 7595, 7595, 7595, 7595, 8664, 8664, 8664, 8664, 6417, 6417,
   2551, 9351, 9351, 10253, 10253, 10253, 8095, 8095, 8095, 7595, 7595, 7595, 7595, 7595, 8664, 13551, 13551, 13551, 6417, 6417,
   6748, 6748, 9351, 9351, 9350, 9350, 8656, 8656, 8656, 8656, 8656, 7595, 7595, 9346, 9346, 13551, 13551, 13551, 6417, 6417,
   6748, 6748, 5544, 5544, 5544, 9350, 8656, 8656, 8656, 8656, 17864, 17864, 9346, 9346, 9346, 13551, 13551, 13551, 6417, 6417,
   6748, 5544, 5544, 5544, 5544, 7583, 8656, 8656, 8656, 8656, 17864, 17864, 9346, 9346, 9346, 5828, 13551, 13551, 13551, 10245,
   4610, 5544, 5544, 5544, 5544, 7583, 7583, 7583, 6411, 6411, 17864, 17864, 9346, 9346, 9346, 5828, 5828, 10245, 10245, 10245,
   4610, 5544, 5544, 5544, 7583, 7583, 7583, 7583, 6411, 6411, 8652, 8652, 6737, 6737, 6737, 5828, 5828, 10245, 17856, 17856 };
  double* cell_areas_in = new double[20*20]{
    1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,
//
    1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,
  };
  double* connection_volume_thresholds_in = new double[20*20];
  std::fill_n(connection_volume_thresholds_in,20*20,0.0);
  double* flood_volume_thresholds_in = new double[20*20];
  std::fill_n(flood_volume_thresholds_in,20*20,0.0);
  int* flood_next_cell_lat_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lat_index_in,20*20,-1);
  int* flood_next_cell_lon_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lon_index_in,20*20,-1);
  int* connect_next_cell_lat_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lat_index_in,20*20,-1);
  int* connect_next_cell_lon_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lon_index_in,20*20,-1);
  int* flood_force_merge_lat_index_in = new int[20*20];
  std::fill_n(flood_force_merge_lat_index_in,20*20,-1);
  int* flood_force_merge_lon_index_in = new int[20*20];
  std::fill_n(flood_force_merge_lon_index_in,20*20,-1);
  int* connect_force_merge_lat_index_in = new int[20*20];
  std::fill_n(connect_force_merge_lat_index_in,20*20,-1);
  int* connect_force_merge_lon_index_in = new int[20*20];
  std::fill_n(connect_force_merge_lon_index_in,20*20,-1);
  int* flood_redirect_lat_index_in = new int[20*20];
  std::fill_n(flood_redirect_lat_index_in,20*20,-1);
  int* flood_redirect_lon_index_in = new int[20*20];
  std::fill_n(flood_redirect_lon_index_in,20*20,-1);
  int* connect_redirect_lat_index_in = new int[20*20];
  std::fill_n(connect_redirect_lat_index_in,20*20,-1);
  int* connect_redirect_lon_index_in = new int[20*20];
  std::fill_n(connect_redirect_lon_index_in,20*20,-1);
  bool* flood_local_redirect_in = new bool[20*20];
  std::fill_n(flood_local_redirect_in,20*20,false);
  bool* connect_local_redirect_in = new bool[20*20];
  std::fill_n(connect_local_redirect_in,20*20,false);
  int* additional_flood_redirect_lat_index_in = new int[20*20];
  std::fill_n(additional_flood_redirect_lat_index_in,20*20,-1);
  int* additional_flood_redirect_lon_index_in = new int[20*20];
  std::fill_n(additional_flood_redirect_lon_index_in,20*20,-1);
  int* additional_connect_redirect_lat_index_in = new int[20*20];
  std::fill_n(additional_connect_redirect_lat_index_in,20*20,-1);
  int* additional_connect_redirect_lon_index_in = new int[20*20];
  std::fill_n(additional_connect_redirect_lon_index_in,20*20,-1);
  bool* additional_flood_local_redirect_in = new bool[20*20];
  std::fill_n(additional_flood_local_redirect_in,20*20,false);
  bool* additional_connect_local_redirect_in = new bool[20*20];
  std::fill_n(additional_connect_local_redirect_in,20*20,false);
  merge_types* merge_points_in = new merge_types[20*20];
  std::fill_n(merge_points_in,20*20,no_merge);
//   double* flood_volume_thresholds_expected_out = new double[20*20] {
//    -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,   -1.0,-1.0,
//    -1.0, 2.0,229.0,-1.0,363.0, -1.0,-1.0,-1.0, 2.0,1150.0,   -1.0,-1.0,
//    -1.0, 7.0,278.0,-1.0,231.0, -1.0,-1.0,-1.0, 7.0,866.0,   -1.0,-1.0,
//    -1.0,16.0,375.0,-1.0,189.0, -1.0,-1.0,-1.0, 16.0,726.0,   -1.0,-1.0,
//    -1.0,30.0,477.0,-1.0,115.0, -1.0,-1.0,-1.0, 30.0,589.0, 54.0,30.0,
// //
//    -1.0,50.0,585.0,-1.0,80.0, -1.0,-1.0,-1.0, 50.0,456.0, -1.0,-1.0,
//    -1.0,104.0,700.0,-1.0,50.0, -1.0,-1.0,-1.0, 77.0,338.0, -1.0,-1.0,
//    -1.0,139.0,823.0,-1.0,26.0, -1.0,-1.0,-1.0, 112.0,270.0, -1.0,-1.0,
//    -1.0,183.0,955.0,-1.0, 9.0, -1.0,-1.0,-1.0, 156.0,209.0, -1.0,-1.0,
//    -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,
// //
//    -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,
//    -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0  };
//   double* connection_volume_thresholds_expected_out = new double[20*20]{
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1};
//   int* flood_next_cell_lat_index_expected_out = new int[20*20]{
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1 };
//   int* flood_next_cell_lon_index_expected_out = new int[20*20]{
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1};
//   int* connect_next_cell_lat_index_expected_out = new int[20*20]{
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1 };
//   int* connect_next_cell_lon_index_expected_out = new int[20*20]{
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1 };
//   int* flood_redirect_lat_index_expected_out = new int[20*20]{
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1 };
//   int* flood_redirect_lon_index_expected_out = new int[20*20]{
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1};
//   int* connect_redirect_lat_index_expected_out = new int[20*20]{
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1 };
//   int* connect_redirect_lon_index_expected_out = new int[20*20]{
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1 };
//   bool* flood_local_redirect_expected_out = new bool[20*20]{
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
// //
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
// //
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false };
//   bool* connect_local_redirect_expected_out = new bool[20*20]{
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
// //
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false,
// //
//    false,false,false,false,false, false,false,false,false,false,false,false,
//    false,false,false,false,false, false,false,false,false,false,false,false };
//   merge_types* merge_points_expected_out = new merge_types[20*20]{
//   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
//   no_merge,   no_merge,   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,  no_merge,   no_merge,   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,  no_merge,   no_merge,
//   no_merge,   no_merge,   connection_merge_not_set_flood_merge_as_primary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
//   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
//   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
//   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
//   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,  no_merge,   no_merge,
//   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
//   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,  no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
//   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
//   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,
//   no_merge,   no_merge,  connection_merge_not_set_flood_merge_as_secondary,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge,   no_merge };

//   int* flood_force_merge_lat_index_expected_out = new int[20*20] {

//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1,  4, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1 };
//   int* flood_force_merge_lon_index_expected_out = new int[20*20] {
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1,  3, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1 };
//   int* connect_force_merge_lat_index_expected_out = new int[20*20] {
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1 };
//   int* connect_force_merge_lon_index_expected_out = new int[20*20] {
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
// //
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1,
//     -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1 };
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.setup_fields(minima_in,
                          raw_orography_in,
                          corrected_orography_in,
                          cell_areas_in,
                          connection_volume_thresholds_in,
                          flood_volume_thresholds_in,
                          prior_fine_rdirs_in,
                          prior_fine_catchments_in,
                          coarse_catchment_nums_in,
                          flood_next_cell_lat_index_in,
                          flood_next_cell_lon_index_in,
                          connect_next_cell_lat_index_in,
                          connect_next_cell_lon_index_in,
                          flood_force_merge_lat_index_in,
                          flood_force_merge_lon_index_in,
                          connect_force_merge_lat_index_in,
                          connect_force_merge_lon_index_in,
                          flood_redirect_lat_index_in,
                          flood_redirect_lon_index_in,
                          connect_redirect_lat_index_in,
                          connect_redirect_lon_index_in,
                          additional_flood_redirect_lat_index_in,
                          additional_flood_redirect_lon_index_in,
                          additional_connect_redirect_lat_index_in,
                          additional_connect_redirect_lon_index_in,
                          flood_local_redirect_in,
                          connect_local_redirect_in,
                          additional_flood_local_redirect_in,
                          additional_connect_local_redirect_in,
                          merge_points_in,
                          grid_params_in,
                          coarse_grid_params_in);
  bool* landsea_in = new bool[20*20] {
    true, true, true, true, true, true, true, true, true, true,  true, true, true, true, true,  true, true, true, true, true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
//
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
//
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
//
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true,
    true, true, true, true, true, true, true, true, true, true,  true, true, true, true, true, true, true, true, true, true };
  bool* true_sinks_in = new bool[20*20];
  fill_n(true_sinks_in,20*20,false);
  int* next_cell_lat_index_in = new int[20*20];
  int* next_cell_lon_index_in = new int[20*20];
  short* sinkless_rdirs_out = new short[20*20];
  int* catchment_nums_in = new int[20*20];
  reverse_priority_cell_queue q;
  auto alg4 = new sink_filling_algorithm_4_latlon();
  alg4->setup_flags(false,true,false,false);
  alg4->setup_fields(corrected_orography_in,
                     landsea_in,
                     true_sinks_in,
                     next_cell_lat_index_in,
                     next_cell_lon_index_in,
                     grid_params_in,
                     sinkless_rdirs_out,
                     catchment_nums_in);
  basin_eval.setup_sink_filling_algorithm(alg4);
  basin_eval.evaluate_basins();
  // EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in)
  //             == field<double>(flood_volume_thresholds_expected_out,grid_params_in));
  // EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in)
  //             == field<double>(connection_volume_thresholds_expected_out,grid_params_in));
  // EXPECT_TRUE(field<int>(flood_next_cell_lat_index_in,grid_params_in)
  //             == field<int>(flood_next_cell_lat_index_expected_out,grid_params_in));
  // EXPECT_TRUE(field<int>(flood_next_cell_lon_index_in,grid_params_in)
  //             == field<int>(flood_next_cell_lon_index_expected_out,grid_params_in));
  // EXPECT_TRUE(field<int>(connect_next_cell_lat_index_in,grid_params_in)
  //             == field<int>(connect_next_cell_lat_index_expected_out,grid_params_in));
  // EXPECT_TRUE(field<int>(connect_next_cell_lon_index_in,grid_params_in)
  //             == field<int>(connect_next_cell_lon_index_expected_out,grid_params_in));
  // EXPECT_TRUE(field<int>(flood_redirect_lat_index_in,grid_params_in)
  //             == field<int>(flood_redirect_lat_index_expected_out,grid_params_in));
  // EXPECT_TRUE(field<int>(flood_redirect_lon_index_in,grid_params_in)
  //             == field<int>(flood_redirect_lon_index_expected_out,grid_params_in));
  // EXPECT_TRUE(field<int>(connect_redirect_lat_index_in,grid_params_in)
  //             == field<int>(connect_redirect_lat_index_expected_out,grid_params_in));
  // EXPECT_TRUE(field<int>(connect_redirect_lon_index_in,grid_params_in)
  //             == field<int>(connect_redirect_lon_index_expected_out,grid_params_in));
  // EXPECT_TRUE(field<bool>(flood_local_redirect_in,grid_params_in)
  //             == field<bool>(flood_local_redirect_expected_out,grid_params_in));
  // EXPECT_TRUE(field<bool>(connect_local_redirect_in,grid_params_in)
  //             == field<bool>(connect_local_redirect_expected_out,grid_params_in));
  // EXPECT_TRUE(field<merge_types>(merge_points_in,grid_params_in)
  //             == field<merge_types>(merge_points_expected_out,grid_params_in));
  // EXPECT_TRUE(field<int>(flood_force_merge_lat_index_in,grid_params_in)
  //             == field<int>(flood_force_merge_lat_index_expected_out,grid_params_in));
  // EXPECT_TRUE(field<int>(flood_force_merge_lon_index_in,grid_params_in)
  //             == field<int>(flood_force_merge_lon_index_expected_out,grid_params_in));
  // EXPECT_TRUE(field<int>(connect_force_merge_lat_index_in,grid_params_in)
  //             == field<int>(connect_force_merge_lat_index_expected_out,grid_params_in));
  // EXPECT_TRUE(field<int>(connect_force_merge_lon_index_in,grid_params_in)
  //             == field<int>(connect_force_merge_lon_index_expected_out,grid_params_in));
  // delete grid_params_in; delete coarse_grid_params_in; delete alg4;
  // delete[] coarse_catchment_nums_in; delete[] corrected_orography_in;
  // delete[] raw_orography_in; delete[] minima_in;
  // delete[] prior_fine_rdirs_in; delete[] prior_fine_catchments_in;
  // delete[] connection_volume_thresholds_in; delete[] flood_volume_thresholds_in;
  // delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  // delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  // delete[] flood_force_merge_lat_index_in; delete[] flood_force_merge_lon_index_in;
  // delete[] connect_force_merge_lat_index_in; delete[] connect_force_merge_lon_index_in;
  // delete[] flood_redirect_lat_index_in; delete[] flood_redirect_lon_index_in;
  // delete[] connect_redirect_lat_index_in; delete[] connect_redirect_lon_index_in;
  // delete[] flood_local_redirect_in; delete[] connect_local_redirect_in;
  // delete[] additional_flood_redirect_lat_index_in;
  // delete[] additional_flood_redirect_lon_index_in;
  // delete[] additional_connect_redirect_lat_index_in;
  // delete[] additional_connect_redirect_lon_index_in;
  // delete[] additional_flood_local_redirect_in;
  // delete[] additional_connect_local_redirect_in;
  // delete[] merge_points_in; delete[] flood_volume_thresholds_expected_out;
  // delete[] connection_volume_thresholds_expected_out;
  // delete[] flood_next_cell_lat_index_expected_out;
  // delete[] flood_next_cell_lon_index_expected_out;
  // delete[] connect_next_cell_lat_index_expected_out;
  // delete[] connect_next_cell_lon_index_expected_out;
  // delete[] flood_redirect_lat_index_expected_out;
  // delete[] flood_redirect_lon_index_expected_out;
  // delete[] connect_redirect_lat_index_expected_out;
  // delete[] connect_redirect_lon_index_expected_out;
  // delete[] flood_local_redirect_expected_out;
  // delete[] connect_local_redirect_expected_out;
  // delete[] merge_points_expected_out;
  // delete[] flood_force_merge_lat_index_expected_out;
  // delete[] flood_force_merge_lon_index_expected_out;
  // delete[] connect_force_merge_lat_index_expected_out;
  // delete[] connect_force_merge_lon_index_expected_out;
  // delete[] landsea_in;
  // delete[] true_sinks_in;
  // delete[] next_cell_lat_index_in;
  // delete[] next_cell_lon_index_in;
  // delete[] sinkless_rdirs_out;
  // delete[] catchment_nums_in;
  // delete[] cell_areas_in;
}


} //namespace
