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
  bool* coarse_minima_in = new bool[3*3] {false,false, true,
                                          false, true,false,
                                          true,false,false};
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
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype };
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_add_minima_to_queue(raw_orography_in,
                                          corrected_orography_in,
                                          minima_in,
                                          coarse_minima_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  auto count = 0;
  field<height_types> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(null_htype);
  while (! q.empty()) {
    auto cell = static_cast<basin_cell*>(q.top());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = cell->get_height_type();
    count++;
    q.pop();
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
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
  bool* coarse_minima_in = new bool[3*3] {false,false, true,
                                          false, true,false,
                                          true,false,false};
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
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype };
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_add_minima_to_queue(raw_orography_in,
                                          corrected_orography_in,
                                          minima_in,
                                          coarse_minima_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  auto count = 0;
  field<height_types> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(null_htype);
  while (! q.empty()) {
    auto cell = static_cast<basin_cell*>(q.top());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = cell->get_height_type();
    count++;
    q.pop();
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
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
  bool* coarse_minima_in = new bool[3*3] {false,false, true,
                                          false, true,false,
                                          true,false,false};
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
     null_htype,flood_height,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype };
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_add_minima_to_queue(raw_orography_in,
                                          corrected_orography_in,
                                          minima_in,
                                          coarse_minima_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  auto count = 0;
  field<height_types> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(null_htype);
  while (! q.empty()) {
    auto cell = static_cast<basin_cell*>(q.top());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = cell->get_height_type();
    count++;
    q.pop();
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
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
  bool* coarse_minima_in = new bool[3*3] { true,false, true,
                                          false,false,false,
                                           true,false, true};
  height_types* expected_cells_in_queue = new height_types[9*9]
    {flood_height,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,flood_height,null_htype,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     flood_height,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, flood_height,null_htype,null_htype };
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_add_minima_to_queue(raw_orography_in,
                                          corrected_orography_in,
                                          minima_in,
                                          coarse_minima_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  auto count = 0;
  field<height_types> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(null_htype);
  while (! q.empty()) {
    auto cell = static_cast<basin_cell*>(q.top());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = cell->get_height_type();
    count++;
    q.pop();
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
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
  bool* coarse_minima_in = new bool[3*3] { true,false, true,
                                          false,false,false,
                                           true,false, true};
  height_types* expected_cells_in_queue = new height_types[9*9]
    {null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,flood_height, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,flood_height,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
//
     null_htype,null_htype,flood_height, null_htype,null_htype,null_htype, null_htype,flood_height,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype };
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_add_minima_to_queue(raw_orography_in,
                                          corrected_orography_in,
                                          minima_in,
                                          coarse_minima_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  auto count = 0;
  field<height_types> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(null_htype);
  while (! q.empty()) {
    auto cell = static_cast<basin_cell*>(q.top());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = cell->get_height_type();
    count++;
    q.pop();
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
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
  bool* coarse_minima_in = new bool[3*3] { true,false, true,
                                          false,false,false,
                                           true,false, true};
  height_types* expected_cells_in_queue = new height_types[9*9]
    {null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,flood_height,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
//
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,flood_height,null_htype,
     null_htype,flood_height,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype,
     null_htype,null_htype,null_htype, null_htype,null_htype,null_htype, null_htype,null_htype,null_htype };
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_add_minima_to_queue(raw_orography_in,
                                          corrected_orography_in,
                                          minima_in,
                                          coarse_minima_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  auto count = 0;
  field<height_types> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(null_htype);
  while (! q.empty()) {
    auto cell = static_cast<basin_cell*>(q.top());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = cell->get_height_type();
    count++;
    q.pop();
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
  EXPECT_EQ(3,count);
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
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
  EXPECT_TRUE(field<bool>(completed_cells_in,grid_params_in) == field<bool>(completed_cells_expected_out,grid_params_in));
  EXPECT_EQ(7,count);
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
  center_coords_in = new latlon_coords(4,8);
  q = basin_eval.test_process_neighbors(center_coords_in,
                                        completed_cells_in,
                                        raw_orography_in,
                                        corrected_orography_in,
                                        grid_params_in);
  center_coords_in = new latlon_coords(7,4);
  q = basin_eval.test_process_neighbors(center_coords_in,
                                        completed_cells_in,
                                        raw_orography_in,
                                        corrected_orography_in,
                                        grid_params_in);
  center_coords_in = new latlon_coords(5,1);
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
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
  EXPECT_TRUE(field<bool>(completed_cells_in,grid_params_in) == field<bool>(completed_cells_expected_out,grid_params_in));
  EXPECT_EQ(26,count);
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
  center_coords_in = new latlon_coords(8,4);
  q = basin_eval.test_process_neighbors(center_coords_in,
                                        completed_cells_in,
                                        raw_orography_in,
                                        corrected_orography_in,
                                        grid_params_in);
  center_coords_in = new latlon_coords(4,8);
  q = basin_eval.test_process_neighbors(center_coords_in,
                                        completed_cells_in,
                                        raw_orography_in,
                                        corrected_orography_in,
                                        grid_params_in);
  center_coords_in = new latlon_coords(8,8);
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
  }
  EXPECT_TRUE(cells_in_queue == field<height_types>(expected_cells_in_queue,grid_params_in));
  EXPECT_TRUE(field<bool>(completed_cells_in,grid_params_in) == field<bool>(completed_cells_expected_out,grid_params_in));
  EXPECT_EQ(23,count);
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
  search_coords_in = new latlon_coords(7,1);
  q = basin_eval.test_search_process_neighbors(search_coords_in,
                                               completed_cells_in,
                                               grid_params_in);
  search_coords_in = new latlon_coords(0,4);
  q = basin_eval.test_search_process_neighbors(search_coords_in,
                                               completed_cells_in,
                                               grid_params_in);
  search_coords_in = new latlon_coords(5,8);
  q = basin_eval.test_search_process_neighbors(search_coords_in,
                                               completed_cells_in,
                                               grid_params_in);
  auto count = 0;
  field<bool> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(false);
  while (! q.empty()) {
    auto cell = static_cast<landsea_cell*>(q.front());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = true;
    count++;
    q.pop();
  }
  EXPECT_TRUE(cells_in_queue == field<bool>(expected_cells_in_queue,grid_params_in));
  EXPECT_TRUE(field<bool>(completed_cells_in,grid_params_in) == field<bool>(completed_cells_expected_out,grid_params_in));
  EXPECT_EQ(23,count);
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
  search_coords_in = new latlon_coords(4,0);
  q = basin_eval.test_search_process_neighbors(search_coords_in,
                                               completed_cells_in,
                                               grid_params_in);
  search_coords_in = new latlon_coords(0,4);
  q = basin_eval.test_search_process_neighbors(search_coords_in,
                                               completed_cells_in,
                                               grid_params_in);
  search_coords_in = new latlon_coords(8,8);
  q = basin_eval.test_search_process_neighbors(search_coords_in,
                                               completed_cells_in,
                                               grid_params_in);
  auto count = 0;
  field<bool> cells_in_queue(grid_params_in);
  cells_in_queue.set_all(false);
  while (! q.empty()) {
    auto cell = static_cast<landsea_cell*>(q.front());
    auto coords = cell->get_cell_coords();
    cells_in_queue(coords) = true;
    count++;
    q.pop();
  }
  EXPECT_TRUE(cells_in_queue == field<bool>(expected_cells_in_queue,grid_params_in));
  EXPECT_TRUE(field<bool>(completed_cells_in,grid_params_in) == field<bool>(completed_cells_expected_out,grid_params_in));
  EXPECT_EQ(26,count);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCell) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,0.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,9.1,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 67;
  int cell_number_in = 6;
  double center_cell_height_in = 1.5;
  double center_cell_volume_threshold_in = 8.5;
  double previous_filled_cell_height_in = 1.4;
  coords* previous_center_coords_in = new latlon_coords(7,1);
  coords* center_coords_in = new latlon_coords(7,2);
  basin_cell* center_cell_in = new basin_cell(1.4,connection_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  basin_cell* qcell = static_cast<basin_cell*>(q.top());
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lat(),7);
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lon(),2);
  EXPECT_EQ(static_cast<basin_cell*>(qcell)->get_height_type(),flood_height);
  q.pop();
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 9.1)<1E-7);
  EXPECT_EQ(cell_number_in,6);
  EXPECT_EQ(previous_filled_cell_height_in,1.5);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellTwo) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,0.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,9.1,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 67;
  int cell_number_in = 6;
  double center_cell_height_in = 1.5;
  double center_cell_volume_threshold_in = 8.5;
  double previous_filled_cell_height_in = 1.4;
  coords* previous_center_coords_in = new latlon_coords(7,1);
  coords* center_coords_in = new latlon_coords(7,2);
  basin_cell* center_cell_in = new basin_cell(1.4,flood_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 9.1)<1E-7);
  EXPECT_EQ(cell_number_in,7);
  EXPECT_EQ(previous_filled_cell_height_in,1.5);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellThree) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,0.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,9.1, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 67;
  int cell_number_in = 6;
  double center_cell_height_in = 1.5;
  double center_cell_volume_threshold_in = 8.5;
  double previous_filled_cell_height_in = 1.4;
  coords* previous_center_coords_in = new latlon_coords(7,2);
  coords* center_coords_in = new latlon_coords(7,2);
  basin_cell* center_cell_in = new basin_cell(1.4,flood_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 9.1)<1E-7);
  EXPECT_EQ(cell_number_in,7);
  EXPECT_EQ(previous_filled_cell_height_in,1.5);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellFour) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,0.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             2.1,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 31;
  int cell_number_in = 2;
  double center_cell_height_in = 2.1;
  double center_cell_volume_threshold_in = 1.5;
  double previous_filled_cell_height_in = 1.8;
  coords* previous_center_coords_in = new latlon_coords(3,0);
  coords* center_coords_in = new latlon_coords(0,3);
  basin_cell* center_cell_in = new basin_cell(1.8,connection_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  basin_cell* qcell = static_cast<basin_cell*>(q.top());
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lat(),0);
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lon(),3);
  EXPECT_EQ(static_cast<basin_cell*>(qcell)->get_height_type(),flood_height);
  q.pop();
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 2.1)<1E-7);
  EXPECT_EQ(cell_number_in,2);
  EXPECT_EQ(previous_filled_cell_height_in,2.1);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellFive) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,0.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             2.1,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 31;
  int cell_number_in = 2;
  double center_cell_height_in = 2.1;
  double center_cell_volume_threshold_in = 1.5;
  double previous_filled_cell_height_in = 1.8;
  coords* previous_center_coords_in = new latlon_coords(3,0);
  coords* center_coords_in = new latlon_coords(0,3);
  basin_cell* center_cell_in = new basin_cell(1.8,flood_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 2.1)<1E-7);
  EXPECT_EQ(cell_number_in,3);
  EXPECT_EQ(previous_filled_cell_height_in,2.1);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellSix) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,0.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             2.1,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 31;
  int cell_number_in = 2;
  double center_cell_height_in = 2.1;
  double center_cell_volume_threshold_in = 1.5;
  double previous_filled_cell_height_in = 1.8;
  coords* previous_center_coords_in = new latlon_coords(3,0);
  coords* center_coords_in = new latlon_coords(3,0);
  basin_cell* center_cell_in = new basin_cell(1.8,flood_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 2.1)<1E-7);
  EXPECT_EQ(cell_number_in,3);
  EXPECT_EQ(previous_filled_cell_height_in,2.1);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellSeven) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,0.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 67;
  int cell_number_in = 0;
  double center_cell_height_in = 1.1;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 0.0;
  coords* previous_center_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,2);
  basin_cell* center_cell_in = new basin_cell(1.1,connection_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  basin_cell* qcell = static_cast<basin_cell*>(q.top());
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lat(),8);
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lon(),2);
  EXPECT_EQ(static_cast<basin_cell*>(qcell)->get_height_type(),flood_height);
  q.pop();
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.0)<1E-7);
  EXPECT_EQ(cell_number_in,0);
  EXPECT_EQ(previous_filled_cell_height_in,1.1);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellEight) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,0.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 67;
  int cell_number_in = 0;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_center_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,2);
  basin_cell* center_cell_in = new basin_cell(1.1,flood_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.0)<1E-7);
  EXPECT_EQ(cell_number_in,1);
  EXPECT_EQ(previous_filled_cell_height_in,1.3);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellNine) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,0.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.2, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 67;
  int cell_number_in = 1;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_center_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,flood_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(cell_number_in,2);
  EXPECT_EQ(previous_filled_cell_height_in,1.3);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellTen) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9];
  std::fill_n(flood_volume_thresholds_in,9*9,0.0);
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.2, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 67;
  int cell_number_in = 1;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_center_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,connection_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  basin_cell* qcell = static_cast<basin_cell*>(q.top());
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lat(),8);
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lon(),1);
  EXPECT_EQ(static_cast<basin_cell*>(qcell)->get_height_type(),flood_height);
  q.pop();
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(cell_number_in,1);
  EXPECT_EQ(previous_filled_cell_height_in,1.3);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellEleven) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.5, 0.0,0.0,0.0, 0.0,0.0,0.0};
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.5, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 67;
  int cell_number_in = 1;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_center_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,connection_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.0)<1E-7);
  EXPECT_EQ(cell_number_in,2);
  EXPECT_EQ(previous_filled_cell_height_in,1.1);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellTwelve) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.5, 0.0,0.0,0.0, 0.0,0.0,0.0};
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.5, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 67;
  int cell_number_in = 1;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_center_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,flood_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.0)<1E-7);
  EXPECT_EQ(cell_number_in,2);
  EXPECT_EQ(previous_filled_cell_height_in,1.1);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellThirteen) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.5, 0.0,0.0,0.0, 0.0,0.0,0.0};
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.2, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.5, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 67;
  int cell_number_in = 1;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_center_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,connection_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  basin_cell* qcell = static_cast<basin_cell*>(q.top());
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lat(),8);
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lon(),1);
  EXPECT_EQ(static_cast<basin_cell*>(qcell)->get_height_type(),flood_height);
  q.pop();
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(cell_number_in,1);
  EXPECT_EQ(previous_filled_cell_height_in,1.3);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellFourteen) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.5, 0.0,0.0,0.0, 0.0,0.0,0.0};
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.2, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 67;
  int cell_number_in = 1;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_center_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,flood_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(cell_number_in,2);
  EXPECT_EQ(previous_filled_cell_height_in,1.3);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellFifteen) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.2, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 67;
  int cell_number_in = 1;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_center_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,connection_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  basin_cell* qcell = static_cast<basin_cell*>(q.top());
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lat(),8);
  EXPECT_EQ(static_cast<latlon_coords*>(qcell->get_cell_coords())->get_lon(),1);
  EXPECT_EQ(static_cast<basin_cell*>(qcell)->get_height_type(),flood_height);
  q.pop();
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(cell_number_in,1);
  EXPECT_EQ(previous_filled_cell_height_in,1.3);
}

TEST_F(BasinEvaluationTest, TestProcessingCenterCellSixteen) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  double* flood_volume_thresholds_in      = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};
  double* connection_volume_thresholds_in = new double[9*9];
  std::fill_n(connection_volume_thresholds_in,9*9,0.0);
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
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
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
  int* next_cell_lat_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_expected_out = new int[9*9] {-1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  double* connection_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};

  double* flood_volume_thresholds_expected_out = new double[9*9]
                                                            {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
//
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0,
                                                             0.0,0.0,0.2, 0.0,0.0,0.0, 0.0,0.0,0.0};
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
  int basin_catchment_number_in = 67;
  int cell_number_in = 1;
  double center_cell_height_in = 1.3;
  double center_cell_volume_threshold_in = 0.0;
  double previous_filled_cell_height_in = 1.1;
  coords* previous_center_coords_in = new latlon_coords(8,2);
  coords* center_coords_in = new latlon_coords(8,1);
  basin_cell* center_cell_in = new basin_cell(1.3,flood_height,center_coords_in);
  priority_cell_queue q;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  q = basin_eval.test_process_center_cell(center_cell_in,
                                          center_coords_in,
                                          previous_center_coords_in,
                                          flood_volume_thresholds_in,
                                          connection_volume_thresholds_in,
                                          raw_orography_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          basin_catchment_numbers_in,
                                          center_cell_volume_threshold_in,
                                          cell_number_in,
                                          basin_catchment_number_in,
                                          center_cell_height_in,
                                          previous_filled_cell_height_in,
                                          grid_params_in);
  EXPECT_TRUE(q.empty());
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in) == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in) == field<int>(next_cell_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in)
              == field<int>(basin_catchment_numbers_expected_out,grid_params_in));
  EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in).almost_equal(field<double>(connection_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),1E-7));
  EXPECT_TRUE(fabs(center_cell_volume_threshold_in - 0.2)<1E-7);
  EXPECT_EQ(cell_number_in,2);
  EXPECT_EQ(previous_filled_cell_height_in,1.3);
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectOne) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1,  5,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, 2,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* local_redirect_lat_index_expected_out = new int[9*9]
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
  int* local_redirect_lon_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, 2,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, 1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  coords* center_coords_in = new latlon_coords(5,2);
  coords* previous_center_coords_in = new latlon_coords(6,3);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectTwo) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  0,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* force_merge_lon_index_expected_out = new int[9*9]
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
  int* local_redirect_lat_index_expected_out = new int[9*9]
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
  int* local_redirect_lon_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
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
  coords* center_coords_in = new latlon_coords(0,5);
  coords* previous_center_coords_in = new latlon_coords(1,3);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectThree) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, 3,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, 7,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* local_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1,  4,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* local_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, 8,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
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
  coords* center_coords_in = new latlon_coords(3,7);
  coords* previous_center_coords_in = new latlon_coords(3,6);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectFour) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1,  3,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1,  8,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* local_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1,  4,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* local_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1,  8,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
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
  coords* center_coords_in = new latlon_coords(3,8);
  coords* previous_center_coords_in = new latlon_coords(5,6);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectFive) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1, 2, -1,-1,-1,
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
  int* force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1, 2, -1,-1,-1,
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
  int* local_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1, 1, -1,-1,-1,
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
  int* local_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1, 1, -1,-1,-1,
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
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
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
  coords* center_coords_in = new latlon_coords(2,2);
  coords* previous_center_coords_in = new latlon_coords(0,5);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectSix) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                5,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* local_redirect_lat_index_expected_out = new int[9*9]
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
  int* local_redirect_lon_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                2,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  coords* center_coords_in = new latlon_coords(5,1);
  coords* previous_center_coords_in = new latlon_coords(6,0);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectSeven) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
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
  int* force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1, 1,-1, -1,-1,-1,
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
  int* local_redirect_lat_index_expected_out = new int[9*9]
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
  int* local_redirect_lon_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1, 1,-1, -1,-1,-1,
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
  coords* center_coords_in = new latlon_coords(5,1);
  coords* previous_center_coords_in = new latlon_coords(0,4);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectEight) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
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
                                               -1,-1,-1,  -1,-1,-1, -1,-1,0};
  int* force_merge_lon_index_expected_out = new int[9*9]
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
                                               -1,-1,-1,  -1,-1,-1, -1,-1,7};
  int* local_redirect_lat_index_expected_out = new int[9*9]
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
  int* local_redirect_lon_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
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
                                               -1,-1,-1,  -1,-1,-1, -1,-1,0};
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
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
                                               -1,-1,-1,  -1,-1,-1, -1,-1,2};
  coords* center_coords_in = new latlon_coords(0,7);
  coords* previous_center_coords_in = new latlon_coords(8,8);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectNine) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1,  5,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, 2,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* local_redirect_lat_index_expected_out = new int[9*9]
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
  int* local_redirect_lon_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, 2,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, 1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  coords* center_coords_in = new latlon_coords(5,2);
  coords* previous_center_coords_in = new latlon_coords(6,3);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectTen) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  0,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* force_merge_lon_index_expected_out = new int[9*9]
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
  int* local_redirect_lat_index_expected_out = new int[9*9]
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
  int* local_redirect_lon_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
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
  coords* center_coords_in = new latlon_coords(0,5);
  coords* previous_center_coords_in = new latlon_coords(1,3);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectEleven) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, 3,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, 7,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* local_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1,  4,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* local_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, 8,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
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
  coords* center_coords_in = new latlon_coords(3,7);
  coords* previous_center_coords_in = new latlon_coords(3,6);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectTwelve) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1,  3,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1,  8,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* local_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1,  4,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* local_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1,  8,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
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
  coords* center_coords_in = new latlon_coords(3,8);
  coords* previous_center_coords_in = new latlon_coords(5,6);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectThirteen) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1, 2, -1,-1,-1,
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
  int* force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1, 2, -1,-1,-1,
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
  int* local_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1, 1, -1,-1,-1,
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
  int* local_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1, 1, -1,-1,-1,
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
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
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
  coords* center_coords_in = new latlon_coords(2,2);
  coords* previous_center_coords_in = new latlon_coords(0,5);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectFourteen) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                5,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* local_redirect_lat_index_expected_out = new int[9*9]
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
  int* local_redirect_lon_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                2,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                                1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1, -1,-1,-1};
  coords* center_coords_in = new latlon_coords(5,1);
  coords* previous_center_coords_in = new latlon_coords(6,0);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectFifteen) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
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
  int* force_merge_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1, 1,-1, -1,-1,-1,
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
  int* local_redirect_lat_index_expected_out = new int[9*9]
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
  int* local_redirect_lon_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1, 1,-1, -1,-1,-1,
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
  coords* center_coords_in = new latlon_coords(5,1);
  coords* previous_center_coords_in = new latlon_coords(0,4);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectSixteen) {
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
  int* force_merge_lat_index_in = new int[9*9];
  std::fill_n(force_merge_lat_index_in,9*9,-1);
  int* force_merge_lon_index_in = new int[9*9];
  std::fill_n(force_merge_lon_index_in,9*9,-1);
  int* local_redirect_lat_index_in = new int[9*9];
  std::fill_n(local_redirect_lat_index_in,9*9,-1);
  int* local_redirect_lon_index_in = new int[9*9];
  std::fill_n(local_redirect_lon_index_in,9*9,-1);
  int* non_local_redirect_lat_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lat_index_in,9*9,-1);
  int* non_local_redirect_lon_index_in = new int[9*9];
  std::fill_n(non_local_redirect_lon_index_in,9*9,-1);
  int* force_merge_lat_index_expected_out = new int[9*9]
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
                                               -1,-1,-1,  -1,-1,-1, -1,-1,0};
  int* force_merge_lon_index_expected_out = new int[9*9]
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
                                               -1,-1,-1,  -1,-1,-1, -1,-1,7};
  int* local_redirect_lat_index_expected_out = new int[9*9]
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
  int* local_redirect_lon_index_expected_out = new int[9*9]
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
  int* non_local_redirect_lat_index_expected_out = new int[9*9]
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
                                               -1,-1,-1,  -1,-1,-1, -1,-1,0};
  int* non_local_redirect_lon_index_expected_out = new int[9*9]
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
                                               -1,-1,-1,  -1,-1,-1, -1,-1,2};
  coords* center_coords_in = new latlon_coords(0,7);
  coords* previous_center_coords_in = new latlon_coords(8,8);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest,TestSettingSecondaryRedirect) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
  bool* requires_redirect_indices_in = new bool[9*9];
  std::fill_n(requires_redirect_indices_in,9*9,false);
  int* next_cell_lat_index_expected_out = new int[9*9]
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
  int* next_cell_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,1,-1, -1,-1,-1,
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
  coords* center_coords_in = new latlon_coords(5,1);
  coords* previous_center_coords_in = new latlon_coords(0,4);
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_secondary_redirect(next_cell_lat_index_in,next_cell_lon_index_in,
                                         requires_redirect_indices_in,
                                         center_coords_in,previous_center_coords_in,
                                         grid_params_in);
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in)
              == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in)
              == field<int>(next_cell_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest,TestSettingSecondaryRedirectTwo) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  int* next_cell_lat_index_in = new int[9*9];
  std::fill_n(next_cell_lat_index_in,9*9,-1);
  int* next_cell_lon_index_in = new int[9*9];
  std::fill_n(next_cell_lon_index_in,9*9,-1);
  bool* requires_redirect_indices_in = new bool[9*9];
  std::fill_n(requires_redirect_indices_in,9*9,false);
  int* next_cell_lat_index_expected_out = new int[9*9]
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
  int* next_cell_lon_index_expected_out = new int[9*9]
                                              {-1,-1,-1, -1,1,-1, -1,-1,-1,
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
  coords* center_coords_in = new latlon_coords(5,1);
  coords* previous_center_coords_in = new latlon_coords(0,4);
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_secondary_redirect(next_cell_lat_index_in,next_cell_lon_index_in,
                                         requires_redirect_indices_in,
                                         center_coords_in,previous_center_coords_in,
                                         grid_params_in);
  EXPECT_TRUE(field<int>(next_cell_lat_index_in,grid_params_in)
              == field<int>(next_cell_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(next_cell_lon_index_in,grid_params_in)
              == field<int>(next_cell_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest,TestSettingRemainingRedirects) {
  auto grid_params_in = new latlon_grid_params(12,12,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* local_redirect_lat_index_in = new int[12*12];
  std::fill_n(local_redirect_lat_index_in,12*12,-1);
  int* local_redirect_lon_index_in = new int[12*12];
  std::fill_n(local_redirect_lon_index_in,12*12,-1);
  int* non_local_redirect_lat_index_in = new int[12*12];
  std::fill_n(non_local_redirect_lat_index_in,12*12,-1);
  int* non_local_redirect_lon_index_in = new int[12*12];
  std::fill_n(non_local_redirect_lon_index_in,12*12,-1);
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
                                     0,0,0, 1,1,1, 4,4,4,  0,0,0,
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
  bool* requires_redirect_indices_in = new bool[12*12]
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
  int* next_cell_lat_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1,  2,-1,-1,
                                     -1,-1,-1, -1, 2,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* next_cell_lon_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, 11,-1,-1,
                                     -1,-1,-1, -1, 3,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* local_redirect_lat_index_expected_out = new int[12*12]
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
  int* local_redirect_lon_index_expected_out = new int[12*12]
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
  int* non_local_redirect_lat_index_expected_out = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, 1,-1,-1,
                                     -1,-1,-1, -1, 1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* non_local_redirect_lon_index_expected_out = new int[12*12]
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
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 3,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
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
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_remaining_redirects(basin_catchment_centers_in,
                                          prior_fine_rdirs_in,
                                          requires_redirect_indices_in,
                                          basin_catchment_numbers_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          local_redirect_lat_index_in,
                                          local_redirect_lon_index_in,
                                          non_local_redirect_lat_index_in,
                                          non_local_redirect_lon_index_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest,TestSettingRemainingRedirectsTwo) {
  auto grid_params_in = new latlon_grid_params(12,12,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* local_redirect_lat_index_in = new int[12*12];
  std::fill_n(local_redirect_lat_index_in,12*12,-1);
  int* local_redirect_lon_index_in = new int[12*12];
  std::fill_n(local_redirect_lon_index_in,12*12,-1);
  int* non_local_redirect_lat_index_in = new int[12*12];
  std::fill_n(non_local_redirect_lat_index_in,12*12,-1);
  int* non_local_redirect_lon_index_in = new int[12*12];
  std::fill_n(non_local_redirect_lon_index_in,12*12,-1);
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
  bool* requires_redirect_indices_in = new bool[12*12]
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
  int* next_cell_lat_index_in = new int[12*12]
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
  int* next_cell_lon_index_in = new int[12*12]
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
  int* local_redirect_lat_index_expected_out = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
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
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* local_redirect_lon_index_expected_out = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
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
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* non_local_redirect_lat_index_expected_out = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, 1,-1,-1,
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
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 2,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* non_local_redirect_lon_index_expected_out = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1,  3,-1,-1,
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
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 2,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
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
                                          requires_redirect_indices_in,
                                          basin_catchment_numbers_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          next_cell_lat_index_in,
                                          next_cell_lon_index_in,
                                          local_redirect_lat_index_in,
                                          local_redirect_lon_index_in,
                                          non_local_redirect_lat_index_in,
                                          non_local_redirect_lon_index_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectWrap) {
  auto grid_params_in = new latlon_grid_params(12,12,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* basin_catchment_numbers_in = new int[12*12] {3,3,3, 3,3,8, 8,8,8, 8,1,1,
                                                  3,3,3, 3,8,8, 8,9,9, 1,1,1,
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
  int* force_merge_lat_index_in = new int[12*12];
  std::fill_n(force_merge_lat_index_in,12*12,-1);
  int* force_merge_lon_index_in = new int[12*12];
  std::fill_n(force_merge_lon_index_in,12*12,-1);
  int* local_redirect_lat_index_in = new int[12*12];
  std::fill_n(local_redirect_lat_index_in,12*12,-1);
  int* local_redirect_lon_index_in = new int[12*12];
  std::fill_n(local_redirect_lon_index_in,12*12,-1);
  int* non_local_redirect_lat_index_in = new int[12*12];
  std::fill_n(non_local_redirect_lat_index_in,12*12,-1);
  int* non_local_redirect_lon_index_in = new int[12*12];
  std::fill_n(non_local_redirect_lon_index_in,12*12,-1);
  int* force_merge_lat_index_expected_out = new int[12*12]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,-1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,-1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,-1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,-1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1,-1,-1,0,-1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* force_merge_lon_index_expected_out = new int[12*12]
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
                                               -1,-1,-1, -1,-1,-1, -1,-1, 9, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* local_redirect_lat_index_expected_out = new int[12*12]
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
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* local_redirect_lon_index_expected_out = new int[12*12]
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
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* non_local_redirect_lat_index_expected_out = new int[12*12]
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
                                               -1,-1,-1,  -1,-1,-1, -1,-1,0, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* non_local_redirect_lon_index_expected_out = new int[12*12]
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
                                               -1,-1,-1, -1,-1,-1, -1,-1, 0, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  coords* center_coords_in = new latlon_coords(0,9);
  coords* previous_center_coords_in = new latlon_coords(8,8);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryRedirectNoWrap) {
  auto grid_params_in = new latlon_grid_params(12,12,true);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,true);
  int* basin_catchment_numbers_in = new int[12*12] {3,3,3, 3,3,8, 8,8,8, 8,1,1,
                                                  3,3,3, 3,8,8, 8,9,9, 1,1,1,
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
  int* force_merge_lat_index_in = new int[12*12];
  std::fill_n(force_merge_lat_index_in,12*12,-1);
  int* force_merge_lon_index_in = new int[12*12];
  std::fill_n(force_merge_lon_index_in,12*12,-1);
  int* local_redirect_lat_index_in = new int[12*12];
  std::fill_n(local_redirect_lat_index_in,12*12,-1);
  int* local_redirect_lon_index_in = new int[12*12];
  std::fill_n(local_redirect_lon_index_in,12*12,-1);
  int* non_local_redirect_lat_index_in = new int[12*12];
  std::fill_n(non_local_redirect_lat_index_in,12*12,-1);
  int* non_local_redirect_lon_index_in = new int[12*12];
  std::fill_n(non_local_redirect_lon_index_in,12*12,-1);
  int* force_merge_lat_index_expected_out = new int[12*12]
                                              {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,-1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,-1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,-1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1,-1,-1,-1,
                                               -1,-1,-1,  -1,-1,-1,-1,-1,0,-1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* force_merge_lon_index_expected_out = new int[12*12]
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
                                               -1,-1,-1, -1,-1,-1, -1,-1, 9, -1,-1,-1,
//
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* local_redirect_lat_index_expected_out = new int[12*12]
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
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* local_redirect_lon_index_expected_out = new int[12*12]
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
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* non_local_redirect_lat_index_expected_out = new int[12*12]
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
                                               -1,-1,-1,  -1,-1,-1, -1,-1,2, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* non_local_redirect_lon_index_expected_out = new int[12*12]
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
                                               -1,-1,-1, -1,-1,-1, -1,-1, 2, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                               -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  coords* center_coords_in = new latlon_coords(0,9);
  coords* previous_center_coords_in = new latlon_coords(8,8);
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
  basin_eval.test_set_primary_redirect(basin_catchment_centers_in,
                                       basin_catchment_numbers_in,
                                       coarse_catchment_nums_in,
                                       force_merge_lat_index_in,
                                       force_merge_lon_index_in,
                                       local_redirect_lat_index_in,
                                       local_redirect_lon_index_in,
                                       non_local_redirect_lat_index_in,
                                       non_local_redirect_lon_index_in,
                                       center_coords_in,
                                       previous_center_coords_in,
                                       grid_params_in,
                                       coarse_grid_params_in);
  EXPECT_TRUE(field<int>(force_merge_lat_index_in,grid_params_in)
              == field<int>(force_merge_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(force_merge_lon_index_in,grid_params_in)
              == field<int>(force_merge_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lat_index_in,grid_params_in)
              == field<int>(local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(local_redirect_lon_index_in,grid_params_in)
              == field<int>(local_redirect_lon_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lat_index_in,grid_params_in)
              == field<int>(non_local_redirect_lat_index_expected_out,grid_params_in));
  EXPECT_TRUE(field<int>(non_local_redirect_lon_index_in,grid_params_in)
              == field<int>(non_local_redirect_lon_index_expected_out,grid_params_in));
}

} //namespace
