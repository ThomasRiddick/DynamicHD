/*
 * test_evaluate_basins.cpp
 *
 * Unit test for the basin evaluation C++ code using Google's
 * Google Test Framework
 *
 *  Created on: Mar 11, 2018
 *      Author: thomasriddick using Google's recommended template code
 */

#include "algorithms/basin_evaluation_algorithm.hpp"
#include "algorithms/catchment_computation_algorithm.hpp"
#include "drivers/evaluate_basins.hpp"
#include "gtest/gtest.h"
#include "base/enums.hpp"
#include <cmath>

namespace basin_evaluation_unittests {

// Convert old data to new merge scheme
enum merge_types {no_merge = 0,
                  connection_merge_as_primary_flood_merge_as_primary = 1,
                  connection_merge_as_primary_flood_merge_as_secondary,
                  connection_merge_as_primary_flood_merge_not_set,
                  connection_merge_as_primary_flood_merge_as_both,
                  connection_merge_as_secondary_flood_merge_as_primary,
                  connection_merge_as_secondary_flood_merge_as_secondary,
                  connection_merge_as_secondary_flood_merge_not_set,
                  connection_merge_as_secondary_flood_merge_as_both,
                  connection_merge_not_set_flood_merge_as_primary,
                  connection_merge_not_set_flood_merge_as_secondary,
                  connection_merge_not_set_flood_merge_as_both,
                  connection_merge_as_both_flood_merge_as_primary,
                  connection_merge_as_both_flood_merge_as_secondary,
                  connection_merge_as_both_flood_merge_not_set,
                  connection_merge_as_both_flood_merge_as_both,
                  null_mtype};

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
  // Test adding three minima to the queue for 9 by 9 array
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  double* raw_orography_in = new double[9*9] {2.0,3.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              3.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  double* corrected_orography_in = new double[9*9] {0.0,3.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    3.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
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
  int* prior_fine_catchments_in = new int[9*9];
  fill_n(prior_fine_catchments_in,9*9,1);
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
                                          prior_fine_catchments_in,
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
  delete[] prior_fine_catchments_in;
}

TEST_F(BasinEvaluationTest, TestAddingMinimaToQueueTwo) {
  // Test adding three minima to the queue for 9 by 9 array where flood and
  // connect heights sometimes differ
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  double* raw_orography_in = new double[9*9] {2.0,3.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              3.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  double* corrected_orography_in = new double[9*9] {2.0,3.0,2.0, 2.0,2.0,2.0, 1.0,2.0,2.0,
                                                    3.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
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
  int* prior_fine_catchments_in = new int[9*9];
  fill_n(prior_fine_catchments_in,9*9,1);
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
                                          prior_fine_catchments_in,
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
  delete[] prior_fine_catchments_in;
}

TEST_F(BasinEvaluationTest, TestAddingMinimaToQueueThree) {
  // Test adding four minima to the queue for 9 by 9 array where flood and
  // connect heights sometimes differ
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  double* raw_orography_in = new double[9*9] {2.0,3.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              3.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  double* corrected_orography_in = new double[9*9] {2.0,3.0,2.0, 2.0,2.0,2.0, 1.0,2.0,2.0,
                                                    3.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
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
  int* prior_fine_catchments_in = new int[9*9];
  fill_n(prior_fine_catchments_in,9*9,1);
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
                                          prior_fine_catchments_in,
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
  delete[] prior_fine_catchments_in;
}

TEST_F(BasinEvaluationTest, TestAddingMinimaToQueueFour) {
  // Test adding four minima to the queue for 9 by 9 array where flood and
  // connect heights sometimes differ; some minima are place next to edges of
  // the array
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
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,3.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,3.0,0.0};
  double* corrected_orography_in = new double[9*9] {2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,3.0,
                                                    2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,3.0,0.0};
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
  int* prior_fine_catchments_in = new int[9*9];
  fill_n(prior_fine_catchments_in,9*9,1);
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
                                          prior_fine_catchments_in,
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
  delete[] prior_fine_catchments_in;
}

TEST_F(BasinEvaluationTest, TestAddingMinimaToQueueFive) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  double* raw_orography_in = new double[9*9] {2.0,3.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              3.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  double* corrected_orography_in = new double[9*9] {2.0,3.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    3.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
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
  int* prior_fine_catchments_in = new int[9*9];
  fill_n(prior_fine_catchments_in,9*9,1);
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
                                          prior_fine_catchments_in,
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
  delete[] prior_fine_catchments_in;
}

TEST_F(BasinEvaluationTest, TestAddingMinimaToQueueSix) {
  // Test adding four minima to the queue for 9 by 9 array; one next to
  // an edge
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  double* raw_orography_in = new double[9*9] {2.0,3.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              3.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
//
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                              2.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0};
  double* corrected_orography_in = new double[9*9] {2.0,3.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
                                                    3.0,2.0,2.0, 2.0,2.0,2.0, 2.0,2.0,2.0,
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
  int* prior_fine_catchments_in = new int[9*9];
  fill_n(prior_fine_catchments_in,9*9,1);
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
                                          prior_fine_catchments_in,
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
  delete[] prior_fine_catchments_in;
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
  delete previous_filled_cell_coords_in;
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
  delete previous_filled_cell_coords_in;
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
  delete previous_filled_cell_coords_in;
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
  delete previous_filled_cell_coords_in;
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
  delete previous_filled_cell_coords_in;
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
  delete previous_filled_cell_coords_in;
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
  delete previous_filled_cell_coords_in;
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
  delete previous_filled_cell_coords_in;
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectOne) {
  //A single primary merge with a simply non-local redirect
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(8,4),
                                                new latlon_coords(8,4),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(8,6),
                                                new latlon_coords(2,1),
                                                false);
  secondary_merge = nullptr;
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,3);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  coords* new_center_coords_in = new latlon_coords(5,2);
  coords* center_coords_in = new latlon_coords(6,3);
  coords* previous_filled_cell_coords_in = new latlon_coords(8,3);
  height_types previous_filled_cell_height_type_in = flood_height;
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  int* basin_catchment_numbers_in = new int[9*9] {3,3,3, 3,3,8, 8,9,9,
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,8),
                                                new latlon_coords(1,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,5);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,5),
                                                new latlon_coords(1,5),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(0,7);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectThree) {
  //Test setting a single local primary merge and redirect
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,5),
                                                new latlon_coords(0,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(4,8);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,8),
                                                new latlon_coords(4,8),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(4,1);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,5),
                                                new latlon_coords(0,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(4,8);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,8),
                                                new latlon_coords(4,8),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(0,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,5),
                                                new latlon_coords(0,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,1);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,1),
                                                new latlon_coords(1,1),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(3,1);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,1),
                                                new latlon_coords(1,0),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(8,6),
                                                new latlon_coords(2,1),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,1);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,1),
                                                new latlon_coords(1,0),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(8,6),
                                                new latlon_coords(2,2),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(0,3);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,8),
                                                new latlon_coords(1,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,5);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,5),
                                                new latlon_coords(0,1),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,7);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(8,4),
                                                new latlon_coords(2,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(8,6),
                                                new latlon_coords(2,1),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,2);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,1),
                                                new latlon_coords(1,1),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,5);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,5),
                                                new latlon_coords(1,5),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(0,0);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,5),
                                                new latlon_coords(0,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(4,8);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,8),
                                                new latlon_coords(4,8),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(4,4);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,5),
                                                new latlon_coords(0,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(4,8);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,8),
                                                new latlon_coords(4,8),
                                                true);
  secondary_merge = nullptr;
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(0,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,5),
                                                new latlon_coords(0,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,1);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,1),
                                                new latlon_coords(1,0),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(5,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,1),
                                                new latlon_coords(1,0),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(8,6),
                                                new latlon_coords(2,1),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(7,0);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,1),
                                                new latlon_coords(1,0),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(8,6),
                                                new latlon_coords(2,2),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,3);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,5.0,
                                                   5.0,5.0,5.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,8),
                                                new latlon_coords(1,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,5);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,5),
                                                new latlon_coords(0,2),
                                                false);
  secondary_merge = nullptr;
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,8);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectSeventeen) {
  //Tests using the local connect fall back when non local connect isn't possible
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  int* basin_catchment_numbers_in = new int[9*9] {1,1,1, 1,1,1, 1,1,1,
                                                  1,3,1, 1,1,1, 1,2,1,
                                                  1,3,1, 1,1,1, 1,2,1,
//
                                                  1,3,1, 1,1,1, 1,2,1,
                                                  1,1,1, 1,1,1, 1,1,1,
                                                  1,1,1, 1,1,1, 1,1,1,
//
                                                  1,1,1, 1,1,1, 1,1,1,
                                                  1,1,1, 1,1,1, 1,1,1,
                                                  1,1,1, 1,1,1, 1,1,1 };
  int* coarse_catchment_nums_in = new int[3*3] {2,3,3,
                                                2,1,3,
                                                1,1,1};
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,4.0,
                                                   8.0,5.0,8.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,4),
                                                new latlon_coords(2,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,7);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,7),
                                                new latlon_coords(1,7),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(5,7);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  coords* new_center_coords_in = new latlon_coords(3,7);
  coords* center_coords_in = new latlon_coords(4,7);
  coords* previous_filled_cell_coords_in = new latlon_coords(5,7);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(4,4));
  basin_catchment_centers_in.push_back(new latlon_coords(1,7));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest, TestProcessingSetPrimaryMergeAndRedirectEighteen) {
  //Tests using the local connect fall back when non local connect isn't possible -
  // this version is a control where it is possible
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  int* basin_catchment_numbers_in = new int[9*9] {1,1,1, 1,1,1, 1,1,1,
                                                  1,3,1, 1,1,1, 1,2,1,
                                                  1,3,1, 1,1,1, 1,2,1,
//
                                                  1,3,1, 1,1,1, 1,2,1,
                                                  1,1,1, 1,1,1, 1,1,1,
                                                  1,1,1, 1,1,1, 1,1,1,
//
                                                  1,1,1, 1,1,1, 1,1,1,
                                                  1,1,1, 1,1,1, 1,1,1,
                                                  1,1,1, 1,1,1, 1,1,1 };
  int* coarse_catchment_nums_in = new int[3*3] {2,3,3,
                                                2,1,3,
                                                1,1,1};
  double* prior_coarse_rdirs_in = new double[3*3] {5.0,5.0,4.0,
                                                   8.0,5.0,8.0,
                                                   5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,4),
                                                new latlon_coords(2,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,1);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,1),
                                                new latlon_coords(1,0),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(5,1);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  coords* new_center_coords_in = new latlon_coords(3,1);
  coords* center_coords_in = new latlon_coords(4,1);
  coords* previous_filled_cell_coords_in = new latlon_coords(5,1);
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(4,4));
  basin_catchment_centers_in.push_back(new latlon_coords(1,7));
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_primary_merge_and_redirect(basin_catchment_centers_in,
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
  delete new_center_coords_in; delete center_coords_in; delete previous_filled_cell_coords_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest,TestSettingSecondaryRedirect) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  double* raw_orography_in = new double[9*9];
  std::fill_n(raw_orography_in,9*9,10.0);
  double* corrected_orography_in = new double[9*9];
  std::fill_n(corrected_orography_in,9*9, 9.0);
  double* prior_coarse_rdirs_in = new double[3*3]
                                        { 5.0,5.0,5.0,
                                          5.0,5.0,5.0,
                                          5.0,5.0,5.0 };
  bool* requires_flood_redirect_indices_in = new bool[9*9]
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
  bool* requires_connect_redirect_indices_in = new bool[9*9]
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
  int* flood_next_cell_lat_index_in = new int[9*9]
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
  int* flood_next_cell_lon_index_in = new int[9*9]
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
  int* connect_next_cell_lat_index_in = new int[9*9]
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
  int* connect_next_cell_lon_index_in = new int[9*9]
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
                                               -1,-1,-1,  -1,-1,-1, -1,-1,1};
  int*  coarse_catchment_nums_in = new int[3*3]
                                             { 1, 1, 1,
                                               1, 1, 1,
                                               1, 1, 1 };
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(2,7),
                                                new latlon_coords(2,7),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(0,4); 
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);

  bool* requires_flood_redirect_indices_expected_out = new bool[9*9];
  std::fill_n(requires_flood_redirect_indices_expected_out,9*9,false);
  bool* requires_connect_redirect_indices_expected_out = new bool[9*9];
  std::fill_n(requires_connect_redirect_indices_expected_out,9*9,false);
  coords* new_center_coords_in = new latlon_coords(2,7);
  coords* center_coords_in = new latlon_coords(5,1);
  coords* previous_filled_cell_coords_in = new latlon_coords(0,4);
  coords* target_basin_center_coords = new latlon_coords(2,7);
  height_types previous_filled_cell_height_type_in = flood_height;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_secondary_redirect(prior_coarse_rdirs_in,
                                         flood_next_cell_lat_index_in,
                                         flood_next_cell_lon_index_in,
                                         connect_next_cell_lat_index_in,
                                         connect_next_cell_lon_index_in,
                                         coarse_catchment_nums_in,
                                         requires_flood_redirect_indices_in,
                                         requires_connect_redirect_indices_in,
                                         raw_orography_in,
                                         corrected_orography_in,new_center_coords_in,
                                         center_coords_in,previous_filled_cell_coords_in,
                                         target_basin_center_coords,
                                         previous_filled_cell_height_type_in,
                                         grid_params_in,coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  EXPECT_TRUE(field<bool>(requires_flood_redirect_indices_in,grid_params_in)
               == field<bool>(requires_flood_redirect_indices_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(requires_connect_redirect_indices_in,grid_params_in)
               == field<bool>(requires_connect_redirect_indices_expected_out,grid_params_in));
  delete grid_params_in;
  delete coarse_grid_params_in;
  delete[] prior_coarse_rdirs_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] requires_flood_redirect_indices_in;
  delete[] requires_connect_redirect_indices_in;
  delete[] raw_orography_in; delete[] corrected_orography_in;
  delete[] requires_flood_redirect_indices_expected_out;
  delete[] requires_connect_redirect_indices_expected_out;
  delete[] coarse_catchment_nums_in;
  delete new_center_coords_in; delete center_coords_in;
  delete previous_filled_cell_coords_in;
  delete target_basin_center_coords; 
}

TEST_F(BasinEvaluationTest,TestSettingSecondaryRedirectTwo) {
  auto grid_params_in = new latlon_grid_params(9,9,false);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  double* raw_orography_in = new double[9*9];
  std::fill_n(raw_orography_in,9*9,10.0);
  double* corrected_orography_in = new double[9*9];
  std::fill_n(corrected_orography_in,9*9, 9.0);
  double* prior_coarse_rdirs_in = new double[3*3]
                                        { 5.0,5.0,5.0,
                                          5.0,5.0,5.0,
                                          5.0,5.0,5.0 };
  int* flood_next_cell_lat_index_in = new int[9*9]
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
  int* flood_next_cell_lon_index_in = new int[9*9]
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
  int* connect_next_cell_lat_index_in = new int[9*9]
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
  int* connect_next_cell_lon_index_in = new int[9*9]
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
  int*  coarse_catchment_nums_in = new int[3*3]
                                             { 1, 1, 1,
                                               1, 1, 1,
                                               1, 1, 1 };
  bool* requires_flood_redirect_indices_expected_out = new bool[9*9];
  std::fill_n(requires_flood_redirect_indices_expected_out,9*9,false);
  bool* requires_connect_redirect_indices_expected_out = new bool[9*9];
  std::fill_n(requires_connect_redirect_indices_expected_out,9*9,false);
  bool* requires_flood_redirect_indices_in = new bool[9*9]
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
  bool* requires_connect_redirect_indices_in = new bool[9*9]
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
                                        int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(5,8),
                                                new latlon_coords(5,8),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(0,4); 
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);

  coords* new_center_coords_in = new latlon_coords(5,8);
  coords* center_coords_in = new latlon_coords(5,1);
  coords* previous_filled_cell_coords_in = new latlon_coords(0,4);
  coords* target_basin_center_coords = new latlon_coords(5,8);
  height_types previous_filled_cell_height_type_in = flood_height;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_secondary_redirect(prior_coarse_rdirs_in,
                                         flood_next_cell_lat_index_in,
                                         flood_next_cell_lon_index_in,
                                         connect_next_cell_lat_index_in,
                                         connect_next_cell_lon_index_in,
                                         coarse_catchment_nums_in,
                                         requires_flood_redirect_indices_in,
                                         requires_connect_redirect_indices_in,
                                         raw_orography_in,
                                         corrected_orography_in,new_center_coords_in,
                                         center_coords_in,previous_filled_cell_coords_in,
                                         target_basin_center_coords,
                                         previous_filled_cell_height_type_in,
                                         grid_params_in,coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  EXPECT_TRUE(field<bool>(requires_flood_redirect_indices_in,grid_params_in)
              == field<bool>(requires_flood_redirect_indices_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(requires_connect_redirect_indices_in,grid_params_in)
              == field<bool>(requires_connect_redirect_indices_expected_out,grid_params_in));
  delete grid_params_in;
  delete coarse_grid_params_in;
  delete[] prior_coarse_rdirs_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] requires_flood_redirect_indices_in;
  delete[] requires_connect_redirect_indices_in;
  delete[] raw_orography_in; delete[] corrected_orography_in;
  delete[] requires_flood_redirect_indices_expected_out;
  delete[] requires_connect_redirect_indices_expected_out;
  delete[] coarse_catchment_nums_in;
  delete new_center_coords_in; delete center_coords_in;
  delete previous_filled_cell_coords_in;
  delete target_basin_center_coords;
}

TEST_F(BasinEvaluationTest,TestSettingSecondaryRedirectEqualHeights) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  double* raw_orography_in = new double[9*9];
  std::fill_n(raw_orography_in,9*9,10.0);
  double* corrected_orography_in = new double[9*9];
  std::fill_n(corrected_orography_in,9*9,10.0);
  double* prior_coarse_rdirs_in = new double[3*3]
                                        { 5.0,5.0,5.0,
                                          5.0,5.0,5.0,
                                          5.0,5.0,5.0 };
  int* flood_next_cell_lat_index_in = new int[9*9]
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
  int* flood_next_cell_lon_index_in = new int[9*9]
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
  int* connect_next_cell_lat_index_in = new int[9*9]
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
  int* connect_next_cell_lon_index_in = new int[9*9]
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
  int*  coarse_catchment_nums_in = new int[3*3]
                                             { 1, 1, 1,
                                               1, 1, 1,
                                               1, 1, 1 };
  bool* requires_flood_redirect_indices_in = new bool[9*9]
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
  bool* requires_connect_redirect_indices_in = new bool[9*9]
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
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(7,5),
                                                new latlon_coords(7,5),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(0,4); 
  (*flood_merge_and_redirect_indices_index)(working_coords) = 0;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  bool* requires_flood_redirect_indices_expected_out = new bool[9*9];
  std::fill_n(requires_flood_redirect_indices_expected_out,9*9,false);
  bool* requires_connect_redirect_indices_expected_out = new bool[9*9];
  std::fill_n(requires_connect_redirect_indices_expected_out,9*9,false);
  coords* new_center_coords_in = new latlon_coords(7,5);
  coords* center_coords_in = new latlon_coords(5,1);
  coords* previous_filled_cell_coords_in = new latlon_coords(0,4);
  coords* target_basin_center_coords = new latlon_coords(7,5);
  height_types previous_filled_cell_height_type_in = flood_height;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_secondary_redirect(prior_coarse_rdirs_in,
                                         flood_next_cell_lat_index_in,
                                         flood_next_cell_lon_index_in,
                                         connect_next_cell_lat_index_in,
                                         connect_next_cell_lon_index_in,
                                         coarse_catchment_nums_in,
                                         requires_flood_redirect_indices_in,
                                         requires_connect_redirect_indices_in,
                                         raw_orography_in,
                                         corrected_orography_in,new_center_coords_in,
                                         center_coords_in,previous_filled_cell_coords_in,
                                         target_basin_center_coords,
                                         previous_filled_cell_height_type_in,
                                         grid_params_in,coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  EXPECT_TRUE(field<bool>(requires_flood_redirect_indices_in,grid_params_in)
              == field<bool>(requires_flood_redirect_indices_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(requires_connect_redirect_indices_in,grid_params_in)
              == field<bool>(requires_connect_redirect_indices_expected_out,grid_params_in));
  delete grid_params_in;
  delete coarse_grid_params_in;
  delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] requires_flood_redirect_indices_in;
  delete[] requires_connect_redirect_indices_in;
  delete[] raw_orography_in; delete[] corrected_orography_in;
  delete[] requires_flood_redirect_indices_expected_out;
  delete[] requires_connect_redirect_indices_expected_out;
  delete new_center_coords_in; delete center_coords_in;
  delete previous_filled_cell_coords_in;
  delete target_basin_center_coords;
}

TEST_F(BasinEvaluationTest,TestSettingSecondaryRedirectCorrectedHigher) {
  auto grid_params_in = new latlon_grid_params(9,9,true);
  auto coarse_grid_params_in = new latlon_grid_params(3,3,true);
  double* raw_orography_in = new double[9*9];
  std::fill_n(raw_orography_in,9*9,10.0);
  double* corrected_orography_in = new double[9*9];
  std::fill_n(corrected_orography_in,9*9,11.0);
  double* prior_coarse_rdirs_in = new double[3*3]
                                        { 5.0,5.0,5.0,
                                          5.0,5.0,5.0,
                                          5.0,5.0,5.0 };
  int* flood_next_cell_lat_index_in = new int[9*9]
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
  int* flood_next_cell_lon_index_in = new int[9*9]
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
  int* connect_next_cell_lat_index_in = new int[9*9]
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
  int* connect_next_cell_lon_index_in = new int[9*9]
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
  int*  coarse_catchment_nums_in = new int[3*3]
                                             { 1, 1, 1,
                                               1, 1, 1,
                                               1, 1, 1 };
  bool* requires_flood_redirect_indices_in = new bool[9*9]
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
  bool* requires_connect_redirect_indices_in = new bool[9*9]
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
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(3,2),
                                                new latlon_coords(3,2),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(0,4); 
  (*flood_merge_and_redirect_indices_index)(working_coords) = 0;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  bool* requires_flood_redirect_indices_expected_out = new bool[9*9];
  std::fill_n(requires_flood_redirect_indices_expected_out,9*9,false);
  bool* requires_connect_redirect_indices_expected_out = new bool[9*9];
  std::fill_n(requires_connect_redirect_indices_expected_out,9*9,false);
  coords* new_center_coords_in = new latlon_coords(3,2);
  coords* center_coords_in = new latlon_coords(5,1);
  coords* previous_filled_cell_coords_in = new latlon_coords(0,4);
  coords* target_basin_center_coords = new latlon_coords(3,2);
  height_types previous_filled_cell_height_type_in = flood_height;
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_secondary_redirect(prior_coarse_rdirs_in,
                                         flood_next_cell_lat_index_in,
                                         flood_next_cell_lon_index_in,
                                         connect_next_cell_lat_index_in,
                                         connect_next_cell_lon_index_in,
                                         coarse_catchment_nums_in,
                                         requires_flood_redirect_indices_in,
                                         requires_connect_redirect_indices_in,
                                         raw_orography_in,
                                         corrected_orography_in,new_center_coords_in,
                                         center_coords_in,previous_filled_cell_coords_in,
                                         target_basin_center_coords,
                                         previous_filled_cell_height_type_in,
                                         grid_params_in,coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  EXPECT_TRUE(field<bool>(requires_flood_redirect_indices_in,grid_params_in)
              == field<bool>(requires_flood_redirect_indices_expected_out,grid_params_in));
  EXPECT_TRUE(field<bool>(requires_connect_redirect_indices_in,grid_params_in)
              == field<bool>(requires_connect_redirect_indices_expected_out,grid_params_in));
  delete grid_params_in;
  delete coarse_grid_params_in;
  delete[] prior_coarse_rdirs_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] requires_flood_redirect_indices_in;
  delete[] requires_connect_redirect_indices_in;
  delete[] raw_orography_in; delete[] corrected_orography_in;
  delete[] requires_flood_redirect_indices_expected_out;
  delete[] requires_connect_redirect_indices_expected_out;
  delete[] coarse_catchment_nums_in;
  delete new_center_coords_in; delete center_coords_in;
  delete previous_filled_cell_coords_in;
  delete target_basin_center_coords; 
}

TEST_F(BasinEvaluationTest,TestSettingRemainingRedirects) {
  auto grid_params_in = new latlon_grid_params(12,12,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
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
                                     -1.0, 0.0,-1.0,  1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0, 2.0,
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
  int* flood_next_cell_lon_index_in = new int[12*12]
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
  delete[] flood_redirect_lat_index_expected_out;
  int* flood_redirect_lon_index_expected_out = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1,  3,-1,-1,
-                                    -1,-1,-1, -1, 0,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1, 1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1,  0,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1, 3,-1
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  delete[] flood_redirect_lon_index_expected_out;
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
  delete[] connect_redirect_lat_index_expected_out;
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
  delete[] connect_redirect_lon_index_expected_out;
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
  delete[] flood_local_redirect_expected_out;
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
  delete[] connect_local_redirect_expected_out;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_redirects =
      new vector<merge_and_redirect_indices*>();
  merge_and_redirect_indices* secondary_redirect;
  collected_merge_and_redirect_indices* collected_indices;
  coords* working_coords = nullptr;
  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(2,11),
                                                new latlon_coords(1,3),
                                                false);
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,9); 
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  delete working_coords;
  flood_index++;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(2,3),
                                                new latlon_coords(1,0),
                                                false);
  primary_redirects = new vector<merge_and_redirect_indices*>();
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(2,4); 
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  delete working_coords;
  flood_index++;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(2,1),
                                                new latlon_coords(0,0),
                                                false);
  primary_redirects = new vector<merge_and_redirect_indices*>();
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(3,1); 
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  delete working_coords;
  flood_index++;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(5,2),
                                                new latlon_coords(1,0),
                                                false);
  primary_redirects = new vector<merge_and_redirect_indices*>();
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,6); 
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  delete working_coords;
  flood_index++;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(9,11),
                                                new latlon_coords(3,3),
                                                false);
  primary_redirects = new vector<merge_and_redirect_indices*>();
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(9,10); 
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  delete working_coords;
  flood_index++;

  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  int* coarse_catchment_nums_in = new int[4*4]
                                     {11,11,14,14,
                                      12,11,14,13,
                                      12,12,12,13,
                                      12,12,13,13};
  double* prior_coarse_rdirs_in = new double[4*4] {5.0,5.0,5.0,5.0,
                                                   5.0,5.0,5.0,5.0,
                                                   5.0,5.0,5.0,5.0,
                                                   5.0,5.0,5.0,5.0 };
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
                                          prior_coarse_rdirs_in,
                                          requires_flood_redirect_indices_in,
                                          requires_connect_redirect_indices_in,
                                          basin_catchment_numbers_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] coarse_catchment_nums_in;
  delete[] flood_next_cell_lat_index_in;
  delete[] connect_next_cell_lat_index_in;
  delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lon_index_in;
  delete[] prior_fine_catchments_in;
  delete[] prior_coarse_rdirs_in;
  delete[] prior_fine_rdirs_in;
  delete[] requires_flood_redirect_indices_in; delete[] requires_connect_redirect_indices_in;
  delete[] basin_catchment_numbers_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest,TestSettingRemainingRedirectsTwo) {
  auto grid_params_in = new latlon_grid_params(12,12,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* flood_next_cell_lat_index_in = new int[12*12]
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
  int* flood_next_cell_lon_index_in = new int[12*12]
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
  int* connect_next_cell_lat_index_in = new int[12*12]
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
  int* connect_next_cell_lon_index_in = new int[12*12]
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
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0, 0.0,
//
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0, 0.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
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
  delete[] flood_redirect_lat_index_expected_out;
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
  delete[] flood_redirect_lon_index_expected_out;
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
  delete[] connect_redirect_lat_index_expected_out;
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
  delete[] connect_redirect_lon_index_expected_out;
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
  delete[] flood_local_redirect_expected_out;
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
  delete[] connect_local_redirect_expected_out;
  int flood_index = 0;
  int connect_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_redirects =
      new vector<merge_and_redirect_indices*>();
  merge_and_redirect_indices* secondary_redirect;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(2,11),
                                                new latlon_coords(0,3),
                                                false);
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,9);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  delete working_coords;
  flood_index++;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,2),
                                                new latlon_coords(1,0),
                                                false);
  primary_redirects = new vector<merge_and_redirect_indices*>();
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(2,4); 
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  delete working_coords;
  flood_index++;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(5,2),
                                                new latlon_coords(1,0),
                                                false);
  primary_redirects = new vector<merge_and_redirect_indices*>();
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,6); 
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  delete working_coords;
  flood_index++;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(8,8),
                                                new latlon_coords(2,2),
                                                false);
  primary_redirects = new vector<merge_and_redirect_indices*>();
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(9,10); 
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  delete working_coords;
  flood_index++;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(2,11),
                                                new latlon_coords(0,3),
                                                false);
  primary_redirects = new vector<merge_and_redirect_indices*>();
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  connect_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,9); 
  (*connect_merge_and_redirect_indices_index)(working_coords) = connect_index;
  delete working_coords;
  connect_index++;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,2),
                                                new latlon_coords(1,0),
                                                false);
  primary_redirects = new vector<merge_and_redirect_indices*>();
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  connect_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(2,4); 
  (*connect_merge_and_redirect_indices_index)(working_coords) = connect_index;
  delete working_coords;
  connect_index++;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(5,2),
                                                new latlon_coords(1,0),
                                                false);
  primary_redirects = new vector<merge_and_redirect_indices*>();
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  connect_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,6); 
  (*connect_merge_and_redirect_indices_index)(working_coords) = connect_index;
  delete working_coords;
  connect_index++;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(8,8),
                                                new latlon_coords(2,2),
                                                false);
  primary_redirects = new vector<merge_and_redirect_indices*>();
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  connect_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(9,10); 
  (*connect_merge_and_redirect_indices_index)(working_coords) = connect_index;
  delete working_coords;
  connect_index++;

  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  int* coarse_catchment_nums_in = new int[4*4]
                                     {11,11,14,14,
                                      12,11,14,13,
                                      12,12,12,13,
                                      12,12,13,13};
  double* prior_coarse_rdirs_in = new double[4*4] {5.0,5.0,5.0,5.0,
                                                   5.0,5.0,5.0,5.0,
                                                   5.0,5.0,5.0,5.0,
                                                   5.0,5.0,5.0,5.0 };
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(3,0));
  basin_catchment_centers_in.push_back(new latlon_coords(10,7));
  basin_catchment_centers_in.push_back(new latlon_coords(8,10));
  basin_catchment_centers_in.push_back(new latlon_coords(0,11));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_remaining_redirects(basin_catchment_centers_in,
                                          prior_fine_rdirs_in,
                                          prior_coarse_rdirs_in,
                                          requires_flood_redirect_indices_in,
                                          requires_connect_redirect_indices_in,
                                          basin_catchment_numbers_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] coarse_catchment_nums_in; delete[] prior_fine_catchments_in;
  delete[] basin_catchment_numbers_in; delete[] prior_fine_rdirs_in;
  delete[] requires_flood_redirect_indices_in; delete[] requires_connect_redirect_indices_in;
  delete[] prior_coarse_rdirs_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest,TestSettingRemainingRedirectsThree) {
  //Test defaulting to a local redirect
  auto grid_params_in = new latlon_grid_params(12,12,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* flood_next_cell_lat_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1, 4,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1, 4,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* prior_fine_catchments_in = new int[12*12]
                                    {1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 2,2,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,2,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,2,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,2,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1 };
  int* basin_catchment_numbers_in = new int[12*12]
                                    {1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1 };
  double*  prior_fine_rdirs_in = new double[12*12]
                                    {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0, 5.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0, 8.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                     -1.0,-1.0,-1.0, -1.0, 8.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0, 8.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  bool* requires_flood_redirect_indices_in = new bool[12*12]
                                    {false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false, true,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
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
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_redirects =
      new vector<merge_and_redirect_indices*>();
  merge_and_redirect_indices* secondary_redirect;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,4),
                                                new latlon_coords(1,1),
                                                false);
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(7,4);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  int* coarse_catchment_nums_in = new int[4*4]
                                     {2,2,1,1,
                                      1,2,1,1,
                                      1,1,1,1,
                                      1,1,1,1};
  double* prior_coarse_rdirs_in = new double[4*4] {5.0,6.0,5.0,5.0,
                                                   5.0,8.0,5.0,5.0,
                                                   5.0,5.0,5.0,5.0,
                                                   5.0,5.0,5.0,5.0 };
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,3));
  basin_catchment_centers_in.push_back(new latlon_coords(1,4));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_remaining_redirects(basin_catchment_centers_in,
                                          prior_fine_rdirs_in,
                                          prior_coarse_rdirs_in,
                                          requires_flood_redirect_indices_in,
                                          requires_connect_redirect_indices_in,
                                          basin_catchment_numbers_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] coarse_catchment_nums_in; delete[] prior_fine_catchments_in;
  delete[] basin_catchment_numbers_in; delete[] prior_fine_rdirs_in;
  delete[] requires_flood_redirect_indices_in; delete[] requires_connect_redirect_indices_in;
  delete[] prior_coarse_rdirs_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}


TEST_F(BasinEvaluationTest,TestSettingRemainingRedirectsFour) {
  //Test defaulting to a local redirect - control case
  auto grid_params_in = new latlon_grid_params(12,12,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* flood_next_cell_lat_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1, 4,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1, 4,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* prior_fine_catchments_in = new int[12*12]
                                    {1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 2,2,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,2,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,2,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,2,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1 };
  int* basin_catchment_numbers_in = new int[12*12]
                                    {1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1 };
  double*  prior_fine_rdirs_in = new double[12*12]
                                    {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0, 5.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0, 8.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                     -1.0,-1.0,-1.0, -1.0, 8.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0, 8.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  bool* requires_flood_redirect_indices_in = new bool[12*12]
                                    {false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false, true,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
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
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_redirects =
      new vector<merge_and_redirect_indices*>();
  merge_and_redirect_indices* secondary_redirect;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,4),
                                                new latlon_coords(1,1),
                                                false);
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(7,4);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  int* coarse_catchment_nums_in = new int[4*4]
                                     {1,2,1,1,
                                      1,2,1,1,
                                      1,1,1,1,
                                      1,1,1,1};
  double* prior_coarse_rdirs_in = new double[4*4] {5.0,5.0,5.0,5.0,
                                                   5.0,8.0,5.0,5.0,
                                                   5.0,5.0,5.0,5.0,
                                                   5.0,5.0,5.0,5.0 };
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,3));
  basin_catchment_centers_in.push_back(new latlon_coords(1,4));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_remaining_redirects(basin_catchment_centers_in,
                                          prior_fine_rdirs_in,
                                          prior_coarse_rdirs_in,
                                          requires_flood_redirect_indices_in,
                                          requires_connect_redirect_indices_in,
                                          basin_catchment_numbers_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] coarse_catchment_nums_in; delete[] prior_fine_catchments_in;
  delete[] basin_catchment_numbers_in; delete[] prior_fine_rdirs_in;
  delete[] requires_flood_redirect_indices_in; delete[] requires_connect_redirect_indices_in;
  delete[] prior_coarse_rdirs_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest,TestSettingRemainingRedirectsFive) {
  //Test defaulting to a local redirect
  auto grid_params_in = new latlon_grid_params(12,12,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* flood_next_cell_lat_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1, 4,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1, 4,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* prior_fine_catchments_in = new int[12*12]
                                    {1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 2,2,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,2,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,2,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,2,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1 };
  int* basin_catchment_numbers_in = new int[12*12]
                                    {1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,0,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,0,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,0,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,0,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1 };
  double*  prior_fine_rdirs_in = new double[12*12]
                                    {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0, 0.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0, 8.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                     -1.0,-1.0,-1.0, -1.0, 8.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0, 8.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  bool* requires_flood_redirect_indices_in = new bool[12*12]
                                    {false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false, true,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
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
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_redirects =
      new vector<merge_and_redirect_indices*>();
  merge_and_redirect_indices* secondary_redirect;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,4),
                                                new latlon_coords(1,1),
                                                false);
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(7,4);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  int* coarse_catchment_nums_in = new int[4*4]
                                     {1,2,1,1,
                                      1,2,1,1,
                                      1,1,1,1,
                                      1,1,1,1};
  double* prior_coarse_rdirs_in = new double[4*4] {5.0,6.0,0.0,5.0,
                                                   5.0,8.0,5.0,5.0,
                                                   5.0,5.0,5.0,5.0,
                                                   5.0,5.0,5.0,5.0 };
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,3));
  basin_catchment_centers_in.push_back(new latlon_coords(1,4));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_remaining_redirects(basin_catchment_centers_in,
                                          prior_fine_rdirs_in,
                                          prior_coarse_rdirs_in,
                                          requires_flood_redirect_indices_in,
                                          requires_connect_redirect_indices_in,
                                          basin_catchment_numbers_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] coarse_catchment_nums_in; delete[] prior_fine_catchments_in;
  delete[] basin_catchment_numbers_in; delete[] prior_fine_rdirs_in;
  delete[] requires_flood_redirect_indices_in; delete[] requires_connect_redirect_indices_in;
  delete[] prior_coarse_rdirs_in;
  while (! basin_catchment_centers_in.empty()){
    coords* center_coords = basin_catchment_centers_in.back();
    basin_catchment_centers_in.pop_back();
    delete center_coords;
  }
}

TEST_F(BasinEvaluationTest,TestSettingRemainingRedirectsSix) {
  //Test defaulting to a local redirect - control case
  auto grid_params_in = new latlon_grid_params(12,12,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* flood_next_cell_lat_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1, 4,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1};
  int* flood_next_cell_lon_index_in = new int[12*12]
                                    {-1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1, 4,-1, -1,-1,-1, -1,-1,-1,
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
//
                                     -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1,-1,
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
  int* prior_fine_catchments_in = new int[12*12]
                                    {1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 2,2,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,2,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,2,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,2,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1 };
  int* basin_catchment_numbers_in = new int[12*12]
                                    {1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,3,1, 1,1,1, 1,1,1,
//
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1,
                                     1,1,1, 1,1,1, 1,1,1, 1,1,1 };
  double*  prior_fine_rdirs_in = new double[12*12]
                                    {-1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0, 5.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0, 8.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                     -1.0,-1.0,-1.0, -1.0, 8.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0, 8.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
//
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0,
                                     -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0, -1.0,-1.0,-1.0};
  bool* requires_flood_redirect_indices_in = new bool[12*12]
                                    {false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
                                     false,false,false, false, true,false, false,false,false, false,false,false,
                                     false,false,false, false,false,false, false,false,false, false,false,false,
//
                                     false,false,false, false,false,false, false,false,false, false,false,false,
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
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_redirects =
    new vector<merge_and_redirect_indices*>();
  merge_and_redirect_indices* secondary_redirect;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;

  secondary_redirect =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,4),
                                                new latlon_coords(1,1),
                                                false);
  collected_indices = new collected_merge_and_redirect_indices(primary_redirects,
                                                               secondary_redirect,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(7,4);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  int* coarse_catchment_nums_in = new int[4*4]
                                     {1,2,1,1,
                                      1,2,1,1,
                                      1,1,1,1,
                                      1,1,1,1};
  double* prior_coarse_rdirs_in = new double[4*4] {5.0,5.0,5.0,5.0,
                                                   5.0,8.0,5.0,5.0,
                                                   5.0,5.0,5.0,5.0,
                                                   5.0,5.0,5.0,5.0 };
  vector<coords*> basin_catchment_centers_in;
  basin_catchment_centers_in.push_back(new latlon_coords(1,1));
  basin_catchment_centers_in.push_back(new latlon_coords(1,3));
  basin_catchment_centers_in.push_back(new latlon_coords(1,4));
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.test_set_remaining_redirects(basin_catchment_centers_in,
                                          prior_fine_rdirs_in,
                                          prior_coarse_rdirs_in,
                                          requires_flood_redirect_indices_in,
                                          requires_connect_redirect_indices_in,
                                          basin_catchment_numbers_in,
                                          prior_fine_catchments_in,
                                          coarse_catchment_nums_in,
                                          flood_next_cell_lat_index_in,
                                          flood_next_cell_lon_index_in,
                                          connect_next_cell_lat_index_in,
                                          connect_next_cell_lon_index_in,
                                          grid_params_in,
                                          coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] coarse_catchment_nums_in; delete[] prior_fine_catchments_in;
  delete[] basin_catchment_numbers_in; delete[] prior_fine_rdirs_in;
  delete[] requires_flood_redirect_indices_in; delete[] requires_connect_redirect_indices_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[4*4]{5.0,5.0,5.0,5.0,
                                                  5.0,5.0,5.0,5.0,
                                                  5.0,5.0,5.0,5.0,
                                                  5.0,5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,8),
                                                new latlon_coords(1,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,2);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(6,2),
                                                new latlon_coords(0,0),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(7,8);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[4*4]{5.0,5.0,5.0,5.0,
                                                  5.0,5.0,5.0,5.0,
                                                  5.0,5.0,5.0,5.0,
                                                  5.0,5.0,5.0,5.0 };
  height_types previous_filled_cell_height_type_in = flood_height;
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,8),
                                                new latlon_coords(1,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,2);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(6,2),
                                                new latlon_coords(2,2),
                                                false);
  secondary_merge = nullptr;
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(7,8);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
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
                                                 prior_coarse_rdirs_in,
                                                 basin_catchment_numbers_in,
                                                 coarse_catchment_nums_in,
                                                 new_center_coords_in,
                                                 center_coords_in,
                                                 previous_filled_cell_coords_in,
                                                 previous_filled_cell_height_type_in,
                                                 grid_params_in,
                                                 coarse_grid_params_in);
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();

  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in;
  delete[] basin_catchment_numbers_in; delete[] coarse_catchment_nums_in;
  delete[] prior_coarse_rdirs_in;
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
  double* prior_coarse_rdirs_in = new double[4*4]{
    5.0,5.0,5.0,5.0,
    5.0,5.0,5.0,5.0,
    5.0,5.0,5.0,5.0,
    5.0,5.0,5.0,5.0 };
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
  double* connection_heights_in = new double[20*20];
  std::fill_n(connection_heights_in,20*20,0.0);
  double* flood_heights_in = new double[20*20];
  std::fill_n(flood_heights_in,20*20,0.0);
  int* flood_next_cell_lat_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lat_index_in,20*20,-1);
  int* flood_next_cell_lon_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lon_index_in,20*20,-1);
  int* connect_next_cell_lat_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lat_index_in,20*20,-1);
  int* connect_next_cell_lon_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lon_index_in,20*20,-1);
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
  2,  3,   5,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,   2,   2,   5,   5,  -1,  -1, -1,   1,
 -1,  4,   2,  -1,  -1,  8,  2,  -1,  -1,  -1, -1,  2,   8,   3,   9,   2,  -1,  -1, -1,  -1,
 -1,  3,   4,  -1,  -1,  6,  4,   6,   3,  -1, -1, -1,   7,   7,   4,   3,   6,  -1, -1,  -1,
 -1,  6,   5,   3,   6,  7, -1,   4,   4,  -1,  5,  7,   4,   6,   5,   8,   4,   3, -1,  -1,
  7,  6,   6,  -1,   5,  5,  6,   5,   5,  -1,  5,  5,   4,   5,   6,   6,   5,   5, -1,  -1,
 -1,  6,   7,  -1,  -1,  6, -1,   4,  -1,  -1, -1,  3,   6,   6,   8,   7,   3,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1,  9,   6,   7,   8,   6,   7,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,   8,   8,  -1,  -1,  -1,  -1, -1,  -1,
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
 19,  5,   2,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  15,  12,   9,   3,  -1,  -1, -1,  19,
 -1,  2,   2,  -1,  -1, 18,  1,  -1,  -1,  -1, -1, 13,  12,  12,  13,  14,  -1,  -1, -1,  -1,
 -1,  2,   1,  -1,  -1,  8,  5,  10,   6,  -1, -1, -1,  12,  11,  13,  14,  17,  -1, -1,  -1,
 -1,  2,   1,   5,   7,  5, -1,   6,   8,  -1, 15, 14,  14,  12,  12,  15,  15,  11, -1,  -1,
  2,  0,   1,  -1,   4,  5,  4,   7,   8,  -1, 10, 11,  12,  14,  13,  14,  16,  17, -1,  -1,
 -1,  4,   1,  -1,  -1,  6, -1,   7,  -1,  -1, -1, 13,  11,  15,  14,  13,  15,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, 12,  10,  15,  13,  16,  16,  -1, -1,  -1,
 -1, -1,  -1,  -1,  -1, -1, -1,  -1,  -1,  -1, -1, -1,  16,  11,  -1,  -1,  -1,  -1, -1,  -1,
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
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1,  4,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
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
    -1, -1, -1, -1, -1,  -1,      -1,  -1,  -1, 16,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,
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
  double* connection_heights_expected_out = new double[20*20]{
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 7, 4 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  };
  double* flood_heights_expected_out = new double[20*20]{
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
     1, 8, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 5, 6, 0, 0, 0, 2,
     0, 3, 3, 0, 0, 9, 8, 0, 0, 0, 0, 6, 3, 3, 5, 5, 0, 0, 0, 0,
     0, 3, 3, 0, 0, 3, 3, 5, 3, 0, 0, 0, 2, 3, 3, 4, 5, 0, 0, 0,
     0, 3, 3, 7, 2, 2, 0, 3, 3, 0, 3, 3, 3, 2, 3, 4, 4, 6, 0, 0,
     4, 4, 4, 0, 2, 1, 2, 2, 3, 0, 3, 2, 2, 3, 3, 3, 4, 5, 0, 0,
     0, 6, 4, 0, 0, 2, 0, 4, 0, 0, 0, 3, 2, 3, 3, 3, 5, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 3, 3, 3, 4, 5, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 4, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 0, 0,
     0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(17,3),
                                                new latlon_coords(3,0),
                                                false);
  primary_merges = new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(15,3);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(7,14),
                                                new latlon_coords(2,3),
                                                false);
  primary_merges = new vector<merge_and_redirect_indices*>;

  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(7,1);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(7,14),
                                                new latlon_coords(7,14),
                                                true);
  primary_merges = new vector<merge_and_redirect_indices*>;

  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(4,7);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(7,14),
                                                new latlon_coords(7,14),
                                                true);
  primary_merges = new vector<merge_and_redirect_indices*>;

  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(5,11);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(5,13),
                                                new latlon_coords(5,13),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(7,14);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;
   primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(6,5),
                                                new latlon_coords(6,5),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(2,14);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(2,0),
                                                new latlon_coords(1,0),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(2,15);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(8,18),
                                                new latlon_coords(1,3),
                                                false);
  primary_merges = new vector<merge_and_redirect_indices*>;

  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(3,5);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(16,15),
                                                new latlon_coords(3,3),
                                                false);
  primary_merges = new vector<merge_and_redirect_indices*>;

  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(13,18);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  delete working_coords;
  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.setup_fields(minima_in,
                          raw_orography_in,
                          corrected_orography_in,
                          cell_areas_in,
                          connection_volume_thresholds_in,
                          flood_volume_thresholds_in,
                          connection_heights_in,
                          flood_heights_in,
                          prior_fine_rdirs_in,
                          prior_coarse_rdirs_in,
                          prior_fine_catchments_in,
                          coarse_catchment_nums_in,
                          flood_next_cell_lat_index_in,
                          flood_next_cell_lon_index_in,
                          connect_next_cell_lat_index_in,
                          connect_next_cell_lon_index_in,
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
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
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
  EXPECT_TRUE(field<double>(connection_heights_expected_out,grid_params_in)
              == field<double>(connection_heights_in,grid_params_in));
  EXPECT_TRUE(field<double>(flood_heights_expected_out,grid_params_in)
              == field<double>(flood_heights_in,grid_params_in));
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in; delete alg4;
  delete[] coarse_catchment_nums_in; delete[] corrected_orography_in;
  delete[] raw_orography_in; delete[] minima_in;
  delete[] flood_heights_in; delete[] connection_heights_in;
  delete[] prior_coarse_rdirs_in;
  delete[] prior_fine_rdirs_in; delete[] prior_fine_catchments_in;
  delete[] connection_volume_thresholds_in; delete[] flood_volume_thresholds_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flood_volume_thresholds_expected_out;
  delete[] connection_volume_thresholds_expected_out;
  delete[] flood_next_cell_lat_index_expected_out;
  delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out;
  delete[] connect_next_cell_lon_index_expected_out;
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
  double* prior_coarse_rdirs_in = new double[4*4]{
    5.0,5.0,5.0,5.0,
    5.0,5.0,5.0,5.0,
    5.0,5.0,5.0,5.0,
    5.0,5.0,5.0,5.0 };
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
  double* connection_heights_in = new double[20*20];
  std::fill_n(connection_heights_in,20*20,0.0);
  double* flood_heights_in = new double[20*20];
  std::fill_n(flood_heights_in,20*20,0.0);
  int* flood_next_cell_lat_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lat_index_in,20*20,-1);
  int* flood_next_cell_lon_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lon_index_in,20*20,-1);
  int* connect_next_cell_lat_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lat_index_in,20*20,-1);
  int* connect_next_cell_lon_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lon_index_in,20*20,-1);
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
  double* connection_heights_expected_out = new double[20*20]{
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  double* flood_heights_expected_out = new double[20*20]{
     0,  0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  2, 11, 0,12, 0, 0, 0, 2, 18, 0, 0,  2, 0,  0, 0, 0, 0, 0, 0,
     0,  3, 12, 0, 9, 0, 0, 0, 3, 16, 0, 0,  3, 0,  0, 0, 0, 0, 0, 0,
     0,  4, 13, 0, 8, 0, 0, 0, 4, 15, 0, 0,  4, 0,  0, 0, 0, 0, 0, 0,
     0,  5, 14, 0, 6, 0, 0, 0, 5, 14, 6, 5,  5, 0,  0, 0, 0, 0, 0, 0,
     0,  6, 15, 0, 5, 0, 0, 0, 6, 13, 0, 0,  7, 0,  0, 0, 0, 0, 0, 0,
     0,  8, 16, 0, 4, 0, 0, 0, 7, 12, 0, 0,  8, 0,  0, 0, 0, 0, 0, 0,
     0,  9, 17, 0, 3, 0, 0, 0, 8, 11, 0, 0,  9, 0,  0, 0, 0, 6, 0, 0,
     0, 10, 18, 0, 2, 0, 0, 0, 9, 10, 0, 0, 12, 0,  0, 0, 5, 7, 0, 0,
     0,  0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0,  0, 4, 8, 0, 0, 0,
     0,  0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0,  3, 9, 0, 0, 0, 0,
     0,  2, 18, 0, 3, 0, 0, 0, 0,  0, 0, 0,  0, 2, 10, 0, 0, 0, 0, 0,
     0,  3, 16, 0, 4, 0, 0, 0, 0,  0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  4, 15, 0, 5, 0, 0, 2, 0,  6, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  5, 14, 0, 6, 0, 0, 3, 4,  4, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  6, 13, 0, 8, 0, 0, 4, 0,  3, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  7, 12, 0, 9, 0, 0, 5, 0,  2, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  8, 11, 0,10, 0, 0, 0, 0,  0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  9, 10, 0,12, 0, 0, 0, 0,  0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0 };
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(13,13),
                                                new latlon_coords(2,3),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(11,14);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,12),
                                                new latlon_coords(1,12),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,9);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,8),
                                                new latlon_coords(1,1),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,12);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(3,14),
                                                new latlon_coords(0,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,9);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(11,4),
                                                new latlon_coords(11,4),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(16,2);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(11,1),
                                                new latlon_coords(3,0),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(18,4);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(18,6),
                                                new latlon_coords(3,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(11,2);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(13,7),
                                                new latlon_coords(13,7),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(14,9);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(16,9),
                                                new latlon_coords(2,1),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(15,7);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(18,7),
                                                new latlon_coords(3,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(13,9);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,1),
                                                new latlon_coords(1,1),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,4);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(8,4),
                                                new latlon_coords(0,0),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(2,2);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(3,19),
                                                new latlon_coords(1,0),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,2);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.setup_fields(minima_in,
                          raw_orography_in,
                          corrected_orography_in,
                          cell_areas_in,
                          connection_volume_thresholds_in,
                          flood_volume_thresholds_in,
                          connection_heights_in,
                          flood_heights_in,
                          prior_fine_rdirs_in,
                          prior_coarse_rdirs_in,
                          prior_fine_catchments_in,
                          coarse_catchment_nums_in,
                          flood_next_cell_lat_index_in,
                          flood_next_cell_lon_index_in,
                          connect_next_cell_lat_index_in,
                          connect_next_cell_lon_index_in,
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
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
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
  EXPECT_TRUE(field<double>(connection_heights_expected_out,grid_params_in)
              == field<double>(connection_heights_in,grid_params_in));
  EXPECT_TRUE(field<double>(flood_heights_expected_out,grid_params_in)
              == field<double>(flood_heights_in,grid_params_in));
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in; delete alg4;
  delete[] coarse_catchment_nums_in; delete[] corrected_orography_in;
  delete[] raw_orography_in; delete[] minima_in;
  delete[] flood_heights_in; delete[] connection_heights_in;
  delete[] prior_coarse_rdirs_in;
  delete[] prior_fine_rdirs_in; delete[] prior_fine_catchments_in;
  delete[] connection_volume_thresholds_in; delete[] flood_volume_thresholds_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flood_volume_thresholds_expected_out;
  delete[] connection_volume_thresholds_expected_out;
  delete[] flood_next_cell_lat_index_expected_out;
  delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out;
  delete[] connect_next_cell_lon_index_expected_out;
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
  double* prior_coarse_rdirs_in = new double[4*4]{
    5.0,5.0,5.0,5.0,
    5.0,5.0,5.0,5.0,
    5.0,5.0,5.0,5.0,
    5.0,5.0,5.0,5.0 };
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
  double* connection_heights_in = new double[20*20];
  std::fill_n(connection_heights_in,20*20,0.0);
  double* flood_heights_in = new double[20*20];
  std::fill_n(flood_heights_in,20*20,0.0);
  int* flood_next_cell_lat_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lat_index_in,20*20,-1);
  int* flood_next_cell_lon_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lon_index_in,20*20,-1);
  int* connect_next_cell_lat_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lat_index_in,20*20,-1);
  int* connect_next_cell_lon_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lon_index_in,20*20,-1);
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
  double* connection_heights_expected_out = new double[20*20]{
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  double* flood_heights_expected_out = new double[20*20]{
     0,  0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  2, 11, 0,12, 0, 0, 0, 2, 18, 0, 0,  2, 0,  0, 0, 0, 0, 0, 0,
     0,  3, 12, 0, 9, 0, 0, 0, 3, 16, 0, 0,  3, 0,  0, 0, 0, 0, 0, 0,
     0,  4, 13, 0, 8, 0, 0, 0, 4, 15, 0, 0,  4, 0,  0, 0, 0, 0, 0, 0,
     0,  5, 14, 0, 6, 0, 0, 0, 5, 14, 6, 5,  5, 0,  0, 0, 0, 0, 0, 0,
     0,  6, 15, 0, 5, 0, 0, 0, 6, 13, 0, 0,  7, 0,  0, 0, 0, 0, 0, 0,
     0,  8, 16, 0, 4, 0, 0, 0, 7, 12, 0, 0,  8, 0,  0, 0, 0, 0, 0, 0,
     0,  9, 17, 0, 3, 0, 0, 0, 8, 11, 0, 0,  9, 0,  0, 0, 0, 6, 0, 0,
     0, 10, 18, 0, 2, 0, 0, 0, 9, 10, 0, 0, 12, 0,  0, 0, 5, 7, 0, 0,
     0,  0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0,  0, 4, 8, 0, 0, 0,
     0,  0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0,  3, 9, 0, 0, 0, 0,
     0,  2, 18, 0, 3, 0, 0, 0, 0,  0, 0, 0,  0, 2, 10, 0, 0, 0, 0, 0,
     0,  3, 16, 0, 4, 0, 0, 0, 0,  0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  4, 15, 0, 5, 0, 0, 2, 0,  6, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  5, 14, 0, 6, 0, 0, 3, 4,  4, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  6, 13, 0, 8, 0, 0, 4, 0,  3, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  7, 12, 0, 9, 0, 0, 5, 0,  2, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  8, 11, 0,10, 0, 0, 0, 0,  0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  9, 10, 0,12, 0, 0, 0, 0,  0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0,
     0,  0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0 };
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(13,13),
                                                new latlon_coords(2,3),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(11,14);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,12),
                                                new latlon_coords(1,12),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,9);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,8),
                                                new latlon_coords(1,1),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,12);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(3,14),
                                                new latlon_coords(0,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,9);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(11,4),
                                                new latlon_coords(11,4),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(16,2);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(11,1),
                                                new latlon_coords(3,0),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(18,4);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(18,6),
                                                new latlon_coords(3,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(11,2);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(13,7),
                                                new latlon_coords(13,7),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(14,9);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(16,9),
                                                new latlon_coords(2,1),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(15,7);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(18,7),
                                                new latlon_coords(3,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(13,9);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,1),
                                                new latlon_coords(1,1),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,4);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(8,4),
                                                new latlon_coords(0,0),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(2,2);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(3,19),
                                                new latlon_coords(1,0),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,2);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.setup_fields(minima_in,
                          raw_orography_in,
                          corrected_orography_in,
                          cell_areas_in,
                          connection_volume_thresholds_in,
                          flood_volume_thresholds_in,
                          connection_heights_in,
                          flood_heights_in,
                          prior_fine_rdirs_in,
                          prior_coarse_rdirs_in,
                          prior_fine_catchments_in,
                          coarse_catchment_nums_in,
                          flood_next_cell_lat_index_in,
                          flood_next_cell_lon_index_in,
                          connect_next_cell_lat_index_in,
                          connect_next_cell_lon_index_in,
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
  EXPECT_TRUE(field<double>(connection_heights_expected_out,grid_params_in)
              == field<double>(connection_heights_in,grid_params_in));
  EXPECT_TRUE(field<double>(flood_heights_expected_out,grid_params_in)
              == field<double>(flood_heights_in,grid_params_in));
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  delete grid_params_in; delete coarse_grid_params_in; delete alg4;
  delete[] coarse_catchment_nums_in; delete[] corrected_orography_in;
  delete[] raw_orography_in; delete[] minima_in;
  delete[] flood_heights_in; delete[] connection_heights_in;
  delete[] prior_coarse_rdirs_in;
  delete[] prior_fine_rdirs_in; delete[] prior_fine_catchments_in;
  delete[] connection_volume_thresholds_in; delete[] flood_volume_thresholds_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flood_volume_thresholds_expected_out;
  delete[] connection_volume_thresholds_expected_out;
  delete[] flood_next_cell_lat_index_expected_out;
  delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out;
  delete[] connect_next_cell_lon_index_expected_out;
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
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 2821.9, 2549.44, 2472.69, 2688.66, 2683.13, 2719.41, 2683.89, 2244.59, 2483.89, 2689.2, 2432.77, 2797.78, 2544.55, 2494.41, 2536.93, 2465.66, 2440.65, 2225.69, 0.0,
   0.0, 2748.9, 2447.34, 2755.49, 2874.7, 2346.54, 2536.99, 2721.65, 2468.33, 2546.6, 2963.46, 2564.8, 2937.53, 3036.52, 2731.27, 2529.42, 2821.21, 2742.64, 2499.66, 0.0,
   0.0, 2395.91, 2848.61, 2678.34, 3011.72, 2613.43, 2745.54, 2989.27, 2920.23, 2291.84, 2545.44, 2789.72, 2346.59, 2786.36, 2684.83, 2438.92, 2610.69, 2689.3, 2602.33, 0.0,
   0.0, 2483.7, 2735.31, 2976.6, 3233.42, 2668.57, 2292.6, 2740.1, 2857.81, 2743.6, 2444.06, 2573.87, 2855.08, 2613.29, 2701.57, 3352.55, 2564, 3025.03, 2427.56, 0.0,
//
   0.0, 2267.33, 2782.81, 2527.5, 2766.48, 2957.41, 3343.05, 3141.05, 2566.35, 2650.34, 2742.92, 2280.02, 2626.72, 2881.62, 3167.12, 3115.05, 2838.05, 2636.08, 2783.24, 0.0,
   0.0, 2820.65, 2911.99, 2642.95, 3150.44, 2533.18, 3067.33, 3084.39, 2840.64, 2760.65, 2403.02, 2529.29, 3511.31, 2271.61, 2227.3, 2508.7, 2858.88, 3293.69, 3142.58, 0.0,
   0.0, 3114.9, 3131.08, 2820.27, 3287.25, 3384.07, 3141.46, 3457.63, 2889.2, 2867.08, 2273.87, 3345.01, 3061.76, 3106.28, 2781.76, 3295.93, 3217.44, 2903.97, 2791.47, 0.0,
   0.0, 2906.96, 2959.25, 2909.48, 2775.41, 2819.85, 2863.38, 3402.57, 3294.53, 3408.99, 3257.53, 2952.45, 2855.42, 2938.21, 2984.79, 2621.18, 3244.9, 3160.94, 2213.43, 0.0,
   0.0, 2762.45, 2828.82, 2774.3, 2822.31, 3045.13, 2921.17, 2639.54, 3239.07, 3116.82, 2887.34, 2887.21, 3021.7, 2964.61, 2807.67, 2814.2, 2900.88, 2604.8, 3330.18, 0.0,
//
   0.0, 2773.8, 2415.13, 2698.35, 2815.15, 2832.7, 2767.04, 2951.54, 3424.48, 2821.08, 3117.57, 3071.27, 3405.28, 3236.87, 2979.43, 2855.74, 2865.74, 2706.25, 2816.8, 0.0,
   0.0, 2488.32, 2026.33, 2827.07, 3217.7, 2745.57, 3005.12, 2625.65, 2892.49, 2968.37, 3117.16, 2917.88, 2897.28, 3336.53, 2734.44, 3487.47, 2808.68, 2663.77, 3230.12, 0.0,
   0.0, 2752.69, 2822.87, 3190.11, 2833.7, 2757.82, 2936.43, 3039.44, 2797.05, 2715.93, 2975.38, 2853.35, 2857.33, 3466.81, 3222.28, 2746.64, 2664.91, 2942.43, 3019.91, 0.0,
   0.0, 2739.07, 2827.06, 2792.29, 2271.22, 2805.07, 2486.66, 2276.78, 2765.03, 2963.27, 2653.15, 2579.8, 3302.41, 3137.43, 3141.73, 2825.02, 3057.72, 2786.84, 2690.5, 0.0,
   0.0, 2888.7, 2546.79, 2908.24, 2780.61, 2906.86, 3314.75, 3106.97, 3033.48, 3041.77, 2841.53, 2667.07, 2880.55, 2972.32, 3179.55, 3117.97, 2951.07, 3388.83, 3449.22, 0.0,
//
   0.0, 2768.02, 2303.46, 2803.46, 2622.72, 2292.72, 2667.95, 2582.67, 2951.93, 2800.26, 3151.58, 2882.46, 3030.87, 3141.97, 3126.41, 3341.3, 2686.55, 2545.24, 3390.11, 0.0,
   0.0, 2414.69, 2470.74, 2611.94, 2647.45, 2423.39, 2608.3, 2276.45, 2485.98, 2584.66, 3133.45, 3426.55, 2667.1, 2962.01, 2948.8, 2967.54, 3158.43, 2586.37, 2798.46, 0.0,
   0.0, 2770.73, 2372, 1766.48, 2738.81, 3142.5, 2609.9, 2975.05, 2681.23, 2820.64, 2538.64, 2740.82, 2776.82, 2492.22, 3088.22, 2828.46, 3058.16, 3223.74, 2993.81, 0.0,
   0.0, 2293.92, 1545.11, 2160.44, 2771.72, 2602.59, 2410.17, 2656.02, 2711.45, 2800.98, 2867.34, 2683.17, 2811.27, 2497.78, 2911.4, 2693.63, 3010.6, 2708.76, 2661.58, 0.0,
   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
};

  double* raw_orography_in = new double[20*20]
  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
   0.0, 2821.9, 2549.44, 2472.69, 2688.66, 2683.13, 2719.41, 2683.89, 2244.59, 2483.89, 2689.2, 2432.77, 2797.78, 2544.55, 2494.41, 2536.93, 2465.66, 2440.65, 2225.69, 0.0,
   0.0, 2748.9, 2447.34, 2755.49, 2874.7, 2346.54, 2536.99, 2721.65, 2468.33, 2546.6, 2963.46, 2564.8, 2937.53, 3036.52, 2731.27, 2529.42, 2821.21, 2742.64, 2499.66, 0.0,
   0.0, 2395.91, 2848.61, 2678.34, 3011.72, 2613.43, 2745.54, 2989.27, 2920.23, 2291.84, 2545.44, 2789.72, 2346.59, 2786.36, 2684.83, 2438.92, 2610.69, 2689.3, 2602.33, 0.0,
   0.0, 2483.7, 2735.31, 2976.6, 3233.42, 2668.57, 2292.6, 2740.1, 2857.81, 2743.6, 2444.06, 2573.87, 2855.08, 2613.29, 2701.57, 3352.55, 2564, 3025.03, 2427.56, 0.0,
   0.0, 2267.33, 2782.81, 2527.5, 2766.48, 2957.41, 3343.05, 3141.05, 2566.35, 2650.34, 2742.92, 2280.02, 2626.72, 2881.62, 3167.12, 3115.05, 2838.05, 2636.08, 2783.24, 0.0,
   0.0, 2820.65, 2911.99, 2642.95, 3150.44, 2533.18, 3067.33, 3084.39, 2840.64, 2760.65, 2403.02, 2529.29, 3511.31, 2271.61, 2227.3, 2508.7, 2858.88, 3293.69, 3142.58, 0.0,
   0.0, 3114.9, 3131.08, 2820.27, 3287.25, 3384.07, 3141.46, 3457.63, 2889.2, 2867.08, 2273.87, 3345.01, 3061.76, 3106.28, 2781.76, 3295.93, 3217.44, 2903.97, 2791.47, 0.0,
   0.0, 2906.96, 2959.25, 2909.48, 2775.41, 2819.85, 2863.38, 3402.57, 3294.53, 3408.99, 3257.53, 2952.45, 2855.42, 2938.21, 2984.79, 2621.18, 3244.9, 3160.94, 2213.43, 0.0,
  0.0, 2762.45, 2828.82, 2774.3, 2822.31, 3045.13, 2921.17, 2639.54, 3239.07, 3116.82, 2887.34, 2887.21, 3021.7, 2964.61, 2807.67, 2814.2, 2900.88, 2604.8, 3330.18, 0.0,
  0.0, 2773.8, 2415.13, 2698.35, 2815.15, 2832.7, 2767.04, 2951.54, 3424.48, 2821.08, 3117.57, 3071.27, 3405.28, 3236.87, 2979.43, 2855.74, 2865.74, 2706.25, 2816.8, 0.0,
  0.0, 2488.32, 2026.33, 2827.07, 3217.7, 2745.57, 3005.12, 2625.65, 2892.49, 2968.37, 3117.16, 2917.88, 2897.28, 3336.53, 2734.44, 3487.47, 2808.68, 2663.77, 3230.12, 0.0,
  0.0, 2752.69, 2822.87, 3190.11, 2833.7, 2757.82, 2936.43, 3039.44, 2797.05, 2715.93, 2975.38, 2853.35, 2857.33, 3466.81, 3222.28, 2746.64, 2664.91, 2942.43, 3019.91, 0.0,
  0.0, 2739.07, 2827.06, 2792.29, 2271.22, 2805.07, 2486.66, 2276.78, 2765.03, 2963.27, 2653.15, 2579.8, 3302.41, 3137.43, 3141.73, 2825.02, 3057.72, 2786.84, 2690.5, 0.0,
  0.0, 2888.7, 2546.79, 2908.24, 2780.61, 2906.86, 3314.75, 3106.97, 3033.48, 3041.77, 2841.53, 2667.07, 2880.55, 2972.32, 3179.55, 3117.97, 2951.07, 3388.83, 3449.22, 0.0,
  0.0, 2768.02, 2303.46, 2803.46, 2622.72, 2292.72, 2667.95, 2582.67, 2951.93, 2800.26, 3151.58, 2882.46, 3030.87, 3141.97, 3126.41, 3341.3, 2686.55, 2545.24, 3390.11, 0.0,
  0.0, 2414.69, 2470.74, 2611.94, 2647.45, 2423.39, 2608.3, 2276.45, 2485.98, 2584.66, 3133.45, 3426.55, 2667.1, 2962.01, 2948.8, 2967.54, 3158.43, 2586.37, 2798.46, 0.0,
  0.0, 2770.73, 2372, 1766.48, 2738.81, 3142.5, 2609.9, 2975.05, 2681.23, 2820.64, 2538.64, 2740.82, 2776.82, 2492.22, 3088.22, 2828.46, 3058.16, 3223.74, 2993.81, 0.0,
  0.0, 2293.92, 1545.11, 2160.44, 2771.72, 2602.59, 2410.17, 2656.02, 2711.45, 2800.98, 2867.34, 2683.17, 2811.27, 2497.78, 2911.4, 2693.63, 3010.6, 2708.76, 2661.58, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

  bool* minima_in = new bool[20*20]
  {false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
   false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
   false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
   false, false, false, false, false, false, false, false, false, true, false, false, true, false, false, true, false, false, false, false,
   false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false,
   false, false, false, true, false, false, false, false, true, false, false, true, false, false, false, false, false, false, false, false,
   false, false, false, false, false, true, false, false, false, false, false, false, false, false, true, false, false, false, false, false,
   false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false,
   false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, true, false, false, false, false,
   false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false,
   false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false,
   false, false, true, false, false, true, false, true, false, false, false, false, false, false, true, false, false, true, false, false,
   false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
   false, false, false, false, true, false, false, true, false, false, false, true, false, false, false, false, false, false, false, false,
   false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
   false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, true, false, false,
   false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false,
   false, false, false, false, false, false, false, false, false, false, true, false, false, true, false, false, false, false, false, false,
   false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,false, false, false, false, false,
   false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false };
  double* prior_fine_rdirs_in = new double[20*20] {
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   8,   7,   1,   3,   2,   1,   6,   5,   4,   6,   8,   4,   9,   8,   9,   8,   7,   9,   0,
    0,   2,   1,   4,   6,   5,   4,   9,   8,   7,   1,   3,   2,   1,   3,   2,   1,   9,   8,   0,
    0,   4,   4,   7,   9,   3,   2,   1,   6,   5,   4,   6,   5,   4,   6,   5,   4,   3,   9,   0,
    0,   2,   1,   2,   1,   6,   5,   4,   9,   8,   3,   2,   1,   7,   9,   8,   7,   6,   9,   0,
    0,   4,   4,   5,   4,   9,   8,   7,   5,   3,   6,   5,   3,   3,   2,   1,   1,   9,   8,   0,
    0,   1,   7,   8,   7,   5,   4,   9,   8,   3,   2,   1,   6,   6,   5,   4,   4,   8,   7,   0,
    0,   4,   9,   8,   9,   8,   7,   9,   9,   6,   5,   4,   9,   9,   8,   7,   7,   3,   2,   0,
    0,   7,   1,   2,   1,   4,   3,   2,   1,   9,   8,   7,   5,   9,   6,   5,   3,   6,   3,   0,
    0,   3,   2,   1,   1,   3,   6,   5,   4,   2,   1,   9,   8,   6,   9,   8,   6,   9,   8,   0,
    0,   3,   2,   1,   4,   2,   3,   2,   1,   5,   4,   8,   7,   3,   2,   1,   9,   8,   7,   0,
    0,   6,   5,   4,   7,   5,   6,   5,   4,   2,   1,   2,   1,   6,   5,   3,   6,   5,   4,   0,
    0,   9,   8,   7,   2,   1,   3,   2,   1,   3,   3,   2,   1,   9,   8,   6,   9,   8,   7,   0,
    0,   3,   2,   6,   5,   4,   6,   5,   4,   6,   6,   5,   4,   7,   9,   9,   8,   7,   3,   0,
    0,   3,   2,   9,   8,   7,   9,   8,   7,   9,   9,   8,   7,   4,   9,   3,   3,   2,   3,   0,
    0,   1,   5,   4,   6,   5,   3,   2,   1,   1,   1,   8,   7,   1,   2,   6,   6,   5,   6,   0,
    0,   4,   3,   2,   1,   8,   6,   5,   4,   4,   2,   1,   3,   2,   1,   9,   9,   8,   9,   0,
    0,   3,   2,   1,   4,   3,   9,   8,   7,   7,   5,   4,   6,   5,   4,   2,   9,   8,   7,   0,
    0,   6,   1,   4,   7,   3,   2,   1,   2,   1,   8,   7,   9,   8,   7,   3,   4,   6,   9,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
  };
  double* prior_coarse_rdirs_in = new double[4*4]{
    5.0,5.0,5.0,5.0,
    5.0,5.0,5.0,5.0,
    5.0,5.0,5.0,5.0,
    5.0,5.0,5.0,5.0 };
  int* prior_fine_catchments_in = new int[20*20] {
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   0, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 8, 0,
   0, 2, 2, 2, 3, 3, 3, 4, 4, 4, 9, 9, 9, 9, 10, 10, 10, 8, 11, 0,
   0, 2, 2, 2, 3, 12, 12, 12, 9, 9, 9, 9, 9, 9, 10, 10, 10, 13, 14, 0,
   0, 15, 15, 16, 16, 12, 12, 12, 9, 9, 17, 17, 17, 18, 10, 10, 10, 19, 19, 0,
   0, 15, 15, 16, 16, 12, 12, 12, 20, 21, 17, 17, 22, 22, 22, 22, 22, 19, 19, 0,
   0, 23, 15, 16, 16, 24, 24, 20, 20, 21, 21, 21, 22, 22, 22, 22, 22, 19, 19, 0,
   0, 23, 16, 16, 24, 24, 24, 20, 21, 21, 21, 21, 22, 22, 22, 22, 22, 25, 25, 0,
   0, 23, 26, 26, 26, 26, 27, 27, 27, 21, 21, 21, 28, 22, 29, 29, 25, 25, 25, 0,
   0, 26, 26, 26, 26, 30, 27, 27, 27, 31, 31, 28, 28, 29, 29, 29, 25, 25, 25, 0,
   0, 26, 26, 26, 26, 33, 30, 30, 30, 31, 31, 28, 28, 32, 32, 32, 25, 25, 25, 0,
   0, 26, 26, 26, 26, 33, 30, 30, 30, 34, 34, 34, 34, 32, 32, 35, 35, 35, 35, 0,
   0, 26, 26, 26, 36, 36, 37, 37, 37, 34, 34, 34, 34, 32, 32, 35, 35, 35, 35, 0,
   0, 38, 38, 36, 36, 36, 37, 37, 37, 34, 34, 34, 34, 34, 35, 35, 35, 35, 40, 0,
   0, 38, 38, 36, 36, 36, 37, 37, 37, 34, 34, 34, 34, 34, 35, 39, 39, 39, 40, 0,
   0, 41, 38, 38, 42, 42, 43, 43, 43, 43, 43, 34, 34, 44, 44, 39, 39, 39, 40, 0,
   0, 41, 45, 45, 45, 42, 43, 43, 43, 43, 46, 46, 44, 44, 44, 39, 39, 39, 40, 0,
   0, 45, 45, 45, 45, 48, 43, 43, 43, 43, 46, 46, 44, 44, 44, 47, 39, 39, 39, 0,
   0, 45, 45, 45, 45, 48, 48, 48, 49, 49, 46, 46, 44, 44, 44, 47, 47, 50, 50, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  double* cell_areas_in = new double[20*20]{
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
//
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
//
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
//
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1
  };
  double* connection_volume_thresholds_in = new double[20*20];
  std::fill_n(connection_volume_thresholds_in,20*20,0.0);
  double* flood_volume_thresholds_in = new double[20*20];
  std::fill_n(flood_volume_thresholds_in,20*20,0.0);
  double* connection_heights_in = new double[20*20];
  std::fill_n(connection_heights_in,20*20,0.0);
  double* flood_heights_in = new double[20*20];
  std::fill_n(flood_heights_in,20*20,0.0);
  int* flood_next_cell_lat_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lat_index_in,20*20,-1);
  int* flood_next_cell_lon_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lon_index_in,20*20,-1);
  int* connect_next_cell_lat_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lat_index_in,20*20,-1);
  int* connect_next_cell_lon_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lon_index_in,20*20,-1);
  double* flood_volume_thresholds_expected_out = new double[20*20] {
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, 190.45,343.33,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, 563.89,-1.0,-1.0,-1.0,152.22, -1.0,-1.0,218.21,-1.0,-1.0, 90.5,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, 636.69,320.83,-1.0,-1.0,-1.0, 273.57,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
//
   -1.0,-1.0,-1.0,115.45,-1.0, -1.0,-1.0,-1.0,83.99,-1.0, -1.0,123.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,300.17,-1.0, 233.3,-1.0,-1.0,-1.0,-1.0, 246.12,-1.0,-1.0,518.49,44.31, 872.55,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, 129.15,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,31.79,-1.0,-1.0, 160.58,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,127.5,-1.0,-1.0, 86.86,32.05,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
//
   -1.0,-1.0,535.18,-1.0,-1.0, -1.0,719.16,-1.0,-1.0,66.26, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,388.8,-1.0,-1.0, 12.25,-1.0,141.39,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,12.2, -1.0,-1.0,1.14,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, 522.65,-1.0,-1.0,-1.0,444.17, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,83.82,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,486.6, -1.0,752.2,209.88,542.75,-1.0, 101.19,73.35,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,247.77,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
//
   -1.0,-1.0,111.23,-1.0,-1.0, 130.67,-1.0,408.88,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,577.22,41.13,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, 500.49,539.88,209.53,402.91,527.08, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,241.49,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, 46.02,-1.0,-1.0,5.56,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0
 };
  double* connection_volume_thresholds_expected_out = new double[20*20]{
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
//
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
//
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
//
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,
   -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0 };
  int* flood_next_cell_lat_index_expected_out = new int[20*20]{
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,   2,   3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,   4,  -1,  -1,  -1,   4,  -1,  -1,   1,  -1,  -1,   1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,   0,   2,  -1,  -1,  -1,   1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
//
     -1,  -1,  -1,   6,  -1,  -1,  -1,  -1,   6,  -1,  -1,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,   5,  -1,   4,  -1,  -1,  -1,  -1,   3,  -1,  -1,   6,   6,   5,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,   9,  -1,  -1,   6,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  11,  -1,  -1,  11,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
//
     -1,  -1,  12,  -1,  -1,  -1,  15,  -1,  -1,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  10,  -1,  -1,  13,  -1,  11,  -1,  -1,  -1,  -1,  -1,  -1,  12,  -1,  -1,  12,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  13,  -1,  -1,  -1,  13,  -1,  -1,  -1,  -1,  -1,  -1,   9,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  12,  -1,  13,  13,  10,  -1,  14,  13,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  12,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
//
     -1,  -1,  17,  -1,  -1,  16,  -1,  16,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  17,  16,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  16,  18,  16,  15,  16,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  15,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  16,  -1,  -1,  19,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1  };
  int* flood_next_cell_lon_index_expected_out = new int[20*20]{
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,   6,   5,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,   5,  -1,  -1,  -1,  10,  -1,  -1,  11,  -1,  -1,  16,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,   6,   5,  -1,  -1,  -1,   8,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
//
     -1,  -1,  -1,   3,  -1,  -1,  -1,  -1,  10,  -1,  -1,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,   1,  -1,   5,  -1,  -1,  -1,  -1,   9,  -1,  -1,  15,  13,  11,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  11,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  11,  -1,  -1,  14,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,   7,  -1,  -1,   7,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
//
     -1,  -1,   0,  -1,  -1,  -1,   5,  -1,  -1,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,   2,  -1,  -1,   4,  -1,   5,  -1,  -1,  -1,  -1,  -1,  -1,  16,  -1,  -1,  16,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,   8,  -1,  -1,  -1,   7,  -1,  -1,  -1,  -1,  -1,  -1,  17,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,   5,  -1,   4,   6,   6,  -1,  11,  10,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
//
     -1,  -1,   0,  -1,  -1,   5,  -1,   9,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  19,  17,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,   7,   6,   8,   7,   6,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  16,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,   8,  -1,  -1,  14,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
     -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1
 };
  int* connect_next_cell_lat_index_expected_out = new int[20*20]{
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
    -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1,  -1, -1, -1, -1, -1
  };
  int* connect_next_cell_lon_index_expected_out = new int[20*20]{
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
  double* connection_heights_expected_out = new double[20*20]{
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  double* flood_heights_expected_out = new double[20*20]{
     0, 0,  0,      0,      0,  0,      0,      0,    0,      0,      0,     0,     0,   0,      0,    0,      0,       0,     0,  0,
     0, 0,  0,      0,      0,  0,      0,      0,    0,      0,      0,     0,     0,   0,      0,    0,      0,       0,     0,  0,
     0, 0,  0,      0,      0,2536.99,2613.43,  0,    0,      0,      0,     0,     0,   0,      0,    0,      0,       0,     0,  0,
     0, 0,  0,      0,      0,2668.57,  0,      0,    0,    2444.06,  0,     0,   2564.8,0,      0,   2529.42, 0,       0,     0,  0,
     0, 0,  0,      0,      0,2683.13,2613.43,  0,    0,      0,   2468.33,  0,     0,   0,      0,    0,      0,       0,     0,  0,
     0, 0,  0,2642.95,      0,  0,      0,      0,  2650.34,  0,      0,   2403.02, 0,   0,      0,    0,      0,       0,     0,  0,
     0, 0,  0,2735.31,      0,2766.48,  0,      0,    0,      0,   2444.06,  0,     0, 2508.7,2271.61,2626.72, 0,       0,     0,  0,
     0, 0,  0,      0,      0,  0,      0,      0,    0,      0,   2403.02,  0,     0,   0,      0,    0,      0,       0,     0,  0,
     0, 0,  0,      0,      0,  0,      0,      0,    0,      0,      0,     0,  2887.21,0,      0,   2781.76, 0,       0,     0,  0,
     0, 0,  0,      0,      0,  0,      0,   2767.04, 0,      0,   2892.49,2887.34, 0,   0,      0,    0,      0,       0,     0,  0,
     0, 0,2488.32,  0,      0,  0,    2780.61,  0,    0,    2887.34,  0,     0,     0,   0,      0,    0,      0,       0,     0,  0,
     0, 0,2415.13,  0,      0,2757.82,  0,   2767.04, 0,      0,      0,     0,     0,   0,   2746.64, 0,      0,      2664.91,0,  0,
     0, 0,  0,      0,      0,2765.03,  0,      0,    0,    2765.03,  0,     0,     0,   0,      0,    0,     2706.25,  0,     0,  0,
     0, 0,  0,      0,2757.82,  0,    2757.82,2486.66,2767.04,0,   2667.07,2653.15, 0,   0,      0,    0,      0,       0,     0,  0,
     0, 0,  0,      0,      0,  0,      0,      0,    0,      0,      0,   2715.93, 0,   0,      0,    0,      0,       0,     0,  0,
     0, 0,2414.69,  0,      0,2423.39,  0,    2584.66,0,      0,      0,     0,     0,   0,      0,    0,     2798.46, 2586.37,0,  0,
     0, 0,  0,      0,      0,2608.3,2609.9,2485.98,2582.67,2608.3,   0,     0,     0,   0,      0,    0,      0,      2686.55,0,  0,
     0, 0,  0,      0,      0,  0,      0,      0,    0,      0,   2584.66,  0,     0, 2497.78,  0,    0,      0,       0,     0,  0,
     0, 0,  0,      0,      0,  0,      0,      0,    0,      0,      0,     0,     0,   0,      0,    0,      0,       0,     0,  0,
     0, 0,  0,      0,      0,  0,      0,      0,    0,      0,      0,     0,     0,   0,      0,    0,      0,       0,     0,  0 };
  int flood_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;

  //0
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(17,0),
                                                new latlon_coords(3,0),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(15,2);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //1
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(12,0),
                                                new latlon_coords(2,0),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(10,2);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //2
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(19,14),
                                                new latlon_coords(3,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(17,13);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //3
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,16),
                                                new latlon_coords(0,3),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(3,15);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //4
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,11),
                                                new latlon_coords(0,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(3,12);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //5
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(5,1),
                                                new latlon_coords(1,0),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,3);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //6
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(12,16),
                                                new latlon_coords(2,3),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(11,14);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //7
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(9,17),
                                                new latlon_coords(1,3),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(12,16);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //8

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,5),
                                                new latlon_coords(0,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,5);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;



  //9
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(2,5),
                                                new latlon_coords(2,5),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(4,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //10
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(4,6),
                                                new latlon_coords(4,6),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(2,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //11
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(0,6),
                                                new latlon_coords(0,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(4,5);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //12
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(6,14),
                                                new latlon_coords(1,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,15);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //13
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(6,10),
                                                new latlon_coords(1,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(5,8);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //14
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(5,11),
                                                new latlon_coords(1,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,15);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //15
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(5,11),
                                                new latlon_coords(5,11),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(7,10);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //16
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(7,10),
                                                new latlon_coords(7,10),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(5,11);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //17
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(3,9),
                                                new latlon_coords(3,9),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,10);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //18
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(5,11),
                                                new latlon_coords(1,2),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(3,9);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //19
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,8),
                                                new latlon_coords(0,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(4,10);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //20
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(17,19),
                                                new latlon_coords(3,3),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(15,16);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //21
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(10,9),
                                                new latlon_coords(10,9),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(9,11);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //22
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(8,12),
                                                new latlon_coords(8,12),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(10,9);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //23
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(11,7),
                                                new latlon_coords(2,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(9,10);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //24
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(11,7),
                                                new latlon_coords(11,7),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(9,7);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //25
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(13,4),
                                                new latlon_coords(3,2),
                                                false);
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(9,7),
                                                new latlon_coords(1,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(11,7);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //26
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(13,4),
                                                new latlon_coords(3,2),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(12,9);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //27
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(13,7),
                                                new latlon_coords(2,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(11,5);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //28
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(13,4),
                                                new latlon_coords(13,4),
                                                true);
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(11,5),
                                                new latlon_coords(11,5),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(13,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //29
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(13,7),
                                                new latlon_coords(13,7),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(13,4);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //30
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(13,11),
                                                new latlon_coords(2,2),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(12,5);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //31
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(11,7),
                                                new latlon_coords(11,7),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(13,8);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //32
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(15,5),
                                                new latlon_coords(3,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(10,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //33
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(16,7),
                                                new latlon_coords(16,7),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(17,10);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //34
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(16,7),
                                                new latlon_coords(16,7),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(16,5);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //35
  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(17,10),
                                                new latlon_coords(3,1),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(15,7);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //36
    primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(15,5),
                                                new latlon_coords(15,5),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(16,9);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  //37
  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(18,6),
                                                new latlon_coords(3,1),
                                                false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(16,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.setup_fields(minima_in,
                          raw_orography_in,
                          corrected_orography_in,
                          cell_areas_in,
                          connection_volume_thresholds_in,
                          flood_volume_thresholds_in,
                          connection_heights_in,
                          flood_heights_in,
                          prior_fine_rdirs_in,
                          prior_coarse_rdirs_in,
                          prior_fine_catchments_in,
                          coarse_catchment_nums_in,
                          flood_next_cell_lat_index_in,
                          flood_next_cell_lon_index_in,
                          connect_next_cell_lat_index_in,
                          connect_next_cell_lon_index_in,
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
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),0.0001));
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
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  EXPECT_TRUE(field<double>(connection_heights_expected_out,grid_params_in)
              == field<double>(connection_heights_in,grid_params_in));
  EXPECT_TRUE(field<double>(flood_heights_expected_out,grid_params_in)
              == field<double>(flood_heights_in,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in; delete alg4;
  delete[] coarse_catchment_nums_in; delete[] corrected_orography_in;
  delete[] raw_orography_in; delete[] minima_in;
  delete[] flood_heights_in; delete[] connection_heights_in;
  delete[] prior_coarse_rdirs_in;
  delete[] prior_fine_rdirs_in; delete[] prior_fine_catchments_in;
  delete[] connection_volume_thresholds_in; delete[] flood_volume_thresholds_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flood_volume_thresholds_expected_out;
  delete[] connection_volume_thresholds_expected_out;
  delete[] flood_next_cell_lat_index_expected_out;
  delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out;
  delete[] connect_next_cell_lon_index_expected_out;
  delete[] landsea_in;
  delete[] true_sinks_in;
  delete[] next_cell_lat_index_in;
  delete[] next_cell_lon_index_in;
  delete[] sinkless_rdirs_out;
  delete[] catchment_nums_in;
  delete[] cell_areas_in;
}

// TEST_F(BasinEvaluationTest, TestEvaluateBasinsFive) {
//   int ncells = 80;
//   int* cell_neighbors = new int[80*3] {
//     //1
//     5,7,2,
//     //2
//     1,10,3,
//     //3
//     2,13,4,
//     //4
//     3,16,5,
//     //5
//     4,19,1,
//     //6
//     20,21,7,
//     //7
//     1,6,8,
//     //8
//     7,23,9,
//     //9
//     8,25,10,
//     //10
//     2,9,11,
//     //11
//     10,27,12,
//     //12
//     11,29,13,
//     //13
//     3,12,14,
//     //14
//     13,31,15,
//     //15
//     14,33,16,
//     //16
//     4,15,17,
//     //17
//     16,35,18,
//     //18
//     17,37,19,
//     //19
//     5,18,20,
//     //20
//     19,39,6,
//     //21
//     6,40,22,
//     //22
//     21,41,23,
//     //23
//     8,22,24,
//     //24
//     23,43,25,
//     //25
//     24,26,9,
//     //26
//     25,45,27,
//     //27
//     11,26,28,
//     //28
//     27,47,29,
//     //29
//     12,28,30,
//     //30
//     29,49,31,
//     //31
//     14,30,32,
//     //32
//     31,51,33,
//     //33
//     15,32,34,
//     //34
//     33,53,35,
//     //35
//     17,34,36,
//     //36
//     35,55,37,
//     //37
//     18,36,38,
//     //38
//     37,57,39,
//     //39
//     20,38,40,
//     //40
//     39,59,21,
//     //41
//     22,60,42,
//     //42
//     41,61,43,
//     //43
//     24,42,44,
//     //44
//     43,63,45,
//     //45
//     26,44,46,
//     //46
//     45,64,47,
//     //47
//     28,46,48,
//     //48
//     47,66,49,
//     //49
//     30,48,50,
//     //50
//     49,67,51,
//     //51
//     32,50,52,
//     //52
//     51,69,53,
//     //53
//     34,52,54,
//     //54
//     53,70,55,
//     //55
//     36,54,56,
//     //56
//     55,72,57,
//     //57
//     38,56,58,
//     //58
//     57,73,59,
//     //59
//     40,58,60,
//     //60
//     59,75,41,
//     //61
//     42,75,62,
//     //62
//     61,76,63,
//     //63
//     44,62,64,
//     //64
//     46,63,65,
//     //65
//     64,77,66,
//     //66
//     48,65,67,
//     //67
//     50,66,68,
//     //68
//     67,78,69,
//     //69
//     52,68,70,
//     //70
//     54,69,71,
//     //71
//     70,79,72,
//     //72
//     56,71,73,
//     //73
//     58,72,74,
//     //74
//     73,80,75,
//     //75
//     60,74,61,
//     //76
//     62,80,77,
//     //77
//     65,76,78,
//     //78
//     68,77,79,
//     //79
//     71,78,80,
//     //80
//     74,79,76
//   };
//   int* prior_fine_rdirs_in = new int[80] {
//     //1
//     8,
//     //2
//     13,
//     //3
//     13,
//     //4
//     13,
//     //5
//     19,
//     //6
//     8,
//     //7
//     8,
//     //8
//     24,
//     //9
//     24,
//     //10
//     13,
//     //11
//     13,
//     //12
//     13,
//     //13
//     -2,
//     //14
//     13,
//     //15
//     13,
//     //16
//     13,
//     //17
//     36,
//     //18
//     36,
//     //19
//     37,
//     //20
//     37,
//     //21
//     8,
//     //22
//     24,
//     //23
//     24,
//     //24
//     64,
//     //25
//     45,
//     //26
//     45,
//     //27
//     45,
//     //28
//     49,
//     //29
//     49,
//     //30
//     13,
//     //31
//     13,
//     //32
//     30,
//     //33
//     52,
//     //34
//     55,
//     //35
//     55,
//     //36
//     55,
//     //37
//     55,
//     //38
//     55,
//     //39
//     37,
//     //40
//     38,
//     //41
//     61,
//     //42
//     61,
//     //43
//     64,
//     //44
//     64,
//     //45
//     64,
//     //46
//     64,
//     //47
//     64,
//     //48
//     64,
//     //49
//     -2,
//     //50
//     49,
//     //51
//     30,
//     //52
//     54,
//     //53
//     55,
//     //54
//     55,
//     //55
//     0,
//     //56
//     55,
//     //57
//     55,
//     //58
//     38,
//     //59
//     38,
//     //60
//     59,
//     //61
//     63,
//     //62
//     64,
//     //63
//     64,
//     //64
//     -2,
//     //65
//     64,
//     //66
//     38,
//     //67
//     49,
//     //68
//     52,
//     //69
//     55,
//     //70
//     55,
//     //71
//     55,
//     //72
//     55,
//     //73
//     56,
//     //74
//     58,
//     //75
//     58,
//     //76
//     64,
//     //77
//     64,
//     //78
//     68,
//     //79
//     71,
//     //80
//     71
//   };
//   int* prior_coarse_rdirs_in = new int[80];
//   std::copy_n(prior_fine_rdirs_in,80,prior_coarse_rdirs_in);
//   double* raw_orography_in = new double[80] {
//     //1
//     10.0,
//     //2
//     10.0,
//     //3
//     10.0,
//     //4
//     10.0,
//     //5
//     10.0,
//     //6
//     10.0,
//     //7
//     10.0,
//     //8
//     10.0,
//     //9
//     10.0,
//     //10
//     10.0,
//     //11
//     10.0,
//     //12
//     10.0,
//     //13
//      3.0,
//     //14
//     10.0,
//     //15
//     10.0,
//     //16
//     10.0,
//     //17
//     10.0,
//     //18
//     10.0,
//     //19
//     10.0,
//     //20
//     10.0,
//     //21
//     10.0,
//     //22
//     10.0,
//     //23
//     10.0,
//     //24
//     10.0,
//     //25
//     10.0,
//     //26
//     10.0,
//     //27
//     10.0,
//     //28
//     10.0,
//     //29
//     10.0,
//     //30
//      6.0,
//     //31
//     10.0,
//     //32
//     10.0,
//     //33
//     10.0,
//     //34
//     10.0,
//     //35
//     10.0,
//     //36
//     10.0,
//     //37
//     10.0,
//     //38
//     10.0,
//     //39
//     10.0,
//     //40
//     10.0,
//     //41
//     10.0,
//     //42
//     10.0,
//     //43
//     10.0,
//     //44
//     10.0,
//     //45
//      4.0,
//     //46
//      4.0,
//     //47
//      7.0,
//     //48
//     10.0,
//     //49
//      5.0,
//     //50
//      9.0,
//     //51
//     10.0,
//     //52
//      8.0,
//     //53
//     10.0,
//     //54
//      6.0,
//     //55
//      0.0,
//     //56
//     10.0,
//     //57
//     10.0,
//     //58
//     10.0,
//     //59
//     10.0,
//     //60
//     10.0,
//     //61
//     10.0,
//     //62
//     10.0,
//     //63
//      4.0,
//     //64
//      3.0,
//     //65
//     10.0,
//     //66
//     10.0,
//     //67
//     10.0,
//     //68
//     10.0,
//     //69
//     10.0,
//     //70
//     10.0,
//     //71
//     10.0,
//     //72
//     10.0,
//     //73
//     10.0,
//     //74
//     10.0,
//     //75
//     10.0,
//     //76
//      5.0,
//     //77
//     10.0,
//     //78
//     10.0,
//     //79
//     10.0,
//     //80
//     10.0
//   };
//   double* corrected_orography_in = new double[80];
//   std::copy_n(raw_orography_in,80,corrected_orography_in);
//   bool* minima_in = new bool[80] {
//     //1
//     false,
//     //2
//     false,
//     //3
//     false,
//     //4
//     false,
//     //5
//     false,
//     //6
//     false,
//     //7
//     false,
//     //8
//     false,
//     //9
//     false,
//     //10
//     false,
//     //11
//     false,
//     //12
//     false,
//     //13
//      true,
//     //14
//     false,
//     //15
//     false,
//     //16
//     false,
//     //17
//     false,
//     //18
//     false,
//     //19
//     false,
//     //20
//     false,
//     //21
//     false,
//     //22
//     false,
//     //23
//     false,
//     //24
//     false,
//     //25
//     false,
//     //26
//     false,
//     //27
//     false,
//     //28
//     false,
//     //29
//     false,
//     //30
//     false,
//     //31
//     false,
//     //32
//     false,
//     //33
//     false,
//     //34
//     false,
//     //35
//     false,
//     //36
//     false,
//     //37
//     false,
//     //38
//     false,
//     //39
//     false,
//     //40
//     false,
//     //41
//     false,
//     //42
//     false,
//     //43
//     false,
//     //44
//     false,
//     //45
//     false,
//     //46
//     false,
//     //47
//     false,
//     //48
//     false,
//     //49
//      true,
//     //50
//     false,
//     //51
//     false,
//     //52
//     false,
//     //53
//     false,
//     //54
//     false,
//     //55
//     false,
//     //56
//     false,
//     //57
//     false,
//     //58
//     false,
//     //59
//     false,
//     //60
//     false,
//     //61
//     false,
//     //62
//     false,
//     //63
//     false,
//     //64
//      true,
//     //65
//     false,
//     //66
//     false,
//     //67
//     false,
//     //68
//     false,
//     //69
//     false,
//     //70
//     false,
//     //71
//     false,
//     //72
//     false,
//     //73
//     false,
//     //74
//     false,
//     //75
//     false,
//     //76
//     false,
//     //77
//     false,
//     //78
//     false,
//     //79
//     false,
//     //80
//     false,
//   };
//   int* basin_numbers_expected_out = new int[80] {
//     //1
//     0,
//     //2
//     0,
//     //3
//     0,
//     //4
//     0,
//     //5
//     0,
//     //6
//     0,
//     //7
//     0,
//     //8
//     0,
//     //9
//     0,
//     //10
//     0,
//     //11
//     0,
//     //12
//     0,
//     //13
//     2,
//     //14
//     0,
//     //15
//     0,
//     //16
//     0,
//     //17
//     0,
//     //18
//     0,
//     //19
//     0,
//     //20
//     0,
//     //21
//     0,
//     //22
//     0,
//     //23
//     0,
//     //24
//     0,
//     //25
//     0,
//     //26
//     0,
//     //27
//     0,
//     //28
//     0,
//     //29
//     0,
//     //30
//     3,
//     //31
//     0,
//     //32
//     0,
//     //33
//     0,
//     //34
//     0,
//     //35
//     0,
//     //36
//     0,
//     //37
//     0,
//     //38
//     0,
//     //39
//     0,
//     //40
//     0,
//     //41
//     0,
//     //42
//     0,
//     //43
//     0,
//     //44
//     0,
//     //45
//     1,
//     //46
//     1,
//     //47
//     3,
//     //48
//     0,
//     //49
//     3,
//     //50
//     0,
//     //51
//     0,
//     //52
//     0,
//     //53
//     0,
//     //54
//     0,
//     //55
//     0,
//     //56
//     0,
//     //57
//     0,
//     //58
//     0,
//     //59
//     0,
//     //60
//     0,
//     //61
//     0,
//     //62
//     0,
//     //63
//     1,
//     //64
//     1,
//     //65
//     0,
//     //66
//     0,
//     //67
//     0,
//     //68
//     0,
//     //69
//     0,
//     //70
//     0,
//     //71
//     0,
//     //72
//     0,
//     //73
//     0,
//     //74
//     0,
//     //75
//     0,
//     //76
//     1,
//     //77
//     0,
//     //78
//     0,
//     //79
//     0,
//     //80
//     0
//   };
//   double* flood_volume_thresholds_expected_out = new double[80] {
//     //1
//     -1.0,
//     //2
//     -1.0,
//     //3
//     -1.0,
//     //4
//     -1.0,
//     //5
//     -1.0,
//     //6
//     -1.0,
//     //7
//     -1.0,
//     //8
//     -1.0,
//     //9
//     -1.0,
//     //10
//     -1.0,
//     //11
//     -1.0,
//     //12
//     -1.0,
//     //13
//      3.0,
//     //14
//     -1.0,
//     //15
//     -1.0,
//     //16
//     -1.0,
//     //17
//     -1.0,
//     //18
//     -1.0,
//     //19
//     -1.0,
//     //20
//     -1.0,
//     //21
//     -1.0,
//     //22
//     -1.0,
//     //23
//     -1.0,
//     //24
//     -1.0,
//     //25
//     -1.0,
//     //26
//     -1.0,
//     //27
//     -1.0,
//     //28
//     -1.0,
//     //29
//     -1.0,
//     //30
//      4.0,
//     //31
//     -1.0,
//     //32
//     -1.0,
//     //33
//     -1.0,
//     //34
//     -1.0,
//     //35
//     -1.0,
//     //36
//     -1.0,
//     //37
//     -1.0,
//     //38
//     -1.0,
//     //39
//     -1.0,
//     //40
//     -1.0,
//     //41
//     -1.0,
//     //42
//     -1.0,
//     //43
//     -1.0,
//     //44
//     -1.0,
//     //45
//      5.0,
//     //46
//      1.0,
//     //47
//     22.0,
//     //48
//     -1.0,
//     //49
//      1.0,
//     //50
//     -1.0,
//     //51
//     -1.0,
//     //52
//     -1.0,
//     //53
//     -1.0,
//     //54
//     -1.0,
//     //55
//     -1.0,
//     //56
//     -1.0,
//     //57
//     -1.0,
//     //58
//     -1.0,
//     //59
//     -1.0,
//     //60
//     -1.0,
//     //61
//     -1.0,
//     //62
//     -1.0,
//     //63
//      1.0,
//     //64
//      1.0,
//     //65
//     -1.0,
//     //66
//     -1.0,
//     //67
//     -1.0,
//     //68
//     -1.0,
//     //69
//     -1.0,
//     //70
//     -1.0,
//     //71
//     -1.0,
//     //72
//     -1.0,
//     //73
//     -1.0,
//     //74
//     -1.0,
//     //75
//     -1.0,
//     //76
//     15.0,
//     //77
//     -1.0,
//     //78
//     -1.0,
//     //79
//     -1.0,
//     //80
//     -1.0
//   };
//   int* flood_next_cell_index_expected_out = new int[80] {
//     //1
//     -1,
//     //2
//     -1,
//     //3
//     -1,
//     //4
//     -1,
//     //5
//     -1,
//     //6
//     -1,
//     //7
//     -1,
//     //8
//     -1,
//     //9
//     -1,
//     //10
//     -1,
//     //11
//     -1,
//     //12
//     -1,
//     //13
//     49,
//     //14
//     -1,
//     //15
//     -1,
//     //16
//     -1,
//     //17
//     -1,
//     //18
//     -1,
//     //19
//     -1,
//     //20
//     -1,
//     //21
//     -1,
//     //22
//     -1,
//     //23
//     -1,
//     //24
//     -1,
//     //25
//     -1,
//     //26
//     -1,
//     //27
//     -1,
//     //28
//     -1,
//     //29
//     -1,
//     //30
//     47,
//     //31
//     -1,
//     //32
//     -1,
//     //33
//     -1,
//     //34
//     -1,
//     //35
//     -1,
//     //36
//     -1,
//     //37
//     -1,
//     //38
//     -1,
//     //39
//     -1,
//     //40
//     -1,
//     //41
//     -1,
//     //42
//     -1,
//     //43
//     -1,
//     //44
//     -1,
//     //45
//     76,
//     //46
//     45,
//     //47
//     52,
//     //48
//     -1,
//     //49
//     30,
//     //50
//     -1,
//     //51
//     -1,
//     //52
//     -1,
//     //53
//     -1,
//     //54
//     -1,
//     //55
//     -1,
//     //56
//     -1,
//     //57
//     -1,
//     //58
//     -1,
//     //59
//     -1,
//     //60
//     -1,
//     //61
//     -1,
//     //62
//     -1,
//     //63
//     46,
//     //64
//     63,
//     //65
//     -1,
//     //66
//     -1,
//     //67
//     -1,
//     //68
//     -1,
//     //69
//     -1,
//     //70
//     -1,
//     //71
//     -1,
//     //72
//     -1,
//     //73
//     -1,
//     //74
//     -1,
//     //75
//     -1,
//     //76
//     49,
//     //77
//     -1,
//     //78
//     -1,
//     //79
//     -1,
//     //80
//     -1
//   };
//   int* flood_force_merge_index_expected_out = new int[80] {
//     //1
//     -1,
//     //2
//     -1,
//     //3
//     -1,
//     //4
//     -1,
//     //5
//     -1,
//     //6
//     -1,
//     //7
//     -1,
//     //8
//     -1,
//     //9
//     -1,
//     //10
//     -1,
//     //11
//     -1,
//     //12
//     -1,
//     //13
//     -1,
//     //14
//     -1,
//     //15
//     -1,
//     //16
//     -1,
//     //17
//     -1,
//     //18
//     -1,
//     //19
//     -1,
//     //20
//     -1,
//     //21
//     -1,
//     //22
//     -1,
//     //23
//     -1,
//     //24
//     -1,
//     //25
//     -1,
//     //26
//     -1,
//     //27
//     -1,
//     //28
//     -1,
//     //29
//     -1,
//     //30
//     64,
//     //31
//     -1,
//     //32
//     -1,
//     //33
//     -1,
//     //34
//     -1,
//     //35
//     -1,
//     //36
//     -1,
//     //37
//     -1,
//     //38
//     -1,
//     //39
//     -1,
//     //40
//     -1,
//     //41
//     -1,
//     //42
//     -1,
//     //43
//     -1,
//     //44
//     -1,
//     //45
//     -1,
//     //46
//     -1,
//     //47
//     -1,
//     //48
//     -1,
//     //49
//     13,
//     //50
//     -1,
//     //51
//     -1,
//     //52
//     -1,
//     //53
//     -1,
//     //54
//     -1,
//     //55
//     -1,
//     //56
//     -1,
//     //57
//     -1,
//     //58
//     -1,
//     //59
//     -1,
//     //60
//     -1,
//     //61
//     -1,
//     //62
//     -1,
//     //63
//     -1,
//     //64
//     -1,
//     //65
//     -1,
//     //66
//     -1,
//     //67
//     -1,
//     //68
//     -1,
//     //69
//     -1,
//     //70
//     -1,
//     //71
//     -1,
//     //72
//     -1,
//     //73
//     -1,
//     //74
//     -1,
//     //75
//     -1,
//     //76
//     -1,
//     //77
//     -1,
//     //78
//     -1,
//     //79
//     -1,
//     //80
//     -1
//   };
//   int* flood_redirect_index_expected_out = new int[80] {
//     //1
//     -1,
//     //2
//     -1,
//     //3
//     -1,
//     //4
//     -1,
//     //5
//     -1,
//     //6
//     -1,
//     //7
//     -1,
//     //8
//     -1,
//     //9
//     -1,
//     //10
//     -1,
//     //11
//     -1,
//     //12
//     -1,
//     //13
//     49,
//     //14
//     -1,
//     //15
//     -1,
//     //16
//     -1,
//     //17
//     -1,
//     //18
//     -1,
//     //19
//     -1,
//     //20
//     -1,
//     //21
//     -1,
//     //22
//     -1,
//     //23
//     -1,
//     //24
//     -1,
//     //25
//     -1,
//     //26
//     -1,
//     //27
//     -1,
//     //28
//     -1,
//     //29
//     -1,
//     //30
//     64,
//     //31
//     -1,
//     //32
//     -1,
//     //33
//     -1,
//     //34
//     -1,
//     //35
//     -1,
//     //36
//     -1,
//     //37
//     -1,
//     //38
//     -1,
//     //39
//     -1,
//     //40
//     -1,
//     //41
//     -1,
//     //42
//     -1,
//     //43
//     -1,
//     //44
//     -1,
//     //45
//     -1,
//     //46
//     -1,
//     //47
//     52,
//     //48
//     -1,
//     //49
//     13,
//     //50
//     -1,
//     //51
//     -1,
//     //52
//     -1,
//     //53
//     -1,
//     //54
//     -1,
//     //55
//     -1,
//     //56
//     -1,
//     //57
//     -1,
//     //58
//     -1,
//     //59
//     -1,
//     //60
//     -1,
//     //61
//     -1,
//     //62
//     -1,
//     //63
//     -1,
//     //64
//     -1,
//     //65
//     -1,
//     //66
//     -1,
//     //67
//     -1,
//     //68
//     -1,
//     //69
//     -1,
//     //70
//     -1,
//     //71
//     -1,
//     //72
//     -1,
//     //73
//     -1,
//     //74
//     -1,
//     //75
//     -1,
//     //76
//     49,
//     //77
//     -1,
//     //78
//     -1,
//     //79
//     -1,
//     //80
//     -1
//   };
//   bool* flood_local_redirect_expected_out = new bool[80] {
//     //1
//     true,
//     //2
//     true,
//     //3
//     true,
//     //4
//     true,
//     //5
//     true,
//     //6
//     true,
//     //7
//     true,
//     //8
//     true,
//     //9
//     true,
//     //10
//     true,
//     //11
//     true,
//     //12
//     true,
//     //13
//     true,
//     //14
//     true,
//     //15
//     true,
//     //16
//     true,
//     //17
//     true,
//     //18
//     true,
//     //19
//     true,
//     //20
//     true,
//     //21
//     true,
//     //22
//     true,
//     //23
//     true,
//     //24
//     true,
//     //25
//     true,
//     //26
//     true,
//     //27
//     true,
//     //28
//     true,
//     //29
//     true,
//     //30
//     true,
//     //31
//     true,
//     //32
//     true,
//     //33
//     true,
//     //34
//     true,
//     //35
//     true,
//     //36
//     true,
//     //37
//     true,
//     //38
//     true,
//     //39
//     true,
//     //40
//     true,
//     //41
//     true,
//     //42
//     true,
//     //43
//     true,
//     //44
//     true,
//     //45
//     true,
//     //46
//     true,
//     //47
//     false,
//     //48
//     true,
//     //49
//     true,
//     //50
//     true,
//     //51
//     true,
//     //52
//     true,
//     //53
//     true,
//     //54
//     true,
//     //55
//     true,
//     //56
//     true,
//     //57
//     true,
//     //58
//     true,
//     //59
//     true,
//     //60
//     true,
//     //61
//     true,
//     //62
//     true,
//     //63
//     true,
//     //64
//     true,
//     //65
//     true,
//     //66
//     true,
//     //67
//     true,
//     //68
//     true,
//     //69
//     true,
//     //70
//     true,
//     //71
//     true,
//     //72
//     true,
//     //73
//     true,
//     //74
//     true,
//     //75
//     true,
//     //76
//     true,
//     //77
//     true,
//     //78
//     true,
//     //79
//     true,
//     //80
//     true
//   };
//   int* merge_points_expected_out = new int[80] {
//     //1
//     0,
//     //2
//     0,
//     //3
//     0,
//     //4
//     0,
//     //5
//     0,
//     //6
//     0,
//     //7
//     0,
//     //8
//     0,
//     //9
//     0,
//     //10
//     0,
//     //11
//     0,
//     //12
//     0,
//     //13
//     10,
//     //14
//     0,
//     //15
//     0,
//     //16
//     0,
//     //17
//     0,
//     //18
//     0,
//     //19
//     0,
//     //20
//     0,
//     //21
//     0,
//     //22
//     0,
//     //23
//     0,
//     //24
//     0,
//     //25
//     0,
//     //26
//     0,
//     //27
//     0,
//     //28
//     0,
//     //29
//     0,
//     //30
//     9,
//     //31
//     0,
//     //32
//     0,
//     //33
//     0,
//     //34
//     0,
//     //35
//     0,
//     //36
//     0,
//     //37
//     0,
//     //38
//     0,
//     //39
//     0,
//     //40
//     0,
//     //41
//     0,
//     //42
//     0,
//     //43
//     0,
//     //44
//     0,
//     //45
//     0,
//     //46
//     0,
//     //47
//     10,
//     //48
//     0,
//     //49
//     9,
//     //50
//     0,
//     //51
//     0,
//     //52
//     0,
//     //53
//     0,
//     //54
//     0,
//     //55
//     0,
//     //56
//     0,
//     //57
//     0,
//     //58
//     0,
//     //59
//     0,
//     //60
//     0,
//     //61
//     0,
//     //62
//     0,
//     //63
//     0,
//     //64
//     0,
//     //65
//     0,
//     //66
//     0,
//     //67
//     0,
//     //68
//     0,
//     //69
//     0,
//     //70
//     0,
//     //71
//     0,
//     //72
//     0,
//     //73
//     0,
//     //74
//     0,
//     //75
//     0,
//     //76
//     10,
//     //77
//     0,
//     //78
//     0,
//     //79
//     0,
//     //80
//     0
//   };
//   double* cell_areas_in = new double[80];
//   std::fill_n(cell_areas_in,80,1.0);
//   int* secondary_neighboring_cell_indices_in = new int[80*9];
//   int* prior_fine_catchments_in = new int[80];
//   icon_single_index_grid_params* grid_params_in =
//       new icon_single_index_grid_params(80,cell_neighbors,
//                                         true,secondary_neighboring_cell_indices_in);
//   grid_params_in->icon_single_index_grid_calculate_secondary_neighbors();
//   delete[] secondary_neighboring_cell_indices_in;
//   secondary_neighboring_cell_indices_in = grid_params_in->get_secondary_neighboring_cell_indices();
//   int* fine_neighboring_cell_indices_in = new int[80*3];
//   std::copy_n(cell_neighbors,80*3,fine_neighboring_cell_indices_in);
//   int* coarse_neighboring_cell_indices_in = new int[80*3];
//   std::copy_n(cell_neighbors,80*3,coarse_neighboring_cell_indices_in);
//   int* fine_secondary_neighboring_cell_indices_in = new int[80*9];
//   std::copy_n(secondary_neighboring_cell_indices_in,80*9,
//               fine_secondary_neighboring_cell_indices_in);
//   int* coarse_secondary_neighboring_cell_indices_in = new int[80*9];
//   std::copy_n(secondary_neighboring_cell_indices_in,80*9,
//               coarse_secondary_neighboring_cell_indices_in);
//   int* mapping_from_fine_to_coarse_grid = new int[80];
//   for (int i = 0; i < 80;i++) mapping_from_fine_to_coarse_grid[i] = i+1;
//   auto alg = catchment_computation_algorithm_icon_single_index();
//   alg.setup_fields(prior_fine_catchments_in,
//                    prior_fine_rdirs_in,
//                    grid_params_in);
//   alg.compute_catchments();
//   int* coarse_catchment_nums_in = new int[80];
//   std::copy_n(prior_fine_catchments_in,80,
//               coarse_catchment_nums_in);
//   double* connection_volume_thresholds_in = new double[80];
//   double* flood_volume_thresholds_in = new double[80];
//   int* flood_next_cell_index_in = new int[80];
//   int* connect_next_cell_index_in = new int[80];
//   int* flood_force_merge_index_in = new int[80];
//   int* connect_force_merge_index_in = new int[80];
//   int* flood_redirect_index_in = new int[80];
//   int* connect_redirect_index_in = new int[80];
//   int* additional_flood_redirect_index_in = new int[80];
//   int* additional_connect_redirect_index_in = new int[80];
//   bool* flood_local_redirect_in = new bool[80];
//   bool* connect_local_redirect_in = new bool[80];
//   bool* additional_flood_local_redirect_in = new bool[80];
//   bool* additional_connect_local_redirect_in = new bool[80];
//   int* merge_points_out_int = new int[80];
//   int* basin_catchment_numbers_in = new int[80];
//   double* connection_volume_thresholds_expected_out = new double[80];
//   std::fill_n(connection_volume_thresholds_expected_out,80,-1.0);
//   int* connect_next_cell_index_expected_out = new int[80];
//   std::fill_n(connect_next_cell_index_expected_out,80,-1);
//   int* connect_force_merge_index_expected_out = new int[80];
//   std::fill_n(connect_force_merge_index_expected_out,80,-1);
//   int* connect_redirect_index_expected_out = new int[80];
//   std::fill_n(connect_redirect_index_expected_out,80,-1);
//   int* additional_flood_redirect_index_expected_out = new int[80];
//   std::fill_n(additional_flood_redirect_index_expected_out,80,-1);
//   int* additional_connect_redirect_index_expected_out = new int[80];
//   std::fill_n(additional_connect_redirect_index_expected_out,80,-1);
//   bool* connect_local_redirect_expected_out = new bool[80];
//   std::fill_n(connect_local_redirect_expected_out,80,true);
//   bool* additional_flood_local_redirect_expected_out = new bool[80];
//   std::fill_n(additional_flood_local_redirect_expected_out,80,true);
//   bool* additional_connect_local_redirect_expected_out = new bool[80];
//   std::fill_n(additional_connect_local_redirect_expected_out,80,true);
//   icon_single_index_evaluate_basins(minima_in,
//                                     raw_orography_in,
//                                     corrected_orography_in,
//                                     cell_areas_in,
//                                     connection_volume_thresholds_in,
//                                     flood_volume_thresholds_in,
//                                     prior_fine_rdirs_in,
//                                     prior_coarse_rdirs_in,
//                                     prior_fine_catchments_in,
//                                     coarse_catchment_nums_in,
//                                     flood_next_cell_index_in,
//                                     connect_next_cell_index_in,
//                                     ncells,ncells,
//                                     fine_neighboring_cell_indices_in,
//                                     coarse_neighboring_cell_indices_in,
//                                     fine_secondary_neighboring_cell_indices_in,
//                                     coarse_secondary_neighboring_cell_indices_in,
//                                     mapping_from_fine_to_coarse_grid,
//                                     basin_catchment_numbers_in);
//   EXPECT_TRUE(field<int>(basin_catchment_numbers_in,grid_params_in) ==
//               field<int>(basin_numbers_expected_out,grid_params_in));
//   EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in) ==
//               field<double>(flood_volume_thresholds_expected_out,grid_params_in));
//   EXPECT_TRUE(field<double>(connection_volume_thresholds_in,grid_params_in) ==
//               field<double>(connection_volume_thresholds_expected_out,grid_params_in));
//   EXPECT_TRUE(field<int>(flood_next_cell_index_in,grid_params_in) ==
//               field<int>(flood_next_cell_index_expected_out,grid_params_in));
//   EXPECT_TRUE(field<int>(connect_next_cell_index_in,grid_params_in) ==
//               field<int>(connect_next_cell_index_expected_out,grid_params_in));
//   EXPECT_TRUE(false);
//   delete grid_params_in;
// }

TEST_F(BasinEvaluationTest, TestEvaluateBasinsSix) {
  auto grid_params_in = new latlon_grid_params(20,20,false);
  auto coarse_grid_params_in = new latlon_grid_params(4,4,false);
  int* coarse_catchment_nums_in = new int[4*4] {4,1,2,3,
                                                4,4,6,5,
                                                4,4,6,5,
                                                7,7,7,8};
  double* corrected_orography_in = new double[20*20]
  {9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0,
   9.0,9.0,9.0,3.0,3.0, 3.0,9.0,1.0,1.0,9.0, 1.0,1.0,1.0,1.0,9.0, 9.0,2.0,2.0,2.0,9.0,
   9.0,1.0,9.0,3.0,3.0, 3.0,6.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,7.0, 7.0,2.0,2.0,2.0,9.0,
   9.0,1.0,9.0,3.0,3.0, 3.0,9.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,9.0, 9.0,2.0,2.0,2.0,9.0,
   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,1.0,1.0,1.0,9.0, 9.0,2.0,2.0,2.0,9.0,

   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,1.0,1.0,1.0,9.0, 9.0,9.0,6.0,9.0,9.0,
   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,5.0,9.0,9.0, 9.0,3.0,3.0,3.0,9.0,
   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0,
   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0,
   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0,

   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0,
   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0,
   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0,
   9.0,7.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,8.0,9.0,
   9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 9.0,9.0,9.0,9.0,7.0, 7.0,7.0,7.0,7.0,7.0,

   9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0,
   9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0,
   9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,8.0,7.0, 0.0,0.0,0.0,0.0,0.0,
   9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0,
   9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0
};

  double* raw_orography_in = new double[20*20]
  {9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0,
   9.0,9.0,9.0,3.0,3.0, 3.0,9.0,1.0,1.0,9.0, 1.0,1.0,1.0,1.0,9.0, 9.0,2.0,2.0,2.0,9.0,
   9.0,1.0,9.0,3.0,3.0, 3.0,6.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,7.0, 7.0,2.0,2.0,2.0,9.0,
   9.0,1.0,9.0,3.0,3.0, 3.0,9.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0,9.0, 9.0,2.0,2.0,2.0,9.0,
   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,1.0,1.0,1.0,9.0, 9.0,2.0,2.0,2.0,9.0,

   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,1.0,1.0,1.0,9.0, 9.0,9.0,6.0,9.0,9.0,
   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,5.0,9.0,9.0, 9.0,3.0,3.0,3.0,9.0,
   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0,
   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0,
   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0,

   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0,
   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0,
   9.0,1.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,4.0,4.0,4.0,9.0, 9.0,3.0,3.0,3.0,9.0,
   9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,8.0,9.0,
   9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 9.0,9.0,9.0,9.0,7.0, 7.0,7.0,7.0,7.0,7.0,

   9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0,
   9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0,
   9.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,6.0,6.0, 6.0,6.0,6.0,8.0,7.0, 0.0,0.0,0.0,0.0,0.0,
   9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0,
   9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,9.0, 9.0,9.0,9.0,9.0,7.0, 0.0,0.0,0.0,0.0,0.0 };

  bool* minima_in = new bool[20*20]
  {  false, false, false, false, false,  false, false, false, false, false,
        false, false, false, false, false,
          false, false, false, false, false,
     false, false, false,  true, false,  false, false,  true, false, false,
      false, false, false, false, false,
        false,  true, false, false, false,
     false,  true, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false,  true, false, false, false,
        false, true, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false,  true, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false,
     false, false, false, false, false,  false, false, false, false, false,
      false, false, false, false, false,
        false, false, false, false, false };
  double* prior_fine_rdirs_in = new double[20*20] {
       1., 4., 3., 2., 1., 1., 3., 2., 1., 1., 2., 1., 1., 1., 1., 3.,
       2., 1., 1., 1.,
       3., 2., 1., 5., 4., 4., 6., 5., 4., 4., 1., 4., 4., 4., 4., 6.,
       5., 4., 4., 4.,
       6., 5., 4., 8., 7., 7., 9., 8., 7., 7., 4., 7., 7., 7., 7., 9.,
       8., 7., 7., 7.,
       9., 8., 7., 8., 7., 7., 9., 8., 7., 7., 7., 7., 7., 7., 7., 9.,
       8., 7., 7., 7.,
       9., 8., 7., 8., 7., 7., 9., 8., 7., 7., 7., 7., 7., 7., 7., 9.,
       8., 7., 7., 7.,
       9., 8., 7., 2., 1., 4., 4., 4., 4., 4., 9., 8., 7., 7., 7., 9.,
       8., 7., 7., 7.,
       9., 8., 7., 1., 4., 7., 7., 7., 7., 7., 9., 8., 7., 7., 7., 6.,
       5., 4., 4., 4.,
       9., 8., 7., 4., 7., 7., 7., 7., 7., 7., 6., 5., 4., 4., 4., 9.,
       8., 7., 7., 7.,
       9., 8., 7., 7., 7., 7., 7., 7., 7., 7., 9., 8., 7., 7., 7., 9.,
       8., 7., 7., 7.,
       9., 8., 7., 8., 7., 7., 7., 7., 7., 7., 9., 8., 7., 7., 7., 9.,
       8., 7., 7., 7.,
       9., 8., 7., 8., 7., 7., 7., 7., 7., 7., 9., 8., 7., 7., 7., 9.,
       8., 7., 7., 7.,
       9., 8., 7., 8., 7., 7., 7., 7., 7., 7., 9., 8., 7., 7., 7., 9.,
       8., 7., 7., 7.,
       9., 8., 7., 8., 7., 7., 7., 7., 7., 7., 9., 8., 7., 7., 7., 9.,
       8., 7., 7., 7.,
       9., 8., 7., 1., 1., 1., 1., 1., 1., 1., 9., 8., 7., 7., 7., 9.,
       8., 7., 7., 7.,
       1., 5., 4., 4., 4., 4., 4., 4., 4., 4., 4., 1., 1., 1., 3., 2.,
       1., 1., 1., 1.,
       4., 8., 7., 7., 7., 7., 7., 7., 7., 7., 7., 4., 4., 4., 6., 3.,
       2., 1., 1., 1.,
       7., 8., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 9., 6.,
       3., 2., 1., 1.,
       7., 8., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 9., 9.,
       6., 3., 2., 1.,
       1., 8., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 9., 9.,
       9., 6., 3., 2.,
       4., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 9., 9., 9., 9.,
       9., 9., 6., 0.
  };
  double* prior_coarse_rdirs_in = new double[4*4]{
    5,5,5,5,
    2,2,5,5,
    5,4,8,8,
    8,7,4,0 };
  int* prior_fine_catchments_in = new int[20*20] {
   3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
   4, 4, 4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
   4, 4, 4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
   4, 4, 4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
   4, 4, 4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5,
   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5,
   4, 4, 4, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5,
   8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8,
   8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8,
   8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8,
   8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8,
   8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8,
   8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8 };
  double* cell_areas_in = new double[20*20]{
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
//
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
//
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
//
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,
    1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1
  };
  double* connection_volume_thresholds_in = new double[20*20];
  std::fill_n(connection_volume_thresholds_in,20*20,0.0);
  double* flood_volume_thresholds_in = new double[20*20];
  std::fill_n(flood_volume_thresholds_in,20*20,0.0);
  double* connection_heights_in = new double[20*20];
  std::fill_n(connection_heights_in,20*20,0.0);
  double* flood_heights_in = new double[20*20];
  std::fill_n(flood_heights_in,20*20,0.0);
  int* flood_next_cell_lat_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lat_index_in,20*20,-1);
  int* flood_next_cell_lon_index_in = new int[20*20];
  std::fill_n(flood_next_cell_lon_index_in,20*20,-1);
  int* connect_next_cell_lat_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lat_index_in,20*20,-1);
  int* connect_next_cell_lon_index_in = new int[20*20];
  std::fill_n(connect_next_cell_lon_index_in,20*20,-1);
  double* flood_volume_thresholds_expected_out = new double[20*20] {
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0,  0.0,  0.0,  27.0, -1.0,  0.0,  0.0, -1.0,  0.0,  0.0,  0.0, 104.0, -1.0, -1.0,  0.0,   0.0, 0.0, -1.0,
         -1.0,  0.0, -1.0,  0.0,  0.0,  0.0, 204.0, 0.0, 0.0, 0.0,     0.0,  0.0, 0.0,    0.0, 188.0, 97.0,  0.0, 0.0, 0.0, -1.0,
         -1.0,  0.0, -1.0,  0.0,  0.0,  0.0,  -1.0, 0.0, 0.0, 0.0,     0.0,  0.0, 0.0,    0.0,  -1.0,  -1.0,  0.0, 0.0, 0.0, -1.0,
         -1.0,  0.0, -1.0, -1.0, -1.0, -1.0,  -1.0,-1.0,-1.0,-1.0,    -1.0,  0.0, 0.0,    0.0,  -1.0,  -1.0, 48.0, 0.0, 0.0, -1.0,
         -1.0, 0.0, -1.0,  -1.0, -1.0, -1.0,  -1.0,-1.0,-1.0,-1.0,    -1.0,  0.0, 0.0,    0.0,  -1.0,  -1.0, -1.0,97.0,-1.0, -1.0,
         -1.0, 0.0, -1.0,  -1.0, -1.0, -1.0,  -1.0,-1.0,-1.0,-1.0,    -1.0, -1.0,149.0,  -1.0,  -1.0,  -1.0,  0.0, 0.0, 0.0, -1.0,
         -1.0, 0.0, -1.0,  -1.0, -1.0, -1.0,  -1.0,-1.0,-1.0,-1.0,    -1.0,  0.0,  0.0,   0.0,  -1.0, -1.0,   0.0, 0.0, 0.0, -1.0,
         -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,   -1.0,  0.0,  0.0,  0.0, -1.0, -1.0,  0.0,  0.0,  0.0, -1.0,
         -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,   -1.0,  0.0,  0.0,  0.0, -1.0, -1.0,  0.0,  0.0,  0.0, -1.0,
         -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,   -1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0,
         -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,   -1.0, 0.0,  0.0,  0.0, -1.0, -1.0,  0.0,  0.0,  0.0, -1.0,
         -1.0, 66.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, 18.0,  0.0, 0.0, -1.0,   -1.0, 63.0, 0.0, 0.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0,          -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0,           0.0, 0.0, 45.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,            0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,            0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
 };
  double* connection_volume_thresholds_expected_out = new double[20*20]{
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, 101.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0  };
  int* flood_next_cell_lat_index_expected_out = new int[20*20]{
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, 2, 3, 3, -1, 2, 3, -1, 4, 5, 5, 6, -1, -1, 2, 3, 4, -1,
         -1, 3, -1, 1, 2, 1, 3, 1, 2, 3, 1, 1, 1, 1, 14, 2, 1, 2, 1, -1,
         -1, 4, -1, 2, 3, 3, -1, 2, 3, 3, 2, 2, 2, 2, -1, -1, 2, 3, 3, -1,
         -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 3, 3, -1, -1, 6, 4, 4, -1,
         -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 4, -1, -1, -1, 2, -1, -1,
         -1, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, 9, 6, 10, -1,
         -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, -1, -1, 8, 6, 6, -1,
         -1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, 8, 7, -1, -1, 7, 8, 7, -1,
         -1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 9, -1, -1, 8, 9, 9, -1,
         -1, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 10, 10, -1, -1, 11, 10, 10, -1,
         -1, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 11, 11, -1, -1, 12, 11, 11, -1,
         -1, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5, 12, 12, -1, -1, 5, 12, 12, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, 15, 16, 17, 17, 17, 17, 17, 17, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, 14, 15, 14, 17, 14, 14, 14, 14, 14, 17, 17, 13, -1, -1, -1, -1, -1, -1, -1,
         -1, 15, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, -1, -1, -1, -1, -1, -1, -1,
         -1, 14, 16, 17, 17, 16, 16, 16, 16, 16, 16, 16, 16, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1  };
  int* flood_next_cell_lon_index_expected_out = new int[20*20]{
          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, 4, 5, 7, -1, 8, 9, -1, 11, 12, 13, 12, -1, -1, 17, 18, 18, -1,
         -1, 1, -1, 4, 3, 5, 16, 8, 7, 10, 10, 11, 12, 13, 19, 14, 17, 16, 18, -1,
         -1, 1, -1, 5, 3, 4, -1, 9, 7, 8, 10, 11, 12, 13, -1, -1, 18, 16, 17, -1,
         -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 12, 13, -1, -1, 18, 16, 17, -1,
         -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 11, 13, -1, -1, -1, 15, -1, -1,
         -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, -1, -1, -1, 18, 16, 18, -1,
         -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 13, -1, -1, 17, 17, 18, -1,
         -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 11, 13, -1, -1, 17, 16, 18, -1,
         -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 11, 12, -1, -1, 18, 16, 17, -1,
         -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 11, 12, -1, -1, 18, 16, 17, -1,
         -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 11, 12, -1, -1, 18, 16, 17, -1,
         -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 11, 12, -1, -1, 17, 16, 17, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, 2, 1, 3, 1, 5, 6, 7, 8, 9, 11, 12, 1, -1, -1, -1, -1, -1, -1, -1,
         -1, 3, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, -1, -1, -1, -1, -1,
         -1, 4, 4, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
 };
  int* connect_next_cell_lat_index_expected_out = new int[20*20]{
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
  };
  int* connect_next_cell_lon_index_expected_out = new int[20*20]{
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  double* connection_heights_expected_out = new double[20*20]{
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  double* flood_heights_expected_out = new double[20*20]{
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 3, 3, 6, 0, 1, 1, 0, 1, 1, 1, 5, 0, 0, 2, 2, 2, 0,
     0, 1, 0, 3, 3, 3, 7, 1, 1, 1, 1, 1, 1, 1, 8, 7, 2, 2, 2, 0,
     0, 1, 0, 3, 3, 3, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 0,
     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 6, 2, 2, 0,
     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 7, 0, 0,
     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 3, 3, 3, 0,
     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 3, 3, 3, 0,
     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 3, 3, 3, 0,
     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 3, 3, 3, 0,
     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 3, 3, 3, 0,
     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 3, 3, 3, 0,
     0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 4, 0, 0, 6, 3, 3, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0, 0, 0, 0, 0, 0, 0,
     0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0,
     0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  int flood_index = 0;
  int connect_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(14,1),
                                                new latlon_coords(14,1),
                                                true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(12,1);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(2,1),
                                                new latlon_coords(2,0),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(15,12);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,7),
                                                new latlon_coords(1,7),
                                                true);

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,5);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,7),
                                                new latlon_coords(0,1),
                                                false);

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(12,11);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(7,11),
                                                new latlon_coords(7,11),
                                                true);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,13);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,3),
                                                new latlon_coords(1,1),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,12);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(7,16),
                                                new latlon_coords(1,3),
                                                false);

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(2,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(7,16),
                                                new latlon_coords(7,16),
                                                true);

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(4,16);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,16),
                                                new latlon_coords(0,3),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(12,16);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  primary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,7),
                                                new latlon_coords(0,1),
                                                false);
  secondary_merge = nullptr;

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(5,17);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(14,19),
                                                new latlon_coords(3,3),
                                                false);

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(2,14);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(18,14),
                                                new latlon_coords(3,3),
                                                false);

  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  connect_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(13,1);
  (*connect_merge_and_redirect_indices_index)(working_coords) = connect_index;
  connect_index++;
  delete working_coords;

  merges_and_redirects merges_and_redirects_expected_out =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  auto basin_eval = latlon_basin_evaluation_algorithm();
  basin_eval.setup_fields(minima_in,
                          raw_orography_in,
                          corrected_orography_in,
                          cell_areas_in,
                          connection_volume_thresholds_in,
                          flood_volume_thresholds_in,
                          connection_heights_in,
                          flood_heights_in,
                          prior_fine_rdirs_in,
                          prior_coarse_rdirs_in,
                          prior_fine_catchments_in,
                          coarse_catchment_nums_in,
                          flood_next_cell_lat_index_in,
                          flood_next_cell_lon_index_in,
                          connect_next_cell_lat_index_in,
                          connect_next_cell_lon_index_in,
                          grid_params_in,
                          coarse_grid_params_in);
  bool* landsea_in = new bool[20*20] {
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
//
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
//
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
//
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,false,
    false,false,false,false,false, false,false,false,false,false, false,false,false,false,false, false,false,false,false,true };
  bool* true_sinks_in = new bool[20*20];
  fill_n(true_sinks_in,20*20,false);
  int* next_cell_lat_index_in = new int[20*20];
  int* next_cell_lon_index_in = new int[20*20];
  short* sinkless_rdirs_out = new short[20*20];
  int* catchment_nums_in = new int[20*20];
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
  EXPECT_TRUE(field<double>(flood_volume_thresholds_in,grid_params_in).almost_equal(field<double>(flood_volume_thresholds_expected_out,grid_params_in),0.0001));
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
  merges_and_redirects* merges_and_redirects_out =
    basin_eval.get_basin_merges_and_redirects();
  EXPECT_TRUE(merges_and_redirects_expected_out ==
              *merges_and_redirects_out);
  EXPECT_TRUE(field<double>(connection_heights_expected_out,grid_params_in)
              == field<double>(connection_heights_in,grid_params_in));
  EXPECT_TRUE(field<double>(flood_heights_expected_out,grid_params_in)
              == field<double>(flood_heights_in,grid_params_in));
  delete grid_params_in; delete coarse_grid_params_in; delete alg4;
  delete[] coarse_catchment_nums_in; delete[] corrected_orography_in;
  delete[] raw_orography_in; delete[] minima_in;
  delete[] flood_heights_in; delete[] connection_heights_in;
  delete[] prior_coarse_rdirs_in;
  delete[] prior_fine_rdirs_in; delete[] prior_fine_catchments_in;
  delete[] connection_volume_thresholds_in; delete[] flood_volume_thresholds_in;
  delete[] flood_next_cell_lat_index_in; delete[] flood_next_cell_lon_index_in;
  delete[] connect_next_cell_lat_index_in; delete[] connect_next_cell_lon_index_in;
  delete[] flood_volume_thresholds_expected_out;
  delete[] connection_volume_thresholds_expected_out;
  delete[] flood_next_cell_lat_index_expected_out;
  delete[] flood_next_cell_lon_index_expected_out;
  delete[] connect_next_cell_lat_index_expected_out;
  delete[] connect_next_cell_lon_index_expected_out;
  delete[] landsea_in;
  delete[] true_sinks_in;
  delete[] next_cell_lat_index_in;
  delete[] next_cell_lon_index_in;
  delete[] sinkless_rdirs_out;
  delete[] catchment_nums_in;
  delete[] cell_areas_in;
}

TEST_F(BasinEvaluationTest, TestConvertingMergesToArray) {
  auto grid_params_in = new latlon_grid_params(20,20,false);
  int flood_index = 0;
  int connect_index = 0;
  int* connect_merge_indices_as_array = new int[6] {1,0,61,62,63,64};
  int* flood_merge_indices_as_array = new int[240]
    { 1,1,1,2,3,4, -1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,
     -1,-1,-1,-1,-1,-1, 1,0,5,6,7,8, -1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,
      1,1,9,10,11,12, 1,0,13,14,15,16, -1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,
     -1,-1,-1,-1,-1,-1, 1,1,17,18,19,20, 1,0,21,22,23,24, -1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,
     -1,-1,-1,-1,-1,-1, 1,1,25,26,27,28, 1,0,29,30,31,32, 1,0,33,34,35,36,    1,0,37,38,39,40,
      1,1,41,42,43,44, -1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,
     -1,-1,-1,-1,-1,-1, 1,0,45,46,47,48, -1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,
      1,1,49,50,51,52, 1,0,53,54,55,56, 1,0,57,58,59,60, -1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1 };
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,2),
                                                               new latlon_coords(3,4),
                                                               true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,2);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge = nullptr;
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(5,6),
                                                         new latlon_coords(7,8),
                                                         false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(2,3);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(9,10),
                                                               new latlon_coords(11,12),
                                                               true);
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(13,14),
                                                         new latlon_coords(15,16),
                                                         false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(3,4);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge = nullptr;
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(17,18),
                                                         new latlon_coords(19,20),
                                                         true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(21,22),
                                                         new latlon_coords(23,24),
                                                         false);
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(4,5);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge = nullptr;
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(25,26),
                                                         new latlon_coords(27,28),
                                                         true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(29,30),
                                                         new latlon_coords(31,32),
                                                         false);
  primary_merges->push_back(primary_merge);
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(33,34),
                                                         new latlon_coords(35,36),
                                                         false);
  primary_merges->push_back(primary_merge);
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(37,38),
                                                         new latlon_coords(39,40),
                                                         false);
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(5,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(41,42),
                                                               new latlon_coords(43,44),
                                                               true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,7);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

    secondary_merge = nullptr;
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(45,46),
                                                         new latlon_coords(47,48),
                                                         false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(7,8);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(49,50),
                                                               new latlon_coords(51,52),
                                                               true);
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(53,54),
                                                         new latlon_coords(55,56),
                                                         false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(57,58),
                                                         new latlon_coords(59,60),
                                                         false);
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,9);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(61,62),
                                                               new latlon_coords(63,64),
                                                               false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  connect_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(2,1);
  (*connect_merge_and_redirect_indices_index)(working_coords) = connect_index;
  connect_index++;
  delete working_coords;

  merges_and_redirects working_merges_and_redirects =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  auto merges_and_redirects_as_array_pair_one = working_merges_and_redirects.get_merges_and_redirects_as_array(true); 
  auto merges_and_redirects_as_array_one = merges_and_redirects_as_array_pair_one->second; 
  for (auto i =0; i < 240; i++){
    EXPECT_EQ((merges_and_redirects_as_array_one)[i],
               flood_merge_indices_as_array[i]);
  }
  auto merges_and_redirects_as_array_pair_two = working_merges_and_redirects.get_merges_and_redirects_as_array(false); 
  auto merges_and_redirects_as_array_two = merges_and_redirects_as_array_pair_two->second;
  for (auto i =0; i < 6; i++){
    EXPECT_EQ((merges_and_redirects_as_array_two)[i],
               connect_merge_indices_as_array[i]);
  }
  delete grid_params_in;
  delete[] merges_and_redirects_as_array_one; 
  delete merges_and_redirects_as_array_pair_one->first;
  delete merges_and_redirects_as_array_pair_one;
  delete[] merges_and_redirects_as_array_two; 
  delete merges_and_redirects_as_array_pair_two->first;
  delete merges_and_redirects_as_array_pair_two;
  delete[] connect_merge_indices_as_array;
  delete[] flood_merge_indices_as_array;
}

} //namespace
