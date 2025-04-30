/*
 * water_redistribution_algorithm.hpp
 *
 *  Created on: May 30, 2019
 *      Author: thomasriddick
 */

#ifndef INCLUDE_WATER_REDISTRIBUTION_ALGORITHM_HPP_
#define INCLUDE_WATER_REDISTRIBUTION_ALGORITHM_HPP_

#include "base/grid.hpp"
#include "base/coords.hpp"
#include "base/field.hpp"

using namespace std;

class water_redistribution_algorithm {
public:
  ~water_redistribution_algorithm();
  void run_water_redistribution();
  ///Setup fields
  void setup_fields(int* lake_numbers_in,
                    bool* lake_centers_in,
                    double* water_to_redistribute_in,
                    double* water_redistributed_to_lakes_in,
                    double* water_redistributed_to_rivers_in,
                    grid_params* grid_params_in,
                    grid_params* coarse_grid_params_in);
protected:
  void prepare_lake_center_list();
  void redistribute_water();
  grid* _grid = nullptr;
  grid* _coarse_grid = nullptr;
  grid_params* _grid_params = nullptr;
  field<int>* lake_numbers = nullptr;
  field<bool>* lake_centers = nullptr;
  field<double>* water_to_redistribute = nullptr;
  field<double>* water_redistributed_to_lakes = nullptr;
  field<double>* water_redistributed_to_rivers = nullptr;
  vector<coords*> lake_center_list;
  int null_lake_number = -1;
};

#endif /* INCLUDE_WATER_REDISTRIBUTION_ALGORITHM_HPP_ */
