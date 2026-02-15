/*
 * water_redistribution_algorithm.cpp
 *
 *  Created on: May 30, 2019
 *      Author: thomasriddick
 */

#include "algorithms/water_redistribution_algorithm.hpp"

using namespace std;

water_redistribution_algorithm::~water_redistribution_algorithm(){
  delete _grid;
  delete _coarse_grid;
  delete lake_numbers;
  delete lake_centers;
  delete water_to_redistribute;
  delete water_redistributed_to_lakes;
  delete water_redistributed_to_rivers;
}

void water_redistribution_algorithm::setup_fields(int* lake_numbers_in,
                                                  bool* lake_centers_in,
                                                  double* water_to_redistribute_in,
                                                  double* water_redistributed_to_lakes_in,
                                                  double* water_redistributed_to_rivers_in,
                                                  grid_params* grid_params_in,
                                                  grid_params* coarse_grid_params_in){
  _grid_params = grid_params_in;
  _grid = grid_factory(grid_params_in);
  _coarse_grid = grid_factory(coarse_grid_params_in);
  lake_numbers = new field<int>(lake_numbers_in,grid_params_in);
  lake_centers = new field<bool>(lake_centers_in,grid_params_in);
  water_to_redistribute = new field<double>(water_to_redistribute_in,
                                            grid_params_in);
  water_redistributed_to_lakes = new field<double>(water_redistributed_to_lakes_in,
                                                  grid_params_in);
  water_redistributed_to_rivers = new field<double>(water_redistributed_to_rivers_in,
                                                   coarse_grid_params_in);
}

void water_redistribution_algorithm::run_water_redistribution(){
  cout << "Entering Water Redistribution C++ Code" << endl;
  prepare_lake_center_list();
  redistribute_water();
  for (int i = 0; i < (int)lake_center_list.size(); i++){
    if(lake_center_list[i]) delete lake_center_list[i];
  };
}

void water_redistribution_algorithm::prepare_lake_center_list(){
  int number_of_lakes = lake_numbers->get_max_element()+1;
  lake_center_list.assign(number_of_lakes,nullptr);
  _grid->for_all([&](coords* coords_in){
    int lake_number = (*lake_numbers)(coords_in);
    if ( lake_number != null_lake_number && (*lake_centers)(coords_in)) {
      lake_center_list[(*lake_numbers)(coords_in)] = coords_in;
    } else delete coords_in;
  });
}

void water_redistribution_algorithm::redistribute_water(){
  _grid->for_all([&](coords* coords_in){
    double water_to_redistribute_in_this_cell =
      (*water_to_redistribute)(coords_in);
    if(water_to_redistribute_in_this_cell> 0.0){
      int lake_number = (*lake_numbers)(coords_in);
      if(lake_number != null_lake_number){
        coords* new_lake_center_coords = lake_center_list[lake_number];
        (*water_redistributed_to_lakes)(new_lake_center_coords) +=
          water_to_redistribute_in_this_cell;
      } else {
        coords* new_coarse_center_coords =
          _coarse_grid->convert_fine_coords(coords_in,_grid_params);
        (*water_redistributed_to_rivers)(new_coarse_center_coords) +=
          water_to_redistribute_in_this_cell;
        delete new_coarse_center_coords;
      }
    }
    delete coords_in;
  });
}
