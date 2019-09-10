/*
 * burn_post_processing_algorithm.hpp
 *
 *  Created on: Aug 23, 2019
 *      Author: thomasriddick
 */

#ifndef INCLUDE_BURN_POST_PROCESSING_ALGORITHM_HPP_
#define INCLUDE_BURN_POST_PROCESSING_ALGORITHM_HPP_

#include <stack>
#include "coords.hpp"
#include "field.hpp"
#include "enums.hpp"

using namespace std;

class basin {
public:
  void initialise_basin(field<int>* basin_numbers_in,
                        field<int>* coarse_catchment_numbers_in,
                        field<bool>* completed_cells_in,
                        field<bool>* redirect_targets_in,
                        field<bool>* minima_in,
                        field<bool>* use_flood_height_only_in,
                        field<merge_types>* merge_points_in,
                        bool* processed_basins_in,
                        bool* coarse_catchments_in,
                        grid* _coarse_grid_in,
                        grid* _fine_grid_in,
                        grid_params* coarse_grid_params_in,
                        grid_params* fine_grid_params_in);
  void check_basin_for_loops();
  int build_and_simplify_basin(int basin_count);
  void set_minimum_coords(coords* coords_in)
    { minimum_coords = coords_in; }
  void set_basin_number(int basin_number_in)
    { basin_number = basin_number_in; }
  int get_basin_number() { return basin_number; }
  bool get_local_redirect() { return local_redirect; }
  coords* get_basin_redirect_coords() { return redirect_coords; }
  void assign_new_primary_basin(basin* primary_basin_in)
    { primary_basin = primary_basin_in; }
  basin* get_primary_basin() { return primary_basin; }
  basin* get_basin_at_coords(coords* coords_in);
  height_types get_basin_redirect_height_type()
    { return redirect_height_type; }
  int get_minimum_coarse_catchment_number()
    { return minimum_coarse_catchment_number; }
  void find_minimum_coarse_catchment_number();
  basic_merge_types get_cell_merge_type(coords* coords_in,height_types height_type_in);
  vector<basin*> get_basins_within_coarse_cell(coords* coarse_coords_in);
  virtual bool is_minimum(coords* coords_in) = 0;
  virtual bool is_outflow(coords* coords_in) = 0;
  virtual coords* find_target_minimum_coords(coords* coords_in) = 0;
  virtual coords* find_next_fine_cell_downstream(coords* coords_in) = 0;
  virtual coords* find_next_coarse_cell_downstream(coords* coords_in) = 0;
  virtual coords* get_next_cell(coords* coords_in,
                        height_types height_type_in) = 0;
  virtual coords* find_redirect_coords(coords* coords_in,
                               height_types height_type_in) = 0;
  virtual coords* get_primary_merge_target(coords* coords_in,
                                 height_types height_type_in) = 0;
  virtual void set_new_redirect(coords* coords_in,
                                height_types height_type_in) = 0;
  virtual void remove_current_cell_connection_from_ladder(coords* last_relevant_cell_in,
                                                          coords* current_cell_in,
                                                          coords* next_cell_in,
                                                          height_types last_relevant_cell_height_type_in) = 0;
  virtual basin* create_basin() = 0;
protected:
  vector<basin*> basins;
  field<int>* basin_numbers = nullptr;
  field<int>* coarse_catchment_numbers = nullptr;
  field<bool>* completed_cells = nullptr;
  field<bool>* redirect_targets = nullptr;
  field<bool>* minima = nullptr;
  field<bool>* use_flood_height_only = nullptr;
  field<merge_types>* merge_points = nullptr;
  coords* minimum_coords = nullptr;
  coords* redirect_coords = nullptr;
  bool* processed_basins = nullptr;
  bool* coarse_catchments = nullptr;
  grid* _coarse_grid = nullptr;
  grid* _fine_grid = nullptr;
  grid_params* coarse_grid_params = nullptr;
  grid_params* fine_grid_params = nullptr;
  basin* primary_basin = nullptr;
  height_types redirect_height_type;
  bool local_redirect;
  int basin_number;
  int minimum_coarse_catchment_number;
};

class latlon_basin : public basin {
public:
  void initialise_basin(field<int>* basin_numbers_in,
                        field<int>* coarse_catchment_numbers_in,
                        field<bool>* completed_cells_in,
                        field<bool>* redirect_targets_in,
                        field<bool>* minima_in,
                        field<bool>* use_flood_height_only_in,
                        field<merge_types>* merge_points_in,
                        bool* processed_basins_in,
                        bool* coarse_catchments_in,
                        grid* _coarse_grid_in,
                        grid* _fine_grid_in,
                        grid_params* coarse_grid_params_in,
                        grid_params* fine_grid_params_in,
                        field<int>* fine_rdirs_in,
                        field<int>* coarse_rdirs_in,
                        field<int>* connect_redirect_lat_index_in,
                        field<int>* connect_redirect_lon_index_in,
                        field<int>* flood_redirect_lat_index_in,
                        field<int>* flood_redirect_lon_index_in,
                        field<int>* connect_next_cell_lat_index_in,
                        field<int>* connect_next_cell_lon_index_in,
                        field<int>* flood_next_cell_lat_index_in,
                        field<int>* flood_next_cell_lon_index_in,
                        field<int>* connect_force_merge_lat_index_in,
                        field<int>* connect_force_merge_lon_index_in,
                        field<int>* flood_force_merge_lat_index_in,
                        field<int>* flood_force_merge_lon_index_in,
                        int* basin_minimum_lats_in,
                        int* basin_minimum_lons_in);
  coords* find_target_minimum_coords(coords* coords_in);
  coords* find_next_fine_cell_downstream(coords* coords_in);
  coords* find_next_coarse_cell_downstream(coords* coords_in);
  coords* find_redirect_coords(coords* coords_in,
                               height_types height_type_in);
  coords* get_next_cell(coords* coords_in,
                        height_types height_type_in);
  coords* get_primary_merge_target(coords* coords_in,
                                   height_types height_type_in);
  void set_new_redirect(coords* coords_in,
                        height_types height_type_in);
  void remove_current_cell_connection_from_ladder(coords* last_relevant_cell_in,
                                                  coords* current_cell_in,
                                                  coords* next_cell_in,
                                                  height_types last_relevant_cell_height_type_in);
  bool is_minimum(coords* coords_in)
    { return (((*coarse_rdirs)(coords_in)) == mininum_rdir); }
  bool is_outflow(coords* coords_in)
    { return (((*coarse_rdirs)(coords_in)) == outflow_rdir); }
  basin* create_basin() { return new latlon_basin(); }
private:
  field<int>* fine_rdirs = nullptr;
  field<int>* coarse_rdirs = nullptr;
  field<int>* connect_redirect_lat_index = nullptr;
  field<int>* connect_redirect_lon_index = nullptr;
  field<int>* flood_redirect_lat_index = nullptr;
  field<int>* flood_redirect_lon_index = nullptr;
  field<int>* connect_next_cell_lat_index = nullptr;
  field<int>* connect_next_cell_lon_index = nullptr;
  field<int>* flood_next_cell_lat_index = nullptr;
  field<int>* flood_next_cell_lon_index = nullptr;
  field<int>* connect_force_merge_lat_index = nullptr;
  field<int>* connect_force_merge_lon_index = nullptr;
  field<int>* flood_force_merge_lat_index = nullptr;
  field<int>* flood_force_merge_lon_index = nullptr;
  int* basin_minimum_lats = nullptr;
  int* basin_minimum_lons = nullptr;
  int outflow_rdir =  0;
  int mininum_rdir = -2;
};

class basin_post_processing_algorithm {
public:
  void build_and_simplify_basins();
  void check_for_loops();
  virtual basin* create_basin() = 0;
protected:
  stack<coords*> minima_q;
  vector<basin*> basins;
  field<int>* basin_numbers = nullptr;
  field<int>* coarse_catchment_numbers = nullptr;
  field<bool>* completed_cells = nullptr;
  field<bool>* redirect_targets = nullptr;
  field<bool>* minima = nullptr;
  field<bool>* use_flood_height_only = nullptr;
  field<merge_types>* merge_points = nullptr;
  bool* processed_basins = nullptr;
  bool* coarse_catchments = nullptr;
  grid* _coarse_grid = nullptr;
  grid* _fine_grid = nullptr;
  grid_params* coarse_grid_params = nullptr;
  grid_params* fine_grid_params = nullptr;
  basin* current_basin = nullptr;
  int num_basins;
  int num_catchments;
};

class latlon_basin_post_processing_algorithm :
  public basin_post_processing_algorithm {
private:
  field<int>* fine_rdirs = nullptr;
  field<int>* coarse_rdirs = nullptr;
  field<int>* connect_redirect_lat_index = nullptr;
  field<int>* connect_redirect_lon_index = nullptr;
  field<int>* flood_redirect_lat_index = nullptr;
  field<int>* flood_redirect_lon_index = nullptr;
  field<int>* connect_next_cell_lat_index = nullptr;
  field<int>* connect_next_cell_lon_index = nullptr;
  field<int>* flood_next_cell_lat_index = nullptr;
  field<int>* flood_next_cell_lon_index = nullptr;
  field<int>* connect_force_merge_lat_index = nullptr;
  field<int>* connect_force_merge_lon_index = nullptr;
  field<int>* flood_force_merge_lat_index = nullptr;
  field<int>* flood_force_merge_lon_index = nullptr;
  int* basin_minimum_lats = nullptr;
  int* basin_minimum_lons = nullptr;
public:
  basin* create_basin();
};

#endif //INCLUDE_BURN_POST_PROCESSING_ALGORITHM_HPP_
