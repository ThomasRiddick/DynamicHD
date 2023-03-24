/*
 * lake_filling_algorithm.hpp
 *
 *  Created on: Feb 21, 2018
 *      Author: thomasriddick
 */

#ifndef INCLUDE_BASIN_EVALUATION_ALGORITHM_HPP_
#define INCLUDE_BASIN_EVALUATION_ALGORITHM_HPP_

#include <queue>
#include <vector>
#include "base/cell.hpp"
#include "base/grid.hpp"
#include "base/field.hpp"
#include "base/enums.hpp"
#include "base/priority_cell_queue.hpp"
#include "base/merges_and_redirects.hpp"
#include "algorithms/sink_filling_algorithm.hpp"
using namespace std;

/// The basin evaluation algorithms main class
class basin_evaluation_algorithm {
public:
  ///Class destructor
	virtual ~basin_evaluation_algorithm();
  ///Setup the fields common to all version of the class
	void setup_fields(bool* minima_in,
                    double* raw_orography_in,
                    double* corrected_orography_in,
                    double* cell_areas_in,
                    double* connection_volume_thresholds_in,
                    double* flood_volume_thresholds_in,
                    int* prior_fine_catchments_in,
                    int* coarse_catchment_nums_in,
                    grid_params* grid_params_in,
                    grid_params* coarse_grid_params_in);
  /// Setup a sink filling algorithm to use to determine the order to process basins in
  void setup_sink_filling_algorithm(sink_filling_algorithm_4* sink_filling_alg_in);
  ///Main routine to evaluate basins
	void evaluate_basins();
  ///Retrieve the lake numbers field
  int* retrieve_lake_numbers();
  ///Method to test adding minima to queue
	queue<cell*> test_add_minima_to_queue(double* raw_orography_in,
                                        double* corrected_orography_in,
                                        bool* minima_in,
                                        int* prior_fine_catchments_in,
                                        sink_filling_algorithm_4*
                                        sink_filling_alg_in,
                                        grid_params* grid_params_in,
                                        grid_params* coarse_grid_params_in);
  ///Method to testing processing neighbors
	priority_cell_queue test_process_neighbors(coords* center_coords_in,
                                             bool*   completed_cells_in,
                                             double* raw_orography_in,
                                             double* corrected_orography_in,
                                             grid_params* grid_params_in);
  /// Method to test processing neighbors during the search algorithm used to
  /// find and set non-local redirects
	queue<landsea_cell*> test_search_process_neighbors(coords* search_coords_in,
	                                                   bool* search_completed_cells_in,
	                              										 grid_params* grid_params_in);
  merges_and_redirects* get_basin_merges_and_redirects()
    { return basin_merges_and_redirects; }
protected:
  ///Virtual setter for flood next cell index of previous cell
	virtual void set_previous_cells_flood_next_cell_index(coords* coords_in) = 0;
  ///Virtual setter for connect next cell index of previous cell
  virtual void set_previous_cells_connect_next_cell_index(coords* coords_in) = 0;
  ///Virtual getter for the next cell to fill in the filling order for a given cell
	virtual coords* get_cells_next_cell_index_as_coords(coords* coords_in,
                                                      height_types height_type_in) = 0;
  /// Returns a true if this cell is a sink or ocean cell and also sets the downstream
  /// coordinates of this cell
	virtual bool check_for_sinks_and_set_downstream_coords(coords* coords_in) = 0;
  /// Check if given coords are sink in the coarse river directions
  virtual bool coarse_cell_is_sink(coords* coords_in) = 0;
  /// Run algorithm over a single basin
	void evaluate_basin();
  /// Initialize the necessary variables for running over a basin
  void initialize_basin();
  /// Generate a list of minima ordered by a sink filling algorithm
  /// then split them up according to catchment
  void generate_minima();
  /// Add the minima for a given catchment to a queue of basin cells
	void add_minima_for_catchment_to_queue();
  /// Update basin variables and fields for a given cell
	void process_center_cell();
  /// Process the neighbors of a given cell
	void process_neighbors();
  /// If this neighbor is not completed then add to the queue and mark as
  /// completed
	void process_neighbor();
  /// Process a given cells neighbors when searching for additional merges
  /// at a given level
  void process_level_neighbors();
  /// Process a given neighbor when searching for additional merges at a
  /// given level
  void process_level_neighbor();
  void rebuild_center_cell(coords* coords_in,height_types height_type_in);
  void process_neighbors_when_rebuilding_basin(coords* coords_in);
  void set_previous_filled_cell_basin_number();
  void read_new_center_cell_variables();
  void update_center_cell_variables();
  void update_previous_filled_cell_variables();
  void rebuild_secondary_basin(int root_secondary_basin_number);
  void process_primary_merge(int target_basin_number,
                             int root_target_basin_number);
  void process_secondary_merge(coords* other_basin_entry_coords);
  void reprocess_secondary_merge(int root_secondary_basin_number,
                                 int target_primary_basin_number);
  void set_primary_merge(merge_and_redirect_indices* working_redirect,
                         int target_basin_number_in);
	void set_remaining_redirects();
	void set_secondary_redirect(merge_and_redirect_indices* working_redirect,
                              coords* redirect_coords,
                              coords* target_basin_center_coords,
                              height_types redirect_height_type);
  void set_preliminary_secondary_redirect(coords* other_basin_entry_coords);
	void set_primary_redirect(merge_and_redirect_indices* working_redirect,
                            int target_basin_number_in);
	void find_and_set_non_local_redirect_index_from_coarse_catchment_num(merge_and_redirect_indices*
                                                                       working_redirect,
                                                                       coords* current_center_coords,
                                                                       int coarse_catchment_number);
	void search_process_neighbors();
	void search_process_neighbor();
  bool process_all_merges_at_given_level();
  bool already_in_basin(coords* coords_in,height_types height_type_in);
	void find_and_set_previous_cells_non_local_redirect_index(merge_and_redirect_indices* working_redirect,
                                                            coords* current_center_coords,
                                                            coords* target_basin_center_coords);
  vector<queue<coords*>*> minima_coords_queues;
	queue<cell*> minima_q;
	priority_cell_queue q;
	queue<landsea_cell*> search_q;
  queue<basin_cell*> level_q;
  priority_cell_queue level_nbr_q;
  queue<pair<int,int>*> primary_merge_q;
	grid_params* _grid_params = nullptr;
	grid_params* _coarse_grid_params = nullptr;
	grid* _grid = nullptr;
	grid* _coarse_grid = nullptr;
	field<bool>* minima = nullptr;
	field<bool>* completed_cells = nullptr;
  field<bool>* level_completed_cells = nullptr;
  field<bool>* flooded_cells = nullptr;
  field<bool>* connected_cells = nullptr;
  field<bool>* basin_flooded_cells = nullptr;
  field<bool>* basin_connected_cells = nullptr;
	field<bool>* search_completed_cells = nullptr;
	field<bool>* requires_flood_redirect_indices = nullptr;
  field<bool>* requires_connect_redirect_indices = nullptr;
	field<double>* raw_orography = nullptr;
	field<double>* corrected_orography = nullptr;
  field<double>* cell_areas = nullptr;
	field<double>* connection_volume_thresholds = nullptr;
	field<double>* flood_volume_thresholds = nullptr;
	field<int>* basin_numbers = nullptr;
	field<int>* coarse_catchment_nums = nullptr;
	field<int>* prior_fine_catchments = nullptr;
  field<int>* catchments_from_sink_filling = nullptr;
  basin_cell* minimum = nullptr;
	basin_cell* center_cell = nullptr;
	coords* center_coords = nullptr;
  coords* previous_filled_cell_coords = nullptr;
  coords* new_center_coords = nullptr;
	coords* search_coords = nullptr;
  coords* level_coords = nullptr;
  coords* null_coords = nullptr;
  coords* downstream_coords = nullptr;
	vector<coords*>* neighbors_coords = nullptr;
	vector<coords*>* search_neighbors_coords = nullptr;
	vector<coords*> basin_catchment_centers;
  vector<coords*> basin_sink_points;
  vector<vector<pair<coords*,bool>*>*> basin_connect_and_fill_orders;
  vector<pair<coords*,bool>*>* basin_connect_and_fill_order;
  disjoint_set_forest basin_connections;
  merges_and_redirects* basin_merges_and_redirects = nullptr;
  height_types new_center_cell_height_type;
  height_types center_cell_height_type;
  height_types previous_filled_cell_height_type;
  sink_filling_algorithm_4* sink_filling_alg;
	int basin_number;
  int  catchments_from_sink_filling_catchment_num;
  bool secondary_merge_found;
  double lake_area;
  double new_center_cell_height;
	double center_cell_height;
	double previous_filled_cell_height;
	double center_cell_volume_threshold;
  double surface_height;
	const int null_catchment = 0;
};

class latlon_basin_evaluation_algorithm : public basin_evaluation_algorithm {
public:
	virtual ~latlon_basin_evaluation_algorithm();
  void setup_fields(bool* minima_in,
                    double* raw_orography_in,
                    double* corrected_orography_in,
                    double* cell_areas_in,
                    double* connection_volume_thresholds_in,
                    double* flood_volume_thresholds_in,
                    double* prior_fine_rdirs_in,
                    double* prior_coarse_rdirs_in,
                    int* prior_fine_catchments_in,
                    int* coarse_catchment_nums_in,
                    int* flood_next_cell_lat_index_in,
                    int* flood_next_cell_lon_index_in,
                    int* connect_next_cell_lat_index_in,
                    int* connect_next_cell_lon_index_in,
                    grid_params* grid_params_in,
                    grid_params* coarse_grid_params_in);
	priority_cell_queue test_process_center_cell(basin_cell* center_cell_in,
                                               coords* center_coords_in,
	                                             coords* previous_filled_cell_coords_in,
                                               double* flood_volume_thresholds_in,
                                               double* connection_volume_thresholds_in,
                                               double* raw_orography_in,
                                               double* cell_areas_in,
                                               int* flood_next_cell_lat_index_in,
                                               int* flood_next_cell_lon_index_in,
                                               int* connect_next_cell_lat_index_in,
                                               int* connect_next_cell_lon_index_in,
                                               int* basin_numbers_in,
                                               bool* flooded_cells_in,
                                               bool* connected_cells_in,
                                               double& center_cell_volume_threshold_in,
                                               double& lake_area_in,
                                               int basin_number_in,
                                               double center_cell_height_in,
                                               double& previous_filled_cell_height_in,
                                               height_types& previous_filled_cell_height_type_in,
                                               grid_params* grid_params_in);
	void test_set_primary_merge_and_redirect(vector<coords*> basin_catchment_centers_in,
                                           double* prior_coarse_rdirs_in,
                                           int* basin_numbers_in,
                                           int* coarse_catchment_nums_in,
                                           coords* new_center_coords_in,
                                           coords* center_coords_in,
                                           coords* previous_filled_cell_coords_in,
                                           height_types previous_filled_cell_height_type_in,
                                           grid_params* grid_params_in,
                                           grid_params* coarse_grid_params_in);
	void test_set_secondary_redirect(double* prior_coarse_rdirs_in,
                                   int* flood_next_cell_lat_index_in,
                                   int* flood_next_cell_lon_index_in,
                                   int* connect_next_cell_lat_index_in,
                                   int* connect_next_cell_lon_index_in,
                                   int* coarse_catchment_nums_in,
                                   bool* requires_flood_redirect_indices_in,
                                   bool* requires_connect_redirect_indices_in,
                                   double* raw_orography_in,
                                   double* corrected_orography_in,
                                   coords* new_center_coords_in,
                                   coords* center_coords_in,
                                   coords* previous_filled_cell_coords_in,
                                   coords* target_basin_center_coords,
                                   height_types& previous_filled_cell_height_type_in,
                                   grid_params* grid_params_in,
                                   grid_params* coarse_grid_params_in);
	void test_set_remaining_redirects(vector<coords*> basin_catchment_centers_in,
                                    double* prior_fine_rdirs_in,
                                    double* prior_coarse_rdirs_in,
                                    bool* requires_flood_redirect_indices_in,
                                    bool* requires_connect_redirect_indices_in,
                                    int* basin_numbers_in,
                                    int* prior_fine_catchments_in,
                                    int* coarse_catchment_nums_in,
                                    int* flood_next_cell_lat_index_in,
                                    int* flood_next_cell_lon_index_in,
                                    int* connect_next_cell_lat_index_in,
                                    int* connect_next_cell_lon_index_in,
                                    grid_params* grid_params_in,
                                    grid_params* coarse_grid_params_in);
private:
	void set_previous_cells_flood_next_cell_index(coords* coords_in);
  void set_previous_cells_connect_next_cell_index(coords* coords_in);
	bool check_for_sinks_and_set_downstream_coords(coords* coords_in);
  bool coarse_cell_is_sink(coords* coords_in);
	coords* get_cells_next_cell_index_as_coords(coords* coords_in,
                                              height_types height_type_in);
  void output_diagnostics_for_grid_section(int min_lat,int max_lat,
                                           int min_lon,int max_lon);
	field<double>* prior_fine_rdirs = nullptr;
  field<double>* prior_coarse_rdirs = nullptr;
	field<int>* flood_next_cell_lat_index = nullptr;
	field<int>* flood_next_cell_lon_index = nullptr;
  field<int>* connect_next_cell_lat_index = nullptr;
  field<int>* connect_next_cell_lon_index = nullptr;
};

class icon_single_index_basin_evaluation_algorithm : public basin_evaluation_algorithm {
public:
  virtual ~icon_single_index_basin_evaluation_algorithm();
  void setup_fields(bool* minima_in,
                    double* raw_orography_in,
                    double* corrected_orography_in,
                    double* cell_areas_in,
                    double* connection_volume_thresholds_in,
                    double* flood_volume_thresholds_in,
                    int* prior_fine_rdirs_in,
                    int* prior_coarse_rdirs_in,
                    int* prior_fine_catchments_in,
                    int* coarse_catchment_nums_in,
                    int* flood_next_cell_index_in,
                    int* connect_next_cell_index_in,
                    grid_params* grid_params_in,
                    grid_params* coarse_grid_params_in);
private:
  void set_previous_cells_flood_next_cell_index(coords* coords_in);
  void set_previous_cells_connect_next_cell_index(coords* coords_in);
  bool check_for_sinks_and_set_downstream_coords(coords* coords_in);
  bool coarse_cell_is_sink(coords* coords_in);
  coords* get_cells_next_cell_index_as_coords(coords* coords_in,
                                              height_types height_type_in);
  void output_diagnostics_for_grid_section(int min_lat,int max_lat,
                                           int min_lon,int max_lon);
  field<int>* prior_fine_rdirs = nullptr;
  field<int>* prior_coarse_rdirs = nullptr;
  field<int>* flood_next_cell_index = nullptr;
  field<int>* connect_next_cell_index = nullptr;
  const int true_sink_value = -5;
  const int outflow_value = -1;
};

#endif /*INCLUDE_BASIN_EVALUATION_ALGORITHM_HPP_*/
