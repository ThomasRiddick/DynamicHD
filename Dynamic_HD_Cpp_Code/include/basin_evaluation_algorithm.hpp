/*
 * lake_filling_algorithm.hpp
 *
 *  Created on: Feb 21, 2018
 *      Author: thomasriddick
 */

#include <queue>
#include "cell.hpp"
#include "grid.hpp"
#include "field.hpp"
#include "enums.hpp"
#include "priority_cell_queue.hpp"

class basin_evaluation_algorithm {
public:
	virtual ~basin_evaluation_algorithm();
	void setup_fields(bool* minima_in,
                    double* raw_orography_in,
                    double* corrected_orography_in,
                    double* connection_volume_thresholds_in,
                    double* flood_volume_thresholds_in,
                    int* prior_fine_catchments_in,
                    int* coarse_catchment_nums_in,
                    bool* flood_local_redirect_in,
                    bool* connect_local_redirect_in,
                    merge_types* merge_points_in,
                    grid_params* grid_params_in,
                    grid_params* coarse_grid_params_in);
	void evaluate_basins();
	priority_cell_queue test_add_minima_to_queue(double* raw_orography_in,
                                               double* corrected_orography_in,
                                               bool* minima_in,
                                               grid_params* grid_params_in,
                                               grid_params* coarse_grid_params_in);
	priority_cell_queue test_process_neighbors(coords* center_coords_in,
                                             bool*   completed_cells_in,
                                             double* raw_orography_in,
                                             double* corrected_orography_in,
                                             grid_params* grid_params_in);
	queue<landsea_cell*> test_search_process_neighbors(coords* search_coords_in,
	                                                   bool* search_completed_cells_in,
	                              										 grid_params* grid_params_in);
protected:
	virtual void set_previous_cells_flood_next_cell_index() = 0;
  virtual void set_previous_cells_connect_next_cell_index() = 0;
	virtual void set_previous_cells_flood_force_merge_index() = 0;
  virtual void set_previous_cells_connect_force_merge_index() = 0;
  virtual void set_previous_cells_redirect_index(coords* initial_fine_coords,
                                                 coords* target_coords,
                                                 height_types height_type) = 0;
	virtual coords* get_cells_next_cell_index_as_coords(coords* coords_in,
                                                      height_types height_type_in) = 0;
  virtual coords* get_cells_redirect_index_as_coords(coords* coords_in,
                                                     height_types height_type_in) = 0;
	virtual bool check_for_sinks(coords* coords_in) = 0;
  bool skip_center_cell();
	void evaluate_basin();
	void add_minima_to_queue();
	void process_center_cell();
	void process_neighbors();
	void process_neighbor();
  void set_merge_type(basic_merge_types current_merge_type);
  void set_primary_merge();
	void set_remaining_redirects();
	void set_secondary_redirect();
	void set_primary_redirect();
  void set_previous_cells_redirect_type(coords* initial_fine_coords,height_types height_type,
                                        redirect_type local_redirect);
	void find_and_set_non_local_redirect_index_from_coarse_catchment_num(coords* initial_center_coords,
	                                                                		 coords* current_center_coords,
                                                                       height_types initial_center_height_type,
	                                                     								 int coarse_catchment_number);
	void search_process_neighbors();
	void search_process_neighbor();
	void find_and_set_previous_cells_non_local_redirect_index(coords* initial_center_coords,
	                                                     			coords* current_center_coords,
	                                                     			coords* catchment_center_coords,
                                                            height_types initial_center_height_type);
	priority_cell_queue minima_q;
	priority_cell_queue q;
	queue<landsea_cell*> search_q;
	grid_params* _grid_params = nullptr;
	grid_params* _coarse_grid_params = nullptr;
	grid* _grid = nullptr;
	grid* _coarse_grid = nullptr;
	field<bool>* minima = nullptr;
	field<bool>* completed_cells = nullptr;
  field<bool>* flooded_cells = nullptr;
  field<bool>* connected_cells = nullptr;
	field<bool>* search_completed_cells = nullptr;
	field<bool>* requires_flood_redirect_indices = nullptr;
  field<bool>* requires_connect_redirect_indices = nullptr;
  field<bool>* flood_local_redirect = nullptr;
  field<bool>* connect_local_redirect = nullptr;
	field<double>* raw_orography = nullptr;
	field<double>* corrected_orography = nullptr;
	field<double>* connection_volume_thresholds = nullptr;
	field<double>* flood_volume_thresholds = nullptr;
	field<int>* basin_catchment_numbers = nullptr;
	field<int>* coarse_catchment_nums = nullptr;
	field<int>* prior_fine_catchments = nullptr;
	field<merge_types>* merge_points = nullptr;
	basin_cell* minimum = nullptr;
	basin_cell* center_cell = nullptr;
	coords* center_coords = nullptr;
  coords* previous_filled_cell_coords = nullptr;
  coords* new_center_coords = nullptr;
	coords* search_coords = nullptr;
	vector<coords*>* neighbors_coords = nullptr;
	vector<coords*>* search_neighbors_coords = nullptr;
	vector<coords*> basin_catchment_centers;
  height_types center_cell_height_type;
  height_types previous_filled_cell_height_type;
	int cell_number;
	int basin_catchment_number;
	double center_cell_height;
	double previous_filled_cell_height;
	double center_cell_volume_threshold;
	const int null_catchment = 0;
};

class latlon_basin_evaluation_algorithm : public basin_evaluation_algorithm {
public:
	virtual ~latlon_basin_evaluation_algorithm();
  void setup_fields(bool* minima_in,
                    double* raw_orography_in,
                    double* corrected_orography_in,
                    double* connection_volume_thresholds_in,
                    double* flood_volume_thresholds_in,
                    double* prior_fine_rdirs_in,
                    int* prior_fine_catchments_in,
                    int* coarse_catchment_nums_in,
                    int* flood_next_cell_lat_index_in,
                    int* flood_next_cell_lon_index_in,
                    int* connect_next_cell_lat_index_in,
                    int* connect_next_cell_lon_index_in,
                    int* flood_force_merge_lat_index_in,
                    int* flood_force_merge_lon_index_in,
                    int* connect_force_merge_lat_index_in,
                    int* connect_force_merge_lon_index_in,
                    int* flood_redirect_lat_index_in,
                    int* flood_redirect_lon_index_in,
                    int* connect_redirect_lat_index_in,
                    int* connect_redirect_lon_index_in,
                    bool* flood_local_redirect_in,
                    bool* connect_local_redirect_in,
                    merge_types* merge_points_in,
                    grid_params* grid_params_in,
                    grid_params* coarse_grid_params_in);
	priority_cell_queue test_process_center_cell(basin_cell* center_cell_in,
                                               coords* center_coords_in,
	                                             coords* previous_filled_cell_coords_in,
                                               double* flood_volume_thresholds_in,
                                               double* connection_volume_thresholds_in,
                                               double* raw_orography_in,
                                               int* flood_next_cell_lat_index_in,
                                               int* flood_next_cell_lon_index_in,
                                               int* connect_next_cell_lat_index_in,
                                               int* connect_next_cell_lon_index_in,
                                               int* basin_catchment_numbers_in,
                                               bool* flooded_cells_in,
                                               bool* connected_cells_in,
                                               double& center_cell_volume_threshold_in,
                                               int& cell_number_in,
                                               int basin_catchment_number_in,
                                               double center_cell_height_in,
                                               double& previous_filled_cell_height_in,
                                               height_types& previous_filled_cell_height_type_in,
                                               grid_params* grid_params_in);
	void test_set_primary_merge_and_redirect(vector<coords*> basin_catchment_centers_in,
                                           int* basin_catchment_numbers_in,
                                           int* coarse_catchment_nums_in,
                                           int* flood_force_merge_lat_index_in,
                                           int* flood_force_merge_lon_index_in,
                                           int* connect_force_merge_lat_index_in,
                                           int* connect_force_merge_lon_index_in,
                                           int* flood_redirect_lat_index_in,
                                           int* flood_redirect_lon_index_in,
                                           int* connect_redirect_lat_index_in,
                                           int* connect_redirect_lon_index_in,
                                           bool* flood_local_redirect_in,
                                           bool* connect_local_redirect_in,
                                           merge_types* merge_points_in,
                                           coords* new_center_coords_in,
                                           coords* center_coords_in,
                                           coords* previous_filled_cell_coords_in,
                                           height_types previous_filled_cell_height_type_in,
                                           grid_params* grid_params_in,
                                           grid_params* coarse_grid_params_in);
	void test_set_secondary_redirect(int* flood_next_cell_lat_index_in,
                                   int* flood_next_cell_lon_index_in,
                                   int* connect_next_cell_lat_index_in,
                                   int* connect_next_cell_lon_index_in,
                                   int* flood_redirect_lat_in,
                                   int* flood_redirect_lon_in,
                                   int* connect_redirect_lat_in,
                                   int* connect_redirect_lon_in,
                                   bool* requires_flood_redirect_indices_in,
                                   bool* requires_connect_redirect_indices_in,
                                   double* raw_orography_in,
                                   double* corrected_orography_in,
                                   coords* new_center_coords_in,
                                   coords* center_coords_in,
                                   coords* previous_filled_cell_coords_in,
                                   height_types& previous_filled_cell_height_type_in,
                                   grid_params* grid_params_in);
	void test_set_remaining_redirects(vector<coords*> basin_catchment_centers_in,
                                    double* prior_fine_rdirs_in,
                                    bool* requires_flood_redirect_indices_in,
                                    bool* requires_connect_redirect_indices_in,
                                    int* basin_catchment_numbers_in,
                                    int* prior_fine_catchments_in,
                                    int* coarse_catchment_nums_in,
                                    int* flood_next_cell_lat_index_in,
                                    int* flood_next_cell_lon_index_in,
                                    int* connect_next_cell_lat_index_in,
                                    int* connect_next_cell_lon_index_in,
                                    int* flood_redirect_lat_index_in,
                                    int* flood_redirect_lon_index_in,
                                    int* connect_redirect_lat_index_in,
                                    int* connect_redirect_lon_index_in,
                                    bool* flood_local_redirect_in,
                                    bool* connect_local_redirect_in,
                                    grid_params* grid_params_in,
                                    grid_params* coarse_grid_params_in);
private:
	void set_previous_cells_flood_next_cell_index();
  void set_previous_cells_connect_next_cell_index();
	void set_previous_cells_flood_force_merge_index();
  void set_previous_cells_connect_force_merge_index();
	void set_previous_cells_redirect_index(coords* initial_fine_coords,coords* target_coords,
                                         height_types height_type);
	bool check_for_sinks(coords* coords_in);
	coords* get_cells_next_cell_index_as_coords(coords* coords_in,
                                              height_types height_type_in);
  coords* get_cells_redirect_index_as_coords(coords* coords_in,
                                             height_types height_type_in);
	field<double>* prior_fine_rdirs = nullptr;
	field<int>* flood_next_cell_lat_index = nullptr;
	field<int>* flood_next_cell_lon_index = nullptr;
  field<int>* connect_next_cell_lat_index = nullptr;
  field<int>* connect_next_cell_lon_index = nullptr;
	field<int>* flood_force_merge_lat_index = nullptr;
	field<int>* flood_force_merge_lon_index = nullptr;
  field<int>* connect_force_merge_lat_index = nullptr;
  field<int>* connect_force_merge_lon_index = nullptr;
	field<int>* flood_redirect_lat_index = nullptr;
	field<int>* flood_redirect_lon_index = nullptr;
	field<int>* connect_redirect_lat_index = nullptr;
	field<int>* connect_redirect_lon_index = nullptr;
};
