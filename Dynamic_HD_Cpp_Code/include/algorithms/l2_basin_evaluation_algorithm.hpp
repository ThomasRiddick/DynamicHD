#include<vector>
#include<map>
#include<functional>
#include<set>
#include "base/cell.hpp"
#include "base/coords.hpp"
#include "base/enums.hpp"
#include "base/grid.hpp"
#include "base/field.hpp"
#include "base/disjoint_set.hpp"
#include "base/priority_cell_queue.hpp"
using namespace std;

const int null_lake_number = -1;

const bool print_with_offsets = true;


class filling_order_entry {
public:
  filling_order_entry(coords* cell_coords_in, height_types height_type_in,
                      double volume_threshold_in, double cell_height_in) :
    cell_coords(cell_coords_in), height_type(height_type_in),
    volume_threshold(volume_threshold_in), cell_height(cell_height_in) {}
  ~filling_order_entry() {delete cell_coords;}
  coords* cell_coords;
  height_types height_type;
  double volume_threshold;
  double cell_height;
  void print(ostream& outstr);
  friend ostream& operator<< (ostream& outstr, filling_order_entry& entry) {
    entry.print(outstr);
  return outstr;
  };
};

class store_to_array {
public:
  store_to_array();
  void set_null_coords(coords* null_coords_in);
  void add_object();
  void complete_object();
  void add_number(int number_in);
  void add_coords(coords* coords_in, int array_offset, int additional_lat_offset=0);
  void add_field(vector<int>* field_in);
  void add_outflow_points_dict(map<int,pair<coords*,bool>*> &outflow_points_in,
                               grid* grid_in,
                               int array_offset,
                               int additional_lat_offset=0);
  void add_filling_order(vector<filling_order_entry*> &filling_order_in,
                         grid* grid_in,int array_offset=0,
                         int additional_lat_offset=0);
  vector<double>* complete_array();
private:
  vector<vector<double>*> objects;
  vector<double>* working_object;
  const int height_type_offset = 1;
  coords* null_coords = nullptr;
};

class lake_variables {
public:
  lake_variables(int lake_number_in,
                 coords* center_coords_in,
                 double lake_lower_boundary_height_in,
                 int primary_lake_in=null_lake_number,
                 set<int>* secondary_lakes_in=new set<int>());
  ~lake_variables();
  void set_primary_lake(int primary_lake_in);
  void set_potential_exit_points(vector<coords*>* potential_exit_points_in);
  void set_filled_lake_area();
  int lake_number;
  coords* center_coords;
  //Keep references to lakes as number not objects
  int primary_lake;
  set<int>* secondary_lakes;
  double center_cell_volume_threshold;
  double lake_area;
  map<int,pair<coords*,bool>*> outflow_points;
  vector<filling_order_entry*> filling_order;
  double lake_lower_boundary_height;
  double filled_lake_area;

  //Working variables - don't need to exported
  //lake area
  map<int,coords*> spill_points;
  vector<coords*>* potential_exit_points;
  vector<coords*> list_of_cells_in_lake;
  void print(ostream& outstr);
  friend ostream& operator<< (ostream& outstr, lake_variables& lake) {
    lake.print(outstr);
    return outstr;
  };
};

class simple_search {
public:
  simple_search(grid* grid_in,
                grid_params* grid_params_in);
  coords* search(function<bool(coords*)> target_found_func,
                 function<bool(coords*)> ignore_nbr_func_in,
                 coords* start_point);
  ~simple_search();
private:
  void search_process_neighbors();
  function<bool(coords*)> ignore_nbr_func;
  grid* _grid = nullptr;
  field<bool>* search_completed_cells = nullptr;
  //Allow for a fast reset of the array
  vector<coords*> search_completed_cell_list;
  queue<landsea_cell*> search_q;
  coords* search_coords;
};


class basin_evaluation_algorithm {
public:
  basin_evaluation_algorithm(bool* minima_in,
                             double* raw_orography_in,
                             double* corrected_orography_in,
                             double* cell_areas_in,
                             int* prior_fine_catchment_nums_in,
                             int* coarse_catchment_nums_in,
                             int* catchments_from_sink_filling_in,
                             int additional_lat_offset_in,
                             grid_params* grid_params_in,
                             grid_params* coarse_grid_params_in);
  virtual ~basin_evaluation_algorithm();
  void evaluate_basins();
  void initialize_basin(lake_variables* lake);
  void search_for_outflows_on_level(int lake_number);
  void process_level_neighbors(int lake_number,
                               coords* level_coords);
  void process_center_cell(lake_variables* lake);
  void process_neighbors();
  void set_outflows();
  coords* find_non_local_outflow_point(coords* first_cell_beyond_rim_coords);
  vector<double>* get_lakes_as_array();
  field<bool>* get_lake_mask();
  int get_number_of_lakes();
  field<int>* get_lake_numbers();
  vector<lake_variables*> get_lakes();
  virtual pair<bool,coords*> check_for_sinks_and_get_downstream_coords(coords* coords_in) = 0;
  virtual bool check_if_fine_cell_is_sink(coords* coords_in) = 0;
protected:
  grid_params* _grid_params = nullptr;
  grid_params* _coarse_grid_params = nullptr;
  grid* _grid = nullptr;
  grid* _coarse_grid = nullptr;
  coords* null_coords = nullptr;
private:
  field<bool>* minima = nullptr;
  field<bool>* completed_cells = nullptr;
  field<bool>* level_completed_cells = nullptr;
  field<bool>* cells_in_lake = nullptr;
  field<double>*  raw_orography = nullptr;
  field<double>* corrected_orography = nullptr;
  field<double>* cell_areas = nullptr;
  field<int>* prior_fine_catchment_nums = nullptr;
  field<int>* lake_numbers = nullptr;
  field<bool>* lake_mask = nullptr;
  field<int>* coarse_catchment_nums = nullptr;
  field<int>* catchments_from_sink_filling = nullptr;
  disjoint_set_forest* lake_connections = nullptr;
  vector<coords*> sink_points;
  vector<lake_variables*> lakes;
  queue<lake_variables*> lake_q;
  priority_cell_queue q;
  //Mostly FIFO access but once need to iterate over
  //entire container hence vector not queue
  vector<basin_cell*> level_q;
  vector<coords*> level_completed_cell_list;
  simple_search* search_alg = nullptr;
  simple_search* coarse_search_alg = nullptr;
  basin_cell* center_cell;
  coords* center_coords;
  double center_cell_height;
  height_types center_cell_height_type;
  coords* previous_cell_coords;
  height_types previous_cell_height_type;
  double previous_cell_height;
  coords* new_center_coords;
  height_types new_center_cell_height_type;
  double new_center_cell_height;
  double searched_level_height;
  vector<int> outflow_lake_numbers;
  vector<coords*>* potential_exit_points;
  double catchments_from_sink_filling_catchment_num;
  int additional_lat_offset = 0;
};

class latlon_basin_evaluation_algorithm :
    public basin_evaluation_algorithm {
public:
  virtual ~latlon_basin_evaluation_algorithm();
  latlon_basin_evaluation_algorithm(bool* minima_in,
                                    double* raw_orography_in,
                                    double* corrected_orography_in,
                                    double* cell_areas_in,
                                    int* prior_fine_rdirs_in,
                                    int* prior_fine_catchment_nums_in,
                                    int* coarse_catchment_nums_in,
                                    int* catchments_from_sink_filling_in,
                                    int additional_lat_offset_in,
                                    grid_params* grid_params_in,
                                    grid_params* coarse_grid_params_in);
  pair<bool,coords*> check_for_sinks_and_get_downstream_coords(coords* coords_in);
  bool check_if_fine_cell_is_sink(coords* coords_in);
private:
  field<int>* prior_fine_rdirs = nullptr;
  const int true_sink_value = 5;
};

class single_index_basin_evaluation_algorithm :
    public basin_evaluation_algorithm {
public:
  virtual ~single_index_basin_evaluation_algorithm() {};
  single_index_basin_evaluation_algorithm(bool* minima_in,
                                          double* raw_orography_in,
                                          double* corrected_orography_in,
                                          double* cell_areas_in,
                                          int* prior_next_cell_indices_in,
                                          int* prior_fine_catchment_nums_in,
                                          int* coarse_catchment_nums_in,
                                          int* catchments_from_sink_filling_in,
                                          grid_params* grid_params_in,
                                          grid_params* coarse_grid_params_in);
  pair<bool,coords*> check_for_sinks_and_get_downstream_coords(coords* coords_in);
  bool check_if_fine_cell_is_sink(coords* coords_in);
private:
  field<int>* prior_next_cell_indices = nullptr;
  const int true_sink_value = -5;
  const int outflow_value = -2;
};
