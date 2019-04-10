/*
 * river_direction_determination_algorithm.hpp
 *
 *  Created on: Dec 6, 2018
 *      Author: thomasriddick
 */
#include "grid.hpp"
#include "field.hpp"
#include "priority_cell_queue.hpp"

class river_direction_determination_algorithm {
public:
  //destructor
  virtual ~river_direction_determination_algorithm()
    {delete land_sea; delete true_sinks; delete orography; delete completed_cells;
     delete _grid;};
  //setup logical options
  void setup_flags(bool always_flow_to_sea_in,
                   bool use_diagonal_nbrs_in,
                   bool mark_pits_as_true_sinks_in);
  //setup fields used
  void setup_fields(double* orography_in,
                    bool* land_sea_in,
                    bool* true_sinks_in,
                    grid_params* grid_params_in);
  //run the main algorithm
  void determine_river_directions();
protected:
  // find the river direction associated with a particular cell
  void find_river_direction(coords* coords_in);
  // check if (sea) cell has other cells flowing to it
  bool point_has_inflows(coords* coords_in);
  // use queued cells to resolve the river directions in flat areas
  void resolve_flat_areas();
  // mark the river direction between the initial coords and the destination
  // coordinates
  virtual void mark_river_direction(coords* initial_coords,
                                    coords* destination_coords) = 0;
  // Mark a sink point with the appropriate code at coords in
  virtual void mark_as_sink_point(coords* coords_in) = 0;
  // Mark an outflow point with the appropriate code at coords in
  virtual void mark_as_outflow_point(coords* coords_in) = 0;
  // Mark an ocean point with the appropriate code at coords in
  virtual void mark_as_ocean_point(coords* coords_in) = 0;
  //Get the coordinates of the cell downstream of a given set of coordinates
  //using the river direction generated
  virtual coords* get_downstream_cell_coords(coords* coords_in) = 0;
  //A input grid_parameter object to specify what kind of grid is required
  void process_neighbors(coords* center_coords,
                         double flat_height);
  queue<coords*> q;
  grid_params* _grid_params = nullptr;
  ///The grid created from the input grid parameters
  grid* _grid = nullptr;
  //The land sea mask; sea being true and land false
  field<bool>* land_sea;
  //True sinks points marked with true otherwise false
  field<bool>* true_sinks;
  //The DEM being used
  field<double>* orography;
  //Cells with river direction already assigned
  field<bool>* completed_cells;
  //A queue of potential exit points from flat areas
  priority_cell_queue_type potential_exit_points;
  //River always flow to a neighboring sea point even if another point is lower
  bool always_flow_to_sea = true;
  //Consider diagonal neighbors when iterating over neighbors
  bool use_diagonal_nbrs = true;
  //Mark any pits found as a true sink (otherwise the discovery of a pit will be
  //an error)
  bool mark_pits_as_true_sinks = true;
};

class river_direction_determination_algorithm_latlon :
public river_direction_determination_algorithm {
public:
  virtual ~river_direction_determination_algorithm_latlon()
    {delete rdirs;};
  //Setup the necessary fields to run a the lat-lon version of
  //the algorithm
  void setup_fields(double* rdirs_in,
                    double* orography_in,
                    bool* land_sea_in,
                    bool* true_sinks_in,
                    grid_params* grid_params_in);
protected:
  //Implementations of methods of the base calss
  void mark_river_direction(coords* initial_coords,
                                    coords* destination_coords);
  void mark_as_sink_point(coords* coords_in);
  void mark_as_outflow_point(coords* coords_in);
  void mark_as_ocean_point(coords* coords_in);
  coords* get_downstream_cell_coords(coords* coords_in);
  //The river directions generated
  field<double>* rdirs;
  // Code for a sink point
  const double sink_point_code = 5.0;
  // Code for an ocean point with rivers flowing into it
  const double outflow_code    = 0.0;
  // Code for an ocean point without river flowing into it
  const double ocean_code      = -1.0;
};

class river_direction_determination_algorithm_icon_single_index :
  public river_direction_determination_algorithm {
public:
  virtual ~river_direction_determination_algorithm_icon_single_index()
    {delete next_cell_index;};
  void setup_fields(int* next_cell_index_in,
                    double* orography_in,
                    bool* land_sea_in,
                    bool* true_sinks_in,
                    grid_params* grid_params_in);
protected:
  //Implementations of methods of the base calss
  void mark_river_direction(coords* initial_coords,
                            coords* destination_coords);
  void mark_as_sink_point(coords* coords_in);
  void mark_as_outflow_point(coords* coords_in);
  void mark_as_ocean_point(coords* coords_in);
  coords* get_downstream_cell_coords(coords* coords_in);
  //The river directions in the form of the index of the
  //downstream cell that each given cell should flow to
  field<int>* next_cell_index = nullptr;
  // Code for a sink point
  const int true_sink_value = -5;
  // Code for an ocean point with rivers flowing into it
  const int outflow_value = -1;
  // Code for an ocean point without river flowing into it
  const int ocean_value   = -2;
};
