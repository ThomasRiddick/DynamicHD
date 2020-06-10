#ifndef INCLUDE_OROGRAPHY_CREATION_ALGORITHM_HPP_
#define INCLUDE_OROGRAPHY_CREATION_ALGORITHM_HPP_

#include "grid.hpp"
#include "field.hpp"
#include "priority_cell_queue.hpp"

class orography_creation_algorithm {
public:
  void setup_flags(double sea_level_in);
  void setup_fields(bool* landsea_in,
                    double* incline_in,
                    double* orography_in,
                    grid_params* grid_params_in);
  ~orography_creation_algorithm();
  void reset();
  void create_orography();
private:
  void process_neighbors(vector<coords*>* neighbors_coords);
  void add_landsea_edge_cells_to_q();
  ///Main priority queue
  priority_cell_queue q;
  cell* center_cell = nullptr;
  ///Coordinates of the central cell being processed
  coords* center_coords;
  double center_height;
  ///A input grid_parameter object to specify what kind of grid is required
  grid_params* _grid_params = nullptr;
  ///The grid created from the input grid parameters
  grid* _grid = nullptr;
  field<bool>* landsea = nullptr;
  field<bool>* completed_cells = nullptr;
  field<double>* inclines = nullptr;
  field<double>* orography = nullptr;
  double sea_level = 0.0;
};

#endif /* INCLUDE_OROGRAPHY_CREATION_ALGORITHM_ALGORITHM_HPP_ */
