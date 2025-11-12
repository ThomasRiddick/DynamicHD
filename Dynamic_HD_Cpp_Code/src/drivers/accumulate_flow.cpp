#include "drivers/accumulate_flow.hpp"
#include "algorithms/flow_accumulation_algorithm.hpp"
#include "base/convert_rdirs_to_indices.hpp"
#include "base/grid.hpp"

void accumulate_flow_icon_single_index(int ncells,
                                       int* neighboring_cell_indices_in,
                                       int* input_next_cell_index,
                                       int* output_cumulative_flow,
                                       int* bifurcated_next_cell_index) {
  cout << "Entering Flow Accumulation C++ Code" << endl;
  for (int i = 0; i < ncells; i++) {
    if (input_next_cell_index[i] ==  0 || input_next_cell_index[i] == -1 ||
        input_next_cell_index[i] == -2 || input_next_cell_index[i] == -5 )
      input_next_cell_index[i] = -3;
  }
  bool use_secondary_neighbors = true;
  int* secondary_neighboring_cell_indices = nullptr;
  auto grid_params_obj =
    new icon_single_index_grid_params(ncells,
                                      neighboring_cell_indices_in,
                                      use_secondary_neighbors,
                                      secondary_neighboring_cell_indices);
  icon_single_index_flow_accumulation_algorithm flow_acc_alg;
  flow_acc_alg.setup_fields(input_next_cell_index,
                            output_cumulative_flow,
                            grid_params_obj,
                            bifurcated_next_cell_index);
  flow_acc_alg.generate_cumulative_flow(false);
  if (bifurcated_next_cell_index) {
    flow_acc_alg.update_bifurcated_flows();
  }
  delete grid_params_obj;
}

void accumulate_flow_latlon(int nlat, int nlon,
                            int* input_river_directions,
                            int* output_cumulative_flow) {
  cout << "Entering Flow Accumulation C++ Code" << endl;
  int* next_cell_index_lat = new int[nlat*nlon];
  int* next_cell_index_lon = new int[nlat*nlon];
  convert_rdirs_to_latlon_indices(nlat,nlon,
                                  input_river_directions,
                                  next_cell_index_lat,
                                  next_cell_index_lon);
  for (int i = 0; i < nlat*nlon; i++) {
    if (input_river_directions[i] ==  0 ||
        input_river_directions[i] == -1 ||
        input_river_directions[i] == -2 ||
        input_river_directions[i] == 5){
      next_cell_index_lat[i] = -2;
      next_cell_index_lon[i] = -2;
    }
  }
  auto grid_params_obj = new latlon_grid_params(nlat,nlon);
  latlon_flow_accumulation_algorithm flow_acc_alg;
  flow_acc_alg.setup_fields(next_cell_index_lat,
                            next_cell_index_lon,
                            output_cumulative_flow,
                            grid_params_obj);
  flow_acc_alg.generate_cumulative_flow(false);
  delete grid_params_obj;
  delete[] next_cell_index_lat;
  delete[] next_cell_index_lon;
}


