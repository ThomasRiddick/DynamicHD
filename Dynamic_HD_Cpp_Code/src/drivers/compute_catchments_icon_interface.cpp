#include <iostream>
#include <fstream>
#include <unistd.h>
#include "base/grid.hpp"
#include "algorithms/catchment_computation_algorithm.hpp"

using namespace std;

void compute_catchments_icon_cython_interface(int ncells,
                                              int* next_cell_index_in,
                                              int* catchment_numbers_out,
                                              int* neighboring_cell_indices_in,
                                              int sort_catchments_by_size_in_int,
                                              string loop_log_filepath,
                                              int generate_selected_subcatchments_only_in_int,
                                              string subcatchment_list_filepath){
  bool sort_catchments_by_size_in = bool(sort_catchments_by_size_in_int);
  bool generate_selected_subcatchments_only_in = \
    generate_selected_subcatchments_only_in_int;
  bool use_secondary_neighbors = true;
  int* secondary_neighboring_cell_indices = nullptr;
  auto grid_params_obj =
    new icon_single_index_grid_params(ncells,
                                      neighboring_cell_indices_in,
                                      use_secondary_neighbors,
                                      secondary_neighboring_cell_indices);
  auto alg = catchment_computation_algorithm_icon_single_index();
  alg.setup_fields(catchment_numbers_out,
                   next_cell_index_in,grid_params_obj);
  if (! generate_selected_subcatchments_only_in) alg.compute_catchments();
  else {
    cout << "Reading selected cells from:" << endl;
    cout << subcatchment_list_filepath << endl;
    vector<int> selected_cells = vector<int>();
    ifstream subcatchment_list_file(subcatchment_list_filepath);
    string line;
    while (getline(subcatchment_list_file,line)) {
      selected_cells.push_back(stoi(line));
    }
    int catchment_number = 0;
    for(vector<int>::iterator i = selected_cells.begin();
                              i != selected_cells.end(); ++i){
      catchment_number++;
      alg.compute_cell_catchment(*i,catchment_number);
    }
  }
  if (sort_catchments_by_size_in) alg.renumber_catchments_by_size();
  vector<int>* loop_numbers = nullptr;
  loop_numbers = alg.identify_loops();
  ofstream loop_log_file;
  loop_log_file.open(loop_log_filepath);
  loop_log_file << "Loops found in catchments:" << endl;
  for (auto i = loop_numbers->begin(); i != loop_numbers->end(); ++i)
    loop_log_file << to_string(*i) << endl;
  loop_log_file.close();
  delete loop_numbers;
  delete grid_params_obj;
}
