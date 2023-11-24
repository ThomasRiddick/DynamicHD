#include <iostream>
#include <fstream>
#include <unistd.h>
#include <regex>
#include "base/grid.hpp"
#include "drivers/bifurcate_rivers_basic.hpp"

using namespace std;

void bifurcate_rivers_basic_icon_cython_interface(int ncells,
                                                  int* neighboring_cell_indices_in,
                                                  int* next_cell_index_in,
                                                  int* cumulative_flow_in,
                                                  int* landsea_mask_in_int,
                                                  int* number_of_outflows_out,
                                                  int* bifurcations_next_cell_index_out,
                                                  string mouth_positions_filepath,
                                                  int minimum_cells_from_split_to_main_mouth_in,
                                                  int maximum_cells_from_split_to_main_mouth_in,
                                                  double cumulative_flow_threshold_fraction_in){
  map<int,vector<int>> river_mouths;
  regex primary_mouth_regex("primary mouth:\\s*([0-9]+)");
  regex secondary_mouth_regex("secondary mouth:\\s*([0-9]+)");
  regex comment_regex("\\s*#");
  smatch primary_mouth_match;
  smatch secondary_mouth_match;
  int primary_mouth_index = -1;
  int secondary_mouth_index;
  vector<int>* secondary_mouths_vector = new vector<int>();
  cout << "Reading mouth position from file:" << endl;
  cout << mouth_positions_filepath << endl;
  ifstream mouth_position_file(mouth_positions_filepath);
  string line;
  while (getline(mouth_position_file,line)) {
    if (! regex_search(line,comment_regex)){
      if (regex_search(line,primary_mouth_match,primary_mouth_regex)){
        if(primary_mouth_index != -1){
          river_mouths.insert(pair<int,vector<int>>(primary_mouth_index,*secondary_mouths_vector));
          delete secondary_mouths_vector;
          secondary_mouths_vector = new vector<int>();
        }
        primary_mouth_index = stoi(primary_mouth_match[1]);
      } else if (regex_search(line,secondary_mouth_match,secondary_mouth_regex) &&
                 primary_mouth_index != -1){
        secondary_mouth_index = stoi(secondary_mouth_match[1]);
        secondary_mouths_vector->push_back(secondary_mouth_index);
      } else {
        cout << "Invalid mouth location file format" << endl;
        exit(EXIT_FAILURE);
      }
    }
  }
  if(primary_mouth_index != -1){
    river_mouths.insert(pair<int,vector<int>>(primary_mouth_index,*secondary_mouths_vector));
    delete secondary_mouths_vector;
  }
  bool* landsea_mask_in = new bool[ncells];
  for (int i = 0;i < ncells;i++) {
    landsea_mask_in[i] = ! bool(landsea_mask_in_int[i]);
  }
  icon_single_index_bifurcate_rivers_basic(river_mouths,
                                           next_cell_index_in,
                                           bifurcations_next_cell_index_out,
                                           cumulative_flow_in,
                                           number_of_outflows_out,
                                           landsea_mask_in,
                                           cumulative_flow_threshold_fraction_in,
                                           minimum_cells_from_split_to_main_mouth_in,
                                           maximum_cells_from_split_to_main_mouth_in,
                                           ncells,
                                           neighboring_cell_indices_in);
}
