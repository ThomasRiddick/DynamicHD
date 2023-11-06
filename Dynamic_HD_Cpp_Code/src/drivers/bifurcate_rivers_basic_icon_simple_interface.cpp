#include <iostream>
#include <fstream>
#include <unistd.h>
#include <regex>
#include "base/grid.hpp"
#include "drivers/bifurcate_rivers_basic.hpp"

using namespace std;

// string UNITS = "units";
// string GRID_TYPE = "grid_type";
// string LONG_NAME = "long_name";
// string COORDINATES = "coordinates";
// string STANDARD_NAME = "standard_name";
// string BOUNDS = "bounds";

// string METRES = "m";
// string RADIAN = "radian";
// string UNSTRUCTURED = "unstructured";
// string LATITUDE = "latitude";
// string CENTER_LATITUDE = "center latitude";
// string LONGITUDE = "longitude";
// string CENTER_LONGITUDE = "center longitude";

// void print_usage(){
//     cout <<
//     "Usage: " << endl;
//     cout <<
//     "./Bifurcate_Rivers_Basic_SI_Exec  [next_cell_index_filepath]" << endl;
//     cout <<
//     "[cumulative_flow_filepath] [landsea_mask_filepath]" << endl;
//     cout <<
//     "[output_number_of_outflows_filepath] [output_next_cell_index_filepath]" << endl;
//     cout <<
//     "[output_bifurcated_next_cell_index_filepath] [grid_params_filepath]" << endl;
//     cout <<
//     "[mouth_positions_filepath] [next_cell_index_fieldname]" << endl;
//     cout <<
//     "[cumulative_flow_fieldname] [landsea_mask_fieldname]" << endl;
//     cout <<
//     "[minimum_cells_from_split_to_main_mouth_string]" << endl;
//     cout <<
//     "[maximum_cells_from_split_to_main_mouth_string]" << endl;
//     cout <<
//     "[cumulative_flow_threshold_fraction_string]" << endl;
// }

// void print_help(){
//   print_usage();
//   cout << "Bifurcates selected river mouths"
//        << endl;
//   cout << "Arguments:" << endl;
//   cout << "next_cell_index_filepath - " << endl;
//   cout << "cumulative_flow_filepath - " << endl;
//   cout << "landsea_mask_filepath - " << endl;
//   cout << "output_number_of_outflows_filepath - " << endl;
//   cout << "output_next_cell_index_filepath - " << endl;
//   cout << "output_bifurcated_next_cell_index_filepath - " << endl;
//   cout << "grid_params_filepath - " << endl;
//   cout << "mouth_positions_filepath - " << endl;
//   cout << "next_cell_index_fieldname - " << endl;
//   cout << "cumulative_flow_fieldname - " << endl;
//   cout << "landsea_mask_fieldname - " << endl;
//   cout << "minimum_cells_from_split_to_main_mouth_string - " << endl;
//   cout << "maximum_cells_from_split_to_main_mouth_string - " << endl;
//   cout << "cumulative_flow_threshold_fraction_string - " << endl;
// }

// int main(int argc, char *argv[]){
//   cout << "ICON river bifurcation tool" << endl;
//   int opts;
//   while ((opts = getopt(argc,argv,"h")) != -1){
//     if (opts == 'h'){
//       print_help();
//       exit(EXIT_FAILURE);
//     }
//   }
//   if(argc<15) {
//     cout << "Not enough arguments" << endl;
//     print_usage();
//     cout << "Run with option -h for help" << endl;
//     exit(EXIT_FAILURE);
//   }
//   if(argc>15) {
//     cout << "Too many arguments" << endl;
//     print_usage();
//     cout << "Run with option -h for help" << endl;
//     exit(EXIT_FAILURE);
//   }
//   string next_cell_index_filepath(argv[1]);
//   string cumulative_flow_filepath(argv[2]);
//   string landsea_mask_filepath(argv[3]);
//   string output_number_of_outflows_filepath(argv[4]);
//   string output_next_cell_index_filepath(argv[5]);
//   string output_bifurcated_next_cell_index_filepath(argv[6]);
//   string grid_params_filepath(argv[7]);
//   string mouth_positions_filepath(argv[8]);
//   string next_cell_index_fieldname(argv[9]);
//   string cumulative_flow_fieldname(argv[10]);
//   string landsea_mask_fieldname(argv[11]);
//   string minimum_cells_from_split_to_main_mouth_string(argv[12]);
//   string maximum_cells_from_split_to_main_mouth_string(argv[13]);
//   string cumulative_flow_threshold_fraction_string(argv[14]);
  // cout << "Using minimum_cells_from_split_to_main_mouth= "
  //      << minimum_cells_from_split_to_main_mouth << endl;
  // cout << "Using maximum_cells_from_split_to_main_mouth= "
  //      << maximum_cells_from_split_to_main_mouth << endl;
  // cout << "Using cumulative_flow_threshold_fraction= "
  //      << cumulative_flow_threshold_fraction << endl;
void bifurcate_rivers_basic_icon_si_cython_interface(int ncells,
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
