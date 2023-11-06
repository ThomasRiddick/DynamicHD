// #include <iostream>
// #include <fstream>
// #include <unistd.h>
#include <netcdf>
#include "base/grid.hpp"
#include "algorithms/river_direction_determination_algorithm.hpp"

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
//     "./Determine_River_Directions_SI_Exec [next cell index out file path]" <<
//     "[orography file path]" << endl;
//     cout <<
//     " [landsea file path] [true sinks filepath] [grid params file path]" << endl;
//     cout <<
//     " [orography field name] [landsea field name] [true sinks field name]" << endl;
//     cout <<
//     " [fractional landsea mask flag] [always flow to sea flag]" << endl;
//     cout <<
//     " [use secondary neighbors flag] [mark pits as true sinks flag]" << endl;
// }

// void print_help(){
//   print_usage();
//   cout << "Determine river directions on a ICON icosahedral grid using a down slope" << endl
//        << "routing. Includes the resolution of flat areas and the possibility of marking" << endl
//        << "depressions as true sink points (terminal lakes of endorheic basins." << endl;
//   cout << "Arguments:" << endl;
//   cout << "next cell index out file path - Full path to target output file for the next" << endl
//        << " cell index values; these are the ICON equivalent of river directions." << endl;
//   cout << "orography file path - Full path to the input orography file" << endl;
//   cout << "landsea file path - Full path to input landsea mask file" << endl;
//   cout << "true sinks filepath - Full path to input true sinks file" << endl;
//   cout << "grid params file path - Full path to the grid description file for the ICON" << endl
//        << " grid being used" << endl;
//   cout << "orography field name - Name of orography field within the orography file" << endl;
//   cout << "landsea field name - Name of the landsea mask field within the landsea"
//        << " mask file" << endl;
//   cout << "true sinks field name - Name of the true sinks field within the true sinks"
//        << " file" << endl;
//   cout << "fractional landsea mask flag - land sea mask expresses fraction of land"
//        << "as a floating point number (default false)" << endl;
//   cout << "always flow to sea flag - alway mark flow direction towards a neighboring" << endl
//        << " ocean point even when a neighboring land point has a lower elevation"
//        << " (default true)" << endl;
//   cout << "use secondary neighbors flag - Use the 9 additional neighbors which"
//          << "share vertices"
//          << endl << "but not edges with a cell (default: true)" << endl;
//   cout << "mark pits as true sinks flag - mark any depression found as a true sink point"
//        << " (default true)" << endl;
// }

// int main(int argc, char *argv[]){
//     cout << "ICON river direction determination tool" << endl;
//   int opts;
//   while ((opts = getopt(argc,argv,"h")) != -1){
//     if (opts == 'h'){
//       print_help();
//       exit(EXIT_FAILURE);
//     }
//   }
//   if(argc<9) {
//     cout << "Not enough arguments" << endl;
//     print_usage();
//     cout << "Run with option -h for help" << endl;
//     exit(EXIT_FAILURE);
//   }
//   if(argc>13) {
//     cout << "Too many arguments" << endl;
//     print_usage();
//     cout << "Run with option -h for help" << endl;
//     exit(EXIT_FAILURE);
//   }
//   string next_cell_index_out_filepath(argv[1]);
//   string orography_filepath(argv[2]);
//   string landsea_filepath(argv[3]);
//   string true_sinks_filepath(argv[4]);
//   string grid_params_filepath(argv[5]);
//   string orography_fieldname(argv[6]);
//   string landsea_fieldname(argv[7]);
//   string true_sinks_fieldname(argv[8]);
//   bool fractional_landsea_mask_in = false;
//   bool always_flow_to_sea_in = true;
//   bool use_secondary_neighbors_in  = true;
//   bool mark_pits_as_true_sinks_in = false;
//   if (argc >= 10){
//       string fractional_landsea_mask_string(argv[9]);
//       fractional_landsea_mask_in = bool(stoi(fractional_landsea_mask_string));
//     if (argc >= 11){
//         string always_flow_to_sea_string(argv[10]);
//         always_flow_to_sea_in = bool(stoi(always_flow_to_sea_string));
//       if (argc >= 12){
//         string use_secondary_neighbors_string(argv[11]);
//         use_secondary_neighbors_in = bool(stoi(use_secondary_neighbors_string));
//         if (argc >= 13){
//           string mark_pits_as_true_sinks_string(argv[12]);
//           mark_pits_as_true_sinks_in = bool(stoi(mark_pits_as_true_sinks_string));
//         }
//       }
//     }
//   }

void determine_river_directions_icon_si_cython_interface(int ncells,
                                                         double* orography_in,
                                                         int* landsea_in_int,
                                                         double* landsea_in_double,
                                                         int* true_sinks_in_int,
                                                         int* neighboring_cell_indices_in,
                                                         int* next_cell_index_out,
                                                         int fractional_landsea_mask_in_int,
                                                         int always_flow_to_sea_in_int,
                                                         int mark_pits_as_true_sinks_in_int){
  bool use_secondary_neighbors = true;
  int* secondary_neighboring_cell_indices = nullptr;
  bool fractional_landsea_mask_in = bool(fractional_landsea_mask_in_int);
  auto grid_params_obj =
    new icon_single_index_grid_params(ncells,
                                      neighboring_cell_indices_in,
                                      use_secondary_neighbors,
                                      secondary_neighboring_cell_indices);
  bool* landsea_in = new bool[ncells];
  if (fractional_landsea_mask_in) {
    //invert landsea mask
    for (auto i = 0; i <ncells;i++){
      if (landsea_in_double[i] < 0.5) landsea_in_double[i] = 0.0;
      else landsea_in_double[i] = 1.0;
      landsea_in[i] = ! bool(landsea_in_double[i]);
    }
  } else {
    //invert landsea mask
    for (auto i = 0; i <ncells;i++){
      landsea_in[i] = ! bool(landsea_in_int[i]);
    }
  }
  bool* true_sinks_in = new bool[ncells];
  for (auto i = 0; i < ncells;i++){
    true_sinks_in[i] = bool(true_sinks_in_int[i]);
  }
  auto alg = river_direction_determination_algorithm_icon_single_index();
  alg.setup_flags(bool(always_flow_to_sea_in_int),use_secondary_neighbors,
                  bool(mark_pits_as_true_sinks_in_int));
  alg.setup_fields(next_cell_index_out,orography_in,landsea_in,true_sinks_in,
                   grid_params_obj);
  alg.determine_river_directions();
  delete grid_params_obj;
}
