#include <iostream>
#include <fstream>
#include <unistd.h>
#include "base/grid.hpp"
#include "algorithms/catchment_computation_algorithm.hpp"

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
//     "./Compute_Catchments_SI_Exec [next cell index filepath] [catchment numbers out filepath]"
//     << endl <<
//     "                             [grid params filepath] [next cell index fieldname]"
//     << endl <<
//     "                             ([use secondary neighbors flag ] [loop log file path]"
//     << endl <<
//     "                              [sort catchments by size flag] )" << endl;

// }

// void print_help(){
//   print_usage();
//   cout << "Generate the catchments of a set of river direction on the ICON icosahedral grid."
//        << endl;
//   cout << "Arguments:" << endl;
//   cout << "next cell index file path - Full path to the next cell index file path; the " << endl
//        << "next cell index values are the ICON equivalent of river directions" << endl;
//   cout << "catchment numbers out file path - Full path to the target output catchment"
//        << " numbers" << endl;
//   cout << "grid params file path - Full path to the grid description file for the ICON" << endl
//        << " grid being used" << endl;
//   cout << "next cell index field name - Field name of the next cell index values in the"
//        << " specified file." << endl;
//   cout << "use secondary neighbors flag - Count secondary neighbors (meeting a cell only"
//        << "at a point as neighbors (1=true,0=false)" << endl;
//   cout << "loop log file path - Full path to text file to write loops to" << endl;
//   cout << "sort catchments by size flag - renumber the catchments so they are ordered by size"
//        << "from the largest to the smallest" << endl;
//   cout << "Subcatchment list file path (optional) - Instead of generating full catchment set only"
//        << "generate the subset of catchments for the points (no necessarily river mouths) listed in"
//        << "this file" << endl;
// }

// int main(int argc, char *argv[]){
//   cout << "ICON catchment computation tool" << endl;
//   int opts;
//   bool output_loop_file = false;
//   while ((opts = getopt(argc,argv,"h")) != -1){
//     if (opts == 'h'){
//       print_help();
//       exit(EXIT_FAILURE);
//     }
//   }
//   if(argc<5) {
//     cout << "Not enough arguments" << endl;
//     print_usage();
//     cout << "Run with option -h for help" << endl;
//     exit(EXIT_FAILURE);
//   }
//   if(argc>9) {
//     cout << "Too many arguments" << endl;
//     print_usage();
//     cout << "Run with option -h for help" << endl;
//     exit(EXIT_FAILURE);
//   }
//   string next_cell_index_filepath(argv[1]);
//   string catchment_numbers_out_filepath(argv[2]);
//   string grid_params_filepath(argv[3]);
//   string next_cell_index_fieldname(argv[4]);
//   bool use_secondary_neighbors_in;
//   if (argc == 6 || argc == 7 || argc == 8 || argc == 9) {
//     string use_secondary_neighbors_string(argv[5]);
//     use_secondary_neighbors_in = bool(stoi(use_secondary_neighbors_string));
//   } else use_secondary_neighbors_in = true;
//   if (use_secondary_neighbors_in) cout << "Using secondary neighbors" << endl;
//   string loop_log_filepath;
//   if (argc == 7 || argc == 8 || argc == 9) {
//     loop_log_filepath = argv[6];
//     output_loop_file = true;
//   }
//   bool sort_catchments_by_size;
//   if (argc == 8 || argc == 9){
//     string sort_catchments_by_size_string(argv[7]);
//     sort_catchments_by_size = bool(stoi(sort_catchments_by_size_string));
//   } else sort_catchments_by_size = false;
//   bool generate_selected_subcatchments_only;
//   string subcatchment_list_filepath;
//   if (argc == 9){
//     subcatchment_list_filepath = argv[8];
//     generate_selected_subcatchments_only = true;
//     if (output_loop_file){
//       cout << "Loops not searched for when creating catchments for selected cells only" <<
//               "Loop log file path argument ignored!" << endl;
//       output_loop_file = false;
//     }
//   } else generate_selected_subcatchments_only = false;
//   ifstream ofile(catchment_numbers_out_filepath.c_str());
//   if (ofile){
//     cout << "Outfile already exists - please delete or specify a different name" << endl;
//     exit(1);
//   }
//   cout << "Loading grid parameters from:" << endl;
//   cout << grid_params_filepath << endl;
void compute_catchments_icon_si_cython_interface(int ncells,
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
