#include <iostream>
#include <fstream>
#include <unistd.h>
#include <netcdf>
#include "grid.hpp"
#include "river_direction_determination_algorithm.hpp"

using namespace std;

string UNITS = "units";
string GRID_TYPE = "grid_type";
string LONG_NAME = "long_name";
string COORDINATES = "coordinates";
string STANDARD_NAME = "standard_name";
string BOUNDS = "bounds";

string METRES = "m";
string RADIAN = "radian";
string UNSTRUCTURED = "unstructured";
string LATITUDE = "latitude";
string CENTER_LATITUDE = "center latitude";
string LONGITUDE = "longitude";
string CENTER_LONGITUDE = "center longitude";

void print_usage(){
    cout <<
    "Usage: " << endl;
    cout <<
    "./Determine_River_Directions_SI_Exec [next cell index out file path]" <<
    "[orography file path]" << endl;
    cout <<
    " [landsea file path] [true sinks filepath] [grid params file path]" << endl;
    cout <<
    " [orography field name] [landsea field name] [true sinks field name]" << endl;
    cout <<
    " [fractional landsea mask flag] [always flow to sea flag]" << endl;
    cout <<
    " [use secondary neighbors flag] [mark pits as true sinks flag]" << endl;
}

void print_help(){
  print_usage();
  cout << "Determine river directions on a ICON icosahedral grid using a down slope" << endl
       << "routing. Includes the resolution of flat areas and the possibility of marking" << endl
       << "depressions as true sink points (terminal lakes of endorheic basins." << endl;
  cout << "Arguments:" << endl;
  cout << "next cell index out file path - Full path to target output file for the next" << endl
       << " cell index values; these are the ICON equivalent of river directions." << endl;
  cout << "orography file path - Full path to the input orography file" << endl;
  cout << "landsea file path - Full path to input landsea mask file" << endl;
  cout << "true sinks filepath - Full path to input true sinks file" << endl;
  cout << "grid params file path - Full path to the grid description file for the ICON" << endl
       << " grid being used" << endl;
  cout << "orography field name - Name of orography field within the orography file" << endl;
  cout << "landsea field name - Name of the landsea mask field within the landsea"
       << " mask file" << endl;
  cout << "true sinks field name - Name of the true sinks field within the true sinks"
       << " file" << endl;
  cout << "fractional landsea mask flag - land sea mask expresses fraction of land"
       << "as a floating point number (default false)" << endl;
  cout << "always flow to sea flag - alway mark flow direction towards a neighboring" << endl
       << " ocean point even when a neighboring land point has a lower elevation"
       << " (default true)" << endl;
  cout << "use secondary neighbors flag - Use the 9 additional neighbors which"
         << "share vertices"
         << endl << "but not edges with a cell (default: true)" << endl;
  cout << "mark pits as true sinks flag - mark any depression found as a true sink point"
       << " (default true)" << endl;
}

int main(int argc, char *argv[]){
    cout << "ICON river direction determination tool" << endl;
  int opts;
  while ((opts = getopt(argc,argv,"h")) != -1){
    if (opts == 'h'){
      print_help();
      exit(EXIT_FAILURE);
    }
  }
  if(argc<9) {
    cout << "Not enough arguments" << endl;
    print_usage();
    cout << "Run with option -h for help" << endl;
    exit(EXIT_FAILURE);
  }
  if(argc>13) {
    cout << "Too many arguments" << endl;
    print_usage();
    cout << "Run with option -h for help" << endl;
    exit(EXIT_FAILURE);
  }
  string next_cell_index_out_filepath(argv[1]);
  string orography_filepath(argv[2]);
  string landsea_filepath(argv[3]);
  string true_sinks_filepath(argv[4]);
  string grid_params_filepath(argv[5]);
  string orography_fieldname(argv[6]);
  string landsea_fieldname(argv[7]);
  string true_sinks_fieldname(argv[8]);
  bool fractional_landsea_mask_in = false;
  bool always_flow_to_sea_in = true;
  bool use_secondary_neighbors_in  = true;
  bool mark_pits_as_true_sinks_in = false;
  if (argc >= 10){
      string fractional_landsea_mask_string(argv[9]);
      fractional_landsea_mask_in = bool(stoi(fractional_landsea_mask_string));
    if (argc >= 11){
        string always_flow_to_sea_string(argv[10]);
        always_flow_to_sea_in = bool(stoi(always_flow_to_sea_string));
      if (argc >= 12){
        string use_secondary_neighbors_string(argv[11]);
        use_secondary_neighbors_in = bool(stoi(use_secondary_neighbors_string));
        if (argc >= 13){
          string mark_pits_as_true_sinks_string(argv[12]);
          mark_pits_as_true_sinks_in = bool(stoi(mark_pits_as_true_sinks_string));
        }
      }
    }
  }
  auto alg = river_direction_determination_algorithm_icon_single_index();
  cout << "Loading grid parameters from:" << endl;
  cout << grid_params_filepath << endl;
  auto grid_params_in =
    new icon_single_index_grid_params(grid_params_filepath,use_secondary_neighbors_in);
  int ncells = grid_params_in->get_ncells();
  cout << "Loading orography from:" << endl;
  cout << orography_filepath << endl;
  NcFile orography_file(orography_filepath.c_str(), NcFile::read);
  NcVar orography_var = orography_file.getVar(orography_fieldname.c_str());
  auto orography_in = new double[ncells];
  orography_var.getVar(orography_in);
  NcVar clat = orography_file.getVar("clat");
  NcVar clon = orography_file.getVar("clon");
  NcVar clat_bnds = orography_file.getVar("clat_bnds");
  NcVar clon_bnds = orography_file.getVar("clon_bnds");
  double clat_local[ncells];
  clat.getVar(clat_local);
  double clon_local[ncells];
  clon.getVar(clon_local);
  double clat_bnds_local[ncells*3];
  clat_bnds.getVar(&clat_bnds_local);
  double clon_bnds_local[ncells*3];
  clon_bnds.getVar(&clon_bnds_local);
  cout << "Loading landsea mask from:" << endl;
  cout << landsea_filepath << endl;
  NcFile landsea_file(landsea_filepath.c_str(), NcFile::read);
  NcVar landsea_var = landsea_file.getVar(landsea_fieldname.c_str());
  double* landsea_in_double;
  int* landsea_in_int;
  auto landsea_in =  new bool[ncells];
  if (fractional_landsea_mask_in) {
    landsea_in_double = new double[ncells];
    landsea_var.getVar(landsea_in_double);
    //invert landsea mask
    for (auto i = 0; i <ncells;i++){
      if (landsea_in_double[i] < 0.5) landsea_in_double[i] = 0.0;
      landsea_in[i] = ! bool(landsea_in_double[i]);
    }
  } else {
    landsea_in_int = new int[ncells];
    landsea_var.getVar(landsea_in_int);
    //invert landsea mask
    for (auto i = 0; i <ncells;i++){
      landsea_in[i] = ! bool(landsea_in_int[i]);
    }
  }
  cout << "Loading true sinks from:" << endl;
  cout << true_sinks_filepath << endl;
  NcFile true_sinks_file(true_sinks_filepath.c_str(), NcFile::read);
  NcVar true_sinks_var = true_sinks_file.getVar(true_sinks_fieldname.c_str());
  auto true_sinks_in_int = new int[ncells];
  true_sinks_var.getVar(true_sinks_in_int);
  auto true_sinks_in = new bool[ncells];
  for (auto i = 0; i < ncells;i++){
    true_sinks_in[i] = bool(true_sinks_in_int[i]);
  }
  auto next_cell_index_out = new int[ncells];
  alg.setup_flags(always_flow_to_sea_in,use_secondary_neighbors_in,
                  mark_pits_as_true_sinks_in);
  alg.setup_fields(next_cell_index_out,orography_in,landsea_in,true_sinks_in,
                   grid_params_in);
  alg.determine_river_directions();
  NcFile output_next_cell_index_file(next_cell_index_out_filepath.c_str(), NcFile::newFile);
  NcDim index = output_next_cell_index_file.addDim("ncells",ncells);
  NcDim vertices = output_next_cell_index_file.addDim("vertices",3);
  NcVar next_cell_index_out_var = output_next_cell_index_file.addVar("next_cell_index",ncInt,index);
  next_cell_index_out_var.putAtt(LONG_NAME,"next cell index");
  next_cell_index_out_var.putAtt(UNITS,METRES);
  next_cell_index_out_var.putAtt(GRID_TYPE,UNSTRUCTURED);
  next_cell_index_out_var.putAtt(COORDINATES,"clat clon");
  next_cell_index_out_var.putVar(next_cell_index_out);
  NcVar clat_out = output_next_cell_index_file.addVar("clat",ncDouble,index);
  NcVar clon_out = output_next_cell_index_file.addVar("clon",ncDouble,index);
  NcVar clat_bnds_out =
    output_next_cell_index_file.addVar("clat_bnds",ncDouble,vector<NcDim>{index,vertices});
  NcVar clon_bnds_out =
    output_next_cell_index_file.addVar("clon_bnds",ncDouble,vector<NcDim>{index,vertices});
  clat_out.putVar(clat_local);
  clat_out.putAtt(STANDARD_NAME,LATITUDE);
  clat_out.putAtt(LONG_NAME,CENTER_LATITUDE);
  clat_out.putAtt(UNITS,RADIAN);
  clat_out.putAtt(BOUNDS,"clat_bnds");
  clon_out.putVar(clon_local);
  clon_out.putAtt(STANDARD_NAME,LONGITUDE);
  clon_out.putAtt(LONG_NAME,CENTER_LONGITUDE);
  clon_out.putAtt(UNITS,RADIAN);
  clon_out.putAtt(BOUNDS,"clon_bnds");
  clat_bnds_out.putVar(clat_bnds_local);
  clon_bnds_out.putVar(clon_bnds_local);
}
