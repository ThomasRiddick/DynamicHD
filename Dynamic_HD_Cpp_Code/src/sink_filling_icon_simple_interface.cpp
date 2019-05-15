#include <iostream>
#include <fstream>
#include <unistd.h>
#include <netcdfcpp.h>
#include "grid.hpp"
#include "sink_filling_algorithm.hpp"

using namespace std;

NcToken UNITS = "units";
NcToken GRID_TYPE = "grid_type";
NcToken LONG_NAME = "long_name";
NcToken COORDINATES = "coordinates";
NcToken STANDARD_NAME = "standard_name";
NcToken BOUNDS = "bounds";

NcToken METRES = "m";
NcToken RADIAN = "radian";
NcToken UNSTRUCTURED = "unstructured";
NcToken LATITUDE = "latitude";
NcToken CENTER_LATITUDE = "center latitude";
NcToken LONGITUDE = "longitude";
NcToken CENTER_LONGITUDE = "center longitude";

void print_usage(){
    cout <<
    "Usage: " << endl;
    cout <<
    "./Fill_Sinks_Icon_SI_Exec [input orography file path] [landsea file path]" << endl;
    cout <<
    " [true sinks file path] [output orography file path] [grid parameters file path]" << endl;
    cout <<
    " [input orography field name] [input landsea mask fieldname]" << endl;
    cout <<
    " [input true sinks field name] [set land sea as no data flag] [add slope flag]" << endl;
    cout <<
    "[epsilon] [use secondary neighbors flag] [fractional landsea mask flag]" << endl;
}

void print_help(){
    print_usage();
    cout << "Fill the sinks in an orography using an accelerated" << endl;
    cout << "priority flood technique on a ICON icosohedral grid" << endl;
    cout << "Arguments:" << endl;
    cout << "input orography file path - full path to input orography file" << endl;
    cout << "landsea file path - full path to land-sea input file" << endl;
    cout << "true sinks file path - full path to true sink input file" << endl;
    cout << "output orography file path - full path to target output orography file" << endl;
    cout << "grid parameters file path - full path to file containing ICON grid parameters"
         << endl << " for resolution being used" << endl;
    cout << "input orography field name - name of input orography field within"
         << " specified file" << endl;
    cout << "input landsea mask field name - name of landsea mask field within"
         << " specified file" << endl;
    cout << "input true sinks field name - name of true sinks field within"
         << " specified file" << endl;
    cout << "set land sea as no data flag - flag to turn on and off setting"
         << " all landsea points to no data" << endl;
    cout << "add slope flag - flag to turn on and off adding a slight slope"
         << " when filling sinks" << endl;
    cout << "epsilon - additional height added to each progressive cell when"
         << " adding a slight slope" << endl;
    cout << "use secondary neighbors flag - Use the 9 additional neighbors which"
         << "share vertices"
         << endl << "but not edges with a cell (default: true)" << endl;
    cout << "fractional landsea mask - land sea mask expresses fraction of land"
         << "as a floating point number" << endl
         << "which requires conversion to a binary mask" << endl;
}

int main(int argc, char *argv[]){
  cout << "ICON sink filling tool" << endl;
  int opts;
  while ((opts = getopt(argc,argv,"h")) != -1){
    if (opts == 'h'){
      print_help();
      exit(EXIT_FAILURE);
    }
  }
  if(argc<12) {
    cout << "Not enough arguments" << endl;
    print_usage();
    cout << "Run with option -h for help" << endl;
    exit(EXIT_FAILURE);
  }
  if(argc>14) {
    cout << "Too many arguments" << endl;
    print_usage();
    cout << "Run with option -h for help" << endl;
    exit(EXIT_FAILURE);
  }
  string orography_in_filepath(argv[1]);
  string landsea_in_filepath(argv[2]);
  string true_sinks_in_filepath(argv[3]);
  string orography_out_filepath(argv[4]);
  string grid_params_filepath(argv[5]);
  string orography_in_fieldname(argv[6]);
  string landsea_in_fieldname(argv[7]);
  string true_sinks_in_fieldname(argv[8]);
  string set_ls_as_no_data_flag_string(argv[9]);
  string add_slope_string(argv[10]);
  string epsilon_string(argv[11]);
  bool fractional_landsea_mask_in = false;
  bool use_secondary_neighbors_in;
  if (argc == 13 || argc == 14) {
    string use_secondary_neighbors_string(argv[12]);
    use_secondary_neighbors_in = bool(stoi(use_secondary_neighbors_string));
    if (argc == 14) {
    string fractional_landsea_mask_string(argv[13]);
    fractional_landsea_mask_in = bool(stoi(fractional_landsea_mask_string));
    }
  } else use_secondary_neighbors_in = true;
  ifstream ofile(orography_out_filepath.c_str());
  if (ofile) {
    cout << "Outfile already exists - please delete or specify a different name" << endl;
    exit(1);
  }
  bool set_ls_as_no_data_flag = bool(stoi(set_ls_as_no_data_flag_string));
  bool add_slope_in = bool(stoi(add_slope_string));
  double epsilon_in = stod(epsilon_string);
  bool tarasov_mod = false;
  cout << "Loading grid parameters from:" << endl;
  cout << grid_params_filepath << endl;
  icon_single_index_grid_params* grid_params_in =
    new icon_single_index_grid_params(grid_params_filepath,use_secondary_neighbors_in);
  int ncells = grid_params_in->get_ncells();
  cout << "Loading orography from:" << endl;
  cout << orography_in_filepath << endl;
  NcFile orography_file(orography_in_filepath.c_str(), NcFile::ReadOnly);
  if ( ! orography_file.is_valid()) throw runtime_error("Invalid orography file");
  NcVar *orography_var = orography_file.get_var(orography_in_fieldname.c_str());
  auto orography_in = new double[ncells];
  orography_var->get(orography_in,ncells);
  NcVar *clat = orography_file.get_var("clat");
  NcVar *clon = orography_file.get_var("clon");
  NcVar *clat_bnds = orography_file.get_var("clat_bnds");
  NcVar *clon_bnds = orography_file.get_var("clon_bnds");
  double clat_local[ncells];
  clat->get(&clat_local[0],ncells);
  double clon_local[ncells];
  clon->get(&clon_local[0],ncells);
  double clat_bnds_local[ncells*3];
  clat_bnds->get(&clat_bnds_local[0],ncells,3);
  double clon_bnds_local[ncells*3];
  clon_bnds->get(&clon_bnds_local[0],ncells,3);
  cout << "Loading landsea mask from:" << endl;
  cout << landsea_in_filepath << endl;
  NcFile landsea_file(landsea_in_filepath.c_str(), NcFile::ReadOnly);
  if ( ! landsea_file.is_valid()) throw runtime_error("Invalid land-sea file");
  NcVar *landsea_var = landsea_file.get_var(landsea_in_fieldname.c_str());
  double* landsea_in_double;
  int* landsea_in_int;
  if (fractional_landsea_mask_in) {
    landsea_in_double = new double[ncells];
    landsea_var->get(landsea_in_double,ncells);
  } else {
    landsea_in_int = new int[ncells];
    landsea_var->get(landsea_in_int,ncells);
  }
  cout << "Loading true sinks from:" << endl;
  cout << true_sinks_in_filepath << endl;
  NcFile true_sinks_file(true_sinks_in_filepath.c_str(), NcFile::ReadOnly);
  if ( ! true_sinks_file.is_valid()) throw runtime_error("Invalid true sinks file");
  NcVar *true_sinks_var = true_sinks_file.get_var(true_sinks_in_fieldname.c_str());
  auto true_sinks_in_int = new int[ncells];
  true_sinks_var->get(true_sinks_in_int,ncells);
  auto landsea_in =  new bool[ncells];
  //invert landsea mask
  for (auto i = 0; i <ncells;i++){
    if (fractional_landsea_mask_in) {
      if (landsea_in_double[i] < 0.5) landsea_in_double[i] = 0.0;
      landsea_in[i] = ! bool(landsea_in_double[i]);
    } else landsea_in[i] = ! bool(landsea_in_int[i]);
  }
  auto true_sinks_in = new bool[ncells];
  for (auto i = 0; i < ncells;i++){
    true_sinks_in[i] = bool(true_sinks_in_int[i]);
  }
  auto alg1 = sink_filling_algorithm_1_icon_single_index();
  alg1.setup_flags(set_ls_as_no_data_flag,tarasov_mod,add_slope_in,epsilon_in);
  alg1.setup_fields(orography_in,landsea_in,true_sinks_in,grid_params_in);
  alg1.fill_sinks();

  NcFile output_orography_file(orography_out_filepath.c_str(), NcFile::New);
  if ( ! output_orography_file.is_valid()) throw runtime_error("Can't write to output orography file");
  NcDim* index = output_orography_file.add_dim("ncells",ncells);
  NcDim* vertices = output_orography_file.add_dim("vertices",3);
  NcVar* orography_out_var = output_orography_file.add_var("cell_elevation",ncDouble,index);
  orography_out_var->add_att(LONG_NAME,"elevation at the cell centers");
  orography_out_var->add_att(UNITS,METRES);
  orography_out_var->add_att(GRID_TYPE,UNSTRUCTURED);
  orography_out_var->add_att(COORDINATES,"clat clon");
  orography_out_var->put(&orography_in[0],ncells);
  NcVar *clat_out = output_orography_file.add_var("clat",ncDouble,index);
  NcVar *clon_out = output_orography_file.add_var("clon",ncDouble,index);
  NcVar *clat_bnds_out = output_orography_file.add_var("clat_bnds",ncDouble,index,vertices);
  NcVar *clon_bnds_out = output_orography_file.add_var("clon_bnds",ncDouble,index,vertices);
  clat_out->put(clat_local,ncells);
  clat_out->add_att(STANDARD_NAME,LATITUDE);
  clat_out->add_att(LONG_NAME,CENTER_LATITUDE);
  clat_out->add_att(UNITS,RADIAN);
  clat_out->add_att(BOUNDS,"clat_bnds");
  clon_out->put(clon_local,ncells);
  clon_out->add_att(STANDARD_NAME,LONGITUDE);
  clon_out->add_att(LONG_NAME,CENTER_LONGITUDE);
  clon_out->add_att(UNITS,RADIAN);
  clon_out->add_att(BOUNDS,"clon_bnds");
  clat_bnds_out->put(clat_bnds_local,ncells,3);
  clon_bnds_out->put(clon_bnds_local,ncells,3);
}
