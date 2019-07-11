#include <iostream>
#include <fstream>
#include <unistd.h>
#include <netcdf>
#include "grid.hpp"
#include "catchment_computation_algorithm.hpp"

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
    "./Compute_Catchments_SI_Exec [next cell index filepath] [catchment numbers out filepath]"
    << endl <<
    "                             [grid params filepath] [next cell index fieldname]" << endl;
}

void print_help(){
  print_usage();
  cout << "Generate the catchments of a set of river direction on the ICON icosahedral grid."
       << endl;
  cout << "Arguments:" << endl;
  cout << "next cell index file path - Full path to the next cell index file path; the " << endl
       << "next cell index values are the ICON equivalent of river directions" << endl;
  cout << "catchment numbers out file path - Full path to the target output catchment"
       << " numbers" << endl;
  cout << "grid params file path - Full path to the grid description file for the ICON" << endl
       << " grid being used" << endl;
  cout << "next cell index field name - Field name of the next cell index values in the"
       << " specified file." << endl;
}

int main(int argc, char *argv[]){
  cout << "ICON catchment computation tool" << endl;
  int opts;
  while ((opts = getopt(argc,argv,"h")) != -1){
    if (opts == 'h'){
      print_help();
      exit(EXIT_FAILURE);
    }
  }
  if(argc<5) {
    cout << "Not enough arguments" << endl;
    print_usage();
    cout << "Run with option -h for help" << endl;
    exit(EXIT_FAILURE);
  }
  if(argc>6) {
    cout << "Too many arguments" << endl;
    print_usage();
    cout << "Run with option -h for help" << endl;
    exit(EXIT_FAILURE);
  }
  string next_cell_index_filepath(argv[1]);
  string catchment_numbers_out_filepath(argv[2]);
  string grid_params_filepath(argv[3]);
  string next_cell_index_fieldname(argv[4]);
  bool use_secondary_neighbors_in;
  if (argc == 6) {
    string use_secondary_neighbors_string(argv[5]);
    use_secondary_neighbors_in = bool(stoi(use_secondary_neighbors_string));
  } else use_secondary_neighbors_in = true;
  ifstream ofile(catchment_numbers_out_filepath.c_str());
  if (ofile){
    cout << "Outfile already exists - please delete or specify a different name" << endl;
    exit(1);
  }
  cout << "Loading grid parameters from:" << endl;
  cout << grid_params_filepath << endl;
  auto grid_params_in =
    new icon_single_index_grid_params(grid_params_filepath,use_secondary_neighbors_in);
  int ncells = grid_params_in->get_ncells();
  cout << "Loading next cell indices from:" << endl;
  cout << next_cell_index_filepath << endl;
  NcFile next_cell_index_file(next_cell_index_filepath.c_str(), NcFile::read);
  NcVar next_cell_index_var = next_cell_index_file.getVar(next_cell_index_fieldname.c_str());
  auto next_cell_index_in = new int[ncells];
  next_cell_index_var.getVar(next_cell_index_in);
  NcVar clat = next_cell_index_file.getVar("clat");
  NcVar clon = next_cell_index_file.getVar("clon");
  NcVar clat_bnds = next_cell_index_file.getVar("clat_bnds");
  NcVar clon_bnds = next_cell_index_file.getVar("clon_bnds");
  double clat_local[ncells];
  clat.getVar(&clat_local);
  double clon_local[ncells];
  clon.getVar(&clon_local);
  double clat_bnds_local[ncells*3];
  clat_bnds.getVar(&clat_bnds_local);
  double clon_bnds_local[ncells*3];
  clon_bnds.getVar(&clon_bnds_local);
  auto catchment_numbers_out = new int[ncells];
  auto alg = catchment_computation_algorithm_icon_single_index();
  alg.setup_fields(catchment_numbers_out,
                   next_cell_index_in,grid_params_in);
  alg.compute_catchments();
  NcFile output_catchment_numbers_file(catchment_numbers_out_filepath.c_str(), NcFile::newFile);
  NcDim index = output_catchment_numbers_file.addDim("ncells",ncells);
  NcDim vertices = output_catchment_numbers_file.addDim("vertices",3);
  NcVar catchment_numbers_out_var = output_catchment_numbers_file.addVar("catchment",ncInt,index);
  catchment_numbers_out_var.putAtt(LONG_NAME,"elevation at the cell centers");
  catchment_numbers_out_var.putAtt(UNITS,METRES);
  catchment_numbers_out_var.putAtt(GRID_TYPE,UNSTRUCTURED);
  catchment_numbers_out_var.putAtt(COORDINATES,"clat clon");
  catchment_numbers_out_var.putVar(&catchment_numbers_out);
  NcVar clat_out = output_catchment_numbers_file.addVar("clat",ncDouble,index);
  NcVar clon_out = output_catchment_numbers_file.addVar("clon",ncDouble,index);
  NcVar clat_bnds_out =
    output_catchment_numbers_file.addVar("clat_bnds",ncDouble,vector<NcDim>{index,vertices});
  NcVar clon_bnds_out =
    output_catchment_numbers_file.addVar("clon_bnds",ncDouble,vector<NcDim>{index,vertices});
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
