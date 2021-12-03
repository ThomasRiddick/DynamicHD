#include <iostream>
#include <fstream>
#include <unistd.h>
#include <netcdf>
#include <regex>
#include "base/grid.hpp"
#include "drivers/bifurcate_rivers_basic.hpp"

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
    "./Bifurcate_Rivers_Basic_SI_Exec  [next_cell_index_filepath]" << endl;
    cout <<
    "[cumulative_flow_filepath] [landsea_mask_filepath]" << endl;
    cout <<
    "[output_number_of_outflows_filepath] [output_next_cell_index_filepath]" << endl;
    cout <<
    "[output_bifurcated_next_cell_index_filepath] [grid_params_filepath]" << endl;
    cout <<
    "[mouth_positions_filepath] [next_cell_index_fieldname]" << endl;
    cout <<
    "[cumulative_flow_fieldname] [landsea_mask_fieldname]" << endl;
    cout <<
    "[minimum_cells_from_split_to_main_mouth_string]" << endl;
    cout <<
    "[maximum_cells_from_split_to_main_mouth_string]" << endl;
    cout <<
    "[cumulative_flow_threshold_fraction_string]" << endl;
}

void print_help(){
  print_usage();
  cout << "Bifurcates selected river mouths"
       << endl;
  cout << "Arguments:" << endl;
  cout << "next_cell_index_filepath - " << endl;
  cout << "cumulative_flow_filepath - " << endl;
  cout << "landsea_mask_filepath - " << endl;
  cout << "output_number_of_outflows_filepath - " << endl;
  cout << "output_next_cell_index_filepath - " << endl;
  cout << "output_bifurcated_next_cell_index_filepath - " << endl;
  cout << "grid_params_filepath - " << endl;
  cout << "mouth_positions_filepath - " << endl;
  cout << "next_cell_index_fieldname - " << endl;
  cout << "cumulative_flow_fieldname - " << endl;
  cout << "landsea_mask_fieldname - " << endl;
  cout << "minimum_cells_from_split_to_main_mouth_string - " << endl;
  cout << "maximum_cells_from_split_to_main_mouth_string - " << endl;
  cout << "cumulative_flow_threshold_fraction_string - " << endl;
}

int main(int argc, char *argv[]){
  cout << "ICON river bifurcation tool" << endl;
  int opts;
  while ((opts = getopt(argc,argv,"h")) != -1){
    if (opts == 'h'){
      print_help();
      exit(EXIT_FAILURE);
    }
  }
  if(argc<15) {
    cout << "Not enough arguments" << endl;
    print_usage();
    cout << "Run with option -h for help" << endl;
    exit(EXIT_FAILURE);
  }
  if(argc>15) {
    cout << "Too many arguments" << endl;
    print_usage();
    cout << "Run with option -h for help" << endl;
    exit(EXIT_FAILURE);
  }
  string next_cell_index_filepath(argv[1]);
  string cumulative_flow_filepath(argv[2]);
  string landsea_mask_filepath(argv[3]);
  string output_number_of_outflows_filepath(argv[4]);
  string output_next_cell_index_filepath(argv[5]);
  string output_bifurcated_next_cell_index_filepath(argv[6]);
  string grid_params_filepath(argv[7]);
  string mouth_positions_filepath(argv[8]);
  string next_cell_index_fieldname(argv[9]);
  string cumulative_flow_fieldname(argv[10]);
  string landsea_mask_fieldname(argv[11]);
  string minimum_cells_from_split_to_main_mouth_string(argv[12]);
  string maximum_cells_from_split_to_main_mouth_string(argv[13]);
  string cumulative_flow_threshold_fraction_string(argv[14]);
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
  int minimum_cells_from_split_to_main_mouth = stoi(minimum_cells_from_split_to_main_mouth_string);
  int maximum_cells_from_split_to_main_mouth = stoi(maximum_cells_from_split_to_main_mouth_string);
  double cumulative_flow_threshold_fraction = stod(cumulative_flow_threshold_fraction_string);
  cout << "Using minimum_cells_from_split_to_main_mouth= "
       << minimum_cells_from_split_to_main_mouth << endl;
  cout << "Using maximum_cells_from_split_to_main_mouth= "
       << maximum_cells_from_split_to_main_mouth << endl;
  cout << "Using cumulative_flow_threshold_fraction= "
       << cumulative_flow_threshold_fraction << endl;
  cout << "Loading grid parameters from:" << endl;
  cout << grid_params_filepath << endl;
  bool use_secondary_neighbors = true;
  auto grid_params_obj =
    new icon_single_index_grid_params(grid_params_filepath,use_secondary_neighbors);
  int ncells = grid_params_obj->get_ncells();
  int* neighboring_cell_indices = grid_params_obj->get_neighboring_cell_indices();
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
  double* clat_local = new double[ncells];
  clat.getVar(clat_local);
  double* clon_local = new double[ncells];
  clon.getVar(clon_local);
  double* clat_bnds_local = new double[ncells*3];
  clat_bnds.getVar(clat_bnds_local);
  double* clon_bnds_local = new double[ncells*3];
  clon_bnds.getVar(clon_bnds_local);
  cout << "Loading cumulative flow from:" << endl;
  cout << cumulative_flow_filepath << endl;
  NcFile cumulative_flow_file(cumulative_flow_filepath.c_str(), NcFile::read);
  NcVar cumulative_flow_var = cumulative_flow_file.getVar(cumulative_flow_fieldname.c_str());
  int* cumulative_flow_in = new int[ncells];
  cumulative_flow_var.getVar(cumulative_flow_in);
  cout << "Loading landsea mask from:" << endl;
  cout << landsea_mask_filepath << endl;
  NcFile landsea_mask_file(landsea_mask_filepath.c_str(), NcFile::read);
  NcVar landsea_mask_var = landsea_mask_file.getVar(landsea_mask_fieldname.c_str());
  int* landsea_mask_in_int = new int[ncells];
  bool* landsea_mask_in = new bool[ncells];
  landsea_mask_var.getVar(landsea_mask_in_int);
  for (int i = 0;i < ncells;i++) {
    landsea_mask_in[i] = ! bool(landsea_mask_in_int[i]);
  }
  delete[] landsea_mask_in_int;
  int* number_of_outflows_out = new int[ncells];
  int* bifurcations_next_cell_index_out = new int[ncells*11];
  icon_single_index_bifurcate_rivers_basic(river_mouths,
                                           next_cell_index_in,
                                           bifurcations_next_cell_index_out,
                                           cumulative_flow_in,
                                           number_of_outflows_out,
                                           landsea_mask_in,
                                           cumulative_flow_threshold_fraction,
                                           minimum_cells_from_split_to_main_mouth,
                                           maximum_cells_from_split_to_main_mouth,
                                           ncells,
                                           neighboring_cell_indices);
  NcFile output_number_of_outflows_file(output_number_of_outflows_filepath.c_str(), NcFile::newFile);
  NcDim index = output_number_of_outflows_file.addDim("ncells",ncells);
  NcDim vertices = output_number_of_outflows_file.addDim("vertices",3);
  NcVar number_of_outflows_out_var = output_number_of_outflows_file.addVar("num_outflows",ncInt,index);
  number_of_outflows_out_var.putAtt(LONG_NAME,"number of cell flowing out of cell");
  number_of_outflows_out_var.putAtt(UNITS,METRES);
  number_of_outflows_out_var.putAtt(GRID_TYPE,UNSTRUCTURED);
  number_of_outflows_out_var.putAtt(COORDINATES,"clat clon");
  number_of_outflows_out_var.putVar(number_of_outflows_out);
  NcVar clat_out = output_number_of_outflows_file.addVar("clat",ncDouble,index);
  NcVar clon_out = output_number_of_outflows_file.addVar("clon",ncDouble,index);
  NcVar clat_bnds_out =
    output_number_of_outflows_file.addVar("clat_bnds",ncDouble,vector<NcDim>{index,vertices});
  NcVar clon_bnds_out =
    output_number_of_outflows_file.addVar("clon_bnds",ncDouble,vector<NcDim>{index,vertices});
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
  NcFile output_next_cell_index_file(output_next_cell_index_filepath.c_str(), NcFile::newFile);
  index = output_next_cell_index_file.addDim("ncells",ncells);
  vertices = output_next_cell_index_file.addDim("vertices",3);
  NcVar next_cell_index_out_var = output_next_cell_index_file.addVar("next_cell_index",ncInt,index);
  next_cell_index_out_var.putAtt(LONG_NAME,"next cell index");
  next_cell_index_out_var.putAtt(UNITS,METRES);
  next_cell_index_out_var.putAtt(GRID_TYPE,UNSTRUCTURED);
  next_cell_index_out_var.putAtt(COORDINATES,"clat clon");
  next_cell_index_out_var.putVar(next_cell_index_in);
  clat_out = output_next_cell_index_file.addVar("clat",ncDouble,index);
  clon_out = output_next_cell_index_file.addVar("clon",ncDouble,index);
  clat_bnds_out =
    output_next_cell_index_file.addVar("clat_bnds",ncDouble,vector<NcDim>{index,vertices});
  clon_bnds_out =
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
  NcFile output_bifurcated_next_cell_index_file(output_bifurcated_next_cell_index_filepath.c_str(), NcFile::newFile);
  index = output_bifurcated_next_cell_index_file.addDim("ncells",ncells);
  NcDim level = output_bifurcated_next_cell_index_file.addDim("nlevels",11);
  vertices = output_bifurcated_next_cell_index_file.addDim("vertices",3);
  NcVar bifurcated_next_cell_index_out_var = output_bifurcated_next_cell_index_file.addVar("bifurcated_next_cell_index",
                                                                       ncInt,
                                                                       vector<NcDim>{level,index});
  bifurcated_next_cell_index_out_var.putAtt(LONG_NAME,"next cell indices of bifurcations");
  bifurcated_next_cell_index_out_var.putAtt(UNITS,METRES);
  bifurcated_next_cell_index_out_var.putAtt(GRID_TYPE,UNSTRUCTURED);
  bifurcated_next_cell_index_out_var.putAtt(COORDINATES,"clat clon");
  bifurcated_next_cell_index_out_var.putVar(bifurcations_next_cell_index_out);
  clat_out = output_bifurcated_next_cell_index_file.addVar("clat",ncDouble,index);
  clon_out = output_bifurcated_next_cell_index_file.addVar("clon",ncDouble,index);
  clat_bnds_out =
    output_bifurcated_next_cell_index_file.addVar("clat_bnds",ncDouble,vector<NcDim>{index,vertices});
  clon_bnds_out =
    output_bifurcated_next_cell_index_file.addVar("clon_bnds",ncDouble,vector<NcDim>{index,vertices});
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
  delete grid_params_obj;
  delete[] clat_local;
  delete[] clon_local;
  delete[] clat_bnds_local;
  delete[] clon_bnds_local;
}
