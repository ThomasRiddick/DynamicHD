#include <iostream>
#include <fstream>
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

int main(int argc, char *argv[]){
  if(argc<9) throw runtime_error("Not enough arguments");
  if(argc>11) throw runtime_error("Too many arguments");
  string orography_in_filepath(argv[1]);
  string landsea_in_filepath(argv[2]);
  string true_sinks_in_filepath(argv[3]);
  string orography_out_filepath(argv[4]);
  string grid_params_filepath(argv[5]);
  string set_ls_as_no_data_flag_string(argv[6]);
  string add_slope_string(argv[7]);
  string epsilon_string(argv[8]);
  bool fractional_landsea_mask_in = false;
  bool use_secondary_neighbors_in;
  if (argc == 10 || argc == 11) {
    string use_secondary_neighbors_string(argv[9]);
    use_secondary_neighbors_in = bool(stoi(use_secondary_neighbors_string));
    if (argc == 11) {
    string fractional_landsea_mask_string(argv[10]);
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
  icon_single_index_grid_params* grid_params_in =
    new icon_single_index_grid_params(grid_params_filepath,use_secondary_neighbors_in);
  int ncells = grid_params_in->get_ncells();
  NcFile orography_file(orography_in_filepath.c_str(), NcFile::ReadOnly);
  if ( ! orography_file.is_valid()) throw runtime_error("Invalid orography file");
  NcVar *orography_var = orography_file.get_var("cell_elevation");
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
  NcFile landsea_file(landsea_in_filepath.c_str(), NcFile::ReadOnly);
  if ( ! landsea_file.is_valid()) throw runtime_error("Invalid land-sea file");
  NcVar *landsea_var = landsea_file.get_var("lsf");
  double* landsea_in_double;
  int* landsea_in_int;
  if (fractional_landsea_mask_in) {
    landsea_in_double = new double[ncells];
    landsea_var->get(landsea_in_double,ncells);
  } else {
    landsea_in_int = new int[ncells];
    landsea_var->get(landsea_in_int,ncells);
  }
  NcFile true_sinks_file(true_sinks_in_filepath.c_str(), NcFile::ReadOnly);
  if ( ! true_sinks_file.is_valid()) throw runtime_error("Invalid true sinks file");
  NcVar *true_sinks_var = true_sinks_file.get_var("lsf");
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
    true_sinks_in[i] = false; //bool(true_sinks_in_int[i]);
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
