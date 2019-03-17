#include <iostream>
#include <fstream>
#include <netcdfcpp.h>
#include "grid.hpp"
#include "river_direction_determination_algorithm.hpp"

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
  if(argc<5) throw runtime_error("Not enough arguments");
  if(argc>8) throw runtime_error("Too many arguments");
  string next_cell_index_out_filepath(argv[1]);
  string orography_filepath(argv[2]);
  string landsea_filepath(argv[3]);
  string true_sinks_filepath(argv[4]);
  string grid_params_filepath(argv[5]);
  bool fractional_landsea_mask_in = false;
  bool always_flow_to_sea_in = true;
  bool use_secondary_neighbors_in  = true;
  bool mark_pits_as_true_sinks_in = false;
  if (argc >= 7){
      string fractional_landsea_mask_string(argv[6]);
      fractional_landsea_mask_in = bool(stoi(fractional_landsea_mask_string));
    if (argc >= 8){
        string always_flow_to_sea_string(argv[7]);
        always_flow_to_sea_in = bool(stoi(always_flow_to_sea_string));
      if (argc >= 9){
        string use_secondary_neighbors_string(argv[8]);
        use_secondary_neighbors_in = bool(stoi(use_secondary_neighbors_string));
        if (argc >= 10){
          string mark_pits_as_true_sinks_string(argv[9]);
          mark_pits_as_true_sinks_in = bool(stoi(mark_pits_as_true_sinks_string));
        }
      }
    }
  }
  auto alg = river_direction_determination_algorithm_icon_single_index();
  auto grid_params_in =
    new icon_single_index_grid_params(grid_params_filepath,use_secondary_neighbors_in);
  int ncells = grid_params_in->get_ncells();
  NcFile orography_file(orography_filepath.c_str(), NcFile::ReadOnly);
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
  NcFile landsea_file(landsea_filepath.c_str(), NcFile::ReadOnly);
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
  auto landsea_in =  new bool[ncells];
  //invert landsea mask
  for (auto i = 0; i <ncells;i++){
    if (fractional_landsea_mask_in) {
      if (landsea_in_double[i] < 0.5) landsea_in_double[i] = 0.0;
      landsea_in[i] = ! bool(landsea_in_double[i]);
    } else landsea_in[i] = ! bool(landsea_in_int[i]);
  }
  NcFile true_sinks_file(true_sinks_filepath.c_str(), NcFile::ReadOnly);
  if ( ! true_sinks_file.is_valid()) throw runtime_error("Invalid true sinks file");
  NcVar *true_sinks_var = true_sinks_file.get_var("lsf");
  auto true_sinks_in_int = new int[ncells];
  true_sinks_var->get(true_sinks_in_int,ncells);
  auto true_sinks_in = new bool[ncells];
  for (auto i = 0; i < ncells;i++){
    true_sinks_in[i] = false; //bool(true_sinks_in_int[i]);
  }
  auto next_cell_index_out = new int[ncells];
  alg.setup_flags(always_flow_to_sea_in,use_secondary_neighbors_in,
                  mark_pits_as_true_sinks_in);
  alg.setup_fields(next_cell_index_out,orography_in,landsea_in,true_sinks_in,
                   grid_params_in);
  alg.determine_river_directions();
  NcFile output_next_cell_index_file(next_cell_index_out_filepath.c_str(), NcFile::New);
  if ( ! output_next_cell_index_file.is_valid())
    throw runtime_error("Can't write to output next cell index file");
  NcDim* index = output_next_cell_index_file.add_dim("ncells",ncells);
  NcDim* vertices = output_next_cell_index_file.add_dim("vertices",3);
  NcVar* next_cell_index_out_var = output_next_cell_index_file.add_var("next_cell_index",ncInt,index);
  next_cell_index_out_var->add_att(LONG_NAME,"next cell index");
  next_cell_index_out_var->add_att(UNITS,METRES);
  next_cell_index_out_var->add_att(GRID_TYPE,UNSTRUCTURED);
  next_cell_index_out_var->add_att(COORDINATES,"clat clon");
  next_cell_index_out_var->put(&next_cell_index_out[0],ncells);
  NcVar *clat_out = output_next_cell_index_file.add_var("clat",ncDouble,index);
  NcVar *clon_out = output_next_cell_index_file.add_var("clon",ncDouble,index);
  NcVar *clat_bnds_out = output_next_cell_index_file.add_var("clat_bnds",ncDouble,index,vertices);
  NcVar *clon_bnds_out = output_next_cell_index_file.add_var("clon_bnds",ncDouble,index,vertices);
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
