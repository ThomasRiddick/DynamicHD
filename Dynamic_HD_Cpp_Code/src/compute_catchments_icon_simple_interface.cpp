#include <iostream>
#include <fstream>
#include <netcdfcpp.h>
#include "grid.hpp"
#include "catchment_computation_algorithm.hpp"

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
  if(argc<4) throw runtime_error("Not enough arguments");
  if(argc>5) throw runtime_error("Too many arguments");
  string next_cell_index_filepath(argv[1]);
  string catchment_numbers_out_filepath(argv[2]);
  string grid_params_filepath(argv[3]);
  bool use_secondary_neighbors_in;
  if (argc == 5) {
    string use_secondary_neighbors_string(argv[4]);
    use_secondary_neighbors_in = bool(stoi(use_secondary_neighbors_string));
  } else use_secondary_neighbors_in = true;
  ifstream ofile(catchment_numbers_out_filepath.c_str());
  if (ofile){
    cout << "Outfile already exists - please delete or specify a different name" << endl;
    exit(1);
  }
  auto grid_params_in =
    new icon_single_index_grid_params(grid_params_filepath,use_secondary_neighbors_in);
  int ncells = grid_params_in->get_ncells();
  NcFile next_cell_index_file(next_cell_index_filepath.c_str(), NcFile::ReadOnly);
  if ( ! next_cell_index_file.is_valid()) throw runtime_error("Invalid river directions file");
  NcVar *next_cell_index_var = next_cell_index_file.get_var("next_cell_index");
  auto next_cell_index_in = new int[ncells];
  next_cell_index_var->get(next_cell_index_in,ncells);
  NcVar *clat = next_cell_index_file.get_var("clat");
  NcVar *clon = next_cell_index_file.get_var("clon");
  NcVar *clat_bnds = next_cell_index_file.get_var("clat_bnds");
  NcVar *clon_bnds = next_cell_index_file.get_var("clon_bnds");
  double clat_local[ncells];
  clat->get(&clat_local[0],ncells);
  double clon_local[ncells];
  clon->get(&clon_local[0],ncells);
  double clat_bnds_local[ncells*3];
  clat_bnds->get(&clat_bnds_local[0],ncells,3);
  double clon_bnds_local[ncells*3];
  clon_bnds->get(&clon_bnds_local[0],ncells,3);
  auto catchment_numbers_out = new int[ncells];
  auto alg = catchment_computation_algorithm_icon_single_index();
  alg.setup_fields(catchment_numbers_out,
                   next_cell_index_in,grid_params_in);
  alg.compute_catchments();
  NcFile output_catchment_numbers_file(catchment_numbers_out_filepath.c_str(), NcFile::New);
  if ( ! output_catchment_numbers_file.is_valid()) throw runtime_error("Can't write to output catchment_numbers file");
  NcDim* index = output_catchment_numbers_file.add_dim("ncells",ncells);
  NcDim* vertices = output_catchment_numbers_file.add_dim("vertices",3);
  NcVar* catchment_numbers_out_var = output_catchment_numbers_file.add_var("cell_elevation",ncInt,index);
  catchment_numbers_out_var->add_att(LONG_NAME,"elevation at the cell centers");
  catchment_numbers_out_var->add_att(UNITS,METRES);
  catchment_numbers_out_var->add_att(GRID_TYPE,UNSTRUCTURED);
  catchment_numbers_out_var->add_att(COORDINATES,"clat clon");
  catchment_numbers_out_var->put(&catchment_numbers_out[0],ncells);
  NcVar *clat_out = output_catchment_numbers_file.add_var("clat",ncDouble,index);
  NcVar *clon_out = output_catchment_numbers_file.add_var("clon",ncDouble,index);
  NcVar *clat_bnds_out = output_catchment_numbers_file.add_var("clat_bnds",ncDouble,index,vertices);
  NcVar *clon_bnds_out = output_catchment_numbers_file.add_var("clon_bnds",ncDouble,index,vertices);
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
