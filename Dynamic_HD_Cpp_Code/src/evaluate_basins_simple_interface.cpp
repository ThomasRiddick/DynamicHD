/*
 * evaluate_basins_simple_interface.cpp
 *
 *  Created on: November 12, 2019
 *      Author: thomasriddick
 */

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <netcdf>
#include "evaluate_basins.hpp"
#include "grid.hpp"

using namespace std;
using namespace netCDF;

string UNITS = "units";
string GRID_TYPE = "grid_type";
string LONG_NAME = "long_name";
string COORDINATES = "coordinates";
string STANDARD_NAME = "standard_name";
string BOUNDS = "bounds";

string METRESCUBED = "m^3";
string RADIAN = "radian";
string UNSTRUCTURED = "unstructured";
string LATITUDE = "latitude";
string CENTER_LATITUDE = "center latitude";
string LONGITUDE = "longitude";
string CENTER_LONGITUDE = "center longitude";
string NONE = string();

void print_usage(){

}

void print_help(){

}

void write_variable(NcFile* file_in,string var_name_in,string long_name_in,
                    string units_in,NcDim index_in,double* data_in){
  NcVar var = file_in->addVar(var_name_in,ncDouble,index_in);
  var.putAtt(LONG_NAME,"basin volume threshold for cell to overflow");
  if(!units_in.empty()) var.putAtt(UNITS,units_in);
  var.putAtt(GRID_TYPE,UNSTRUCTURED);
  var.putAtt(COORDINATES,"clat clon");
  var.putVar(data_in);
}

void write_variable(NcFile* file_in,string var_name_in,string long_name_in,
                    string units_in,NcDim index_in,int* data_in){
  NcVar var = file_in->addVar(var_name_in,ncInt,index_in);
  var.putAtt(LONG_NAME,"basin volume threshold for cell to overflow");
  if(!units_in.empty()) var.putAtt(UNITS,units_in);
  var.putAtt(GRID_TYPE,UNSTRUCTURED);
  var.putAtt(COORDINATES,"clat clon");
  var.putVar(data_in);
}

int main(int argc, char *argv[]){
  cout << "ICON basin evaluation determination tool" << endl;
  int opts;
  while ((opts = getopt(argc,argv,"h")) != -1){
    if (opts == 'h'){
      print_help();
      exit(EXIT_FAILURE);
    }
  }
  if(argc<11) {
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
  string minima_filepath(argv[1]);
  string orography_filepath(argv[2]);
  string prior_rdirs_filepath(argv[3]);
  string prior_catchments_filepath(argv[4]);
  string basin_para_out_filepath(argv[5]);
  string basin_catchment_numbers_out_filepath(argv[6]);
  string grid_params_filepath(argv[7]);
  string minima_fieldname(argv[8]);
  string orography_fieldname(argv[9]);
  string prior_rdirs_fieldname(argv[10]);
  string prior_catchments_fieldname(argv[11]);
  bool use_secondary_neighbors_in  = true;
  bool use_simple_output_format = true;
  if (argc == 13){
    string simple_output_format_string(argv[12]);
    use_simple_output_format = bool(stoi(simple_output_format_string));
  }

  cout << "Loading grid parameters from:" << endl;
  cout << grid_params_filepath << endl;
  auto grid_params_in =
    new icon_single_index_grid_params(grid_params_filepath,use_secondary_neighbors_in);
  int ncells = grid_params_in->get_ncells();
  NcFile grid_params_file(grid_params_filepath.c_str(), NcFile::read);
  NcVar cell_areas_var = grid_params_file.getVar("cell_area_p");
  auto cell_areas_in = new double[ncells];
  cell_areas_var.getVar(cell_areas_in);
  cout << "Loading minima from:" << endl;
  cout << minima_filepath << endl;
  NcFile minima_file(minima_filepath.c_str(), NcFile::read);
  NcVar minima_var = minima_file.getVar(minima_fieldname.c_str());
  auto minima_in_int = new int[ncells];
  minima_var.getVar(minima_in_int);
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
  double* clat_local = new double[ncells];
  clat.getVar(clat_local);
  double* clon_local = new double[ncells];
  clon.getVar(clon_local);
  double* clat_bnds_local = new double[ncells*3];
  clat_bnds.getVar(clat_bnds_local);
  double* clon_bnds_local = new double[ncells*3];
  clon_bnds.getVar(clon_bnds_local);
  cout << "Loading river directions from" << endl;
  cout << prior_rdirs_filepath << endl;
  NcFile prior_rdirs_file(prior_rdirs_filepath, NcFile::read);
  NcVar prior_rdirs_var = prior_rdirs_file.getVar(prior_rdirs_fieldname.c_str());
  auto prior_rdirs_in = new int[ncells];
  prior_rdirs_var.getVar(prior_rdirs_in);
  cout << "Loading prior catchments from" << endl;
  cout << prior_catchments_filepath << endl;
  NcFile prior_catchments_file(prior_catchments_filepath.c_str(), NcFile::read);
  NcVar  prior_catchments_var =
    prior_catchments_file.getVar(prior_catchments_fieldname);
  auto prior_catchments_in = new int[ncells];
  prior_catchments_var.getVar(prior_catchments_in);
  auto minima_in = new bool[ncells];
  for (int i = 0; i<ncells;i++){
    minima_in[i] = bool(minima_in_int[i]);
  }
  auto connection_volume_thresholds_in = new double[ncells];
  auto flood_volume_thresholds_in = new double[ncells];
  auto flood_next_cell_index_in = new int[ncells];
  auto connect_next_cell_index_in = new int[ncells];
  auto flood_force_merge_index_in = new int[ncells];
  auto connect_force_merge_index_in = new int[ncells];
  auto flood_redirect_index_in = new int[ncells];
  auto connect_redirect_index_in = new int[ncells];
  auto additional_flood_redirect_index_in = new int[ncells];
  auto additional_connect_redirect_index_in = new int[ncells];
  auto flood_local_redirect_in = new bool[ncells];
  auto connect_local_redirect_in = new bool[ncells];
  auto additional_flood_local_redirect_in = new bool[ncells];
  auto additional_connect_local_redirect_in = new bool[ncells];
  auto merge_points_out_int = new int[ncells];
  auto basin_catchment_numbers_in = new int[ncells];
  auto mapping_from_fine_to_coarse_grid = new int[ncells];
  for (int i = 0; i < ncells; i++) {
     mapping_from_fine_to_coarse_grid[i] = i;
  }
  icon_single_index_evaluate_basins(minima_in,orography_in,orography_in,
                                    cell_areas_in,
                                    connection_volume_thresholds_in,
                                    flood_volume_thresholds_in,
                                    prior_rdirs_in,prior_rdirs_in,
                                    prior_catchments_in,
                                    prior_catchments_in,
                                    flood_next_cell_index_in,
                                    connect_next_cell_index_in,
                                    flood_force_merge_index_in,
                                    connect_force_merge_index_in,
                                    flood_redirect_index_in,
                                    connect_redirect_index_in,
                                    additional_flood_redirect_index_in,
                                    additional_connect_redirect_index_in,
                                    flood_local_redirect_in,
                                    connect_local_redirect_in,
                                    additional_flood_local_redirect_in,
                                    additional_connect_local_redirect_in,
                                    merge_points_out_int,
                                    ncells,ncells,
                                    grid_params_in->get_neighboring_cell_indices(),
                                    grid_params_in->get_neighboring_cell_indices(),
                                    grid_params_in->get_secondary_neighboring_cell_indices(),
                                    grid_params_in->get_secondary_neighboring_cell_indices(),
                                    mapping_from_fine_to_coarse_grid,
                                    basin_catchment_numbers_in);
  cout << "Writing basin parameters to:" << endl;
  cout << basin_para_out_filepath << endl;
  NcFile* basin_para_out_file = new NcFile(basin_para_out_filepath.c_str(),
                                           NcFile::newFile);
  NcDim index = basin_para_out_file->addDim("ncells",ncells);
  NcDim vertices = basin_para_out_file->addDim("vertices",3);
  if (use_simple_output_format) {
    write_variable(basin_para_out_file,"basin_volume_threshold",
                   "basin volume threshold for cell to overflow",
                   METRESCUBED,index,flood_volume_thresholds_in);
    write_variable(basin_para_out_file,"next_cell_index",
                   "index of the next cell to fill when this one overflows",
                   NONE,index,flood_next_cell_index_in);
    write_variable(basin_para_out_file,"flood_redirect_index",
                   "index of the cell to redirect water to when this basins overflows",
                   NONE,index,flood_next_cell_index_in);
    write_variable(basin_para_out_file,"overflow_points",
                   "points where a lake overflow",
                   NONE,index,merge_points_out_int);
  } else {
    auto flood_local_redirect_in_int = new int[ncells];
    auto connect_local_redirect_in_int = new int[ncells];
    auto additional_flood_local_redirect_in_int = new int[ncells];
    auto additional_connect_local_redirect_in_int = new int[ncells];
    for (int i = 0; i<ncells;i++){
      flood_local_redirect_in_int[i] = int(flood_local_redirect_in[i]);
      connect_local_redirect_in_int[i] = int(connect_local_redirect_in[i]);
      additional_flood_local_redirect_in_int[i] = int(additional_flood_local_redirect_in[i]);
      additional_connect_local_redirect_in_int[i] = int(additional_connect_local_redirect_in[i]);
    }
    write_variable(basin_para_out_file,"flood_volume_thresholds",
                   "flood volume threshold for cell to overflow",
                   METRESCUBED,index,flood_volume_thresholds_in);
    write_variable(basin_para_out_file,"connection_volume_thresholds",
                   "connect volume threshold for cell to overflow",
                   METRESCUBED,index,connection_volume_thresholds_in);
    write_variable(basin_para_out_file,"flood_next_cell_index",
                   "index of the next cell to fill when this one overflows",
                   NONE,index,flood_next_cell_index_in);
    write_variable(basin_para_out_file,"connect_next_cell_index",
                   "index of the next cell to fill when this one connects",
                   NONE,index,connect_next_cell_index_in);
    write_variable(basin_para_out_file,"merge_points",
                   "points where a lake overflow",
                   NONE,index,merge_points_out_int);
    write_variable(basin_para_out_file,"flood_force_merge_index",
                   "flood force merge index",
                   NONE,index,flood_force_merge_index_in);
    write_variable(basin_para_out_file,"connect_force_merge_index",
                   "connect force merge index",
                   NONE,index,connect_force_merge_index_in);
    write_variable(basin_para_out_file,"flood_redirect_index",
                   "flood redirect index",
                   NONE,index,flood_redirect_index_in);
    write_variable(basin_para_out_file,"connect_redirect_index",
                   "connect redirect index",
                   NONE,index,connect_redirect_index_in);
    write_variable(basin_para_out_file,"additional_flood_redirect_index",
                   "additional flood redirect index",
                   NONE,index,additional_flood_redirect_index_in);
    write_variable(basin_para_out_file,"additional_connect_redirect_index",
                   "additional connect redirect index",
                   NONE,index,additional_connect_redirect_index_in);
    write_variable(basin_para_out_file,"flood_local_redirect",
                   "flood local redirect",
                   NONE,index,flood_local_redirect_in_int);
    write_variable(basin_para_out_file,"connect_local_redirect",
                   "connect local redirect",
                   NONE,index,connect_local_redirect_in_int);
    write_variable(basin_para_out_file,"additional_flood_local_redirect",
                   "additional flood local redirect",
                   NONE,index,additional_flood_local_redirect_in_int);
    write_variable(basin_para_out_file,"additional_connect_local_redirect",
                   "additional connect local redirect",
                   NONE,index,additional_connect_local_redirect_in_int);
    write_variable(basin_para_out_file,"lake_centers",
                   "lake centers",
                   NONE,index,minima_in_int);
  }
  NcVar clat_out = basin_para_out_file->addVar("clat",ncDouble,index);
  NcVar clon_out = basin_para_out_file->addVar("clon",ncDouble,index);
  NcVar clat_bnds_out = basin_para_out_file->addVar("clat_bnds",ncDouble,vector<NcDim>{index,vertices});
  NcVar clon_bnds_out = basin_para_out_file->addVar("clon_bnds",ncDouble,vector<NcDim>{index,vertices});
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
  basin_para_out_file->close();
  cout << "Writing basin catchments to:" << endl;
  cout << basin_catchment_numbers_out_filepath << endl;
  NcFile* basin_catchment_numbers_out_file = new NcFile(basin_catchment_numbers_out_filepath.c_str(),
                                                        NcFile::newFile);
  NcDim index_bc = basin_catchment_numbers_out_file->addDim("ncells",ncells);
  NcDim vertices_bc = basin_catchment_numbers_out_file->addDim("vertices",3);
  write_variable(basin_catchment_numbers_out_file,"basin_catchment_numbers",
                 "basin catchment numbers",
                 NONE,index,basin_catchment_numbers_in);
  NcVar clat_out_bc = basin_catchment_numbers_out_file->addVar("clat",ncDouble,index_bc);
  NcVar clon_out_bc = basin_catchment_numbers_out_file->addVar("clon",ncDouble,index_bc);
  NcVar clat_bnds_out_bc =
    basin_catchment_numbers_out_file->addVar("clat_bnds",ncDouble,vector<NcDim>{index_bc,vertices_bc});
  NcVar clon_bnds_out_bc =
    basin_catchment_numbers_out_file->addVar("clon_bnds",ncDouble,vector<NcDim>{index_bc,vertices_bc});
  clat_out_bc.putVar(clat_local);
  clat_out_bc.putAtt(STANDARD_NAME,LATITUDE);
  clat_out_bc.putAtt(LONG_NAME,CENTER_LATITUDE);
  clat_out_bc.putAtt(UNITS,RADIAN);
  clat_out_bc.putAtt(BOUNDS,"clat_bnds");
  clon_out_bc.putVar(clon_local);
  clon_out_bc.putAtt(STANDARD_NAME,LONGITUDE);
  clon_out_bc.putAtt(LONG_NAME,CENTER_LONGITUDE);
  clon_out_bc.putAtt(UNITS,RADIAN);
  clon_out_bc.putAtt(BOUNDS,"clon_bnds");
  clat_bnds_out_bc.putVar(clat_bnds_local);
  clon_bnds_out_bc.putVar(clon_bnds_local);
  basin_catchment_numbers_out_file->close();
}
