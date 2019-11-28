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

void print_usage(){

}

void print_help(){

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
  if(argc>11) {
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

  cout << "Loading grid parameters from:" << endl;
  cout << grid_params_filepath << endl;
  auto grid_params_in =
    new icon_single_index_grid_params(grid_params_filepath,use_secondary_neighbors_in);
  int ncells = grid_params_in->get_ncells();
  NcFile grid_params_file(grid_params_filepath.c_str(), NcFile::read);
  NcVar cell_areas_var = grid_params_file.getVar("");
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
                                    basin_catchment_numbers_in);
  NcFile basin_para_out_file(basin_para_out_filepath.c_str(),
                             NcFile::newFile);
  NcDim index = basin_para_out_file.addDim("ncells",ncells);
  NcDim vertices = basin_para_out_file.addDim("vertices",3);
  NcVar volume_thresholds_out_var = basin_para_out_file.addVar("volume_thresholds",
                                                                ncDouble,index);
  volume_thresholds_out_var.putAtt(LONG_NAME,"basin volume threshold for cell to overflow");
  volume_thresholds_out_var.putAtt(UNITS,METRESCUBED);
  volume_thresholds_out_var.putAtt(GRID_TYPE,UNSTRUCTURED);
  volume_thresholds_out_var.putAtt(COORDINATES,"clat clon");
  volume_thresholds_out_var.putVar(flood_volume_thresholds_in);
  NcVar next_cell_index_out_var = basin_para_out_file.addVar("next_cell_index",
                                                             ncInt,index);
  next_cell_index_out_var.putAtt(LONG_NAME,
                                 "index of the next cell to fill when this one overflows");
  next_cell_index_out_var.putAtt(GRID_TYPE,UNSTRUCTURED);
  next_cell_index_out_var.putAtt(COORDINATES,"clat clon");
  next_cell_index_out_var.putVar(flood_next_cell_index_in);
  NcVar flood_redirect_index_out_var = basin_para_out_file.addVar("flood_redirect_index",
                                                                  ncInt,index);
  flood_redirect_index_out_var.putAtt(LONG_NAME,
                                      "index of the cell to redirect water to when this basins overflows");
  flood_redirect_index_out_var.putAtt(GRID_TYPE,UNSTRUCTURED);
  flood_redirect_index_out_var.putAtt(COORDINATES,"clat clon");
  flood_redirect_index_out_var.putVar(flood_redirect_index_in);
  NcVar merge_points_out_int_var = basin_para_out_file.addVar("overflow_points",
                                                              ncInt,index);
  merge_points_out_int_var.putAtt(LONG_NAME,
                                  "points where a lake overflows");
  merge_points_out_int_var.putAtt(GRID_TYPE,UNSTRUCTURED);
  merge_points_out_int_var.putAtt(COORDINATES,"clat clon");
  merge_points_out_int_var.putVar(merge_points_out_int);
  NcVar clat_out = basin_para_out_file.addVar("clat",ncDouble,index);
  NcVar clon_out = basin_para_out_file.addVar("clon",ncDouble,index);
  NcVar clat_bnds_out = basin_para_out_file.addVar("clat_bnds",ncDouble,vector<NcDim>{index,vertices});
  NcVar clon_bnds_out = basin_para_out_file.addVar("clon_bnds",ncDouble,vector<NcDim>{index,vertices});
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
  NcFile basin_catchment_numbers_out_file(basin_catchment_numbers_out_filepath.c_str(),
                                          NcFile::newFile);
  NcDim index_bc = basin_catchment_numbers_out_file.addDim("ncells",ncells);
  NcVar basin_catchment_numbers_out_var =
    basin_catchment_numbers_out_file.addVar("basin_catchment_numbers",
                                            ncInt,index_bc);
  basin_catchment_numbers_out_var.putAtt(LONG_NAME,"basin catchment numbers");
  basin_catchment_numbers_out_var.putAtt(GRID_TYPE,UNSTRUCTURED);
  basin_catchment_numbers_out_var.putAtt(COORDINATES,"clat clon");
  basin_catchment_numbers_out_var.putVar(basin_catchment_numbers_in);
  NcVar clat_out_bc = basin_catchment_numbers_out_file.addVar("clat",ncDouble,index_bc);
  NcVar clon_out_bc = basin_catchment_numbers_out_file.addVar("clon",ncDouble,index_bc);
  NcVar clat_bnds_out_bc =
    basin_catchment_numbers_out_file.addVar("clat_bnds",ncDouble,vector<NcDim>{index_bc,vertices});
  NcVar clon_bnds_out_bc =
    basin_catchment_numbers_out_file.addVar("clon_bnds",ncDouble,vector<NcDim>{index_bc,vertices});
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
}
