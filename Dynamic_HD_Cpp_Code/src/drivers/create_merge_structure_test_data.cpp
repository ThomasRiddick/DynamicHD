#include "base/grid.hpp"
#include "base/field.hpp"
#include "base/merges_and_redirects.hpp"
#include "drivers/create_merge_structure_test_data.hpp"

#include <vector>
#include <string>
#if USE_NETCDFCPP
#include <netcdf>
#endif
using namespace std;
#if USE_NETCDFCPP
using namespace netCDF;
#endif

void latlon_create_merge_structure_test_data(string merge_test_data_filepath){

  auto grid_params_in = new latlon_grid_params(20,20,false);
  int flood_index = 0;
  int connect_index = 0;
  field<int>* connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  field<int>* flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
  connect_merge_and_redirect_indices_index->set_all(-1);
  flood_merge_and_redirect_indices_index->set_all(-1);
  vector<collected_merge_and_redirect_indices*>*
    connect_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<collected_merge_and_redirect_indices*>*
    flood_merge_and_redirect_indices_vector =
      new vector<collected_merge_and_redirect_indices*>;
  vector<merge_and_redirect_indices*>* primary_merges = nullptr;
  merge_and_redirect_indices* primary_merge;
  merge_and_redirect_indices* secondary_merge;
  collected_merge_and_redirect_indices* collected_indices = nullptr;
  coords* working_coords = nullptr;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(1,2),
                                                               new latlon_coords(3,4),
                                                               true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(1,2);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge = nullptr;
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(5,6),
                                                         new latlon_coords(7,8),
                                                         false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(2,3);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(9,10),
                                                               new latlon_coords(11,12),
                                                               true);
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(13,14),
                                                         new latlon_coords(15,16),
                                                         false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(3,4);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge = nullptr;
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(17,18),
                                                         new latlon_coords(19,20),
                                                         true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(21,22),
                                                         new latlon_coords(23,24),
                                                         false);
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(4,5);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge = nullptr;
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(25,26),
                                                         new latlon_coords(27,28),
                                                         true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(29,30),
                                                         new latlon_coords(31,32),
                                                         false);
  primary_merges->push_back(primary_merge);
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(33,34),
                                                         new latlon_coords(35,36),
                                                         false);
  primary_merges->push_back(primary_merge);
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(37,38),
                                                         new latlon_coords(39,40),
                                                         false);
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(5,6);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(41,42),
                                                               new latlon_coords(43,44),
                                                               true);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(6,7);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

    secondary_merge = nullptr;
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(45,46),
                                                         new latlon_coords(47,48),
                                                         false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(7,8);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(49,50),
                                                               new latlon_coords(51,52),
                                                               true);
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(53,54),
                                                         new latlon_coords(55,56),
                                                         false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  primary_merges->push_back(primary_merge);
  primary_merge =
    create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(57,58),
                                                         new latlon_coords(59,60),
                                                         false);
  primary_merges->push_back(primary_merge);
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  flood_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(8,9);
  (*flood_merge_and_redirect_indices_index)(working_coords) = flood_index;
  flood_index++;
  delete working_coords;

  secondary_merge =
          create_latlon_merge_and_redirect_indices_for_testing(new latlon_coords(61,62),
                                                               new latlon_coords(63,64),
                                                               false);
  primary_merges =
    new vector<merge_and_redirect_indices*>;
  collected_indices = new collected_merge_and_redirect_indices(primary_merges,
                                                               secondary_merge,
                                                               latlon_merge_and_redirect_indices_factory);
  connect_merge_and_redirect_indices_vector->push_back(collected_indices);
  working_coords = new latlon_coords(2,1);
  (*connect_merge_and_redirect_indices_index)(working_coords) = connect_index;
  connect_index++;
  delete working_coords;
  merges_and_redirects working_merges_and_redirects =
          merges_and_redirects(connect_merge_and_redirect_indices_index,
                               flood_merge_and_redirect_indices_index,
                               connect_merge_and_redirect_indices_vector,
                               flood_merge_and_redirect_indices_vector,
                               grid_params_in);
  #if USE_NETCDFCPP
  pair<tuple<int,int,int>*,int*>* array_and_dimensions =
    working_merges_and_redirects.get_merges_and_redirects_as_array(true);
  cout << "Writing test data to file " << merge_test_data_filepath << endl;
  NcFile merges_and_redirects_file(merge_test_data_filepath.c_str(), NcFile::newFile);
  NcDim flood_first_index =
    merges_and_redirects_file.addDim("flood_first_index",get<0>(*array_and_dimensions->first));
  NcDim flood_second_index =
    merges_and_redirects_file.addDim("flood_second_index",get<1>(*array_and_dimensions->first));
  NcDim flood_third_index =
    merges_and_redirects_file.addDim("flood_third_index",get<2>(*array_and_dimensions->first));
  vector<NcDim> flood_dims;
  flood_dims.push_back(flood_first_index);
  flood_dims.push_back(flood_second_index);
  flood_dims.push_back(flood_third_index);
  NcVar flood_merges_and_redirects_out_var =
    merges_and_redirects_file.addVar("flood_merges_and_redirects",ncInt,flood_dims);
  flood_merges_and_redirects_out_var.putVar(array_and_dimensions->second);
  array_and_dimensions =
    working_merges_and_redirects.get_merges_and_redirects_as_array(false);
  NcDim connect_first_index =
    merges_and_redirects_file.addDim("connect_first_index",get<0>(*array_and_dimensions->first));
  NcDim connect_second_index =
    merges_and_redirects_file.addDim("connect_second_index",get<1>(*array_and_dimensions->first));
  NcDim connect_third_index =
    merges_and_redirects_file.addDim("connect_third_index",get<2>(*array_and_dimensions->first));
  vector<NcDim> connect_dims;
  connect_dims.push_back(connect_first_index);
  connect_dims.push_back(connect_second_index);
  connect_dims.push_back(connect_third_index);
  NcVar connect_merges_and_redirects_out_var =
  merges_and_redirects_file.addVar("connect_merges_and_redirects",ncInt,connect_dims);
  connect_merges_and_redirects_out_var.putVar(array_and_dimensions->second);
  #endif
}
