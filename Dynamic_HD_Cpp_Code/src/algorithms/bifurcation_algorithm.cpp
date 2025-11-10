/*
 * bifurcation_algorithm.cpp
 *
 *  Created on: Oct 19, 2020
 *      Author: thomasriddick
 */

#include <math.h>
#include <stdexcept>
#include "algorithms/bifurcation_algorithm.hpp"
using namespace std;

bifurcation_algorithm::~bifurcation_algorithm() {
  for(map<coords*,vector<coords*>>::iterator i = river_mouths.begin();
                                             i != river_mouths.end(); ++i){
    coords* primary_mouth = i->first;
    delete primary_mouth;
    vector<coords*> secondary_mouths = i->second;
    while(! secondary_mouths.empty()){
      coords* secondary_mouth = secondary_mouths.back();
      secondary_mouths.pop_back();
      delete secondary_mouth;
    }
  }
  delete _grid; delete completed_cells; delete landsea_mask;
  delete major_side_channel_mask; delete main_channel_mask;
  delete cumulative_flow; delete number_of_outflows;
}

bifurcation_algorithm_latlon::~bifurcation_algorithm_latlon() {
  for (int i = 0; i< maximum_bifurcations;i++) delete bifurcation_rdirs[i];
  delete working_rdirs; delete master_rdirs;
}

bifurcation_algorithm_icon_single_index::
  ~bifurcation_algorithm_icon_single_index() {
  for (int i = 0; i< maximum_bifurcations;i++) delete bifurcations_next_cell_index[i];
  delete  working_next_cell_index; delete master_next_cell_index;
}

void bifurcation_algorithm::setup_fields(int* cumulative_flow_in,
                                         int* number_of_outflows_in,
                                         bool* landsea_mask_in,
                                         grid_params* grid_params_in) {
  _grid_params = grid_params_in;
  _grid = grid_factory(_grid_params);
  cumulative_flow = new field<int>(cumulative_flow_in,_grid_params);
  number_of_outflows = new field<int>(number_of_outflows_in,_grid_params);
  number_of_outflows->set_all(1);
  landsea_mask = new field<bool>(landsea_mask_in,_grid_params);
  completed_cells = new field<bool>(_grid_params);
  completed_cells->set_all(false);
  major_side_channel_mask = new field<bool>(_grid_params);
  major_side_channel_mask->set_all(false);
  main_channel_mask = new field<channel_type>(_grid_params);
  main_channel_mask->set_all(not_main_channel);
}

void bifurcation_algorithm::setup_flags(double cumulative_flow_threshold_fraction_in,
                                        int minimum_cells_from_split_to_main_mouth_in,
<<<<<<< HEAD
                                        int maximum_cells_from_split_to_main_mouth_in) {
=======
                                        int maximum_cells_from_split_to_main_mouth_in,
                                        bool remove_main_channel_in) {
>>>>>>> master
  cumulative_flow_threshold_fraction = cumulative_flow_threshold_fraction_in;
  minimum_cells_from_split_to_main_mouth = minimum_cells_from_split_to_main_mouth_in;
  maximum_cells_from_split_to_main_mouth = maximum_cells_from_split_to_main_mouth_in;
  remove_main_channel = remove_main_channel_in;
}

void bifurcation_algorithm::bifurcate_rivers(){
  for(map<coords*,vector<coords*>>::iterator i = river_mouths.begin();
                                             i != river_mouths.end(); ++i){
    bifurcate_river(*i);
  }
}


void bifurcation_algorithm::bifurcate_river(pair<coords*,vector<coords*>> river){
  main_channel_mask->set_all(not_main_channel);
  major_side_channel_mask->set_all(false);
  track_main_channel(river.first);
  vector<coords*> secondary_river_mouths = river.second;
  // separation_from_primary_mouth_comparison comparison_object =
  //   separation_from_primary_mouth_comparison(river->first);
  // sort(river_mouths.begin(),river_mouths.end(),comparison_object)
  if (remove_main_channel) is_first_distributory = true;
  for(vector<coords*>::iterator i = secondary_river_mouths.begin();
                                i != secondary_river_mouths.end(); ++i){
    find_shortest_path_to_main_channel(*i);
  }
  if (remove_main_channel) delete valid_main_channel_start_coords;
}

void bifurcation_algorithm::find_shortest_path_to_main_channel(coords* mouth_coords){
  bool river_mouth_relocated = false;
  completed_cells->set_all(false);
  connection_found = false;
  push_cell(mouth_coords->clone());
  while (!q.empty()){
    cell* center_cell = q.top();
    q.pop();
    center_coords = center_cell->get_cell_coords();
    process_neighbors(false);
    delete center_cell;
  }
  //Allow movement along coastal ocean cells till a path
  //is possible
  if (! connection_found){
    completed_cells->set_all(false);
    push_cell(mouth_coords->clone());
    while (!q.empty()){
      cell* center_cell = q.top();
      q.pop();
      center_coords = center_cell->get_cell_coords();
      process_neighbors(true);
      delete center_cell;
    }
    if (connection_found) river_mouth_relocated = true;
  }
  if (! connection_found) throw runtime_error("Unable to route distributary");
  coords* working_coords = connection_location;
  bool sea_reached = false;
  while(! sea_reached){
    transcribe_river_direction(working_coords);
    (*major_side_channel_mask)(working_coords) = true;
    coords* new_working_coords = get_next_cell_downstream(working_coords);
    if ((*new_working_coords)==(*working_coords)) sea_reached = true;
    delete working_coords;
    working_coords = new_working_coords;
  }
  if (river_mouth_relocated) {
    cout << "River mouth relocated" << endl;
    cout << "Old position: " << *mouth_coords << endl;
    cout << "New position: " << *working_coords << endl;
  }
  //Allow single sea point to recieve multiple distributory
  (*major_side_channel_mask)(working_coords) = false;
  delete working_coords;
}

//No longer providing option for non-diagonal neighbors only as it never gets used
void bifurcation_algorithm::process_neighbors(bool allow_coastal_cells)
{
  neighbors_coords = completed_cells->get_neighbors_coords(center_coords,1);
  while( ! neighbors_coords->empty() ) {
    process_neighbor(allow_coastal_cells);
  }
  delete neighbors_coords;
}

inline void bifurcation_algorithm::process_neighbor(bool allow_coastal_cells)
{
  coords* nbr_coords = neighbors_coords->back();
  neighbors_coords->pop_back();
  if (!( (*major_side_channel_mask)(nbr_coords) ||
         (*completed_cells)(nbr_coords) ||
         (*landsea_mask)(nbr_coords) ||
         ((*main_channel_mask)(nbr_coords) == main_channel_invalid) ||
         connection_found ) ) {
    bool is_connection = false;
    if (remove_main_channel && is_first_distributory) {
      is_connection = ((*nbr_coords) == (*valid_main_channel_start_coords));
    } else {
      is_connection = ((*main_channel_mask)(nbr_coords) ==
                           main_channel_valid);
    }
    if (is_connection) {
      if (remove_main_channel && is_first_distributory) {
        mark_river_direction(nbr_coords,center_coords);
        transcribe_river_direction(nbr_coords);
        is_first_distributory = false;
      } else {
        (*number_of_outflows)(nbr_coords) += 1;
        mark_bifurcated_river_direction(nbr_coords,center_coords);
      }
      connection_found = true;
      connection_location = center_coords->clone();
      while (!q.empty()) {
        cell* center_cell = q.top();
        q.pop();
        delete center_cell;
      }
      delete nbr_coords;
    } else {
      push_cell(nbr_coords);
      (*completed_cells)(nbr_coords) = true;
      mark_river_direction(nbr_coords,center_coords);
    }
  } else if ((! (*completed_cells)(nbr_coords)) &&
             (*landsea_mask)(nbr_coords) &&
             allow_coastal_cells) {
      bool is_coastal_cell = false;
      _grid->for_all_nbrs_wrapped(nbr_coords,
                                  [&](coords* second_nbr_coords){
        if (! (*landsea_mask)(second_nbr_coords)) is_coastal_cell = true;
        delete second_nbr_coords;
      });
      if (is_coastal_cell) {
        push_coastal_cell(nbr_coords);
        (*completed_cells)(nbr_coords) = true;
      } else delete nbr_coords;
  } else delete nbr_coords;
}

void bifurcation_algorithm::track_main_channel(coords* mouth_coords){
  vector<coords*> cells_to_remove_from_main_channel;
  cells_from_mouth = 0;
  push_cell(mouth_coords->clone());
  (*main_channel_mask)(mouth_coords) = main_channel_invalid;
  cells_to_remove_from_main_channel.push_back(mouth_coords->clone());
  while (!q.empty()){
    cell* center_cell = q.top();
    q.pop();
    center_coords = center_cell->get_cell_coords();
    if((*landsea_mask)(center_coords)){
      process_neighbors_track_main_channel(true);
      cumulative_flow_threshold = floor(cumulative_flow_threshold_fraction*
                                        highest_cumulative_flow_nbrs);
    }
    process_neighbors_track_main_channel(false);
    if ((*main_channel_mask)(center_coords) != not_main_channel &&
         next_upstream_cell_coords){
      cells_from_mouth++;
      (*major_side_channel_mask)(next_upstream_cell_coords) = false;
      (*main_channel_mask)(next_upstream_cell_coords) =
        (cells_from_mouth >  minimum_cells_from_split_to_main_mouth &&
         cells_from_mouth <=  maximum_cells_from_split_to_main_mouth) ?
                            main_channel_valid : main_channel_invalid;
      if (remove_main_channel) {
        if (cells_from_mouth <= minimum_cells_from_split_to_main_mouth) {
          cells_to_remove_from_main_channel.push_back(next_upstream_cell_coords->clone());
        } else if ( cells_from_mouth ==
                    minimum_cells_from_split_to_main_mouth + 1) {
          valid_main_channel_start_coords = next_upstream_cell_coords->clone();
        }
      }
    }
    delete center_cell;
  }
  for(vector<coords*>::iterator i = cells_to_remove_from_main_channel.begin();
                                    i != cells_to_remove_from_main_channel.end(); ++i){
    (*main_channel_mask)(*i) = not_main_channel;
    delete *i;
  }
}

//No longer providing option for non-diagonal neighbors only as it never gets used
void bifurcation_algorithm::
  process_neighbors_track_main_channel(bool find_highest_flow_only)
{
  highest_cumulative_flow_nbrs = 0;
  next_upstream_cell_coords = nullptr;
  neighbors_coords = completed_cells->get_neighbors_coords(center_coords,1);
  while( ! neighbors_coords->empty() ) {
    if (find_highest_flow_only) process_neighbor_find_highest_flow();
    else process_neighbor_track_main_channel();
  }
  delete neighbors_coords;
}

inline void bifurcation_algorithm::process_neighbor_track_main_channel() {
  coords* nbr_coords = neighbors_coords->back();
  neighbors_coords->pop_back();
  int cumulative_flow_at_nbr_coords = (*cumulative_flow)(nbr_coords);
  if ( cumulative_flow_at_nbr_coords > cumulative_flow_threshold &&
       cell_flows_into_cell(nbr_coords,center_coords) ){
      push_cell(nbr_coords);
      if ( ! remove_main_channel) (*major_side_channel_mask)(nbr_coords) = true;
      if( cumulative_flow_at_nbr_coords  > highest_cumulative_flow_nbrs  ) {
        highest_cumulative_flow_nbrs = cumulative_flow_at_nbr_coords;
        next_upstream_cell_coords = nbr_coords;
      }
  } else delete nbr_coords;
}

inline void bifurcation_algorithm::process_neighbor_find_highest_flow() {
  coords* nbr_coords = neighbors_coords->back();
  neighbors_coords->pop_back();
  int cumulative_flow_at_nbr_coords = (*cumulative_flow)(nbr_coords);
  if( cumulative_flow_at_nbr_coords  > highest_cumulative_flow_nbrs  ) {
    highest_cumulative_flow_nbrs = cumulative_flow_at_nbr_coords;
  }
  delete nbr_coords;
}

inline void bifurcation_algorithm_latlon::transcribe_river_direction(coords* coords_in){
  (*master_rdirs)(coords_in) = (*working_rdirs)(coords_in);
}

inline void bifurcation_algorithm_icon_single_index::transcribe_river_direction(coords* coords_in){
  (*master_next_cell_index)(coords_in) = (*working_next_cell_index)(coords_in);
}

void bifurcation_algorithm_latlon::mark_bifurcated_river_direction(coords* initial_coords,
                                                                         coords* destination_coords){
  bool successful = false;
  for (int i = 0;i<maximum_bifurcations;i++){
    if ((*bifurcation_rdirs[i])(initial_coords) == no_bifurcation_code) {
      (*bifurcation_rdirs[i])(initial_coords) =
        _grid->calculate_dir_based_rdir(initial_coords,destination_coords);
      successful = true;
      break;
    }
  }
  if (! successful) throw runtime_error("Too many bifurcations at a single point");
}

void bifurcation_algorithm_icon_single_index::mark_bifurcated_river_direction(coords* initial_coords,
                                                                         coords* destination_coords){
  auto destination_generic_1d_coords =
    static_cast<generic_1d_coords*>(destination_coords);
  for (int i = 0;i<maximum_bifurcations;i++){
    if ((*bifurcations_next_cell_index[i])(initial_coords) == no_bifurcation_code) {
      (*bifurcations_next_cell_index[i])(initial_coords) =
        destination_generic_1d_coords->get_index();
      break;
    }
  }
}

inline bool bifurcation_algorithm_latlon::cell_flows_into_cell(coords* source_coords,
                                                                     coords* destination_coords){
  coords* downstream_cell_from_rdir =
    _grid->calculate_downstream_coords_from_dir_based_rdir(source_coords,
                                                           (*master_rdirs)(source_coords));
  bool downstream_cell_equals_destination =
    ((*downstream_cell_from_rdir)==(*destination_coords));
  delete downstream_cell_from_rdir;
  return downstream_cell_equals_destination;
}

inline bool bifurcation_algorithm_icon_single_index::cell_flows_into_cell(coords* source_coords,
                                                                                coords* destination_coords){
  auto destination_generic_1d_coords =
    static_cast<generic_1d_coords*>(destination_coords);
  return ((*master_next_cell_index)(source_coords) == destination_generic_1d_coords->get_index());
}

void bifurcation_algorithm_latlon::mark_river_direction(coords* initial_coords,
                                                       coords* destination_coords){
  (*working_rdirs)(initial_coords) = _grid->calculate_dir_based_rdir(initial_coords,destination_coords);
}

void bifurcation_algorithm_icon_single_index::
     mark_river_direction(coords* initial_coords,coords* destination_coords){
  auto destination_generic_1d_coords =
    static_cast<generic_1d_coords*>(destination_coords);
  (*working_next_cell_index)(initial_coords) = destination_generic_1d_coords->get_index();
}

inline coords* bifurcation_algorithm_latlon::get_next_cell_downstream(coords* initial_coords) {
  coords* new_coords= _grid->calculate_downstream_coords_from_dir_based_rdir(initial_coords,
            (*master_rdirs)(initial_coords));
  coords* new_coords_wrapped = _grid->wrapped_coords(new_coords);
  if (! (*new_coords_wrapped == *new_coords)) delete new_coords;
  return new_coords_wrapped;
}

inline coords* bifurcation_algorithm_icon_single_index::
  get_next_cell_downstream(coords* initial_coords) {
  int next_cell_index = (*master_next_cell_index)(initial_coords);
  coords* new_coords;
  if (next_cell_index > 0) new_coords = new generic_1d_coords(next_cell_index);
  else new_coords = initial_coords->clone();
  return new_coords;
}

void bifurcation_algorithm_latlon::reset_working_flow_directions(){
  working_rdirs->set_all(0.0);
}

void bifurcation_algorithm_icon_single_index::reset_working_flow_directions(){
  working_next_cell_index->set_all(-1);
}

void bifurcation_algorithm_latlon::setup_fields(map<pair<int,int>,
                                                          vector<pair<int,int>>> river_mouths_in,
                                                      double* rdirs_in,
                                                      int* cumulative_flow_in,
                                                      int* number_of_outflows_in,
                                                      bool* landsea_mask_in,
                                                      grid_params* grid_params_in){
  bifurcation_algorithm::setup_fields(cumulative_flow_in,
                                            number_of_outflows_in,
                                            landsea_mask_in,
                                            grid_params_in);
  master_rdirs = new field<double>(rdirs_in,_grid_params);
  working_rdirs = new field<double>(_grid_params);
  working_rdirs->set_all(0.0);
  for (int i = 0; i< maximum_bifurcations; i++){
    bifurcation_rdirs[i] = new field<double>(_grid_params);
    bifurcation_rdirs[i]->set_all(no_bifurcation_code);
  }
  for(map<pair<int,int>,vector<pair<int,int>>>::iterator i = river_mouths_in.begin();
                                                         i != river_mouths_in.end(); ++i){
    vector<coords*> secondary_mouths;
    for(vector<pair<int,int>>::iterator j = i->second.begin();
                                        j != i->second.end(); ++j){
      secondary_mouths.push_back(new latlon_coords(j->first,j->second));
    }
    river_mouths.insert(pair<coords*,
                             vector<coords*>>(new latlon_coords(i->first.first,i->first.second),
                                              secondary_mouths));
  }
}

void bifurcation_algorithm_icon_single_index::setup_fields(map<int,vector<int>> river_mouths_in,
                                                                 int* next_cell_index_in,
                                                                 int* cumulative_flow_in,
                                                                 int* number_of_outflows_in,
                                                                 bool* landsea_mask_in,
                                                                 grid_params* grid_params_in){
  bifurcation_algorithm::setup_fields(cumulative_flow_in,
                                            number_of_outflows_in,
                                            landsea_mask_in,
                                            grid_params_in);
  master_next_cell_index = new field<int>(next_cell_index_in,_grid_params);
  working_next_cell_index = new field<int>(_grid_params);
  working_next_cell_index->set_all(-1);
  for (int i = 0; i< maximum_bifurcations;i++){
    bifurcations_next_cell_index[i] = new field<int>(_grid_params);
    bifurcations_next_cell_index[i]->set_all(no_bifurcation_code);
  }
  for(map<int,vector<int>>::iterator i = river_mouths_in.begin();
                                      i != river_mouths_in.end(); ++i){
    vector<coords*> secondary_mouths;
    for(vector<int>::iterator j = i->second.begin();
                                  j != i->second.end(); ++j){
      secondary_mouths.push_back(new generic_1d_coords(*j));
    }
    river_mouths.insert(pair<coords*,
                             vector<coords*>>(new generic_1d_coords(i->first),
                                              secondary_mouths));
  }
}

double* bifurcation_algorithm_latlon::get_bifurcation_rdirs(){
  int array_size = _grid->get_total_size();
  double* bifurcation_rdirs_out = new double[maximum_bifurcations*array_size];
  for (int i = 0; i< maximum_bifurcations;i++){
    double* bifurcation_rdirs_slice = bifurcation_rdirs[i]->get_array();
    for (int j = 0; j < array_size;j++){
      bifurcation_rdirs_out[j+i*array_size] = bifurcation_rdirs_slice[j];
    }
  }
  return bifurcation_rdirs_out;
}

int* bifurcation_algorithm_icon_single_index::get_bifurcation_next_cell_index(){
  long array_size = (long)_grid->get_total_size();
  int* bifurcation_next_cell_index_out = new int[(long)maximum_bifurcations*array_size];
  for (long i = 0; i< (long)maximum_bifurcations;i++){
    int* bifurcation_next_cell_index_slice = bifurcations_next_cell_index[i]->get_array();
    for (long j = 0; j < array_size;j++){
      bifurcation_next_cell_index_out[j+i*array_size] = bifurcation_next_cell_index_slice[j];
    }
  }
  return bifurcation_next_cell_index_out;
}

int bifurcation_algorithm_latlon::get_maximum_bifurcations(){
  return maximum_bifurcations;
}

int bifurcation_algorithm_icon_single_index::get_maximum_bifurcations(){
  return maximum_bifurcations;
}
