#include "algorithms/flow_accumulation_algorithm.hpp"

flow_accumulation_algorithm::~flow_accumulation_algorithm(){
  delete _grid;
  delete[] dependencies->get_array();
  delete dependencies;
  delete cumulative_flow;
}

void flow_accumulation_algorithm::setup_fields(int* cumulative_flow_in,
                                               grid_params* grid_params_in) {
  _grid_params = grid_params_in;
  _grid = grid_factory(_grid_params);
  int* dependencies_data = new int[_grid->get_total_size()];
  dependencies = new field<int>(dependencies_data,
                                grid_params_in);
  cumulative_flow = new field<int>(cumulative_flow_in,grid_params_in);
  dependencies->set_all(0);
  cumulative_flow->set_all(0);
}

latlon_flow_accumulation_algorithm::
  ~latlon_flow_accumulation_algorithm(){
    delete next_cell_index_lat;
    delete next_cell_index_lon;
    delete external_data_value;
    delete flow_terminates_value;
    delete no_data_value;
    delete no_flow_value;
}

void latlon_flow_accumulation_algorithm::setup_fields(int* next_cell_index_lat_in,
                                                      int* next_cell_index_lon_in,
                                                      int* cumulative_flow_in,
                                                      grid_params* grid_params_in) {
  flow_accumulation_algorithm::setup_fields(cumulative_flow_in,
                                            grid_params_in);
  next_cell_index_lat = new field<int>(next_cell_index_lat_in,grid_params_in);
  next_cell_index_lon = new field<int>(next_cell_index_lon_in,grid_params_in);
  external_data_value = new latlon_coords(-1,-1);
  flow_terminates_value = new latlon_coords(-2,-2);
  no_data_value = new latlon_coords(-3,-3);
  no_flow_value = new latlon_coords(-4,-4);
}

icon_single_index_flow_accumulation_algorithm::
  ~icon_single_index_flow_accumulation_algorithm(){
  for (auto i : bifurcated_next_cell_index) {
    delete[] i->get_array();
    delete i;
  }
  for (auto i : bifurcation_complete) {
    delete[] i->get_array();
    delete i;
  }
  delete next_cell_index;
  delete external_data_value;
  delete flow_terminates_value;
  delete no_data_value;
  delete no_flow_value;
}

void icon_single_index_flow_accumulation_algorithm::
  setup_fields(int* next_cell_index_in,
               int* cumulative_flow_in,
               grid_params* grid_params_in,
               int* bifurcated_next_cell_index_in){
  flow_accumulation_algorithm::setup_fields(cumulative_flow_in,
                                            grid_params_in);
  max_neighbors = 12;
  no_bifurcation_value = -9;
  next_cell_index = new field<int>(next_cell_index_in,grid_params_in);
  external_data_value = new generic_1d_coords(-2);
  flow_terminates_value = new generic_1d_coords(-3);
  no_data_value = new generic_1d_coords(-4);
  no_flow_value = new generic_1d_coords(-5);
  if (bifurcated_next_cell_index_in) {
    for (long i = 0; i < max_neighbors - 1; i++) {
      int* bifurcated_next_cell_index_slice = new int[_grid->get_total_size()];
      copy(bifurcated_next_cell_index_in+(long)_grid->get_total_size()*i,
           bifurcated_next_cell_index_in+(long)_grid->get_total_size()*(i+1l),
           bifurcated_next_cell_index_slice);
      bifurcated_next_cell_index.push_back(
        new field<int>(bifurcated_next_cell_index_slice,grid_params_in));
      bool* bifurcation_complete_slice = new bool[_grid->get_total_size()];
      bifurcation_complete.push_back(
        new field<bool>(bifurcation_complete_slice,grid_params_in));
      bifurcation_complete[i]->set_all(false);
    }
  }
}

void flow_accumulation_algorithm::generate_cumulative_flow(bool set_links){
    _grid->for_all([&](coords* coords_in){
      set_dependencies(coords_in);
      delete coords_in;
    });
    _grid->for_all([&](coords* coords_in){
      add_cells_to_queue(coords_in);
      delete coords_in;
    });
    process_queue();
    if (search_for_loops){
      _grid->for_all([&](coords* coords_in){
         check_for_loops(coords_in);
         delete coords_in;
      });
    }
    if (set_links) {
      _grid->for_all_edge_cells([&](coords* coords_in){
        follow_paths(coords_in);
      });
    }
}

void flow_accumulation_algorithm::set_dependencies(coords* coords_in){
  if ( ! ((*coords_in)==(*get_no_data_value()) ||
              (*coords_in)==(*get_no_flow_value()) ||
              (*coords_in)==(*get_flow_terminates_value()) )) {
    coords* target_coords = get_next_cell_coords(coords_in);
    if ((*target_coords)==(*get_no_data_value()) ) {
      (*cumulative_flow)(coords_in) = acc_no_data_value;
    } else if ((*target_coords)==(*get_no_flow_value())) {
      (*cumulative_flow)(coords_in) = acc_no_flow_value;
    } else if ((*target_coords)!=(*get_flow_terminates_value())) {
      coords* target_of_target_coords = get_next_cell_coords(target_coords);
      if ( ! ((*target_of_target_coords)==(*get_no_flow_value()) ||
              (*target_of_target_coords)==(*get_no_data_value())) ) {
        int dependency = (*dependencies)(target_coords);
        (*dependencies)(target_coords) = dependency + 1;
      }
      delete target_of_target_coords;
    }
    delete target_coords;
  }
}

void flow_accumulation_algorithm::add_cells_to_queue(coords* coords_in){
  coords* target_coords = get_next_cell_coords(coords_in);
  int dependency = (*dependencies)(coords_in);
  if (  dependency == 0 &&
       ((*target_coords) != (*get_no_data_value()) ) ) {
    q.push(coords_in->clone());
    if ( (*target_coords) != (*get_flow_terminates_value())) {
      (*cumulative_flow)(coords_in) = 1;
    }
  }
  delete target_coords;
}

void flow_accumulation_algorithm::process_queue(){
  while ( ! q.empty() ) {
    coords* current_coords = q.front();
    coords* target_coords = get_next_cell_coords(current_coords);
    if ( (*target_coords)==(*get_no_data_value()) ||
         (*target_coords)==(*get_no_flow_value()) ||
         (*target_coords)==(*get_flow_terminates_value())) {
      q.pop();
      delete current_coords;
      delete target_coords;
      continue;
    }
    coords* target_of_target_coords = get_next_cell_coords(target_coords);
    if ( (*target_of_target_coords) == (*get_no_data_value())) {
      q.pop();
      delete current_coords;
      delete target_coords;
      delete target_of_target_coords;
      continue;
    }
    delete target_of_target_coords;
    int dependency = (*dependencies)(target_coords) - 1;
    (*dependencies)(target_coords) = dependency;
    int cumulative_flow_target_coords = (*cumulative_flow)(target_coords);
    int cumulative_flow_current_coords = (*cumulative_flow)(current_coords);
    q.pop();
    delete current_coords;
    if (dependency == 0 &&
        (*target_coords) != (*get_flow_terminates_value())) {
      (*cumulative_flow)(target_coords) = cumulative_flow_target_coords +
                                          cumulative_flow_current_coords + 1;
      q.push(target_coords);
    } else {
      (*cumulative_flow)(target_coords) = cumulative_flow_target_coords +
                                          cumulative_flow_current_coords;
      delete target_coords;
    }
  }
}

void flow_accumulation_algorithm::check_for_loops(coords* coords_in){
  int dependency = (*dependencies)(coords_in);
  if (dependency != 0) {
    label_loop(coords_in);
  }
}

void flow_accumulation_algorithm::follow_paths(coords* initial_coords){
  coords* current_coords = initial_coords->clone();
  coords* target_coords = nullptr;
  int coords_index = generate_coords_index(initial_coords);
  while (true) {
    if ( (*target_coords) == (*get_no_data_value()) ||
         (*target_coords) == (*get_no_flow_value()) ) {
      assign_coords_to_link_array(coords_index,get_flow_terminates_value());
      break;
    }
    target_coords = get_next_cell_coords(current_coords);
    if ( _grid->outside_limits(target_coords) ) {
      if ( (*current_coords) == (*initial_coords) ) {
         assign_coords_to_link_array(coords_index,get_external_flow_value());
      } else {
         assign_coords_to_link_array(coords_index,current_coords);
      }
      break;
    }
    current_coords = target_coords;
  }
  delete current_coords;
}

void flow_accumulation_algorithm::label_loop(coords* start_coords){
    coords* current_coords = start_coords->clone();
    while (true) {
      (*dependencies)(current_coords) = 0;
      (*cumulative_flow)(current_coords) = 0;
      coords* new_current_coords = get_next_cell_coords(current_coords);
      delete current_coords;
      if ((*new_current_coords) == (*start_coords)) {
        delete new_current_coords;
        break;
      }
      current_coords = new_current_coords;
    }
}

coords* flow_accumulation_algorithm::get_external_flow_value() {
    return external_data_value;
}

coords* flow_accumulation_algorithm::get_flow_terminates_value() {
    return flow_terminates_value;
}

coords* flow_accumulation_algorithm::get_no_data_value() {
    return no_data_value;
}

coords* flow_accumulation_algorithm::get_no_flow_value() {
    return no_flow_value;
}

coords* latlon_flow_accumulation_algorithm::
  get_next_cell_coords(coords* coords_in) {
  int next_cell_coords_lat = (*next_cell_index_lat)(coords_in);
  int next_cell_coords_lon = (*next_cell_index_lon)(coords_in);
  return new latlon_coords(next_cell_coords_lat,
                           next_cell_coords_lon);
}

int latlon_flow_accumulation_algorithm::
  generate_coords_index(coords* coords_in) {
  int index;
  latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
  int relative_lat = 1 + latlon_coords_in->get_lat() - tile_min_lat;
  int relative_lon = 1 + latlon_coords_in->get_lon() - tile_min_lon;
  if ( relative_lat == 1 ) {
    index = relative_lon;
  } else if ( relative_lat < tile_width_lat &&
            relative_lat > 1 ) {
    if ( relative_lon == 1 ) {
      index = 2*tile_width_lon + 2*tile_width_lat - relative_lat - 2;
    } else if ( relative_lon == tile_width_lon ) {
      index = tile_width_lon + relative_lat - 1;
    } else {
      throw runtime_error("trying to generate the index of an non edge cell");
    }
  } else if ( relative_lat == tile_width_lat ) {
    index = tile_width_lat + 2*tile_width_lon - relative_lon - 1;
  } else {
      throw runtime_error("trying to generate the index of an non edge cell");
  }
  return index;
}


void latlon_flow_accumulation_algorithm::
  assign_coords_to_link_array(int coords_index,coords* coords_in){
  links[coords_index] = coords_in;
}

coords* icon_single_index_flow_accumulation_algorithm::
  get_next_cell_coords(coords* coords_in) {
  int next_cell_coords_index = (*next_cell_index)(coords_in);
  return new generic_1d_coords(next_cell_coords_index);
}

int icon_single_index_flow_accumulation_algorithm::
  generate_coords_index(coords* coords_in) {
    throw runtime_error("Function generate_coords_index not yet implemented for icon grid");
}

void icon_single_index_flow_accumulation_algorithm::
  assign_coords_to_link_array(int coords_index,coords* coords_in){
  links[coords_index] = coords_in;
}

void flow_accumulation_algorithm::update_bifurcated_flows(){
  _grid->for_all([&](coords* coords_in){
    check_for_bifurcations_in_cell(coords_in);
    delete coords_in;
  });
}

void flow_accumulation_algorithm::
  check_for_bifurcations_in_cell(coords* coords_in){
    if (is_bifurcated(coords_in)) {
      for (int i=0; i < max_neighbors - 1; i++) {
        if (is_bifurcated(coords_in,i)) {
          coords* target_coords = get_next_cell_bifurcated_coords(coords_in,i);
          update_bifurcated_flow(target_coords,
                                 (*cumulative_flow)(coords_in));
          (*bifurcation_complete[i])(coords_in) = true;
          delete target_coords;
        }
      }
    }
}

void flow_accumulation_algorithm::
  update_bifurcated_flow(coords* initial_coords,
                         int additional_accumulated_flow){
  coords* target_coords = nullptr;
  coords* current_coords = initial_coords->clone();
  while (true) {
    int cumulative_flow_value = (*cumulative_flow)(current_coords);
    (*cumulative_flow)(current_coords) = cumulative_flow_value +
                                         additional_accumulated_flow;
    if (is_bifurcated(current_coords)) {
      for (int i=0;i < max_neighbors - 1; i++) {
          if (is_bifurcated(current_coords,i) &&
              (*bifurcation_complete[i])(current_coords)) {
            target_coords = get_next_cell_bifurcated_coords(current_coords,i);
            update_bifurcated_flow(target_coords,
                                   additional_accumulated_flow);
            delete target_coords;
        }
      }
    }
    target_coords = get_next_cell_coords(current_coords);
    delete current_coords;
    current_coords = target_coords;
    if ((*current_coords)==(*get_flow_terminates_value())) break;
  }
  delete current_coords;
}

bool latlon_flow_accumulation_algorithm::is_bifurcated(coords* coords_in,
                                                       int layer_in){
  if (layer_in != -1) {
    int bifurcated_next_cell_index_lat_value =
      (*bifurcated_next_cell_index_lat[layer_in])(coords_in);
    return (bifurcated_next_cell_index_lat_value
            != no_bifurcation_value);
  } else {
    for (int i = 0; i < max_neighbors - 1; i++) {
      int bifurcated_next_cell_index_lat_value =
        (*bifurcated_next_cell_index_lat[i])(coords_in);
      if (bifurcated_next_cell_index_lat_value
          != no_bifurcation_value) return true;
    }
  }
  return false;
}

bool icon_single_index_flow_accumulation_algorithm::
  is_bifurcated(coords* coords_in,int layer_in) {
  if (layer_in != -1) {
    int bifurcated_next_cell_index_value =
      (*bifurcated_next_cell_index[layer_in])(coords_in);
    return (bifurcated_next_cell_index_value
            != no_bifurcation_value);
  } else {
    for (int i = 0;i < max_neighbors - 1;i++) {
      int bifurcated_next_cell_index_value =
        (*bifurcated_next_cell_index[i])(coords_in);
      if (bifurcated_next_cell_index_value
          != no_bifurcation_value) return true;
    }
  }
  return false;
}

coords* latlon_flow_accumulation_algorithm::
  get_next_cell_bifurcated_coords(coords* coords_in,
                                  int layer_in) {
  int bifurcated_next_cell_coords_lat =
      (*bifurcated_next_cell_index_lat[layer_in])(coords_in);
  int bifurcated_next_cell_coords_lon =
      (*bifurcated_next_cell_index_lon[layer_in])(coords_in);
  return new latlon_coords(bifurcated_next_cell_coords_lat,
                           bifurcated_next_cell_coords_lon);
}

coords* icon_single_index_flow_accumulation_algorithm::
  get_next_cell_bifurcated_coords(coords* coords_in,
                                  int layer_in) {
  int bifurcated_next_cell_coords_index =
      (*bifurcated_next_cell_index[layer_in])(coords_in);
  return new generic_1d_coords(bifurcated_next_cell_coords_index);
}
