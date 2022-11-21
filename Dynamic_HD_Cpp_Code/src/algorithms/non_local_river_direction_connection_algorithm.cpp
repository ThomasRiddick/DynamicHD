/*
 * non_local_river_direction_connection_algorithm.cpp
 *
 *  Created on: Oct 8, 2022
 *      Author: Thomas Riddick
 */

void non_local_river_direction_connection_algorithm::resolve_disconnects() {

}

void non_local_river_direction_connection_algorithm::resolve_disconnect(coords* target_coords,
                                                                        coords* starting_coords) {
  completed_cells->set_all(false);
  push_cell(starting_coords->clone());
  while (!q.empty()){
    cell* center_cell = q.top();
    q.pop();
    center_coords = center_cell->get_cell_coords();
    process_neighbors();
    delete center_cell;
  }
  coords* working_coords = connection_location;
  bool starting_coords_reached = false;
  while(! starting_coords_reached){
    transcribe_river_direction(working_coords);
    coords* new_working_coords = get_next_cell_downstream(working_coords);
    if ((*starting_coords)==(*working_coords)) starting_coords_reached = true;
    delete working_coords;
    working_coords = new_working_coords;
  }
  delete working_coords;
  calculate_new_disconnects()
}
  // For each cell disconnected in a different catchment create a initial disconnect between
  // the upstream cell and its downstream cell; then process this list linking any multiple
  // disconnects to form the final list of new disconnects

/* ************************
   ************************ */

//Some useful snippets to use for this
  while (((*working_coords) != (*target_coords)) ||
           (*working_coords_catchment == target_catchment &&
            *working_coords_cumulative_flow > target coords cumulative flow)){
}

//No longer providing option for non-diagonal neighbors only as it never gets used
void process_neighbors()
{
  neighbors_coords = completed_cells->get_neighbors_coords(center_coords,1);
  while( ! neighbors_coords->empty() ) {
    process_neighbor();
  }
  delete neighbors_coords;
}

inline void process_neighbor()
{
  coords* nbr_coords = neighbors_coords->back();
  neighbors_coords->pop_back();
  if (!( (*major_side_channel_mask)(nbr_coords) ||
         (*completed_cells)(nbr_coords) ||
         (*landsea_mask)(nbr_coords) ||
         ((*main_channel_mask)(nbr_coords) == main_channel_invalid) ||
         connection_found ) ) {
    if ((*main_channel_mask)(nbr_coords) == main_channel_valid) {
      (*number_of_outflows)(nbr_coords) += 1;
      mark_bifurcated_river_direction(nbr_coords,center_coords);
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
  } else delete nbr_coords;
}

void push_cell(coords* cell_coords){
      //Here orography is overloaded to mean path length (the cell class having been
      q.push(new landsea_cell(cell_coords));
}

void track_main_channel(coords* mouth_coords){
  cells_from_mouth = 0;
  push_cell(mouth_coords->clone());
  (*main_channel_mask)(mouth_coords) = main_channel_invalid;
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
    }
    delete center_cell;
  }
}
