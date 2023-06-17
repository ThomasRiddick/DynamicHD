/*
 * non_local_river_direction_connection_algorithm.cpp
 *
 *  Created on: Oct 8, 2022
 *      Author: Thomas Riddick
 */

ALSO NEED ACC + RDIR TREE

bool disconnect_list::check_for_disconnects_by_source(coords* source_coords,
                                                      coords* target_coords){
  for(vector<pair<coords*,coords*>>::iterator i =
      disconnects.begin();
      i != disconnects.end(); ++i){
    if ((*i)->first == source_coords){
      target_coords = (i*)->second
      return true;
    }
  }
}

vector* disconnect_list::check_for_disconnects_by_target(coords* target_coords){
  vector* source_coords = new vector()
  for(vector<pair<coords*,coords*>>::iterator i =
      disconnects.begin();
      i != disconnects.end(); ++i){
    if ((*i)->second == target_source){
      source_coords.push((i*)->first)
      return true;
    }
  }
  return source_coords
}

bool disconnect_list::merge(disconnect_list* new_disconnects){
   =
  for(vector<pair<coords*,coords*>>::iterator i =
      new_disconnects->invalidated_disconnects->begin();
      i !=  new_disconnects->invalidated_disconnects->end(); ++i){
    ITERATE THROUGH disconnects AND REMOVE IF == i
  }
}

vector<coords*> upstream_cell_tree::get_upstream_cells(coords* coords_in,
                                                       disconnect_list* disconnects_in){
  vector* upstream_cell_list = upstream_cells(coords);
  FOR EACH CELL IN UPSTREAM CELL LIST
    if check_for_disconnects_by_source (CELL){
      REMOVE CELL
    }
  vector* additional_cell_list = check_for_disconnects_by_target(coords_in);
  upstream_cell_list.insert(upstream_cell_list.end(),additional_cell_list.begin(),
                            additional_cell_list.end());
  return upstream_cell_list;
}

void non_local_river_direction_connection_algorithm::resolve_disconnects() {
  while(!disconnects.empty()){
    disconnect = disconnects.top();
    disconnect.pop()
    resolve_disconnect(disconnect->first,disconnect->second)
  }
}


void non_local_river_direction_connection_algorithm::resolve_disconnect(coords* starting_coords,
                                                                        coords* target_coords) {
  completed_cells->set_all(false);
  push_cell(starting_coords->clone());
  while (!q.empty()){
    cell* center_cell = q.top();
    center_disconnects = center_cell->get_cell_disconnect_list();
    q.pop();
    center_coords = center_cell->get_cell_coords();
    if ((*target_coords) == (*center_coords)) break;
    process_neighbors();
    delete center_cell;
  }
  new_disconnects = center_disconnects->clone();
  delete center_cell;
  coords* working_coords = target_coords->clone();
  bool starting_coords_reached = false;
  while(! starting_coords_reached){
    transfer_river_direction(working_coords);
    coords* new_working_coords = get_next_cell_downstream(working_coords);
    if ((*starting_coords)==(*working_coords)) starting_coords_reached = true;
    delete working_coords;
    working_coords = new_working_coords;
  }
  delete working_coords;
  filter_new_disconnects();
  update_variables();
}

void non_local_river_direction_connection_algorithm::mark_initial_disconnects(){

}

//No longer providing option for non-diagonal neighbors only as it never gets used
void non_local_river_direction_connection_algorithm::process_neighbors()
{
  neighbors_coords = completed_cells->get_neighbors_coords(center_coords,1);
  while( ! neighbors_coords->empty() ) {
    process_neighbor();
  }
  delete neighbors_coords;
}

inline void non_local_river_direction_connection_algorithm::process_neighbor()
{
  coords* nbr_coords = neighbors_coords->back();
  neighbors_coords->pop_back();
  if (! (OUTFLOW OR SINK)) {
    nbr_disconnects = center_disconnects->clone()
    coords* nbr_current_target = get_next_cell_downstream(nbr_coords);
    if ( nbr_current_target != center_coords) {
      // Mark disconnect in master list for removal
      if (disconnected_cells(nbr_coords)){
        nbr_connects->mark_disconnect_for_removal(nbr_coords);
      }
      // Now remove any disconnect in the nbr_disconnects list
      nbr_connects->remove_disconnect_if_present(nbr_coords);
      calculate_upstream_disconnects(nbr_coords,
                                     center_coords,
                                     nbr_disconnects);
      mark_working_river_direction(nbr_coords,center_coords);
    }
    push_cell(nbr_coords)
    BROKEN ACC + DISTANCE
  }
  (*completed_cells)(nbr_coords) = true;
  delete nbr_coords;
}

void non_local_river_direction_connection_algorithm::filter_new_disconnects(coords* working_coords){
  IN NEW RDIRS
  TRACE ONE DOWNSTREAM, SET ALL BOOL TO FALSE,MARK IN BOOL,TRACE OTHER AND SEE IF IT INTERSECTS
}

void non_local_river_direction_connection_algorithm::get_next_cell_downstream(coords* working_coords){
  coords* target_coords;
  if (! new_disconnects->check_for_disconnects_by_source(working_coords,target_coords)){
    target_coords = rdirs->get_next_cell_downstream(working_coords);
  }
  return target_coords;
}

void non_local_river_direction_connection_algorithm::
    calculate_upstream_disconnects(coords* coords_in,coords* target_coords_in,
                                   disconnect_list* disconnects_in){
  vector<coords*> upstream_cells->get_upstream_cells(coords_in,disconnects_in);
  ITERATE OVER UPSTREAM CELLS WITH ITERATOR{
    disconnects_in->add_disconnect(*i,target_coords,disconnects);
    if (disconnected_cells(*i)){
      disconnects_in->mark_disconnect_for_removal(*i);
    }
    disconnects_in->remove_disconnect_if_present(*i);
  }
}

void non_local_river_direction_connection_algorithm::update_variables(){
  disconnects->merge(new_disconnects);
  update overall disconnect list, acc and rdirs
}
