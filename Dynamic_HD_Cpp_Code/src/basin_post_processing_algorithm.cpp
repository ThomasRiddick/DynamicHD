
MINIMA IS STACK!

void basin_post_processing_algorithm::build_and_simplify_basins(){
  int basin_count = 1;
  while(!minima.empty()){
    basin.set_minimum_coords(minima.top());
    basin.set_basin_number(basin_count);
    minima.pop();
    basin.build_and_simplify_basin();
    basin.find_minimum_coarse_catchment_number();
    basins.push(basin);
    basin_count++;
  }
  _grid->for_all([&](coords* coords_in){
    if ((*delayed_merges_to_process)(coords)){ CONNECT OR FLOOD?
      target_basin = get_target_basin(merge_target?, CONNECT OR FLOOD?)
      target_basin->assign_new_primary_basin(basin_number? primary_basin_number?);
    }
  });
}

void basin_post_processing_algorithm::check_for_loops(){
  coarse_catchments SET ALL FALSE
  while(!basins.empty()){
    basin = basins.top();
    basins.pop();
    coarse_catchment_number = get_minimum_coarse_catchment_number()
    coarse_catchments[coarse_catchment_number - 1] = true;
    if (PRIMARY BASIN)
      redirect_coords = get_basin_non_local_redirect_coords()
      coarse_catchment_number = coarse_catchment_numbers(redirect_coords)
      if (coarse_catchments[coarse_catchment_number - 1]){
        FOLLOW RIVER DOWNSTREAM FROM MINIMUM LOOP
        if (new_coarse_catchment_number !=coarse_catchment_number){
          coarse_catchment_number = new_coarse_catchment_number
          if (! coarse_catchments[coarse_catchment_number - 1]) break;
        }
      basin.set_new_redirect(FINE COORDS FROM FOLLOWING CHANGED TO COARSE COORDS)
    }
  }
}

void basin::build_and_simplify_basin(){
  current_cell = minimum_coords->clone();
  last_relevant_cell = minimum_coords->clone();
  while(merge_type != secondary_merge){
    if (merge_type == primary_merge){
      merge_target = get_merge_target();
        IF READY TO MERGE
        target_basin = get_target_basin(merge_target, CONNECT OR FLOOD?)
        target_basin->assign_new_primary_basin(basin_number);
        ELSE
          (*delayed_merges_to_process)(merge_target) CONNECT OR FLOOD?
    } else if(height_type == connection_height &&
       merge_type == no_merge &&
       ! redirect_target(current_cell)){
      remove_current_cell_connection_from_ladder();
    }
    next_cell = get_next_cell();
    merge_type = get_cell_merge_type();
  }
  if (local_redirect) MARK LOCAL CONNECTION
  else MARK NON LOCAL CONNECTION
}
