#include "algorithms/basin_post_processing_algorithm.hpp"

void basin_post_processing_algorithm::build_and_simplify_basins(){
  int basin_count = 1;
  while(!minima_q.empty()){
    coords* minimum = minima_q.top();
    minima_q.pop();
    if(!((*completed_cells)(minimum))){
      current_basin = create_basin();
      current_basin->set_minimum_coords(minimum);
      current_basin->set_basin_number(basin_count);
      basin_count = current_basin->build_and_simplify_basin(basin_count);
      current_basin->find_minimum_coarse_catchment_number();
      basins.push_back(current_basin);
    }
  }
}

void basin_post_processing_algorithm::check_for_loops(){
  fill_n(processed_basins,num_basins,false);
  while(!basins.empty()){
    fill_n(coarse_catchments,num_catchments,false);
    current_basin = basins.back();
    basins.pop_back();
    if (processed_basins[current_basin->get_basin_number() - 1]) break;
    current_basin->check_basin_for_loops();
  }
}

void basin::initialise_basin(field<int>* basin_numbers_in,
                             field<int>* coarse_catchment_numbers_in,
                             field<bool>* completed_cells_in,
                             field<bool>* redirect_targets_in,
                             field<bool>* minima_in,
                             field<bool>* use_flood_height_only_in,
                             field<merge_types>* merge_points_in,
                             bool* processed_basins_in,
                             bool* coarse_catchments_in,
                             grid* _coarse_grid_in,
                             grid* _fine_grid_in,
                             grid_params* coarse_grid_params_in,
                             grid_params* fine_grid_params_in){
  basin_numbers = basin_numbers_in;
  coarse_catchment_numbers = coarse_catchment_numbers_in;
  completed_cells = completed_cells_in;
  redirect_targets =  redirect_targets_in;
  minima = minima_in;
  use_flood_height_only = use_flood_height_only_in;
  merge_points = merge_points_in;
  processed_basins = processed_basins_in;
  coarse_catchments = coarse_catchments_in;
  _coarse_grid = _coarse_grid_in;
  _fine_grid = _fine_grid_in;
  coarse_grid_params = coarse_grid_params_in;
  fine_grid_params = fine_grid_params_in;
}

void latlon_basin::initialise_basin(field<int>* basin_numbers_in,
                                    field<int>* coarse_catchment_numbers_in,
                                    field<bool>* completed_cells_in,
                                    field<bool>* redirect_targets_in,
                                    field<bool>* minima_in,
                                    field<bool>* use_flood_height_only_in,
                                    field<merge_types>* merge_points_in,
                                    bool* processed_basins_in,
                                    bool* coarse_catchments_in,
                                    grid* _coarse_grid_in,
                                    grid* _fine_grid_in,
                                    grid_params* coarse_grid_params_in,
                                    grid_params* fine_grid_params_in,
                                    field<int>* fine_rdirs_in,
                                    field<int>* coarse_rdirs_in,
                                    field<int>* connect_redirect_lat_index_in,
                                    field<int>* connect_redirect_lon_index_in,
                                    field<int>* flood_redirect_lat_index_in,
                                    field<int>* flood_redirect_lon_index_in,
                                    field<int>* connect_next_cell_lat_index_in,
                                    field<int>* connect_next_cell_lon_index_in,
                                    field<int>* flood_next_cell_lat_index_in,
                                    field<int>* flood_next_cell_lon_index_in,
                                    field<int>* connect_force_merge_lat_index_in,
                                    field<int>* connect_force_merge_lon_index_in,
                                    field<int>* flood_force_merge_lat_index_in,
                                    field<int>* flood_force_merge_lon_index_in,
                                    int* basin_minimum_lats_in,
                                    int* basin_minimum_lons_in){
  basin::initialise_basin(basin_numbers_in,
                          coarse_catchment_numbers_in,
                          completed_cells_in,
                          redirect_targets_in,
                          minima_in,
                          use_flood_height_only_in,
                          merge_points_in,
                          processed_basins_in,
                          coarse_catchments_in,
                          _coarse_grid_in,
                          _fine_grid_in,
                          coarse_grid_params_in,
                          fine_grid_params_in);
  fine_rdirs = fine_rdirs_in;
  coarse_rdirs = coarse_rdirs_in;
  connect_redirect_lat_index = connect_redirect_lat_index_in;
  connect_redirect_lon_index = connect_redirect_lon_index_in,
  flood_redirect_lat_index = flood_redirect_lat_index_in;
  flood_redirect_lon_index = flood_redirect_lon_index_in;
  connect_next_cell_lat_index = connect_next_cell_lat_index_in;
  connect_next_cell_lon_index = connect_next_cell_lon_index_in;
  flood_next_cell_lat_index = flood_next_cell_lat_index_in;
  flood_next_cell_lon_index = flood_next_cell_lon_index_in;
  connect_force_merge_lat_index = connect_force_merge_lat_index_in;
  connect_force_merge_lon_index = connect_force_merge_lon_index_in;
  flood_force_merge_lat_index = flood_force_merge_lat_index_in;
  flood_force_merge_lon_index = flood_force_merge_lon_index_in;
  basin_minimum_lats = basin_minimum_lats_in;
  basin_minimum_lons = basin_minimum_lons_in;
}

void basin::check_basin_for_loops(){
  basin* current_basin = this;
  while(true){
    processed_basins[current_basin->get_basin_number() - 1] = true;
    int coarse_catchment_number = current_basin->get_minimum_coarse_catchment_number();
    coarse_catchments[coarse_catchment_number - 1] = true;
    coords* redirect_coords = get_basin_redirect_coords();
    if (current_basin->get_local_redirect()) {
      current_basin = get_basin_at_coords(redirect_coords);
      coarse_catchment_number = current_basin->get_minimum_coarse_catchment_number();
    } else {
      coarse_catchment_number = (*coarse_catchment_numbers)(redirect_coords);
      coords* next_fine_cell_downstream;
      coords* new_coarse_cell_downstream;
      coords* old_coarse_cell_downstream = nullptr;
      if(coarse_catchments[coarse_catchment_number - 1]){
        while(coarse_catchments[coarse_catchment_number - 1]){
          next_fine_cell_downstream = find_next_fine_cell_downstream(next_fine_cell_downstream);
          new_coarse_cell_downstream = _coarse_grid->convert_fine_coords(next_fine_cell_downstream,
                                                                         fine_grid_params);
          if ((*new_coarse_cell_downstream) != (*old_coarse_cell_downstream)){
            coarse_catchment_number =
              (*coarse_catchment_numbers)(new_coarse_cell_downstream);
          }
        }
      redirect_coords = _coarse_grid->convert_fine_coords(next_fine_cell_downstream,
                                                          fine_grid_params);
      current_basin->set_new_redirect(redirect_coords,
        current_basin->get_basin_redirect_height_type());
      }
      new_coarse_cell_downstream = redirect_coords;
      while(is_minimum(new_coarse_cell_downstream)){
        if(is_outflow(new_coarse_cell_downstream)) return;
        new_coarse_cell_downstream =
          find_next_coarse_cell_downstream(new_coarse_cell_downstream);
      }
      vector<basin*> downstream_basins = get_basins_within_coarse_cell(new_coarse_cell_downstream);
      while(! downstream_basins.empty()){
        basin* downstream_basin = downstream_basins.back();
        downstream_basins.pop_back();
        downstream_basin->check_basin_for_loops();
      }
    }
  }
}

int basin::build_and_simplify_basin(int basin_count){
  basin* current_basin = this;
  int basin_number = basin_count;
  basin_count++;
  coords* current_cell = minimum_coords->clone();
  coords* last_relevant_cell = minimum_coords->clone();
  (*completed_cells)(minimum_coords) = true;
  (*basin_numbers)(minimum_coords) = basin_number;
  height_types height_type = (*use_flood_height_only)(current_cell) ? flood_height :
                                                                      connection_height;
  basic_merge_types merge_type = get_cell_merge_type(current_cell,height_type);
  height_types last_relevant_cell_height_type;
  height_types next_cell_height_type;
  while(merge_type != merge_as_secondary){
    if (merge_type == merge_as_primary){
      coords*  merge_target = get_primary_merge_target(current_cell,
                                                       height_type);
      basin* target_basin;
        if (!((*completed_cells)(merge_target))) {
          coords* target_minimum_coords = find_target_minimum_coords(merge_target);
          target_basin = create_basin();
          target_basin->set_minimum_coords(target_minimum_coords);
          target_basin->set_basin_number(basin_count);
          basin_count = \
            current_basin->build_and_simplify_basin(basin_count);
          basins.push_back(target_basin);
        } else {
          int target_basin_number = (*basin_numbers)(merge_target);
          basin* working_target_basin = basins[target_basin_number];
          while(! working_target_basin->get_primary_basin()){
            working_target_basin = working_target_basin->get_primary_basin();
          }
          target_basin = working_target_basin;
        }
        target_basin->assign_new_primary_basin(current_basin);
    }
    coords* next_cell = get_next_cell(current_cell,height_type);
    next_cell_height_type =
      ((*completed_cells)(next_cell) ||
      (*use_flood_height_only)(next_cell)) ? flood_height : connection_height;
    if(next_cell_height_type == connection_height &&
       merge_type == basic_no_merge &&
       ! ((*redirect_targets)(current_cell))){
      remove_current_cell_connection_from_ladder(last_relevant_cell,current_cell,
                                                 next_cell,last_relevant_cell_height_type);
    } else {
      last_relevant_cell = current_cell;
      last_relevant_cell_height_type = height_type;
    }
    current_cell = next_cell;
    height_type = next_cell_height_type;
    merge_type = get_cell_merge_type(current_cell,height_type);
    if ((*completed_cells)(current_cell)) {
      (*completed_cells)(current_cell) = true;
    } else {
      (*basin_numbers)(current_cell) = basin_number;
    }
  }
  redirect_coords = find_redirect_coords(current_cell,
                                         height_type);
  return basin_count;
}

void basin::find_minimum_coarse_catchment_number(){
  coords* coarse_minimum_coords = _coarse_grid->convert_fine_coords(minimum_coords,
                                                                    fine_grid_params);
  minimum_coarse_catchment_number = (*coarse_catchment_numbers)(coarse_minimum_coords);
}

basin* basin::get_basin_at_coords(coords* coords_in){
  return basins[(*basin_numbers)(coords_in) - 1];
}

vector<basin*> basin::get_basins_within_coarse_cell(coords* coarse_coords_in){
  vector<basin*> basins_in_coarse_cell;
  _fine_grid->for_all_fine_pixels_in_coarse_cell(coarse_coords_in,
                                                 coarse_grid_params,
                                                 [&](coords* fine_coords_in){
    if((*minima)(fine_coords_in)){
      basins_in_coarse_cell.push_back(get_basin_at_coords(fine_coords_in));
    }
  });
  return basins_in_coarse_cell;
}

coords* latlon_basin::find_target_minimum_coords(coords* coords_in){
  int basin_number = (*basin_numbers)(coords_in);
  int minimum_lat = basin_minimum_lats[basin_number];
  int minimum_lon = basin_minimum_lons[basin_number];
  return new latlon_coords(minimum_lat,minimum_lon);
}

coords* latlon_basin::get_next_cell(coords* coords_in,
                                    height_types height_type_in){
  int lat; int lon;
  if( height_type_in == connection_height){
    lat = (*connect_next_cell_lat_index)(coords_in);
    lon = (*connect_next_cell_lon_index)(coords_in);
  } else {
    lat = (*flood_next_cell_lat_index)(coords_in);
    lon = (*flood_next_cell_lon_index)(coords_in);
  }
  return new latlon_coords(lat,lon);
}

coords* latlon_basin::find_redirect_coords(coords* coords_in,
                                           height_types height_type_in){
  int lat; int lon;
  if( height_type_in == connection_height){
    lat = (*connect_redirect_lat_index)(coords_in);
    lon = (*connect_redirect_lon_index)(coords_in);
  } else {
    lat = (*flood_redirect_lat_index)(coords_in);
    lon = (*flood_redirect_lon_index)(coords_in);
  }
  return new latlon_coords(lat,lon);
}

coords* latlon_basin::get_primary_merge_target(coords* coords_in,
                                               height_types height_type_in){
  int lat; int lon;
  if( height_type_in == connection_height){
    lat = (*connect_force_merge_lat_index)(coords_in);
    lon = (*connect_force_merge_lon_index)(coords_in);
  } else {
    lat = (*flood_force_merge_lat_index)(coords_in);
    lon = (*flood_force_merge_lon_index)(coords_in);
  }
  return new latlon_coords(lat,lon);
}

void latlon_basin::set_new_redirect(coords* coords_in,
                                    height_types height_type_in){
  latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
  if( height_type_in == connection_height){
    (*connect_redirect_lat_index)(coords_in) =
      latlon_coords_in->get_lat();
    (*connect_redirect_lon_index)(coords_in) =
      latlon_coords_in->get_lon();
  } else {
    (*flood_redirect_lat_index)(coords_in) =
      latlon_coords_in->get_lat();
    (*flood_redirect_lon_index)(coords_in) =
      latlon_coords_in->get_lon();
  }
}

void latlon_basin::remove_current_cell_connection_from_ladder(coords* last_relevant_cell_in,
                                                              coords* current_cell_in,
                                                              coords* next_cell_in,
                                                              height_types last_relevant_cell_height_type_in){
  latlon_coords* latlon_next_cell_in = static_cast<latlon_coords*>(next_cell_in);
  if(last_relevant_cell_height_type_in == connection_height) {
    (*connect_next_cell_lat_index)(last_relevant_cell_in) = latlon_next_cell_in->get_lat();
    (*connect_next_cell_lon_index)(last_relevant_cell_in) = latlon_next_cell_in->get_lon();
  } else {
    (*flood_next_cell_lat_index)(last_relevant_cell_in) = latlon_next_cell_in->get_lat();
    (*flood_next_cell_lon_index)(last_relevant_cell_in) = latlon_next_cell_in->get_lon();
  }
  (*connect_next_cell_lat_index)(current_cell_in) = -1;
  (*connect_next_cell_lon_index)(current_cell_in) = -1;
}

coords* latlon_basin::find_next_fine_cell_downstream(coords* coords_in){
  return _fine_grid->calculate_downstream_coords_from_dir_based_rdir(coords_in,(*fine_rdirs)(coords_in));
}

coords* latlon_basin::find_next_coarse_cell_downstream(coords* coords_in){
  return _coarse_grid->calculate_downstream_coords_from_dir_based_rdir(coords_in,(*coarse_rdirs)(coords_in));
}

basic_merge_types basin::get_cell_merge_type(coords* coords_in,height_types height_type_in){
  merge_types merge_type = (*merge_points)(coords_in);
  if (height_type_in == connection_height) {
    switch (merge_type) {
      case no_merge :
      case connection_merge_not_set_flood_merge_as_primary :
      case connection_merge_not_set_flood_merge_as_secondary :
      case connection_merge_not_set_flood_merge_as_both :
        return basic_no_merge;
      case connection_merge_as_primary_flood_merge_as_primary :
      case connection_merge_as_primary_flood_merge_as_secondary :
      case connection_merge_as_primary_flood_merge_not_set :
      case connection_merge_as_primary_flood_merge_as_both :
        return merge_as_primary;
      case connection_merge_as_secondary_flood_merge_as_primary :
      case connection_merge_as_secondary_flood_merge_as_secondary :
      case connection_merge_as_secondary_flood_merge_not_set :
      case connection_merge_as_secondary_flood_merge_as_both :
        return merge_as_secondary;
      case connection_merge_as_both_flood_merge_as_primary :
      case connection_merge_as_both_flood_merge_as_secondary :
      case connection_merge_as_both_flood_merge_not_set :
      case connection_merge_as_both_flood_merge_as_both :
        return merge_as_both;
      case null_mtype:
        throw runtime_error("No merge type defined for these coordinates");
      default:
        throw runtime_error("Merge type not recognized");
    }
  } else if(height_type_in == flood_height) {
    switch (merge_type) {
      case no_merge :
      case connection_merge_as_primary_flood_merge_not_set :
      case connection_merge_as_secondary_flood_merge_not_set :
      case connection_merge_as_both_flood_merge_not_set :
        return basic_no_merge;
      case connection_merge_as_primary_flood_merge_as_primary :
      case connection_merge_as_secondary_flood_merge_as_primary :
      case connection_merge_not_set_flood_merge_as_primary :
      case connection_merge_as_both_flood_merge_as_primary :
        return merge_as_primary;
      case connection_merge_as_primary_flood_merge_as_secondary :
      case connection_merge_as_secondary_flood_merge_as_secondary :
      case connection_merge_not_set_flood_merge_as_secondary :
      case connection_merge_as_both_flood_merge_as_secondary :
        return merge_as_secondary;
      case connection_merge_as_primary_flood_merge_as_both :
      case connection_merge_as_secondary_flood_merge_as_both :
      case connection_merge_not_set_flood_merge_as_both :
      case connection_merge_as_both_flood_merge_as_both :
        return merge_as_both;
      case null_mtype:
        throw runtime_error("No merge type defined for these coordinates");
      default:
        throw runtime_error("Merge type not recognized");
    }
  } else throw runtime_error("Merge type not recognized");
}

basin* latlon_basin_post_processing_algorithm::create_basin(){
  latlon_basin* new_basin = new latlon_basin();
  new_basin->initialise_basin(basin_numbers,
                              coarse_catchment_numbers,
                              completed_cells,
                              redirect_targets,
                              minima,
                              use_flood_height_only,
                              merge_points,
                              processed_basins,
                              coarse_catchments,
                              _coarse_grid,
                              _fine_grid,
                              coarse_grid_params,
                              fine_grid_params,
                              fine_rdirs,
                              coarse_rdirs,
                              connect_redirect_lat_index,
                              connect_redirect_lon_index,
                              flood_redirect_lat_index,
                              flood_redirect_lon_index,
                              connect_next_cell_lat_index,
                              connect_next_cell_lon_index,
                              flood_next_cell_lat_index,
                              flood_next_cell_lon_index,
                              connect_force_merge_lat_index,
                              connect_force_merge_lon_index,
                              flood_force_merge_lat_index,
                              flood_force_merge_lon_index,
                              basin_minimum_lats,
                              basin_minimum_lons);
  return new_basin;
}
