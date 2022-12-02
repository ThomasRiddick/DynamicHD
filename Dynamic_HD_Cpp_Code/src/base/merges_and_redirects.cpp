#include "base/merges_and_redirects.hpp"

latlon_merge_and_redirect_indices::
  latlon_merge_and_redirect_indices(coords* target_coords){
      latlon_coords* latlon_target_coords =
        static_cast<latlon_coords*>(target_coords);
      merge_target_lat_index = latlon_target_coords->get_lat();
      merge_target_lon_index = latlon_target_coords->get_lon();
}

icon_single_index_merge_and_redirect_indices::
  icon_single_index_merge_and_redirect_indices(coords* target_coords){
    generic_1d_coords* icon_single_index_target_coords =
      static_cast<generic_1d_coords*>(target_coords);
    merge_target_cell_index = icon_single_index_target_coords->get_index();
}

void latlon_merge_and_redirect_indices::set_redirect_indices(coords* redirect_coords){
      latlon_coords* latlon_redirect_coords =
        static_cast<latlon_coords*>(redirect_coords);
      redirect_lat_index = latlon_redirect_coords->get_lat();
      redirect_lon_index = latlon_redirect_coords->get_lon();
}

void icon_single_index_merge_and_redirect_indices::set_redirect_indices(coords* redirect_coords){
    generic_1d_coords* icon_single_index_redirect_coords =
      static_cast<generic_1d_coords*>(redirect_coords);
    redirect_cell_index = icon_single_index_redirect_coords->get_index();
}

collected_merge_and_redirect_indices*
  merges_and_redirects::
    get_collected_merge_and_redirect_indices(coords* merge_coords,
                                             height_types merge_height_type,
                                             bool new_entries_permitted){
  int merge_and_redirect_indices_index;
  if (merge_height_type == flood_height) {
    merge_and_redirect_indices_index =
      (*flood_merge_and_redirect_indices_index)(merge_coords);
    if (merge_and_redirect_indices_index == -1){
      if (new_entries_permitted){
        (*flood_merge_and_redirect_indices_index)(merge_coords) = next_free_flood_index;
        flood_merge_and_redirect_indices.push_back(new collected_merge_and_redirect_indices
                                                       (merge_and_redirect_indices_factory));
        next_free_flood_index++;
      } else throw runtime_error("Merge not found");
    }
    return flood_merge_and_redirect_indices[merge_and_redirect_indices_index];
  } else if (merge_height_type == connection_height) {
    merge_and_redirect_indices_index =
      (*connect_merge_and_redirect_indices_index)(merge_coords);
    if (merge_and_redirect_indices_index == -1){
      if (new_entries_permitted){
        (*connect_merge_and_redirect_indices_index)(merge_coords) = next_free_connect_index;
        connect_merge_and_redirect_indices.push_back(new collected_merge_and_redirect_indices
                                                     (merge_and_redirect_indices_factory));
        next_free_connect_index++;
      } else throw runtime_error("Merge not found");
    }
    return connect_merge_and_redirect_indices[merge_and_redirect_indices_index];
  } else throw runtime_error("Cell type not recognized");
}

void merges_and_redirects::set_unmatched_flood_merge(coords* merge_coords){
  collected_merge_and_redirect_indices*
  working_collected_merge_and_redirect_indices =
    get_collected_merge_and_redirect_indices(merge_coords,flood_height,
                                             true);
  working_collected_merge_and_redirect_indices->set_unmatched_secondary_merge(true);
}

void merges_and_redirects::set_unmatched_connect_merge(coords* merge_coords){
  collected_merge_and_redirect_indices*
  working_collected_merge_and_redirect_indices =
    get_collected_merge_and_redirect_indices(merge_coords,connection_height,
                                             true);
  working_collected_merge_and_redirect_indices->set_unmatched_secondary_merge(true);
}
