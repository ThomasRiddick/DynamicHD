#ifndef MERGES_AND_REDIRECTS_HPP_
#define MERGES_AND_REDIRECTS_HPP_

#include <functional>
#include "base/enums.hpp"
#include "base/coords.hpp"
#include "base/field.hpp"
#include "base/disjoint_set.hpp"

using namespace std;

///Data for an individual merge/redirect
class merge_and_redirect_indices {
public:
  void set_local_redirect() { local_redirect = true; }
  void set_non_local_redirect() { local_redirect = false; }
  virtual void set_redirect_indices(coords* redirect_coords) = 0;
protected:
  bool local_redirect;
};

class latlon_merge_and_redirect_indices : public merge_and_redirect_indices {
public:
  latlon_merge_and_redirect_indices(coords* target_coords);
  void set_redirect_indices(coords* redirect_coords);
protected:
  int merge_target_lat_index;
  int merge_target_lon_index;
  int redirect_lat_index;
  int redirect_lon_index;
};

merge_and_redirect_indices* latlon_merge_and_redirect_indices_factory(coords* target_coords){
  return new latlon_merge_and_redirect_indices(target_coords);
}

class icon_single_index_merge_and_redirect_indices : public merge_and_redirect_indices {
public:
  icon_single_index_merge_and_redirect_indices(coords* target_coords);
  void set_redirect_indices(coords* redirect_coords);
protected:
  int merge_target_cell_index;
  int redirect_cell_index;
};

merge_and_redirect_indices* icon_single_index_merge_and_redirect_indices_factory(coords* target_coords){
  return new icon_single_index_merge_and_redirect_indices(target_coords);
}


///A collection of all the merge and redirect data for a given
///point
class collected_merge_and_redirect_indices {
protected:
  vector<merge_and_redirect_indices*> primary_merge_and_redirect_indices;
  merge_and_redirect_indices* secondary_merge_and_redirect_indices = nullptr;
  bool unmatched_secondary_merge = false;
  function<merge_and_redirect_indices*(coords*)>* merge_and_redirect_indices_factory = nullptr;
public:
  collected_merge_and_redirect_indices(function<merge_and_redirect_indices*(coords*)>*
                                       merge_and_redirect_indices_factory_in) :
    merge_and_redirect_indices_factory(merge_and_redirect_indices_factory_in) {}
  void set_unmatched_secondary_merge(bool unmatched_secondary_merge_in)
    { unmatched_secondary_merge = unmatched_secondary_merge_in; }
  bool get_unmatched_secondary_merge(){ return unmatched_secondary_merge; }
  void set_secondary_merge_target_coords(coords* target_coords)
    { secondary_merge_and_redirect_indices =
        (*merge_and_redirect_indices_factory)(target_coords); }
  void set_next_primary_merge_target_index(coords* target_coords) {
    primary_merge_and_redirect_indices.
      push_back((*merge_and_redirect_indices_factory)(target_coords));
  }
  merge_and_redirect_indices* get_latest_primary_merge_and_redirect_indices() {
    return primary_merge_and_redirect_indices.back();
  }
};

class merges_and_redirects {
public:
  merges_and_redirects(function<merge_and_redirect_indices*(coords*)>*
                       merge_and_redirect_indices_factory_in) :
    next_free_connect_index(0), next_free_flood_index(0),
    merge_and_redirect_indices_factory(merge_and_redirect_indices_factory_in) {}
  collected_merge_and_redirect_indices*
    get_collected_merge_and_redirect_indices(coords* merge_coords,
                                             height_types merge_height_type,
                                             bool new_entries_permitted=false);
  void set_unmatched_flood_merge(coords* merge_coords);
  void set_unmatched_connect_merge(coords* merge_coords);
protected:
  int next_free_connect_index;
  int next_free_flood_index;
  field<int>* connect_merge_and_redirect_indices_index = nullptr;
  field<int>* flood_merge_and_redirect_indices_index = nullptr;
  vector<collected_merge_and_redirect_indices*> connect_merge_and_redirect_indices;
  vector<collected_merge_and_redirect_indices*> flood_merge_and_redirect_indices;
  function<merge_and_redirect_indices*(coords*)>* merge_and_redirect_indices_factory = nullptr;
};

#endif /*MERGES_AND_REDIRECTS_HPP_*/
