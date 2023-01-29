#ifndef MERGES_AND_REDIRECTS_HPP_
#define MERGES_AND_REDIRECTS_HPP_

#include <functional>
#include "base/enums.hpp"
#include "base/coords.hpp"
#include "base/field.hpp"
#include "base/disjoint_set.hpp"

using namespace std;

template<typename T> bool compare_object_pointers(const T* lhs,const T* rhs )
{ return *lhs == *rhs; }

///Data for an individual merge/redirect
class merge_and_redirect_indices {
public:
  merge_and_redirect_indices() {}
  merge_and_redirect_indices(bool is_local_redirect) :
    local_redirect(is_local_redirect) {}
  virtual ~merge_and_redirect_indices() {}
  const bool get_local_redirect() const { return local_redirect; }
  void set_local_redirect() { local_redirect = true; }
  void set_non_local_redirect() { local_redirect = false; }
  friend bool operator== (const merge_and_redirect_indices& lhs,
                          const merge_and_redirect_indices& rhs){
    return lhs.equals(rhs);
  };
  virtual void set_redirect_indices(coords* redirect_coords) = 0;
  virtual pair<int,int*>* get_indices_as_array() = 0;
  virtual ostream& print(ostream& out) = 0;
protected:
  virtual bool equals(const merge_and_redirect_indices& rhs) const = 0;
  bool local_redirect;
  friend ostream& operator<<(ostream& out, merge_and_redirect_indices& obj)
    {  obj.print(out); return out; }
};

class latlon_merge_and_redirect_indices : public merge_and_redirect_indices {
public:
  latlon_merge_and_redirect_indices(coords* target_coords);
  latlon_merge_and_redirect_indices(coords* merge_target_coords,
                                    coords* redirect_coords,
                                    bool is_local_redirect);
  virtual ~latlon_merge_and_redirect_indices() {}
  void set_redirect_indices(coords* redirect_coords);
  const int get_merge_target_lat_index() const { return merge_target_lat_index; }
  const int get_merge_target_lon_index() const { return merge_target_lon_index; }
  const int get_redirect_lat_index() const { return redirect_lat_index; }
  const int get_redirect_lon_index() const { return redirect_lon_index; }
  pair<int,int*>* get_indices_as_array();
  ostream& print(ostream& out);
protected:
  bool equals (const merge_and_redirect_indices& rhs) const;
  int merge_target_lat_index;
  int merge_target_lon_index;
  int redirect_lat_index;
  int redirect_lon_index;
};

merge_and_redirect_indices* latlon_merge_and_redirect_indices_factory(coords* target_coords);

merge_and_redirect_indices* create_latlon_merge_and_redirect_indices_for_testing(coords* merge_target_coords,
                                    					         coords* redirect_coords,
                                    				                 bool is_local_redirect);

class icon_single_index_merge_and_redirect_indices : public merge_and_redirect_indices {
public:
  icon_single_index_merge_and_redirect_indices(coords* target_coords);
  icon_single_index_merge_and_redirect_indices(coords* merge_target_coords,
                                               coords* redirect_coords,
                                               bool is_local_redirect);
  virtual ~icon_single_index_merge_and_redirect_indices() {}
  void set_redirect_indices(coords* redirect_coords);
  const int get_merge_target_cell_index() const { return merge_target_cell_index; }
  const int get_redirect_cell_index() const { return redirect_cell_index; }
  pair<int,int*>* get_indices_as_array();
  ostream& print(ostream& out);
protected:
  bool equals(const merge_and_redirect_indices& rhs) const;
  int merge_target_cell_index;
  int redirect_cell_index;
};

merge_and_redirect_indices*
  icon_single_index_merge_and_redirect_indices_factory(coords* target_coords);

bool operator==(vector<merge_and_redirect_indices*>& lhs,
                vector<merge_and_redirect_indices*>& rhs);

///A collection of all the merge and redirect data for a given
///point
class collected_merge_and_redirect_indices {
protected:
  vector<merge_and_redirect_indices*>* primary_merge_and_redirect_indices;
  merge_and_redirect_indices* secondary_merge_and_redirect_indices = nullptr;
  bool unmatched_secondary_merge = false;
  function<merge_and_redirect_indices*(coords*)> merge_and_redirect_indices_factory = nullptr;
public:
  collected_merge_and_redirect_indices(function<merge_and_redirect_indices*(coords*)>
                                       merge_and_redirect_indices_factory_in) :
    merge_and_redirect_indices_factory(merge_and_redirect_indices_factory_in)
    {primary_merge_and_redirect_indices = new vector<merge_and_redirect_indices*>;}
  collected_merge_and_redirect_indices(vector<merge_and_redirect_indices*>*
                                       primary_merge_and_redirect_indices_in,
                                       merge_and_redirect_indices*
                                       secondary_merge_and_redirect_indices_in,
                                       function<merge_and_redirect_indices*(coords*)>
                                       merge_and_redirect_indices_factory_in) :
    primary_merge_and_redirect_indices(primary_merge_and_redirect_indices_in),
    secondary_merge_and_redirect_indices(secondary_merge_and_redirect_indices_in),
    merge_and_redirect_indices_factory(merge_and_redirect_indices_factory_in) {}
  ~collected_merge_and_redirect_indices();
  void set_unmatched_secondary_merge(bool unmatched_secondary_merge_in)
    { unmatched_secondary_merge = unmatched_secondary_merge_in; }
  const bool get_unmatched_secondary_merge() const { return unmatched_secondary_merge; }
  void set_secondary_merge_target_coords(coords* target_coords)
    { secondary_merge_and_redirect_indices =
        merge_and_redirect_indices_factory(target_coords); }
  void set_next_primary_merge_target_index(coords* target_coords) {
    primary_merge_and_redirect_indices->
      push_back(merge_and_redirect_indices_factory(target_coords));
  }
  merge_and_redirect_indices* get_latest_primary_merge_and_redirect_indices() {
    return primary_merge_and_redirect_indices->back();
  }
  const merge_and_redirect_indices* get_secondary_merge_and_redirect_indices() const {
    return secondary_merge_and_redirect_indices;
  }
  merge_and_redirect_indices* get_secondary_merge_and_redirect_indices() {
    return secondary_merge_and_redirect_indices;
  }
  vector<merge_and_redirect_indices*>* get_primary_merge_and_redirect_indices() const {
    return primary_merge_and_redirect_indices;
  }

  pair<pair<int,int>*,int*>* get_collection_as_array();

  //Note equality does not check the merge and redirect factories are equal
  friend bool operator== (const collected_merge_and_redirect_indices& lhs,
                          const collected_merge_and_redirect_indices& rhs);

  friend ostream& operator<<(ostream& out, collected_merge_and_redirect_indices& obj);
};

class merges_and_redirects {
public:
  merges_and_redirects(function<merge_and_redirect_indices*(coords*)>
                       merge_and_redirect_indices_factory_in,
                       grid_params* grid_params_in);
  merges_and_redirects(field<int>* connect_merge_and_redirect_indices_index_in,
                       field<int>* flood_merge_and_redirect_indices_index_in,
                       vector<collected_merge_and_redirect_indices*>*
                          connect_merge_and_redirect_indices_in,
                       vector<collected_merge_and_redirect_indices*>*
                          flood_merge_and_redirect_indices_in,
                       grid_params* grid_params_in);
  ~merges_and_redirects();
  collected_merge_and_redirect_indices*
    get_collected_merge_and_redirect_indices(coords* merge_coords,
                                             height_types merge_height_type,
                                             bool new_entries_permitted=false);
  void set_unmatched_flood_merge(coords* merge_coords);
  void set_unmatched_connect_merge(coords* merge_coords);
  field<int>* get_connect_merge_and_redirect_indices_index() const
    { return connect_merge_and_redirect_indices_index; }
  field<int>* get_flood_merge_and_redirect_indices_index() const
    { return flood_merge_and_redirect_indices_index; }
  vector<collected_merge_and_redirect_indices*>*
    get_connect_merge_and_redirect_indices() const
    { return connect_merge_and_redirect_indices; }
  vector<collected_merge_and_redirect_indices*>*
    get_flood_merge_and_redirect_indices() const
    { return flood_merge_and_redirect_indices; }
  // pair<tuple<int,int,int>,int*>* get_flood_merges_and_redirects_as_array();
  // pair<tuple<int,int,int>,int*>* get_connect_merges_and_redirects_as_array();
  bool operator==(const merges_and_redirects& rhs);
  friend ostream& operator<<(ostream& out, merges_and_redirects& obj);
protected:
  int next_free_connect_index;
  int next_free_flood_index;
  field<int>* connect_merge_and_redirect_indices_index = nullptr;
  field<int>* flood_merge_and_redirect_indices_index = nullptr;
  vector<collected_merge_and_redirect_indices*>* connect_merge_and_redirect_indices;
  vector<collected_merge_and_redirect_indices*>* flood_merge_and_redirect_indices;
  function<merge_and_redirect_indices*(coords*)> merge_and_redirect_indices_factory;
};

bool operator==(vector<collected_merge_and_redirect_indices*>& lhs,
                vector<collected_merge_and_redirect_indices*>& rhs);

#endif /*MERGES_AND_REDIRECTS_HPP_*/
