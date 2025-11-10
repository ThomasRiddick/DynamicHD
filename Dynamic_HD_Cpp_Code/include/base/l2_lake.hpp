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



class lake {
public:
  vector<coords*> lake_cells;
  vector<coords*> lake_next_cell_to_fill;
  vector<bool> lake_cells_connection_only;
  vector<double> lake_fill_heights;
  double sill_height;
  int connects_to_lake;
  bool connection_set;
  bool local_redirect;
  coords* redirect_location;
  bool is_leaf_node;
  pair<int,int> sublakes;
  friend bool operator== (const merge_and_redirect_indices& lhs,
                          const merge_and_redirect_indices& rhs){
    return lhs.equals(rhs);
  };
  virtual void set_redirect_indices(coords* redirect_coords) = 0;
  virtual pair<int,int*>* get_indices_as_array() = 0;
  virtual ostream& print(ostream& out) = 0;
  virtual void add_offsets_to_lat_indices(int offset_non_local,
                                          int offset_local) = 0;
protected:
  virtual bool equals(const merge_and_redirect_indices& rhs) const = 0;
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
  void add_offsets_to_lat_indices(int offset_non_local,int offset_local);
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
  //Does nothing for ICON grid
  void add_offsets_to_lat_indices(int offset_non_local,int offset_local){};
protected:
  bool equals(const merge_and_redirect_indices& rhs) const;
  int merge_target_cell_index;
  int redirect_cell_index;
};

merge_and_redirect_indices*
  icon_single_index_merge_and_redirect_indices_factory(coords* target_coords);

bool operator==(vector<merge_and_redirect_indices*>& lhs,
                vector<merge_and_redirect_indices*>& rhs);

pair<int*,float*> get_lakes_as_array(vector<lake*>*);

#endif /*MERGES_AND_REDIRECTS_HPP_*/
