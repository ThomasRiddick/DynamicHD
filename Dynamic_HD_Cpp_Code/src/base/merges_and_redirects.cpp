#include "base/merges_and_redirects.hpp"
#include <algorithm>

using namespace std;

latlon_merge_and_redirect_indices::
  latlon_merge_and_redirect_indices(coords* target_coords){
      latlon_coords* latlon_target_coords =
        static_cast<latlon_coords*>(target_coords);
      merge_target_lat_index = latlon_target_coords->get_lat();
      merge_target_lon_index = latlon_target_coords->get_lon();
}

latlon_merge_and_redirect_indices::
  latlon_merge_and_redirect_indices(coords* merge_target_coords,
                                    coords* redirect_coords,
                                    bool is_local_redirect) :
        merge_and_redirect_indices(is_local_redirect) {
      latlon_coords* latlon_merge_target_coords =
        static_cast<latlon_coords*>(merge_target_coords);
      merge_target_lat_index = latlon_merge_target_coords->get_lat();
      merge_target_lon_index = latlon_merge_target_coords->get_lon();
      latlon_coords* latlon_redirect_coords =
        static_cast<latlon_coords*>(redirect_coords);
      redirect_lat_index = latlon_redirect_coords->get_lat();
      redirect_lon_index = latlon_redirect_coords->get_lon();
}

icon_single_index_merge_and_redirect_indices::
  icon_single_index_merge_and_redirect_indices(coords* target_coords){
    generic_1d_coords* icon_single_index_target_coords =
      static_cast<generic_1d_coords*>(target_coords);
    merge_target_cell_index = icon_single_index_target_coords->get_index();
}

icon_single_index_merge_and_redirect_indices::
  icon_single_index_merge_and_redirect_indices(coords* merge_target_coords,
                                               coords* redirect_coords,
                                               bool is_local_redirect) :
        merge_and_redirect_indices(is_local_redirect) {
    generic_1d_coords* icon_single_index_merge_target_coords =
      static_cast<generic_1d_coords*>(merge_target_coords);
    merge_target_cell_index = icon_single_index_merge_target_coords->get_index();
    generic_1d_coords* icon_single_index_redirect_coords =
      static_cast<generic_1d_coords*>(redirect_coords);
    redirect_cell_index = icon_single_index_redirect_coords->get_index();
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

bool latlon_merge_and_redirect_indices::
      equals (const merge_and_redirect_indices& rhs) const {
  const latlon_merge_and_redirect_indices* latlon_rhs =
    static_cast<const latlon_merge_and_redirect_indices*>(&rhs);
  return (merge_target_lat_index == latlon_rhs->get_merge_target_lat_index() &&
          merge_target_lon_index == latlon_rhs->get_merge_target_lon_index() &&
          redirect_lat_index == latlon_rhs->get_redirect_lat_index() &&
          redirect_lon_index == latlon_rhs->get_redirect_lon_index() &&
          local_redirect == latlon_rhs->get_local_redirect()); };

bool icon_single_index_merge_and_redirect_indices::
      equals (const merge_and_redirect_indices& rhs) const {
  const icon_single_index_merge_and_redirect_indices* icon_single_index_rhs =
    static_cast<const icon_single_index_merge_and_redirect_indices*>(&rhs);
  return (merge_target_cell_index ==
            icon_single_index_rhs->get_merge_target_cell_index() &&
          redirect_cell_index == icon_single_index_rhs->get_redirect_cell_index() &&
          local_redirect == icon_single_index_rhs->get_local_redirect());
}

pair<int,int*>* latlon_merge_and_redirect_indices::get_indices_as_array(){
  //First number is to indicate this array is filled when positioned
  //within a larger array (where some spaces are unfilled)
  return new pair<int,int*>(6,new int[6] { 1,
                                           int(local_redirect),
                                           merge_target_lat_index,
                                           merge_target_lon_index,
                                           redirect_lat_index,
                                           redirect_lon_index });
}

pair<int,int*>* icon_single_index_merge_and_redirect_indices::get_indices_as_array(){
  //First number is to indicate this array is filled when positioned
  //within a larger array (where some spaces are unfilled)
  return new pair<int,int*>(4, new int[4] { 1,
                                            int(local_redirect),
                                            merge_target_cell_index,
                                            redirect_cell_index });
}

ostream& latlon_merge_and_redirect_indices::print(ostream& out) {
  return out << "merge_target: " << get_merge_target_lat_index() << ","  <<
                                    get_merge_target_lon_index() << endl <<
                "redirect: "     << get_redirect_lat_index()     << ","  <<
                                    get_redirect_lon_index()     << endl <<
                "local: "          << get_local_redirect()         << endl;

}

ostream& icon_single_index_merge_and_redirect_indices::print(ostream& out) {
  return out << "merge_target: " << get_merge_target_cell_index() << endl  <<
                "redirect: "     << get_redirect_cell_index()     << endl  <<
                "local"          << get_local_redirect()          << endl;

}

merge_and_redirect_indices* latlon_merge_and_redirect_indices_factory(coords* target_coords){
  return new latlon_merge_and_redirect_indices(target_coords);
}

merge_and_redirect_indices* icon_single_index_merge_and_redirect_indices_factory(coords* target_coords){
  return new icon_single_index_merge_and_redirect_indices(target_coords);
}

merge_and_redirect_indices* create_latlon_merge_and_redirect_indices_for_testing(coords* merge_target_coords,
                                    					         coords* redirect_coords,
                                    				                 bool is_local_redirect){
  merge_and_redirect_indices* working_indices = new latlon_merge_and_redirect_indices(merge_target_coords,
                                                                                      redirect_coords,
                                                                                      is_local_redirect);
  delete merge_target_coords;
  delete redirect_coords;
  return working_indices;
}

bool operator==(vector<merge_and_redirect_indices*>& lhs,
                vector<merge_and_redirect_indices*>& rhs){
  return (equal(lhs.begin(),lhs.end(),rhs.begin(),
                compare_object_pointers<merge_and_redirect_indices>) &&
          lhs.size() == rhs.size());
}

pair<tuple<int,int,int>*,int*>*
collected_merge_and_redirect_indices::get_collection_as_array(){
  int merge_and_redirect_indices_size = 0;
  pair<int,int*>* secondary_merge_dimension_and_array = nullptr;
  if (secondary_merge_and_redirect_indices) {
    secondary_merge_dimension_and_array =
      secondary_merge_and_redirect_indices->get_indices_as_array();
    merge_and_redirect_indices_size = secondary_merge_dimension_and_array->first;
  } else {
    merge_and_redirect_indices_size =
      (*primary_merge_and_redirect_indices)[0]->get_indices_as_array()->first;
  }
  int* array = new int[merge_and_redirect_indices_size*
                       (1+primary_merge_and_redirect_indices->size())];
  for (int i = 0; i<merge_and_redirect_indices_size;i++){
    if (secondary_merge_and_redirect_indices) {
      array[i] = secondary_merge_dimension_and_array->second[i];
    } else {
      array[i] = -1;
    }
  }
  for(int i = 0; i < primary_merge_and_redirect_indices->size(); i++){
    int* primary_merge_array =
      (*primary_merge_and_redirect_indices)[i]->get_indices_as_array()->second;
    for (int j = 0; j<merge_and_redirect_indices_size;j++){
        array[((i+1)*merge_and_redirect_indices_size)+j] =
          primary_merge_array[j];
    }
  }
  return new pair<tuple<int,int,int>*,int*>
    (new tuple<int,int,int>(1+primary_merge_and_redirect_indices->size(),
                            merge_and_redirect_indices_size,
                            int(bool(secondary_merge_and_redirect_indices))),
                            array);
}

collected_merge_and_redirect_indices::~collected_merge_and_redirect_indices(){
  for(vector<merge_and_redirect_indices*>::const_iterator i =
      primary_merge_and_redirect_indices->begin();
      i != primary_merge_and_redirect_indices->end();++i){
      delete (*i);
  }
  delete primary_merge_and_redirect_indices;
  delete secondary_merge_and_redirect_indices;
}

inline bool operator== (const collected_merge_and_redirect_indices& lhs,
                        const collected_merge_and_redirect_indices& rhs){
  bool is_equal = (lhs.get_unmatched_secondary_merge() ==
                   rhs.get_unmatched_secondary_merge());
  if (lhs.get_primary_merge_and_redirect_indices()){
    is_equal = (is_equal && (*lhs.get_primary_merge_and_redirect_indices() ==
                             *rhs.get_primary_merge_and_redirect_indices()));
  } else {
    //i.e. if primary merges null in lhs check if it is also null in rhs
    is_equal = (is_equal && ! rhs.get_primary_merge_and_redirect_indices());
  }
  if (lhs.get_secondary_merge_and_redirect_indices()){
    return (is_equal &&
            (*lhs.get_secondary_merge_and_redirect_indices() ==
             *rhs.get_secondary_merge_and_redirect_indices()));
  } else {
    //i.e if secondary merge is null in lhs check if it is also null in rhs
    return (is_equal && ! rhs.get_secondary_merge_and_redirect_indices());
  }
}

ostream& operator<<(ostream& out, collected_merge_and_redirect_indices& obj){
  out << "unmatched secondary merge: " <<
        obj.get_unmatched_secondary_merge() << endl
      << "secondary_merge indices  : " << endl;
  if(obj.get_secondary_merge_and_redirect_indices()) {
    out << (*obj.get_secondary_merge_and_redirect_indices()) << endl;
  } else {
    out << "   null pointer" << endl;
  }
  out << "     * * * * * * * * * *" << endl;
  out << "primary merge and redirect indices :" << endl;
  if (obj.get_primary_merge_and_redirect_indices()){
    for(vector<merge_and_redirect_indices*>::const_iterator i =
          obj.get_primary_merge_and_redirect_indices()->begin();
          i != obj.get_primary_merge_and_redirect_indices()->end();++i){
      out << *(*i) << endl
          << "     # # # # # # # # # #" << endl;
    }
  } else {
    out << "null pointer" << endl;
  }
  return out;
}

merges_and_redirects::merges_and_redirects(function<merge_and_redirect_indices*(coords*)>
                                           merge_and_redirect_indices_factory_in,
                                           grid_params* grid_params_in) :
  next_free_connect_index(0), next_free_flood_index(0),
  merge_and_redirect_indices_factory(merge_and_redirect_indices_factory_in) {
    connect_merge_and_redirect_indices_index = new field<int>(grid_params_in);
    flood_merge_and_redirect_indices_index = new field<int>(grid_params_in);
    connect_merge_and_redirect_indices_index->set_all(-1);
    flood_merge_and_redirect_indices_index->set_all(-1);
    connect_merge_and_redirect_indices =
      new vector<collected_merge_and_redirect_indices*>;
    flood_merge_and_redirect_indices =
      new vector<collected_merge_and_redirect_indices*>;
  }

merges_and_redirects::
  merges_and_redirects(field<int>* connect_merge_and_redirect_indices_index_in,
                       field<int>* flood_merge_and_redirect_indices_index_in,
                       vector<collected_merge_and_redirect_indices*>*
                          connect_merge_and_redirect_indices_in,
                       vector<collected_merge_and_redirect_indices*>*
                          flood_merge_and_redirect_indices_in,
                       grid_params* grid_params_in) :
    next_free_connect_index(0), next_free_flood_index(0),
    connect_merge_and_redirect_indices_index(connect_merge_and_redirect_indices_index_in),
    flood_merge_and_redirect_indices_index(flood_merge_and_redirect_indices_index_in),
    connect_merge_and_redirect_indices(connect_merge_and_redirect_indices_in),
    flood_merge_and_redirect_indices(flood_merge_and_redirect_indices_in)
{}

merges_and_redirects::~merges_and_redirects(){
   delete connect_merge_and_redirect_indices_index;
   delete flood_merge_and_redirect_indices_index; 
   for(vector<collected_merge_and_redirect_indices*>::const_iterator i =
     connect_merge_and_redirect_indices->begin();
     i != connect_merge_and_redirect_indices->end();++i){
     delete (*i);
   }
   for(vector<collected_merge_and_redirect_indices*>::const_iterator i =
       flood_merge_and_redirect_indices->begin();
       i != flood_merge_and_redirect_indices->end();++i){
     delete (*i);
   }
   delete connect_merge_and_redirect_indices;
   delete flood_merge_and_redirect_indices;
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
        flood_merge_and_redirect_indices->push_back(new collected_merge_and_redirect_indices
                                                        (merge_and_redirect_indices_factory));
        next_free_flood_index++;
        return flood_merge_and_redirect_indices->back();
      } else throw runtime_error("Merge not found");
    }
    return (*flood_merge_and_redirect_indices)[merge_and_redirect_indices_index];
  } else if (merge_height_type == connection_height) {
    merge_and_redirect_indices_index =
      (*connect_merge_and_redirect_indices_index)(merge_coords);
    if (merge_and_redirect_indices_index == -1){
      if (new_entries_permitted){
        (*connect_merge_and_redirect_indices_index)(merge_coords) = next_free_connect_index;
        connect_merge_and_redirect_indices->push_back(new collected_merge_and_redirect_indices
                                                          (merge_and_redirect_indices_factory));
        next_free_connect_index++;
        return connect_merge_and_redirect_indices->back();
      } else throw runtime_error("Merge not found");
    }
    return (*connect_merge_and_redirect_indices)[merge_and_redirect_indices_index];
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

pair<tuple<int,int,int>*,int*>*
merges_and_redirects::
    get_merges_and_redirects_as_array(bool get_flood_merges_and_redirects){
  int max_primary_merges_at_single_point = 0;
  vector<pair<tuple<int,int,int>*,int*>*> array_slices;
  pair<tuple<int,int,int>*,int*>* collection = nullptr;
  vector<collected_merge_and_redirect_indices*>*
    working_merge_and_redirect_indices = get_flood_merges_and_redirects ?
      flood_merge_and_redirect_indices :
      connect_merge_and_redirect_indices;
  for (vector<collected_merge_and_redirect_indices*>::const_iterator i =
      working_merge_and_redirect_indices->begin();
       i != working_merge_and_redirect_indices->end();++i){
    collection = (*i)->get_collection_as_array();
    if ( get<1>(*collection->first) > max_primary_merges_at_single_point){
      max_primary_merges_at_single_point = get<1>(*collection->first);
    }
    array_slices.push_back(collection);
  }
  int* array = new int[array_slices.size()*
                       max_primary_merges_at_single_point*
                       get<1>(*array_slices[0]->first)];
  fill_n(array,array_slices.size()*
               max_primary_merges_at_single_point*
               get<1>(*array_slices[0]->first),0);
  //Would be better to iterate using [] for vector
  int j = 0;
  for(vector<pair<tuple<int,int,int>*,int*>*>::const_iterator i = array_slices.begin();
      i != array_slices.end();++i){
    for (int k =0; k < get<0>(*(*i)->first); k++){
      array[j*max_primary_merges_at_single_point*get<1>(*(*i)->first)+
            get<0>(*(*i)->first)*get<1>(*(*i)->first)+k] =
            ((*i)->second)[get<0>(*(*i)->first)*get<1>(*(*i)->first)+k];
    }
    j++;
  }
  return new pair<tuple<int,int,int>*,int*>(new tuple<int,int,int>(array_slices.size(),
                                                                  max_primary_merges_at_single_point,
                                                                  get<1>(*array_slices[0]->first)),
                                                                  array);
}

bool merges_and_redirects::operator==(const merges_and_redirects& rhs){
  return (*connect_merge_and_redirect_indices_index ==
          *rhs.get_connect_merge_and_redirect_indices_index()) &&
         (*flood_merge_and_redirect_indices_index ==
          *rhs.get_flood_merge_and_redirect_indices_index()) &&
         (*connect_merge_and_redirect_indices ==
          *rhs.get_connect_merge_and_redirect_indices()) &&
         (*flood_merge_and_redirect_indices ==
          *rhs.get_flood_merge_and_redirect_indices());
}

ostream& operator<<(ostream& out, merges_and_redirects& obj){
  out << "connect merge indices index: " <<
          *obj.get_connect_merge_and_redirect_indices_index() << endl
      << "flood merge indices index:  " <<
          *obj.get_flood_merge_and_redirect_indices_index() << endl
      << "connect merge indices:   " << endl;
  for(vector<collected_merge_and_redirect_indices*>::const_iterator i =
        obj.get_connect_merge_and_redirect_indices()->begin();
      i != obj.get_connect_merge_and_redirect_indices()->end();++i){
    out << *(*i) << endl;
    out << endl << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl << endl;
  }
  out << "=================================" << endl;
  out << "=================================" << endl << endl;
  out << "flood merge indices:  " << endl;
  for(vector<collected_merge_and_redirect_indices*>::const_iterator i =
      obj.get_flood_merge_and_redirect_indices()->begin();
      i != obj.get_flood_merge_and_redirect_indices()->end();++i){
    out << *(*i) << endl;
    out << endl << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl << endl;
  }
  return out;
}


bool operator==(vector<collected_merge_and_redirect_indices*>& lhs,
                vector<collected_merge_and_redirect_indices*>& rhs) {
  return (equal(lhs.begin(),lhs.end(),rhs.begin(),
                compare_object_pointers<collected_merge_and_redirect_indices>) &&
          lhs.size() == rhs.size());
}
