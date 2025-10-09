//New version of basin evaluation algorithm with tree
//structure closer to that of algorithm of Barnes et al

/*
 * basin_evaluation_algorithm.cpp
 *
 *  Created on: Feb 21, 2018
 *      Author: thomasriddick
 */
// Note the algorithm works on 3 sets of cells. Each threshold is the threshold to
// start filling a new cell. The new cell can either be a flood or a connect cell;
// connect means it simply connects other cells by a set of lateral and diagonal channels
// connecting all 8 surrounding cells but whose width has negible effect on the cells
// ability to hold water. Flood means flooding the entire cell. The algorithm consider three
// cells at a time. The previous cell is actually the cell currently filling, the center cell
// is the cell target for beginning to fill while the new cell is using for testing if the
// center cell is the sill at the edge of a basin

// It is possible for more than one basin merge to occur at single level. Here level exploration is
// only done in a consitent manner once the first level merge is found. Then all other merges are
// searched for at the same level. Levels where no merge is found, i.e. all surrounding cells not
// in the part of the basin already processed are explored in an ad-hoc manner as this will still
// always give consitent results


#include <unordered_set>
#include <map>
#include <stack>
#include "algorithms/l2_basin_evaluation_algorithm.hpp"

using namespace std;

void filling_order_entry::print(ostream& outstr){
    outstr << "cell_coords= " << *cell_coords;
    outstr << "height_type= " << height_type << endl;
    outstr << "volume_threshold= " << volume_threshold << endl;
    outstr << "cell_height= " << cell_height << endl;
}

store_to_array::store_to_array(){
  objects = vector<vector<double>*>();
  working_object = nullptr;
}

void store_to_array::set_null_coords(coords* null_coords_in){
    null_coords = null_coords_in;
}

void store_to_array::add_object(){
    working_object = new vector<double>{0.0};
    objects.push_back(working_object);
}

void store_to_array::complete_object(){
    (*working_object)[0] = (double)(working_object->size() - 1);
}

void store_to_array::add_number(int number_in){
    working_object->push_back((double)number_in);
}

void store_to_array::add_coords(coords* coords_in, int array_offset,
                                int additional_lat_offset){
    if (coords_in->coords_type ==
        coords_in->coords_types::generic_1d) {
        generic_1d_coords* generic_1d_coords_in =
            static_cast<generic_1d_coords*>(coords_in);
        working_object->push_back((double)generic_1d_coords_in->get_index()+array_offset);
    } else if (coords_in->coords_type ==
               coords_in->coords_types::latlon) {
        latlon_coords* latlon_coords_in =
            static_cast<latlon_coords*>(coords_in);
        working_object->push_back((double)latlon_coords_in->get_lat()+array_offset+
                                  additional_lat_offset);
        working_object->push_back((double)latlon_coords_in->get_lon()+array_offset);
    }
}

void store_to_array::add_field(vector<int>* field_in){
    working_object->push_back((double)field_in->size());
    for(int i : *field_in){
        working_object->push_back((double)i);
    }
}

void store_to_array::add_outflow_points_dict(map<int,pair<coords*,bool>*> &outflow_points_in,
                                             grid* grid_in,
                                             int array_offset,
                                             int additional_lat_offset){
    vector<double>* dict_array = new vector<double>();
    for(map<int,pair<coords*,bool>*>::iterator i = outflow_points_in.begin();
        i != outflow_points_in.end(); ++i){
        vector<double>* entry_as_array = new vector<double>();
        if (i->first >= 0) {
            entry_as_array->push_back((double)i->first+array_offset);
        } else {
            entry_as_array->push_back((double)i->first);
        }
        if (*(i->second->first) == (*null_coords)){
            if (grid_in->get_grid_type() == grid_in->grid_types::icon_single_index){
                entry_as_array->push_back(-1);
            } else {
                entry_as_array->push_back(-1);
                entry_as_array->push_back(-1);
            }
        } else if (grid_in->get_grid_type() == grid_in->grid_types::icon_single_index) {
            generic_1d_coords* generic_1d_entry_coords =
                static_cast<generic_1d_coords*>(i->second->first);
            entry_as_array->push_back((double)generic_1d_entry_coords->get_index()+array_offset);
        } else {
            latlon_coords* latlon_entry_coords =
                static_cast<latlon_coords*>(i->second->first);
            if (i->second->second){
                entry_as_array->push_back((double)latlon_entry_coords->get_lat()+array_offset+
                                          additional_lat_offset);
            } else {
                entry_as_array->push_back((double)latlon_entry_coords->get_lat()+array_offset-1);
            }
            entry_as_array->push_back((double)latlon_entry_coords->get_lon()+array_offset);
        }
        entry_as_array->push_back((double)i->second->second);
        dict_array->insert(dict_array->end(),entry_as_array->begin(),entry_as_array->end());
        delete entry_as_array;
    }
    working_object->push_back((double)outflow_points_in.size());
    working_object->insert(working_object->end(),dict_array->begin(),dict_array->end());
    delete dict_array;
}

void store_to_array::add_filling_order(vector<filling_order_entry*> &filling_order_in,
                                       grid* grid_in,int array_offset,
                                       int additional_lat_offset){
    vector<double> filling_order_array;
    for(vector<filling_order_entry*>::iterator entry = filling_order_in.begin();
        entry != filling_order_in.end(); ++entry){
        if (grid_in->get_grid_type() == grid_in->grid_types::icon_single_index) {
            generic_1d_coords* generic_1d_cell_coords =
                static_cast<generic_1d_coords*>((*entry)->cell_coords);
            filling_order_array.push_back((double)generic_1d_cell_coords->get_index()+array_offset);
        } else {
            latlon_coords* latlon_cell_coords =
                static_cast<latlon_coords*>((*entry)->cell_coords);
            filling_order_array.push_back((double)latlon_cell_coords->get_lat()+array_offset+
                                          additional_lat_offset);
            filling_order_array.push_back((double)latlon_cell_coords->get_lon()+array_offset);
        }
        filling_order_array.push_back((double)(*entry)->height_type+
                                      height_type_offset);
        filling_order_array.push_back((double)(*entry)->volume_threshold);
        filling_order_array.push_back((double)(*entry)->cell_height);
    }
    working_object->push_back((double)filling_order_in.size());
    working_object->insert(working_object->end(),filling_order_array.begin(),
                           filling_order_array.end());
}

vector<double>* store_to_array::complete_array(){
    vector<double>* array = new vector<double>{(double)objects.size()};
    for(vector<vector<double>*>::iterator obj = objects.begin();
        obj != objects.end(); ++obj){
      array->insert(array->end(),(*obj)->begin(),(*obj)->end());
      delete *obj;
    }
    return array;
}

lake_variables::lake_variables(int lake_number_in,
                               coords* center_coords_in,
                               double lake_lower_boundary_height_in,
                               int primary_lake_in,
                               set<int>* secondary_lakes_in) :
                               center_coords(center_coords_in),
                               lake_number(lake_number_in),
                               lake_lower_boundary_height(lake_lower_boundary_height_in),
                               filled_lake_area(-1.0),
                               primary_lake(primary_lake_in),
                               secondary_lakes(secondary_lakes_in),
                               center_cell_volume_threshold(0.0),
                               lake_area(0.0),
                               potential_exit_points(nullptr) {}

void lake_variables::set_primary_lake(int primary_lake_in) {
    primary_lake = primary_lake_in;
}

void lake_variables::set_potential_exit_points(vector<coords*>* potential_exit_points_in) {
    potential_exit_points = potential_exit_points_in;
}

void lake_variables::set_filled_lake_area(){
    filled_lake_area = lake_area;
}

lake_variables::~lake_variables() {
  delete center_coords;
  delete secondary_lakes;
  for (auto entry : outflow_points) {
    delete entry.second->first;
    delete entry.second;
  }
  for (auto entry : filling_order) delete entry;
  for (auto entry : spill_points) delete entry.second;
  for (auto entry : *potential_exit_points) delete entry;
  for (auto entry : list_of_cells_in_lake) delete entry;
  delete potential_exit_points;
}

void lake_variables::print(ostream& outstr){
    outstr << "center_coords= " << *center_coords << endl;
    outstr << "lake_number= " << lake_number << endl;
    outstr << "primary_lake= " << primary_lake << endl;
    outstr << "secondary_lakes= ";
    if (! secondary_lakes->empty()) {
        for (int secondary_lake : *secondary_lakes) outstr << secondary_lake;
    } else outstr << " none";
    outstr << endl;
    outstr << "outflow_points" << endl;
    if (! outflow_points.empty()){
        for (pair<int,pair<coords*,bool>*> outflow_point : outflow_points){
            outstr << outflow_point.first << endl;
            outstr << *outflow_point.second->first;
            outstr << outflow_point.second->second << endl;
        }
    } else outstr << "none";
    outstr << "filling_order= " << endl;
    if (! filling_order.empty()) {
        for (filling_order_entry* entry : filling_order) outstr << *entry;
    } else outstr << " none";
    outstr << endl;
    outstr << "center_cell_volume_threshold= " << center_cell_volume_threshold << endl;
    outstr << "lake_area= " << lake_area << endl;
    //outstr << "spill_points= " << spill_points << endl;
    outstr << "potential_exit_points= " << endl;
    if (! potential_exit_points->empty()) {
        for (coords* exit_point : *potential_exit_points)
            outstr << *exit_point;
    } else outstr << "none";
    outstr << endl;
}

simple_search::simple_search(grid* grid_in,
                             grid_params* grid_params_in){
        search_completed_cells = new field<bool>(grid_params_in);
        search_completed_cells->set_all(null_lake_number);
        _grid = grid_in;
}

coords* simple_search::search(function<bool(coords*)> target_found_func,
                              function<bool(coords*)> ignore_nbr_func_in,
                              coords* start_point){
    if (! search_q.empty()) {
        throw runtime_error("Search Failed");
    }
    search_q.push(new landsea_cell(start_point->clone()));
    search_completed_cells->set_all(false);
    ignore_nbr_func = ignore_nbr_func_in;
    while ( ! search_q.empty()) {
        landsea_cell* search_cell = search_q.front();
        search_q.pop();
        search_coords = search_cell->get_cell_coords();
        if (target_found_func(search_coords)) {
            coords* end_point = search_coords->clone();
            delete search_cell;
            while ( ! search_q.empty()) {
                search_cell = search_q.front();
                search_q.pop();
                delete search_cell;
            }
            return end_point;
        }
        search_process_neighbors();
        delete search_cell;
    }
    throw runtime_error("Search Failed");
}

void simple_search::search_process_neighbors() {
    _grid->for_all_nbrs_general(search_coords,[&](coords* search_nbr_coords){
        if (! ((*search_completed_cells)(search_nbr_coords) ||
               ignore_nbr_func(search_nbr_coords))) {
            search_q.push(new landsea_cell(search_nbr_coords->clone()));
            (*search_completed_cells)(search_nbr_coords) = true;
        }
        delete search_nbr_coords;
    });
}

basin_evaluation_algorithm::basin_evaluation_algorithm(bool* minima_in,
                                                       double* raw_orography_in,
                                                       double* corrected_orography_in,
                                                       double* cell_areas_in,
                                                       int* prior_fine_catchment_nums_in,
                                                       int* coarse_catchment_nums_in,
                                                       int* catchments_from_sink_filling_in,
                                                       int additional_lat_offset_in,
                                                       grid_params* grid_params_in,
                                                       grid_params* coarse_grid_params_in) :
        lake_connections(nullptr), additional_lat_offset(additional_lat_offset_in),
        potential_exit_points(nullptr) {
    _grid_params = grid_params_in;
    _coarse_grid_params = coarse_grid_params_in;
    _grid = grid_factory(_grid_params);
    _coarse_grid = grid_factory(_coarse_grid_params);
    minima = new field<bool>(minima_in,_grid_params);
    raw_orography = new field<double>(raw_orography_in,_grid_params);
    corrected_orography = new field<double>(corrected_orography_in,_grid_params);
    cell_areas = new field<double>(cell_areas_in,_grid_params);
    prior_fine_catchment_nums =
        new field<int>(prior_fine_catchment_nums_in,_grid_params);
    coarse_catchment_nums =
        new field<int>(coarse_catchment_nums_in,_coarse_grid_params);
    catchments_from_sink_filling =
        new field<int>(catchments_from_sink_filling_in,_grid_params);
    lake_numbers = new field<int>(_grid_params);
    lake_numbers->set_all(null_lake_number);
    lake_mask = new field<bool>(_grid_params);
    lake_mask->set_all(false);
    completed_cells = new field<bool>(_grid_params);
    completed_cells->set_all(false);
    cells_in_lake = new field<bool>(_grid_params);
    cells_in_lake->set_all(false);
    level_completed_cells = new field<bool>(_grid_params);
    level_completed_cells->set_all(false);
    search_alg = new simple_search(_grid,
                                   _grid_params);
    coarse_search_alg = new simple_search(_coarse_grid,
                                          _coarse_grid_params);

}

basin_evaluation_algorithm::~basin_evaluation_algorithm(){
    for (lake_variables* lake : lakes) delete lake;
    delete minima;
    delete raw_orography;
    delete corrected_orography;
    delete cell_areas;
    delete prior_fine_catchment_nums;
    delete coarse_catchment_nums;
    delete catchments_from_sink_filling;
    delete lake_numbers;
    delete lake_mask;
    delete completed_cells;
    delete cells_in_lake;
    delete level_completed_cells;
    delete search_alg;
    delete coarse_search_alg;
    delete _coarse_grid;
    delete _grid;
}

void basin_evaluation_algorithm::evaluate_basins(){
    //Leave room for a catchment 0
    int max_catchment_number = prior_fine_catchment_nums->get_max_element();
    for (int i = 0;i <= max_catchment_number;i++){
        sink_points.push_back(null_coords);
    }
    lakes.clear();
    if (! lake_q.empty()) {
        throw runtime_error("Lake queue not empty");
    }
    lake_connections = new disjoint_set_forest();
    lake_numbers->set_all(null_lake_number);
    stack<coords*> minima_q;
    _grid->for_all([&](coords* coords_in){
        if ((*minima)(coords_in)){
            minima_q.push(coords_in);
        } else delete coords_in;
    });
    vector<int> merging_lakes;
    while (! minima_q.empty()) {
        coords* minimum = minima_q.top();
        minima_q.pop();
        int lake_number = lakes.size();
        lake_variables* lake = new lake_variables(lake_number,minimum,(*raw_orography)(minimum));
        lakes.push_back(lake);
        lake_q.push(lake);
        lake_connections->add_set(lake_number);
    }
    while (true) {
        while ( ! lake_q.empty()) {
            lake_variables* lake = lake_q.front();
            lake_q.pop();
            int lake_number = lake->lake_number;
            initialize_basin(lake);
            while (true) {
                if (q.empty()) {
                    throw runtime_error("Basin outflow not found");
                }
                center_cell = static_cast<basin_cell*>(q.top());
                q.pop();
                //Call the newly loaded coordinates and height for the center cell 'new'
                //until after making the test for merges then relabel. Center cell height/coords
                //without the 'new' moniker refers to the previous center cell; previous center cell
                //height/coords the previous previous center cell
                new_center_coords = center_cell->get_cell_coords()->clone();
                new_center_cell_height_type = center_cell->get_height_type();
                new_center_cell_height = center_cell->get_orography();
                //Exit to basin or level area found
                if (new_center_cell_height <= center_cell_height &&
                    searched_level_height != center_cell_height) {
                    search_for_outflows_on_level(lake_number);
                    if ( ! outflow_lake_numbers.empty()) {
                        //Exit(s) found
                        for(vector<int>::iterator entry =
                            outflow_lake_numbers.begin();
                            entry != outflow_lake_numbers.end();++entry){
                            int other_lake_number = *entry;
                            if (other_lake_number != -1) {
                                if (lake_connections->find_root(lake_number) !=
                                    lake_connections->find_root(other_lake_number)) {
                                    lake_connections->make_new_link(lake_number,
                                                                    other_lake_number);
                                    merging_lakes.push_back(lake_number);
                                }
                            }
                        }
                        lake->set_potential_exit_points(potential_exit_points);
                        if (lake->filled_lake_area == -1.0) lake->filled_lake_area = 1.0;
                        for (coords* coords_in : lake->list_of_cells_in_lake) {
                            if ((*raw_orography)(coords_in) < center_cell_height){
                                (*raw_orography)(coords_in) = center_cell_height;
                            }
                            if ((*corrected_orography)(coords_in) < center_cell_height){
                                (*corrected_orography)(coords_in) = center_cell_height;
                            }
                        }
                        delete center_cell;
                        break;
                    } else {
                        delete potential_exit_points;
                        //Don't rescan level later
                        searched_level_height = center_cell_height;
                    }
                }
                //Process neighbors of new center coords
                process_neighbors();
                delete previous_cell_coords;
                previous_cell_coords = center_coords;
                previous_cell_height = center_cell_height;
                previous_cell_height_type = center_cell_height_type;
                center_cell_height_type = new_center_cell_height_type;
                center_cell_height = new_center_cell_height;
                center_coords = new_center_coords;
                process_center_cell(lake);
                delete center_cell;
            }
            delete new_center_coords;
            delete center_coords;
            delete previous_cell_coords;
            while (! q.empty()) {
                center_cell = static_cast<basin_cell*>(q.top());
                q.pop();
                delete center_cell;
            }
        }
        if ( ! merging_lakes.empty()) {
            unordered_set<int> unique_lake_groups;
            for (int merging_lake : merging_lakes){
                unique_lake_groups.insert(lake_connections->find_root(merging_lake));
            }
            for(int lake_group : unique_lake_groups) {
                vector<int>* potential_sublakes_in_lake =
                    lake_connections->get_set(lake_group)->get_set_element_labels();
                set<int>* sublakes_in_lake = new set<int>();
                for (int sublake : *potential_sublakes_in_lake) {
                    if (lakes[sublake]->primary_lake == null_lake_number){
                        sublakes_in_lake->insert(sublake);
                    }
                }
                delete potential_sublakes_in_lake;
                int new_lake_number = lakes.size();
                int first_sublake = *(sublakes_in_lake->lower_bound(-1));
                coords* new_lake_center_coords = lakes[first_sublake]->center_coords;
                lake_variables* new_lake =
                    new lake_variables(new_lake_number,
                                       new_lake_center_coords->clone(),
                                       (*raw_orography)(new_lake_center_coords),
                                       null_lake_number,
                                       sublakes_in_lake);
                lake_connections->add_set(new_lake_number);
                //Note the new lake isn't necessarily the root of the disjointed set
                lake_connections->make_new_link(new_lake_number,first_sublake);
                lakes.push_back(new_lake);
                lake_q.push(new_lake);
                for(set<int>::iterator sublake =
                    sublakes_in_lake->begin();
                    sublake != sublakes_in_lake->end();++sublake){
                    for(set<int>::iterator other_sublake =
                        sublakes_in_lake->begin();
                        other_sublake != sublakes_in_lake->end();++other_sublake){
                        if (*sublake != *other_sublake) {
                            lakes[*sublake]->spill_points[*other_sublake] =
                                search_alg->
                                search(function<bool(coords*)>([this,sublake,other_sublake](coords* coords_in) {
                                       return (*lake_numbers)(coords_in) == *other_sublake; }),
                                       function<bool(coords*)>([this,sublake](coords* coords_in) {
                                       return (*corrected_orography)(coords_in) !=
                                       (*corrected_orography)(lakes[*sublake]->center_coords); }),
                                       lakes[*sublake]->center_coords);
                        }
                    }
                }
                for(set<int>::iterator sublake =
                    sublakes_in_lake->begin();
                    sublake != sublakes_in_lake->end();++sublake){
                        for (coords* coords_in : lakes[*sublake]->list_of_cells_in_lake) {
                            (*lake_numbers)(coords_in) = new_lake->lake_number;
                        }
                    lakes[*sublake]->set_primary_lake(new_lake->lake_number);
                }
            }
            unique_lake_groups.clear();
            merging_lakes.clear();
        } else break;
    }
    set_outflows();
    delete lake_connections;
}

void basin_evaluation_algorithm::initialize_basin(lake_variables* lake) {
    if ( ! q.empty()) throw runtime_error("Queue not empty at initilisation");
    searched_level_height = 0.0;
    completed_cells->set_all(false);
    cells_in_lake->set_all(false);
    lake->center_cell_volume_threshold = 0.0;
    center_coords = lake->center_coords->clone();
    double raw_height = (*raw_orography)(center_coords);
    double corrected_height = (*corrected_orography)(center_coords);
    if (raw_height <= corrected_height) {
        center_cell_height = raw_height;
        center_cell_height_type = flood_height;
    } else {
        center_cell_height = corrected_height;
        center_cell_height_type = connection_height;
    }
    previous_cell_coords = center_coords;
    previous_cell_height_type = center_cell_height_type;
    previous_cell_height = center_cell_height;
    catchments_from_sink_filling_catchment_num =
        (*catchments_from_sink_filling)(center_coords);
    new_center_coords = center_coords;
    new_center_cell_height_type = center_cell_height_type;
    new_center_cell_height = center_cell_height;
    if (center_cell_height_type == connection_height) {
        lake->lake_area = 0.0;
    } else if (center_cell_height_type == flood_height) {
        lake->lake_area = (double)(*cell_areas)(center_coords);
    } else {
        throw runtime_error("Cell type not recognized");
    }
    (*completed_cells)(center_coords) = true;
    //Make partial iteration
    process_neighbors();
    center_cell = static_cast<basin_cell*>(q.top());
    q.pop();
    new_center_coords = center_cell->get_cell_coords()->clone();
    new_center_cell_height_type = center_cell->get_height_type();
    new_center_cell_height = center_cell->get_orography();
    process_neighbors();
    center_cell_height_type = new_center_cell_height_type;
    center_cell_height = new_center_cell_height;
    center_coords = new_center_coords;
    process_center_cell(lake);
    delete center_cell;
}

void basin_evaluation_algorithm::search_for_outflows_on_level(int lake_number) {
    level_completed_cells->set_all(false);
    outflow_lake_numbers.clear();
    potential_exit_points = new vector<coords*>();
    if ( ! level_q.empty()) throw runtime_error("level q not empty at start of search");
    vector<basin_cell*> additional_cells_to_return_to_q;
    additional_cells_to_return_to_q.clear();
    if ( ! q.empty()) {
        while ( ! q.empty()){
            double next_cell_height = static_cast<basin_cell*>(q.top())->get_orography();
            if (next_cell_height > center_cell_height) break;
            if (next_cell_height == center_cell_height &&
                (*lake_numbers)(q.top()->get_cell_coords()) == -1) {
                level_q.push_back(static_cast<basin_cell*>(q.top()));
            } else {
                additional_cells_to_return_to_q.push_back(static_cast<basin_cell*>(q.top()));
            }
            q.pop();
        }
    }
    //put removed items back in q
    for(basin_cell* cell : level_q){
        q.push(cell->clone());
    }
    for(basin_cell* cell : additional_cells_to_return_to_q){
        q.push(cell);
    }
    level_q.push_back(new basin_cell(center_cell_height,
                                     center_cell_height_type,
                                     center_coords->clone()));
    if (center_cell_height == new_center_cell_height) {
        level_q.push_back(new basin_cell(new_center_cell_height,
                                         new_center_cell_height_type,
                                         new_center_coords->clone()));
        if ((*lake_numbers)(new_center_coords) != -1 &&
            (*lake_numbers)(new_center_coords) != lake_number) {
            outflow_lake_numbers.push_back((*lake_numbers)(new_center_coords));
            potential_exit_points->push_back(new_center_coords->clone());
        }
    }
    while ( ! level_q.empty()) {
        basin_cell* level_center_cell = level_q.back();
        level_q.pop_back();
        coords* level_coords = level_center_cell->get_cell_coords();
        (*level_completed_cells)(level_coords) = true;
        process_level_neighbors(lake_number,level_coords);
        delete level_center_cell;
    }
}

void basin_evaluation_algorithm::process_level_neighbors(int lake_number,
                                                         coords* level_coords) {
    _grid->for_all_nbrs_general(level_coords,[&](coords* nbr_coords){
        int nbr_catchment = (*catchments_from_sink_filling)(nbr_coords);
        bool in_different_catchment =
          ( nbr_catchment != catchments_from_sink_filling_catchment_num) &&
          ( nbr_catchment != -1);
        if ((! (*level_completed_cells)(nbr_coords)) &&
            ! in_different_catchment &&
            ! (*cells_in_lake)(nbr_coords)) {
            double raw_height = (*raw_orography)(nbr_coords);
            double corrected_height = (*corrected_orography)(nbr_coords);
            (*level_completed_cells)(nbr_coords) = true;
            double nbr_height; height_types nbr_height_type;
            if (raw_height <= corrected_height) {
                nbr_height = raw_height;
                nbr_height_type = flood_height;
            } else {
                nbr_height = corrected_height;
                nbr_height_type = connection_height;
            }
            if (nbr_height < center_cell_height &&
                (*lake_numbers)(nbr_coords) != lake_number) {
                outflow_lake_numbers.push_back(-1);
                potential_exit_points->push_back(nbr_coords->clone());
            } else if (nbr_height == center_cell_height &&
                  (*lake_numbers)(nbr_coords) != -1 &&
                  (*lake_numbers)(nbr_coords) != lake_number) {
                outflow_lake_numbers.push_back((*lake_numbers)(nbr_coords));
                potential_exit_points->push_back(nbr_coords->clone());
            } else if (nbr_height == center_cell_height) {
                level_q.push_back(new basin_cell(nbr_height,nbr_height_type,
                                                 nbr_coords->clone()));
            }
        }
        delete nbr_coords;
    });
}

void basin_evaluation_algorithm::process_center_cell(lake_variables* lake) {
    (*cells_in_lake)(previous_cell_coords) = true;
    lake->list_of_cells_in_lake.push_back(previous_cell_coords->clone());
    if ((*lake_numbers)(previous_cell_coords) == null_lake_number) {
        (*lake_numbers)(previous_cell_coords) = lake->lake_number;
    }
    lake->center_cell_volume_threshold +=
            lake->lake_area*(center_cell_height-previous_cell_height);
    if (previous_cell_height_type == connection_height) {
        q.push(new basin_cell((*raw_orography)(previous_cell_coords),
               flood_height,previous_cell_coords->clone()));
        lake->filling_order.push_back(new filling_order_entry(previous_cell_coords->clone(),
                                                              connection_height,
                                                              lake->center_cell_volume_threshold,
                                                              center_cell_height));
    } else if (previous_cell_height_type == flood_height) {
        lake->filling_order.push_back(new filling_order_entry(previous_cell_coords->clone(),
                                                              flood_height,
                                                              lake->center_cell_volume_threshold,
                                                              center_cell_height));
    } else {
        throw runtime_error("Cell type not recognized");
    }
    if (center_cell_height_type == flood_height) {
        lake->set_filled_lake_area();
        lake->lake_area += (*cell_areas)(center_coords);
    } else if (center_cell_height_type != connection_height) {
        throw runtime_error("Cell type not recognized");
    }
}

void basin_evaluation_algorithm::process_neighbors() {
    _grid->for_all_nbrs_general(new_center_coords,[&](coords* nbr_coords){
        int nbr_catchment = (*catchments_from_sink_filling)(nbr_coords);
        bool in_different_catchment =
          ( nbr_catchment != catchments_from_sink_filling_catchment_num) &&
          ( nbr_catchment != -1);
        if ((! (*completed_cells)(nbr_coords)) &&
            ! in_different_catchment){
            double raw_height = (*raw_orography)(nbr_coords);
            double corrected_height = (*corrected_orography)(nbr_coords);
            double nbr_height; height_types nbr_height_type;
            if (raw_height <= corrected_height) {
                nbr_height = raw_height;
                nbr_height_type = flood_height;
            } else {
                nbr_height = corrected_height;
                nbr_height_type = connection_height;
            }
            q.push(new basin_cell(nbr_height,nbr_height_type,
                                  nbr_coords->clone()));
            (*completed_cells)(nbr_coords) = true;
        }
        delete nbr_coords;
    });
}

void basin_evaluation_algorithm::set_outflows(){
    for(vector<lake_variables*>::iterator entry = lakes.begin();
        entry != lakes.end();++entry){
        lake_variables* lake = *entry;
        if (lake->primary_lake == null_lake_number) {
            //Arbitrarily choose the first exit point
            coords* first_potential_exit_point = lake->potential_exit_points->front();
            int first_potential_exit_point_lake_number =
                (*lake_numbers)(first_potential_exit_point);
            if (first_potential_exit_point_lake_number != -1) {
                //This means the lake is spilling into an
                //unconnected neighboring lake at a lower level
                //It is possible though very rare that a will spill
                //to two unconnected downstreams lakes in the same
                //overall catchment both at a lower level - in this
                //case arbitrarily use the first
                lake->outflow_points[first_potential_exit_point_lake_number] =
                    new pair<coords*,bool>(null_coords->clone(),true);
            } else {
                coords* first_cell_beyond_rim_coords = first_potential_exit_point;
                lake->outflow_points[-1] = new pair<coords*,bool>
                    (find_non_local_outflow_point(first_cell_beyond_rim_coords),false);
            }
        } else {
            for(map<int,coords*>::iterator entry = lake->spill_points.begin();
                entry != lake->spill_points.end();++entry){
                int other_lake = entry->first;
                coords* spill_point = entry->second;
                coords* spill_point_coarse_coords =
                    _coarse_grid->convert_fine_coords(spill_point,
                                                      _grid_params);
                coords* lake_center_coarse_coords =
                    _coarse_grid->convert_fine_coords(lakes[other_lake]->center_coords,
                                                     _grid_params);
                if ((*spill_point_coarse_coords) == (*lake_center_coarse_coords)) {
                    lake->outflow_points[other_lake] = new pair<coords*,bool>(null_coords->clone(),
                                                                              true);
                } else {
                    lake->outflow_points[other_lake] =
                        new pair<coords*,bool>(find_non_local_outflow_point(spill_point),false);
                }
                delete spill_point_coarse_coords;
                delete lake_center_coarse_coords;
            }
        }
    }
    for (coords* entry : sink_points) {
        if (*entry != *null_coords){
            delete entry;
        }
    }
}

coords* basin_evaluation_algorithm::find_non_local_outflow_point(coords* first_cell_beyond_rim_coords) {
    coords* outflow_coords;
    if (check_if_fine_cell_is_sink(first_cell_beyond_rim_coords)) {
            outflow_coords =
                _coarse_grid->convert_fine_coords(first_cell_beyond_rim_coords,
                                                  _grid_params);
    } else {
        int prior_fine_catchment_num = (*prior_fine_catchment_nums)(first_cell_beyond_rim_coords);
        coords* catchment_outlet_coarse_coords = null_coords;
        if ((*sink_points[prior_fine_catchment_num]) != (*null_coords)) {
            catchment_outlet_coarse_coords = sink_points[prior_fine_catchment_num];
        } else {
            coords* current_coords = first_cell_beyond_rim_coords->clone();
            while(true) {
                bool is_sink; coords* downstream_coords;
                tie(is_sink,downstream_coords) =
                    check_for_sinks_and_get_downstream_coords(current_coords);
                if (is_sink) {
                    catchment_outlet_coarse_coords =
                        _coarse_grid->convert_fine_coords(current_coords,
                                                          _grid_params);
                    delete current_coords;
                    delete downstream_coords;
                    break;
                }
                delete current_coords;
                current_coords = downstream_coords;
            }
            if ((*catchment_outlet_coarse_coords) == (*null_coords)) {
                throw runtime_error("Sink point for non local secondary redirect not found");
            }
            sink_points[prior_fine_catchment_num] = catchment_outlet_coarse_coords;
        }
        int coarse_catchment_number =
            (*coarse_catchment_nums)(catchment_outlet_coarse_coords);
        coords* coarse_first_cell_beyond_rim_coords =
            _coarse_grid->convert_fine_coords(first_cell_beyond_rim_coords,
                                              _grid_params);
        outflow_coords = coarse_search_alg->search(function<bool(coords*)>
                                                   ([this,coarse_catchment_number](coords* coords_in) {
                                                   return (*coarse_catchment_nums)(coords_in) ==
                                                   coarse_catchment_number; }),
                                                   function<bool(coords*)>
                                                   ([](coords* coords_in) { return false; }),
                                                   coarse_first_cell_beyond_rim_coords);
        delete coarse_first_cell_beyond_rim_coords;
    }
    return outflow_coords;
}

vector<double>* basin_evaluation_algorithm::get_lakes_as_array() {
    int array_offset = 1;
    store_to_array store_to_array_inst;
    store_to_array_inst.set_null_coords(null_coords);
    for(vector<lake_variables*>::iterator entry = lakes.begin();
        entry != lakes.end();++entry){
        lake_variables* lake = *entry;
        store_to_array_inst.add_object();
        store_to_array_inst.add_number(lake->lake_number+array_offset);
        if (lake->primary_lake != null_lake_number) {
            store_to_array_inst.add_number(lake->primary_lake+array_offset);
        } else {
            store_to_array_inst.add_number(-1);
        }
        vector<int>* secondary_lakes_with_offset = new vector<int>();
        if (! lake->secondary_lakes->empty()) {
            for (int secondary_lake : *lake->secondary_lakes) {
                secondary_lakes_with_offset->push_back(secondary_lake+array_offset);
            }
            store_to_array_inst.add_field(secondary_lakes_with_offset);
        } else {
            //Add empty field
            store_to_array_inst.add_field(secondary_lakes_with_offset);
        }
        delete secondary_lakes_with_offset;
        store_to_array_inst.add_coords(lake->center_coords,array_offset,
                                       additional_lat_offset);
        store_to_array_inst.add_filling_order(lake->filling_order,_grid,
                                              array_offset,additional_lat_offset);
        store_to_array_inst.add_outflow_points_dict(lake->outflow_points,
                                                    _grid,
                                                    array_offset,
                                                    additional_lat_offset);
        store_to_array_inst.add_number(lake->lake_lower_boundary_height);
        store_to_array_inst.add_number(lake->filled_lake_area);
        store_to_array_inst.complete_object();
    }
    return store_to_array_inst.complete_array();
}


field<bool>* basin_evaluation_algorithm::get_lake_mask(){
    lake_mask->set_all(false);
    _grid->for_all([&](coords* coords_in){
        if ((*lake_numbers)(coords_in) != null_lake_number){
            (*lake_mask)(coords_in) = true;
        }
        delete coords_in;
    });
    return lake_mask;
}

int basin_evaluation_algorithm::get_number_of_lakes(){
    return lakes.size();
}

vector<lake_variables*> basin_evaluation_algorithm::get_lakes(){
    return lakes;
}

field<int>* basin_evaluation_algorithm::get_lake_numbers(){
    return lake_numbers;
}

latlon_basin_evaluation_algorithm::
    latlon_basin_evaluation_algorithm(bool* minima_in,
                                      double* raw_orography_in,
                                      double* corrected_orography_in,
                                      double* cell_areas_in,
                                      int* prior_fine_rdirs_in,
                                      int* prior_fine_catchment_nums_in,
                                      int* coarse_catchment_nums_in,
                                      int* catchments_from_sink_filling_in,
                                      int additional_lat_offset_in,
                                      grid_params* grid_params_in,
                                      grid_params* coarse_grid_params_in) :
        basin_evaluation_algorithm(minima_in,
                                   raw_orography_in,
                                   corrected_orography_in,
                                   cell_areas_in,
                                   prior_fine_catchment_nums_in,
                                   coarse_catchment_nums_in,
                                   catchments_from_sink_filling_in,
                                   additional_lat_offset_in,
                                   grid_params_in,
                                   coarse_grid_params_in) {
    prior_fine_rdirs = new field<int>(prior_fine_rdirs_in,grid_params_in);
    null_coords = new latlon_coords(-1,-1);
}

latlon_basin_evaluation_algorithm::~latlon_basin_evaluation_algorithm(){
    delete prior_fine_rdirs;
    delete null_coords;
}

pair<bool,coords*> latlon_basin_evaluation_algorithm::
                   check_for_sinks_and_get_downstream_coords(coords* coords_in){
    double rdir = (*prior_fine_rdirs)(coords_in);
    coords* downstream_coords = _grid->calculate_downstream_coords_from_dir_based_rdir(coords_in,rdir);
    double next_rdir = (*prior_fine_rdirs)(downstream_coords);
    return pair<bool,coords*>(rdir == 5.0 || next_rdir == 0.0, downstream_coords);
}

bool latlon_basin_evaluation_algorithm::check_if_fine_cell_is_sink(coords* coords_in) {
    int rdir = (*prior_fine_rdirs)(coords_in);
    return rdir == 5;
}

single_index_basin_evaluation_algorithm::
    single_index_basin_evaluation_algorithm(bool* minima_in,
                                            double* raw_orography_in,
                                            double* corrected_orography_in,
                                            double* cell_areas_in,
                                            int* prior_next_cell_indices_in,
                                            int* prior_fine_catchment_nums_in,
                                            int* coarse_catchment_nums_in,
                                            int* catchments_from_sink_filling_in,
                                            grid_params* grid_params_in,
                                            grid_params* coarse_grid_params_in) :
        basin_evaluation_algorithm(minima_in,
                                   raw_orography_in,
                                   corrected_orography_in,
                                   cell_areas_in,
                                   prior_fine_catchment_nums_in,
                                   coarse_catchment_nums_in,
                                   catchments_from_sink_filling_in,0,
                                   grid_params_in,
                                   coarse_grid_params_in),
        true_sink_value(-5) {
    prior_next_cell_indices = new field<int>(prior_next_cell_indices_in,
                                             grid_params_in);
    null_coords = new generic_1d_coords(-1);
}

pair<bool,coords*> single_index_basin_evaluation_algorithm::
        check_for_sinks_and_get_downstream_coords(coords* coords_in) {
    int next_cell_index = (*prior_next_cell_indices)(coords_in);
    coords* downstream_coords =
        _grid->calculate_downstream_coords_from_index_based_rdir(coords_in,next_cell_index);
    int next_next_cell_index = (*prior_next_cell_indices)(downstream_coords);
    return pair<bool,coords*>((next_cell_index == true_sink_value) &&
                              (next_next_cell_index == outflow_value),
                              downstream_coords);
}

bool single_index_basin_evaluation_algorithm::check_if_fine_cell_is_sink(coords* coords_in) {
    int next_cell_index = (*prior_next_cell_indices)(coords_in);
    return (next_cell_index == true_sink_value);
}

// class LatLonEvaluateBasin:

//
