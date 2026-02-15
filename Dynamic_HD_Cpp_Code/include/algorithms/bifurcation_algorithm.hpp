#ifndef INCLUDE_BIFURCATION_ALGORITHM_HPP_
#define INCLUDE_BIFURCATION_ALGORITHM_HPP_

#include <vector>
#include <climits>
#include <map>
#include "base/priority_cell_queue.hpp"
#include "base/enums.hpp"
#include "base/coords.hpp"
#include "base/field.hpp"
#include "base/cell.hpp"

using namespace std;

class bifurcation_algorithm {
  public:
    bifurcation_algorithm() {};
    virtual ~bifurcation_algorithm();
    void setup_fields(int* cumulative_flow_in,
                      int* number_of_outflows_in,
                      bool* landsea_mask_in,
                      grid_params* grid_params_in);
    void setup_flags(double cumulative_flow_threshold_fraction_in,
                     int minimum_cells_from_split_to_main_mouth_in,
                     int maximum_cells_from_split_to_main_mouth_in=INT_MAX,
                     bool remove_main_channel_in = false);
    void bifurcate_rivers();
    virtual int get_maximum_bifurcations() = 0;
  protected:
    virtual void transcribe_river_direction(coords* coords_in) = 0;
    virtual void mark_bifurcated_river_direction(coords* initial_coords,
                                                 coords* destination_coords) = 0;
    virtual bool cell_flows_into_cell(coords* source_coords,
                                      coords* destination_coords) = 0;
    virtual void mark_river_direction(coords* initial_coords,
                                      coords* destination_coords) = 0;
    virtual coords* get_next_cell_downstream(coords* initial_coords) = 0;
    virtual void reset_working_flow_directions() = 0;
    void bifurcate_river(pair<coords*,vector<coords*>> river);
    void find_shortest_path_to_main_channel(coords*);
    void process_neighbors(bool allow_coastal_cells);
    void process_neighbor(bool allow_coastal_cells);
    virtual void push_cell(coords* nbr_coords) = 0;
    virtual void push_coastal_cell(coords* cell_coords) = 0;
    void track_main_channel(coords* mouth_coords);
    void process_neighbors_track_main_channel(bool find_highest_flow_only);
    void process_neighbor_track_main_channel();
    void process_neighbor_find_highest_flow();
    priority_cell_queue q;
    map<coords*,vector<coords*>> river_mouths;
    grid* _grid = nullptr;
    grid_params* _grid_params = nullptr;
    field<bool>* completed_cells = nullptr;
    field<bool>* landsea_mask = nullptr;
    field<bool>* major_side_channel_mask = nullptr;
    field<channel_type>* main_channel_mask = nullptr;
    field<int>*  cumulative_flow = nullptr;
    field<int>*  number_of_outflows = nullptr;
    vector<coords*>* neighbors_coords = nullptr;
    coords* center_coords;
    coords* connection_location;
    coords* next_upstream_cell_coords;
    coords* valid_main_channel_start_coords;
    double cumulative_flow_threshold_fraction;
    int cumulative_flow_threshold;
    int minimum_cells_from_split_to_main_mouth;
    int maximum_cells_from_split_to_main_mouth;
    int highest_cumulative_flow_nbrs;
    int cells_from_mouth;
    bool connection_found;
    bool is_first_distributory;
    bool remove_main_channel;
};

class bifurcation_algorithm_latlon : public virtual bifurcation_algorithm {
  public:
    bifurcation_algorithm_latlon() {};
    virtual ~bifurcation_algorithm_latlon();
    void setup_fields(map<pair<int,int>,
                          vector<pair<int,int>>> river_mouths_in,
                      double* rdirs_in,
                      int* cumulative_flow_in,
                      int* number_of_outflows_in,
                      bool* landsea_mask_in,
                      grid_params* grid_params_in);
    double* get_bifurcation_rdirs();
    int get_maximum_bifurcations();
  protected:
    void transcribe_river_direction(coords* coords_in);
    void mark_bifurcated_river_direction(coords* initial_coords,
                                         coords* destination_coords);
    bool cell_flows_into_cell(coords* source_coords,
                              coords* destination_coords);
    void mark_river_direction(coords* initial_coords,
                              coords* destination_coords);
    coords* get_next_cell_downstream(coords* initial_coords);
    void reset_working_flow_directions();
    field<double>* working_rdirs = nullptr;
    field<double>* master_rdirs = nullptr;
    static const int maximum_bifurcations = 7;
    field<double>* bifurcation_rdirs[maximum_bifurcations];
    static constexpr double no_bifurcation_code = -9.0;
};

class bifurcation_algorithm_icon_single_index : public virtual bifurcation_algorithm {
  public:
    bifurcation_algorithm_icon_single_index() {};
    virtual ~bifurcation_algorithm_icon_single_index();
    void setup_fields(map<int,vector<int>> river_mouths_in,
                      int* next_cell_index_in,
                      int* cumulative_flow_in,
                      int* number_of_outflows_in,
                      bool* landsea_mask_in,
                      grid_params* grid_params_in);
    int* get_bifurcation_next_cell_index();
    int get_maximum_bifurcations();
  protected:
    void transcribe_river_direction(coords* coords_in);
    void mark_bifurcated_river_direction(coords* initial_coords,
                                         coords* destination_coords);
    bool cell_flows_into_cell(coords* source_coords,
                              coords* destination_coords);
    void mark_river_direction(coords* initial_coords,
                              coords* destination_coords);
    coords* get_next_cell_downstream(coords* initial_coords);
    void reset_working_flow_directions();
    field<int>* working_next_cell_index = nullptr;
    field<int>* master_next_cell_index = nullptr;
    static const int maximum_bifurcations = 11;
    field<int>* bifurcations_next_cell_index[maximum_bifurcations];
    static const int no_bifurcation_code = -9;
};

#endif /*INCLUDE_BIFURCATION_ALGORITHM_HPP_*/
