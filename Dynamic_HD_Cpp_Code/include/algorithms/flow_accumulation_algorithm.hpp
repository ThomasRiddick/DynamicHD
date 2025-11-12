#include <vector>
#include <queue>
#include "base/field.hpp"
using namespace std;

class flow_accumulation_algorithm {
  protected:
    grid* _grid = nullptr;
    grid_params* _grid_params = nullptr;
    field<int>* dependencies = nullptr;
    field<int>* cumulative_flow = nullptr;
    vector<field<bool>*> bifurcation_complete;
    queue<coords*> q;
    vector<coords*> links;
    coords* external_data_value = nullptr;
    coords* flow_terminates_value = nullptr;
    coords* no_data_value = nullptr;
    coords* no_flow_value = nullptr;
    int acc_no_data_value = -3;
    int acc_no_flow_value = -4;
    int max_neighbors;
    int no_bifurcation_value;
    bool search_for_loops = true;
  public:
    virtual ~flow_accumulation_algorithm();
    void generate_cumulative_flow(bool set_links);
    void update_bifurcated_flows();
  protected:
    void setup_fields(int* cumulative_flow,
                      grid_params* grid_params_in);
    void set_dependencies(coords* coords_in);
    void add_cells_to_queue(coords* coords_in);
    void process_queue();
    void follow_paths(coords* initial_coords);
    void check_for_loops(coords* coords_in);
    coords* get_external_flow_value();
    coords* get_flow_terminates_value();
    coords* get_no_data_value();
    coords* get_no_flow_value();
    void check_for_bifurcations_in_cell(coords* coords_in);
    void update_bifurcated_flow(coords* coords_in,
                                int additional_accumulated_flow);
    void label_loop(coords* start_coords);
    virtual coords* get_next_cell_coords(coords* coords_in) = 0;
    virtual int generate_coords_index(coords* coords_in) = 0;
    virtual void assign_coords_to_link_array(int coords_index,
                                             coords* coords_in) = 0;
    virtual bool is_bifurcated(coords* coords_in,
                               int layer_in=-1) = 0;
    virtual coords* get_next_cell_bifurcated_coords(coords* coords_in,
                                                    int layer_in) = 0;
};


class latlon_flow_accumulation_algorithm : public flow_accumulation_algorithm{
  public:
    void setup_fields(int* next_cell_index_lat,
                      int* next_cell_index_lon,
                      int* cumulative_flow,
                      grid_params* grid_params_in);
    virtual ~latlon_flow_accumulation_algorithm();
  private:
    int tile_min_lat;
    int tile_max_lat;
    int tile_min_lon;
    int tile_max_lon;
    int tile_width_lat;
    int tile_width_lon;
    field<int>* next_cell_index_lat = nullptr;
    field<int>* next_cell_index_lon = nullptr;
    vector<field<int>*> bifurcated_next_cell_index_lat;
    vector<field<int>*> bifurcated_next_cell_index_lon;
    field<int>* river_directions = nullptr;

    int generate_coords_index(coords* coords_in);
    void assign_coords_to_link_array(int coords_index,
                                     coords* coords_in);
    coords* get_next_cell_coords(coords* coords_in);
    bool is_bifurcated(coords* coords_in,
                       int layer_in=-1);
    coords* get_next_cell_bifurcated_coords(coords* coords_in,
                                            int layer_in);
};

class icon_single_index_flow_accumulation_algorithm : public flow_accumulation_algorithm {
  public:
    void setup_fields(int* next_cell_index,
                      int* cumulative_flow,
                      grid_params* grid_params_in,
                      int* bifurcated_next_cell_index_in = nullptr);
    virtual ~icon_single_index_flow_accumulation_algorithm();
  private:
    field<int>* next_cell_index = nullptr;
    vector<field<int>*> bifurcated_next_cell_index;
    int generate_coords_index(coords* coords_in);
    void assign_coords_to_link_array(int coords_index,
                                     coords* coords_in);
    coords* get_next_cell_coords(coords* coords_in);
    bool is_bifurcated(coords* coords_in,
                       int layer_in=-1);
    coords* get_next_cell_bifurcated_coords(coords* coords_in,
                                            int layer_in);
};
