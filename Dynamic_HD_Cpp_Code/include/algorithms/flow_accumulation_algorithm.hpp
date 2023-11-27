#include <vector>
#include "base/field.hpp"

class flow_accumulation_algorithm {
  protected:
    field<int>* dependencies = nullptr;
    field<int>* cumulative_flow = nullptr;
    vector<field<int>*> bifurcation_complete = nullptr;
    queue<coords*> q;
    vector<coords*> links;
    coords* external_data_value = nullptr;
    coords* flow_terminates_value = nullptr;
    coords* no_data_value = nullptr;
    coords* no_flow_value = nullptr;
    int max_neighbors;
    int no_bifurcation_value;
    bool search_for_loops = true;
  public:
    void generate_cumulative_flow(bool set_links);
    void update_bifurcated_flows(coords* coords_in);
  protected:
    void set_dependencies(coords* coords_in);
    void add_cells_to_queue(coords* coords_in);
    void process_queue();
    void follow_paths(coords* initial_coords);
    coords* get_external_flow_value();
    coords* get_flow_terminates_value();
    coords* get_no_data_value();
    coords* get_no_flow_value();
    bool check_for_bifurcations_in_cell(coords* coords_in);
    void update_bifurcated_flow();
    void label_loop(coords* start_coords);
    virtual coords* get_next_cell_coords(coords* coords_in) = 0;
    virtual int generate_coords_index(coords* coords_in) = 0;
    virtual void assign_coords_to_link_array(coords* coords_index,
                                             coords* coords_in) = 0;
    virtual coords* is_bifurcated(coords* coords_in,
                                  int layer_in=-1) = 0;
    virtual coords* get_next_cell_bifurcated_coords(coords* coords_in,
                                                    int layer_in=-1) = 0;
}


class latlon_flow_accumulation_algorithm : public flow_accumulation_algorithm{
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
    field<int>* river_directions => null()
    coords* generate_coords_index(coords* coords_in);
    void assign_coords_to_link_array(coords* coords_index,
                                     coords* coords_in);
    coords* get_next_cell_coords(coords* coords_in);
    coords* is_bifurcated(coords* coords_in,
                          int layer_in=-1);
    coords* get_next_cell_bifurcated_coords(coords* coords_in,
                                            int layer_in=-1);
}

class icon_single_index_flow_accumulation_algorithm : public flow_accumulation_algorithm {
  private:
    class(subfield), pointer :: next_cell_index => null()
    class(subfield_ptr), pointer, dimension(:) :: bifurcated_next_cell_index => null()
    coords* generate_coords_index(coords* coords_in);
    void assign_coords_to_link_array(coords* coords_index,
                                     coords* coords_in);
    coords* get_next_cell_coords(coords* coords_in);
    coords* is_bifurcated(coords* coords_in,
                          int layer_in=-1);
    coords* get_next_cell_bifurcated_coords(coords* coords_in,
                                            int layer_in=-1);
}
