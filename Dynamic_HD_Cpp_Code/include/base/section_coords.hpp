#include "base/coords.hpp"
#include "base/grid.hpp"

// An abstract class for holding the coordinates of a subsection of a generic
// grid
class section_coords {};

// A concrete subclass of section coords for holding sections of a latitude
// logitude grid
class latlon_section_coords : public section_coords {
protected:
    // The minimum latitude of the section
    int section_min_lat;
    // The minimum longitude of the section
    int section_min_lon;
    // The latitudinal width of the section
    int section_width_lat;
    // The longitudinal width of the section
    int section_width_lon;
    // The actual longitude of the zero line (i.e. vertical edge of the array)
    double zero_line;
public:
  latlon_section_coords() {};
  latlon_section_coords(int section_min_lat_in,int section_min_lon_in,
                        int section_width_lat_in,int section_width_lon_in,
                        double zero_line_in=0.0);
};


class irregular_latlon_section_coords : public latlon_section_coords {
protected:
    int cell_number;
    vector<int>*  list_of_cell_numbers;
    int* cell_numbers;
    int* section_min_lats;
    int* section_min_lons;
    int* section_max_lats;
    int* section_max_lons;
public:
  irregular_latlon_section_coords(int cell_number_in,
                                    int ncells,
                                    int* cell_numbers_in,
                                    int* section_min_lats_in,
                                    int* section_min_lons_in,
                                    int* section_max_lats_in,
                                    int* section_max_lons_in,
                                    int lat_offset_in = 0);
  irregular_latlon_section_coords(vector<int>* list_of_cell_numbers_in,
                                    int ncells,
                                    int* cell_numbers_in,
                                    int* section_min_lats_in,
                                    int* section_min_lons_in,
                                    int* section_max_lats_in,
                                    int* section_max_lons_in);
    ~irregular_latlon_section_coords();
};

class generic_1d_section_coords : public section_coords {
protected:
    int* cell_neighbors;
    int* cell_secondary_neighbors;
    vector<int>* edge_cells;
    int* subfield_indices;
    int* full_field_indices;
    bool* mask;
    int num_points_subarray;
   icon_single_index_grid* grid;
public:
  generic_1d_section_coords(int ncells,
                              int* cell_neighbors_in,
                              int* cell_secondary_neighbors_in,
                              bool* mask_in=nullptr);
    ~generic_1d_section_coords();
    int* get_cell_neighbors();
    int* get_cell_secondary_neighbors();
    bool* get_section_mask();
    vector<int>*  get_edge_cells();
    int* get_subfield_indices();
    int* get_full_field_indices();
};
