#include "base/section_coords.hpp"

latlon_section_coords::latlon_section_coords(int section_min_lat_in,int section_min_lon_in,
                                             int section_width_lat_in,int section_width_lon_in,
                                             double zero_line_in) :
                                             section_min_lat(section_min_lat_in),
                                             section_min_lon(section_min_lon_in),
                                             section_width_lat(section_width_lat_in),
                                             section_width_lon(section_width_lon_in),
                                             zero_line(zero_line_in) {}

irregular_latlon_section_coords::irregular_latlon_section_coords(int cell_number_in,
                                                                 int ncells,
                                                                 int* cell_numbers_in,
                                                                 int* section_min_lats_in,
                                                                 int* section_min_lons_in,
                                                                 int* section_max_lats_in,
                                                                 int* section_max_lons_in,
                                                                 int lat_offset_in) :
        section_max_lons(section_max_lons_in), section_min_lons(section_min_lons_in),
        cell_numbers(cell_numbers_in), cell_number(cell_number_in) {
    section_max_lats = new int[ncells];
    section_min_lats = new int[ncells];
    for (int i = 0; i < ncells; i++) {
        section_max_lats[i] = section_max_lats_in[i] + lat_offset_in;
        section_min_lats[i] = section_min_lats_in[i] + lat_offset_in;
    }
    list_of_cell_numbers = nullptr;
    section_min_lat = section_min_lats[cell_number_in];
    section_min_lon = section_min_lons[cell_number_in];
    section_width_lat =
        section_max_lats[cell_number_in] + 1 -
            section_min_lats[cell_number_in];
    section_width_lon =
        section_max_lons[cell_number_in] + 1 -
            section_min_lons[cell_number_in];
    zero_line = 0.0;
}

irregular_latlon_section_coords::
~irregular_latlon_section_coords() {
    delete section_max_lats;
    delete section_min_lats;
}

irregular_latlon_section_coords::
irregular_latlon_section_coords(vector<int>* list_of_cell_numbers_in,
                                int ncells,
                                int* cell_numbers_in,
                                int* section_min_lats_in,
                                int* section_min_lons_in,
                                int* section_max_lats_in,
                                int* section_max_lons_in) :
        section_min_lons(section_min_lons_in),
        section_max_lons(section_max_lons_in),
        section_min_lats(section_min_lats_in),
        section_max_lats(section_max_lats_in),
        cell_numbers(cell_numbers_in),
        list_of_cell_numbers(list_of_cell_numbers_in) {
    section_min_lats = new int[ncells];
    section_max_lats = new int[ncells];
    cell_number = 0;
    int working_cell_number = (*list_of_cell_numbers_in)[0];
    section_min_lat = section_min_lats_in[working_cell_number];
    section_min_lon = section_min_lons_in[working_cell_number];
    int section_max_lat = section_max_lats_in[working_cell_number];
    int section_max_lon = section_max_lons_in[working_cell_number];
    for (int i = 1; i < ncells; i++) {
        working_cell_number = (*list_of_cell_numbers_in)[i];
        section_min_lat = min(section_min_lats_in[working_cell_number],
                              section_min_lat);
        section_min_lon = min(section_min_lons_in[working_cell_number],
                              section_min_lon);
        section_max_lat = max(section_max_lats_in[working_cell_number],
                              section_max_lat);
        section_max_lon = max(section_max_lons_in[working_cell_number],
                              section_max_lon);
    }
    section_width_lat =
        section_max_lat + 1 - section_min_lat;
    section_width_lon =
        section_max_lon + 1 - section_min_lon;
    zero_line = 0.0;
}

generic_1d_section_coords::generic_1d_section_coords(int ncells,
                                                     int* cell_neighbors_in,
                                                     int* cell_secondary_neighbors_in,
                                                     bool* mask_in) :
        cell_neighbors(cell_neighbors_in),
        cell_secondary_neighbors(cell_secondary_neighbors_in),
        mask(mask_in) {
    edge_cells = nullptr;
    subfield_indices = nullptr;
    full_field_indices = nullptr;
    if (! mask) {
        mask = new bool[ncells];
        for (int i = 0; i < ncells; i++) {
            mask[i] = true;
        }
        num_points_subarray = ncells;
    } else {
        num_points_subarray = 0;
        for (int i = 0; i < ncells; i++) {
            if (mask[i]) num_points_subarray++;
        }
    }
}

generic_1d_section_coords::~generic_1d_section_coords() {
    delete mask;
}

int* generic_1d_section_coords::get_cell_neighbors() {
    if (cell_neighbors) {
        return cell_neighbors;
    } else {
        return grid->generate_subfield_neighbors(num_points_subarray,
                                                 mask,
                                                 get_subfield_indices(),
                                                 get_full_field_indices());
    }
};

bool* generic_1d_section_coords::get_section_mask() {
    return mask;
}

int* generic_1d_section_coords::get_cell_secondary_neighbors() {
    if (cell_secondary_neighbors) {
        return cell_secondary_neighbors;
    } else {
        return grid->generate_subfield_secondary_neighbors(num_points_subarray,
                                                           get_subfield_indices(),
                                                           get_full_field_indices());
    }
}

vector<int>* generic_1d_section_coords::get_edge_cells() {
    if (edge_cells) {
        return edge_cells;
    } else {
        return grid->generate_subfield_edge_cells(num_points_subarray,
                                                  get_cell_neighbors(),
                                                  get_cell_secondary_neighbors());
    }
}

int* generic_1d_section_coords::get_subfield_indices() {
    if (subfield_indices) {
        return subfield_indices;
    } else {
        return grid->generate_subfield_indices(mask);
    }
}

int* generic_1d_section_coords::get_full_field_indices() {
    if (subfield_indices) {
        return full_field_indices;
    } else {
        return grid->generate_full_field_indices(num_points_subarray,
                                                 mask,
                                                 get_subfield_indices());
    }
}
