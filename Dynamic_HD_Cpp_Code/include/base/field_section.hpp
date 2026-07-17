#include "base/coords.hpp"
#include "base/field.hpp"
#include "base/section_coords.hpp"

// A class to manipulate a limited section of a given field of data using coordinates
// that encompass the entire field of data. Actually can generally manipulate the entire
// field of data; it simply stores the boundaries of the section.
template <typename field_type> class field_section : public field<field_type> {
    // Get the value at the given coordinates
    field_type get_value(coords* coords_in);
    void set_value(coords* coords_in,field_type value);
    // For each cell in the section
    void for_all_section(coords*,function<void(coords*)>);
    // Print the entire data field
    void print_field_section();
};

// A concrete subclass of field section for a latitude longitude grid
template <typename field_type> class latlon_field_section : public field_section<field_type> {
    private:
        // Number of latitudinal points in the field
        int nlat;
        // Number of longitudinal points in the field
        int nlon;
        // Minimum latitude of the section of field that is of interest
        int section_min_lat;
        // Minimum longitude of the section of field that is of interest
        int section_min_lon;
        // Maximum latitude of the section of field that is of interest
        int section_max_lat;
        // Maximum longitude of the section of field that is of interest
        int section_max_lon;
        // Wrap the field east west or not
        bool wrap;
    public:
        // Initialize this latitude longitude field section. Arguments are a pointer to
        // the input data array and the section coords of the section of the field that
        // is of interest
        latlon_field_section(field_type* data_in,section_coords* section_coords_in,
                             grid_params* params_in);
        // Return the number of the latitude points in the entire grid
        int get_nlat();
        // Return the number of the longitude points in the entire grid
        int get_nlon();
        // Return the value of the wrap flag
        bool get_wrap();
        // Getter for section minimum latitude
        int get_section_min_lat();
        // Getter for section minimum longitude
        int get_section_min_lon();
        // Getter for section maximum latitude
        int get_section_max_lat();
        // Getter for section maximum longitude
        int get_section_max_lon();
        // Return an unlimited polymorphic pointer to a value at the given latitude
        // longitude coordinates
        field_type get_value();
        // For each cell in the section
        void for_all_section();
        // Print the entire latitude longitude data field
        void print_field_section();
};

// A concrete subclass of field section for an  icon single index grid
template <typename field_type> class icon_single_index_field_section : public field_section<field_type>  {
    public:
        // A mask with false outside the section and true inside
        field<bool>* mask;
        // Number of points in the field
        int num_points;
        vector<field<bool>*>* cell_neighbors;
        vector<field<bool>*>* cell_secondary_neighbors;
    private:
        // Initialize this latitude longitude field section. Arguments are a pointer to
        // the input data array and the section coords of the section of the field that
        // is of interest
        icon_single_index_field_section();
        field<bool>* get_mask();
        void for_all_section();
        void print_field_section();
        // Get the number of points
        int get_num_points();
};
