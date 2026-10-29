#ifndef INCLUDE_SUBFIELD_HPP_
#define INCLUDE_SUBFIELD_HPP_

#include "base/field.hpp"
#include "base/section_coords.hpp"

using namespace std;

template <typename field_type> class subfield {
  protected:
    field<field_type>* internal_field_object;
    grid_params* subfield_grid_params;
  public:
    virtual ~subfield();
    field_type get_value();
    void for_all_edge_cells(function<void(coords*)> func_in);
    bool coords_outside_subfield(coords* coords_in);
    void set_all(field_type value) {internal_field_object->set_all(value);}
    ///Overloaded bracket operator to be a generic indexing operator
    virtual field_type& operator () (coords* coords_in) = 0;
    ///Overloaded bracket operator to be a generic indexing operator (const version)
    virtual field_type operator () (coords* coords_in) const = 0;
    ///Get the underlying array
    field_type* get_array() {return internal_field_object->get_array();}
    ///Get the underlying array (const version)
    field_type* get_array() const {return internal_field_object->get_array();}
};

template <typename field_type>
class latlon_subfield : public subfield<field_type> {
  public:
    latlon_subfield(section_coords* section_coords_in,
                    bool wrap_in);
    ~latlon_subfield() {};
    ///Overloaded bracket operator to be a generic indexing operator
    field_type& operator () (coords* coords_in);
    ///Overloaded bracket operator to be a generic indexing operator (const version)
    field_type operator () (coords* coords_in) const;
  private:
    /// Latitude offset to remove from input coordinates on the full grid to
    /// get coordinates within the grid of the subfield
    int lat_offset;
    /// Longitude offset to remove from input coordinates on the full grid to
    /// get coordinates within the grid of the subfield
    int lon_offset;
    /// Minimum latitude in this subfield
    int lat_min;
    /// Minimum longitude in this subfield
    int lon_min;
    /// Maximum latitude in this subfield
    int lat_max;
    /// Maximum longitude in this subfield
    int lon_max;
    /// Wrap coordinates in this subfield... this should be false unless the subfield
    /// span 360 degree of longitude
    bool wrap;
};

template <typename field_type>
class icon_single_index_subfield : public subfield<field_type> {
  public:
    icon_single_index_subfield(section_coords* section_coords_in,
                               bool wrap) { this->internal_field_object = nullptr;
                                            this->subfield_grid_params = nullptr; };
    ~icon_single_index_subfield() {};
    ///Overloaded bracket operator to be a generic indexing operator
    field_type& operator () (coords* coords_in) {};
    ///Overloaded bracket operator to be a generic indexing operator (const version)
    field_type operator () (coords* coords_in) const {};
  private:
    int* edge_cells;
    int* subfield_indices;
    int* mask;
    generic_1d_coords* outside_subfield_coords;
    int num_points;
    int num_edge_cells;
};

#endif /* INCLUDE_SUBFIELD_HPP_ */
