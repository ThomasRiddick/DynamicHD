#include "base/subfield.hpp"

template <typename field_type>
    subfield<field_type>::~subfield() {
    if (internal_field_object) delete internal_field_object;
    if (subfield_grid_params) delete subfield_grid_params;
};

template <typename field_type>
    latlon_subfield<field_type>::latlon_subfield(section_coords* section_coords_in,
                                                 bool wrap_in) : wrap(wrap_in) {
    latlon_section_coords* latlon_section_coords_in =
        static_cast<latlon_section_coords*>(section_coords_in);
    lat_offset = latlon_section_coords_in->section_min_lat;
    lon_offset = latlon_section_coords_in->section_min_lon;
    lat_min    = latlon_section_coords_in->section_min_lat;
    lat_max    = latlon_section_coords_in->section_width_lat + lat_offset - 1;
    lon_min    = latlon_section_coords_in->section_min_lon;
    lon_max    = latlon_section_coords_in->section_width_lon + lon_offset - 1;
    this->subfield_grid_params =
        new latlon_grid_params(latlon_section_coords_in->section_width_lat,
                               latlon_section_coords_in->section_width_lon,
                               ! wrap_in);
    this->internal_field_object = new field<field_type>(this->subfield_grid_params);
}

template <typename field_type>
    field_type& latlon_subfield<field_type>::operator () (coords* coords_in) {
    latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
    int lat; int lon;
    if ( this->lat_max >= latlon_coords_in->get_lat() &&
         latlon_coords_in->get_lat() >= this->lat_min) {
        lat = latlon_coords_in->get_lat();
    } else if (this->lat_max < latlon_coords_in->get_lat()) {
        lat = this->lat_max;
    } else {
        lat = this->lat_min;
    }
    if ( this->lon_max >= latlon_coords_in->get_lon() && latlon_coords_in->get_lon() >= this->lon_min) {
        lon = latlon_coords_in->get_lon();
    } else if (this->lon_max < latlon_coords_in->get_lon()) {
        if (! this->wrap) {
            lon = this->lon_max;
        } else {
            lon = latlon_coords_in->get_lon() - this->lon_max;
        }
    } else {
        if (! this->wrap) {
            lon = this->lon_min;
        } else {
            lon = this->lon_max - this->lon_min + 1 + latlon_coords_in->get_lon();
        }
    }
    coords* new_coords = new latlon_coords(lat-this->lat_offset,
                                           lon-this->lon_offset);
    field_type& value = (*this->internal_field_object)(new_coords);
    delete new_coords;
    return value;
};

template <typename field_type>
    field_type latlon_subfield<field_type>::operator () (coords* coords_in) const{
    latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
    int lat; int lon;
    if ( this->lat_max >= latlon_coords_in->get_lat() &&
         latlon_coords_in->get_lat() >= this->lat_min) {
        lat = latlon_coords_in->get_lat();
    } else if (this->lat_max < latlon_coords_in->get_lat()) {
        lat = this->lat_max;
    } else {
        lat = this->lat_min;
    }
    if ( this->lon_max >= latlon_coords_in->get_lon() && latlon_coords_in->get_lon() >= this->lon_min) {
        lon = latlon_coords_in->get_lon();
    } else if (this->lon_max < latlon_coords_in->get_lon()) {
        if (! this->wrap) {
            lon = this->lon_max;
        } else {
            lon = latlon_coords_in->get_lon() - this->lon_max;
        }
    } else {
        if (! this->wrap) {
            lon = this->lon_min;
        } else {
            lon = this->lon_max - this->lon_min + 1 + latlon_coords_in->get_lon();
        }
    }
    coords* new_coords = new latlon_coords(lat-this->lat_offset,
                                           lon-this->lon_offset);
    field_type value = (*this->internal_field_object)(new_coords);
    delete new_coords;
    return value;
};

template class latlon_subfield<bool>;
template class latlon_subfield<int>;
