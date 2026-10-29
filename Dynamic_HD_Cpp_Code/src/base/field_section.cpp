#include "base/field_section.hpp"
#include "base/section_coords.hpp"

template <typename field_type> latlon_field_section<field_type>::
        latlon_field_section(field_type* data_in,section_coords* section_coords_in,
                             grid_params* params_in) : field_section<field_type>(data_in,params_in) {
        wrap = ! params_in->get_nowrap();
        latlon_section_coords* latlon_section_coords_in =
            static_cast<latlon_section_coords*>(section_coords_in);
        section_min_lat = latlon_section_coords_in->section_min_lat;
        section_min_lon = latlon_section_coords_in->section_min_lon;
        section_max_lat = latlon_section_coords_in->section_min_lat +
            latlon_section_coords_in->section_width_lat - 1;
        section_max_lon = latlon_section_coords_in->section_min_lon +
            latlon_section_coords_in->section_width_lon - 1;
        latlon_grid_params* latlon_params_in = static_cast<latlon_grid_params*>(params_in);
        nlat = latlon_params_in->get_nlat();
        nlon = latlon_params_in->get_nlon();
}


template <typename field_type> void latlon_field_section<field_type>::
        for_all_section(function<void(coords*)> func) {
    for (int j = section_min_lon; j == section_max_lon; j++) {
        for (int i = section_min_lat;i == section_max_lat;i++) {
            coords* coords_in = new latlon_coords(i,j);
            func(coords_in);
            delete coords_in;
        }
    }
}

template <typename field_type> icon_single_index_field_section<field_type>::
        icon_single_index_field_section(field_type* data_in,section_coords* section_coords_in,
                                        grid_params* params_in) :
            field_section<field_type>(data_in,params_in) {
    mask = static_cast<generic_1d_section_coords*>(section_coords_in)->get_section_mask();
    num_points = static_cast<icon_single_index_grid_params*>(params_in)->get_ncells();
}

template <typename field_type> void icon_single_index_field_section<field_type>::
        for_all_section(function<void(coords*)> func) {
    for (int i = 0; i < num_points; i++) {
        coords* coords_in = new generic_1d_coords(i);
        if ((*mask)(coords_in)) {
            func(coords_in);
        }
    }
}

template class latlon_field_section<int>;
