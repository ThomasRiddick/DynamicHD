#include "algorithms/non_coincident_grid_mapping_algorithm.hpp"
#include "base/grid.hpp"

unstructured_grid_vertex_coords::
  unstructured_grid_vertex_coords(double* vertex_lats_in,
                                  double* vertex_lons_in) {
  vertex_lats = vertex_lats_in;
  vertex_lons = vertex_lons_in;
}

void non_coincident_grid_mapper::generate_pixels_in_cell_mask(coords* cell_coords) {
  create_new_mask();
  coarse_cell_coords = cell_coords;
  generate_cell_bounds();
  generate_areas_to_consider();
  primary_area_to_consider->for_all_section([&](coords* coords_in) {
    check_if_pixel_is_in_cell(coords_in);
  });
  if (secondary_area_to_consider) {
    secondary_area_to_consider->for_all_section([&](coords* coords_in) {
      check_if_pixel_is_in_cell(coords_in);
    });
  }
  if (display_progress) print_progress();
}

void non_coincident_grid_mapper::check_if_pixel_is_in_cell(coords* coords_in) {
  double pixel_center_lat = (*pixel_center_lats)(coords_in);
  double pixel_center_lon = (*pixel_center_lons)(coords_in);
  (*mask)(coords_in) = check_if_pixel_center_is_in_bounds(pixel_center_lat,
                                                          pixel_center_lon);
  delete coords_in;
}

field<int>* non_coincident_grid_mapper::generate_cell_numbers() {
  cell_numbers = new field<int>(fine_grid_params);
  cell_numbers->set_all(0);
  coarse_grid->for_all([&](coords* coords_in) {
    process_cell(coords_in);
  });
  return cell_numbers;
}

void non_coincident_grid_mapper::set_cell_numbers(int* cell_numbers_in) {
    cell_numbers = new field<int>(cell_numbers_in,fine_grid_params);
}

void non_coincident_grid_mapper::process_cell(coords* coords_in) {
    generate_pixels_in_cell_mask(coords_in);
    assign_cell_numbers();
    delete mask;
    delete cell_bounds;
    delete primary_area_to_consider;
    if (secondary_area_to_consider) {
      delete secondary_area_to_consider;
    }
    delete coords_in;
}

void non_coincident_grid_mapper::generate_limits() {
  fine_grid->for_all([&](coords* coords_in) {
    process_pixel_for_limits(coords_in);
  });
}

void non_coincident_grid_mapper::offset_limits(int lat_offset,
                                               int lon_offset) {
  for (int i = 0; i < coarse_grid->get_total_size(); i++){
    section_min_lats->get_array()[i] =
      section_min_lats->get_array()[i] + lat_offset;
    section_max_lats->get_array()[i] =
      section_max_lats->get_array()[i] + lat_offset;
    section_min_lons->get_array()[i] =
      section_min_lons->get_array()[i] + lon_offset;
    section_max_lons->get_array()[i] =
      section_max_lons->get_array()[i] + lon_offset;
  }
}

void icon_icosohedral_cell_latlon_pixel_ncg_mapper::
    process_pixel_for_limits(coords* coords_in) {
  latlon_coords* latlon_coords_in = static_cast<latlon_coords*>(coords_in);
  int i = latlon_coords_in->get_lat();
  int j = latlon_coords_in->get_lon();
  int working_cell_index = (*cell_numbers)(coords_in);
  coords* working_cell_coords =
    new generic_1d_coords(working_cell_index);
  if ((*section_min_lats)(working_cell_coords) > i) {
    (*section_min_lats)(working_cell_coords) = i;
  }
  if ((*section_max_lats)(working_cell_coords) < i) {
    (*section_max_lats)(working_cell_coords) = i;
  }
  if ((*section_min_lons)(working_cell_coords) > j) {
    (*section_min_lons)(working_cell_coords) = j;
  }
  if ((*section_max_lons)(working_cell_coords) < j) {
    (*section_max_lons)(working_cell_coords) = j;
  }
  delete working_cell_coords;
  delete coords_in;
}

void icon_icosohedral_cell_latlon_pixel_ncg_mapper::
    generate_areas_to_consider() {
  double fine_grid_zero_line =
    static_cast<latlon_grid_params*>(fine_grid_params)->get_zero_line();
  int rotated_west_extreme_lon = cell_bounds->west_extreme_lon;
  int rotated_east_extreme_lon = cell_bounds->east_extreme_lon;
  if (rotated_west_extreme_lon < 0  + fine_grid_zero_line) {
    rotated_west_extreme_lon = rotated_west_extreme_lon + 360.0;
  }
  if (rotated_east_extreme_lon < 0 + fine_grid_zero_line) {
    rotated_east_extreme_lon = rotated_east_extreme_lon + 360.0;
  }
  if (rotated_west_extreme_lon > 360.0  + fine_grid_zero_line) {
    rotated_west_extreme_lon = rotated_west_extreme_lon - 360.0;
  }
  if (rotated_east_extreme_lon > 360.0 + fine_grid_zero_line) {
    rotated_east_extreme_lon = rotated_east_extreme_lon - 360.0;
  }
  if (rotated_east_extreme_lon - rotated_west_extreme_lon > 180.0) {
    double temp_for_swapping_lons = rotated_west_extreme_lon;
    rotated_west_extreme_lon = rotated_east_extreme_lon;
    rotated_east_extreme_lon = temp_for_swapping_lons;
  }
  if (rotated_west_extreme_lon < rotated_east_extreme_lon ||
      rotated_east_extreme_lon == 0.0 + fine_grid_zero_line) {
    if (rotated_east_extreme_lon == 0.0 + fine_grid_zero_line) {
      rotated_east_extreme_lon = 360.0 + fine_grid_zero_line;
    }
    primary_area_to_consider =
      generate_area_to_consider(rotated_west_extreme_lon,
                                rotated_east_extreme_lon);
    secondary_area_to_consider = nullptr;
  } else if (rotated_west_extreme_lon > rotated_east_extreme_lon) {
    primary_area_to_consider =
      generate_area_to_consider(0.0 + fine_grid_zero_line,
                                rotated_east_extreme_lon);
    secondary_area_to_consider =
      generate_area_to_consider(rotated_west_extreme_lon,
                                360.0 + fine_grid_zero_line);
  } else throw runtime_error("Error - cell appears to have no width");
}

section_coords* icon_icosohedral_cell_latlon_pixel_ncg_mapper::
    generate_area_to_consider(double area_min_lon,
                              double area_max_lon) {
  section_coords* area_to_consider;
  int area_min_lat_index = 0;
  int area_max_lat_index = 0;
  int area_min_lon_index = 0;
  int area_max_lon_index = 0;
  double working_min_lon = 99999.0;
  double working_max_lon = -99999.0;
  latlon_grid_params* latlon_fine_grid_params =
    static_cast<latlon_grid_params*>(fine_grid_params);
  int nlat = latlon_fine_grid_params->get_nlat();
  int nlon = latlon_fine_grid_params->get_nlon();
  bool minimum_found = false;
  for (int i=0; i < nlat; i++) {
    coords* pixel_in_row_or_column = new latlon_coords(i,0);
    double pixel_center_lat = (*pixel_center_lats)(pixel_in_row_or_column);
    delete pixel_in_row_or_column;
    if ( pixel_center_lat <= cell_bounds->north_extreme_lat
        && ! minimum_found) {
      area_min_lat_index = i;
      minimum_found = true;
    }
    area_max_lat_index = i;
    if (pixel_center_lat <
        cell_bounds->south_extreme_lat) break;
  }
  minimum_found = false;
  for (int i=0; i < nlon; i++) {
    coords* pixel_in_row_or_column = new latlon_coords(0,i);
    double pixel_center_lon = (*pixel_center_lons)(pixel_in_row_or_column);
    delete pixel_in_row_or_column;
    if (pixel_center_lon >= area_min_lon &&
        pixel_center_lon < working_min_lon) {
      area_min_lon_index = i;
      working_min_lon = pixel_center_lon;
    }
    if (pixel_center_lon < area_max_lon &&
        pixel_center_lon > working_max_lon) {
      area_max_lon_index = i;
      working_max_lon = pixel_center_lon;
    }
  }
  area_to_consider =
    new latlon_section_coords(area_min_lat_index,
                              area_min_lon_index,
                              area_max_lat_index - area_min_lat_index + 1,
                              area_max_lon_index - area_min_lon_index + 1);
  return area_to_consider;
}

void  icon_icosohedral_cell_latlon_pixel_ncg_mapper::create_new_mask() {
    mask = new field<bool>(fine_grid_params);
    mask->set_all(false);
}

void icon_icosohedral_cell_latlon_pixel_ncg_mapper::print_progress() {
  generic_1d_coords* generic_1d_coarse_cell_coords =
    static_cast<generic_1d_coords*>(coarse_cell_coords);
  if (generic_1d_coarse_cell_coords->get_index()%
      coarse_grid->get_total_size()/10 == 0) {
    cout << 100*generic_1d_coarse_cell_coords->get_index()/
    coarse_grid->get_total_size() << endl;
  }
}

void icon_icosohedral_cell_latlon_pixel_ncg_mapper::generate_cell_bounds() {
    double vertex_one_lat = get_vertex_coords(coarse_cell_coords,1,true);
    double vertex_two_lat = get_vertex_coords(coarse_cell_coords,2,true);
    double vertex_three_lat = get_vertex_coords(coarse_cell_coords,3,true);
    double vertex_one_lon = get_vertex_coords(coarse_cell_coords,1,false);
    double vertex_two_lon = get_vertex_coords(coarse_cell_coords,2,false);
    double vertex_three_lon = get_vertex_coords(coarse_cell_coords,3,false);
    if (vertex_one_lat == 90.0 || vertex_one_lat == -90.0) {
      vertex_one_lon = vertex_two_lon;
    } else if (vertex_two_lat == 90.0 || vertex_two_lat == -90.0) {
      vertex_two_lon = vertex_three_lon;
    } else if (vertex_three_lat == 90.0 || vertex_three_lat == -90.0) {
      vertex_three_lon = vertex_one_lon;
    }
    if (vertex_two_lon - vertex_one_lon > 180.0) {
      vertex_one_lon = vertex_one_lon + 360.0;
    } else if (vertex_two_lon - vertex_one_lon < -180.0) {
      vertex_two_lon = vertex_two_lon + 360.0;
    }
    if (vertex_three_lon - vertex_one_lon > 180.0) {
      vertex_one_lon = vertex_one_lon + 360.0;
      vertex_two_lon = vertex_two_lon + 360.0;
    } else if (vertex_three_lon - vertex_one_lon < -180.0) {
      vertex_three_lon = vertex_three_lon + 360.0;
    }
    cell_bounds = new bounds();
    cell_bounds->west_extreme_lon =
                     min(min(vertex_one_lon,vertex_two_lon),vertex_three_lon);
    cell_bounds->east_extreme_lon =
                     max(max(vertex_one_lon,vertex_two_lon),vertex_three_lon);
    cell_bounds->south_extreme_lat =
                     min(min(vertex_one_lat,vertex_two_lat),vertex_three_lat);
    cell_bounds->north_extreme_lat =
                     max(max(vertex_one_lat,vertex_two_lat),vertex_three_lat);
}

bool icon_icosohedral_cell_latlon_pixel_ncg_mapper::
      check_if_pixel_center_is_in_bounds(double pixel_center_lat,
                                         double pixel_center_lon) {
    double vertex_one_lat =
      get_vertex_coords(coarse_cell_coords,1,true);
    double vertex_one_lon =
      get_vertex_coords(coarse_cell_coords,1,false);
    double vertex_two_lat =
      get_vertex_coords(coarse_cell_coords,2,true);
    double vertex_two_lon =
      get_vertex_coords(coarse_cell_coords,2,false);
    double vertex_three_lat =
      get_vertex_coords(coarse_cell_coords,3,true);
    double vertex_three_lon =
      get_vertex_coords(coarse_cell_coords,3,false);
    if (vertex_two_lon - vertex_one_lon > 180.0) {
      vertex_two_lon = vertex_two_lon - 360.0;
    } else if(vertex_two_lon - vertex_one_lon < -180.0) {
      vertex_two_lon = vertex_two_lon + 360.0;
    }
    if (vertex_three_lon - vertex_one_lon > 180.0) {
      vertex_three_lon = vertex_three_lon - 360.0;
    } else if(vertex_three_lon - vertex_one_lon < -180.0) {
      vertex_three_lon = vertex_three_lon + 360.0;
    }
    if (pixel_center_lon - vertex_one_lon > 180.0) {
      pixel_center_lon = pixel_center_lon - 360.0;
    } else if(pixel_center_lon - vertex_one_lon < -180.0) {
      pixel_center_lon = pixel_center_lon + 360.0;
    }
    bool is_in_bounds = true;
    for (int i = 0; i < 3; i++) {
      double vertex_temp_lon = vertex_three_lon;
      vertex_three_lon = vertex_two_lon;
      vertex_two_lon = vertex_one_lon;
      vertex_one_lon = vertex_temp_lon;
      double vertex_temp_lat = vertex_three_lat;
      vertex_three_lat = vertex_two_lat;
      vertex_two_lat = vertex_one_lat;
      vertex_one_lat = vertex_temp_lat;
      if (vertex_one_lon - vertex_two_lon == 0 ||
          vertex_two_lat == 90.0 || vertex_two_lat == -90.0) {
        is_in_bounds = is_in_bounds &&
                       signbit(pixel_center_lon - vertex_one_lon) ==
                       signbit(vertex_three_lon - vertex_one_lon);
      } else if (vertex_two_lat - vertex_one_lat == 0  ) {
        is_in_bounds = is_in_bounds &&
                       signbit(pixel_center_lat - vertex_one_lat) ==
                       signbit(vertex_three_lat - vertex_one_lat);
      } else if(vertex_one_lat == 90.0 || vertex_one_lat == -90.0) {
        is_in_bounds = is_in_bounds &&
                       signbit(pixel_center_lon - vertex_two_lon) ==
                       signbit(vertex_three_lon - vertex_two_lon);
      } else {
        double lat_difference_at_pixel_center_lon = pixel_center_lat -
          calculate_line(pixel_center_lon, vertex_one_lon,
                         vertex_two_lon, vertex_one_lat,
                         vertex_two_lat);
        double lat_difference_at_vertex_three_lon = vertex_three_lat -
          calculate_line(vertex_three_lon, vertex_one_lon,
                         vertex_two_lon, vertex_one_lat,
                         vertex_two_lat);
        if (abs(lat_difference_at_pixel_center_lon) < 1.0e-15) {
          is_in_bounds = is_in_bounds &&
                         (lat_difference_at_vertex_three_lon >= 0);
        } else {
          is_in_bounds = is_in_bounds &&
                          signbit(lat_difference_at_pixel_center_lon) ==
                          signbit(lat_difference_at_vertex_three_lon);
        }
      }
    }
  return is_in_bounds;
}

double icon_icosohedral_cell_latlon_pixel_ncg_mapper::
  calculate_line(double x,double x1,double x2,
                 double y1,double y2) {
  return ((y2 - y1)/(x2 - x1))*(x-x1) + y1;
}

double icon_icosohedral_cell_latlon_pixel_ncg_mapper::
    get_vertex_coords(coords* coords_in,
                      int vertex_num_in,
                      bool return_lat) {
  int vertex_num = vertex_num_in;
  double coordinate;
  vertex_coords* vertex_positions = (*cell_vertex_coords)(coords_in);
  unstructured_grid_vertex_coords* unstructured_grid_vertex_positions =
    static_cast<unstructured_grid_vertex_coords*>(vertex_positions);
  double lat_coordinate = unstructured_grid_vertex_positions->
                          vertex_lats[vertex_num-1];
  if (return_lat) {
    coordinate = lat_coordinate;
  } else {
    //If at pole return one of the other coordinates instead
    if (lat_coordinate == 90.0 || lat_coordinate == -90.0) {
      if (vertex_num == 1) {
        vertex_num = 2;
      } else {
        vertex_num = 1;
      }
    }
    coordinate = unstructured_grid_vertex_positions->
                 vertex_lons[vertex_num-1];
    if (longitudal_range_centered_on_zero) {
      if (coordinate < 0.0) {
        coordinate = coordinate + 360.0;
      }
    }
  }
 return coordinate;
}

void icon_icosohedral_cell_latlon_pixel_ncg_mapper::assign_cell_numbers() {
  fine_grid->for_all([&](coords* fine_coords_in) {
    if ((*mask)(fine_coords_in)){
      (*cell_numbers)(fine_coords_in) =
        static_cast<generic_1d_coords*>(coarse_cell_coords)->get_index();
    }
    delete fine_coords_in;
  });
}

tuple<field<int>*,field<int>*,field<int>*,field<int>*>
  icon_icosohedral_cell_latlon_pixel_ncg_mapper::get_limits() {
    return tie(section_min_lats,section_min_lons,section_max_lats,
               section_max_lons);
}

icon_icosohedral_cell_latlon_pixel_ncg_mapper::
    icon_icosohedral_cell_latlon_pixel_ncg_mapper(
    field<double>* pixel_center_lats_in,
    field<double>* pixel_center_lons_in,
    field<vertex_coords_ptr>* cell_vertex_coords_in,
    grid_params* coarse_grid_params_in,
    grid_params* fine_grid_params_in,
    bool longitudal_range_centered_on_zero_in) {
  longitudal_range_centered_on_zero = longitudal_range_centered_on_zero_in;
  pixel_center_lats = pixel_center_lats_in;
  pixel_center_lons = pixel_center_lons_in;
  cell_vertex_coords = cell_vertex_coords_in;
  fine_grid_params = fine_grid_params_in;
  section_min_lats = new field<int>(coarse_grid_params_in);
  section_min_lons = new field<int>(coarse_grid_params_in);
  section_max_lats = new field<int>(coarse_grid_params_in);
  section_max_lons = new field<int>(coarse_grid_params_in);
  section_min_lats->set_all(static_cast<latlon_grid_params*>
                            (fine_grid_params)->get_nlat() + 1);
  section_min_lons->set_all(static_cast<latlon_grid_params*>
                            (fine_grid_params)->get_nlon() + 1);
  section_max_lats->set_all(0);
  section_max_lons->set_all(0);
  coarse_grid = new icon_single_index_grid(coarse_grid_params_in);
  fine_grid = new latlon_grid(fine_grid_params_in);
}
