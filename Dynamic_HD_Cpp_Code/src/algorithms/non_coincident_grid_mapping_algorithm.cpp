unstructured_grid_vertex_coords(double* vertex_lats_in,
                                double* vertex_lons_in) {
  vertex_lats = vertex_lats_in;
  vertex_lons = vertex_lons_in;
}

void generate_pixels_in_cell_mask(coords* cell_coords) {
  create_new_mask();
  coarse_cell_coords = cell_coords;
  generate_cell_bounds();
  generate_areas_to_consider();
  area_to_consider_mask->for_all(check_if_pixel_is_in_cell);
  if (secondary_area_to_consider_mask) {
    secondary_area_to_consider_mask->for_all(check_if_pixel_is_in_cell);
  }
  if (display_progress) print_progress();
}

void check_if_pixel_is_in_cell(coords* coords_in) {
  double pixel_center_lat = (*pixel_center_lats)(coords_in);
  double pixel_center_lon = (*pixel_center_lons)(coords_in);
  (*mask)(coords_in) = check_if_pixel_center_is_in_bounds(pixel_center_lat,
                                                          pixel_center_lon);
  delete coords_in;
}

function generate_cell_numbers() {
  ???
  field_section*  cell_numbers;
  _grid->for_all_section(process_cell);
  return cell_numbers;
}

void set_cell_numbers(???? cell_numbers_in) {
    cell_numbers = cell_numbers_in
}

void process_cell(coords* coords_in) {
    generate_pixels_in_cell_mask(coords_in);
    assign_cell_numbers();
    mask->deallocate_data();
    delete mask;
    delete cell_bounds;
    area_to_consider_mask->destructor();
    delete area_to_consider_mask;
    if (secondary_area_to_consider_mask) {
      secondary_area_to_consider_mask->destructor()
      delete secondary_area_to_consider_mask;
    }
    delete coords_in;
}

void generate_limits() {
  ????
  _grid->for_all_section(process_pixel_for_limits);
}

void offset_limits(int lat_offset) {
  section_min_lats(:) = section_min_lats(:) + lat_offset;
  section_max_lats(:) = section_max_lats(:) + lat_offset;
}

void latlon_process_pixel_for_limits(coords* coords_in) {
  int i = coords_in->lat;
  int j = coords_in->lon;
  int working_cell_number = (*cell_numbers)(coords_in);
  if (section_min_lats(working_cell_number) > i) {
    section_min_lats(working_cell_number) = i;
  }
  if (section_max_lats(working_cell_number) < i) {
    section_max_lats(working_cell_number) = i;
  }
  if (section_min_lons(working_cell_number) > j) {
    section_min_lons(working_cell_number) = j;
  }
  if (section_max_lons(working_cell_number) < j) {
    section_max_lons(working_cell_number) = j;
  }
  delete coords_in;
}

void latlon_pixel_generate_areas_to_consider() {
  double fine_grid_zero_line = latlon_fine_grid_shape->zero_line;
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
    area_to_consider_mask =
      latlon_pixel_generate_area_to_consider(rotated_west_extreme_lon,
                                             rotated_east_extreme_lon);
    secondary_area_to_consider_mask = nullptr;
  } else if (rotated_west_extreme_lon > rotated_east_extreme_lon) {
    area_to_consider_mask =
      latlon_pixel_generate_area_to_consider(0.0 + fine_grid_zero_line,
                                             rotated_east_extreme_lon);
    secondary_area_to_consider_mask =
      latlon_pixel_generate_area_to_consider(rotated_west_extreme_lon,
                                             360.0 + fine_grid_zero_line);
  } else throw runtime_error("Error - cell appears to have no width");
}

function latlon_pixel_generate_area_to_consider(double area_min_lon,
                                                double area_max_lon) {
  class(latlon_section_coords), pointer :: area_to_consider_section_coords
  class(*),pointer,dimension(:,:) :: mask
  class(subfield), pointer  :: area_to_consider
  int area_min_lat_index = 0;
  int area_max_lat_index = 0;
  int area_min_lon_index = 0;
  int area_max_lon_index = 0;
  double working_min_lon = 99999.0;
  double working_max_lon = -99999.0;
  CAST latlon_fine_grid_shape => fine_grid_shape
  int nlat = latlon_fine_grid_shape->section_width_lat
  int nlon = latlon_fine_grid_shape->section_width_lon
  bool minimum_found = false;
  for (int i=1; i <= nlat; i++) {
    coords* pixel_in_row_or_column = new latlon_coords(i,1);
    double pixel_center_lat = pixel_center_lats->get_value(pixel_in_row_or_column)
    delete pixel_in_row_or_column;
    if ( pixel_center_lat <= cell_bounds->north_extreme_lat
        && ! minimum_found) {
      area_min_lat_index = i
      minimum_found = true
    }
    area_max_lat_index = i
    if (pixel_center_lat <
        cell_bounds->south_extreme_lat) break;
  }
  minimum_found = false;
  for (int i=1; i <= nlon; i++) {
    coords* pixel_in_row_or_column = new latlon_coords(1,i);
    double pixel_center_lon = pixel_center_lons->get_value(pixel_in_row_or_column);
    delete pixel_in_row_or_column;
    if (pixel_center_lon >= area_min_lon &&
        pixel_center_lon < working_min_lon) {
      area_min_lon_index = i
      working_min_lon = pixel_center_lon
    }
    if (pixel_center_lon < area_max_lon &&
        pixel_center_lon > working_max_lon) {
      area_max_lon_index = i
      working_max_lon = pixel_center_lon
    }
  }
  allocate(area_to_consider_section_coords,
           source=latlon_section_coords(area_min_lat_index,
                                        area_min_lon_index,
                                        area_max_lat_index - area_min_lat_index + 1,
                                        area_max_lon_index - area_min_lon_index + 1))
  allocate(logical::mask(1:area_max_lat_index - area_min_lat_index + 1,
                         1:area_max_lon_index - area_min_lon_index + 1))
  mask = false
  area_to_consider=> latlon_subfield(mask,area_to_consider_section_coords)
  delete area_to_consider_section_coords;
  return area_to_consider;
}

void latlon_pixel_create_new_mask() {
    allocate(logical::new_mask(1:fine_grid_shape->section_width_lat,
                              1:fine_grid_shape->section_width_lon))
    new_mask = false;
    mask = new latlon_field_section(new_mask,fine_grid_shape)
}

void icon_icosohedral_cell_print_progress() {
  if(mod(coarse_cell_coords->index,cell_vertex_coords->get_num_points()/10) == 0) {
    cout << 100*coarse_cell_coords->index/cell_vertex_coords->get_num_points() << endl;
  }
}

void icon_icosohedral_cell_generate_cell_bounds() {
    double vertex_one_lat = icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,1,true);
    double vertex_two_lat = icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,2,true);
    double vertex_three_lat = icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,3,true);
    double vertex_one_lon = icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,1,false);
    double vertex_two_lon = icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,2,false);
    double vertex_three_lon = icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,3,false);
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
    cell_bounds->west_extreme_lon =
                     min(min(vertex_one_lon,vertex_two_lon),vertex_three_lon);
    cell_bounds->east_extreme_lon =
                     max(max(vertex_one_lon,vertex_two_lon),vertex_three_lon);
    cell_bounds->south_extreme_lat =
                     min(min(vertex_one_lat,vertex_two_lat),vertex_three_lat);
    cell_bounds->north_extreme_lat =
                     max(max(vertex_one_lat,vertex_two_lat),vertex_three_lat);
}

bool icon_icosohedral_cell_check_if_pixel_center_is_in_bounds(double pixel_center_lat,
                                                              double pixel_center_lon) {
    double vertex_one_lat =
      icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,1,true);
    double vertex_one_lon =
      icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,1,false);
    double vertex_two_lat =
      icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,2,true);
    double vertex_two_lon =
      icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,2,false);
    double vertex_three_lat =
      icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,3,true);
    double vertex_three_lon =
      icon_icosohedral_cell_get_vertex_coords(coarse_cell_coords,3,false);
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
    is_in_bounds = true;
    for (int i = 0; i < 3; i++) {
      int vertex_temp_lon = vertex_three_lon;
      vertex_three_lon = vertex_two_lon;
      vertex_two_lon = vertex_one_lon;
      vertex_one_lon = vertex_temp_lon;
      int vertex_temp_lat = vertex_three_lat;
      vertex_three_lat = vertex_two_lat;
      vertex_two_lat = vertex_one_lat;
      vertex_one_lat = vertex_temp_lat;
      if (vertex_one_lon - vertex_two_lon == 0 ||
          vertex_two_lat == 90.0 || vertex_two_lat == -90.0) {
        is_in_bounds = is_in_bounds &&
                       signbit(1.0,pixel_center_lon - vertex_one_lon) ==
                       signbit(1.0,vertex_three_lon - vertex_one_lon);
      } else if (vertex_two_lat - vertex_one_lat == 0  ) {
        is_in_bounds = is_in_bounds &&
                       signbit(1.0,pixel_center_lat - vertex_one_lat) ==
                       signbit(1.0,vertex_three_lat - vertex_one_lat);
      } else if(vertex_one_lat == 90.0 || vertex_one_lat == -90.0) {
        is_in_bounds = is_in_bounds &&
                       signbit(1.0,pixel_center_lon - vertex_two_lon) ==
                       signbit(1.0,vertex_three_lon - vertex_two_lon);
      } else {
        double lat_difference_at_pixel_center_lon = pixel_center_lat -
          icon_icosohedral_cell_calculate_line(pixel_center_lon, vertex_one_lon,
                                               vertex_two_lon, vertex_one_lat,
                                               vertex_two_lat);
        double lat_difference_at_vertex_three_lon = vertex_three_lat -
          icon_icosohedral_cell_calculate_line(vertex_three_lon, vertex_one_lon,
                                               vertex_two_lon, vertex_one_lat,
                                               vertex_two_lat);
        if (lat_difference_at_pixel_center_lon == 0) {
          is_in_bounds = is_in_bounds &&
                         (lat_difference_at_vertex_three_lon >= 0);
        } else {
          is_in_bounds = is_in_bounds &&
                          signbit(1.0,
                                  lat_difference_at_pixel_center_lon) ==
                          signbit(1.0,
                                  lat_difference_at_vertex_three_lon);
        }
      }
    }
  return is_in_bounds;
}

double icon_icosohedral_cell_calculate_line(double x,double x1,double x2,
                                            double y1,double y2) {
  return ((y2 - y1)/(x2 - x1))*(x-x1) + y1;
}

double icon_icosohedral_cell_get_vertex_coords(coords* coords_in,
                                               int vertex_num_in,
                                               bool return_lat) {
  int vertex_num = vertex_num_in;
  double coordinate;
  vertex_coords* vertex_positions = cell_vertex_coords->get_value(coords_in);
  double lat_coordinate = vertex_positions->vertex_lats(vertex_num);
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
    coordinate = vertex_positions->vertex_lons(vertex_num)
    if (longitudal_range_centered_on_zero) {
      if (coordinate < 0.0) {
        coordinate = coordinate + 360.0;
      }
    }
  }
 delete vertex_positions_ptr;
 return coordinate;
}

void icon_icosohedral_cell_latlon_pixel_assign_cell_numbers() {
    select type (mask=>mask)
    type is (latlon_field_section)
      select type (data=>mask->data)
      type is (logical)
        select type(cell_numbers => cell_numbers)
        type is (latlon_field_section)
          select type (cell_numbers_data => cell_numbers->data)
          type is (integer)
            select type (coarse_cell_coords=>coarse_cell_coords)
            type is (generic_1d_coords)
                where(data) cell_numbers_data = coarse_cell_coords->index
            end select
          end select
        end select
      end select
    end select
}

void init_icon_icosohedral_cell_latlon_pixel_ncg_mapper(pixel_center_lats,
                                                        pixel_center_lons,
                                                        cell_vertex_coords,
                                                        fine_grid_shape,
                                                        longitudal_range_centered_on_zero_in = false) {
  class(latlon_field_section), pointer, intent(in)  :: pixel_center_lats
  class(latlon_field_section), pointer, intent(in)  :: pixel_center_lons
  class(icon_single_index_field_section), pointer, intent(inout)  :: cell_vertex_coords
  class(latlon_section_coords),pointer,intent(in) :: fine_grid_shape
  class(*),pointer,dimension(:,:) :: new_cell_numbers
  bool longitudal_range_centered_on_zero = longitudal_range_centered_on_zero_in;
  pixel_center_lats => pixel_center_lats
  pixel_center_lons => pixel_center_lons
  cell_vertex_coords => cell_vertex_coords
  fine_grid_shape => fine_grid_shape
  allocate(integer::new_cell_numbers(1:fine_grid_shape->section_width_lat,
                                     1:fine_grid_shape->section_width_lon))
  allocate(section_min_lats(cell_vertex_coords->get_num_points()))
  allocate(section_min_lons(cell_vertex_coords->get_num_points()))
  allocate(section_max_lats(cell_vertex_coords->get_num_points()))
  allocate(section_max_lons(cell_vertex_coords->get_num_points()))
  section_min_lats(:) = fine_grid_shape->section_width_lat + 1
  section_min_lons(:) = fine_grid_shape->section_width_lon + 1
  section_max_lats(:) = 0
  section_max_lons(:) = 0
  ALL new_cell_numbers = 0
  cell_numbers => latlon_field_section(new_cell_numbers,fine_grid_shape)
}

void icon_icosohedral_cell_latlon_pixel_ncg_mapper_destructor() {
    delete section_min_lats;
    delete section_min_lons;
    delete section_max_lats;
    delete section_max_lons;
    cell_numbers->deallocate_data()
    delete cell_numbers;
}
