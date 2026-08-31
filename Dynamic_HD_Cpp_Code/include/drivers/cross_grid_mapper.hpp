void cross_grid_mapper_latlon_to_icon(double* pixel_center_lats,
                                      double* pixel_center_lons,
                                      double* cell_vertices_lats,
                                      double* cell_vertices_lons,
                                      int* cell_neighbors,
                                      int* output_cell_numbers,
                                      int nlat_fine, int nlon_fine,
                                      int ncells_coarse,
                                      bool longitudal_range_centered_on_zero = false);
