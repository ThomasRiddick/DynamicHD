#include <iostream>
#include <numbers>
#include "drivers/cross_grid_mapper.hpp"
#include "base/grid.hpp"
#include "algorithms/non_coincident_grid_mapping_algorithm.hpp"
using namespace std;
using namespace std::numbers;

const double abs_tol_for_deg = 0.00001;

void cross_grid_mapper_latlon_to_icon(double* pixel_center_lats,
                                      double* pixel_center_lons,
                                      double* cell_vertices_lats,
                                      double* cell_vertices_lons,
                                      int* cell_neighbors,
                                      int* output_cell_numbers,
                                      int nlat_fine, int nlon_fine,
                                      int ncells_coarse,
                                      bool longitudal_range_centered_on_zero) {
    cout << "Entering C++ Cross Grid Mapping Generation Code" << endl;

    double fine_grid_zero_line;
    if (pixel_center_lons[0] < -1.0) {
        fine_grid_zero_line = -180.0;
    } else {
        fine_grid_zero_line = 0.0;
    }
    latlon_grid_params* fine_grid_params =
        new latlon_grid_params(nlat_fine,nlon_fine,true,fine_grid_zero_line);
    bool use_secondary_neighbors = true;
    int* secondary_neighboring_cell_indices = nullptr;
    icon_single_index_grid_params* coarse_grid_params =
        new icon_single_index_grid_params(ncells_coarse,
                                          cell_neighbors,
                                          use_secondary_neighbors,
                                          secondary_neighboring_cell_indices);
    vertex_coords_ptr* cell_vertex_coords_data = new vertex_coords_ptr[ncells_coarse];
    for (int i=0; i<ncells_coarse; i++){
      for(int j = 0; j < 3; j++){
        cell_vertices_lats[i*3+j] = cell_vertices_lats[i*3+j]*(180.0/pi);
        if(cell_vertices_lats[i*3+j] > 90.0 - abs_tol_for_deg) {
          cell_vertices_lats[i*3+j] = 90.0;
        }
        if(cell_vertices_lats[i*3+j] < -90.0 + abs_tol_for_deg) {
          cell_vertices_lats[i*3+j] = -90.0;
        }
        cell_vertices_lons[i*3+j] = cell_vertices_lons[i*3+j]*(180.0/pi);
      }
    }
    for(int i = 0; i < ncells_coarse; i++) {
        cell_vertex_coords_data[i] =
                new unstructured_grid_vertex_coords(&cell_vertices_lats[i*3],
                                                    &cell_vertices_lons[i*3]);
    }
    field<vertex_coords_ptr>* cell_vertex_coords =
        new field<vertex_coords_ptr>(cell_vertex_coords_data,coarse_grid_params);
    double* pixel_center_lats_2d = new double[nlon_fine*nlat_fine];
    for (int i = 0; i < nlon_fine; i++){
      for (int j = 0; j < nlat_fine; j++) {
        pixel_center_lats_2d[i+j*nlon_fine] = pixel_center_lats[j]*(180.0/pi);
      }
    }
    double* pixel_center_lons_2d = new double[nlon_fine*nlat_fine];
    for (int i = 0; i < nlon_fine; i++){
      for (int j = 0; j < nlat_fine; j++) {
        pixel_center_lons_2d[i+j*nlon_fine] = pixel_center_lons[i]*(180.0/pi);
      }
    }
    field<double>* pixel_center_lats_field =
      new field<double>(pixel_center_lats_2d,fine_grid_params);
    field<double>* pixel_center_lons_field =
      new field<double>(pixel_center_lons_2d,fine_grid_params);
    auto ncg_mapper =
        icon_icosohedral_cell_latlon_pixel_ncg_mapper(pixel_center_lats_field,
                                                      pixel_center_lons_field,
                                                      cell_vertex_coords,
                                                      coarse_grid_params,
                                                      fine_grid_params,
                                                      longitudal_range_centered_on_zero);
    field<int>* cell_numbers = ncg_mapper.generate_cell_numbers();
    int* cell_numbers_ptr = cell_numbers->get_array();
    for (int i = 0; i < nlat_fine*nlon_fine; i++){
      output_cell_numbers[i] = cell_numbers_ptr[i];
    }
}
