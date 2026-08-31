import os.path as path
import numpy as np
import cross_grid_mapper_wrapper

def cross_grid_mapper_latlon_to_icon(cell_neighbors,
                                     pixel_center_lats,
                                     pixel_center_lons,
                                     cell_vertices_lats,
                                     cell_vertices_lons,
                                     longitudal_range_centered_on_zero=True):
    nlat = len(pixel_center_lats)
    nlon = len(pixel_center_lons)
    output_cell_numbers = np.empty(shape=(nlat,nlon),dtype=np.int32)
    cross_grid_mapper_wrapper.cross_grid_mapper_latlon_to_icon_cpp(
        np.ascontiguousarray(pixel_center_lats,dtype=np.float64),
        np.ascontiguousarray(pixel_center_lons,dtype=np.float64),
        np.ascontiguousarray(cell_vertices_lats,dtype=np.float64),
        np.ascontiguousarray( cell_vertices_lons,dtype=np.float64),
        np.ascontiguousarray(cell_neighbors,dtype=np.int32),
        output_cell_numbers,
        longitudal_range_centered_on_zero)
    return output_cell_numbers
