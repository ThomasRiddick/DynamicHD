'''
Created on March 23, 2018

@author: thomasriddick
'''
import os
import cdo
import os.path as path
import numpy as np
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.base import field
from Dynamic_HD_Scripts.base import grid
import lake_operators_wrapper
import l2_evaluate_basins_wrapper

def advanced_local_minima_finding_driver(input_orography_file,
                                         input_orography_fieldname,
                                         output_local_minima_file,
                                         output_local_minima_fieldname):
    input_orography = iodriver.advanced_field_loader(input_orography_file,
                                                     field_type='Orography',
                                                     fieldname=input_orography_fieldname)
    output_local_minima = input_orography.find_all_local_minima()
    iodriver.advanced_field_writer(output_local_minima_file,output_local_minima,
                                   fieldname=output_local_minima_fieldname)

def advanced_burn_carved_rivers_driver(input_orography_file,
                                       input_orography_fieldname,
                                       input_rdirs_file,
                                       input_rdirs_fieldname,
                                       input_minima_file,
                                       input_minima_fieldname,
                                       input_lakemask_file,
                                       input_lakemask_fieldname,
                                       output_orography_file,
                                       output_orography_fieldname,
                                       add_slope = False,
                                       max_exploration_range = 0,
                                       minimum_height_change_threshold = 0.0,
                                       short_path_threshold = 0,
                                       short_minimum_height_change_threshold = 0.0):
    input_orography = iodriver.advanced_field_loader(input_orography_file,
                                                     field_type='Orography',
                                                     fieldname=input_orography_fieldname)
    input_rdirs     = iodriver.advanced_field_loader(input_rdirs_file,
                                                     field_type='Orography',
                                                     fieldname=input_rdirs_fieldname)
    input_minima    = iodriver.advanced_field_loader(input_minima_file,
                                                     field_type='Orography',
                                                     fieldname=input_minima_fieldname)
    input_lakemask  = iodriver.advanced_field_loader(input_lakemask_file,
                                                     field_type='Generic',
                                                     fieldname=input_lakemask_fieldname)
    output_orography = field.Field(np.ascontiguousarray(input_orography.get_data(),
                             dtype=np.float64),grid=input_orography.get_grid())
    lake_operators_wrapper.burn_carved_rivers(output_orography.get_data(),
                                              np.ascontiguousarray(input_rdirs.get_data(),
                                                                   dtype=np.float64),
                                              np.ascontiguousarray(input_minima.get_data(),
                                                                   dtype=np.int32),
                                              np.ascontiguousarray(input_lakemask.get_data(),
                                                                   dtype=np.int32),
                                              add_slope,max_exploration_range,
                                              minimum_height_change_threshold,
                                              short_path_threshold,
                                              short_minimum_height_change_threshold)
    iodriver.advanced_field_writer(output_orography_file,output_orography,
                                   fieldname=output_orography_fieldname)

def advanced_fill_lakes_driver(input_minima_file,
                               input_minima_fieldname,
                               input_lakemask_file,
                               input_lakemask_fieldname,
                               input_orography_file,
                               input_orography_fieldname,
                               output_orography_file,
                               output_orography_fieldname,
                               use_highest_possible_lake_water_level=True):
    input_minima    = iodriver.advanced_field_loader(input_minima_file,
                                                     field_type='Orography',
                                                     fieldname=input_minima_fieldname)
    input_lakemask  = iodriver.advanced_field_loader(input_lakemask_file,
                                                     field_type='Orography',
                                                     fieldname=input_lakemask_fieldname)
    input_orography = iodriver.advanced_field_loader(input_orography_file,
                                                     field_type='Orography',
                                                     fieldname=input_orography_fieldname)
    lake_operators_wrapper.fill_lakes(input_minima.get_data(),input_lakemask.get_data(),
                                      input_orography.get_data(),
                                      use_highest_possible_lake_water_level)
    iodriver.advanced_field_writer(output_orography_file,input_orography,
                                   fieldname=output_orography_fieldname)

def reduce_connected_areas_to_points(input_minima_file,
                                     input_minima_fieldname,
                                     output_minima_file,
                                     output_minima_fieldname,
                                     use_diagonals=True):
    input_minima    = iodriver.advanced_field_loader(input_minima_file,
                                                     field_type='Generic',
                                                     fieldname=input_minima_fieldname)
    minima_array = np.ascontiguousarray(input_minima.get_data(),
                                        dtype=np.int32)
    lake_operators_wrapper.reduce_connected_areas_to_points(minima_array,use_diagonals)
    output_minima = field.Field(minima_array,grid=input_minima.get_grid())
    iodriver.advanced_field_writer(output_minima_file,output_minima,
                                   fieldname=output_minima_fieldname)

def evaluate_basins(landsea_in,
                    minima_in,
                    raw_orography_in,
                    corrected_orography_in,
                    cell_areas_in,
                    prior_fine_rdirs_in,
                    prior_fine_catchments_in,
                    coarse_catchment_nums_in):
    fine_grid = raw_orography_in.get_grid()
    fine_shape = raw_orography_in.get_data().shape
    lake_numbers_out = np.zeros(fine_shape,dtype=np.int32,order='C')
    sinkless_rdirs_out_double = np.zeros(fine_shape,dtype=np.float64,order='C')
    lake_mask_out_int = np.zeros(fine_shape,dtype=np.int32,order='C')
    output = {}
    output["number_of_lakes"],output["lakes_as_array"] = \
        l2_evaluate_basins_wrapper.\
        evaluate_basins(np.ascontiguousarray(landsea_in.get_data(),dtype=np.int32),
                        np.ascontiguousarray(minima_in.get_data(),dtype=np.int32),
                        np.ascontiguousarray(raw_orography_in.get_data(),dtype=np.float64),
                        np.ascontiguousarray(corrected_orography_in.get_data(),dtype=np.float64),
                        np.ascontiguousarray(cell_areas_in.get_data(),dtype=np.float64),
                        np.ascontiguousarray(prior_fine_rdirs_in.get_data(),dtype=np.int32),
                        np.ascontiguousarray(prior_fine_catchments_in.get_data(),dtype=np.int32),
                        np.ascontiguousarray(coarse_catchment_nums_in.get_data(),dtype=np.int32),
                        lake_numbers_out,
                        sinkless_rdirs_out_double,
                        lake_mask_out_int)
    output["lake_mask"] = field.Field(lake_mask_out_int.astype(np.bool),fine_grid)
    output["sinkless_rdirs"] = field.Field(sinkless_rdirs_out_double,fine_grid)
    output["lake_numbers"] = field.Field(lake_numbers_out,fine_grid)
    return output

def advanced_water_redistribution_driver(input_lake_numbers_file,
                                         input_lake_numbers_fieldname,
                                         input_lake_centers_file,
                                         input_lake_centers_fieldname,
                                         input_water_to_redistribute_file,
                                         input_water_to_redistribute_fieldname,
                                         output_water_redistributed_to_lakes_file,
                                         output_water_redistributed_to_lakes_fieldname,
                                         output_water_redistributed_to_rivers_file,
                                         output_water_redistributed_to_rivers_fieldname,
                                         coarse_grid_type,**coarse_grid_kwargs):
  lake_numbers = iodriver.advanced_field_loader(input_lake_numbers_file,
                                                field_type='Generic',
                                                fieldname=input_lake_numbers_fieldname)
  lake_centers = iodriver.advanced_field_loader(input_lake_centers_file,
                                                field_type='Generic',
                                                fieldname=input_lake_centers_fieldname)
  water_to_redistribute = \
    iodriver.advanced_field_loader(input_water_to_redistribute_file,
                                   field_type='Generic',
                                   fieldname=input_water_to_redistribute_fieldname)
  fine_grid = lake_numbers.get_grid()
  fine_shape = lake_numbers.get_data().shape
  coarse_grid = grid.makeGrid(coarse_grid_type,**coarse_grid_kwargs)
  water_redistributed_to_lakes = field.Field(np.zeros(fine_shape,dtype=np.float64,order='C'),
                                             fine_grid)
  water_redistributed_to_rivers = field.Field(coarse_grid.create_empty_field(np.float64),coarse_grid)
  lake_operators_wrapper.redistribute_water(np.ascontiguousarray(lake_numbers.get_data(),
                                            dtype=np.int32),
                                            np.ascontiguousarray(lake_centers.get_data(),
                                            dtype=np.int32),
                                            np.ascontiguousarray(water_to_redistribute.get_data(),
                                            dtype=np.float64),
                                            water_redistributed_to_lakes.get_data(),
                                            water_redistributed_to_rivers.get_data())
  iodriver.advanced_field_writer(output_water_redistributed_to_lakes_file,
                                 water_redistributed_to_lakes,
                                 fieldname=output_water_redistributed_to_lakes_fieldname)
  iodriver.advanced_field_writer(output_water_redistributed_to_rivers_file,
                                 water_redistributed_to_rivers,
                                 fieldname=output_water_redistributed_to_rivers_fieldname)

def advanced_shallow_lake_filtering_driver(input_unfilled_orography_file,
                                           input_unfilled_orography_fieldname,
                                           input_filled_orography_file,
                                           input_filled_orography_fieldname,
                                           output_unfilled_orography_file,
                                           output_unfilled_orography_fieldname,
                                           minimum_depth_threshold):
    input_unfilled_orography = \
      iodriver.advanced_field_loader(input_unfilled_orography_file,
                                     field_type='Orography',
                                     fieldname=input_unfilled_orography_fieldname)
    input_filled_orography = \
      iodriver.advanced_field_loader(input_filled_orography_file,
                                     field_type='Orography',
                                     fieldname=input_filled_orography_fieldname)
    output_unfilled_orography = \
      field.Field(np.ascontiguousarray(input_unfilled_orography.get_data(),
                                       dtype=np.float64),
                  grid=input_unfilled_orography.get_grid())
    lake_operators_wrapper.filter_out_shallow_lakes(output_unfilled_orography.get_data(),
                                                    np.ascontiguousarray(input_filled_orography.\
                                                                         get_data(),
                                                                         dtype=np.float64),
                                                    minimum_depth_threshold)
    iodriver.advanced_field_writer(output_unfilled_orography_file,
                                   output_unfilled_orography,
                                   fieldname=output_unfilled_orography_fieldname)

def add_lake_bathymetry(input_orography,
                        input_bathymetry,
                        lake_mask):
  input_orography.mask_field_with_external_mask(lake_mask.get_data())
  output_orography = input_bathymetry
  output_orography.update_field_with_partially_masked_data(input_orography)
  return output_orography

def add_lake_bathymetry_driver(input_orography_file,
                               input_orography_fieldname,
                               input_bathymetry_file,
                               input_bathymetry_fieldname,
                               lake_mask_file,
                               lake_mask_fieldname,
                               output_orography_file,
                               output_orography_fieldname):
  input_orography =  iodriver.advanced_field_loader(input_orography_file,
                                                    field_type='Generic',
                                                    fieldname=input_orography_fieldname)
  input_bathymetry = iodriver.advanced_field_loader(input_bathymetry_file,
                                                    field_type='Generic',
                                                    fieldname=input_bathymetry_fieldname)
  lake_mask = iodriver.advanced_field_loader(lake_mask_file,
                                             field_type='Generic',
                                             fieldname=lake_mask_fieldname)
  lake_mask.change_dtype(bool)
  output_orography = add_lake_bathymetry(input_orography,
                                         input_bathymetry,
                                         lake_mask)
  iodriver.advanced_field_writer(output_orography_file,
                                 output_orography,
                                 fieldname=output_orography_fieldname)

def filter_narrow_lakes(input_unfilled_orography,
                        input_filled_orography,
                        interior_cell_min_masked_neighbors=5,
                        edge_cell_max_masked_neighbors=4,
                        max_range=5,
                        iterations=5):
  unfilled_orography = input_unfilled_orography.copy()
  for _ in range(iterations):
    lake_mask = unfilled_orography.equal_to(input_filled_orography)
    lake_mask.invert_data()
    number_of_lake_neighbors = lake_mask.get_number_of_masked_neighbors()
    edge_cell_mask = number_of_lake_neighbors.\
      less_than_or_equal_to_value(edge_cell_max_masked_neighbors)
    edge_cell_mask = edge_cell_mask.logical_and(lake_mask)
    non_edge_cell_mask = edge_cell_mask.copy()
    non_edge_cell_mask.invert_data()
    non_edge_cell_mask = lake_mask.logical_and(non_edge_cell_mask)
    dilated_interior_cell_mask = number_of_lake_neighbors.\
      greater_than_or_equal_to_value(interior_cell_min_masked_neighbors)
    dilated_interior_cell_mask.dilate(np.array([[True,True,True],
                                        [True,True,True],
                                        [True,True,True]],dtype=bool),
                                       iterations=max_range)
    filtered_lake_mask = non_edge_cell_mask.logical_or(dilated_interior_cell_mask.\
                                                       logical_and(edge_cell_mask))
    masked_filled_orography = input_filled_orography.copy()
    masked_filled_orography.mask_field_with_external_mask(filtered_lake_mask.get_data())
    unfilled_orography.update_field_with_partially_masked_data(masked_filled_orography)
  return unfilled_orography

def advanced_narrow_lake_filtering_driver(input_unfilled_orography_file,
                                          input_unfilled_orography_fieldname,
                                          input_filled_orography_file,
                                          input_filled_orography_fieldname,
                                          output_unfilled_orography_file,
                                          output_unfilled_orography_fieldname,
                                          interior_cell_min_masked_neighbors=5,
                                          edge_cell_max_masked_neighbors=4,
                                          max_range=5,
                                          iterations=5):
  input_unfilled_orography =  \
    iodriver.advanced_field_loader(input_unfilled_orography_file,
                                   field_type='Generic',
                                   fieldname=input_unfilled_orography_fieldname)
  input_filled_orography = \
    iodriver.advanced_field_loader(input_filled_orography_file,
                                   field_type='Generic',
                                   fieldname=input_filled_orography_fieldname)
  output_unfilled_orography = filter_narrow_lakes(input_unfilled_orography,
                                                  input_filled_orography,
                                                  interior_cell_min_masked_neighbors,
                                                  edge_cell_max_masked_neighbors,
                                                  max_range,iterations)
  iodriver.advanced_field_writer(output_unfilled_orography_file,
                                 output_unfilled_orography,
                                 fieldname=output_unfilled_orography_fieldname)

def find_outlet_for_excess_evaporation(coarse_lake_mask,
                                       coarse_rdirs,
                                       coarse_catchments,
                                       coarse_connected_catchments,
                                       coarse_grid_to_jsbach_grid_lat_map,
                                       coarse_grid_to_jsbach_grid_lon_map,
                                       nlat_jsbach,nlon_jsbach):
  lake_cell_numbers = np.full((nlat_jsbach,nlon_jsbach),-1,dtype=np.int32)
  counter = 0
  lake_cell_pixel_list = []
  total_coarse_catchments = np.max(coarse_catchments)
  catchment_center_lats = np.full(total_coarse_catchments+1,-1,dtype=np.int32)
  catchment_center_lons = np.full(total_coarse_catchments+1,-1,dtype=np.int32)
  total_connected_coarse_catchments = np.max(coarse_connected_catchments)
  connected_catchment_outlet_lats = \
    np.full(total_connected_coarse_catchments+1,-1,dtype=np.int32)
  connected_catchment_outlet_lons = \
    np.full(total_connected_coarse_catchments+1,-1,dtype=np.int32)
  for i in range(coarse_rdirs.shape[0]):
    for j in range(coarse_rdirs.shape[1]):
      coarse_rdir = coarse_rdirs[i,j]
      if coarse_rdir == 0:
        coarse_connected_catchment = coarse_connected_catchments[i,j]
        connected_catchment_outlet_lats[coarse_connected_catchment] = i
        connected_catchment_outlet_lons[coarse_connected_catchment] = j
      if coarse_lake_mask[i,j]:
        if coarse_rdir == -2:
          coarse_catchment = coarse_catchments[i,j]
          catchment_center_lats[coarse_catchment] = i
          catchment_center_lons[coarse_catchment] = j
        lat_jsbach = coarse_grid_to_jsbach_grid_lat_map[i,j]
        lon_jsbach = coarse_grid_to_jsbach_grid_lon_map[i,j]
        lake_cell_number = lake_cell_numbers[lat_jsbach,lon_jsbach]
        if lake_cell_number == -1:
          lake_cell_numbers[lat_jsbach,lon_jsbach] = counter
          lake_cell_pixel_list.append([(i,j)])
          counter += 1
        else:
          lake_cell_pixel_list[lake_cell_number].append((i,j))
  catchment_counts = np.zeros(total_coarse_catchments+1,dtype=np.int32)
  excess_evaporation_outlet_lat = \
    np.full((nlat_jsbach,nlon_jsbach),-1,dtype=np.int32)
  excess_evaporation_outlet_lon = \
    np.full((nlat_jsbach,nlon_jsbach),-1,dtype=np.int32)
  for i in range(lake_cell_numbers.shape[0]):
    for j in range(lake_cell_numbers.shape[1]):
      lake_cell_number = lake_cell_numbers[i,j]
      if lake_cell_number >= 0:
        pixels = lake_cell_pixel_list[lake_cell_number]
        catchment_counts[:] = 0
        for pixel in pixels:
          catchment_counts[coarse_catchments[pixel]] += 1
        max_catchment = np.argmax(catchment_counts)
        max_catchment_center_lat = catchment_center_lats[max_catchment]
        max_catchment_center_lon = catchment_center_lons[max_catchment]
        max_connected_catchment = \
          coarse_connected_catchments[max_catchment_center_lat,
                                      max_catchment_center_lon]
        excess_evaporation_outlet_lat[i,j] = \
          connected_catchment_outlet_lats[max_connected_catchment]
        excess_evaporation_outlet_lon[i,j] = \
          connected_catchment_outlet_lons[max_connected_catchment]
  return excess_evaporation_outlet_lat,excess_evaporation_outlet_lon

def advanced_find_outlet_for_excess_evaporation_driver(
      fine_lake_mask_file,
      fine_lake_mask_fieldname,
      coarse_rdirs_file,
      coarse_rdirs_fieldname,
      coarse_catchment_file,
      coarse_catchements_fieldname,
      coarse_connected_catchment_file,
      coarse_connected_catchment_fieldname,
      coarse_grid_to_jsbach_grid_map_file,
      coarse_grid_to_jsbach_grid_lat_map_fieldname
      coarse_grid_to_jsbach_grid_lon_map_fieldname
      excess_evaporation_outlet_file,
      excess_evaporation_outlet_lat_fieldname,
      excess_evaporation_outlet_lon_fieldname,
      nlat_jsbach,nlon_jsbach):
  fine_lake_mask =  \
    iodriver.advanced_field_loader(fine_lake_mask_file,
                                   field_type='Generic',
                                   fieldname=fine_lake_mask_fieldname)
  coarse_lake_mask = utilties.upscale_field(fine_lake_mask,"HD",
                                            "Max",scalenumbers=False)
  coarse_rdirs =  \
    iodriver.advanced_field_loader(coarse_rdirs_file,
                                   field_type='Generic',
                                   fieldname=coarse_rdirs_fieldname)
  coarse_catchments =  \
    iodriver.advanced_field_loader(coarse_catchment_file,
                                   field_type='Generic',
                                   fieldname=coarse_catchements_fieldname)
  coarse_connected_catchments =  \
    iodriver.advanced_field_loader(coarse_connected_catchment_file,
                                   field_type='Generic',
                                   fieldname=coarse_connected_catchment_fieldname)
  coarse_grid_to_jsbach_grid_lat_map =  \
    iodriver.advanced_field_loader(coarse_grid_to_jsbach_grid_map_file,
                                   field_type='Generic',
                                   fieldname=
                                   coarse_grid_to_jsbach_grid_lat_map_fieldname)
  coarse_grid_to_jsbach_grid_lon_map =  \
    iodriver.advanced_field_loader(coarse_grid_to_jsbach_grid_map_file,
                                   field_type='Generic',
                                   fieldname=
                                   coarse_grid_to_jsbach_grid_lon_map_fieldname)
  excess_evaporation_outlet_lat,excess_evaporation_outlet_lon = \
    find_outlet_for_excess_evaporation(coarse_lake_mask,
                                       coarse_rdirs,
                                       coarse_catchments,
                                       coarse_connected_catchments,
                                       coarse_grid_to_jsbach_grid_lat_map,
                                       coarse_grid_to_jsbach_grid_lon_map,
                                       nlat_jsbach,nlon_jsbach)
  excess_evaporation_outlet_lat =  \
    iodriver.advanced_field_loader(excess_evaporation_outlet_file,
                                   field_type='Generic',
                                   fieldname=
                                   excess_evaporation_outlet_lat_fieldname)
  excess_evaporation_outlet_lon =  \
    iodriver.advanced_field_loader(excess_evaporation_outlet_file,
                                   field_type='Generic',
                                   fieldname=
                                   excess_evaporation_outlet_lon_fieldname)





