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
from Dynamic_HD_Scripts.interface.cpp_interface.libs \
    import lake_operators_wrapper
from Dynamic_HD_Scripts.interface.cpp_interface.libs \
    import evaluate_basins_wrapper

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

def advanced_basin_evaluation_driver(input_minima_file,
                                     input_minima_fieldname,
                                     input_raw_orography_file,
                                     input_raw_orography_fieldname,
                                     input_corrected_orography_file,
                                     input_corrected_orography_fieldname,
                                     input_cell_areas_file,
                                     input_cell_areas_fieldname,
                                     input_prior_fine_rdirs_file,
                                     input_prior_fine_rdirs_fieldname,
                                     input_prior_fine_catchments_file,
                                     input_prior_fine_catchments_fieldname,
                                     input_coarse_catchment_nums_file,
                                     input_coarse_catchment_nums_fieldname,
                                     input_coarse_rdirs_file,
                                     input_coarse_rdirs_fieldname,
                                     combined_output_filename,
                                     output_filepath,
                                     output_filelabel,
                                     output_basin_catchment_nums_filepath=None,
                                     output_sinkless_rdirs_filepath=None):
    input_minima    = iodriver.advanced_field_loader(input_minima_file,
                                                     field_type='Generic',
                                                     fieldname=input_minima_fieldname)
    input_raw_orography = iodriver.advanced_field_loader(input_raw_orography_file,
                                                         field_type='Orography',
                                                         fieldname=input_raw_orography_fieldname)
    input_corrected_orography = iodriver.advanced_field_loader(input_corrected_orography_file,
                                                               field_type='Orography',
                                                               fieldname=input_corrected_orography_fieldname)
    input_cell_areas = iodriver.advanced_field_loader(input_cell_areas_file,
                                                      field_type='Generic',
                                                      fieldname=input_cell_areas_fieldname)
    input_prior_fine_rdirs = iodriver.advanced_field_loader(input_prior_fine_rdirs_file,
                                                            field_type='RiverDirections',
                                                            fieldname=input_prior_fine_rdirs_fieldname)
    input_prior_fine_catchments = iodriver.advanced_field_loader(input_prior_fine_catchments_file,
                                                                 field_type='Generic',
                                                                 fieldname=input_prior_fine_catchments_fieldname)
    input_coarse_catchment_nums = iodriver.advanced_field_loader(input_coarse_catchment_nums_file,
                                                                 field_type='Generic',
                                                                 fieldname=input_coarse_catchment_nums_fieldname)
    input_coarse_rdirs =  iodriver.advanced_field_loader(input_coarse_rdirs_file,
                                                         field_type='Generic',
                                                         fieldname=
                                                         input_coarse_rdirs_fieldname)
    fine_grid = input_raw_orography.get_grid()
    fine_shape = input_raw_orography.get_data().shape
    connection_volume_thresholds = field.Field(np.zeros(fine_shape,dtype=np.float64,order='C'),fine_grid)
    flood_volume_thresholds = field.Field(np.zeros(fine_shape,dtype=np.float64,order='C'),fine_grid)
    connection_heights = field.Field(np.zeros(fine_shape,dtype=np.float64,order='C'),fine_grid)
    flood_heights = field.Field(np.zeros(fine_shape,dtype=np.float64,order='C'),fine_grid)
    flood_next_cell_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    flood_next_cell_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    connect_next_cell_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    connect_next_cell_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    connect_merge_and_redirect_indices_index = \
        field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    flood_merge_and_redirect_indices_index = \
        field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    if output_basin_catchment_nums_filepath is not None:
        basin_catchment_numbers = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        if output_sinkless_rdirs_filepath is not None:
            sinkless_rdirs = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        else:
            sinkless_rdirs = None
    else:
        basin_catchment_numbers = None
        sinkless_rdirs = None
    merges_filename = path.join(output_filepath,
                                "merges_" +
                                output_filelabel + ".nc")
    evaluate_basins_wrapper.evaluate_basins(minima_in_int=
                                            np.ascontiguousarray(input_minima.get_data(),dtype=np.int32),
                                            raw_orography_in=
                                            np.ascontiguousarray(input_raw_orography.get_data(),
                                                                 dtype=np.float64),
                                            corrected_orography_in=
                                            np.ascontiguousarray(input_corrected_orography.get_data(),
                                                                 dtype=np.float64),
                                            cell_areas_in=
                                            np.ascontiguousarray(input_cell_areas.get_data(),
                                                                 dtype=np.float64),
                                            connection_volume_thresholds_in=
                                            connection_volume_thresholds.get_data(),
                                            flood_volume_thresholds_in=
                                            flood_volume_thresholds.get_data(),
                                            connection_heights_in=
                                                connection_heights.get_data(),
                                            flood_heights_in=
                                                flood_heights.get_data(),
                                            prior_fine_rdirs_in=
                                            np.ascontiguousarray(input_prior_fine_rdirs.get_data(),
                                                                 dtype=np.float64),
                                            prior_coarse_rdirs_in=
                                            np.ascontiguousarray(input_coarse_rdirs.get_data(),
                                                                 dtype=np.float64),
                                            prior_fine_catchments_in=
                                            np.ascontiguousarray(input_prior_fine_catchments.get_data(),
                                                                 dtype=np.int32),
                                            coarse_catchment_nums_in=
                                            np.ascontiguousarray(input_coarse_catchment_nums.get_data(),
                                                                 dtype=np.int32),
                                            flood_next_cell_lat_index_in=
                                            flood_next_cell_lat_index.get_data(),
                                            flood_next_cell_lon_index_in=
                                            flood_next_cell_lon_index.get_data(),
                                            connect_next_cell_lat_index_in=
                                            connect_next_cell_lat_index.get_data(),
                                            connect_next_cell_lon_index_in=
                                            connect_next_cell_lon_index.get_data(),
                                            connect_merge_and_redirect_indices_index_in=
                                            connect_merge_and_redirect_indices_index.get_data(),
                                            flood_merge_and_redirect_indices_index_in=
                                            flood_merge_and_redirect_indices_index.get_data(),
                                            merges_filepath=merges_filename,
                                            basin_catchment_numbers_in=
                                            basin_catchment_numbers.get_data(),
                                            sinkless_rdirs_in=
                                            sinkless_rdirs.get_data())
    connection_volume_thresholds_filename = path.join(output_filepath,
                                                      "connect_vts_" +
                                                      output_filelabel + ".nc")
    iodriver.advanced_field_writer(connection_volume_thresholds_filename,
                                   connection_volume_thresholds,
                                   fieldname='connection_volume_thresholds')
    flood_volume_thresholds_filename = path.join(output_filepath,
                                                 "flood_vts_" +
                                                 output_filelabel + ".nc")
    iodriver.advanced_field_writer(flood_volume_thresholds_filename,
                                   flood_volume_thresholds,
                                   fieldname='flood_volume_thresholds')
    connection_heights_filename = path.join(output_filepath,
                                            "connect_h_" +
                                            output_filelabel + ".nc")
    iodriver.advanced_field_writer(connection_heights_filename,
                                   connection_heights,
                                   fieldname='connection_heights')
    flood_heights_filename = path.join(output_filepath,
                                       "flood_h_" +
                                       output_filelabel + ".nc")
    iodriver.advanced_field_writer(flood_heights_filename,
                                   flood_heights,
                                   fieldname='flood_heights')
    corrected_heights_filename = path.join(output_filepath,
                                           "corrected_h_" +
                                           output_filelabel + ".nc")
    iodriver.advanced_field_writer(corrected_heights_filename,
                                   input_corrected_orography,
                                   fieldname='corrected_heights')
    raw_heights_filename = path.join(output_filepath,
                                     "raw_h_" +
                                     output_filelabel + ".nc")
    iodriver.advanced_field_writer(raw_heights_filename,
                                   input_raw_orography,
                                   fieldname='raw_heights')
    flood_next_cell_lat_index_filename = path.join(output_filepath,
                                                   "flood_nci_lat_" +
                                                   output_filelabel + ".nc")
    iodriver.advanced_field_writer(flood_next_cell_lat_index_filename,
                                   flood_next_cell_lat_index,
                                   fieldname='flood_next_cell_lat_index')
    flood_next_cell_lon_index_filename = path.join(output_filepath,
                                                   "flood_nci_lon_" +
                                                   output_filelabel + ".nc")
    iodriver.advanced_field_writer(flood_next_cell_lon_index_filename,
                                   flood_next_cell_lon_index,
                                   fieldname='flood_next_cell_lon_index')
    connect_next_cell_lat_index_filename = path.join(output_filepath,
                                                     "connect_nci_lat_" +
                                                     output_filelabel + ".nc")
    iodriver.advanced_field_writer(connect_next_cell_lat_index_filename,
                                   connect_next_cell_lat_index,
                                   fieldname='connect_next_cell_lat_index')
    connect_next_cell_lon_index_filename = path.join(output_filepath,
                                                     "connect_nci_lon_" +
                                                     output_filelabel + ".nc")
    iodriver.advanced_field_writer(connect_next_cell_lon_index_filename,
                                   connect_next_cell_lon_index,
                                   fieldname='connect_next_cell_lon_index')
    connect_merge_and_redirect_indices_index_filename = path.join(output_filepath,
                                                                  "connect_mergeredir_indices_index_" +
                                                                  output_filelabel + ".nc")
    iodriver.advanced_field_writer(connect_merge_and_redirect_indices_index_filename,
                                   connect_merge_and_redirect_indices_index,
                                   fieldname='connect_merge_and_redirect_indices_index')
    flood_merge_and_redirect_indices_index_filename = path.join(output_filepath,
                                                                  "flood_mergeredir_indices_index_" +
                                                                  output_filelabel + ".nc")
    iodriver.advanced_field_writer(flood_merge_and_redirect_indices_index_filename,
                                   flood_merge_and_redirect_indices_index,
                                   fieldname='flood_merge_and_redirect_indices_index')
    lake_centers_filename = path.join(output_filepath,
                                      "lake_centers_" +
                                      output_filelabel + ".nc")
    iodriver.advanced_field_writer(lake_centers_filename,
                                   input_minima,
                                   fieldname="lake_centers")
    individual_field_filenames = [connection_volume_thresholds_filename,
                                  flood_volume_thresholds_filename,
                                  connection_heights_filename,
                                  flood_heights_filename,
                                  corrected_heights_filename,
                                  raw_heights_filename,
                                  flood_next_cell_lat_index_filename,
                                  flood_next_cell_lon_index_filename,
                                  connect_next_cell_lat_index_filename,
                                  connect_next_cell_lon_index_filename,
                                  connect_merge_and_redirect_indices_index,
                                  flood_merge_and_redirect_indices_index,
                                  lake_centers_filename,
                                  merges_filename]
    cdo_inst = cdo.Cdo()
    cdo_inst.merge(input=" ".join(individual_field_filenames),
                   output=combined_output_filename)
    for individual_field_filename in individual_field_filenames:
      os.remove(individual_field_filename)
    if output_basin_catchment_nums_filepath is not None:
        iodriver.advanced_field_writer(output_basin_catchment_nums_filepath,
                                       basin_catchment_numbers,
                                       fieldname="basin_catchment_numbers")
        if output_sinkless_rdirs_filepath is not None:
            iodriver.advanced_field_writer(output_sinkless_rdirs_filepath,
                                           sinkless_rdirs,
                                           fieldname="rdir")

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
  lake_mask.change_dtype(np.bool)
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
                                        [True,True,True]],dtype=np.bool_),
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
