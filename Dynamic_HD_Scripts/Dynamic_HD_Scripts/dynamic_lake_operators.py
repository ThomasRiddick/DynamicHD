'''
Created on March 23, 2018

@author: thomasriddick
'''
import iodriver
import libs.lake_operators_wrapper as lake_operators_wrapper  #@UnresolvedImport
import libs.evaluate_basins_wrapper as evaluate_basins_wrapper
import numpy as np
import field

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
                                       output_orography_fieldname):
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
                                                                   dtype=np.int32))
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
                                     input_prior_fine_rdirs_file,
                                     input_prior_fine_rdirs_fieldname,
                                     input_prior_fine_catchments_file,
                                     input_prior_fine_catchments_fieldname,
                                     input_coarse_catchment_nums_file,
                                     input_coarse_catchment_nums_fieldname):
    input_minima    = iodriver.advanced_field_loader(input_minima_file,
                                                     field_type='Generic',
                                                     fieldname=input_minima_fieldname)
    input_raw_orography = iodriver.advanced_field_loader(input_raw_orography_file,
                                                         field_type='Orography',
                                                         fieldname=input_raw_orography_fieldname)
    input_corrected_orography = iodriver.advanced_field_loader(input_corrected_orography_file,
                                                               field_type='Orography',
                                                               fieldname=input_corrected_orography_fieldname)
    input_prior_fine_rdirs = iodriver.advanced_field_loader(input_prior_fine_rdirs_file,
                                                            field_type='RiverDirections',
                                                            fieldname=input_prior_fine_rdirs_fieldname,
                                                            grid_desc_file="/Users/thomasriddick/Documents/data/HDdata/grids/grid_10min.txt")
    print "watch for the nasty hack"
    input_prior_fine_catchments = iodriver.advanced_field_loader(input_prior_fine_catchments_file,
                                                                 field_type='Generic',
                                                                 fieldname=input_prior_fine_catchments_fieldname,
                                                                 grid_desc_file="/Users/thomasriddick/Documents/data/HDdata/grids/grid_10min.txt")
    print "watch for the nasty hack"
    input_coarse_catchment_nums = iodriver.advanced_field_loader(input_coarse_catchment_nums_file,
                                                                 field_type='Generic',
                                                                 fieldname=input_coarse_catchment_nums_fieldname)
    fine_grid = input_raw_orography.get_grid()
    fine_shape = input_raw_orography.get_data().shape
    connection_volume_thresholds = field.Field(np.zeros(fine_shape,dtype=np.float64,order='C'),fine_grid)
    flood_volume_thresholds = field.Field(np.zeros(fine_shape,dtype=np.float64,order='C'),fine_grid)
    flood_next_cell_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    flood_next_cell_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    connect_next_cell_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    connect_next_cell_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    flood_force_merge_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    flood_force_merge_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    connect_force_merge_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    connect_force_merge_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    flood_redirect_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    flood_redirect_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    connect_local_redirect_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    connect_local_redirect_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    flood_local_redirect = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    connect_local_redirect = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    merge_points = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    evaluate_basins_wrapper.evaluate_basins(minima_in_int=
                                            np.ascontiguousarray(input_minima.get_data(),dtype=np.int32),
                                            raw_orography_in=
                                            np.ascontiguousarray(input_raw_orography.get_data(),
                                                                 dtype=np.float64),
                                            corrected_orography_in=
                                            np.ascontiguousarray(input_corrected_orography.get_data(),
                                                                 dtype=np.float64),
                                            connection_volume_thresholds_in=
                                            connection_volume_thresholds.get_data(),
                                            flood_volume_thresholds_in=
                                            flood_volume_thresholds.get_data(),
                                            prior_fine_rdirs_in=
                                            np.ascontiguousarray(input_prior_fine_rdirs.get_data(),
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
                                            flood_force_merge_lat_index_in=
                                            flood_force_merge_lat_index.get_data(),
                                            flood_force_merge_lon_index_in=
                                            flood_force_merge_lon_index.get_data(),
                                            connect_force_merge_lat_index_in=
                                            connect_force_merge_lat_index.get_data(),
                                            connect_force_merge_lon_index_in=
                                            connect_force_merge_lon_index.get_data(),
                                            flood_redirect_lat_index_in=
                                            flood_redirect_lat_index.get_data(),
                                            flood_redirect_lon_index_in=
                                            flood_redirect_lon_index.get_data(),
                                            connect_local_redirect_lat_index_in=
                                            connect_local_redirect_lat_index.get_data(),
                                            connect_local_redirect_lon_index_in=
                                            connect_local_redirect_lon_index.get_data(),
                                            flood_local_redirect_out_int=
                                            flood_local_redirect.get_data(),
                                            connect_local_redirect_out_int=
                                            connect_local_redirect.get_data(),
                                            merge_points_out_int=
                                            merge_points.get_data())
