'''
Created on March 23, 2018

@author: thomasriddick
'''
import iodriver
import libs.lake_operators_wrapper as lake_operators_wrapper  #@UnresolvedImport
import libs.evaluate_basins_wrapper as evaluate_basins_wrapper
import numpy as np
import field
import os
import os.path as path
import cdo

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
                                     input_coarse_catchment_nums_fieldname,
                                     combined_output_filename,
                                     output_filepath,
                                     output_filelabel):
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
    connect_redirect_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    connect_redirect_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    additional_flood_redirect_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    additional_flood_redirect_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    additional_connect_redirect_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    additional_connect_redirect_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    flood_local_redirect = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    connect_local_redirect = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    additional_flood_local_redirect = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
    additional_connect_local_redirect = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
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
                                            connect_redirect_lat_index_in=
                                            connect_redirect_lat_index.get_data(),
                                            connect_redirect_lon_index_in=
                                            connect_redirect_lon_index.get_data(),
                                            additional_flood_redirect_lat_index_in=
                                            additional_flood_redirect_lat_index.get_data(),
                                            additional_flood_redirect_lon_index_in=
                                            additional_flood_redirect_lon_index.get_data(),
                                            additional_connect_redirect_lat_index_in=
                                            additional_connect_redirect_lat_index.get_data(),
                                            additional_connect_redirect_lon_index_in=
                                            additional_connect_redirect_lon_index.get_data(),
                                            flood_local_redirect_out_int=
                                            flood_local_redirect.get_data(),
                                            connect_local_redirect_out_int=
                                            connect_local_redirect.get_data(),
                                            additional_flood_local_redirect_out_int=
                                            additional_flood_local_redirect.get_data(),
                                            additional_connect_local_redirect_out_int=
                                            additional_connect_local_redirect.get_data(),
                                            merge_points_out_int=
                                            merge_points.get_data())
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
    flood_force_merge_lat_index_filename = path.join(output_filepath,
                                                     "flood_fmi_lat_" +
                                                     output_filelabel + ".nc")
    iodriver.advanced_field_writer(flood_force_merge_lat_index_filename,
                                   flood_force_merge_lat_index,
                                   fieldname='flood_force_merge_lat_index')
    flood_force_merge_lon_index_filename = path.join(output_filepath,
                                                     "flood_fmi_lon_" +
                                                     output_filelabel + ".nc")
    iodriver.advanced_field_writer(flood_force_merge_lon_index_filename,
                                   flood_force_merge_lon_index,
                                   fieldname='flood_force_merge_lon_index')
    connect_force_merge_lat_index_filename = path.join(output_filepath,
                                                       "connect_fmi_lat_" +
                                                       output_filelabel + ".nc")
    iodriver.advanced_field_writer(connect_force_merge_lat_index_filename,
                                   connect_force_merge_lat_index,
                                   fieldname='connect_force_merge_lat_index')
    connect_force_merge_lon_index_filename = path.join(output_filepath,
                                                       "connect_fmi_lon_" +
                                                       output_filelabel + ".nc")
    iodriver.advanced_field_writer(connect_force_merge_lon_index_filename,
                                   connect_force_merge_lon_index,
                                   fieldname='connect_force_merge_lon_index')
    flood_redirect_lat_index_filename = path.join(output_filepath,
                                                  "flood_ri_lat_" +
                                                  output_filelabel + ".nc")
    iodriver.advanced_field_writer(flood_redirect_lat_index_filename,
                                   flood_redirect_lat_index,
                                   fieldname='flood_redirect_lat_index')
    flood_redirect_lon_index_filename = path.join(output_filepath,
                                                  "flood_ri_lon_" +
                                                  output_filelabel + ".nc")
    iodriver.advanced_field_writer(flood_redirect_lon_index_filename,
                                   flood_redirect_lon_index,
                                   fieldname='flood_redirect_lon_index')
    connect_redirect_lat_index_filename = path.join(output_filepath,
                                                    "connect_ri_lat_" +
                                                    output_filelabel + ".nc")
    iodriver.advanced_field_writer(connect_redirect_lat_index_filename,
                                   connect_redirect_lat_index,
                                   fieldname='connect_redirect_lat_index')
    connect_redirect_lon_index_filename = path.join(output_filepath,
                                                    "connect_ri_lon_" +
                                                    output_filelabel + ".nc")
    iodriver.advanced_field_writer(connect_redirect_lon_index_filename,
                                   connect_redirect_lon_index,
                                   fieldname='connect_redirect_lon_index')
    additional_flood_redirect_lat_index_filename = path.join(output_filepath,
                                                  "additional_flood_ri_lat_" +
                                                  output_filelabel + ".nc")
    iodriver.advanced_field_writer(additional_flood_redirect_lat_index_filename,
                                   additional_flood_redirect_lat_index,
                                   fieldname='additional_flood_redirect_lat_index')
    additional_flood_redirect_lon_index_filename = path.join(output_filepath,
                                                  "additional_flood_ri_lon_" +
                                                  output_filelabel + ".nc")
    iodriver.advanced_field_writer(additional_flood_redirect_lon_index_filename,
                                   additional_flood_redirect_lon_index,
                                   fieldname='additional_flood_redirect_lon_index')
    additional_connect_redirect_lat_index_filename = path.join(output_filepath,
                                                    "additional_connect_ri_lat_" +
                                                    output_filelabel + ".nc")
    iodriver.advanced_field_writer(additional_connect_redirect_lat_index_filename,
                                   additional_connect_redirect_lat_index,
                                   fieldname='additional_connect_redirect_lat_index')
    additional_connect_redirect_lon_index_filename = path.join(output_filepath,
                                                    "additional_connect_ri_lon_" +
                                                    output_filelabel + ".nc")
    iodriver.advanced_field_writer(additional_connect_redirect_lon_index_filename,
                                   additional_connect_redirect_lon_index,
                                   fieldname='additional_connect_redirect_lon_index')
    flood_local_redirect_filename = path.join(output_filepath,
                                              "flood_local_r_" +
                                              output_filelabel + ".nc")
    iodriver.advanced_field_writer(flood_local_redirect_filename,
                                   flood_local_redirect,
                                   fieldname='flood_local_redirect')
    connect_local_redirect_filename = path.join(output_filepath,
                                                "connect_local_r_" +
                                                output_filelabel + ".nc")
    iodriver.advanced_field_writer(connect_local_redirect_filename,
                                   connect_local_redirect,
                                   fieldname='connect_local_redirect')
    additional_flood_local_redirect_filename = path.join(output_filepath,
                                              "additional_flood_local_r_" +
                                              output_filelabel + ".nc")
    iodriver.advanced_field_writer(additional_flood_local_redirect_filename,
                                   additional_flood_local_redirect,
                                   fieldname='additional_flood_local_redirect')
    additional_connect_local_redirect_filename = path.join(output_filepath,
                                                "additional_connect_local_r_" +
                                                output_filelabel + ".nc")
    iodriver.advanced_field_writer(additional_connect_local_redirect_filename,
                                   additional_connect_local_redirect,
                                   fieldname='additional_connect_local_redirect')
    merge_points_filename = path.join(output_filepath,
                                      "merge_points_" +
                                      output_filelabel + ".nc")
    iodriver.advanced_field_writer(merge_points_filename,
                                   merge_points,
                                   fieldname='merge_points')
    lake_centers_filename = path.join(output_filepath,
                                      "lake_centers_" +
                                      output_filelabel + ".nc")
    iodriver.advanced_field_writer(lake_centers_filename,
                                   input_minima,
                                   fieldname="lake_centers")
    individual_field_filenames = [connection_volume_thresholds_filename,
                                  flood_volume_thresholds_filename,
                                  flood_next_cell_lat_index_filename,
                                  flood_next_cell_lon_index_filename,
                                  connect_next_cell_lat_index_filename,
                                  connect_next_cell_lon_index_filename,
                                  flood_force_merge_lat_index_filename,
                                  flood_force_merge_lon_index_filename,
                                  connect_force_merge_lat_index_filename,
                                  connect_force_merge_lon_index_filename,
                                  flood_redirect_lat_index_filename,
                                  flood_redirect_lon_index_filename,
                                  connect_redirect_lat_index_filename,
                                  connect_redirect_lon_index_filename,
                                  additional_flood_redirect_lat_index_filename,
                                  additional_flood_redirect_lon_index_filename,
                                  additional_connect_redirect_lat_index_filename,
                                  additional_connect_redirect_lon_index_filename,
                                  flood_local_redirect_filename,
                                  connect_local_redirect_filename,
                                  additional_flood_local_redirect_filename,
                                  additional_connect_local_redirect_filename,
                                  merge_points_filename,
                                  lake_centers_filename]
    cdo_inst = cdo.Cdo()
    cdo_inst.merge(input=" ".join(individual_field_filenames),
                   output=combined_output_filename)
    for individual_field_filename in individual_field_filenames:
      os.remove(individual_field_filename)
