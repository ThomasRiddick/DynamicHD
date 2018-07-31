'''
Created on March 23, 2018

@author: thomasriddick
'''
import iodriver
import libs.lake_operators_wrapper as lake_operators_wrapper  #@UnresolvedImport
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
