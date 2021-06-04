'''
Driver for orography creation C++ code
Created on May 23, 2020

@author: thomasriddick
'''
from Dynamic_HD_Scripts.libs import create_orography_wrapper
from Dynamic_HD_Scripts import iodriver
from Dynamic_HD_Scripts import field
import numpy as np

def advanced_orography_creation_driver(landsea_mask_filename,
                                       inclines_filename,
                                       orography_filename,
                                       landsea_mask_fieldname,
                                       inclines_fieldname,
                                       orography_fieldname):
    landsea_in = iodriver.advanced_field_loader(landsea_mask_filename,
                                                field_type="Generic",
                                                fieldname=landsea_mask_fieldname,
                                                adjust_orientation=True)
    inclines_in = iodriver.advanced_field_loader(inclines_filename,
                                                 field_type="Generic",
                                                 fieldname=inclines_fieldname,
                                                 adjust_orientation=True)
    orography_in = field.makeEmptyField('Generic',np.float64,landsea_in.get_grid())
    create_orography_wrapper.create_orography(np.ascontiguousarray(landsea_in.get_data(),
                                                                   dtype=np.int32),
                                              np.ascontiguousarray(inclines_in.get_data(),
                                                                   dtype=np.float64),
                                              np.ascontiguousarray(orography_in.get_data(),
                                                                   dtype=np.float64))
    iodriver.advanced_field_writer(orography_filename,
                                   orography_in,
                                   fieldname=orography_fieldname)
