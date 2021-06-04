'''
Created on March 17, 2019

@author: thomasriddick
'''

from Dynamic_HD_Scripts import field
from Dynamic_HD_Scripts import iodriver
import numpy as np
from Dynamic_HD_Scripts.libs import determine_river_directions_wrapper

def determine_river_directions(orography,lsmask,truesinks=None,
                               always_flow_to_sea=True,
                               use_diagonal_nbrs=True,
                               mark_pits_as_true_sinks=True):
    rdirs = field.makeEmptyField('RiverDirections',np.float64,orography.get_grid())
    if truesinks is None:
        truesinks = field.makeEmptyField('Generic',np.float64,orography.get_grid())
    determine_river_directions_wrapper.determine_river_directions(rdirs.get_data(),
                                                                  orography.get_data(),
                                                                  lsmask.get_data().astype(dtype=np.int32,
                                                                                           order='C',
                                                                                           copy=False),
                                                                  truesinks.get_data().astype(dtype=np.int32,
                                                                                           order='C',
                                                                                           copy=False),
                                                                  always_flow_to_sea,
                                                                  use_diagonal_nbrs,
                                                                  mark_pits_as_true_sinks)
    return rdirs

def advanced_river_direction_determination_driver(rdirs_filename,
                                                  orography_filename,
                                                  lsmask_filename,
                                                  truesinks_filename=None,
                                                  rdirs_fieldname=None,
                                                  orography_fieldname=None,
                                                  lsmask_fieldname=None,
                                                  truesinks_fieldname=None,
                                                  always_flow_to_sea=True,
                                                  use_diagonal_nbrs=True,
                                                  mark_pits_as_true_sinks=True):
    orography = iodriver.advanced_field_loader(orography_filename,
                                               field_type='Orography',
                                               fieldname=orography_fieldname,
                                               adjust_orientation=True)
    lsmask = iodriver.advanced_field_loader(lsmask_filename,
                                            field_type='Generic',
                                            fieldname=lsmask_fieldname,
                                            adjust_orientation=True)
    if truesinks_filename is None:
        truesinks = None
    else:
        truesinks = iodriver.advanced_field_loader(truesinks_filename,
                                                   field_type='Generic',
                                                   fieldname=truesinks_filename,
                                                   adjust_orientation=True)
    rdirs = determine_river_directions(orography,
                                       lsmask,
                                       truesinks,
                                       always_flow_to_sea,
                                       use_diagonal_nbrs,
                                       mark_pits_as_true_sinks)
    iodriver.advanced_field_writer(rdirs_filename,rdirs,
                                   fieldname=rdirs_fieldname)

