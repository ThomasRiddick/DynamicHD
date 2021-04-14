'''
Routines to extract lake volumes from lake data
Created on July 29, 2020

@author: thomasriddick
'''
import numpy as np
from .libs import create_connected_lsmask_wrapper
from .libs import lake_operators_wrapper
from . import compute_catchments as cc
from . import determine_river_directions
from . import field
from . import iodriver
import tempfile
import os

def extract_lake_volumes(flood_volume_thresholds,
                         basin_catchment_numbers,merge_types):
    lake_mask = basin_catchment_numbers.greater_than_value(0)
    lake_volumes =  field.makeEmptyField("Generic",np.float64,
                                         grid_type=lake_mask.get_grid())
    lake_mask_reduced = lake_mask.copy()
    lake_mask_reduced.change_dtype(np.int32)
    lake_operators_wrapper.reduce_connected_areas_to_points(lake_mask_reduced.get_data(),True)
    lake_mask_inv = lake_mask.copy()
    lake_mask_inv.invert_data()
    orography =  field.makeEmptyField("Generic",np.float64,
                                      grid_type=lake_mask.get_grid())
    orography.get_data()[lake_mask_inv.get_data()] = 3.0
    orography.get_data()[lake_mask.get_data()] = 1.0
    orography.get_data()[lake_mask_reduced.get_data().astype(bool)] = 0.0
    lake_mask_inv.change_dtype(np.int32)
    rdirs = determine_river_directions.determine_river_directions(orography,
                                                                  lake_mask_reduced,
                                                                  truesinks=None,
                                                                  always_flow_to_sea=True,
                                                                  use_diagonal_nbrs=True,
                                                                  mark_pits_as_true_sinks=False)
    rdirs.get_data()[lake_mask_inv.get_data()] = -1
    simplified_basin_catchment_numbers = None
    temporary_file = tempfile.NamedTemporaryFile(delete = False)
    try:
        temporary_file.close()
        dummy_loop_log_filepath = temporary_file.name
        simplified_basin_catchment_numbers = \
            field.Field(cc.compute_catchments_cpp(rdirs.get_data(),loop_logfile=dummy_loop_log_filepath),
                                                  grid=lake_mask.get_grid())
    finally:
        os.remove(temporary_file.name)
    for i in range(1,simplified_basin_catchment_numbers.find_maximum()+1):
        if i in simplified_basin_catchment_numbers.get_data():
            single_lake_mask = lake_mask.copy()
            single_lake_mask.\
                get_data()[simplified_basin_catchment_numbers.get_data() != i] = False
            single_lake_flood_volume_thresholds = flood_volume_thresholds.copy()
            single_lake_flood_volume_thresholds.\
                get_data()[np.logical_not(single_lake_mask.get_data())] = 0.0
            single_lake_flood_volume_thresholds.\
                get_data()[np.logical_and(merge_types.get_data() != 10,
                                          merge_types.get_data() != 11)] = 0.0
            lake_volumes.get_data()[single_lake_mask.get_data()] = np.sum(single_lake_flood_volume_thresholds.get_data())
    return lake_volumes

def lake_volume_extraction_driver(lake_parameters_filepath,
                                  basin_catchment_numbers_filepath,
                                  lake_volumes_out_filepath):
    flood_volume_thresholds = iodriver.advanced_field_loader(lake_parameters_filepath,
                                                             field_type='Generic',
                                                             fieldname="flood_volume_thresholds")
    merge_types = iodriver.advanced_field_loader(lake_parameters_filepath,
                                                 field_type='Generic',
                                                 fieldname="merge_points")
    basin_catchment_numbers = iodriver.advanced_field_loader(basin_catchment_numbers_filepath,
                                                             field_type="Generic",
                                                             fieldname="basin_catchment_numbers")
    lake_volumes = extract_lake_volumes(flood_volume_thresholds,
                                        basin_catchment_numbers,merge_types)
    iodriver.advanced_field_writer(lake_volumes_out_filepath,lake_volumes,
                                   fieldname="lake_volume")




