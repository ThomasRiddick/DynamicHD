'''
Routines to extract lake volumes from lake data
Created on July 29, 2020

@author: thomasriddick
'''
import numpy as np
import tempfile
import os
from netCDF4 import Dataset
from Dynamic_HD_Scripts.base import field
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.tools import compute_catchments as cc
from Dynamic_HD_Scripts.tools import determine_river_directions
from Dynamic_HD_Scripts.tools import connect_coarse_lake_catchments as cclc
import create_connected_lsmask_wrapper
import lake_operators_wrapper

def extract_lake_volumes(flood_volume_thresholds,
                         basin_catchment_numbers,
                         flood_merge_and_redirect_indices_index,
                         merges_and_redirects):
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
    secondary_merges = field.Field(np.full(lake_mask.get_data().shape,False,
                                           dtype=bool),
                                   grid=lake_mask.get_grid())
    iterator = np.nditer(flood_merge_and_redirect_indices_index.get_data(),
                         flags=["multi_index"])
    for i in iterator:
        if i != -1:
            secondary_merges.get_data()[iterator.multi_index[0],
                                        iterator.multi_index[1]] = \
                merges_and_redirects[i].secondary_merge
    for i in range(1,simplified_basin_catchment_numbers.find_maximum()+1):
        if i in simplified_basin_catchment_numbers.get_data():
            single_lake_mask = lake_mask.copy()
            single_lake_mask.\
                get_data()[simplified_basin_catchment_numbers.get_data() != i] = False
            single_lake_flood_volume_thresholds = flood_volume_thresholds.copy()
            single_lake_flood_volume_thresholds.\
                get_data()[np.logical_not(single_lake_mask.get_data())] = 0.0
            single_lake_flood_volume_thresholds.\
                get_data()[np.logical_not(secondary_merges.get_data())] = 0.0
            lake_volumes.get_data()[single_lake_mask.get_data()] = np.sum(single_lake_flood_volume_thresholds.get_data())
    return lake_volumes

def lake_volume_extraction_driver(lake_parameters_filepath,
                                  basin_catchment_numbers_filepath,
                                  lake_volumes_out_filepath):
    flood_volume_thresholds = iodriver.advanced_field_loader(lake_parameters_filepath,
                                                             field_type='Generic',
                                                             fieldname="flood_volume_thresholds")
    flood_merge_and_redirect_indices_index = \
        iodriver.advanced_field_loader(lake_parameters_filepath,
                                       field_type='Generic',
                                       fieldname=
                                       "flood_merge_and_redirect_indices_index")
    with Dataset(lake_parameters_filepath,mode='r',format='NETCDF4') as dataset:
        merges_and_redirects_array = \
            np.array(dataset.variables["flood_merges_and_redirects"][:,:,:])
    merges_and_redirects = \
        cclc.create_merge_indices_collections_from_array(merges_and_redirects_array)
    basin_catchment_numbers = iodriver.advanced_field_loader(basin_catchment_numbers_filepath,
                                                             field_type="Generic",
                                                             fieldname="basin_catchment_numbers")
    lake_volumes = extract_lake_volumes(flood_volume_thresholds,
                                        basin_catchment_numbers,
                                        flood_merge_and_redirect_indices_index,
                                        merges_and_redirects)
    iodriver.advanced_field_writer(lake_volumes_out_filepath,lake_volumes,
                                   fieldname="lake_volume")




