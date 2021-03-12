'''
Routines to connect coarse catchments ending in a terminal lake to
the catchment that lake overflows into when full
Created on February 13, 2021

@author: thomasriddick
'''
from collections import Counter
import iodriver
import compute_catchments as cc
import numpy as np
import field

class CatchmentTrees(object):

    def __init__(self):
        self.primary_catchments = {}
        self.all_catchments = {}

    def add_link(self,from_catchment,to_catchment):
        if from_catchment in self.primary_catchments:
            from_catchment_obj = self.primary_catchments.pop(from_catchment)
        else:
            from_catchment_obj = CatchmentNode()
            self.all_catchments[from_catchment] = from_catchment_obj
        if to_catchment in self.all_catchments:
            to_catchment_obj = self.all_catchments[to_catchment]
        else:
            to_catchment_obj = CatchmentNode()
            self.primary_catchments[to_catchment] = to_catchment_obj
            self.all_catchments[to_catchment] = to_catchment_obj
        from_catchment_obj.add_supercatchment(to_catchment,to_catchment_obj)
        to_catchment_obj.add_subcatchment(from_catchment,from_catchment_obj)

    def get_nested_dictionary_of_subcatchments(self):
        subcatchment_dict = {}
        for catchment_num,catchment_obj in self.primary_catchments.items():
           subcatchment_dict[catchment_num] = \
            catchment_obj.get_nested_dictionary_of_subcatchments()
        return  subcatchment_dict

class CatchmentNode(object):

    def __init__(self,supercatchment_num_in=None,supercatchment_obj_in=None):
        self.supercatchment_num = supercatchment_num_in
        self.supercatchment_obj = supercatchment_obj_in
        self.subcatchments = {}

    def add_supercatchment(self,supercatchment_num_in,supercatchment_obj_in):
        self.supercatchment_num = supercatchment_num_in
        self.supercatchment_obj = supercatchment_obj_in

    def add_subcatchment(self,subcatchment_number,subcatchment_obj):
        self.subcatchments[subcatchment_number] = subcatchment_obj

    def get_all_subcatchment_nums(self):
        if self.subcatchments:
            subcatchments_nums = self.subcatchments.keys()
            for subcatchment_obj in self.subcatchments.values():
                subcatchments_nums += subcatchment_obj.get_all_subcatchment_nums()
            return subcatchments_nums
        else:
            return []

    def get_nested_dictionary_of_subcatchments(self):
        subcatchment_dict = {}
        if self.subcatchments:
            for catchment_num,catchment_obj in self.subcatchments.items():
               subcatchment_dict[catchment_num] = \
                catchment_obj.get_nested_dictionary_of_subcatchments()
        return  subcatchment_dict

def connect_coarse_lake_catchments_driver(coarse_catchments_filepath,
                                          lake_parameters_filepath,
                                          basin_catchment_numbers_filepath,
                                          river_directions_filepath,
                                          connected_coarse_catchments_out_filename,
                                          coarse_catchments_fieldname,
                                          connected_coarse_catchments_out_fieldname,
                                          basin_catchment_numbers_fieldname,
                                          river_directions_fieldname,
                                          scale_factor = 3):
    coarse_catchments = iodriver.advanced_field_loader(coarse_catchments_filepath,
                                                       field_type='Generic',
                                                       fieldname=\
                                                       coarse_catchments_fieldname)
    basin_catchment_numbers = iodriver.advanced_field_loader(basin_catchment_numbers_filepath,
                                                   field_type='Generic',
                                                   fieldname=\
                                                   basin_catchment_numbers_fieldname)
    lake_centers = iodriver.advanced_field_loader(lake_parameters_filepath,
                                                  field_type='Generic',
                                                  fieldname=
                                                  "lake_centers")
    flood_next_cell_index_lat = iodriver.advanced_field_loader(lake_parameters_filepath,
                                                               field_type='Generic',
                                                               fieldname=
                                                               "flood_next_cell_lat_index")
    flood_next_cell_index_lon = iodriver.advanced_field_loader(lake_parameters_filepath,
                                                               field_type='Generic',
                                                               fieldname=
                                                               "flood_next_cell_lon_index")
    flood_redirect_lat = iodriver.advanced_field_loader(lake_parameters_filepath,
                                                        field_type='Generic',
                                                        fieldname=
                                                        "flood_redirect_lat_index")
    flood_redirect_lon = iodriver.advanced_field_loader(lake_parameters_filepath,
                                                        field_type='Generic',
                                                        fieldname=
                                                        "flood_redirect_lon_index")
    additional_flood_redirect_lat = iodriver.advanced_field_loader(lake_parameters_filepath,
                                                        field_type='Generic',
                                                        fieldname=
                                                        "additional_flood_redirect_lat_index")
    additional_flood_redirect_lon = iodriver.advanced_field_loader(lake_parameters_filepath,
                                                        field_type='Generic',
                                                        fieldname=
                                                        "additional_flood_redirect_lon_index")
    local_redirect = iodriver.advanced_field_loader(lake_parameters_filepath,
                                                    field_type='Generic',
                                                    fieldname=\
                                                    "flood_local_redirect")
    additional_local_redirect = iodriver.advanced_field_loader(lake_parameters_filepath,
                                                               field_type='Generic',
                                                               fieldname=\
                                                               "additional_flood_local_redirect")
    merge_types = iodriver.advanced_field_loader(lake_parameters_filepath,
                                                 field_type='Generic',
                                                 fieldname=\
                                                 "merge_points")
    river_directions = iodriver.advanced_field_loader(river_directions_filepath,
                                                      field_type='Generic',
                                                      fieldname=\
                                                      river_directions_fieldname)
    catchments = connect_coarse_lake_catchments(coarse_catchments,lake_centers,basin_catchment_numbers,
                                                flood_next_cell_index_lat,flood_next_cell_index_lon,
                                                flood_redirect_lat,flood_redirect_lon,
                                                additional_flood_redirect_lat,
                                                additional_flood_redirect_lon,
                                                local_redirect,additional_local_redirect,
                                                merge_types,river_directions,scale_factor)
    iodriver.advanced_field_writer(connected_coarse_catchments_out_filename,
                                   field=catchments,
                                   fieldname=connected_coarse_catchments_out_fieldname)


#Remember - Tuples trigger basic indexing, lists don't
def connect_coarse_lake_catchments(coarse_catchments,lake_centers,basin_catchment_numbers,
                                   flood_next_cell_index_lat,flood_next_cell_index_lon,
                                   flood_redirect_lat,flood_redirect_lon,
                                   additional_flood_redirect_lat,additional_flood_redirect_lon,
                                   local_redirect,additional_local_redirect,
                                   merge_types,river_directions,scale_factor = 3):
    lake_centers_array = np.argwhere(lake_centers.get_data())
    lake_centers_list = [lake_centers_array[i,:].tolist()
                         for i in range(lake_centers_array.shape[0])]
    overflow_catchments = field.makeEmptyField(field_type='Generic',dtype=np.int64,
                                               grid_type=lake_centers.get_grid())
    for lake_center_coords in lake_centers_list:
        basin_number = basin_catchment_numbers.get_data()[tuple(lake_center_coords)]
        while True:
            secondary_merge_coords = np.argwhere(
                                     np.logical_and(
                                     np.logical_or(merge_types.get_data() == 10,
                                                   merge_types.get_data() == 11),
                                     basin_catchment_numbers.get_data() == basin_number))[0,:].tolist()
            double_merge = np.any(np.logical_and(merge_types.get_data() == 11,
                                  basin_catchment_numbers.get_data() == basin_number))
            basin_number = basin_catchment_numbers.get_data()[flood_next_cell_index_lat.get_data()[tuple(secondary_merge_coords)],
                                                              flood_next_cell_index_lon.get_data()[tuple(secondary_merge_coords)]]
            if basin_number == 0:
                is_local_redirect = ((local_redirect.get_data()[tuple(secondary_merge_coords)]) if not double_merge else
                                     (additional_local_redirect.get_data()[tuple(secondary_merge_coords)]))
                if is_local_redirect:
                    if double_merge:
                        basin_number = \
                            basin_catchment_numbers.\
                            get_data()[additional_flood_redirect_lat.get_data()[tuple(secondary_merge_coords)],
                                       additional_flood_redirect_lon.get_data()[tuple(secondary_merge_coords)]]
                    else:
                        basin_number = \
                            basin_catchment_numbers.\
                            get_data()[flood_redirect_lat.get_data()[tuple(secondary_merge_coords)],
                                       flood_redirect_lon.get_data()[tuple(secondary_merge_coords)]]
                else:
                    if double_merge:
                        overflow_catchment = \
                            coarse_catchments.get_data()[additional_flood_redirect_lat.get_data()[tuple(secondary_merge_coords)],
                                                         additional_flood_redirect_lon.get_data()[tuple(secondary_merge_coords)]]

                    else:
                        overflow_catchment = \
                            coarse_catchments.get_data()[flood_redirect_lat.get_data()[tuple(secondary_merge_coords)],
                                                         flood_redirect_lon.get_data()[tuple(secondary_merge_coords)]]
                    break
        overflow_catchments.get_data()[tuple(lake_center_coords)] = overflow_catchment
    #specific to latlon grid
    sink_points_array = np.argwhere(river_directions.get_data() == 5)
    sink_points_list = [sink_points_array[i,:].tolist()
                         for i in range(sink_points_array.shape[0])]
    catchment_trees = CatchmentTrees()
    for sink_point in sink_points_list:
        sink_point_coarse_catchment = coarse_catchments.get_data()[tuple(sink_point)]
        #Specific to lat-lon grids
        overflow_catch_fine_cells_in_coarse_cell = overflow_catchments.get_data()[sink_point[0]*scale_factor:(sink_point[0]+1)*scale_factor,
                                                                                  sink_point[1]*scale_factor:(sink_point[1]+1)*scale_factor]
        overflow_catchment_list = overflow_catch_fine_cells_in_coarse_cell[overflow_catch_fine_cells_in_coarse_cell != 0].tolist()
        if not overflow_catchment_list:
            continue
        overflow_catchment_counters = Counter(overflow_catchment_list)
        highest_count = max(overflow_catchment_counters.values())
        overflow_catchment = [ value for value,count in overflow_catchment_counters.items() if count == highest_count][0]
        if sink_point_coarse_catchment == overflow_catchment:
            continue
        catchment_trees.add_link(sink_point_coarse_catchment,overflow_catchment)
    for supercatchment_number,tree in catchment_trees.primary_catchments.items():
        for subcatchments_num in tree.get_all_subcatchment_nums():
            coarse_catchments.get_data()[subcatchments_num == coarse_catchments.get_data()] = \
                supercatchment_number
    return field.Field(cc.renumber_catchments_by_size(coarse_catchments.get_data()),type="Generic",
                       grid=coarse_catchments.get_grid())
