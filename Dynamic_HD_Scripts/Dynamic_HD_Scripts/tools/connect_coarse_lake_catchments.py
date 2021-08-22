'''
Routines to connect coarse catchments ending in a terminal lake to
the catchment that lake overflows into when full. Also reconnects
cumulative flows
Created on February 13, 2021

@author: thomasriddick
'''
from collections import Counter
import numpy as np
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.base import field
from Dynamic_HD_Scripts.interface.cpp_interface.libs \
    import follow_streams_wrapper
from Dynamic_HD_Scripts.tools import compute_catchments as cc

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

    def pop_leaves(self):
        leaves = {catchment:node for catchment,node in
                  self.all_catchments.items() if not node.subcatchments}
        self.all_catchments = {catchment:node for catchment,node in
                               self.all_catchments.items() if node.subcatchments}
        for catchment_obj in list(self.all_catchments.values()):
            for leaf_catchment_num in leaves.keys():
                catchment_obj.subcatchments.pop(leaf_catchment_num,None)
        return leaves

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
            subcatchments_nums = list(self.subcatchments.keys())
            for subcatchment_obj in list(self.subcatchments.values()):
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

def update_cumulative_flow(upstream_catchment_center,
                           downstream_catchment_entry_point,
                           cumulative_flow,river_directions):
    additional_cumulative_flow = cumulative_flow.get_data()[tuple(upstream_catchment_center)]
    downstream_catchment_entry_point_in_field = \
        np.zeros(river_directions.get_data().shape,dtype=np.int32)
    downstream_catchment_entry_point_in_field[tuple(downstream_catchment_entry_point)] = 1
    downstream_cells_out = np.zeros(river_directions.get_data().shape,dtype=np.int32)
    follow_streams_wrapper.follow_streams(river_directions.get_data().astype(np.float64),
                                          downstream_catchment_entry_point_in_field,
                                          downstream_cells_out,True)
    cumulative_flow.get_data()[downstream_cells_out == 1] += additional_cumulative_flow

def connect_coarse_lake_catchments_driver(coarse_catchments_filepath,
                                          lake_parameters_filepath,
                                          basin_catchment_numbers_filepath,
                                          river_directions_filepath,
                                          connected_coarse_catchments_out_filename,
                                          coarse_catchments_fieldname,
                                          connected_coarse_catchments_out_fieldname,
                                          basin_catchment_numbers_fieldname,
                                          river_directions_fieldname,
                                          cumulative_flow_filepath = None,
                                          connected_cumulative_flow_out_filepath=None,
                                          cumulative_flow_fieldname = None,
                                          connected_cumulative_flow_out_fieldname=None,
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
    if cumulative_flow_filepath is not None:
        cumulative_flow = iodriver.advanced_field_loader(cumulative_flow_filepath,
                                                         field_type='Generic',
                                                         fieldname=\
                                                         cumulative_flow_fieldname)
    catchments, corrected_cumulative_flow = \
        connect_coarse_lake_catchments(coarse_catchments,lake_centers,basin_catchment_numbers,
                                       flood_next_cell_index_lat,flood_next_cell_index_lon,
                                       flood_redirect_lat,flood_redirect_lon,
                                       additional_flood_redirect_lat,
                                       additional_flood_redirect_lon,
                                       local_redirect,additional_local_redirect,
                                       merge_types,river_directions,scale_factor,
                                       cumulative_flow=(cumulative_flow if cumulative_flow_filepath
                                                        is not None else None),
                                       correct_cumulative_flow=(True if cumulative_flow_filepath
                                                                is not None else False))
    iodriver.advanced_field_writer(connected_coarse_catchments_out_filename,
                                   field=catchments,
                                   fieldname=connected_coarse_catchments_out_fieldname)
    if cumulative_flow_filepath is not None:
        iodriver.advanced_field_writer(connected_cumulative_flow_out_filepath,
                                       field=corrected_cumulative_flow,
                                       fieldname=connected_cumulative_flow_out_fieldname)


#Remember - Tuples trigger basic indexing, lists don't
def connect_coarse_lake_catchments(coarse_catchments,lake_centers,basin_catchment_numbers,
                                   flood_next_cell_index_lat,flood_next_cell_index_lon,
                                   flood_redirect_lat,flood_redirect_lon,
                                   additional_flood_redirect_lat,additional_flood_redirect_lon,
                                   local_redirect,additional_local_redirect,
                                   merge_types,river_directions,scale_factor = 3,
                                   correct_cumulative_flow=False,
                                   cumulative_flow=None):
    if correct_cumulative_flow:
        if cumulative_flow is None or river_directions is None:
            raise RuntimeError("Required input files for cumulative flow correction not provided")
        old_coarse_catchments = coarse_catchments.copy()
    lake_centers_array = np.argwhere(lake_centers.get_data())
    lake_centers_list = [lake_centers_array[i,:].tolist()
                         for i in range(lake_centers_array.shape[0])]
    overflow_catchments = field.makeEmptyField(field_type='Generic',dtype=np.int64,
                                               grid_type=lake_centers.get_grid())
    overflow_coords_lats = field.makeEmptyField(field_type='Generic',dtype=np.int64,
                                                grid_type=lake_centers.get_grid())
    overflow_coords_lons = field.makeEmptyField(field_type='Generic',dtype=np.int64,
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
                        overflow_coords = (additional_flood_redirect_lat.get_data()[tuple(secondary_merge_coords)],
                                           additional_flood_redirect_lon.get_data()[tuple(secondary_merge_coords)])
                    else:
                        overflow_catchment = \
                            coarse_catchments.get_data()[flood_redirect_lat.get_data()[tuple(secondary_merge_coords)],
                                                         flood_redirect_lon.get_data()[tuple(secondary_merge_coords)]]
                        overflow_coords = (flood_redirect_lat.get_data()[tuple(secondary_merge_coords)],
                                           flood_redirect_lon.get_data()[tuple(secondary_merge_coords)])
                    break
        overflow_catchments.get_data()[tuple(lake_center_coords)] = overflow_catchment
        overflow_coords_lats.get_data()[tuple(lake_center_coords)] = overflow_coords[0]
        overflow_coords_lons.get_data()[tuple(lake_center_coords)] = overflow_coords[1]
    #specific to latlon grid
    sink_points_array = np.argwhere(np.logical_or(river_directions.get_data() == 5,
                                                  river_directions.get_data() == -2))
    sink_points_list = [sink_points_array[i,:].tolist()
                         for i in range(sink_points_array.shape[0])]
    catchment_trees = CatchmentTrees()
    sink_point_cumulative_flow_redirect_lat = field.makeEmptyField(field_type='Generic',dtype=np.int64,
                                                                   grid_type=coarse_catchments.get_grid())
    sink_point_cumulative_flow_redirect_lon = field.makeEmptyField(field_type='Generic',dtype=np.int64,
                                                                   grid_type=coarse_catchments.get_grid())
    for sink_point in sink_points_list:
        sink_point_coarse_catchment = coarse_catchments.get_data()[tuple(sink_point)]
        #Specific to lat-lon grids
        overflow_catch_fine_cells_in_coarse_cell = overflow_catchments.get_data()[sink_point[0]*scale_factor:(sink_point[0]+1)*scale_factor,
                                                                                  sink_point[1]*scale_factor:(sink_point[1]+1)*scale_factor]
        overflow_coords_lat_fine_cells_in_coarse_cell = overflow_coords_lats.get_data()[sink_point[0]*scale_factor:(sink_point[0]+1)*scale_factor,
                                                                                       sink_point[1]*scale_factor:(sink_point[1]+1)*scale_factor]
        overflow_coords_lon_fine_cells_in_coarse_cell = overflow_coords_lons.get_data()[sink_point[0]*scale_factor:(sink_point[0]+1)*scale_factor,
                                                                                       sink_point[1]*scale_factor:(sink_point[1]+1)*scale_factor]
        overflow_catchment_list = overflow_catch_fine_cells_in_coarse_cell[overflow_catch_fine_cells_in_coarse_cell != 0].tolist()
        if not overflow_catchment_list:
            continue
        overflow_catchment_counters = Counter(overflow_catchment_list)
        highest_count = max(overflow_catchment_counters.values())
        overflow_catchment = [ value for value,count in overflow_catchment_counters.items() if count == highest_count][0]
        if sink_point_coarse_catchment == overflow_catchment:
            continue
        overflow_catchment_fine_coords_within_coarse_cell = \
            tuple(np.argwhere(overflow_catch_fine_cells_in_coarse_cell == overflow_catchment)[0,:].tolist())
        sink_point_cumulative_flow_redirect_lat.get_data()[tuple(sink_point)] = \
            overflow_coords_lat_fine_cells_in_coarse_cell[overflow_catchment_fine_coords_within_coarse_cell]
        sink_point_cumulative_flow_redirect_lon.get_data()[tuple(sink_point)] = \
           overflow_coords_lon_fine_cells_in_coarse_cell[overflow_catchment_fine_coords_within_coarse_cell]
        catchment_trees.add_link(sink_point_coarse_catchment,overflow_catchment)
    for supercatchment_number,tree in catchment_trees.primary_catchments.items():
        for subcatchments_num in tree.get_all_subcatchment_nums():
            coarse_catchments.get_data()[subcatchments_num == coarse_catchments.get_data()] = \
                supercatchment_number
    if correct_cumulative_flow:
        while catchment_trees.all_catchments:
            upstream_catchments = catchment_trees.pop_leaves()
            for upstream_catchment in upstream_catchments:
                if np.any(np.logical_and(np.logical_or(river_directions.get_data() == 5,
                                                       river_directions.get_data() == -2),
                                         old_coarse_catchments.get_data() == upstream_catchment)):
                    upstream_catchment_center = \
                        tuple(np.argwhere(np.logical_and(np.logical_or(river_directions.get_data() == 5,
                                                                       river_directions.get_data() == -2),
                                          old_coarse_catchments.get_data() == upstream_catchment))[0,:].tolist())
                    update_cumulative_flow(upstream_catchment_center,
                                           (sink_point_cumulative_flow_redirect_lat.get_data()[upstream_catchment_center],
                                            sink_point_cumulative_flow_redirect_lon.get_data()[upstream_catchment_center]),
                                           cumulative_flow,river_directions)
        return field.Field(cc.renumber_catchments_by_size(coarse_catchments.get_data()),type="Generic",
                       grid=coarse_catchments.get_grid()),cumulative_flow
    return field.Field(cc.renumber_catchments_by_size(coarse_catchments.get_data()),type="Generic",
                       grid=coarse_catchments.get_grid()),None
