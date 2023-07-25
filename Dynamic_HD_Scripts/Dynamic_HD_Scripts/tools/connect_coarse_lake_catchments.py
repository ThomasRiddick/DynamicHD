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
from netCDF4 import Dataset

class MergeAndRedirectIndices:
    pass

class LatLonMergeAndRedirectIndices(MergeAndRedirectIndices):

    def __init__(self,
                 is_primary_merge=False,
                 local_redirect=False,
                 merge_target_lat_index=-1,
                 merge_target_lon_index=-1,
                 redirect_lat_index=-1,
                 redirect_lon_index=-1):
        self.is_primary_merge = is_primary_merge
        self.local_redirect = local_redirect
        self.merge_target_lat_index = merge_target_lat_index
        self.merge_target_lon_index = merge_target_lon_index
        self.redirect_lat_index = redirect_lat_index
        self.redirect_lon_index = redirect_lon_index

    def read_merge_and_redirect_indices_from_array(self,is_primary_merge,array_in):
        self.is_primary_merge = is_primary_merge
        self.local_redirect =  (array_in[1]==1)
        self.merge_target_lat_index = array_in[2]
        self.merge_target_lon_index = array_in[3]
        self.redirect_lat_index = array_in[4]
        self.redirect_lon_index = array_in[5]

class  MergeAndRedirectIndicesCollection:
    def __init__(self,
                 primary_merge_and_redirect_indices,
                 secondary_merge_and_redirect_indices):
      self.primary_merge = (primary_merge_and_redirect_indices is not None)
      self.secondary_merge = (secondary_merge_and_redirect_indices is not None)
      self.primary_merge_and_redirect_indices = \
        primary_merge_and_redirect_indices
      self.secondary_merge_and_redirect_indices = \
        secondary_merge_and_redirect_indices

def create_merge_indices_collections_from_array(array_in):
    merge_and_redirect_indices_collections = []
    for row in array_in:
        primary_merge_and_redirect_indices = []
        for j in range(row.shape[0]):
            if row[j,0] == 1:
                working_merge_and_redirect_indices = LatLonMergeAndRedirectIndices()
                working_merge_and_redirect_indices.read_merge_and_redirect_indices_from_array((j != 1),row[j,:])
                if j == 0:
                    secondary_merge_and_redirect_indices = working_merge_and_redirect_indices
                else:
                    primary_merge_and_redirect_indices.append(working_merge_and_redirect_indices)
            elif j == 0:
                secondary_merge_and_redirect_indices = None
        primary_merge_and_redirect_indices = (primary_merge_and_redirect_indices
                                              if len(primary_merge_and_redirect_indices) != 0
                                              else None)
        working_merge_and_redirect_indices_collection = \
          MergeAndRedirectIndicesCollection(primary_merge_and_redirect_indices,
                                            secondary_merge_and_redirect_indices)
        merge_and_redirect_indices_collections.append(working_merge_and_redirect_indices_collection)
    return merge_and_redirect_indices_collections

class CatchmentTrees:

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

class CatchmentNode:

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

def update_river_directions(upstream_catchment_center,
                            downstream_catchment_entry_point,
                            river_directions,
                            corrected_river_directions,
                            rdirs_jump_next_cell_indices):
    #Use code 10 to indicate jump
    corrected_river_directions.get_data()[tuple(upstream_catchment_center)] = \
        river_directions.get_data()[tuple(upstream_catchment_center)]
    rdirs_jump_next_cell_indices[0].get_data()[tuple(upstream_catchment_center)] = \
        downstream_catchment_entry_point[0]
    rdirs_jump_next_cell_indices[1].get_data()[tuple(upstream_catchment_center)] = \
        downstream_catchment_entry_point[1]


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
                                          corrected_river_directions_filepath=None,
                                          corrected_river_directions_fieldname=None,
                                          rdirs_jump_next_cell_indices_filepath=None,
                                          rdirs_jump_next_cell_indices_fieldname=None,
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
    flood_merge_and_redirect_indices_index = \
        iodriver.advanced_field_loader(lake_parameters_filepath,
                                       field_type='Generic',
                                       fieldname=
                                       "flood_merge_and_redirect_indices_index")
    with Dataset(lake_parameters_filepath,mode='r',format='NETCDF4') as dataset:
        merges_and_redirects_array = \
            np.array(dataset.variables["flood_merges_and_redirects"][:,:,:])
    merges_and_redirects = \
        create_merge_indices_collections_from_array(merges_and_redirects_array)
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
                                       flood_merge_and_redirect_indices_index,
                                       merges_and_redirects,river_directions,scale_factor,
                                       cumulative_flow=cumulative_flow,
                                       correct_cumulative_flow=(True if cumulative_flow_filepath
                                                                is not None else False),
                                       corrected_river_directions = corrected_river_directions_filepath,
                                       rdirs_jump_next_cell_indices = rdirs_jump_next_cell_indices_filepath,
                                       correct_rdirs=(True if corrected_river_directions_filepath
                                                      is not None else False))
    iodriver.advanced_field_writer(connected_coarse_catchments_out_filename,
                                   field=catchments,
                                   fieldname=connected_coarse_catchments_out_fieldname)
    if cumulative_flow_filepath is not None:
        iodriver.advanced_field_writer(connected_cumulative_flow_out_filepath,
                                       field=corrected_cumulative_flow,
                                       fieldname=connected_cumulative_flow_out_fieldname)
    if corrected_river_directions_filepath is not None:
        iodriver.advanced_field_writer(corrected_river_directions_filepath,
                                       field=corrected_river_directions,
                                       fieldname=corrected_river_directions_fieldname)
    if rdirs_jump_next_cell_indices_filepath is not None:
        iodriver.advanced_field_writer(rdirs_jump_next_cell_indices_filepath,
                                       field=rdirs_jump_next_cell_indices[0],
                                       fieldname=rdirs_jump_next_cell_indices_fieldname+"lat")
        iodriver.advanced_field_writer(rdirs_jump_next_cell_indices_filepath,
                                       field=rdirs_jump_next_cell_indices[1],
                                       fieldname=rdirs_jump_next_cell_indices_fieldname+"lon")


#Remember - Tuples trigger basic indexing, lists don't
def connect_coarse_lake_catchments(coarse_catchments,lake_centers,basin_catchment_numbers,
                                   flood_next_cell_index_lat,flood_next_cell_index_lon,
                                   flood_merge_and_redirect_indices_index,
                                   merges_and_redirects,river_directions,scale_factor = 3,
                                   correct_cumulative_flow=False,
                                   cumulative_flow=None,
                                   correct_rdirs=False,
                                   corrected_river_directions=None,
                                   rdirs_jump_next_cell_indices=None):
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
    merge_coords_list = \
        np.argwhere(flood_merge_and_redirect_indices_index.get_data() != -1).tolist()

    secondary_merge_locations = field.makeEmptyField(field_type='Generic',dtype=np.bool_,
                                                     grid_type=lake_centers.get_grid())
    secondary_merge_locations.set_all(False)
    for merge_coords in merge_coords_list:
        working_index = flood_merge_and_redirect_indices_index.get_data()[tuple(merge_coords)]
        if merges_and_redirects[working_index].secondary_merge:
            secondary_merge_locations.data[tuple(merge_coords)] = True
    for lake_center_coords in lake_centers_list:
        basin_number = basin_catchment_numbers.get_data()[tuple(lake_center_coords)]
        while True:
            secondary_merge_coords = np.argwhere(np.logical_and(secondary_merge_locations.get_data(),
                                                                basin_catchment_numbers.get_data() ==
                                                                basin_number))[0,:].tolist()
            working_secondary_merge_index = flood_merge_and_redirect_indices_index.\
                                            get_data()[tuple(secondary_merge_coords)]
            working_secondary_merge = merges_and_redirects[working_secondary_merge_index].\
                                      secondary_merge_and_redirect_indices
            basin_number = basin_catchment_numbers.get_data()[working_secondary_merge.merge_target_lat_index,
                                                              working_secondary_merge.merge_target_lon_index]
            if basin_number == 0:
                if working_secondary_merge.local_redirect:
                    basin_number = \
                        basin_catchment_numbers.\
                        get_data()[working_secondary_merge.redirect_lat_index,
                                   working_secondary_merge.redirect_lon_index]
                else:
                    overflow_catchment = \
                        coarse_catchments.get_data()[working_secondary_merge.redirect_lat_index,
                                                     working_secondary_merge.redirect_lon_index]
                    overflow_coords = (working_secondary_merge.redirect_lat_index,
                                       working_secondary_merge.redirect_lon_index)
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
    coarse_catchments_field = field.Field(cc.renumber_catchments_by_size(coarse_catchments.get_data()),type="Generic",
                                          grid=coarse_catchments.get_grid())
    if correct_cumulative_flow or correct_rdirs:
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
                    if correct_cumulative_flow:
                        update_cumulative_flow(upstream_catchment_center,
                                               (sink_point_cumulative_flow_redirect_lat.get_data()[upstream_catchment_center],
                                                sink_point_cumulative_flow_redirect_lon.get_data()[upstream_catchment_center]),
                                               cumulative_flow,river_directions)
                    if correct_rdirs:
                        update_river_directions(upstream_catchment_center,
                                                (sink_point_cumulative_flow_redirect_lat.get_data()[upstream_catchment_center],
                                                 sink_point_cumulative_flow_redirect_lon.get_data()[upstream_catchment_center]),
                                                 river_directions,
                                                 corrected_river_directions,
                                                 rdirs_jump_next_cell_indices)
        if correct_cumulative_flow and correct_rdirs:
            return coarse_catchments_field,cumulative_flow,corrected_rdirs,rdirs_jump_next_cell_indices
        elif correct_cumulative_flow:
            return coarse_catchments_field,cumulative_flow
        else:
            return coarse_catchments_field,corrected_rdirs,rdirs_jump_next_cell_indices
    return coarse_catchments_field
