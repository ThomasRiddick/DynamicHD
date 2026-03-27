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
import follow_streams_wrapper
from Dynamic_HD_Scripts.tools import compute_catchments as cc
from netCDF4 import Dataset

class Redirect:

    def __init__(self,use_local_redirect,
                 local_redirect_target_lake_number,
                 non_local_redirect_target):
        self.use_local_redirect = use_local_redirect
        self.local_redirect_target_lake_number = local_redirect_target_lake_number
        self.non_local_redirect_target = non_local_redirect_target

    def __eq__(self,rhs):
        return (self.use_local_redirect == rhs.use_local_redirect and
                self.local_redirect_target_lake_number ==
                rhs.local_redirect_target_lake_number and
                self.non_local_redirect_target ==
                rhs.non_local_redirect_target)

    def __str__(self):
        return (f"{self.use_local_redirect} "
                f"{self.local_redirect_target_lake_number} "
                f"{self.non_local_redirect_target}")

# Lake here is equivalent to LakeParameters in the Julia version of the Lake
# Model
class Lake:

    def __init__(self,lake_number,primary_lake,
                 secondary_lakes,center_coords,
                 filling_order,outflow_points,
                 lake_lower_boundary_height,
                 filled_lake_area,scale_factor):
        center_cell_coarse_coords = \
            tuple([1 + (coord-1)//scale_factor for coord in center_coords])
        is_primary = (primary_lake == -1)
        if is_primary and len(outflow_points) > 1:
            raise RuntimeError('Primary lake has more'
                               'than one outflow point')
        is_leaf = (len(secondary_lakes) == 0)
        self.center_coords = center_coords
        self.center_cell_coarse_coords = center_cell_coarse_coords
        self.lake_number = lake_number
        self.is_primary = is_primary
        self.is_leaf = is_leaf
        self.primary_lake = primary_lake
        self.secondary_lakes = secondary_lakes
        self.filling_order = filling_order
        self.outflow_points = outflow_points
        self.lake_lower_boundary_height = lake_lower_boundary_height
        self.filled_lake_area = filled_lake_area

    def find_top_level_primary_lake_number(self,lakes):
        if self.is_primary:
          return self.lake_number
        else:
          primary_lake = lakes[self.primary_lake - 1]
          return primary_lake.find_top_level_primary_lake_number(lakes)

    def __str__(self):
        return (f"\n{self.center_coords}\n"
                f"{self.center_cell_coarse_coords}\n"
                f"{self.lake_number}\n"
                f"{self.is_primary}\n"
                f"{self.is_leaf}\n"
                f"{self.primary_lake}\n"
                f"{self.secondary_lakes}\n"
                f"{self.filling_order}\n"
                f"{self.outflow_points}\n"
                f"{self.lake_lower_boundary_height}\n"
                f"{self.filled_lake_area}")

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

class ArrayDecoder:

    def __init__(self,array):
        self.array = array
        self.current_index = 1
        self.object_count = 0
        self.object_start_index = 0
        self.expected_total_objects = array[0]
        self.expected_object_length = 0

    def start_next_object(self):
        self.expected_object_length = self.array[self.current_index]
        self.current_index += 1
        self.object_count += 1
        #Expected object length excludes the first entry (i.e. the length itself)
        self.object_start_index = self.current_index

    def finish_object(self):
        if (self.expected_object_length !=
            self.current_index - self.object_start_index):
            raise RuntimeError("Object read incorrectly - length"
                               " doesn't match expectation")

    def finish_array(self):
        if self.object_count != self.expected_total_objects:
            raise RuntimeError("Array read incorrectly - number of object"
                               " doesn't match expectation")

        if len(self.array) != self.current_index:
            raise RuntimeError("Array read incorrectly - length doesn't"
                               " match expectation")

    def read_float(self):
        val = self.array[self.current_index]
        self.current_index += 1
        return val

    def read_integer(self):
        return int(self.read_float())

    def read_bool(self):
        return (self.read_float() != 0)

    def read_coords(self,single_index=False):
        if single_index:
            return (self.read_integer(),)
        else:
            y = int(self.array[self.current_index])
            x  = int(self.array[self.current_index+1])
            self.current_index += 2
            return (y,x)

    def read_field(self,integer_field=False):
      field_length = int(self.array[self.current_index])
      self.current_index += 1
      field = self.array[int(self.current_index):int(self.current_index)+field_length]
      self.current_index += field_length
      if integer_field:
        return field
      else:
        return field

    def read_outflow_points_dict(self,single_index=False):
        length = int(self.array[self.current_index])
        self.current_index += 1
        entry_length =  3 if single_index else 4
        offset = 1 if single_index else 2
        outflow_points = {}
        for _ in range(length):
            entry = \
                self.array[self.current_index:self.current_index+entry_length]
            self.current_index += entry_length
            lake_number = int(entry[0])
            is_local = (entry[1+offset] != 0)
            if not is_local:
                coords_array =  [int(entry[1])] if single_index \
                                 else [int(x) for x in entry[1:3]]
                coords = tuple(coords_array)
            else:
                coords = (-1,-1) if not single_index else (-1,)
            redirect = Redirect(is_local,lake_number,coords)
            outflow_points[lake_number] = redirect
        return outflow_points

    def read_filling_order(self,single_index=False):
        length = int(self.array[self.current_index])
        self.current_index += 1
        entry_length = 4 if single_index else 5
        for _ in range(length):
            self.current_index += entry_length
        #Just return placeholder as filling order isn't currently required
        return []

def get_lake_parameters_from_array(array,scale_factor=3,
                                   single_index=False):
    decoder = ArrayDecoder(array)
    lake_parameters = []
    for _ in range(int(decoder.expected_total_objects)):
        decoder.start_next_object()
        lake_number = decoder.read_integer()
        primary_lake = decoder.read_integer()
        secondary_lakes = decoder.read_field(integer_field=True)
        center_coords = decoder.read_coords(single_index=single_index)
        filling_order = \
          decoder.read_filling_order(single_index=single_index)

        outflow_points = \
          decoder.read_outflow_points_dict(single_index=single_index)
        lake_lower_boundary_height = decoder.read_float()
        filled_lake_area = decoder.read_float()
        decoder.finish_object()
        lake_parameters.append(Lake(lake_number,
                                    primary_lake,
                                    secondary_lakes,
                                    center_coords,
                                    filling_order,
                                    outflow_points,
                                    lake_lower_boundary_height,
                                    filled_lake_area,
                                    scale_factor))
    decoder.finish_array()
    return lake_parameters

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

def mark_jumps(upstream_catchment_center,
               downstream_catchment_entry_point,
               rdirs_jump_next_cell_indices):
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
                                          rdirs_jump_next_cell_indices_filepath=None,
                                          rdirs_jump_next_cell_indices_fieldname=None,
                                          coarse_lake_outflows_fieldname=None,
                                          scale_factor = 3):
    coarse_catchments = iodriver.advanced_field_loader(coarse_catchments_filepath,
                                                       field_type='Generic',
                                                       fieldname=\
                                                       coarse_catchments_fieldname)
    basin_catchment_numbers = iodriver.advanced_field_loader(basin_catchment_numbers_filepath,
                                                   field_type='Generic',
                                                   fieldname=\
                                                   basin_catchment_numbers_fieldname)
    river_directions = iodriver.advanced_field_loader(river_directions_filepath,
                                                      field_type='Generic',
                                                      fieldname=\
                                                      river_directions_fieldname)
    if cumulative_flow_filepath is not None:
        cumulative_flow = iodriver.advanced_field_loader(cumulative_flow_filepath,
                                                         field_type='Generic',
                                                         fieldname=\
                                                         cumulative_flow_fieldname)
    if rdirs_jump_next_cell_indices_filepath is not None:
        rdirs_jump_next_cell_indices = (field.makeField(np.full(cumulative_flow.get_data().shape,-1),
                                                        'Generic',grid_type=cumulative_flow.get_grid()),
                                        field.makeField(np.full(cumulative_flow.get_data().shape,-1),
                                                        'Generic',grid_type=cumulative_flow.get_grid()))
    else:
        rdirs_jump_next_cell_indices = None
    lakes = None
    results = \
        connect_coarse_lake_catchments(lakes,
                                       coarse_catchments,
                                       basin_catchment_numbers,
                                       river_directions=river_directions,
                                       scale_factor=scale_factor,
                                       cumulative_flow=cumulative_flow,
                                       correct_cumulative_flow=(True if cumulative_flow_filepath
                                                                is not None else False),
                                       mark_rdir_jumps=(True if rdirs_jump_next_cell_indices_filepath
                                                        is not None else False),
                                       rdirs_jump_next_cell_indices=rdirs_jump_next_cell_indices)
    if rdirs_jump_next_cell_indices_filepath is not None or cumulative_flow_filepath is not None:
        if (rdirs_jump_next_cell_indices_filepath is not None and
            cumulative_flow_filepath is not None):
            catchments, corrected_cumulative_flow,\
            rdirs_jump_next_cell_indices,coarse_lake_outflows = results
        elif cumulative_flow_filepath is not None:
            catchments, corrected_cumulative_flow = results
        else:
            catchments, rdirs_jump_next_cell_indices,coarse_lake_outflows = results
    else:
        catchments = results
    iodriver.advanced_field_writer(connected_coarse_catchments_out_filename,
                                   field=catchments,
                                   fieldname=connected_coarse_catchments_out_fieldname)
    if cumulative_flow_filepath is not None:
        iodriver.advanced_field_writer(connected_cumulative_flow_out_filepath,
                                       field=corrected_cumulative_flow,
                                       fieldname=connected_cumulative_flow_out_fieldname)
    if rdirs_jump_next_cell_indices_filepath is not None:
        iodriver.advanced_field_writer(rdirs_jump_next_cell_indices_filepath,
                                       field=[*rdirs_jump_next_cell_indices,
                                              coarse_lake_outflows],
                                       fieldname=[rdirs_jump_next_cell_indices_fieldname+"lat",
                                                  rdirs_jump_next_cell_indices_fieldname+"lon",
                                                  coarse_lake_outflows_fieldname])

#Remember - Tuples trigger basic indexing, lists don't
def connect_coarse_lake_catchments(lakes,
                                   coarse_catchments,
                                   basin_catchment_numbers,
                                   river_directions=None,
                                   scale_factor=3,
                                   correct_cumulative_flow=False,
                                   cumulative_flow=None,
                                   mark_rdir_jumps=False,
                                   rdirs_jump_next_cell_indices=None):
    if correct_cumulative_flow:
        if cumulative_flow is None or river_directions is None:
            raise RuntimeError("Required input files for cumulative flow correction not provided")
        old_coarse_catchments = coarse_catchments.copy()
    overflow_catchments = field.makeEmptyField(field_type='Generic',dtype=np.int64,
                                               grid_type=
                                               basin_catchment_numbers.get_grid())
    overflow_coords_lats = field.makeEmptyField(field_type='Generic',dtype=np.int64,
                                                grid_type=
                                                basin_catchment_numbers.get_grid())
    overflow_coords_lons = field.makeEmptyField(field_type='Generic',dtype=np.int64,
                                                grid_type=
                                                basin_catchment_numbers.get_grid())
    if mark_rdir_jumps:
        coarse_lake_outflows = field.makeEmptyField(field_type='Generic',dtype=np.bool_,
                                                    grid_type=coarse_catchments.get_grid())
        coarse_lake_outflows.set_all(False)
    for lake in lakes:
        if not lake.is_leaf:
            continue
        lake_center_coords = lake.center_coords
        basin_number = basin_catchment_numbers.get_data()\
                        [tuple([coord - 1 for coord in lake.center_coords])]
        working_lake = lake
        while True:
            primary_lake = \
                lakes[working_lake.find_top_level_primary_lake_number(lakes) - 1]
            #Primary lake will only have one outflow
            key = list(primary_lake.outflow_points)[0]
            if primary_lake.outflow_points[key].use_local_redirect:
                working_lake = lakes[primary_lake.outflow_points[key].\
                                     local_redirect_target_lake_number  - 1]
                if working_lake.lake_number != primary_lake.outflow_points[key].\
                                               local_redirect_target_lake_number:
                    raise RuntimeError('Lakes not ordered as expected')
            else:
                if mark_rdir_jumps:
                    coarse_lake_outflows.get_data()[tuple([ coord - 1 for coord in \
                                                    primary_lake.outflow_points[key].\
                                                    non_local_redirect_target])] = True
                overflow_catchment = \
                    coarse_catchments.get_data()[tuple([coord - 1 for coord in \
                                                 primary_lake.outflow_points[key].\
                                                 non_local_redirect_target])]
                overflow_coords = primary_lake.outflow_points[key].\
                                  non_local_redirect_target
                break
        overflow_catchments.get_data()[tuple([coord - 1 for coord in \
                                             lake_center_coords])] = overflow_catchment
        #Include array offset
        overflow_coords_lats.get_data()[tuple([coord - 1 for coord in \
                                              lake_center_coords])] = \
                                        overflow_coords[0] - 1
        overflow_coords_lons.get_data()[tuple([coord - 1 for coord in \
                                              lake_center_coords])] = \
                                        overflow_coords[1] - 1
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
        overflow_catch_fine_cells_in_coarse_cell = \
            overflow_catchments.get_data()[sink_point[0]*scale_factor:
                                           (sink_point[0]+1)*scale_factor,
                                           sink_point[1]*scale_factor:
                                           (sink_point[1]+1)*scale_factor]
        overflow_coords_lat_fine_cells_in_coarse_cell = \
            overflow_coords_lats.get_data()[sink_point[0]*scale_factor:
                                            (sink_point[0]+1)*scale_factor,
                                            sink_point[1]*scale_factor:
                                            (sink_point[1]+1)*scale_factor]
        overflow_coords_lon_fine_cells_in_coarse_cell = \
            overflow_coords_lons.get_data()[sink_point[0]*scale_factor:
                                            (sink_point[0]+1)*scale_factor,
                                            sink_point[1]*scale_factor:
                                            (sink_point[1]+1)*scale_factor]
        overflow_catchment_list = \
            overflow_catch_fine_cells_in_coarse_cell[overflow_catch_fine_cells_in_coarse_cell != 0].tolist()
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
    coarse_catchments_field = \
        field.Field(cc.renumber_catchments_by_size(coarse_catchments.get_data()),
                    type="Generic",grid=coarse_catchments.get_grid())
    if correct_cumulative_flow or mark_rdir_jumps:
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
                                               (sink_point_cumulative_flow_redirect_lat.get_data()\
                                                [upstream_catchment_center],
                                                sink_point_cumulative_flow_redirect_lon.get_data()\
                                                [upstream_catchment_center]),
                                               cumulative_flow,river_directions)
                    if mark_rdir_jumps:
                        mark_jumps(upstream_catchment_center,
                                   (sink_point_cumulative_flow_redirect_lat.\
                                    get_data()[upstream_catchment_center],
                                    sink_point_cumulative_flow_redirect_lon.\
                                    get_data()[upstream_catchment_center]),
                                    rdirs_jump_next_cell_indices)
        if correct_cumulative_flow and mark_rdir_jumps:
            return coarse_catchments_field,cumulative_flow,rdirs_jump_next_cell_indices,coarse_lake_outflows
        elif correct_cumulative_flow:
            return coarse_catchments_field,cumulative_flow
        else:
            return coarse_catchments_field,rdirs_jump_next_cell_indices,coarse_lake_outflows
    return coarse_catchments_field
