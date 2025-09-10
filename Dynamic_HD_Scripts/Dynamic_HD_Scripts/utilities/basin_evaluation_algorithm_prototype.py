import numpy as np
from copy import copy
from heapq import heappush,heappop
from enum import auto, Enum
from functools import total_ordering
#Note - copied library files from the latest version of master
#hence they don't match this version of the code (but can
#still be used here)
from fill_sinks_wrapper import fill_sinks_cpp_func

class StoreToArray:

  def __init__(self):
    self.objects = []
    self.working_object = None

  def add_object(self):
    self.working_object = [0.0]
    self.objects.append(self.working_object)

  def complete_object(self):
    self.working_object[0] = float(len(self.working_object) - 1)

  def add_number(self,number_in):
    self.working_object.append(float(number_in))

  def add_coords(self,coords_in):
    if type(coords_in) is int:
        self.working_object.append(float(coords_in))
    elif len(coords_in) == 2:
        self.working_object.append(float(coords_in[0]))
        self.working_object.append(float(coords_in[1]))

  def add_field(self,field_in):
    self.working_object.append(float(len(field_in)))
    self.working_object.extend([float(val) for val in field_in])

  def add_outflow_points_dict(self,outflow_points_in,
                              single_index=False,
                              array_offset=0):
    dict_array = []
    for key,val in outflow_points_in.items():
        entry_as_array = [float((key+array_offset) if key >= 0 else key)]
        if val[0] is None:
            if single_index:
                entry_as_array.append(-1.0)
            else:
                entry_as_array.extend([-1.0,-1.0])
        elif val[0] is int:
            if not single_index:
                raise RuntimeError("Wrong numbers of coordinates in "
                                   "outflows points")
            entry_as_array.append(float(val[0]+array_offset))
        else:
            if single_index:
                raise RuntimeError("Wrong numbers of coordinates in "
                                   "outflows points")
            entry_as_array.extend([float(val[0][0]+array_offset),
                                   float(val[0][1]+array_offset)])
        entry_as_array.append(float(val[1]))
        dict_array.extend(entry_as_array)
    self.working_object.append(float(len(outflow_points_in.keys())))
    self.working_object.extend(dict_array)

  def add_filling_order(self,filling_order_in,array_offset=0):
    filling_order_array = []
    for entry in filling_order_in:
        if type(entry[0]) is int:
            filling_order_array.append(float(entry[0]+array_offset))
        else:
            filling_order_array.extend([float(entry[0][0]+array_offset),
                                        float(entry[0][1]+array_offset)])
        filling_order_array.append(float(entry[1].value))
        filling_order_array.append(float(entry[2]))
        filling_order_array.append(float(entry[3]))
    self.working_object.append(float(len(filling_order_in)))
    self.working_object.extend(filling_order_array)

  def complete_array(self):
    array = [float(len(self.objects))]
    for obj in self.objects:
      array.extend(obj)
    return array

class Grid:
    pass

class LatLonGrid(Grid):

    def __init__(self,shape):
        self.shape = shape

    def get_neighbors_coords(self,coords_in):
        y = coords_in[0]
        x = coords_in[1]
        neighbors = []
        for i in range(-1,2):
            for j in range(-1,2):
                if ((i == 0 and j == 0) or
                     y+j < 0 or
                     y+j >= self.shape[0]):
                    pass
                elif x+i < 0:
                    neighbors.append((y+j,self.shape[1]+i))
                elif (x+i >= self.shape[1]):
                    neighbors.append((y+j,-1+i))
                else:
                    neighbors.append((y+j,x+i))
        return neighbors

    def convert_fine_coords(self,coords_in,fine_grid_in):
        y_scale = fine_grid_in.shape[0]/self.shape[0]
        x_scale = fine_grid_in.shape[1]/self.shape[1]
        return (int(coords_in[0]//y_scale),int(coords_in[1]//x_scale))

    def calculate_downstream_coords_from_dir_based_rdir(self,coords_in,rdir):
        y = coords_in[0]
        x = coords_in[1]
        if rdir == 5 or rdir <= 0:
            return coords_in
        if rdir == 7 or rdir == 8 or rdir == 9:
            j = -1
        elif rdir == 4 or rdir == 6:
            j = 0
        elif rdir == 1 or rdir == 2 or rdir == 3:
            j = 1
        if rdir == 1 or rdir == 4 or rdir == 7:
            i = -1
        elif rdir == 2 or rdir == 8:
            i = 0
        elif rdir == 3 or rdir == 6 or rdir == 9:
            i = 1
        if(y+j < 0 or
           y+j >= self.shape[0]):
            return coords_in
        elif x+i < 0:
            return (y+j,self.shape[1]+i)
        elif x+i >= self.shape[1]:
            return (y+j,-1+i)
        else:
            return (y+j,x+i)

class HeightType(Enum):
    flood_height = auto()
    connection_height = auto()

class Lake:

    def __init__(self,
                 lake_number,
                 center_coords,
                 lake_lower_boundary_height,
                 primary_lake=None,
                 secondary_lakes=None):
        #Keep references to lakes as number not objects
        self.center_coords = center_coords
        self.lake_number = lake_number
        self.primary_lake = primary_lake
        self.secondary_lakes  = secondary_lakes
        self.outflow_points = {}
        self.filling_order = []
        self.lake_lower_boundary_height = lake_lower_boundary_height
        self.filled_lake_area = None
        #Working variables - don't need to exported
        self.center_cell_volume_threshold = 0.0
        self.lake_area = 0.0
        self.spill_points = {}
        self.potential_exit_points = []

    def set_primary_lake(self,primary_lake):
        self.primary_lake = primary_lake

    def set_potential_exit_points(self,potential_exit_points):
        self.potential_exit_points = potential_exit_points

    def set_filled_lake_area(self):
        self.filled_lake_area = self.lake_area

    def __repr__(self):
        return (f"center_coords={self.center_coords}\n"
                f"lake_number={self.lake_number}\n"
                f"primary_lake={self.primary_lake}\n"
                f"secondary_lakes={self.secondary_lakes}\n"
                f"outflow_points={self.outflow_points}\n"
                f"filling_order={self.filling_order}\n"
                f"center_cell_volume_threshold={self.center_cell_volume_threshold}\n"
                f"lake_area={self.lake_area}\n"
                f"spill_points={self.spill_points}\n"
                f"potential_exit_points{self.potential_exit_points}\n")


class Cell:

    def get_cell_coords(self):
        return self.cell_coords

@total_ordering
class BasinCell(Cell):

    def __init__(self,cell_height,cell_height_type,cell_coords):
        self.cell_coords = cell_coords
        self.cell_height_type = cell_height_type
        self.cell_height = cell_height

    def __gt__(self,other):
        return self.cell_height > other.cell_height

    def __eq__(self,other):
        return self.cell_height == other.cell_height

    def get_height(self):
        return self.cell_height

    def get_height_type(self):
        return self.cell_height_type

class LandSeaCell(Cell):

    def __init__(self,cell_coords):
        self.cell_coords = cell_coords

class DisjointSet:

    def __init__(self,label_in):
        self.label = label_in
        self.size = 1
        self.root = self
        self.nodes = []

    def get_root(self):
        return self.root

    def set_root(self,x):
        self.root = x

    def add_node(self,x):
        self.nodes.append(x)

    def add_nodes(self,extra_nodes):
        self.nodes.extend(extra_nodes)

    def get_nodes(self):
        return self.nodes

    def increase_size(self,size_increment_in):
        self.size += size_increment_in

    def get_size(self):
        return self.size

    def get_label(self):
        return self.label

    def get_set_element_labels(self):
        labels = [self.label]
        for node in self.nodes:
            labels.append(node.get_label())
        return labels

class DisjointSetForest:

    def __init__(self):
        self.sets = []

    def find_root(self,x):
      root = x
      while root.get_root() != root:
        working_ptr = root.get_root()
        working_ptr.set_root(root.get_root().get_root())
        root = working_ptr
      return root

    def link(self,x,y):
      root_x = self.find_root(x)
      root_y = self.find_root(y)
      if (root_x == root_y):
        return False
      root_y.set_root(root_x)
      root_x.increase_size(root_y.get_size())
      root_x.add_node(root_y)
      root_x.add_nodes(root_y.get_nodes())
      return True

    def add_set(self,label_in):
      if self.get_set(label_in):
        return
      new_set = DisjointSet(label_in)
      self.sets.append(new_set)

    def get_set(self,label_in):
      for node in self.sets:
        if node.get_label() == label_in:
            return node
      return None

    def make_new_link(self,label_x,label_y):
      x = self.get_set(label_x)
      y = self.get_set(label_y)
      return self.link(x,y)

    def find_root_from_label(self,label_in):
      x = self.get_set(label_in)
      root_x = self.find_root(x)
      return root_x.get_label()

    def for_elements_in_set(self,root,func):
      if root.get_root() != root:
        raise RuntimeError("Given set is not label of a root set")
      func(root.get_label())
      for node in root.get_nodes():
        func(node.get_label())

    def for_elements_in_set(self,root_label,func):
      root = self.get_set(root_label)
      self.for_elements_in_set(root,func)

    def check_subset_has_elements(self,label_of_element,element_labels):
        check_passed = True
        root = self.find_root(self.get_set(label_of_element))
        if len(root.get_nodes()) + 1 != len(element_labels):
            return False
        if element_labels[0] != root.get_label():
            return False
        for i,node in enumerate(root.get_nodes()):
            i += 1
            if element_labels[i] != node.get_label():
                check_passed = False
        return check_passed

class SimpleSearch:

    def __init__(self,grid):
        self.search_completed_cells = np.full(grid.shape,False)
        self.grid = grid

    def __call__(self,
                 target_found_func,
                 ignore_nbr_func,
                 start_point):
        self.search_q = []
        self.search_q.append(LandSeaCell(start_point))
        self.search_completed_cells[:,:] = False
        self.ignore_nbr_func = ignore_nbr_func
        while len(self.search_q) > 0:
            search_cell = self.search_q[0]
            self.search_q.pop(0)
            self.search_coords = search_cell.get_cell_coords()
            if target_found_func(self.search_coords):
                return self.search_coords
            self.search_process_neighbors()
        raise RuntimeError("Search Failed")

    def search_process_neighbors(self):
        search_neighbors_coords = \
            self.grid.get_neighbors_coords(self.search_coords)
        while len(search_neighbors_coords) > 0:
            search_nbr_coords = search_neighbors_coords.pop()
            if not (self.search_completed_cells[search_nbr_coords] or
                    self.ignore_nbr_func(search_nbr_coords)):
                self.search_q.append(LandSeaCell(search_nbr_coords))
                self.search_completed_cells[search_nbr_coords] = True

class BasinEvaluationAlgorithm:

    def __init__(self,
                 minima,
                 raw_orography,
                 corrected_orography,
                 cell_areas,
                 prior_fine_catchment_nums,
                 coarse_catchment_nums,
                 catchments_from_sink_filling):
        self.minima = minima
        self.raw_orography = raw_orography
        self.corrected_orography = corrected_orography
        self.cell_areas = cell_areas
        self.prior_fine_catchment_nums = prior_fine_catchment_nums
        self.sink_points = [None for _ in range(np.max(self.prior_fine_catchment_nums))]
        self.coarse_catchment_nums = coarse_catchment_nums
        self.catchments_from_sink_filling = catchments_from_sink_filling
        self.lakes = []
        self.lake_q = []
        self.lake_connections = None
        self.q = None
        self.level_q = None
        self.null_lake_number = -1
        self.lake_numbers = np.full(self.raw_orography.shape,self.null_lake_number,
                                    dtype=np.int32)
        self.completed_cells = np.full(self.raw_orography.shape,False,dtype=np.bool)
        self.cells_in_lake = np.full(self.raw_orography.shape,False,dtype=np.bool)
        self.level_completed_cells = np.full(self.raw_orography.shape,False,dtype=np.bool)
        self.search_alg = SimpleSearch(self.grid)
        self.coarse_search_alg = SimpleSearch(self.coarse_grid)

    def evaluate_basins(self):
        self.lakes = []
        self.lake_q = []
        self.lake_connections = DisjointSetForest()
        self.lake_numbers[:,:] = self.null_lake_number
        minima_q = [(int(element[0]),int(element[1])) for element in list(np.argwhere(self.minima))]
        merging_lakes = []
        while len(minima_q) > 0:
            minimum = minima_q[-1]
            minima_q.pop()
            lake_number = len(self.lakes)
            lake = Lake(lake_number,minimum,self.raw_orography[minimum])
            self.lakes.append(lake)
            self.lake_q.append(lake)
            self.lake_connections.add_set(lake_number)
        while True:
            while len(self.lake_q) > 0:
                lake = self.lake_q[0]
                lake_number = lake.lake_number
                self.lake_q.pop(0)
                self.initialize_basin(lake)
                while True:
                    if len(self.q) <= 0:
                        raise RuntimeError("Basin outflow not found")
                    center_cell = heappop(self.q)
                    #Call the newly loaded coordinates and height for the center cell 'new'
                    #until after making the test for merges then relabel. Center cell height/coords
                    #without the 'new' moniker refers to the previous center cell; previous center cell
                    #height/coords the previous previous center cell
                    self.new_center_coords = center_cell.get_cell_coords()
                    self.new_center_cell_height_type = center_cell.get_height_type()
                    self.new_center_cell_height = center_cell.get_height()
                    #Exit to basin or level area found
                    if (self.new_center_cell_height <= self.center_cell_height and
                        self.searched_level_height != self.center_cell_height):
                            outflow_lake_numbers,potential_exit_points = \
                                self.search_for_outflows_on_level(lake_number)
                            if len(outflow_lake_numbers) > 0:
                                #Exit(s) found
                                for other_lake_number in outflow_lake_numbers:
                                    if other_lake_number != -1:
                                        if (self.lake_connections.find_root_from_label(lake_number) !=
                                            self.lake_connections.find_root_from_label(other_lake_number)):
                                            self.lake_connections.make_new_link(lake_number,other_lake_number)
                                            merging_lakes.append(lake_number)
                                lake.set_potential_exit_points(potential_exit_points)
                                lake.set_filled_lake_area()
                                self.raw_orography[
                                    np.logical_and(self.cells_in_lake,
                                                   self.raw_orography < self.center_cell_height)] = \
                                        self.center_cell_height
                                self.corrected_orography[
                                    np.logical_and(self.cells_in_lake,
                                                   self.corrected_orography < self.center_cell_height)] = \
                                        self.center_cell_height
                                break
                            else:
                                #Don't rescan level later
                                self.searched_level_height = self.center_cell_height
                    #Process neighbors of new center coords
                    self.process_neighbors()
                    self.previous_cell_coords = self.center_coords
                    self.previous_cell_height = self.center_cell_height
                    self.previous_cell_height_type = self.center_cell_height_type
                    self.center_cell_height_type = self.new_center_cell_height_type
                    self.center_cell_height = self.new_center_cell_height
                    self.center_coords = self.new_center_coords
                    self.process_center_cell(lake)
            if len(merging_lakes) > 0:
                unique_lake_groups = {g for g in [self.lake_connections.find_root_from_label(l)
                                                  for l in merging_lakes]}
                for lake_group in unique_lake_groups:
                    sublakes_in_lake = \
                        {sublake for sublake in
                         self.lake_connections.get_set(lake_group).get_set_element_labels()
                         if self.lakes[sublake].primary_lake is None }
                    new_lake_number = len(self.lakes)
                    new_lake_center_coords = self.lakes[list(sublakes_in_lake)[0]].center_coords
                    new_lake = Lake(new_lake_number,
                                    new_lake_center_coords,
                                    self.raw_orography[new_lake_center_coords],
                                    primary_lake=None,
                                    secondary_lakes=sublakes_in_lake)
                    self.lake_connections.add_set(new_lake_number)
                    #Note the new lake isn't necessarily the root of the disjointed set
                    self.lake_connections.make_new_link(new_lake_number,list(sublakes_in_lake)[0])
                    self.lakes.append(new_lake)
                    self.lake_q.append(new_lake)
                    for sublake in sublakes_in_lake:
                        for other_sublake in sublakes_in_lake:
                            if sublake != other_sublake:
                                self.lakes[sublake].spill_points[other_sublake] = \
                                    self.search_alg(lambda coords :
                                                    self.lake_numbers[coords] == other_sublake,
                                                    lambda coords :
                                                    float(self.corrected_orography[coords]) !=
                                                    float(self.corrected_orography[
                                                          self.lakes[sublake].center_coords]),
                                                    self.lakes[sublake].center_coords)
                    for sublake in sublakes_in_lake:
                        self.lake_numbers[self.lake_numbers == sublake] = new_lake.lake_number
                        self.lakes[sublake].set_primary_lake(new_lake.lake_number)
                merging_lakes = []
            else:
                break
        self.set_outflows()

    def initialize_basin(self,lake):
        self.q = []
        self.searched_level_height = 0.0
        self.completed_cells[:,:] = False
        self.cells_in_lake[:,:] = False
        lake.center_cell_volume_threshold = 0.0
        self.center_coords = lake.center_coords
        raw_height = float(self.raw_orography[self.center_coords])
        corrected_height = float(self.corrected_orography[self.center_coords])
        if raw_height <= corrected_height:
            self.center_cell_height = raw_height
            self.center_cell_height_type = HeightType.flood_height
        else:
            self.center_cell_height = corrected_height
            self.center_cell_height_type = HeightType.connection_height
        self.previous_cell_coords = self.center_coords
        self.previous_cell_height_type = self.center_cell_height_type
        self.previous_cell_height = self.center_cell_height
        self.catchments_from_sink_filling_catchment_num = \
            self.catchments_from_sink_filling[self.center_coords]
        self.new_center_coords = self.center_coords
        self.new_center_cell_height_type = self.center_cell_height_type
        self.new_center_cell_height = self.center_cell_height
        if self.center_cell_height_type == HeightType.connection_height:
            lake.lake_area = 0.0
        elif self.center_cell_height_type == HeightType.flood_height:
            lake.lake_area = float(self.cell_areas[self.center_coords])
        else:
            raise RuntimeError("Cell type not rcognized")
        self.completed_cells[self.center_coords] = True
        #Make partial iteration
        self.process_neighbors()
        self.center_cell = heappop(self.q)
        self.new_center_coords = self.center_cell.get_cell_coords()
        self.new_center_cell_height_type = self.center_cell.get_height_type()
        self.new_center_cell_height = self.center_cell.get_height()
        self.process_neighbors()
        self.center_cell_height_type = self.new_center_cell_height_type
        self.center_cell_height = self.new_center_cell_height
        self.center_coords = self.new_center_coords
        self.process_center_cell(lake)

    def search_for_outflows_on_level(self,lake_number):
        self.level_completed_cells[:,:] = False
        self.outflow_lake_numbers = []
        self.potential_exit_points = []
        self.level_q = []
        additional_cells_to_return_to_q = []
        if len(self.q) > 0:
            while (len(self.q) > 0 and
                   self.q[0].get_height() <= self.center_cell_height):
                if (self.lake_numbers[self.q[0].get_cell_coords()] == -1 and
                    self.q[0].get_height() == self.center_cell_height):
                    self.level_q.append(heappop(self.q))
                else:
                    additional_cells_to_return_to_q.append(heappop(self.q))
        #put remove items back in q
        for cell in self.level_q:
            heappush(self.q,cell)
        for cell in additional_cells_to_return_to_q:
            heappush(self.q,cell)
        self.level_q.append(BasinCell(self.center_cell_height,
                                      self.center_cell_height_type,
                                      self.center_coords))
        if self.center_cell_height == self.new_center_cell_height:
            self.level_q.append(BasinCell(self.new_center_cell_height,
                                          self.new_center_cell_height_type,
                                          self.new_center_coords))
            if (self.lake_numbers[self.new_center_coords] != -1 and
                self.lake_numbers[self.new_center_coords] != lake_number):
                self.outflow_lake_numbers.append(self.lake_numbers[self.new_center_coords])
                self.potential_exit_points.append(self.new_center_coords)
        while len(self.level_q) > 0:
            level_center_cell = self.level_q.pop()
            self.level_coords = level_center_cell.get_cell_coords()
            self.level_completed_cells[self.level_coords] = True
            self.process_level_neighbors(lake_number)
        return self.outflow_lake_numbers,self.potential_exit_points

    def process_level_neighbors(self,lake_number):
        neighbors_coords = \
            self.grid.get_neighbors_coords(self.level_coords)
        while len(neighbors_coords) > 0:
            nbr_coords = neighbors_coords[-1]
            neighbors_coords.pop(-1)
            if (not self.level_completed_cells[nbr_coords] and
                not self.cells_in_lake[nbr_coords]):
                raw_height = float(self.raw_orography[nbr_coords])
                corrected_height = float(self.corrected_orography[nbr_coords])
                self.level_completed_cells[nbr_coords] = True
                if raw_height <= corrected_height:
                    nbr_height = raw_height
                    nbr_height_type = HeightType.flood_height
                else:
                    nbr_height = corrected_height
                    nbr_height_type = HeightType.connection_height
                if (nbr_height < self.center_cell_height and
                    self.lake_numbers[nbr_coords] != lake_number):
                    self.outflow_lake_numbers.append(-1)
                    self.potential_exit_points.append(nbr_coords)
                elif (nbr_height == self.center_cell_height and
                      self.lake_numbers[nbr_coords] != -1 and
                      self.lake_numbers[nbr_coords] != lake_number):
                    self.outflow_lake_numbers.append(self.lake_numbers[nbr_coords])
                    self.potential_exit_points.append(nbr_coords)
                elif nbr_height == self.center_cell_height:
                    self.level_q.append(BasinCell(nbr_height,nbr_height_type,
                                                  nbr_coords))

    def process_center_cell(self,lake):
        self.cells_in_lake[self.previous_cell_coords] = True
        if (self.lake_numbers[self.previous_cell_coords] == self.null_lake_number):
            self.lake_numbers[self.previous_cell_coords] = lake.lake_number
        lake.center_cell_volume_threshold += \
                lake.lake_area*(self.center_cell_height-self.previous_cell_height)
        if self.previous_cell_height_type == HeightType.connection_height:
            heappush(self.q,BasinCell(float(self.raw_orography[self.previous_cell_coords]),
                                      HeightType.flood_height,self.previous_cell_coords))
            lake.filling_order.append((self.previous_cell_coords,
                                       HeightType.connection_height,lake.center_cell_volume_threshold,
                                       self.center_cell_height))
        elif self.previous_cell_height_type == HeightType.flood_height:
            lake.filling_order.append((self.previous_cell_coords,
                                       HeightType.flood_height,lake.center_cell_volume_threshold,
                                       self.center_cell_height))
        else:
            raise RuntimeError("Cell type not recognized")
        if (self.center_cell_height_type == HeightType.flood_height):
            lake.lake_area += float(self.cell_areas[self.center_coords])
        elif self.center_cell_height_type != HeightType.connection_height:
            raise RuntimeError("Cell type not recognized")

    def process_neighbors(self):
        neighbors_coords = self.grid.get_neighbors_coords(self.new_center_coords)
        while len(neighbors_coords) > 0:
            nbr_coords = neighbors_coords[-1]
            neighbors_coords.pop()
            nbr_catchment = self.catchments_from_sink_filling[nbr_coords]
            in_different_catchment = \
              ( nbr_catchment != self.catchments_from_sink_filling_catchment_num) and \
              ( nbr_catchment != -1)
            if (not self.completed_cells[nbr_coords] and
                not in_different_catchment):
                raw_height = float(self.raw_orography[nbr_coords])
                corrected_height = float(self.corrected_orography[nbr_coords])
                if raw_height <= corrected_height:
                    nbr_height = raw_height
                    nbr_height_type = HeightType.flood_height
                else:
                    nbr_height = corrected_height
                    nbr_height_type = HeightType.connection_height
                heappush(self.q,BasinCell(nbr_height,nbr_height_type,
                                          nbr_coords))
                self.completed_cells[nbr_coords] = True

    def set_outflows(self):
        for lake in self.lakes:
            if lake.primary_lake is None:
                #Arbitrarily choose the first exit point
                first_potential_exit_point = lake.potential_exit_points[0]
                first_potential_exit_point_lake_number = \
                    self.lake_numbers[first_potential_exit_point]
                if first_potential_exit_point_lake_number != -1:
                    #This means the lake is spilling into an
                    #unconnected neighboring lake at a lower level
                    #It is possible though very rare that a will spill
                    #to two unconnected downstreams lakes in the same
                    #overall catchment both at a lower level - in this
                    #case arbitrarily use the first
                    lake.outflow_points[first_potential_exit_point_lake_number] = \
                        (None,True)
                else:
                    first_cell_beyond_rim_coords = first_potential_exit_point
                    lake.outflow_points[-1] = (self.find_non_local_outflow_point(first_cell_beyond_rim_coords),False)
            else:
                for other_lake,spill_point in lake.spill_points.items():
                    spill_point_coarse_coords = self.coarse_grid.convert_fine_coords(spill_point,
                                                                                     self.grid)
                    lake_center_coarse_coords = \
                        self.coarse_grid.convert_fine_coords(self.lakes[other_lake].center_coords,
                                                             self.grid)
                    if spill_point_coarse_coords == lake_center_coarse_coords:
                        lake.outflow_points[other_lake] = (None,True)
                    else:
                        lake.outflow_points[other_lake] = (self.find_non_local_outflow_point(spill_point),False)

    def find_non_local_outflow_point(self,first_cell_beyond_rim_coords):
        if self.check_if_fine_cell_is_sink(first_cell_beyond_rim_coords):
                outflow_coords = \
                    self.coarse_grid.convert_fine_coords(first_cell_beyond_rim_coords,
                                                         self.grid)
        else:
            prior_fine_catchment_num = self.prior_fine_catchment_nums[first_cell_beyond_rim_coords]
            catchment_outlet_coarse_coords = None
            if self.sink_points[prior_fine_catchment_num - 1] is not None:
                catchment_outlet_coarse_coords = self.sink_points[prior_fine_catchment_num - 1]
            else:
                current_coords = copy(first_cell_beyond_rim_coords)
                while(True):
                    is_sink,downstream_coords = \
                        self.check_for_sinks_and_get_downstream_coords(current_coords)
                    if is_sink:
                        catchment_outlet_coarse_coords = \
                            self.coarse_grid.convert_fine_coords(current_coords,self.grid)
                        break
                    current_coords = downstream_coords
                if catchment_outlet_coarse_coords is None:
                    raise RuntimeError("Sink point for non local secondary redirect not found")
                self.sink_points[prior_fine_catchment_num - 1] = catchment_outlet_coarse_coords
            coarse_catchment_number = \
                self.coarse_catchment_nums[catchment_outlet_coarse_coords]
            coarse_first_cell_beyond_rim_coords = \
                self.coarse_grid.convert_fine_coords(first_cell_beyond_rim_coords,
                                                     self.grid)
            outflow_coords = self.coarse_search_alg(lambda coords :
                                                    self.coarse_catchment_nums[coords] ==
                                                    coarse_catchment_number,
                                                    lambda coords : False,
                                                    coarse_first_cell_beyond_rim_coords)
        return outflow_coords

    def get_lakes_as_array(self):
        array_offset = 1
        store_to_array = StoreToArray()
        for lake in self.lakes:
            store_to_array.add_object()
            store_to_array.add_number(lake.lake_number+array_offset)
            if lake.primary_lake is not None:
                store_to_array.add_number(lake.primary_lake+array_offset)
            else:
                store_to_array.add_number(-1)
            if lake.secondary_lakes is not None:
                store_to_array.add_field([x+array_offset
                                         for x in lake.secondary_lakes])
            else:
                store_to_array.add_field([])
            store_to_array.add_coords([x+array_offset for x in lake.center_coords])
            store_to_array.add_filling_order(lake.filling_order,
                                             array_offset=array_offset)
            store_to_array.add_outflow_points_dict(lake.outflow_points,
                                                   array_offset=array_offset)
            store_to_array.add_number(lake.lake_lower_boundary_height)
            store_to_array.add_number(lake.filled_lake_area)
            store_to_array.complete_object()
        return store_to_array.complete_array()



    def get_lake_mask(self):
        return self.lake_numbers != self.null_lake_number

    def get_number_of_lakes(self):
        return len(self.lakes)

    def get_lake_numbers(self):
        return self.lake_numbers

class LatLonBasinEvaluationAlgorithm(BasinEvaluationAlgorithm):

    def __init__(self,
                 minima,
                 raw_orography,
                 corrected_orography,
                 cell_areas,
                 prior_fine_rdirs,
                 prior_fine_catchment_nums,
                 coarse_catchment_nums,
                 catchments_from_sink_filling):
        self.grid = LatLonGrid(raw_orography.shape)
        self.coarse_grid = LatLonGrid(coarse_catchment_nums.shape)
        super().__init__(minima,
                         raw_orography,
                         corrected_orography,
                         cell_areas,
                         prior_fine_catchment_nums,
                         coarse_catchment_nums,
                         catchments_from_sink_filling)
        self.prior_fine_rdirs = prior_fine_rdirs

    def check_for_sinks_and_get_downstream_coords(self,coords_in):
        rdir = self.prior_fine_rdirs[coords_in]
        downstream_coords = self.grid.calculate_downstream_coords_from_dir_based_rdir(coords_in,rdir)
        next_rdir = self.prior_fine_rdirs[downstream_coords]
        return rdir == 5 or next_rdir == 0,downstream_coords

    def check_if_fine_cell_is_sink(self,coords_in):
        rdir = self.prior_fine_rdirs[coords_in]
        return rdir == 5

class SingleIndexBasinEvaluationAlgorithm(BasinEvaluationAlgorithm):

    def __init__(self,
                 minima,
                 raw_orography,
                 corrected_orography,
                 cell_areas,
                 prior_next_cell_indices,
                 prior_fine_catchment_nums,
                 coarse_catchment_nums,
                 catchments_from_sink_filling):
        super().__init__(minima,
                         raw_orography,
                         corrected_orography,
                         cell_areas,
                         prior_fine_catchment_nums,
                         coarse_catchment_nums,
                         catchments_from_sink_filling)
        self.prior_next_cell_indices = prior_next_cell_indices
        self.true_sink_value = -5

    def check_for_sinks_and_set_downstream_coords(self,coords_in):
        next_cell_index = self.prior_next_cell_indices[coords_in]
        downstream_coords = grid.calculate_downstream_coords_from_index_based_rdir(coords_in,next_cell_index)
        next_next_cell_index = next_cell_index[downstream_coords]
        return (next_cell_index == true_sink_value) and (next_next_cell_index == outflow_value)

    def check_if_fine_cell_is_sink(self,coords_in):
        next_cell_index = self.prior_next_cell_indices[coords_in]
        return (next_cell_index == self.true_sink_value)

class LatLonEvaluateBasin:

    @staticmethod
    def evaluate_basins(landsea_in,
                        minima_in,
                        raw_orography_in,
                        corrected_orography_in,
                        cell_areas_in,
                        prior_fine_rdirs_in,
                        prior_fine_catchments_in,
                        coarse_catchment_nums_in,
                        return_algorithm_object=False):
        catchments_from_sink_filling_in = \
            np.full(minima_in.shape,-1,dtype=np.int32)
        fill_sinks_cpp_func(orography_array=
                            corrected_orography_in,
                            method = 4,
                            use_ls_mask = True,
                            landsea_in = landsea_in,
                            set_ls_as_no_data_flag = False,
                            use_true_sinks = False,
                            true_sinks_in =
                            np.full(minima_in.shape,False,dtype=np.int32),
                            rdirs_in =
                            np.zeros(minima_in.shape,dtype=np.float64),
                            next_cell_lat_index_in =
                            np.zeros(minima_in.shape,dtype=np.int32),
                            next_cell_lon_index_in =
                            np.zeros(minima_in.shape,dtype=np.int32),
                            catchment_nums_in =
                            catchments_from_sink_filling_in,
                            prefer_non_diagonal_initial_dirs = False)
        alg = LatLonBasinEvaluationAlgorithm(minima_in,
                                             raw_orography_in,
                                             corrected_orography_in,
                                             cell_areas_in,
                                             prior_fine_rdirs_in,
                                             prior_fine_catchments_in,
                                             coarse_catchment_nums_in,
                                             catchments_from_sink_filling_in)
        alg.evaluate_basins()
        output = {"lakes_as_array":alg.get_lakes_as_array(),
                  "number_of_lakes":alg.get_number_of_lakes(),
                  "lake_mask":alg.get_lake_mask(),
                  "lake_numbers":alg.get_lake_numbers()}
        if return_algorithm_object:
            return output,alg
        else:
            return output

