class Lake:

    def __init__(self,
                 lake_number,
                 center_coords,
                 is_leaf=True,
                 primary_lake=None,
                 secondary_lakes=None):
        #Keep references to lakes as number not objects
        self.center_coords = center_coords
        self.lake_number = lake_number
        self.is_leaf = is_leaf
        self.primary_lake = primary_lake
        self.secondary_lakes  = secondary_lakes

    def set_primary_lake(self,primary_lake)
        self.primary_lake = primary_lake

class BasinCell:

    def __init__(cell_coords,cell_height_type,cell_height):
        self.cell_coords = center_coords
        self.cell_height_type = cell_height_type
        self.cell_height = cell_height

class DisjointSet:

    def __init__(self,label_in):
        self.label = label_in
        self.size = 1
        self.root = self
        nodes = []

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
      root_y->set_root(root_x)
      root_x->increase_size(root_y->get_size())
      root_x->add_node(root_y)
      root_x->add_nodes(root_y->get_nodes())
      return True

    def add_set(self,label_in):
      if self.get_set(label_in):
        return
      new_set = DisjointSet(label_in)
      sets.append(new_set)

    def get_set(self,label_in):
      for node in sets:
        if node.get_label() == label_in:
            return node
      return None

    def make_new_link(self,label_x,label_y):
      x = self.get_set(label_x)
      y = self.get_set(label_y)
      return self.link(x,y)

    def find_root(self,label_in):
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
        if root.get_nodes().size() + 1 != element_labels.size():
            return False
        if element_labels[0] != root.get_label():
            return False
        for i,node in enumerate(root.get_nodes()):
            i += 1
            if element_labels[i] != node.get_label():
                check_passed = False
        return check_passed

class BasinEvaluationAlgorithm:

    def evaluate_basins(self):
        while not self.mimima.empty():
            minimum = self.minima.top()
            self.minima.heappop()
            lake_number = len(self.lakes)
            self.lakes.append(Lake(lake_number,minimum))
            self.lake_q.append(lake_number)
            self.lake_connections.add_set(lake_number)
        while not self.lake_q.empty():
            lake = lake_q.top()
            lake_q.pop()
            self.initialize_basin(lake)
            while True:
                if self.q.empty():
                    raise RuntimeError("Basin outflow not found")
                self.center_cell = self.q.top()
                self.q.heappop()
                #Call the newly loaded coordinates and height for the center cell 'new'
                #until after making the test for merges then relabel. Center cell height/coords
                #without the 'new' moniker refers to the previous center cell; previous center cell
                #height/coords the previous previous center cell
                new_center_coords = self.center_cell.get_cell_coords().clone()
                new_center_cell_height_type = self.center_cell.get_height_type()
                new_center_cell_height = self.center_cell.get_height()
                #Exit to basin or level area found
                if (new_center_cell_height <= self.center_cell_height and
                    searched_level_height != self.center_cell_height):
                        outflow_lake_numbers = \
                            self.search_for_outflows_on_level(self.q
                                                              self.center_coords,
                                                              self.center_cell_height)
                        if len(outflow_lake_numbers) > 0:
                            #Exit(s) found
                            for other_lake_number in outflow_lake_numbers:
                                if other_lake_number != -1
                                    self.lake_connections.make_new_link(lake_number,other_basin_number)
                                    merging_lakes.append(lake_number)
                            self.fill_lake_orography(self.lake)
                            break
                        else:
                            #Don't rescan level later
                            searched_level_height = self.center_cell_height
                #Process neighbors of new center coords
                process_neighbors()
                previous_filled_cell_coords = center_coords.clone()
                previous_filled_cell_height = self.center_cell_height
                previous_filled_cell_height_type = self.center_cell_height_type
                center_cell_height_type = new_center_cell_height_type
                center_cell_height = new_center_cell_height
                center_coords = new_center_coords
                process_center_cell()
            if len(merging_lakes) > 0:
                unique_lake_groups = {g for g in [self.lake_connections.find_root(l)
                                                  for l in merging_lakes]}
                for lake_group in unique_lake_groups:
                    sublakes_in_lake = [sublake for sublake in self.lake_connections.get_set(lake_group)
                                        if sublake.primary_lake is None ]
                    new_lake_number = len(lakes)
                    new_lake = Lake(new_lake_number,
                                    lakes[sublakes_in_lake[0]].center_coords,
                                    is_leaf=False,
                                    primary_lake=None,
                                    secondary_lakes=sublakes_in_lake)
                    #Note the new lake isn't necessarily the root of the disjointed set
                    lake_connections.make_new_link(new_lake_number)
                    lakes.append(new_lake)
                    for sublake in sublakes_in_lake:
                        for other_sublake in sublakes_in_lakes
                            if sublake != other_sublake:
                                self.find_spill_point(sublake,other_sublake)
                    for sublake in sublakes_in_lake:
                        lake_numbers[lake_numbers == sublake] = new_lake.lake_number
                        lakes[sublake].set_primary_lake(new_lake.lake_number)
                merging_lakes = []
            else:
                break

    def initialize_basin(self.lake):
        self.completed_cells[:,:] = False
        self.center_cell_volume_threshold = 0.0
        self.center_coords = lake.center_coords.clone()
        raw_height = raw_orography[self.center_coords]
        corrected_height = corrected_orography[self.center_coords]
        if raw_height <= corrected_height:
            self.center_cell_height = raw_height
            self.center_cell_height_type = flood_height
        else:
            self.center_cell_height = corrected_height
            self.center_cell_height_type = connection_height
        self.previous_filled_cell_coords = center_coords.clone()
        self.previous_filled_cell_height_type = center_cell_height_type
        self.previous_filled_cell_height = center_cell_height
        self.catchments_from_sink_filling_catchment_num =
            catchments_from_sink_filling[center_coords]
        new_center_coords = center_coords.clone()
        new_center_cell_height_type = center_cell_height_type
        new_center_cell_height = center_cell_height
        if self.center_cell_height_type == connection_height:
            self.lake_area = 0.0
        else if self.center_cell_height_type == flood_height:
            self.lake_area = cell_areas[center_coords]
        else:
            raise RuntimeError("Cell type not recognized")
        self.completed_cells[center_coords] = True
        #Make partial first and second iteration
        self.process_neighbors()
        self.center_cell = self.q.top()
        self.q.heappop()
        new_center_coords = self.center_cell.get_cell_coords().clone()
        new_center_cell_height_type = self.center_cell.get_height_type()
        new_center_cell_height = self.center_cell.get_height()
        center_cell_height_type = new_center_cell_height_type
        center_cell_height = new_center_cell_height
        center_coords = new_center_coords
        self.process_center_cell()

    def search_for_outflows_on_level(self,q
                                     center_cell,
                                     center_cell_height):
        self.outflow_lake_number = []
        level_q.push_back(center_cell)
        while self.q[0].get_height() == center_cell_height:
            level_q.push_back(self.q.heappop())
        for cell in level_q:
            self.q.heappush(cell)
        while not level_q.empty():
            level_center_cell = level_q.pop()
            self.process_level_neighbors(level_center_cell.get_coords())
        return self.outflow_lake_numbers

    def process_level_neighbors(self,level_coords):
        neighbors_coords = raw_orography.get_neighbors_coords(level_coords,1)
        while not neighbors_coords.empty():
            nbr_coords = neighbors_coords.back()
            neighbors_coords.pop_back()
            if (not level_completed_cells[nbr_coords] and
                not self.completed_cells[nbr_coords]):
                raw_height = raw_orography[nbr_coords]
                corrected_height = corrected_orography[nbr_coords]
                level_completed_cells[nbr_coords] = True
                if raw_height <= corrected_height:
                    nbr_height = raw_height
                    nbr_height_type = flood_height
                else:
                    nbr_height = corrected_height
                    nbr_height_type = connection_height
                if nbr_height == self.center_cell_height:
                    level_q.push(BasinCell(nbr_height,nbr_height_type,
                                           nbr_coords))
                else if (nbr_height < self.center_cell_height and
                         lake_numbers[nbr_coords] != -1):
                    self.outflow_lake_numbers.push_back(-1)
                else if (nbr_height < self.center_cell_height or
                         lake_numbers[nbr_coords] != -1):
                    self.outflow_lake_numbers.push_back(lake_numbers[nbr_coords])

    def process_center_cell():
        if (basin_numbers[previous_filled_cell_coords] == null_catchment):
            basin_numbers[previous_filled_cell_coords] = basin_number
        center_cell_volume_threshold +=
                lake_area*(center_cell_height-previous_filled_cell_height)
        if previous_filled_cell_height_type == connection_height:
            self.q.heappush(BasinCell(raw_orography[previous_filled_cell_coords],
                                      flood_height,previous_filled_cell_coords.clone()))
            self.filling_order.append((coords_in,flood_height,center_cell_volume_threshold,
                                       center_cell_height))
        else if previous_filled_cell_height_type == flood_height:
            self.filling_order.append((coords_in,connect_height,center_cell_volume_threshold,
                                       center_cell_height))
        else:
            raise RuntimeError("Cell type not recognized")
        if (center_cell_height_type == flood_height):
            lake_area += cell_areas[center_coords]
        else if center_cell_height_type != connection_height:
            raise RuntimeError("Cell type not recognized")

    def process_neighbors(self):
        neighbors_coords = raw_orography.get_neighbors_coords(new_center_coords,1)
        while not neighbors_coords.empty():
            nbr_coords = neighbors_coords.back()
            neighbors_coords.pop_back()
            nbr_catchment = catchments_from_sink_filling[nbr_coords]
            in_different_catchment =
              ( nbr_catchment != catchments_from_sink_filling_catchment_num) and
              ( nbr_catchment != -1)
            if (not self.completed_cells[nbr_coords] and
                not in_different_catchment):
                raw_height = raw_orography[nbr_coords]
                corrected_height = corrected_orography[nbr_coords]
                if raw_height <= corrected_height:
                    nbr_height = raw_height
                    nbr_height_type = flood_height
                else:
                    nbr_height = corrected_height
                    nbr_height_type = connection_height
                self.q.heappush(BasinCell(nbr_height,nbr_height_type,
                                          nbr_coords))
                self.completed_cells[nbr_coords] = True
