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
        self.spill_points = {}
        self.potential_exit_points = []
        self.outflow_points = {}

    def set_primary_lake(self,primary_lake)
        self.primary_lake = primary_lake

    def set_potential_exit_points(self,potential_exit_points)
        self.potential_exit_points = potential_exit_points

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
      root_y.set_root(root_x)
      root_x.increase_size(root_y.get_size())
      root_x.add_node(root_y)
      root_x.add_nodes(root_y.get_nodes())
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



class SimpleSearch:

    def __init__(self):
        self.search_completed_cells =

    def __call__(self,
                 target_found_func,
                 ignore_nbr_func,
                 start_point):
        self.search_q =
        self.search_q.push(landsea_cell(start_point))
        self.search_completed_cells[:,:] = False
        self.ignore_nbr_func = ignore_nbr_func
        while not self.search_q.empty():
            search_cell = self.search_q.front()
            self.search_q.pop()
            search_coords = search_cell.get_cell_coords()
            if target_found_func(search_coords)
                return search_coords
            self.search_process_neighbors(search_coords)

    def search_process_neighbors(self,search_coords):
        search_neighbors_coords = \
            self.search_completed_cells.get_neighbors_coords(search_coords)
        while not search_neighbors_coords.empty():
            search_nbr_coords = search_neighbors_coords.back()
            search_neighbors_coords.pop_back()
            if not (self.search_completed_cells[search_nbr_coords] or
                    self.ignore_nbr_func(search_nbr_coords)):
                self.search_q.push(landsea_cell(search_nbr_coords))
                self.search_completed_cells[search_nbr_coords] = True

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
                        outflow_lake_numbers,potential_exit_points = \
                            self.search_for_outflows_on_level(self.q
                                                              self.center_coords,
                                                              self.center_cell_height)
                        if len(outflow_lake_numbers) > 0:
                            #Exit(s) found
                            for other_lake_number in outflow_lake_numbers:
                                if other_lake_number != -1
                                    self.lake_connections.make_new_link(lake_number,other_basin_number)
                                    merging_lakes.append(lake_number)
                            self.lake.set_potential_exit_points(potential_exit_points)
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
                    sublakes_in_lake = {sublake for sublake in self.lake_connections.get_set(lake_group)
                                        if sublake.primary_lake is None }
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
                                lakes[sublake].spill_points[other_sublake] = \
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
        self.outflow_lake_numbers = []
        self.potential_exit_points = []
        level_q.push_back(center_cell)
        while self.q[0].get_height() == center_cell_height:
            level_q.push_back(self.q.heappop())
        for cell in level_q:
            self.q.heappush(cell)
        while not level_q.empty():
            level_center_cell = level_q.pop()
            self.process_level_neighbors(level_center_cell.get_coords())
        return self.outflow_lake_numbers,self.potential_exit_points

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
                    self.potential_exit_points = nbr_coords
                else if (nbr_height < self.center_cell_height or
                         lake_numbers[nbr_coords] != -1):
                    self.outflow_lake_numbers.push_back(lake_numbers[nbr_coords])
                    self.potential_exit_points = nbr_coords

    def process_center_cell(self):
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

    def find_spill_point(self,sublake,other_sublake):
        return self.search_alg(lamba coords :
                               self.lake_numbers[coords] == other_sublake,
                               lambda coords :
                               self.corrected_height[coords] !=
                               self.corrected_height[self.lakes[sublake].center_coords],
                               self.lakes[sublake].center_coords)

    def set_outflows(self):
        for lake in self.lakes
            if lake.primary_lake:
                for potential_exit_point in potential_exit_points:
                    if self.lake_numbers[potential_exit_points] != -1:
                        raise RuntimeError("Lake on top of heirarchy trying"
                                           "to spill into another lake")
                first_cell_beyond_rim_coords = potential_exit_points[0]
                fine_catchment_num = self.fine_catchments[first_cell_beyond_rim_coords]
                lake.outflow_points[-1] = (self.find_non_local_outflow_point(first_cell_beyond_rim_coords),False)
            else:
                for other_lake,spill_point in lake.spill_points.items()
                    spill_point_coarse_coords = _coarse_grid.convert_fine_coords(spill_point,
                                                                                 _grid_params)
                    lake_center_coarse_coords = _coarse_grid.convert_fine_coords(other_lake.center_coords,
                                                                                 _grid_params)
                    if spill_point_coarse_coords == lake_center_coarse_coords:
                        lake.outflow_points[other_lake] = (None,True)
                    else:
                        lake.outflow_points[other_lake] = (self.find_non_local_outflow_point(spill_point),False)

    def find_non_local_outflow_point(self,first_cell_beyond_rim_coords)
        if check_if_fine_cell_is_sink(first_cell_beyond_rim_coords):
                outflow_coords = \
                    _coarse_grid.convert_fine_coords(first_cell_beyond_rim_coords,
                                                     _grid_params)
        else:
            prior_fine_catchment_num = prior_fine_catchment_numbers[first_cell_beyond_rim_coords]
            if sink_points[prior_fine_catchment_num - 1] is not None:
                catchment_outlet_coarse_coords = sink_points[prior_fine_catchment_num - 1]
            else:
                catchment_outlet_coarse_coords = None
                current_coords = first_cell_beyond_rim_coords.clone()
                while(True):
                    if self.check_for_sinks_and_get_downstream_coords(current_coords):
                        catchment_outlet_coarse_coords,downstream_coords = \
                            _coarse_grid.convert_fine_coords(current_coords,
                                                              _grid_params)
                        break
                    current_coords = downstream_coords
                if not catchment_outlet_coarse_coords:
                    raise RuntimeError("Sink point for non local secondary redirect not found")
                sink_points[prior_fine_catchment_num - 1] = catchment_outlet_coarse_coords
            coarse_catchment_number = \
                coarse_catchment_nums[catchment_outlet_coarse_coords]
            outflow_coords = self.search_alg(lamba coords :
                                             coarse_catchment_nums[coords] == coarse_catchment_number,
                                             lamba coords : False,
                                             first_cell_beyond_rim_coords.clone())
        return outflow_coords


    def check_for_sinks_and_get_downstream_coords(self,coords_in):
        rdir = prior_fine_rdirs[coords_in]
        downstream_coords = _grid.calculate_downstream_coords_from_dir_based_rdir(coords_in,rdir)
        next_rdir = prior_fine_rdirs[downstream_coords]
        return rdir == 5.0 or next_rdir == 0.0,downstream_coords

    def coarse_cell_is_sink(self,coords_in):
        return prior_coarse_rdirs[coords_in] == 5.0

    def set_previous_cells_flood_next_cell_index(self,coords_in):
        latlon_coords_in = coords_in
        flood_next_cell_lat_index[previous_filled_cell_coords] = latlon_coords_in.get_lat()
        flood_next_cell_lon_index[previous_filled_cell_coords] = latlon_coords_in.get_lon()

    def set_previous_cells_connect_next_cell_index(self,coords_in):
        latlon_coords_in = coords_in
        connect_next_cell_lat_index[previous_filled_cell_coords] = latlon_coords_in.get_lat()
        connect_next_cell_lon_index[previous_filled_cell_coords] = latlon_coords_in.get_lon()

    def get_cells_next_cell_index_as_coords(self,coords_in,
                                            height_type_in):

        latlon_coords_in = coords_in
        if height_type_in == flood_height:
            return latlon_coords(flood_next_cell_lat_index[latlon_coords_in],
                                 flood_next_cell_lon_index[latlon_coords_in])
        else if height_type_in == connection_height:
            return latlon_coords(connect_next_cell_lat_index[latlon_coords_in],
                                 connect_next_cell_lon_index[latlon_coords_in])
        else:
            raise RuntimeError("Height type not recognized")

    def check_for_sinks_and_set_downstream_coords(self,coords_in):
        rdir = prior_fine_rdirs[coords_in]
        downstream_coords = _grid.calculate_downstream_coords_from_index_based_rdir(coords_in,rdir)
        next_rdir = prior_fine_rdirs[downstream_coords]
        return rdir == true_sink_value || next_rdir == outflow_value

    def coarse_cell_is_sink(self,coords_in):
        rdir = prior_coarse_rdirs[coords_in]
        return rdir == true_sink_value

    def set_previous_cells_flood_next_cell_index(self,coords_in):
        generic_1d_coords_in = coords_in
        flood_next_cell_index[previous_filled_cell_coords] = generic_1d_coords_in.get_index()

    def set_previous_cells_connect_next_cell_index(self,coords_in):
        generic_1d_coords_in = coords_in
        connect_next_cell_index[previous_filled_cell_coords] = generic_1d_coords_in.get_index()

    def get_cells_next_cell_index_as_coords(self,coords_in,
                                            height_type_in):

        generic_1d_coords_in = coords_in
        if height_type_in == flood_height:
            return generic_1d_coords(flood_next_cell_index[generic_1d_coords_in])
        else if height_type_in == connection_height:
            return generic_1d_coords(connect_next_cell_index[generic_1d_coords_in])
        else:
            raise RuntimeError("Height type not recognized")
