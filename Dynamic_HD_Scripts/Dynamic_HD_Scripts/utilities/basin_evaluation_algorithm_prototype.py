class AssignLakeNumber():

    lake_number = -1

    def __call__(self):
        self.lake_number += 1
        return self.lake_number

    def __reset__(self):
        self.lake_number = -1

class Lake:

    def __init__(self,
                 is_leaf=True,
                 primary_lake=None,
                 secondary_lakes=None):
        self.is_leaf = is_leaf
        self.primary_connection_set = primary_connection_set
        self.primary_lake = primary_lake
        if self.primary_lake is not None:
            if self.primary_lake.root is None:
                self.root = self.primary_lake
            else:
                self.root = self.primary_lake.root
        else:
            self.root = self
        self.secondary_lakes  = None
        self.outflow_coords_list = None
        self.sill_point = None
        self.lake_number = AssignLakeNumber()

    def set_outflow_coords_list(self,outflow_coords_list):
        self.outflow_coords_list = outflow_coords_list

    def set_root(self,root):
        self.root = root

    def find_root(self):
        if self.primary_lake is None:
            return self
        elif self.root is not self:
            #Check root is up to date
            new_root = self.root.find_root():
            if self.root is not new_root:
                self.root.set_root(new_root)
                self.root = new_root
        else:
            #Use root of primary
            self.root = self.primary_lake.find_root():
            if self.root is not self.primary_lake:
                self.primary.set_root(self.root)
        return self.root

    def set_sill_point(self,coords):
        self.sill_point = coords

class basin_evaluation_algorithm:

    def evaluate_basins(self):
        while True
            merging_lakes = []
            self.initialize_basin()
            self.minimum = self.minima.top()
            self.minima.pop()
            self.q.append(next_basin)
            self.initialize_basin()
            while True:
                if self.q.empty():
                    raise RuntimeError("Basin outflow not found")
                self.center_cell = self.q.top()
                self.q.pop()
                #Call the newly loaded coordinates and height for the center cell 'new'
                #until after making the test for merges then relabel. Center cell height/coords
                #without the 'new' moniker refers to the previous center cell; previous center cell
                #height/coords the previous previous center cell
                self.read_new_center_cell_variables()
                #Exit to basin or level area found
                if (new_center_cell_height <= surface_height and
                    searched_level_height != surface_height):
                        outflow_coords_list = self.search_for_outflow_on_level(self.q
                                                                               self.center_coords,
                                                                               self.center_cell_height)
                        if len(outflow_coords_list) > 0:
                            #Exit found
                            self.lake.set_outflow_coords_list(outflow_coords_list)
                            for outflow_coords in outflow_coords_list
                                other_lake_number = lake_numbers[outflow_coords]
                                if other_lake_number != 0
                                    self.lake_connections.make_new_link(lake_number,other_basin_number)
                                    merging_lakes.append(lake_number)
                                    self.lake.set_sill_point(self.center_coords)
                            self.fill_lake_orography(self.lake)
                            break
                        else:
                            #Don't rescan level later
                            searched_level_height = surface_height
                process_neighbors()
                update_previous_filled_cell_variables()
                update_center_cell_variables()
                process_center_cell()
            if len(merging_lakes) > 0:
                unique_lake_groups = {g for g in [self.lake_connections.find_root(l)
                                                  for l in merging_lakes]}
                for lake_group in unique_lake_groups:
                    sublakes_in_lake = self.lake_connections.get_set(lake_group)
                    lake = Lake(is_leaf=False,
                                primary_lake=None,
                                secondary_lakes=)
                    lakes.append(lake)
                    minima.append(lakes[lake_number].sill_point)
                    for sublake in sublakes_in_lake:
                        lake_numbers[lake_numbers == sublake] = lake.lake_number
            else:
                break

    def initialize_basin(self):
        self.center_cell = self.minimum.clone()
        self.completed_cells[:,:] = False
        self.basin_flooded_cells[:,:] = False
        self.basin_connected_cells[:,:] = False
        self.center_cell_volume_threshold = 0.0
        self.basin_connections.add_set(lake.lake_number)
        self.center_coords = center_cell.get_cell_coords().clone()
        self.center_cell_height_type = center_cell.get_height_type()
        self.center_cell_height = center_cell.get_orography()
        self.surface_height = center_cell_height
        self.previous_filled_cell_coords = center_coords.clone()
        self.previous_filled_cell_height_type = center_cell_height_type
        self.previous_filled_cell_height = center_cell_height
        self.catchments_from_sink_filling_catchment_num =
            catchments_from_sink_filling[center_coords]
        self.read_new_center_cell_variables()
        if self.center_cell_height_type == connection_height:
            self.basin_connected_cells[center_coords] = True
            self.lake_area = 0.0
        else if self.center_cell_height_type == flood_height:
            self.basin_flooded_cells[center_coords] = True
            self.lake_area = cell_areas[center_coords]
        else:
            raise RuntimeError("Cell type not recognized")
        self.completed_cells[center_coords] = True
        #Make partial first and second iteration
        self.process_neighbors()
        self.center_cell = self.q.top()
        self.q.pop()
        self.read_new_center_cell_variables()
        self.process_neighbors()
        self.update_center_cell_variables()
        self.process_center_cell()

    #q is input only... must not change it... instead copy relevant variables
    def search_for_outflow_on_level(self,q
                                    center_coords,
                                    center_cell_height):

    def process_level_neighbors(self):
        neighbors_coords = raw_orography.get_neighbors_coords(level_coords,1)
        while not neighbors_coords.empty():
            nbr_coords = neighbors_coords.back()
            neighbors_coords.pop_back()
            if not level_completed_cells[nbr_coords]:
                        raw_height = raw_orography[nbr_coords]
                        corrected_height = corrected_orography[nbr_coords]
                        level_completed_cells[nbr_coords] = True
                        if raw_height <= corrected_height:
                            nbr_height = raw_height
                            nbr_height_type = flood_height
                        else:
                            nbr_height = corrected_height
                            nbr_height_type = connection_height
                        if nbr_height == surface_height:
                            level_q.push(BasinCell(nbr_height,nbr_height_type,
                                                   nbr_coords))
                        else if (nbr_height < surface_height):
                            level_nbr_q.push(BasinCell(nbr_height,nbr_height_type,
                                                       nbr_coords))

    def read_new_center_cell_variables(self):
        new_center_coords = self.center_cell.get_cell_coords().clone()
        new_center_cell_height_type = self.center_cell.get_height_type()
        new_center_cell_height = self.center_cell.get_orography()

    def update_previous_filled_cell_variables(self):
        previous_filled_cell_coords = center_coords.clone()
        previous_filled_cell_height = center_cell_height
        previous_filled_cell_height_type = center_cell_height_type

    def update_center_cell_variables(self):
        center_cell_height_type = new_center_cell_height_type
        center_cell_height = new_center_cell_height
        center_coords = new_center_coords
        if center_cell_height > surface_height:
            surface_height = center_cell_height

    def process_center_cell():
        set_previous_filled_cell_basin_number()
        center_cell_volume_threshold +=
                lake_area*(center_cell_height-previous_filled_cell_height)
        if previous_filled_cell_height_type == connection_height:
            connection_volume_thresholds[previous_filled_cell_coords] =
                center_cell_volume_threshold
            connection_heights[previous_filled_cell_coords] =
                center_cell_height
            self.q.push(BasinCell(raw_orography[previous_filled_cell_coords],
                                  flood_height,previous_filled_cell_coords.clone()))
            set_previous_cells_connect_next_cell_index(center_coords)
        else if previous_filled_cell_height_type == flood_height:
            flood_volume_thresholds[previous_filled_cell_coords] =
                center_cell_volume_threshold
            flood_heights[previous_filled_cell_coords] =
                center_cell_height
            set_previous_cells_flood_next_cell_index(center_coords)
        else:
            raise RuntimeError("Cell type not recognized")
        if center_cell_height_type == connection_height:
            connected_cells[center_coords] = True
            basin_connected_cells[center_coords] = True
        else if (center_cell_height_type == flood_height):
            flooded_cells[center_coords] = True
            basin_flooded_cells[center_coords] = True
            lake_area += cell_areas[center_coords]
        else:
            raise RuntimeError("Cell type not recognized")

    def set_previous_filled_cell_basin_number(self):
        if (basin_numbers[previous_filled_cell_coords]
                == null_catchment):
            basin_numbers[previous_filled_cell_coords] =
                basin_number

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
                self.q.push(BasinCell(nbr_height,nbr_height_type,
                                      nbr_coords))
                self.completed_cells[nbr_coords] = True
