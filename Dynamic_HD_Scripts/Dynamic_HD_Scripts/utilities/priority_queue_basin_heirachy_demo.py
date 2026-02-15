class Forest:

    def __init__(self):
        self.trees = []

    def find_node(self):

class Node:

    def __init__(self):
        self.direct_subnodes = []
        self.all_subnodes = []

def generate_basin_hierarchy_layer(cell_heights,cell_areas,
                                   depressions_center_coords,
                                   icon_grid=False,wrapped_grid_for_latlon=False,
                                   grid_shape_for_latlon=None,
                                   cell_neighbors_for_icon=None,
                                   edge_cells=None,
                                   print_output=False,
                                   basin_forest=None,
                                   basin_numbers=None):
    if basin_forest is None:
        basin_forest = Forest()
        cell_heights = cell_heights.clone()
        basin_numbers = np.zeros(cell_heights.shape,dtype=bool)
    raw_connections_list = []
    for i,depression_center_coords in enumerate(depression_center_coords):
        (order_of_cells_to_fill,cell_filling_volume_thresholds,
         cell_filling_height_thresholds,lake_area,connections,
         cells_in_basin) = \
            generate_filling_order(cell_heights,cell_areas,depression_center_coords,
                                   icon_grid=icon_grid,
                                   wrapped_grid_for_latlon=wrapped_grid_for_latlon,
                                   grid_shape_for_latlon=grid_shape_for_latlon,
                                   cell_neighbors_for_icon=cell_neighbors_for_icon,
                                   edge_cells=edge_cells,
                                   print_output=print_output)
        if connections:
            raw_connections_list.append((i,connections))
        basin_numbers[cells_in_basin] = i
    for basin_connections in raw_connections_list:
        connected_basins = { basin_numbers[connection_coords]
                             for connection_coords in basin_connections[1]}
        for connected_basin in connected_basins
        basin_forest.add_connection(i,connected_basin)
    depressions_center_coords = find_coords_in_basins(basin_forest.get_unprocessed_nodes(),
                                                      basin_numbers)
    if depressions_center_coords:
        generate_basin_hierarchy_layer(cell_heights,cell_areas,
                                       depressions_center_coords,
                                       icon_grid=icon_grid,
                                       wrapped_grid_for_latlon=wrapped_grid_for_latlon,
                                       grid_shape_for_latlon=grid_shape_for_latlon,
                                       cell_neighbors_for_icon=cell_neighbors_for_icon,
                                       edge_cells=edge_cells,
                                       print_output=print_output,
                                       basin_forest=basin_forest,
                                       basin_numbers=basin_numbers)

def find_coords_in_basin(basins_list,
                         basin_numbers_array):
    return [np.argwhere(basin_numbers_array == basin_number)[0]
            for basin_number in basins_list]



