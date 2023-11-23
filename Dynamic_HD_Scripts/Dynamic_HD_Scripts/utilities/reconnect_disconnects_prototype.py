def check_for_trivial_reconnects():

def route_along_boundaries(start_coords,end_coords):

def expand_route(path,start,end):
	for edge_index in path:
		potential_cells_for_pathing_mask[cells_bounding_edge[edge_index,0]] = True
		potential_cells_for_pathing_mask[cells_bounding_edge[edge_index,1]] = True
	completed_cells = np.zeros(catchments.shape,dtype=bool)
	changed_cells = np.zeros(catchments.shape,dtype=bool)
	done = False
	while (((not own_catchment_q.empty()) or (not headwater_cell_q.empty()) or
	        (not other_catchments_q.empty())) and not done): 
		while not own_catchment_q.empty():
			done = process_q(own_catchment_q,catchment,end)
			if done:
				break;
		if not headwater_catchment_q.empty():
			done = process_q(headwater_catchment_q,catchment,end)
		else:	
			done = process_q(other_catchments_q,catchment,end)
	end ---> Follow back path and mark changed and new rdir and push all new disconnects
	redo all boundaries of changed_cells 

def process_q(q,catchment,end):
	for cell in q:
		if cell.cell_coords == end:
			return True
		nbrs = get_neighbors(cell.cell_coords)
		for nbr in nbrs:
			if not completed_cells[nbr]:
				completed_cells[nbr] = True
				if catchments[nbr] == catchment:	
					own_catchment_q.push(Cell(nbr,cell.cell_coords))
		return False
			
	

def remove_disconnects(disconnects_q):
	for edge_index in edges:
		if not complete_edges[edge_index]:
			cell_one_coords = cells_bounding_edge[edge_index,0]
			cell_two_coords = cells_bounding_edge[edge_index,1]
			boundaries[edge_index] = (catchments[cell_one_coords] != catchment[cell_two_coords]) 
	while not disconnects_q.empty():
		check_for_trivial_reconnect(disconnect.start,disconnect.end)
		path = route_along_boundaries(disconnect.start,disconnect.end)
		expand_route(path,disconnect.start,disconnect.end)
