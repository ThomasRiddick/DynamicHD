from queue import PriorityQueue
import numpy as np

# This main function computes the order the cells fill in and all other necessary information
# Conceptually the lake is broken into a stack of horizontal slabs, one for each cell that is added
# starting with the smallest one cell slab at the botton, then with a two cell slab on top of that 
# and so forth. The area, depth and thus volume of each slab is calculated and lists are made of the
# order the cells fill and of threshold volume for each cell to start filling and of their heights
# Unfortunately in python the priority queue only works for integers (unlike other languages)
# Hence can only use integer heights. For decimals heights you could multiple by say 100 and then
# divide the resulting volumes by 100 and you'd get the same results as if you'd worked with decimals
# to the 2 decimal place.
# Can only handle 1 depression at a time. For a complex depression with two mimima you could process 1 as 
# normal, will the cells in that depression to the height of the outflow then process the other and to get
# the total lake volume add the volume of the two calculations together.
def generate_filling_order(cell_heights,cell_areas,depression_center_coords,
			   icon_grid=False,wrapped_grid_for_latlon=False,
			   grid_shape_for_latlon=None,
			   cell_neighbors_for_icon=None,
			   edge_cells=None,
			   print_output=False):
	#The order in which the lake cells will fill
	order_of_cells_to_fill = []
	#The threshold for each cell to *start* filling, listed in filling order
	cell_filling_volume_thresholds = []
	#The height of each cell, listed in filling order
	cell_filling_height_thresholds = []
	#The area of the lake once the nth cell in the filling order has filled
	lake_area = []
	#A list to return the cells this basin overflows into; set to none if it isn't connected to any
	connections = None
	#An array of cells already processed (2D for a latlon grid, 1D for an icon grid). Start filled
	#with falses 
	processed_cells = np.zeros(cell_heights.shape,dtype=bool)
	#A Priority queue, a queueing structure that keeps entries ordered (in ascending order) by the size of 
	#their first element in this case by the height of the cells it is filled with. 
	q = PriorityQueue()
	#Add the cell at the bottom of the depression the queue. Record its height and its position
	q.put((cell_heights[depression_center_coords],depression_center_coords))
	#Initialise various running total for this lake
	total_volume = 0.0
	total_area = 0.0
	#Initialise these values of height for the current and previous working cells
	previous_height = cell_heights[depression_center_coords]
	current_height = cell_heights[depression_center_coords]
	while not q.empty():
		#Remove the first entry in the queue i.e. the lowest cell awaiting processing 
		working_cell = q.get()	
		#Get the cell's coords
		working_cell_coords = working_cell[1]
		#Check this cell hasn't already been dealt with... If more than one neighbor adds 
		#the cell to the queue it can have multiple entries but we only want to process it the
		#first time
		if processed_cells[working_cell_coords]:
			#If it was processed before skip to the next iteration of the main loop
			continue
		#Get the height of the current cell
		current_height = working_cell[0]
		#Check we haven't reached the edge of the basin. This cell being lower than the previous cell would
		#indicate the previous cell was the outflow point of the basin
		if current_height < previous_height:
			print("Top of depression reached")
			overflow_coords = order_of_cells_to_fill[-1]
			print("Depression overflow location",format(overflow_coords))
			if connections:
				connections.append(working_cell_coords)
			else:
				connections = [working_cell_coords]
			#Check in case there is more than one overflow at exactly the same level
			extra_connections = check_for_multiple_connections(q,previous_height,
			                                                   overflow_coords,
			                                                   working_cell_coords,
			                                                   processed_cells,
								   							   icon_grid,
								   							   wrapped_grid_for_latlon,
								   							   grid_shape_for_latlon,
								   							   cell_neighbors_for_icon)
			if extra_connections:
				connections.extend(extra_connection)
			break
		#Check we haven't exited the area under consideration - for a global grid this will never occur
		if edge_cells[working_cell_coords]:
			print("Edge of area of interest reached")
			print("Exit location ",format(order_of_cells_to_fill[-1]))
			break
		#Calculate the new total area of the lake required to start filling this cell
		#This will be the change in height from this cell compared to the last times 
		#the area of the lake including the last cell processed but not this one
		#Note the area of the working cell isn't include in this calculation... we are
                #working out the volume at which it would *start* filling
		total_volume +=	total_area*(current_height - previous_height)	
		#Now add this cell's area to the total area
		total_area += cell_areas[working_cell_coords]
		#Fill the various lists of values for this cell
		order_of_cells_to_fill.append(working_cell_coords)
		cell_filling_volume_thresholds.append(total_volume)
		cell_filling_height_thresholds.append(current_height)
		lake_area.append(total_area)
		if print_output:
			print(f"Processing cell {working_cell_coords}")
			print(f"Height of current cell {current_height}")
			print(f"Total depression volume required to start filling this cell {total_volume}") 
			print(f"New total area of lake (including this cell) {total_area}")
			print("Of course all previously processed cells are still filling too")
		#Mark this cell as processed
		processed_cells[working_cell_coords] = True
		#Save the current height as previous height ready for the next iteration
		previous_height = current_height
		#Iterate over a list of neighbors of this cell
		for neighbor_coords in get_neighbor_coords_list(working_cell_coords,
								icon_grid,
								wrapped_grid_for_latlon,
								grid_shape_for_latlon,
								cell_neighbors_for_icon):
			#If neighbor isn't processed already add it to the queue including its height and
			#its location. The cell will automatically be placed in the queue it according to its height
			if not processed_cells[neighbor_coords]:
				if print_output:
					print(f" Adding neighbor to queue: {neighbor_coords}")
				q.put((cell_heights[neighbor_coords],neighbor_coords))	
	#Final list of cell in lake is just the list of processed_cells
	cells_in_basin = processed_cells
	#Return results of running this function
	return (order_of_cells_to_fill,cell_filling_volume_thresholds,
	        cell_filling_height_thresholds,lake_area,connections,
	        cells_in_basin)

def get_neighbor_coords_list(working_cell_coords,
			     icon_grid,
			     wrapped_grid_for_latlon,
			     grid_shape_for_latlon,
		 	     cell_neighbors_for_icon):
	if icon_grid:
		#For the ICON grid simply take neighbors from a list filtering out any entries indicating a missing
		#neighbor
		return [coords for coords in cell_neighbors_for_icon[working_cell_coords,:] if coords != -1]
	else:
		#For lat-lon generate a list of the 8 neighbors around the cell excluding any that are outside the
		#grid; wrapping the longitude values if that option is set
		cell_neighbors = []
		for i in range(3):	
			for j in range(3):
				if i == 1 and j == 1:
					continue
				new_lat = working_cell_coords[0] + i - 1
				new_lon = working_cell_coords[1] + j - 1
				if (new_lon < 0 or new_lon >= grid_shape_for_latlon[1]):
					if wrapped_grid_for_latlon:
						if new_lon < 0:
							new_lon = grid_shape_for_latlon[1] - 1	
						else:
							new_lon = 0
					else:
						continue
				if(new_lat < 0 or new_lat >= grid_shape_for_latlon[0]):
					continue 
				cell_neighbors.append((new_lat,new_lon))
		return cell_neighbors

def calculate_secondary_neighbors(neighboring_cell_indices):
	#Calculate the secondary neighbors for icon grid given primary neighbors
	ncells = neighboring_cell_indices.shape[0]
	secondary_neighboring_cell_indices = np.zeros((ncells,9),dtype=np.int64)
	for index_over_grid in range(ncells):
		#Six secondary neighbors are neighbors of primary neighbors
		for index_over_primary_nbrs in range(3):
			primary_neighbor_index = \
				neighboring_cell_indices[index_over_grid,index_over_primary_nbrs]
			valid_secondary_nbr_count = 0;
			for index_over_secondary_nbrs in range(3):
				#Actually process this 2 rather than 3 times primary neighbor index (by skipping one iteration) as we miss 
				#out 1 secondary neighbor for each primary neighbor 
				secondary_neighbor_index = \
					neighboring_cell_indices[primary_neighbor_index,index_over_secondary_nbrs]
				if secondary_neighbor_index != index_over_grid:
					#Note this leaves gaps for the remaining three secondary neighbors
					secondary_neighboring_cell_indices[index_over_grid,
									   3*index_over_primary_nbrs+valid_secondary_nbr_count] = secondary_neighbor_index
					valid_secondary_nbr_count += 1
		#Three secondary neighbors are common neighbors of the existing secondary neighbors
		gap_index = 2
		#Last secondary neighbor is as yet unfilled so loop only up to an index of 7
		index_over_secondary_nbrs= -1	
		while index_over_secondary_nbrs < 7:
			index_over_secondary_nbrs += 1
			#skip as yet unfilled entries in the secondary neighbors array
			if (index_over_secondary_nbrs+1)%3 == 0:
				index_over_secondary_nbrs += 1
			first_secondary_neighbor_index = \
				secondary_neighboring_cell_indices[index_over_grid,index_over_secondary_nbrs]
	  		#Last secondary neighbor is as yet unfilled so loop only up to an index of 7
			second_index_over_secondary_nbrs = index_over_secondary_nbrs + 1
			while second_index_over_secondary_nbrs < 7:
				second_index_over_secondary_nbrs += 1
				if (second_index_over_secondary_nbrs+1)%3 == 0: 
					second_index_over_secondary_nbrs += 1
				second_secondary_neighbor_index = \
					secondary_neighboring_cell_indices[index_over_grid,second_index_over_secondary_nbrs]
				#Some tertiary neighbors are also secondary neighbors
				for index_over_tertiary_nbrs in range(3):
					tertiary_neighbor_index = \
						neighboring_cell_indices[first_secondary_neighbor_index,
					                           	 index_over_tertiary_nbrs] 
					#Test to see if this one of the twelve 5-point vertices in the grid
					if second_secondary_neighbor_index == tertiary_neighbor_index:
						secondary_neighboring_cell_indices[index_over_grid,gap_index] = -1;
						gap_index += 3
						continue
					for second_index_over_tertiary_nbrs in range(3):
						second_tertiary_neighbor_index = \
							neighboring_cell_indices[second_secondary_neighbor_index, 
					                             	         second_index_over_tertiary_nbrs] 
						if second_tertiary_neighbor_index == tertiary_neighbor_index:
							secondary_neighboring_cell_indices[index_over_grid,gap_index] = tertiary_neighbor_index 
							gap_index += 3
	return secondary_neighboring_cell_indices

def check_for_multiple_connections(q,previous_height,
                                   overflow_coords,
                                   first_outflow_downstream_coords,
			                       processed_cells,
								   icon_grid,
								   wrapped_grid_for_latlon,
								   grid_shape_for_latlon,
								   cell_neighbors_for_icon):
	cell_list = [item for item in q].append((previous_height,overflow_coords))
	extra_connections = []
	for cell in cell_list:
		working_cell_height = cell[0]
		working_cell_coords = cell[1]
		if working_cell_height == previous_height:
			#Iterate over a list of neighbors of this cell
			for neighbor_coords in get_neighbor_coords_list(working_cell_coords,
															icon_grid,
															wrapped_grid_for_latlon,
															grid_shape_for_latlon,
															cell_neighbors_for_icon):
				if (not processed_cells[neighbor_coords] and
					neighbor_coords =/ ffirst_outflow_downstream_coords):
					if print_output:
						print(f" Additional outflow to : {neighbor_coords}")
					extra_connections.append(neighbor_coords)
	return extra_connections


def print_icon_array(array):
	#Print icon arrays in a understandable form
	print("    " + " ".join([str(array[i]) + "       " for i in range(5)]))	
	print(" " + " ".join([str(array[i]) + " " for i in range(5,20)]))	
	print(" ".join([str(array[i]) for i in range(20,40)]))	
	print(" ".join([str(array[i]) for i in range(40,60)]))	
	print(" " + " ".join([str(array[i]) + " " for i in range(60,75)]))	
	print("    " + " ".join([str(array[i]) + "       " for i in range(75,80)]))	

if __name__ == '__main__':
	#Print the output during running to show working?
	print_output = False
	#Run the latitude longitude example
	run_latlon_example=True
	#Run the icon grid example
	run_icon_grid_example=True
	if run_latlon_example:
		print("------------------------------------------------------------------")
		print("Running Example on LatLon Grid")
		#Prep some test data
		cell_heights = np.array([[100,100,100, 100,100,100, 100,100,100],
                                 	 [100,100,100, 100,100,100, 100,100,100],
                                 	 [100,100, 71,  76, 83, 65, 100,100,100],

                                 	 [100, 68, 58,  57, 56, 61,  80,100,100],
                                 	 [100,100, 59,  51, 55, 59,  79,100,100],
                                 	 [100,100, 73,  66, 81, 82, 100,100,100],

                                 	 [100,100, 74, 100,100, 84, 100,100,100],
                                 	 [100,100, 75, 100,100, 85, 100,100,100],
                                 	 [100,100,100, 100,100, 82, 100,100,100]])
		cell_areas = np.ones(cell_heights.shape,dtype=np.float64)
		edge_cells = np.zeros(cell_heights.shape,dtype=bool)
		depression_center_coords = (5-1,4-1)
		#Run main routine
		order_of_cells_to_fill,cell_filling_volume_thresholds,cell_filling_height_thresholds,lake_area = \
			generate_filling_order(cell_heights,cell_areas,depression_center_coords,
			       	       	       icon_grid=False,wrapped_grid_for_latlon=False,
			       	       	       grid_shape_for_latlon=cell_heights.shape,
			       	       	       cell_neighbors_for_icon=None,
			       	       	       edge_cell=None,
			       	       	       print_output=print_output)
		#Print results
		print("------------------------------------------------------------------")
		print("Results")
		print("Cell Coords, Volume of Lake for Filling to Start, Height of Cell, Fraction of Grid that is Lake Once Cell Fills")
		total_area = np.sum(cell_areas)
		for coords,volume,height,area in zip(order_of_cells_to_fill,
						cell_filling_volume_thresholds,
						cell_filling_height_thresholds,
						lake_area):
			fraction = area/total_area
			print(f"{coords}, {volume}, {height}, {fraction:6.3f}")
		print("Cell Filling Order")
		cell_filling_order = np.zeros(cell_heights.shape,dtype=np.int64)
		for cell_number,coords in enumerate(order_of_cells_to_fill,start=1):
			cell_filling_order[coords] = cell_number
		print(cell_filling_order)
	if run_icon_grid_example:
		print("------------------------------------------------------------------")
		print("Running Example on Icon Grid")
		print("Watch out for the -1 array offset in Python!")
		#Prep the grid information - I think this is a R2B0 grid possibly??
		#These are the cell numbers from Fortran - will remove one from them below to give python cell numbers
		cell_neighbors = np.array([  
					   #1
    					   [5,7,2],
    					   #2
    					   [1,10,3],
    				           #3
    					   [2,13,4],
    					   #4
    					   [3,16,5],
    					   #5
    					   [4,19,1],
    					   #6
    					   [20,21,7],
                                           #7
    					   [1,6,8],
    					   #8
    					   [7,23,9],
    					   #9
    					   [8,25,10],
    					   #10
    					   [2,9,11],
    					   #11
    					   [10,27,12],
    					   #12
    					   [11,29,13],
    					   #13
    					   [3,12,14],
    					   #14
    					   [13,31,15],
    					   #15
    					   [14,33,16],
    					   #16
    					   [4,15,17],
    					   #17
    					   [16,35,18],
    					   #18
    					   [17,37,19],
    					   #19
    					   [5,18,20],
    					   #20
    					   [19,39,6],
    					   #21
    					   [6,40,22],
    					   #22
    					   [21,41,23],
    					   #23
    					   [8,22,24],
    					   #24
    					   [23,43,25],
    					   #25
    					   [24,26,9],
    					   #26
    					   [25,45,27],
    					   #27
    					   [11,26,28],
    					   #28
    					   [27,47,29],
    					   #29
    					   [12,28,30],
    					   #30
    					   [29,49,31],
    					   #31
    					   [14,30,32],
    					   #32
    					   [31,51,33],
    					   #33
    					   [15,32,34],
    					   #34
    					   [33,53,35],
    					   #35
    					   [17,34,36],
    					   #36
    					   [35,55,37],
    					   #37
    					   [18,36,38],
    					   #38
    					   [37,57,39],
    					   #39
    					   [20,38,40],
    					   #40
    					   [39,59,21],
    					   #41
    					   [22,60,42],
    					   #42
    					   [41,61,43],
    					   #43
    					   [24,42,44],
    					   #44
    					   [43,63,45],
    					   #45
    					   [26,44,46],
    					   #46
    					   [45,64,47],
    					   #47
    					   [28,46,48],
    					   #48
    					   [47,66,49],
    					   #49
    					   [30,48,50],
    					   #50
    					   [49,67,51],
    					   #51
    					   [32,50,52],
    					   #52
    					   [51,69,53],
    					   #53
    					   [34,52,54],
    					   #54
    					   [53,70,55],
    					   #55
    					   [36,54,56],
    					   #56
    					   [55,72,57],
    					   #57
    					   [38,56,58],
    					   #58
    					   [57,73,59],
    					   #59
    					   [40,58,60],
    					   #60
    					   [59,75,41],
    					   #61
    					   [42,75,62],
    					   #62
    					   [61,76,63],
    					   #63
    					   [44,62,64],
    					   #64
    					   [46,63,65],
    					   #65
    					   [64,77,66],
    					   #66
    					   [48,65,67],
    					   #67
    					   [50,66,68],
    					   #68
    					   [67,78,69],
    					   #69
    					   [52,68,70],
    					   #70
    					   [54,69,71],
    					   #71
    					   [70,79,72],
    					   #72
    					   [56,71,73],
    					   #73
    					   [58,72,74],
    					   #74
    					   [73,80,75],
    					   #75
    					   [60,74,61],
    					   #76
    					   [62,80,77],
    					   #77
    					   [65,76,78],
    					   #78
    					   [68,77,79],
    					   #79
    					   [71,78,80],
    					   #80
    					   [74,79,76]])
		#Python uses -1 array offset (cells 1 to 80 have indices 0 to 79)
		cell_neighbors -= 1	
		secondary_neighbors = calculate_secondary_neighbors(cell_neighbors) 
		cell_neighbors = np.concatenate((cell_neighbors,secondary_neighbors),1)
		cell_numbers = np.arange(80)
		print("Cell Numbers")
		print_icon_array(cell_numbers)
		#Prep some test data
		cell_heights = np.array([    100,        100,        100,        100,        100,       
 				 	  100, 55, 54, 65, 62, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                                        100,58,54,51,53,100,66, 100,63,100,100,100,100,100,100,100,100,100,100,100,
                                        57,100,56,52,61,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,
 				 	  100, 52, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
					    53,        54,        100,        100,        100 ])
		cell_areas = np.ones(cell_heights.shape,dtype=np.float64)
		edge_cells = np.zeros(cell_heights.shape,dtype=bool)
		depression_center_coords = (24 -1)
		#Run main routine
		order_of_cells_to_fill,cell_filling_volume_thresholds,cell_filling_height_thresholds,lake_area = \
			generate_filling_order(cell_heights,cell_areas,depression_center_coords,
			       	       	       icon_grid=True,wrapped_grid_for_latlon=False,
			       	       	       grid_shape_for_latlon=None,
			       	       	       cell_neighbors_for_icon=cell_neighbors,
			       	       	       edge_cell=None,
			       	       	       print_output=print_output)
		#Print results
		print("------------------------------------------------------------------")
		print("Results")
		print("Cell Coords, Volume of Lake for Filling to Start, Height of Cell, Fraction of Grid that is Lake Once Cell Fills")
		total_area = np.sum(cell_areas)
		for coords,volume,height,area in zip(order_of_cells_to_fill,
						cell_filling_volume_thresholds,
						cell_filling_height_thresholds,
						lake_area):
			fraction = area/total_area
			print(f"{coords}, {volume}, {height}, {fraction:6.3f}")
		print("Cell Filling Order")
		cell_filling_order = np.zeros(cell_heights.shape,dtype=np.int64)
		for cell_number,coords in enumerate(order_of_cells_to_fill,start=1):
			cell_filling_order[coords] = cell_number
		print_icon_array(cell_filling_order)
