module IdentifyBifurcatedRiverMouths

struct Cells
	cell_indices::Array{CartesianIndex}
	cell_neighbors::Array{CartesianIndex}
	cell_coords::@NamedTuple{lats::Array{Float64},lons::Array{Float64}}
	cell_vertices::Array{@NamedTuple{lats::Array{Float64},lons::Array{Float64}}}
	is_wrapped_cell::Array{Bool}
	cell_extremes::@NamedTuple{min_lats::Array{Float64},max_lats::Array{Float64},
				   									 min_lons::Array{Float64},max_lons::Array{Float64}}
	function Cells(cell_indices::Array{CartesianIndex},
								 cell_neighbors::Array{CartesianIndex},
								 cell_coords::@NamedTuple{lats::Array{Float64},lons::Array{Float64}},
								 cell_vertices::@NamedTuple{lats::Array{Float64},lons::Array{Float64}})
		restructured_cell_vertices::Array{@NamedTuple{lats::Array{Float64},lons::Array{Float64}}} =
			@NamedTuple{lats::Array{Float64},
									lons::Array{Float64}}[ (lats = cell_vertices.lats[:,i],
									                        lons = cell_vertices.lons[:,i]) for i = 1:3 ]
		cell_extremes::@NamedTuple{min_lats::Array{Float64},max_lats::Array{Float64},
				   									   min_lons::Array{Float64},max_lons::Array{Float64}} =
				   					(min_lats = minimum(cell_vertices.lats,dims=2),
										 max_lats = maximum(cell_vertices.lats,dims=2),
							    	 min_lons = minimum(cell_vertices.lons,dims=2),
							    	 max_lons = maximum(cell_vertices.lons,dims=2))
		is_wrapped_cell::Array{Bool} = fill(false,size(cell_indices))
		for i = 1:size(cell_vertices.lons,1)
			if abs(cell_vertices.lons[i,3] - cell_vertices.lons[i,2]) > 180.0 ||
				 abs(cell_vertices.lons[i,3] - cell_vertices.lons[i,1]) > 180.0
				is_wrapped_cell[i] = true
				if abs(cell_vertices.lons[i,3] - cell_vertices.lons[i,2]) > 180.0 &&
					 abs(cell_vertices.lons[i,3] - cell_vertices.lons[i,1]) > 180.0
					cell_extremes.min_lons[i] = cell_vertices.lons[i,3] > cell_vertices.lons[i,2] ?
																		  cell_vertices.lons[i,3] :
																		  min(cell_vertices.lons[i,2],cell_vertices.lons[i,1])
					cell_extremes.max_lons[i] = cell_vertices.lons[i,3] > cell_vertices.lons[i,2] ?
																		  max(cell_vertices.lons[i,2],cell_vertices.lons[i,1]) :
																		  cell_vertices.lons[i,3]
				elseif abs(cell_vertices.lons[i,2] - cell_vertices.lons[i,3]) > 180.0 &&
				 	     abs(cell_vertices.lons[i,2] - cell_vertices.lons[i,1]) > 180.0
				 	cell_extremes.min_lons[i] = cell_vertices.lons[i,2] > cell_vertices.lons[i,3] ?
																		  cell_vertices.lons[i,2] :
																		  min(cell_vertices.lons[i,3],cell_vertices.lons[i,1])
					cell_extremes.max_lons[i] = cell_vertices.lons[i,2] > cell_vertices.lons[i,3] ?
																		  max(cell_vertices.lons[i,3],cell_vertices.lons[i,1]) :
																		  cell_vertices.lons[i,2]
				elseif abs(cell_vertices.lons[i,1] - cell_vertices.lons[i,3]) > 180.0 &&
					     abs(cell_vertices.lons[i,1] - cell_vertices.lons[i,2]) > 180.0
				 	cell_extremes.min_lons[i] = cell_vertices.lons[i,1] > cell_vertices.lons[i,3] ?
																		  cell_vertices.lons[i,1] :
																		  min(cell_vertices.lons[i,3],cell_vertices.lons[i,2])
					cell_extremes.max_lons[i] = cell_vertices.lons[i,1] > cell_vertices.lons[i,3] ?
																		  max(cell_vertices.lons[i,3],cell_vertices.lons[i,2]) :
																		  cell_vertices.lons[i,1]
				else
  				throw(UserError())
  			end
			end
		end
		return new(cell_indices,cell_neighbors,cell_coords,
		           restructured_cell_vertices,is_wrapped_cell,
		           cell_extremes)
	end
end

const Line = @NamedTuple{start_point::@NamedTuple{lat::Float64,
								     															lon::Float64},
							 			 			 end_point::@NamedTuple{lat::Float64,
								     															lon::Float64}}

struct RiverDelta
	name::String
	lines::Vector{Vector{Line}}
	function RiverDelta(name::String,
										  lines_in::Vector{Vector{Vector{Float64}}})
		lines::Vector{Vector{Line}} = Vector{Line}[]
		for line in lines_in
			line_sections::Vector{Line} = Line[]
			for i = 1:length(line)-1
				push!(line_sections,(start_point=(lat=line[i][1],lon=line[i][2]),
				 						 				 end_point=(lat=line[i+1][1],lon=line[i+1][2])))
			end
			push!(lines,line_sections)
		end
		new(name,lines)
	end
end

function find_cells_on_line_section(line_section::@NamedTuple{start_point::
																														  @NamedTuple{lat::Float64,
																																				  lon::Float64},
																															end_point::
																															@NamedTuple{lat::Float64,
																																			    lon::Float64}},
																		cells::Cells)
	min_lat::Float64 = min(line_section.start_point.lat,line_section.end_point.lat)
	max_lat::Float64 = max(line_section.start_point.lat,line_section.end_point.lat)
	min_lon::Float64 = min(line_section.start_point.lon,line_section.end_point.lon)
	max_lon::Float64 = max(line_section.start_point.lon,line_section.end_point.lon)
	local is_wrapped_line::Bool
	if abs(max_lon - min_lon) > 180.0
			new_min_lon::Float64 = max_lon
			max_lon = min_lon
			min_lon = new_min_lon
			is_wrapped_line = true
	else
			is_wrapped_line = false
	end
	filtered_cell_indices::Array{CartesianIndex} =
		filter(i -> check_if_line_section_passes_within_cell_extremes(
		              cells.cell_extremes.min_lats[i],
		             	cells.cell_extremes.max_lats[i],
						      cells.cell_extremes.min_lons[i],
						      cells.cell_extremes.max_lons[i],
						      cells.is_wrapped_cell[i],
						      min_lat,max_lat,min_lon,max_lon,
						      is_wrapped_line),
					 cells.cell_indices)
	cells_on_line_section::Array{CartesianIndex} = CartesianIndex[]
	for i in filtered_cell_indices
		#Note the map is over the set of vertices of a given triangle
		if check_if_line_section_intersects_cell(line_section,
																						 is_wrapped_line,
																						 map(cell_vertex_coords ->
									  												 	   (lat=cell_vertex_coords.lats[i],
									   															lon=cell_vertex_coords.lons[i]),
									  												     cells.cell_vertices),
									  												 cells.is_wrapped_cell[i])
			push!(cells_on_line_section,i)
		end
	end
	return cells_on_line_section
end

function check_if_line_section_passes_within_cell_extremes(cell_min_lat::Float64,
	                                                         cell_max_lat::Float64,
																													 cell_min_lon::Float64,
																													 cell_max_lon::Float64,
																													 is_wrapped_cell::Bool,
																													 line_min_lat::Float64,
																													 line_max_lat::Float64,
																													 line_min_lon::Float64,
																													 line_max_lon::Float64,
																													 is_wrapped_line::Bool)
	is_in_bounds::Bool = cell_max_lat > line_min_lat &&
		  						     cell_min_lat < line_max_lat
	if is_wrapped_line && is_wrapped_cell
		return is_in_bounds
	elseif is_wrapped_line || is_wrapped_cell
		is_in_bounds = is_in_bounds &&
									 (cell_max_lon > line_min_lon ||
									  cell_min_lon < line_max_lon)
	else
		is_in_bounds = is_in_bounds &&
									 cell_max_lon > line_min_lon &&
									 cell_min_lon < line_max_lon
	end
	return is_in_bounds
end

function check_if_line_section_intersects_cell(input_line_section::@NamedTuple{
																							 		start_point::@NamedTuple{lat::Float64,
										     																									 lon::Float64},
							 		 																end_point::@NamedTuple{lat::Float64,
									             																					 lon::Float64}},
									             								 is_wrapped_line::Bool,
					       															 input_cell_vertices::Array{
					       															 		@NamedTuple{lat::Float64,
					       															 								lon::Float64}},
					       															 is_wrapped_cell::Bool)
	local cell_vertices::Array{@NamedTuple{lat::Float64,lon::Float64}}
	local line_section::@NamedTuple{start_point::@NamedTuple{lat::Float64,
										     																	 lon::Float64},
							 		 								end_point::@NamedTuple{lat::Float64,
									             													 lon::Float64}}
	if is_wrapped_line
		line_section = (start_point = (lat = input_line_section.start_point.lat,
		                               lon = input_line_section.start_point.lon < 0.0 ?
		                               			 input_line_section.start_point.lon + 360.0 :
		                               			 input_line_section.start_point.lon),
										end_point = (lat = input_line_section.end_point.lat,
																 lon = input_line_section.end_point.lon < 0.0 ?
		                               		 input_line_section.end_point.lon + 360.0 :
		                               		 input_line_section.end_point.lon))
	else
		line_section = input_line_section
	end
	if is_wrapped_cell || is_wrapped_line
		if line_section.start_point.lon > 0.0 || is_wrapped_line
			cell_vertices = [ (lat = input_cell_vertices[i].lat,
									       lon = input_cell_vertices[i].lon < 0.0 ?
									             input_cell_vertices[i].lon + 360.0 :
									             input_cell_vertices[i].lon) for i = 1:3 ]
		else
			cell_vertices = [ (lat = input_cell_vertices[i].lat,
									       lon = input_cell_vertices[i].lon > 0.0 ?
									             input_cell_vertices[i].lon - 360.0 :
									             input_cell_vertices[i].lon) for i = 1:3 ]
		end
	else
		cell_vertices = input_cell_vertices
	end
	line_intersects_cell::Bool,divided_vertex_index::Int64 =
		check_if_line_intersects_cell(line_section,cell_vertices)
	if ! line_intersects_cell
		return false
	end
	if divided_vertex_index == 0
		return true
	end
	divided_vertex::@NamedTuple{lat::Float64,lon::Float64} =
		cell_vertices[divided_vertex_index]
	other_vertices::Array{@NamedTuple{lat::Float64,lon::Float64}} =
		[cell_vertices[i] for i=1:3 if i != divided_vertex_index]
	intersection_found::Bool = false
	for other_vertex in other_vertices
		if (point_line_determinant_sign(line_section.start_point.lat,
				           									line_section.start_point.lon,
                               	    other_vertex.lat,other_vertex.lon,
			       	       	   						divided_vertex.lat,divided_vertex.lon) !=
		    point_line_determinant_sign(line_section.end_point.lat,
				           									line_section.end_point.lon,
                               	    other_vertex.lat,other_vertex.lon,
			       	       	   						divided_vertex.lat,divided_vertex.lon))
			intersection_found = true
		end
	end
	return intersection_found
end

function check_if_line_intersects_cell(line::@NamedTuple{start_point::
							 																					 @NamedTuple{lat::Float64,
								     																					       lon::Float64},
							 																					 end_point::
							 																					 @NamedTuple{lat::Float64,
								     																								 lon::Float64}},
				       												 cell_vertices::Array{@NamedTuple{lat::Float64,
																																				lon::Float64}})
	norm_det_sum::Int64 = 0
	norm_dets::Array{Tuple{Int64,Int64}} = Int64[]
	for i = 1:3
		norm_det = point_line_determinant_sign(cell_vertices[i].lat,
				             	       							 cell_vertices[i].lon,
                               	           line.start_point.lat,
						      												 line.start_point.lon,
			       	       	     	       				 line.end_point.lat,
						       												 line.end_point.lon)
		if norm_det == 0
			return true,0
		end
		norm_det_sum += norm_det
		push!(norm_dets,(i,norm_det))
	end
	if abs(norm_det_sum) == 1
		return true, filter(nd -> nd[2] != sign(norm_det_sum),norm_dets)[1][1]
	else
		return false,-1
	end
end

function point_line_determinant_sign(px::Float64,py::Float64,lx1::Float64,ly1::Float64,
                                     lx2::Float64,ly2::Float64)
	return Int64(sign((px - lx1)*(ly2 - ly1) - (py - ly1)*(lx2 - lx1)))
end

function calculate_separation_measure(cell_index::CartesianIndex,cells::Cells,
			      	      									point::@NamedTuple{lat::Float64,lon::Float64})
	#Calculate (D/R)^2 (D = distance, R = Earths Radius) instead of D to reduce computation
	delta_lat::Float64 = cells.cell_coords.lats[cell_index] - point.lat 
	delta_lon::Float64 = cells.cell_coords.lons[cell_index] - point.lon 
	if delta_lon > 180.0
		delta_lon = cells.cell_coords.lons[cell_index] - 360.0 - point.lon
	elseif delta_lon < -180.0
		delta_lon = cells.cell_coords.lons[cell_index] + 360.0 - point.lon
	end
	return delta_lat^2 + (cos(0.5*(cells.cell_coords.lats[cell_index] + point.lat))*delta_lon)^2
end

function search_for_river_mouth_location_on_line_section(
																line_section::@NamedTuple{
																	start_point::@NamedTuple{lat::Float64,lon::Float64},
																	end_point::@NamedTuple{lat::Float64,lon::Float64}},
																cells::Cells,
																lsmask::Array{Bool},
							 									river_mouth_indices::Array{CartesianIndex})
	cells_on_line_section_indices::Array{CartesianIndex} = find_cells_on_line_section(line_section,cells)
	append!(cells_on_line_section_indices,
	        Set(CartesianIndex[cells.cell_neighbors[i,j] for i in cells_on_line_section_indices,j=1:3
	            if !(cells.cell_neighbors[i,j] in cells_on_line_section_indices) ]))
	sort!(cells_on_line_section_indices,by=x->calculate_separation_measure(x,cells,line_section.start_point))
	for cell_indices in cells_on_line_section_indices
		if lsmask[cell_indices]
			is_coastal_cell::Bool = false
			for neighbor in [cells.cell_neighbors[cell_indices,i] for i=1:3]
				is_coastal_cell = is_coastal_cell || ! lsmask[neighbor]	
			end
			if is_coastal_cell
				push!(river_mouth_indices,cell_indices)
				return true
			else
				error("Have reached the ocean without passing a coastal cell!")
				return false
			end
		end 
	end
	return false
end 

function identify_bifurcated_river_mouths(river_deltas::Array{RiverDelta},
																					cells::Cells,
																					lsmask::Array{Bool})
	river_mouth_indices_for_all_rivers::Dict{String,Array{CartesianIndex}} =
																			Dict{String,Array{CartesianIndex}}()
	for delta in river_deltas
		river_mouth_indices::Array{CartesianIndex} = Array{CartesianIndex}[]
		for line in delta.lines
			for line_section in line
				if search_for_river_mouth_location_on_line_section(line_section,
																													 cells,
																													 lsmask,
																													 river_mouth_indices)
					break
				end
			end
		end
		river_mouth_indices_for_all_rivers[delta.name] = river_mouth_indices
	end
	return river_mouth_indices_for_all_rivers
end

end
