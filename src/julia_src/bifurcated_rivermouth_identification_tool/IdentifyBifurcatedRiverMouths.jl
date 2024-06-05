module IdentifyBifurcatedRiverMouths

struct Cells
	cell_indices::Array{CartesianIndices}
	cell_neighbors::Array{Array{CartesianIndices}}
	cell_coords::@NamedTuple{lats::Array{Float64},lons::Array{Float64}}
	cell_vertices::Array{@NamedTuple{lat::Array{Float64},lon::Array{Float64}}}	
	cell_extremes::@NamedTuple{min_lats::Array{Float64},max_lats::Array{Float64},
				   min_lons::Array{Float64},max_lons::Array{Float64}}
end


#LOAD A TOML FILE WITH TREES DEFINING LINES ENININATING FROM POINTS FAR INLAND DOWN THE DEEPEST CHANNELS FORKING OUTWARDS

function point_line_determinant_sign(px::Float64,py::Float64,lx1::Float64,ly1::Float64,
                                     lx2::Float64,ly2::Float64)
	return Int64(sign((px - lx1)*(ly2 - ly1) - (py - ly1)*(lx2 - lx1)))
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

function check_if_line_section_intersects_cell(line_section::@NamedTuple{start_point::
									 @NamedTuple{lat::Float64,
										     lon::Float64},
							 		 end_point::
									 @NamedTuple{lat::Float64,
									             lon::Float64}},
					       cell_vertices::Array{@NamedTuple{lat::Float64,
										lon::Float64}})
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

function find_cells_on_line_section(line_section::NamedTuple{NamedTuple{Int64,Int64}},cells::Cells)
	min_lat::Float64 = mininum(line_section.start_point.lat,line_section.end_point.lat)
	max_lat::Float64 = maximum(line_section.start_point.lat,line_section.end_point.lat)
	min_lon::Float64 = mininum(line_section.start_point.lon,line_section.end_point.lon)
	max_lon::Float64 = maximum(line_section.start_point.lon,line_section.end_point.lon)
	filtered_cell_indices::Array{CartesianIndices} = filter(i -> cell.cell_extremes.max_lats[i] > min_lat &&
						      	     	     cell.cell_extremes.min_lats[i] < max_lat && 
						      	     	     cell.cell_extremes.max_lons[i] > min_lon &&
						      	     	     cell.cell_extremes.min_lons[i] < max_lon, 
								cells.cell_indices)
	cells_on_line_section::Array{CartesianIndices} = [] 
	for i in filtered_cell_indices
		if check_if_line_section_intersects_cell(line_section,map(cell_vertex_coords -> 
									  (lat=cell_vertex_coords.lat[i],
									   lon=cell_vertex_coords.lon[i]), 
									  cells.cell_vertices))
			push!(cells_on_line_section,i)
		end
	end
	return cells_on_line_section
end

function calculate_separation_measure(cell_index::CartesianIndices,cells::Cells,	
			      	      point::@NamedTuple{lat:Float64,lon::Float64})
	#Calculate (D/R)^2 (D = distance, R = Earths Radius) instead of D to reduce computation
	delta_lat::Float64 = cells.cell_coords.lats[cell_index] - point.lat 
	delta_lon::Float64 = cells.cell_coords.lons[cell_index] - point.lon 
	return delta_lat^2 + (cos(0.5*(cells.cell_coords.lats[cell_index] + point.lat))*delta_lon)^2
end

function search_for_river_mouth_location_on_line_section(line_section::NamedTuple{NamedTuple{Int64,Int64}},cells::Array{Cells},
							 river_mouth_indices::Array{CartesianIndices})
	cells_on_line_section_indices::Array{CartesianIndices} = find_cells_on_line_section(line_section,cells)
	append!(cells_on_line_section_indices,[cells.cell_neighbors[i][j] for i=1:3,j in cells_on_line_section_indices])
	sort!(cells_on_line_section_indices,by=x->calculate_separation_measure(x,cells,line_section.start_point))
	for cell_indices in cells_on_line_section_indices
		if lsmask[cell_indices]
			is_coastal_cell::Bool = false
			for neighbor in [cells.cell_neighbors[i][cell_indices] for i=1:3] 
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

end
