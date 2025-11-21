module IdentifyBifurcatedRiverMouths

CartesianIndexOrNothing = Union{CartesianIndex,Nothing}

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
                        if cell_vertices.lats[i,1] > 85.0 || cell_vertices.lats[i,1] < -80.0
        is_wrapped_cell[i] = false
      elseif abs(cell_vertices.lons[i,3] - cell_vertices.lons[i,2]) > 180.0 ||
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
            error()
          end
      end
    end
    return new(cell_indices,cell_neighbors,cell_coords,
               restructured_cell_vertices,is_wrapped_cell,
               cell_extremes)
  end
end

#Often referred to as a 'line section' below, i.e. a single element
#of a multi section line
const Line = @NamedTuple{start_point::@NamedTuple{lat::Float64,
                                                  lon::Float64},
                           end_point::@NamedTuple{lat::Float64,
                                                  lon::Float64}}

struct RiverDelta
  name::String
  reverse_search::Bool
  lines::Vector{Vector{Line}}
  function RiverDelta(name::String,
                      reverse_search::Bool,
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
    new(name,reverse_search,lines)
  end
end

function find_cells_on_line_section(line_section::Line,
                                    cells::Cells,
                                    previous_section_cells_on_line::Array{CartesianIndex})
  displacement::Float64 = 0.001
  displaced_line_section::Line = (start_point=(lat=line_section.start_point.lat+displacement,
                                               lon=line_section.start_point.lon+displacement),
                                  end_point=(lat=line_section.end_point.lat+displacement,
                                             lon=line_section.end_point.lon+displacement))
  min_lon::Float64 = min(line_section.start_point.lon,line_section.end_point.lon)
  max_lon::Float64 = max(line_section.start_point.lon,line_section.end_point.lon) + displacement
  local is_wrapped_line::Bool
  if abs(max_lon - min_lon) > 180.0
      is_wrapped_line = true
  else
      is_wrapped_line = false
  end
  cells_on_line_section::Array{CartesianIndex} = CartesianIndex[]
  q::Vector{CartesianIndex} = CartesianIndex[]
  completed_cells::BitArray = falses(size(cells.cell_indices))
  initial_cell::CartesianIndex =
    find_cell_containing_point(line_section.start_point.lat,
                               line_section.start_point.lon,cells,
                               previous_section_cells_on_line)
  displaced_initial_cell::CartesianIndex =
    find_cell_containing_point(displaced_line_section.start_point.lat,
                               displaced_line_section.start_point.lon,cells,
                               previous_section_cells_on_line)
  push!(cells_on_line_section,initial_cell)
  completed_cells[initial_cell] = true
  if displaced_initial_cell != initial_cell
    completed_cells[displaced_initial_cell] = true
    push!(cells_on_line_section,displaced_initial_cell)
  end
  add_unprocessed_neighbors_to_q(q,completed_cells,initial_cell,cells)
  if displaced_initial_cell != initial_cell
    add_unprocessed_neighbors_to_q(q,completed_cells,displaced_initial_cell,cells)
  end
  while length(q) > 0
    i = pop!(q)
    #Note the map is over the set of vertices of a given triangle
    if check_if_line_section_intersects_cell(line_section,
                                             is_wrapped_line,
                                             map(cell_vertex_coords::
                                                 @NamedTuple{lats::Array{Float64},
                                                             lons::Array{Float64}} ->
                                                 (lat=cell_vertex_coords.lats[i],
                                                  lon=cell_vertex_coords.lons[i]),
                                                 cells.cell_vertices),
                                             cells.is_wrapped_cell[i]) ||
       check_if_line_section_intersects_cell(displaced_line_section,
                                             is_wrapped_line,
                                             map(cell_vertex_coords::
                                                 @NamedTuple{lats::Array{Float64},
                                                             lons::Array{Float64}} ->
                                                 (lat=cell_vertex_coords.lats[i],
                                                  lon=cell_vertex_coords.lons[i]),
                                                 cells.cell_vertices),
                                             cells.is_wrapped_cell[i])
      push!(cells_on_line_section,i)
      add_unprocessed_neighbors_to_q(q,completed_cells,i,cells)
    end
  end
  return cells_on_line_section
end

function add_unprocessed_neighbors_to_q(q::Vector{CartesianIndex},
                                        completed_cells::BitArray,
                                        center_cell::CartesianIndex,
                                        cells::Cells)
  for_all_neighbors(center_cell,cells.cell_neighbors) do neighbor_indices::CartesianIndex
    if ! completed_cells[neighbor_indices]
      push!(q,neighbor_indices)
      completed_cells[neighbor_indices] = true
    end
  end
end


function check_if_line_section_intersects_cell(input_line_section::Line,
                                               is_wrapped_line::Bool,
                                               input_cell_vertices::Array{
                                                  @NamedTuple{lat::Float64,
                                                              lon::Float64}},
                                               is_wrapped_cell::Bool)
  local cell_vertices::Array{@NamedTuple{lat::Float64,lon::Float64}}
  local line_section::Line
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
  if is_wrapped_cell
    if line_section.start_point.lon > 0.0
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
  elseif is_wrapped_line
    min_lon::Float64 = minimum([input_cell_vertices[i].lon for i = 1:3 ])
    if min_lon < 0
      cell_vertices = [ (lat = input_cell_vertices[i].lat,
                         lon = input_cell_vertices[i].lon + 360.0) for i = 1:3 ]
    else
      cell_vertices = input_cell_vertices
    end
  else
    cell_vertices = input_cell_vertices
  end
  line_intersects_cell::Bool,divided_vertex_index::Int64,
    intersects_cell_vertex::Bool =
    check_if_line_intersects_cell(line_section,cell_vertices)
  if ! line_intersects_cell
    return false
  end
  local other_vertices::Array{@NamedTuple{lat::Float64,lon::Float64}}
  if intersects_cell_vertex
    line_min_lat::Float64 = min(line_section.start_point.lat,
                                line_section.end_point.lat)
    line_max_lat::Float64 = max(line_section.start_point.lat,
                                line_section.end_point.lat)
    line_min_lon::Float64 = min(line_section.start_point.lon,
                                line_section.end_point.lon)
    line_max_lon::Float64 = max(line_section.start_point.lon,
                                line_section.end_point.lon)
    if divided_vertex_index < 0
      #Line exactly parallel to cell edge
      parallel_vertices::Array{@NamedTuple{lat::Float64,lon::Float64}} =
        [cell_vertices[i] for i=1:3 if i != -divided_vertex_index]
      edge_min_lat::Float64 = min(parallel_vertices[1].lat,
                                  parallel_vertices[2].lat)
      edge_max_lat::Float64 = max(parallel_vertices[1].lat,
                                  parallel_vertices[2].lat)
      edge_min_lon::Float64 = min(parallel_vertices[1].lon,
                                  parallel_vertices[2].lon)
      edge_max_lon::Float64 = max(parallel_vertices[1].lon,
                                  parallel_vertices[2].lon)
      return (edge_min_lat <= line_max_lat &&
              edge_max_lat >= line_min_lat &&
              edge_min_lon <= edge_max_lon &&
              edge_max_lon >= line_min_lon)
    else
      #Line intersects one vertex
      intersected_cell_vertex::@NamedTuple{lat::Float64,lon::Float64} =
        cell_vertices[divided_vertex_index]
      if line_min_lat <= intersected_cell_vertex.lat &&
         line_max_lat >= intersected_cell_vertex.lat &&
         line_min_lon <= intersected_cell_vertex.lon &&
         line_max_lon >= intersected_cell_vertex.lon
        #Line section reaches that vertex
         return true
      else
        #Does line section pass between other two vertices?
        other_vertices =
          [cell_vertices[i] for i=1:3 if i != divided_vertex_index]
        if point_line_determinant_sign(line_section.start_point.lat,
                                       line_section.start_point.lon,
                                       other_vertices[1].lat,other_vertices[1].lon,
                                       other_vertices[2].lat,other_vertices[2].lon) !=
           point_line_determinant_sign(line_section.end_point.lat,
                                       line_section.end_point.lon,
                                       other_vertices[1].lat,other_vertices[1].lon,
                                       other_vertices[2].lat,other_vertices[2].lon)
          local line_vertex_in_opposite_plane::@NamedTuple{lat::Float64,lon::Float64}
          if point_line_determinant_sign(line_section.start_point.lat,
                                         line_section.start_point.lon,
                                         other_vertices[1].lat,other_vertices[1].lon,
                                         other_vertices[2].lat,other_vertices[2].lon) !=
             point_line_determinant_sign(intersected_cell_vertex.lat,
                                         intersected_cell_vertex.lon,
                                         other_vertices[1].lat,other_vertices[1].lon,
                                         other_vertices[2].lat,other_vertices[2].lon)
             line_vertex_in_opposite_plane = line_section.start_point
          else
             line_vertex_in_opposite_plane = line_section.end_point
          end
          return point_line_determinant_sign(line_vertex_in_opposite_plane.lat,
                                             line_vertex_in_opposite_plane.lon,
                                             other_vertices[1].lat,other_vertices[1].lon,
                                             intersected_cell_vertex.lat,
                                             intersected_cell_vertex.lon) !=
                 point_line_determinant_sign(line_vertex_in_opposite_plane.lat,
                                             line_vertex_in_opposite_plane.lon,
                                             other_vertices[2].lat,other_vertices[2].lon,
                                             intersected_cell_vertex.lat,
                                             intersected_cell_vertex.lon)
        else
          return false
        end
      end
    end
  else
    divided_vertex::@NamedTuple{lat::Float64,lon::Float64} =
      cell_vertices[divided_vertex_index]
    other_vertices =
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
end

function check_if_line_intersects_cell(line::Line,
                                       cell_vertices::Array{@NamedTuple{lat::Float64,
                                                                        lon::Float64}})
  norm_det_sum::Int64 = 0
  norm_dets::Array{Tuple{Int64,Int64}} = Int64[]
  det_zero_count::Int64 = 0
  zero_det_index::Int64 = -1
  non_zero_det_index::Int64 = -1
  for i = 1:3
    norm_det = point_line_determinant_sign(cell_vertices[i].lat,
                                           cell_vertices[i].lon,
                                           line.start_point.lat,
                                           line.start_point.lon,
                                           line.end_point.lat,
                                           line.end_point.lon)
    if norm_det == 0
      det_zero_count += 1
      zero_det_index = i
    else
      non_zero_det_index = i
    end
    norm_det_sum += norm_det
    push!(norm_dets,(i,norm_det))
  end
  if det_zero_count == 2
    return true,-non_zero_det_index,true
  elseif det_zero_count == 1
    return true,zero_det_index,true
  end
  if abs(norm_det_sum) == 1
    return true, filter(nd::Tuple{Int64,Int64} ->
                        nd[2] != sign(norm_det_sum),norm_dets)[1][1],false
  else
    return false,0,false
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

function for_all_secondary_neighbors(function_on_neighbor::Function,
                                     cell_indices::CartesianIndex,
                                     cell_neighbors::Array{CartesianIndex})
  for i=1:3
    neighbor::CartesianIndex = cell_neighbors[cell_indices,i]
    for j=1:3
      secondary_neighbor::CartesianIndex = cell_neighbors[neighbor,j]
      if secondary_neighbor == cell_indices
        continue
      end
      function_on_neighbor(secondary_neighbor)
      for k=1:3
        tertiary_neighbor::CartesianIndex = cell_neighbors[secondary_neighbor,k]
        if tertiary_neighbor == neighbor
          continue
        end
        for l=1:3
          other_neighbor::CartesianIndex = cell_neighbors[cell_indices,l]
          if i >= l
            continue
          end
          for m=1:3
            other_secondary_neighbor::CartesianIndex = cell_neighbors[other_neighbor,m]
            for n=1:3
              other_tertiary_neighbor::CartesianIndex = cell_neighbors[other_secondary_neighbor,n]
              if other_tertiary_neighbor == other_neighbor
                continue
              end
              if other_tertiary_neighbor == tertiary_neighbor
                function_on_neighbor(tertiary_neighbor)
              end
            end
          end
        end
      end
    end
  end
end

function for_all_neighbors(function_on_neighbor::Function,
                           cell_indices::CartesianIndex,
                           cell_neighbors::Array{CartesianIndex})
  for neighbor in [cell_neighbors[cell_indices,i] for i=1:3]
    function_on_neighbor(neighbor)
  end
  for_all_secondary_neighbors(function_on_neighbor,
                              cell_indices,
                              cell_neighbors)
end

function find_cell_containing_point(point_lat::Float64,point_lon::Float64,cells::Cells,
                                    previous_section_cells_on_line::Array{CartesianIndex})
  local cell_indices::Array{CartesianIndex}
  if length(previous_section_cells_on_line) > 0
    cell_indices = CartesianIndex[]
    append!(cell_indices,previous_section_cells_on_line)
    for center_cell_indices in previous_section_cells_on_line
      for_all_neighbors(center_cell_indices,
                        cells.cell_neighbors) do neighbor_indices::CartesianIndex
        push!(cell_indices,neighbor_indices)
      end
    end
  else
    cell_indices =
      filter(i::CartesianIndex ->
             check_if_point_is_within_cell_extremes(
                cells.cell_extremes.min_lats[i],
                cells.cell_extremes.max_lats[i],
                cells.cell_extremes.min_lons[i],
                cells.cell_extremes.max_lons[i],
                cells.is_wrapped_cell[i],
                point_lat,point_lon),
             cells.cell_indices)
  end
  for i in cell_indices
    cell_vertices::Array{@NamedTuple{lat::Float64,
                                     lon::Float64}} =
      map(cell_vertex_coords::
          @NamedTuple{lats::Array{Float64},
          lons::Array{Float64}} ->
          (lat=cell_vertex_coords.lats[i],
           lon=cell_vertex_coords.lons[i]),
          cells.cell_vertices)
    if cells.is_wrapped_cell[i]
      if point_lon > 0.0
        modified_cell_vertices = [ (lat = cell_vertices[i].lat,
                                    lon = cell_vertices[i].lon < 0.0 ?
                                    cell_vertices[i].lon + 360.0 :
                                    cell_vertices[i].lon) for i = 1:3 ]
      else
        modified_cell_vertices = [ (lat = cell_vertices[i].lat,
                                    lon = cell_vertices[i].lon > 0.0 ?
                                    cell_vertices[i].lon - 360.0 :
                                    cell_vertices[i].lon) for i = 1:3 ]
      end
    else
      modified_cell_vertices = cell_vertices
    end
    if check_if_point_is_in_cell(point_lat,point_lon,modified_cell_vertices)
      return i
    end
  end
  error("No cell containing point found")
  return 0
end

function check_if_point_is_within_cell_extremes(cell_min_lat::Float64,
                                                cell_max_lat::Float64,
                                                cell_min_lon::Float64,
                                                cell_max_lon::Float64,
                                                is_wrapped_cell::Bool,
                                                point_lat::Float64,
                                                point_lon::Float64)
  is_in_bounds::Bool = cell_max_lat >= point_lat &&
                       cell_min_lat <= point_lat
  if is_wrapped_cell
    is_in_bounds = is_in_bounds &&
                   (cell_max_lon >= point_lon ||
                    cell_min_lon <= point_lon)
  else
    is_in_bounds = is_in_bounds &&
                   cell_max_lon >= point_lon &&
                   cell_min_lon <= point_lon
  end
  return is_in_bounds
end

function check_if_point_is_in_cell(point_lat::Float64,point_lon::Float64,
                                   cell_vertices::Array{@NamedTuple{lat::Float64,
                                                                    lon::Float64}})
  norm_det_sum::Int64 = 0
  abs_norm_det_sum::Int64 = 0
  for i = 1:3
    j = i + 1
    if j > 3
      j = 1
    end
    norm_det = point_line_determinant_sign(point_lat,
                                           point_lon,
                                           cell_vertices[i].lat,
                                           cell_vertices[i].lon,
                                           cell_vertices[j].lat,
                                           cell_vertices[j].lon)
    norm_det_sum += norm_det
    abs_norm_det_sum += abs(norm_det)
  end
  if abs_norm_det_sum == 1
    #On a corner
    return true
  elseif abs_norm_det_sum == 2
    #On a line
    if abs(norm_det_sum) == 2
      return true
    else
      return false
    end
  elseif abs(norm_det_sum) == 3
    return true
  else
    return false
  end
end

function check_connection(cell_indices::CartesianIndex,
                          inland_cell_indices::CartesianIndex,
                          cells::Cells,
                          lsmask::Array{Bool})
  q::Vector{CartesianIndex} = CartesianIndex[cell_indices]
  completed_cells::BitArray = falses(size(lsmask))
  completed_cells[cell_indices] = true
  connection_found::Bool = false
  while length(q) > 0 && ! connection_found
    center_cell_indices = pop!(q)
    for_all_neighbors(center_cell_indices,
                      cells.cell_neighbors) do neighbor_indices::CartesianIndex
      if ! completed_cells[neighbor_indices]
        completed_cells[neighbor_indices] = true
        if inland_cell_indices == neighbor_indices
          connection_found = true
        end
        if ! lsmask[neighbor_indices]
          push!(q,neighbor_indices)
        end
      end
    end
  end
  return connection_found
end



function search_for_river_mouth_location_on_line_section(
                                line_section::Line,
                                cells::Cells,
                                lsmask::Array{Bool},
                                river_mouth_indices::Array{CartesianIndex},
                                #river_mouth_indices::SharedArray{CartesianIndex},
                                #line_index::Int64,
                                previous_section_cells_on_line::Array{CartesianIndex},
                                reverse_search::Bool=false,
                                inland_cell_indices::CartesianIndexOrNothing=nothing)
  cells_on_line_section_indices::Array{CartesianIndex} =
    find_cells_on_line_section(line_section,cells,previous_section_cells_on_line)
  if length(cells_on_line_section_indices) == 0
    return false
  end
  sort!(cells_on_line_section_indices,by=x->calculate_separation_measure(x,cells,line_section.start_point),
        rev=reverse_search)
  previous_cells_on_lines_section = cells_on_line_section_indices
  passing_disconnected_land_cells::Bool = false
  #For a reverse search need to check if we are starting on a
  #disconnected land cell
  if reverse_search
    cell_indices::CartesianIndex = cells_on_line_section_indices[1]
    if ! lsmask[cell_indices] &&
       ! check_connection(cell_indices,
                          inland_cell_indices,cells,lsmask)
      passing_disconnected_land_cells = true
    end
  end
  for cell_indices in cells_on_line_section_indices
    if lsmask[cell_indices]
      is_coastal_cell::Bool = false
      passing_disconnected_land_cells = false
      for neighbor in [cells.cell_neighbors[cell_indices,i] for i=1:3]
        is_coastal_cell = is_coastal_cell || ! lsmask[neighbor] 
      end
      if is_coastal_cell
        if reverse_search && ! check_connection(cell_indices,inland_cell_indices,
                                                cells,lsmask)
          passing_disconnected_land_cells = true
          continue
        end
        push!(river_mouth_indices,cell_indices)
        #river_mouth_indices[line_index] = cell_indices
        return true
      else
        is_secondary_coastal_cell::Bool = false
        for_all_secondary_neighbors(cell_indices,
            cells.cell_neighbors) do secondary_neighbor_indices::CartesianIndex
          is_secondary_coastal_cell = is_secondary_coastal_cell || ! lsmask[secondary_neighbor_indices]
        end
        if is_secondary_coastal_cell
          if reverse_search && ! check_connection(cell_indices,inland_cell_indices,
                                                  cells,lsmask)
            passing_disconnected_land_cells = true
            continue
          end
          push!(river_mouth_indices,cell_indices)
          #river_mouth_indices[line_index] = cell_indices
          return true
        elseif ! reverse_search
          error("Have reached the ocean without passing a coastal cell!")
          return false
        end
      end
    elseif reverse_search && ! passing_disconnected_land_cells
      error("Have reached the land without passing a coastal cell!")
      return false
    end 
  end
  return false
end 

function identify_bifurcated_river_mouths(river_deltas::Array{RiverDelta},
                                          cells::Cells,
                                          lsmask::Array{Bool})
  river_mouth_indices_for_all_rivers::Dict{String,Array{CartesianIndex}} =
                                      Dict{String,Array{CartesianIndex}}()
  # river_mouth_indices_for_all_rivers_array::Array{(String,Array{CartesianIndex})} =
  #   pmap(river_deltas) do delta::RiverDelta
  for delta in river_deltas
    println("Processing $(delta.name)")
    river_mouth_indices::Array{CartesianIndex} = Array{CartesianIndex}[]
    # river_mouth_indices::SharedArray{CartesianIndex} =
    #   SharedArray{CartesianIndex}((length(delta.lines)))
    local inland_cell_indices::CartesianIndexOrNothing
    if delta.reverse_search
      #Arbitarily choose the first line for finding the inland cell indices
      inland_cell_indices =
        find_cell_containing_point(delta.lines[1][1].start_point.lat,
                                   delta.lines[1][1].start_point.lon,
                                   cells,CartesianIndex[])
    else
      inland_cell_indices = nothing
    end
    #@sync @distributed for i,line in enumerate(delta.lines)
    for line in delta.lines
      local modified_line::Vector{Line}
      if delta.reverse_search
        modified_line = reverse(line)
      else
        modified_line = line
      end
      previous_section_cells_on_line::Array{CartesianIndex} = CartesianIndex[]
      for line_section in modified_line
        if search_for_river_mouth_location_on_line_section(line_section,
                                                           cells,
                                                           lsmask,
                                                           river_mouth_indices,
                                                           #i,
                                                           previous_section_cells_on_line,
                                                           delta.reverse_search,
                                                           inland_cell_indices)
          break
        end
      end
    end
    river_mouth_indices_for_all_rivers[delta.name] = river_mouth_indices
    #return (delta.name,river_mouth_indices)
  end
  # for (name,indices) in river_mouth_indices_for_all_rivers_array
  #   river_mouth_indices_for_all_rivers[name] = indices
  # end
  return river_mouth_indices_for_all_rivers
end

end
