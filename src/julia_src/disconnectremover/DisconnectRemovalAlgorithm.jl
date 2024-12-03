module DisconnectRemovalALgorithm

struct ArrayOfObjects{T}
  list_index_array::Array{Int64}
  list_of_objects::Vector{T}
  function ArrayOfObjects{T}(dims::Tuple)
    list_index_array = fill(-1,dims)
    list_of_objects = []
    new(list_index_array,list_of_objects)
  end
end

function set_value_at_point(array::ArrayOfObjects{T},
                            coords::CartesianIndices,
                            values::T) where {T}
  index::Int64 = array.list_index_array[coords]
  if index >= 0
    array.list_of_objects[index] = values
  else
    array.list_index_array[coords] = length(array.list_of_objects)
    push!(array.list_of_objects,values)
  end
end

function get_values_at_point(array::ArrayOfObjects{T},
                             coords::CartesianIndices) where {T}
  index::Int64 = array.list_index_array[coords]
  if index >= 0:
    return array.list_of_objects[index]
  else:
    return nothing
end

struct Vertex(coords,edges)
  coords::CartesianIndices
  valid_edges::Vector{Int64}???
  paths_on_edges::Dict{????}
  total_paths::Int64
  function Vertex(coords::CartesianIndices,edges::???)
    paths_on_edges = {valid_edge:[] for valid_edge in valid_edges}
    new(coords,edges,paths_on_edges,0)
  end
end

function calculate_outer_and_inner_paths(vertex::Vertex,edge_in,edge_in_path_index)
  if edge_in_path_index != 1 &&
     edge_in_path_index != length(vertex.paths_on_edges[edge_in])
    outer_path::Int64 = vertex.paths_on_edges[edge_in][edge_in_path_index - 1]
    inner_path::Int64 = vertex.paths_on_edges[edge_in][edge_in_path_index]
  else
    outer_edge_inner_path = vertex.paths_on_edges[edge_in - 1][-1]
    inner_edge_outer_path = vertex.paths_on_edges[edge_in + 1][0]
    if edge_in_path_index == 1 &&
       edge_in_path_index != length(vertex.paths_on_edges[edge_in])
      outer_path = outer_edge_inner_path
      inner_path = inner_edge_outer_path
    elseif edge_in_path_index == 1
      outer_path = outer_edge_inner_path
      inner_path = vertex.paths_on_edges[edge_in][edge_in_path_index]
    elseif edge_in_path_index != length(vertex.paths_on_edges[edge_in])
      outer_path = vertex.paths_on_edges[edge_in][edge_in_path_index - 1]
      inner_path = inner_edge_outer_path
    end
  end
  return outer_path,inner_path
end

function add_path_through_vertex(vertex::Vertex,
                                 path,edge_in,edge_in_path_index,edge_out;
                                 calculate_index_only=False)
  if ! calculate_index_only
    vertex.paths_on_edges[edge_in].insert(edge_in_path_index,path) ???
  end
  if vertex.total_paths == 0
    if ! calculate_index_only
      vertex.total_paths = 1
      vertex.paths_on_edges[edge_out].insert(0,path)??? (0->1 offset change?)
    end
  else
    if ! calculate_index_only
      vertex.total_paths += 1
    end
    outer_path,inner_path = vertex.calculate_outer_and_inner_paths(edge_in,edge_in_path_index)
    if outer_path in vertex.paths_on_edges[edge_out] &&
       inner_path in vertex.paths_on_edges[edge_out]
      inner_path_index = vertex.paths_on_edges[edge_out].index(inner_path) ??
      outer_path_index = vertex.paths_on_edges[edge_out].index(outer_path) ??
      if outer_path_index + 1 != inner_path_index
        error("Inconsistent path indices")
      end
      if ! calculate_index_only
        vertex.paths_on_edges[edge_out].insert(inner_path_index,path) ???
      end
      return inner_path_index
    elseif outer_path in vertex.paths_on_edges[edge_out]
      outer_path_index = vertex.paths_on_edges[edge_out].index(outer_path) ???
      if outer_path_index != length(vertex.paths_on_edges[edge_out]) - 1
        error("Inconsistent path indices")
      end
      if ! calculate_index_only
        vertex.paths_on_edges[edge_out].insert(outer_path_index+1,path)
      end
      return outer_path_index + 1
    elseif inner_path in vertex.paths_on_edges[edge_out]
      inner_path_index = vertex.paths_on_edges[edge_out].index(inner_path)
      if inner_path_index != 0
        error("Inconsistent path indices")
      end
      if ! calculate_index_only
        vertex.paths_on_edges[edge_out].insert(0,path) ??? index offset
      end
      return 0
    else
      if length(vertex.paths_on_edges[edge_out]) > 0
        error("Inconsistent path indices") ???
      end
      if ! calculate_index_only
        vertex.paths_on_edges[edge_out].insert(0,path) ??? index offset
      end
      return 0
    end
end

function get_valid_exit_edges(vertex::Vertex,path,edge_in,edge_in_path_index)
  other_edges = [edge for edge in vertex.valid_edges if ! edge != edge_in] ??
  if vertex.total_paths == 0
    return other_edges
  else
    outer_path,inner_path = vertex.calculate_outer_and_inner_paths(edge_in,edge_in_path_index)
    if outer_path in vertex.paths_on_edges[edge_in] &&
       inner_path in vertex.paths_on_edges[edge_in]
      for edge in other_edges
        if inner_path in vertex.paths_on_edges[edge]
          if outer_path in vertex.paths_on_edges[edge]
            return [edge]
          else
            inner_edge = edge
            for edge_two in other_edges
              if outer_path in vertex.paths_on_edges[edge_two]
                outer_edge = edge_two
              end
            end
          end
        end
      end
      inner_edge_index = vertex.valid_edges.index(inner_edge)
      outer_edge_index = vertex.valid_edges.index(outer_edge)
      if outer_edge_index > inner_edge_index
        return vertex.valid_edges[inner_edge_index:outer_edge_index+1]
      else if inner_edge_index > outer_edge_index
        combined_edges = vertex.valid_edges[0:outer_edge_index+1]
        combined_edges.extend(valid_edges[inner_edge_index:])
        return combined_edges
      else
        error("Inconsistent path indices")
      end
    else if outer_path in vertex.paths_on_edges[edge_in]
      for edge in other_edges:
        if inner_path in vertex.paths_on_edges[edge]
          inner_edge = edge
        end
      end
      inner_edge_index = vertex.valid_edges.index(inner_edge)
      outer_edge_index = vertex.valid_edges.index(edge_in)
      if outer_edge_index < inner_edge_index
        return vertex.valid_edges[outer_edge_index:inner_edge_index+1]
      end
      else if inner_edge_index < outer_edge_index
        combined_edges = vertex.valid_edges[0:inner_edge_index+1]
        combined_edges.extend(valid_edges[outer_edge_index:])
        return combined_edges
      else
        error("Inconsistent path indices")
      end
    else if inner_path in vertex.paths_on_edges[edge_in]
      for edge in other_edges:
        if outer_path in vertex.paths_on_edges[edge]
          outer_edge = edge
        end
      end
      inner_edge_index = vertex.valid_edges.index(edge_in)
      outer_edge_index = vertex.valid_edges.index(outer_edge)
      if outer_edge_index < inner_edge_index
        return vertex.valid_edges[outer_edge_index:inner_edge_index+1]
      else if inner_edge_index < outer_edge_index
        combined_edges = vertex.valid_edges[0:inner_edge_index+1]
        combined_edges.extend(valid_edges[outer_edge_index:])
        return combined_edges
      else
        error("Inconsistent path indices")
      end
    else
      for edge in other_edges
        if outer_path in vertex.paths_on_edges[edge]
          outer_edge = edge
        end
        if inner_path in vertex.paths_on_edges[edge]
          inner_edge = edge
        end
      end
      inner_edge_index = vertex.valid_edges.index(inner_edge)
      outer_edge_index = vertex.valid_edges.index(outer_edge)
      if outer_edge_index < inner_edge_index
        return vertex.valid_edges[outer_edge_index:inner_edge_index+1]
      else if inner_edge_index < outer_edge_index
        combined_edges = vertex.valid_edges[0:inner_edge_index+1]
        combined_edges.extend(valid_edges[outer_edge_index:])
        return combined_edges
      else
        error("Inconsistent path indices")
      end
    end
  end
end

function process_neighbors(q::Vector{CartesianIndex},
                           vertex::Vertex,
                           visited_vertices)
  valid_edges = get_valid_exit_edges(vertex::Vertex,path,
                                     edge_in,
                                     edge_in_path_index)
  filter!(e->!visited_vertices[e],valid_edges)
  append!(q,valid_edges)
  for valid_edge in valid_edges
    visited_vertices[valid_edge] = true
  end
end

DO WE NEED EDGE OBJECTS??

disconnect_source_coords::CartesianIndex
disconnect_sink_coords::CartesianIndex
visited_vertices::Array
target_vertices::Array
q = Vector{CartesianIndex}
sizehint!(q,grid.ncells*vertices_per_cell)
add_disconnect_source_corners_to_q(q,disconnect_source_coords)
add_disconnect_sink_corners_to_target_verties(q,target_vertex)
first_vertex = popfirst!(q)
add_source_at_vertex(first_vertex,edge_in)
visited_vertices[first_vertex] = true
process_neighbors(q,first_vertex,visited_vertices)
while ! q.size()
  vertex = popfirst!(q)
  edge_out = get_edge_out_from_old_vertex_to_vertex(edge_out)
  add_path_through_vertex(old_vertex::Vertex,
                          path_num,old_edge_in,
                          old_edge_in_path_index,
                          edge_out)
  old_vertex = vertex
  old_edge_in = edge_in
  old_edge_in_path_index = edge_in_path_index
end

Expanding edges priority
- in same catchment
- has no inflows
- other using same path finding on small scale for introducing reconnect for other
  catchment

for other
  new disconnect from upstream to downstream

end
