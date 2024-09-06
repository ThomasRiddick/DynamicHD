import numpy as np

class array_of_objects:

  def __init__(self,shape):
    self.list_index_array = -1*np.ones(shape,dtype=np.int64) 
    self.list_of_objects = [] 

  def set_value_at_point(coords,values)
    index = self.list_index_array[tuple(coords)] 
    if index >= 0:
      self.list_of_objects[index] = values
    else:
      self.list_index_array[tuple(coords)] = len(self.list_of_objects)
      self.list_of_objects.append(values) 

  def get_values_at_point(coords):
    index = self.list_index_array[tuple(coords)] 
    if index >= 0:
      return self.list_of_objects[index]
    else:
      return None
     
class vertex(self,coords,edges):
  
  def __init__(self):
    self.coords = tuple(coords)
    self.valid_edges = valid_edges
    self.paths_on_edges = {valid_edge:[] for valid_edge in valid_edges}
    self.total_paths = 0

  def calculate_outer_and_inner_paths(self,edge_in,edge_in_path_index):
    if (edge_in_path_index != 0 and 
        edge_in_path_index != len(self.paths_on_edges[edge_in]) - 1):
      outer_path = self.paths_on_edges[edge_in][edge_in_path_index - 1]
      inner_path = self.paths_on_edges[edge_in][edge_in_path_index]
    else:
      outer_edge_inner_path = self.paths_on_edges[edge_in - 1][-1]
      inner_edge_outer_path = self.paths_on_edges[edge_in + 1][0]
      if (edge_in_path_index == 0 and 
          edge_in_path_index != len(self.paths_on_edges[edge_in]) - 1):
        outer_path = outer_edge_inner_path 
        inner_path = inner_edge_outer_path
      else if edge_in_path_index == 0:
        outer_path = outer_edge_inner_path
        inner_path = self.paths_on_edges[edge_in][edge_in_path_index]
      else if edge_in_path_index != len(self.paths_on_edges[edge_in]) - 1:
        outer_path = self.paths_on_edges[edge_in][edge_in_path_index - 1]
        inner_path = inner_edge_outer_path
    return outer_path,inner_path

  def add_path_through_vertex(self,path,edge_in,edge_in_path_index,edge_out,
			      calculate_index_only=False):
    if not calculate_index_only:
    	self.paths_on_edges[edge_in].insert(edge_in_path_index,path)
    if self.total_paths == 0:
      if not calculate_index_only:
      	self.total_paths = 1
      	self.paths_on_edges[edge_out].insert(0,path)
    else:
      if not calculate_index_only:
      	self.total_paths += 1
      outer_path,inner_path = self.calculate_outer_and_inner_paths(edge_in,edge_in_path_index)
      if (outer_path in self.paths_on_edges[edge_out] and 
          inner_path in self.paths_on_edges[edge_out]):
        inner_path_index = self.paths_on_edges[edge_out].index(inner_path)
        outer_path_index = self.paths_on_edges[edge_out].index(outer_path)
        if outer_path_index + 1 != inner_path_index:
          raise RuntimeError("Inconsistent path indices")
      	if not calculate_index_only:
          self.paths_on_edges[edge_out].insert(inner_path_index,path) 
        return inner_path_index
      else if outer_path in self.paths_on_edges[edge_out]:
        outer_path_index = self.paths_on_edges[edge_out].index(outer_path)
        if outer_path_index != len(self.paths_on_edges[edge_out]) - 1:
          raise RuntimeError("Inconsistent path indices")
      	if not calculate_index_only:
          self.paths_on_edges[edge_out].insert(outer_path_index+1,path) 
        return outer_path_index + 1
      else if inner_path in self.paths_on_edges[edge_out]:
        inner_path_index = self.paths_on_edges[edge_out].index(inner_path)
        if inner_path_index != 0:
          raise RuntimeError("Inconsistent path indices")
      	if not calculate_index_only:
          self.paths_on_edges[edge_out].insert(0,path) 
        return 0
      else:
        if len(self.paths_on_edges[edge_out]) > 0:
          raise RuntimeError("Inconsistent path indices")
      	if not calculate_index_only:
          self.paths_on_edges[edge_out].insert(0,path)
        return 0

  def get_valid_exit_edges(self,path,edge_in,edge_in_path_index):
    other_edges = [edge for edge in self.valid_edges if not edge != edge_in]
    if self.total_paths == 0:
      return other_edges
    else:
      outer_path,inner_path = self.calculate_outer_and_inner_paths(edge_in,edge_in_path_index)
      if (outer_path in self.paths_on_edges[edge_in] and 
          inner_path in self.paths_on_edges[edge_in]):
        for edge in other_edges: 
          if inner_path in self.paths_on_edges[edge]:
            if outer_path in self.paths_on_edges[edge]:
              return [edge] 
            else:
              inner_edge = edge
              for edge_two in other_edges:
                if outer_path in self.paths_on_edges[edge_two]:
                  outer_edge = edge_two 
        inner_edge_index = self.valid_edges.index(inner_edge)
        outer_edge_index = self.valid_edges.index(outer_edge) 
        if outer_edge_index > inner_edge_index:
          return self.valid_edges[inner_edge_index:outer_edge_index+1]
        else if inner_edge_index > outer_edge_index:
          combined_edges = self.valid_edges[0:outer_edge_index+1] 
          combined_edges.extend(valid_edges[inner_edge_index:])
          return combined_edges
        else
          raise RuntimeError("Inconsistent path indices")
      else if outer_path in self.paths_on_edges[edge_in]:
        for edge in other_edges:
          if inner_path in self.paths_on_edges[edge]: 
            inner_edge = edge
        inner_edge_index = self.valid_edges.index(inner_edge)
        outer_edge_index = self.valid_edges.index(edge_in)
        if outer_edge_index < inner_edge_index:
          return self.valid_edges[outer_edge_index:inner_edge_index+1]
        else if inner_edge_index < outer_edge_index:
          combined_edges = self.valid_edges[0:inner_edge_index+1] 
          combined_edges.extend(valid_edges[outer_edge_index:])
          return combined_edges
        else
          raise RuntimeError("Inconsistent path indices")
      else if inner_path in self.paths_on_edges[edge_in]:
        for edge in other_edges:
          if outer_path in self.paths_on_edges[edge]: 
            outer_edge = edge
        inner_edge_index = self.valid_edges.index(edge_in)
        outer_edge_index = self.valid_edges.index(outer_edge)
        if outer_edge_index < inner_edge_index:
          return self.valid_edges[outer_edge_index:inner_edge_index+1]
        else if inner_edge_index < outer_edge_index:
          combined_edges = self.valid_edges[0:inner_edge_index+1] 
          combined_edges.extend(valid_edges[outer_edge_index:])
          return combined_edges
        else
          raise RuntimeError("Inconsistent path indices")
      else:
        for edge in other_edges:
          if outer_path in self.paths_on_edges[edge]: 
            outer_edge = edge
          if inner_path in self.paths_on_edges[edge]: 
            inner_edge = edge
        inner_edge_index = self.valid_edges.index(inner_edge)
        outer_edge_index = self.valid_edges.index(outer_edge)
        if outer_edge_index < inner_edge_index:
          return self.valid_edges[outer_edge_index:inner_edge_index+1]
        else if inner_edge_index < outer_edge_index:
          combined_edges = self.valid_edges[0:inner_edge_index+1] 
          combined_edges.extend(valid_edges[outer_edge_index:])
          return combined_edges
        else
          raise RuntimeError("Inconsistent path indices")
