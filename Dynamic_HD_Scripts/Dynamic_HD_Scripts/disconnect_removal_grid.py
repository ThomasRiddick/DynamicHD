'''
Created on Apr 14, 2017

@author: thomasriddick
'''

import grid
from abc import abstractmethod
import numpy as np

class DisconnectRemovalGrid(grid.Grid):
    '''
    Public Methods:
    return_coords_of_differences
    get_value_at_coords
    return_target_coords_of_index_based_rdirs
    set_target_coords_of_index_based_rdirs
    progress_made
    distance_between_points_squared
    increment_value
    get_neighbors_flowing_to_point
    get_downstream_neighbor
    get_neighbors
    set_value_at_coords
    attempt_direct_reconnect
    attempt_extended_direct_reconnect
    attempt_indirect_reconnect
    points_neighbor
    coords_as_string
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        
    @abstractmethod
    def return_coords_of_differences(self,first_rdirs_field,second_rdirs_field):
        pass
    
    @abstractmethod
    def get_value_at_coords(self,data,coords):
        pass
    
    @abstractmethod
    def return_target_coords_of_index_based_rdirs(self,rdirs_field,input_coords):
        pass
    
    @abstractmethod
    def set_target_coords_of_index_based_rdirs(self,rdirs_field,input_coords,target_coords):
        pass
    
    def progress_made(self,neighbor_coords,point_coords,target_coords):
        return (self.distance_between_points_squared(point_coords,target_coords) -
                self.distance_between_points_squared(neighbor_coords,target_coords))

    @abstractmethod
    def distance_between_points_squared(self,pointone,pointtwo):
        pass
    
    @abstractmethod    
    def increment_value(self,data,coords):
        pass
    
    @abstractmethod    
    def get_neighbors_flowing_to_point(self,rdirs,point):
        pass
    
    @abstractmethod    
    def get_downstream_neighbor(self,rdirs,point):
        pass
    
    @abstractmethod    
    def get_neighbors(self,point):
        pass
    
    @abstractmethod
    def set_value_at_coords(self,data,point,value):
        pass
    
    @abstractmethod
    def attempt_direct_reconnect(self,disconnected_neighbor,downstream_neighbor,rdirs):
        pass
    
    @abstractmethod
    def attempt_extended_direct_reconnect(self,disconnected_neighbor,reconnected_points,
                                          new_cotat_catchments,cotat_rdirs):
        pass
    
    @abstractmethod
    def attempt_indirect_reconnect(self,disconnected_neighbor,new_cotat_catchments,cotat_rdirs):
        pass
    
    @abstractmethod
    def points_neighbor(self,point_one,point_two):
        pass
    
    @abstractmethod
    def coords_as_string(self,coords):
        pass
        
class LatLongDisconnectRemovealGrid(DisconnectRemovalGrid,grid.LatLongGrid):
    """ 
    Public Methods:
    """
    
    def return_coords_of_differences(self,first_rdirs_field,second_rdirs_field):
        return np.argwhere((first_rdirs_field[:,:,0] != second_rdirs_field[:,:,0]) | 
                           (first_rdirs_field[:,:,1] != second_rdirs_field[:,:,1]))
        
    def get_value_at_coords(self,data,coords):
        return self.data[coords]
    
    def return_target_coords_of_index_based_rdirs(self,rdirs_field,input_coords):
        return rdirs_field[input_coords + [0]],rdirs_field[input_coords + [1]]
    
    def set_target_coords_of_index_based_rdirs(self,rdirs_field,input_coords,target_coords):
        rdirs_field[input_coords + [0]] = target_coords[0]
        rdirs_field[input_coords + [1]] = target_coords[1]
        
    def distance_between_points_squared(self,pointone,pointtwo):
        return (pointone[0] - pointtwo[0])**2 + (pointone[1] - pointtwo[1])**2
    
    def increment_value(self,data,coords):
        data[coords] += 1
        
    def get_neighbors_flowing_to_point(self,rdirs,point):
        neighbors_flowing_to_point = []
        for neighbor in self.get_neighbors(point):
            if (rdirs[neighbor+[0]] == point[0] and
                rdirs[neighbor+[1]] == point[1]):
                neighbors_flowing_to_point.append(neighbor)

    def get_downstream_neighbor(self,rdirs,point):
        return rdirs[point+[0]],rdirs[point+[1]]
    
    def get_neighbors(self,point):
        return [(i + point[0],j + point[1]) for i in range(-1,2) for j in range(-1,2) if not (i==0 and j==0)]

    def set_value_at_coords(self,data,point,value):
        data[point] = value
        
    def attempt_direct_reconnect(self,disconnected_neighbor,downstream_neighbor,rdirs):
        if self.points_neighbor(disconnected_neighbor,downstream_neighbor):
            self.set_target_coords_of_index_based_rdirs(rdirs,disconnected_neighbor,
                                                        downstream_neighbor)
            return True 
        else:
            return False
    
    def attempt_extended_direct_reconnect(self,disconnected_neighbor,reconnected_points,
                                          new_cotat_catchments,cotat_rdirs):
        for neighbor_of_neighbor in self.get_neighbors(disconnected_neighbor):
            if (reconnected_points[neighbor_of_neighbor] and
                new_cotat_catchments[neighbor_of_neighbor] == 
                new_cotat_catchments[disconnected_neighbor]): 
                self.set_target_coords_of_index_based_rdirs(cotat_rdirs,
                                                            disconnected_neighbor,
                                                            neighbor_of_neighbor)
                return True
        return False
    
    def attempt_indirect_reconnect(self,disconnected_neighbor,new_cotat_catchments,cotat_rdirs):
        for neighbor_of_neighbor in self.get_neighbors(disconnected_neighbor):
            if (new_cotat_catchments[neighbor_of_neighbor] == 
                new_cotat_catchments[disconnected_neighbor]):
                self.set_target_coords_of_index_based_rdirs(cotat_rdirs,
                                                            disconnected_neighbor,
                                                            neighbor_of_neighbor)
                return True
        return False
    
    def points_neighbor(self,point_one,point_two):
        for value in [p1_index - p2_index for p1_index,p2_index in zip(point_one,point_two)]:
            if abs(value) > 1:
                return False
        return True
    
    def coords_as_string(self,coords):
        return "i: {0} j: {1}".format(*coords)