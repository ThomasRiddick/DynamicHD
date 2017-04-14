'''
Created on Apr 14, 2017

@author: thomasriddick
'''

import grid
import remove_disconnects
from abc import abstractmethod
import numpy as np
import math

class DisconnectRemovalGrid(grid.Grid):
    '''
    Public Methods:
    return_coords_of_differences
    get_value_at_coords
    return_target_coords_of_index_based_rdirs
    set_target_coords_of_index_based_rdirs
    prep_path_generator
    compute_rerouting
    increment_value
    get_neighbors_flowing_to_point
    get_downstream_neighbor
    get_neighbors
    set_value_at_coords
    attempt_direct_reconnect
    attempt_extended_direct_reconnect
    attempt_indirect_reconnect
    points_neighbor
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
    
    @abstractmethod
    def prep_path_generator(self,disconnect,downstream_reconnection_target,disconnect_size,
                  cotat_disconnected_catchment_num,reroute_point_func,evaluate_rerouting_func,
                  rerouting_start_points_and_biases):
        pass 
    
    @abstractmethod
    def compute_rerouting(self,start_point,point_to_avoid,divert_right):
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
    
    def prep_path_generator(self,disconnect,downstream_reconnection_target,disconnect_size,
                            cotat_disconnected_catchment_num,reroute_point_func,
                            evaluate_rerouting_func,rerouting_start_points_and_biases):
        required_delta_i =  downstream_reconnection_target[0] - disconnect[0]
        required_delta_j =  downstream_reconnection_target[1] - disconnect[1] 
        initial_i = disconnect[0]
        initial_j = disconnect[1]
        if (abs(required_delta_i) > abs(required_delta_j)):
            next_path_step_generator = self.generate_next_step_on_path(initial_i,
                                                                       initial_j,
                                                                       required_delta_i, 
                                                                       required_delta_j,True,
                                                                       reroute_point_func,
                                                                       evaluate_rerouting_func,
                                                                       rerouting_start_points_and_biases)
        else:
            next_path_step_generator = self.generate_next_step_on_path(initial_i,
                                                                       initial_j,
                                                                       required_delta_j, 
                                                                       required_delta_i,False,
                                                                       reroute_point_func,
                                                                       evaluate_rerouting_func,
                                                                       rerouting_start_points_and_biases)
        return next_path_step_generator

    def update_path_generation_variables(self,initial_more_frequent_index,
                                         initial_less_frequent_index,
                                         required_more_frequent_index,
                                         required_less_frequent_index):
        more_frequent_index = initial_more_frequent_index
        less_frequent_index = initial_less_frequent_index
        point = (initial_more_frequent_index,initial_less_frequent_index)
        if (required_less_frequent_index != 0):
            frac_lfi_step_per_mfi_step = less_frequent_index/more_frequent_index
        else:
            frac_lfi_step_per_mfi_step = 0 
        return more_frequent_index,less_frequent_index,frac_lfi_step_per_mfi_step,point

    def generate_next_step_on_path(self,initial_more_frequent_index,initial_less_frequent_index,
                                   required_more_frequent_index,
                                   required_less_frequent_index,
                                   i_is_more_frequent_index,reroute_point_func,
                                   evaluate_rerouting_func,
                                   rerouting_start_points_and_biases):
        more_frequent_index,less_frequent_index,frac_lfi_step_per_mfi_step,point =\
            self.update_path_generation_variables(initial_more_frequent_index,
                                                  initial_less_frequent_index,
                                                  required_more_frequent_index,
                                                  required_less_frequent_index)
        while (more_frequent_index < required_more_frequent_index or 
               less_frequent_index < required_less_frequent_index):
            if(frac_lfi_step_per_mfi_step*more_frequent_index >= less_frequent_index + 1):
                less_frequent_index += math.copysign(1,required_less_frequent_index)
            more_frequent_index += math.copysign(1,required_more_frequent_index)
            previous_point = point 
            if (i_is_more_frequent_index): point = more_frequent_index,less_frequent_index
            else: point = less_frequent_index,more_frequent_index
            if not reroute_point_func(point):
                yield point
            else:
                rerouting = remove_disconnects.RemoveDisconnects.\
                    find_optimal_rerouting(start_point=previous_point,
                                           point_to_avoid=point,
                                           rerouting_start_points_and_biases=
                                           rerouting_start_points_and_biases,
                                           reroute_point_func,evaluate_rerouting_func,
                                           self.compute_rerouting)
                more_frequent_index,less_frequent_index,frac_lfi_step_per_mfi_step,point =\
                    self.update_path_generation_variables(rerouting[-1][0],
                                                          rerouting[-1][1],
                                                          required_more_frequent_index,
                                                          required_less_frequent_index)
                for reroute_point in rerouting:
                    yield reroute_point
    
    def compute_rerouting(self,start_point,point_to_avoid,divert_right):
        delta_i = point_to_avoid[0] - start_point[0]
        delta_j = point_to_avoid[1] - start_point[1]
        if delta_i == 0:
            delta_i = self.compute_non_diagonal_change_in_index(delta_j,divert_right)
        elif delta_j == 0:
            delta_j = self.compute_non_diagonal_change_in_index(-1*delta_i,divert_right)
        elif delta_i == delta_j:
            delta_i,delta_j =\
                self.compute_diagonal_change_in_indices_with_same_sign(delta_i,delta_j,
                                                                       divert_right) 
        else:
            delta_i,delta_j =\
                self.compute_diagonal_change_in_indices_with_opposite_sign(delta_i,delta_j,
                                                                            divert_right) 
    
    def compute_non_diagonal_change_in_index(self,other_index,divert_right):
        if (other_index > 0):
            return  1 if divert_right else -1
        else:
            return -1 if divert_right else 1
        
    def compute_diagonal_change_in_indices_with_same_sign(self,index_one,index_two,divert_right):
        if (index_one > 0):
            return 1,0 if divert_right else 0,1
        else:
            return -1,0 if divert_right else 0,-1
    
    def compute_diagonal_change_in_indices_with_opposite_sign(self,index_one,
                                                              index_two,divert_right):
        if (index_one > 0):
            return 0,-1 if divert_right else (1,0)
        else:
            return -1,0 if divert_right else (0,1)    
        
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