'''
Created on Mar 29, 2017

@author: thomasriddick
'''

import numpy as np
import field
import heapq
import copy
from __builtin__ import None

class RemoveDisconnects(object):
    '''
    classdocs
    '''
    
    disconnect_size_threshold = 2

    def __init__(self, params):
        '''
        Constructor
        '''
        
    def run_disconnect_removal(self):
        self.find_disconnects()
        while (self.disconnect) > 0:
            self.order_disconnects()
            largest_disconnect = self.disconnects.pop()
            if self.yamazaki_flowtocell.get_value(largest_disconnect) < self.disconnect_size_threshold: 
                break
            path,cotat_disconnected_catchment_num,downstream_reconnection_target =\
                 self.find_path(largest_disconnect)
            if path is None:
                print "Couldn't solve disconnect at:"  +\
                    self.grid.coords_as_string(largest_disconnect)
                continue
            self.mark_path(path, largest_disconnect, downstream_reconnection_target)
            self.attempt_to_reroute_disconnected_cells(path, 
                    cotat_disconnected_catchment_num)

    
    def find_disconnects(self): 
        self.disconnects = [(dc,True) for dc in 
                            self.yamazaki_rdirs_field.find_disconnects(self.old_cotat_rdirs)]
    
    def order_disconnects(self):
        disconnect_magnitudes = [[disconnect,self.yamazaki_flowtocell.get_value(disconnect[0])]
                                 for disconnect in self.disconnects]
        disconnect_magnitudes.sort(key=lambda disconnect_magnitude: disconnect_magnitude[1])
        self.disconnects = [disconnect_magnitude[0] 
                            for disconnect_magnitude in disconnect_magnitudes] 

    def find_path(self,disconnect):
        if disconnect[1]:
            downstream_reconnection_target = self.yamazaki_rdirs_field.get_target_coords(disconnect[0])
        else:
            downstream_reconnection_target = self.new_disconnects_map.get_target_coords(disconnect[0])
        disconnect_size = self.yamazaki_flowtocell.get_value(disconnect)
        cotat_disconnected_catchment_num = self.cotat_catchments.get_value(downstream_reconnection_target)
        path_generator = \
            self.GeneratePath(disconnect,downstream_reconnection_target,
                              lambda x: self.reroute_point(x,disconnect_size,
                                        cotat_disconnected_catchment_num),
                              lambda x: self.evaluate_rerouting(x,
                                        cotat_disconnected_catchment_num)) 
        path = path_generator.generate_path()
        return path,cotat_disconnected_catchment_num,downstream_reconnection_target
        
    def reroute_point(self,point,disconnect_size,cotat_disconnected_catchment_num):
        if point is None:
            return True
        if self.true_sinks_field.is_true_sink(point):
            return True 
        elif self.cotat_catchments.get_value(point) == cotat_disconnected_catchment_num:
            return False
        elif self.modification_counts.get_value(point) > self.max_allowed_modifications_of_point:
            if self.yamazaki_flowtocell.get_value(point) >= disconnect_size: 
                return True
            else:
                return False
    
    def evaluate_rerouting(self,point,cotat_disconnected_catchment_num):
        if self.cotat_catchments.get_value(point) == cotat_disconnected_catchment_num:
            return 0 
        else:
            return self.yamazaki_flowtocell.get_value(point)
    
    def mark_path(self,path,disconnect,downstream_reconnection_target):         
        previous_point = downstream_reconnection_target
        for point in path:
            self.modification_counts.increment_value(point)
            self.new_cotat_rdirs.direct_rdir_to_point(point,previous_point)
            previous_point = point
        self.new_cotat_rdirs.direct_rdir_to_point(disconnect,previous_point)
        
    def attempt_to_reroute_disconnected_cells(self,path,primary_cotat_disconnected_catchment_num):
        self.reconnected_points = field.Field(self.grid.create_empty_field(np.bool_),
                                              grid=self.grid)
        for point in path:
            if(self.old_cotat_catchments.get_value(point) == 
               primary_cotat_disconnected_catchment_num):
                continue
            disconnected_neighbors = self.new_cotat_rdirs.get_neighbors_flowing_to_point(point)
            downstream_neighbor = self.old_cotat_rdirs.get_downstream_neighbor(point)
            second_downstream_neighbor = self.old_cotat_rdirs.get_downstream_neighbor(downstream_neighbor)
            if (self.new_cotat_catchments.get_value(downstream_neighbor) == 
                self.old_cotat_catchments.get_value(downstream_neighbor)):
                disconnects_solved = []
                for disconnected_neighbor in disconnected_neighbors:
                    reconnect_successful = self.grid.attempt_direct_reconnect(disconnected_neighbor,downstream_neighbor,
                                                                              self.new_cotat_rdirs.get_data())
                    if reconnect_successful is not None:
                        self.reconnected_points.set_value(disconnected_neighbor,True)
                        disconnects_solved.append(disconnected_neighbor)
                disconnected_neighbors = [dc for dc in disconnected_neighbors if dc not in disconnects_solved] 
            if (self.new_cotat_catchments.get_value(second_downstream_neighbor) == 
                self.old_cotat_catchments.get_value(second_downstream_neighbor)):
                disconnects_solved = []
                for disconnected_neighbor in disconnected_neighbors:
                    reconnect_successful = self.grid.attempt_direct_reconnect(disconnected_neighbor,
                                                                              second_downstream_neighbor,
                                                                              self.new_cotat_rdirs.get_data())
                    if reconnect_successful is not None:
                        self.reconnected_points.set_value(disconnected_neighbor,True)
                        disconnects_solved.append(disconnected_neighbor)
                disconnected_neighbors = [dc for dc in disconnected_neighbors if dc not in disconnects_solved] 
            self.attempt_extended_direct_reconnect(disconnected_neighbors)
            self.attempt_indirect_reconnect(disconnected_neighbors)
            self.attempt_extended_direct_reconnect(disconnected_neighbors)
            if len(disconnected_neighbors) > 0:
                for disconnected_neighbor in disconnected_neighbors:
                    self.disconnects.append((disconnected_neighbors,False))
                    nearest_possible_reconnect = self.reconnected_points.\
                        find_nearest_possible_reconnect(disconnected_neighbor)
                    self.new_disconnects_map.set_value(disconnected_neighbor,nearest_possible_reconnect)
                
    def attempt_extended_direct_reconnects(self,disconnected_neighbors):
        disconnects_solved = []
        extended_direct_connect_made=True
        while(extended_direct_connect_made==True and len(disconnected_neighbors) > 0):
            extended_direct_connect_made=False
            for disconnected_neighbor in disconnected_neighbors:
                reconnect_successful = self.grid.attempt_extended_direct_reconnect(disconnected_neighbor,
                                                                                   self.reconnected_points.get_data(),
                                                                                   self.new_cotat_catchments.get_data(),
                                                                                   self.new_cotat_rdirs.get_data())
                if reconnect_successful is not None:
                    self.reconnected_points.set_value(disconnected_neighbor,True)
                    disconnects_solved.append(disconnected_neighbor)
                    extended_direct_connect_made=True
            disconnected_neighbors = [dc for dc in disconnected_neighbors if dc not in disconnects_solved] 

    def attempt_indirect_reconnects(self,disconnected_neighbors):
        disconnects_solved = []
        indirect_connect_made=True
        while(indirect_connect_made==True and len(disconnected_neighbors) > 0):
            indirect_connect_made=False
            for disconnected_neighbor in disconnected_neighbors:
                reconnect_successful = self.grid.attempt_indirect_reconnect(disconnected_neighbor,
                                                                            self.new_cotat_catchments.get_data(),
                                                                            self.new_cotat_rdirs.get_data())
                if reconnect_successful is not None:
                    self.reconnected_points.set_value(disconnected_neighbor,True)
                    disconnects_solved.append(disconnected_neighbor)
                    indirect_connect_made=True
            disconnected_neighbors = [dc for dc in disconnected_neighbors if dc not in disconnects_solved] 
            disconnects_solved = []
            
class GeneratePath(object):
    
    frustration_constant = 5
    search_abandonment_threshold = 25
    
    def __init__(self,initial_coords,target_coords,reroute_point_func,evaluate_rerouting_func,
                 grid):
        self.q = []
        self.initial_coords = initial_coords
        self.target_coords = target_coords
        self.reroute_point_func = reroute_point_func
        self.evaluate_rerouting_func = evaluate_rerouting_func
        self.grid = grid
        
    def generate_path(self):
        heapq.heappush(self.q,(0,self.initial_coords,[self.initial_coords],0))
        lowest_cost_found = 0
        lowest_cost_path = None
        while len(self.q) > 0:
            path_cost_including_frustration,point_coords,path_to_point,path_cost = heapq.heappop(self.q)
            neighbors = list((neighbor for neighbor in self.grid.get_neighbors(point_coords) 
                              if neighbor not in set(path_to_point)))
            if lowest_cost_path is not None:
                if (path_cost_including_frustration > lowest_cost_found 
                    + self.search_abandonment_threshold):
                    return lowest_cost_path[1:]
            for neighbor_coords in neighbors:
                if neighbor_coords == self.target_coords:
                    if lowest_cost_path is not None:
                        if path_cost < lowest_cost_path: 
                            lowest_cost_found = path_to_point
                            lowest_cost_found = path_cost
                    else:
                        lowest_cost_path = path_to_point 
                    continue
                if self.reroute_point_func(neighbor_coords):
                    continue
                neighbor_path_cost = path_cost +  self.evaluate_rerouting_func(neighbor_coords)
                progress = self.grid.progress_made(neighbor_coords,point_coords,self.target_coords)
                if progress > 0:
                    neighbor_path_cost_including_frustration = neighbor_path_cost
                elif progress == 0:
                    neighbor_path_cost_including_frustration = path_cost_including_frustration +\
                                                                self.frustration_constant
                else:
                    neighbor_path_cost_including_frustration = path_cost_including_frustration +\
                                                                self.frustration_constant*2
                neighbor_path_to_point = copy.deepcopy(path_to_point) 
                neighbor_path_to_point.append(neighbor_coords)
                heapq.heappush(self.q,(neighbor_path_cost_including_frustration,
                               neighbor_coords,neighbor_path_to_point,neighbor_path_cost)) 
        return None