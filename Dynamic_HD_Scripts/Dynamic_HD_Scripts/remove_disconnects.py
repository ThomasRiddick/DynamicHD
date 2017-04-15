'''
Created on Mar 29, 2017

@author: thomasriddick
'''
import numpy as np
import field
import collections
from __builtin__ import None

class RemoveDisconnects(object):
    '''
    classdocs
    '''
    
    left_bias_value  = -1
    no_bias_value    =  0
    right_bias_value =  1


    def __init__(self, params):
        '''
        Constructor
        '''
        
    def run_disconnect_removal(self):
        self.find_disconnects()
        while (self.disconnect) > 0:
            self.order_disconnects()
            largest_disconnect = self.disconnects.pop()
            if self.yamazaki_flowtocell.get_value(largest_disconnect) < 2: 
                break
            path,cotat_disconnected_catchment_num = self.find_path(largest_disconnect)
            if path is None:
                print "Couldn't solve disconnect at:"  +\
                    self.grid.coords_as_string(largest_disconnect)
                continue
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
        rerouting_start_points_and_biases = collections.OrderedDict()
        next_path_step_generator = \
            self.grid.prep_path_generator(disconnect,downstream_reconnection_target,
                                          disconnect_size,cotat_disconnected_catchment_num,
                                          lambda x: self.reroute_point(x,disconnect_size,
                                                    cotat_disconnected_catchment_num),
                                          lambda x: self.evaluate_rerouting(x,
                                                    cotat_disconnected_catchment_num),
                                          rerouting_start_points_and_biases) 
        path = list(next_path_step_generator)
        while path[-1] is None:
            path = list(next_path_step_generator)
            if len(rerouting_start_points_and_biases) == 0:
                return None,None
        for point in path:
                self.modification_counts.increment_value(point)
        return path,cotat_disconnected_catchment_num
        
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
            
    @classmethod
    def find_optimal_rerouting_step(cls,start_point,point_to_avoid,initial_bias,
                                    reroute_point_func,evaluate_rerouting_func,
                                    compute_rerouting_func):
        bias = initial_bias
        rerouting = []
        direct_rerouting = True
        cumulative_rerouting_penalty = 0
        while True:
            if bias != cls.left_bias_value: 
                righthand_rerouting = compute_rerouting_func(start_point,point_to_avoid,divert_right=True)
            else:
                righthand_rerouting = None
            if bias != cls.right_bias_value:
                lefthand_rerouting =  compute_rerouting_func(start_point,point_to_avoid,divert_right=False)
            else:
                lefthand_rerouting = None
            righthand_rerouting_valid = not reroute_point_func(righthand_rerouting)
            lefthand_rerouting_valid  = not reroute_point_func(lefthand_rerouting)
            valid_rerouting = righthand_rerouting or lefthand_rerouting
            if righthand_rerouting_valid and lefthand_rerouting_valid:
                if (bias != cls.left_bias_value):
                    rerouting.append(righthand_rerouting)
                    bias = cls.right_bias_value
                else:
                    rerouting.append(lefthand_rerouting)
                    bias = cls.left_bias_value
            elif righthand_rerouting_valid:
                rerouting.append(righthand_rerouting)
                bias = cls.right_bias_value
            elif lefthand_rerouting_valid:
                rerouting.append(lefthand_rerouting)
                bias = cls.left_bias_value
            else:
                if not direct_rerouting: 
                    if initial_bias != cls.no_bias_value:
                        return None
                    else:
                        second_attempt_output = \
                            cls.find_optimal_rerouting(start_point,point_to_avoid, 
                                                       initial_bias=\
                                                       cls.left_bias_value if bias == 
                                                       cls.right_bias_value else cls.right_bias_value,
                                                       reroute_point_func=reroute_point_func,
                                                       evaluate_rerouting_func=evaluate_rerouting_func)
                        if second_attempt_output is None:
                            return None
                        else:
                            return second_attempt_output[:2]
                else: 
                    direct_rerouting = False
            if direct_rerouting:
                cumulative_rerouting_penalty += evaluate_rerouting_func(rerouting[-1])
                if initial_bias != cls.no_bias_value:
                    return rerouting,bias,cumulative_rerouting_penalty
                else:
                    second_attempt_output = \
                            cls.find_optimal_rerouting(start_point,point_to_avoid,
                                                       initial_bias=\
                                                       cls.left_bias_value if bias == 
                                                       cls.right_bias_value else cls.right_bias_value,
                                                       reroute_point_func=reroute_point_func,
                                                       evaluate_rerouting_func=evaluate_rerouting_func)
                    if (second_attempt_output is not None and 
                        second_attempt_output[2] < cumulative_rerouting_penalty):
                        return second_attempt_output[:2] 
                    else:
                        return rerouting,bias
            elif valid_rerouting:
                cumulative_rerouting_penalty += evaluate_rerouting_func(rerouting[-1])
                direct_rerouting = True
    
    @classmethod
    def find_optimal_rerouting(cls,previous_point,rerouting_start_points_and_biases,
                               diagonal_step,reroute_point_func,evaluate_rerouting_func):
        if previous_point == next(reversed(rerouting_start_points_and_biases)): 
            if (rerouting_start_points_and_biases[previous_point] == 
                cls.no_bias_value):
                rerouting_start_points_and_biases.popitem()
                yield None
                break
            else:
                left_bias  = cls.left_bias_value
                right_bias = cls.right_bias_value 
                bias = left_bias if rerouting_start_points_and_biases[previous_point] ==\
                                    right_bias else right_bias
        else:
            bias = cls.no_bias_value
        optimal_rerouting_output = cls.\
            find_optimal_rerouting_step(previous_point,
                                        diagonal_step=diagonal_step,
                                        bias=bias,reroute_point_func=reroute_point_func,
                                        evaluate_rerouting_func=evaluate_rerouting_func)
        if optimal_rerouting_output is None:
            yield None
            break
        rerouting,route_bias = optimal_rerouting_output
        if not set(rerouting).isdisjoint(rerouting_start_points_and_biases.keys()):
            yield None
            break
        if bias == cls.no_bias_value:
            rerouting_start_points_and_biases[previous_point] = route_bias
        else:
            rerouting_start_points_and_biases[previous_point] =\
                cls.no_bias_value