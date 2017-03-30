'''
Created on Mar 29, 2017

@author: thomasriddick
'''
import numpy as np
import field

class RemoveDisconnects(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        
    def run_disconnect_removal(self):
        self.find_disconnects()
        self.order_disconnects()
    
    def find_disconnects(self): 
        self.disconnects = [(dc,True) for dc in 
                            self.yamazaki_rdirs_field.find_disconnects(self.old_cotat_rdirs)]
    
    def order_disconnects(self):
        disconnect_magnitudes = [[disconnect,self.yamazaki_flowtocell.get_value(disconnect)]
                                 for disconnect in self.disconnects]
        disconnect_magnitudes.sort(key=lambda disconnect_magnitude: disconnect_magnitude[1])
        self.disconnects = [disconnect_magnitude[0] 
                            for disconnect_magnitude in disconnect_magnitudes] 

    def find_path(self,disconnect):
        if disconnect[1]:
            downstream_reconnection_target = self.yamazaki_rdirs_field.get_coords(disconnect[0])
        else:
            downstream_reconnection_target = self.new_disconnects_map.get_coords(disconnect[0])
        disconnect_size = self.yamazaki_flowtocell.get_value(disconnect)
        cotat_disconnected_catchment_num = self.cotat_catchments(downstream_reconnection_target)
        path = self.grid.find_path(disconnect,downstream_reconnection_target,
                                   disconnect_size,cotat_disconnected_catchment_num,
                                   self.reroute_point,self.evaluate_rerouting) 
        if path is not None:
            for point in path:
                self.modification_counts.increment_value(point)
        return path
        
    def reroute_point(self,point,disconnect_size,cotat_disconnected_catchment_num):
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
        self.reconnected_points = field.Field(self.grid.grid.create_empty_field(np.bool_),
                                              grid=self.grid)
        for point in path:
            if(self.old_cotat_catchments.get_value(primary_cotat_disconnected_catchment_num) == 
               primary_cotat_disconnected_catchment_num):
                break
            disconnected_neighbors = self.new_cotat_rdirs.get_neighbors_flowing_to_point(point)
            downstream_neighbor = self.old_cotat_rdirs.get_downstream_neighbor(point)
            if (self.new_cotat_catchments.get_value(downstream_neighbor) == 
                self.old_cotat_catchments.get_value(downstream_neighbor)):
                disconnects_solved = []
                for disconnected_neighbor in disconnected_neighbors:
                    reconnected_point = self.grid.attempt_direct_reconnect(disconnected_neighbor,downstream_neighbor,
                                                                           self.new_cotat_rdirs.get_data())
                    if reconnected_point is not None:
                        self.reconnected_points.set_value(reconnected_point,True)
                        disconnects_solved.append(reconnected_point)
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
                reconnected_point = self.grid.attempt_extended_direct_reconnect(disconnected_neighbor,
                                                                                self.reconnected_points,
                                                                                self.new_cotat_catchments,
                                                                                self.new_cotat_rdirs.get_data())
                if reconnected_point is not None:
                    self.reconnected_points.set_value(reconnected_point,True)
                    disconnects_solved.append(reconnected_point)
                    extended_direct_connect_made=True
            disconnected_neighbors = [dc for dc in disconnected_neighbors if dc not in disconnects_solved] 

    def attempt_indirect_reconnects(self,disconnected_neighbors):
        disconnects_solved = []
        indirect_connect_made=True
        while(indirect_connect_made==True and len(disconnected_neighbors) > 0):
            indirect_connect_made=False
            for disconnected_neighbor in disconnected_neighbors:
                reconnected_point = self.grid.attempt_indirect_reconnect(disconnected_neighbor,
                                                                         self.new_cotat_catchments,
                                                                         self.new_cotat_rdirs.get_data())
                if reconnected_point is not None:
                    self.reconnected_points.set_value(reconnected_point,True)
                    disconnects_solved.append(reconnected_point)
                    indirect_connect_made=True
            disconnected_neighbors = [dc for dc in disconnected_neighbors if dc not in disconnects_solved] 
            disconnects_solved = []
