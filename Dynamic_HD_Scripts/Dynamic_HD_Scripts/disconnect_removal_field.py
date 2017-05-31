'''
Created on Apr 13, 2017

@author: thomasriddick
'''

import field

class FieldForDCR(field.Field): 

    def get_value(self,coords):
        return self.grid.get_value_at_coords(self.data,coords)
    
    def set_value(self,coords,value):
        self.grid.set_value_at_coords(self.data,coords,value)

class IndexBasedRiverDirections(field.Field):

    def find_disconnects(self,original_cotat_rdirs):
        """Find disconnect by comparing yamazaki upscaled flow directions with cotat upscaled flow directions"""
        return self.grid.return_coords_of_differences(self.data,self.original_cotat_rdirs.get_data()) 
    
    def get_target_coords(self,input_coords):
        return self.grid.return_target_coords_of_index_based_rdirs(self.data,input_coords)
    
    def get_neighbors_flowing_to_point(self,point):
        return self.grid.get_neighbors_flowing_to_point(self.data,point)
        
    def get_downstream_neighbor(self,point):
        return self.grid.get_downstream_neighbor(self.data,point)
    
    def direct_rdir_to_point(self,input_point,target_point):
        self.grid.set_target_coords_of_index_based_rdirs(self.data,input_point,target_point)

class ModificationCounts(FieldForDCR): 
    
    def increment_value(self,coords):
        self.grid.increment_value(self.data,coords)
        
class TrueSinksForDCR(FieldForDCR):
    
    def is_true_sink(self,coords):
        return self.get_value(coords)
        