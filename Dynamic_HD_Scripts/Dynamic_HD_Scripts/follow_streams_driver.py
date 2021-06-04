'''
Created on Feb 12, 2020

@author: thomasriddick
'''

import numpy as np
from Dynamic_HD_Scripts import field
from Dynamic_HD_Scripts.libs import follow_streams_wrapper

def follow_streams(rdirs_in,cumulative_flow):
    downstream_cells_out = np.zeros(rdirs_in.get_data().shape,dtype=np.int32)
    follow_streams_wrapper.follow_streams(rdirs_in.get_data(),
                                          cumulative_flow.\
                                          get_cells_with_loops().astype(np.int32),
                                          downstream_cells_out)
    return field.makeField(downstream_cells_out,field_type="Generic",
                           grid_type=rdirs_in.get_grid())
