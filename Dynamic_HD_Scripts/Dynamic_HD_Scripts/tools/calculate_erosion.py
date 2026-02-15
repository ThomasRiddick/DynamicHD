import numpy as np
import scipy.ndimage as ndimage
from Dynamic_HD_Scripts.utilities import  downscale_ls_mask

def calculate_erosion(erodable_material,
                      sill_flow,
                      river_flow,
                      erosion_parameters,
                      fine_grid_type,
                      **fine_grid_kwargs):
    downscaled_river_flow = downscale_ls_mask(river_flow,fine_grid_type,**fine_grid_kwargs)
    maxflow = np.maximum(downscaled_river_flow.get_data(),sill_flow.get_data())
    if erosion_range > 0:
        stencil = make a stencil based on erosion_range
        maxflow = ndimage.maximum_filter(maxflow,mode=""(wrap+nearest???))
    if connected_cells:
        for cell_cluster in connected_cells:
            highest_maxflow = 0.0
            for cell in cell_cluster:
                if maxflow[cell] > highest_maxflow:
                    highest_maxflow = maxflow[cell]
            for cell in cell_cluster:
                maxflow[cell] = highest_maxflow
    if use_constant_erosion_threshold:
        erosion_threshold = constant_erosion_threshold
    else:
        erosion_threshold =
    if instant_erosion_above_threshold:
        erodable_material.get_data()[maxflow > erosion_threshold] = 0.0
    else:
        flow_over_threshold = np.where(maxflow > erosion_threshold,
                                       maxflow - erosion_threshold,
                                       np.zeros(maxflow.shape))
        erodable_material.get_data() =
            erodable_material.get_data() - erosion_rate*flow_over_threshold
        erodable_material.get_data()[erodable_material.get_data() < 0] = 0.0
