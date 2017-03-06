'''
Created on Mar 4, 2017

@author: thomasriddick
'''

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotting_tools as pts
import river_comparison_plotting_routines as rc_pts

def make_basic_flowmap_comparison_plot(ax,flowmap_ref_field,flowmap_data_field,minflowcutoff,
                                       first_datasource_name,second_datasource_name,lsmask=None,
                                       return_image_array_instead_of_plotting=False):
    flowmap_ref_field[flowmap_ref_field < minflowcutoff] = 1
    flowmap_ref_field[flowmap_ref_field >= minflowcutoff] = 2
    flowmap_ref_field[np.logical_and(flowmap_data_field >= minflowcutoff,
                                     flowmap_ref_field == 2)] = 3
    flowmap_ref_field[np.logical_and(flowmap_data_field >= minflowcutoff,
                                     flowmap_ref_field != 3)] = 4                                
    if lsmask is not None:
        flowmap_ref_field[lsmask == 1] = 0
    if return_image_array_instead_of_plotting:
        return flowmap_ref_field
    cmap = mpl.colors.ListedColormap(['blue','peru','black','white','purple'])
    bounds = range(6)
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    ax.imshow(flowmap_ref_field,cmap=cmap,norm=norm,interpolation='none')
    plt.title('Cells with cumulative flow greater than or equal to {0}'.format(minflowcutoff))
    pts.remove_ticks(ax)
    ax.format_coord = pts.OrogCoordFormatter(0,0)
    mappable = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)
    mappable.set_array(flowmap_ref_field)
    dvdr = make_axes_locatable(ax)
    cax = dvdr.append_axes("right", size=0.2, pad=0.05)
    cb = plt.colorbar(mappable,cax=cax)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    tic_loc = np.arange(6) + 0.5
    tic_labels = ['Sea', 'Land','{} River Path'.format(first_datasource_name),
                  'Common River Path','{} River Path'.format(second_datasource_name)]
    cb.set_ticks(tic_loc) 
    cb.set_ticklabels(tic_labels)
    
def add_selected_catchment_to_existing_plot(image_array,data_catchment_field,
                                            ref_catchment_field,
                                            data_catchment_field_original_scale,
                                            data_original_scale_flowtocellfield,
                                            rdirs_field,data_rdirs_field,pair,
                                            catchment_grid_changed,grid_type,
                                            data_original_scale_grid_type='HD',
                                            data_original_scale_grid_kwargs={},
                                            **grid_kwargs):
    points_to_mark = [pair[0].get_coords(),pair[1].get_coords()]
    ref_catchment_num,data_catchment_num,_ = rc_pts.find_catchment_numbers(ref_catchment_field,data_catchment_field,
                                                                           data_catchment_field_original_scale,
                                                                           data_original_scale_flowtocellfield,pair,
                                                                           catchment_grid_changed,
                                                                           data_original_scale_grid_type=\
                                                                           data_original_scale_grid_type,
                                                                           data_original_scale_grid_kwargs=\
                                                                           data_original_scale_grid_kwargs,
                                                                           grid_type=grid_type,**grid_kwargs)
    catchment_section,_ = rc_pts.select_catchment(ref_catchment_field,
                                                  data_catchment_field,
                                                  rdirs_field,
                                                  ref_catchment_num,
                                                  data_catchment_num,
                                                  points_to_mark,
                                                  data_true_sinks=data_rdirs_field,
                                                  allow_new_sink_points=False,
                                                  alternative_catchment_bounds=(0,data_catchment_field.shape[0],
                                                                                0,data_catchment_field.shape[1]))
    return combine_image(image_array,catchment_section)

def combine_image(original_image,catchment_section):
    original_image[((original_image == 1) | (original_image == 0)) & (catchment_section == 2)] = 5
    original_image[((original_image == 1) | (original_image == 0)) & (catchment_section == 3)] = 6
    original_image[((original_image == 1) | (original_image == 0)) & (catchment_section == 4)] = 7
    return original_image

def plot_composite_image(ax,image,minflowcutoff,first_datasource_name,second_datasource_name):
    cmap = mpl.colors.ListedColormap(['blue','peru','black','white','purple','red','grey','green'])
    bounds = range(9)
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    ax.imshow(image,cmap=cmap,norm=norm,interpolation='none')
    plt.title('Cells with cumulative flow greater than or equal to {0}'.format(minflowcutoff))
    pts.remove_ticks(ax)
    ax.format_coord = pts.OrogCoordFormatter(0,0)
    mappable = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)
    mappable.set_array(image)
    dvdr = make_axes_locatable(ax)
    cax = dvdr.append_axes("right", size=0.2, pad=0.05)
    cb = plt.colorbar(mappable,cax=cax)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    tic_loc = np.arange(6) + 0.5
    tic_labels = ['Sea', 'Land','{} River Path'.format(first_datasource_name),
                  'Common River Path','{} River Path'.format(second_datasource_name)]
    cb.set_ticks(tic_loc) 
    cb.set_ticklabels(tic_labels)
    