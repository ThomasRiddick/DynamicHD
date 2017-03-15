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
                                       return_image_array_instead_of_plotting=False,
                                       no_antarctic_rivers=True,colors=None,add_title=True):
    flowmap_ref_field[flowmap_ref_field < minflowcutoff] = 1
    flowmap_ref_field[flowmap_ref_field >= minflowcutoff] = 2
    flowmap_ref_field[np.logical_and(flowmap_data_field >= minflowcutoff,
                                     flowmap_ref_field == 2)] = 3
    flowmap_ref_field[np.logical_and(flowmap_data_field >= minflowcutoff,
                                     flowmap_ref_field != 3)] = 4                                
    if no_antarctic_rivers:
        flowmap_ref_field[310:,:] = 1
    if lsmask is not None:
        flowmap_ref_field[lsmask == 1] = 0
    if return_image_array_instead_of_plotting:
        return flowmap_ref_field
    if colors is None:
        cmap = mpl.colors.ListedColormap(['blue','peru','black','white','purple'])
    else:
        cmap = mpl.colors.ListedColormap(colors.basic_flowmap_comparison_plot_colors)
    bounds = range(6)
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    ax.imshow(flowmap_ref_field,cmap=cmap,norm=norm,interpolation='none')
    if add_title:
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
                                            catchment_grid_changed,use_alt_color,
                                            alt_color_num,
                                            use_single_color_for_discrepancies,
                                            use_only_one_color_for_flowmap,
                                            grid_type,
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
    return combine_image(image_array,catchment_section,use_alt_color,alt_color_num,use_single_color_for_discrepancies,
                         use_only_one_color_for_flowmap)

def combine_image(original_image,catchment_section,use_alt_color,alt_color_num,use_single_color_for_discrepancies,
                  use_only_one_color_for_flowmap):
    if use_only_one_color_for_flowmap:
        original_image[original_image == 2] = 1
    original_image[((original_image == 1) | (original_image == 0)) & (catchment_section == 2)] = 5
    original_image[((original_image == 1) | (original_image == 0)) & (catchment_section == 3)] =\
        6 if not use_alt_color else alt_color_num
    original_image[((original_image == 1) | (original_image == 0)) & (catchment_section == 4)] =\
        5 if use_single_color_for_discrepancies else 7
    return original_image

def plot_composite_image(ax,image,minflowcutoff,first_datasource_name,second_datasource_name,
                         use_single_color_for_discrepancies,use_only_one_color_for_flowmap,
                         use_title=True,colors=None):
    if use_single_color_for_discrepancies:
        image[image == 8] = 7
        image[image == 9] = 8
        if use_only_one_color_for_flowmap:
            if colors is None:
                flowmap_and_catchment_colors = ['lightblue','peru','blue','red',
                                                'grey','darkgrey','lightgrey']
            else:
                flowmap_and_catchment_colors = colors.flowmap_and_catchments_colors_single_color_flowmap
            cmap = mpl.colors.ListedColormap(flowmap_and_catchment_colors)
            bounds = range(8)
            image[image == 2] = 1
            image[(image == 3) | (image == 4)] = 2
            image[image > 4] = image[image > 4] - 2
        else:
            if colors is None:
                flowmap_and_catchment_colors = ['lightblue','peru','black','blue','purple','red',
                                                'grey','darkgrey','lightgrey']
            else:
                flowmap_and_catchment_colors = colors.flowmap_and_catchments_colors
            cmap = mpl.colors.ListedColormap(flowmap_and_catchment_colors)
            bounds = range(10)
    else:
        cmap = mpl.colors.ListedColormap(['lightblue','peru','black','blue','purple','red',
                                          'grey','green','darkgrey','lightgrey'])
        bounds = range(11)
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    ax.imshow(image,cmap=cmap,norm=norm,interpolation='none')
    if use_title:
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
    num_colors = 10 if use_single_color_for_discrepancies else 11
    if use_only_one_color_for_flowmap:
        num_colors -= 2 
    tic_loc = (np.arange(num_colors) + 0.5)
    if use_only_one_color_for_flowmap:
        tic_labels = ['Sea', 'Land','{} River Path'.format(second_datasource_name)]
    else:
        tic_labels = ['Sea', 'Land','{} River Path'.format(first_datasource_name),
                      'Common River Path','{} River Path'.format(second_datasource_name)]
    if use_single_color_for_discrepancies:
        tic_labels.extend(["Discrepancy in Catchments","Common Catchment","Common Catchment","Common Catchment"])
    else:
        tic_labels.extend(["Reference Catchment","Common Catchment","Data Catchment","Common Catchment","Common Catchment"])
    cb.set_ticks(tic_loc) 
    cb.set_ticklabels(tic_labels)
    