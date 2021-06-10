'''
Created on Mar 4, 2017

@author: thomasriddick
'''

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.pyplot as plt
from HD_Plots.utilities import plotting_tools as pts
from HD_Plots.utitlies import river_comparison_plotting_routines as rc_pts

def make_basic_flowmap_comparison_plot(ax,flowmap_ref_field,flowmap_data_field,minflowcutoff,
                                       first_datasource_name,second_datasource_name,lsmask=None,
                                       return_image_array_instead_of_plotting=False,
                                       no_antarctic_rivers=True,colors=None,add_title=True,
                                       glacier_mask=None,second_lsmask=None):
    flowmap_ref_field_copy = np.copy(flowmap_ref_field)
    flowmap_ref_field[flowmap_ref_field_copy < minflowcutoff] = 1
    if glacier_mask is not None:
        flowmap_ref_field[(glacier_mask == 100.) |
                          (glacier_mask == 1) ] = -1 if return_image_array_instead_of_plotting else 5
    flowmap_ref_field[flowmap_ref_field_copy >= minflowcutoff] = 2
    flowmap_ref_field[np.logical_and(flowmap_data_field >= minflowcutoff,
                                     flowmap_ref_field == 2)] = 3
    flowmap_ref_field[np.logical_and(flowmap_data_field >= minflowcutoff,
                                     flowmap_ref_field != 3)] = 4
    if no_antarctic_rivers:
        if glacier_mask is not None:
            flowmap_ref_field[310:,:] = 1
            flowmap_ref_field[310:,:][(glacier_mask[310:,:] == 100.) |
                                      (glacier_mask[310:,:] == 1 )] =\
                                       -1 if return_image_array_instead_of_plotting else 5
        else:
            flowmap_ref_field[310:,:] = 1
    if lsmask is not None:
        flowmap_ref_field[lsmask == 1] = 0
        if second_lsmask is not None:
          extra_lsmask = np.zeros(second_lsmask.shape)
          extra_lsmask[np.logical_and(second_lsmask == 1,
                                      lsmask != 1)] = 1
          extra_lsmask[np.logical_and(second_lsmask != 1,
                                      lsmask == 1)] = 2
    if return_image_array_instead_of_plotting:
      if lsmask is None or second_lsmask is None:
        return flowmap_ref_field
      else:
        return flowmap_ref_field,extra_lsmask
    if colors is None:
        cmap = mpl.colors.ListedColormap(['blue','peru','black','white','purple'])
    else:
        cmap = mpl.colors.ListedColormap(colors.basic_flowmap_comparison_plot_colors)
    bounds = list(range(6))
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    ax.imshow(flowmap_ref_field,cmap=cmap,norm=norm,interpolation='none',rasterized=True)
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

def add_extra_flowmap(image_array,extra_ls_mask):
    current_max = np.amax(image_array)
    image_array[np.logical_and(image_array == 1,extra_ls_mask == 1)] = current_max+1
    image_array[np.logical_and(image_array == 0,extra_ls_mask == 2)] = current_max+2
    return image_array


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
                         use_title=True,colors=None,use_only_one_common_catchment_label=True,
                         difference_in_catchment_label="Discrepancy",remove_ticks=False,
                         flowmap_grid=None,plot_glaciers=False,second_ls_mask=False,
                         no_extra_sea=True):
    if second_ls_mask:
      max = image.max()
      image[image == max] = - 2
      image[image == max - 1] = - 3
    if use_single_color_for_discrepancies:
        image[image == 8] = 7
        image[image == 9] = 8
        image[image == -1] = 9
        if use_only_one_color_for_flowmap:
            if colors is None:
                flowmap_and_catchment_colors = ['lightblue','peru','blue','red',
                                                'grey','darkgrey','lightgrey']
                if plot_glaciers:
                    flowmap_and_catchment_colors += ['white']
            else:
                if plot_glaciers:
                    flowmap_and_catchment_colors = colors.\
                    flowmap_and_catchments_colors_single_color_flowmap_with_glac
                else:
                    flowmap_and_catchment_colors = colors.\
                    flowmap_and_catchments_colors_single_color_flowmap
            cmap = mpl.colors.ListedColormap(flowmap_and_catchment_colors)
            color_list = flowmap_and_catchment_colors
            bounds = list(range(8)) if not plot_glaciers else list(range(9))
            image[image == 2] = 1
            image[(image == 3) | (image == 4)] = 2
            image[image > 4] = image[image > 4] - 2
        else:
            if colors is None:
                flowmap_and_catchment_colors = ['lightblue','peru','black','blue','purple','red',
                                                'grey','darkgrey','lightgrey']
                if plot_glaciers:
                    flowmap_and_catchment_colors += ['white']
            else:
                if plot_glaciers:
                    flowmap_and_catchment_colors = colors.flowmap_and_catchments_colors_with_glac
                else:
                    flowmap_and_catchment_colors = colors.flowmap_and_catchments_colors
            cmap = mpl.colors.ListedColormap(flowmap_and_catchment_colors)
            color_list = flowmap_and_catchment_colors
            bounds = list(range(10)) if not plot_glaciers else list(range(11))
    else:
        image[image == -1] = 10
        color_list = ['lightblue','peru','black','blue','purple','red',
                      'grey','green','darkgrey','lightgrey']
        if plot_glaciers:
            color_list += ['white']
        cmap = mpl.colors.ListedColormap(color_list)
        bounds = list(range(11)) if not plot_glaciers else list(range(12))
    if second_ls_mask:
        image[image > 1] = image[image > 1] + 1
        image[image == -3] = 2
        if not no_extra_sea:
          image[image > 0] = image[image > 0] + 1
          image[image == -2] = 1
        if colors is None:
          extra_sea = ['green']
          extra_land = ['orange']
        else:
          extra_sea = colors.extra_sea_color
          extra_land = colors.extra_land_color
        if no_extra_sea:
          color_list[2:2] = extra_land
          extra_colors = 1
        else:
          color_list[1:1] = extra_sea
          color_list[3:3] = extra_land
          extra_colors = 2
        bounds = list(range(len(bounds)+ extra_colors))
        cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    ax.imshow(image,cmap=cmap,norm=norm,interpolation='none',rasterized=True)
    if use_title:
        plt.title('Cells with cumulative flow greater than or equal to {0}'.format(minflowcutoff))
    if remove_ticks:
        pts.remove_ticks(ax)
    else:
            axis_tick_label_scale_factor=\
            flowmap_grid.get_scale_factor_for_geographic_coords() if flowmap_grid is not None else 0.5
            ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(60/axis_tick_label_scale_factor,0))
            ax.yaxis.set_major_locator(mpl.ticker.IndexLocator(30/axis_tick_label_scale_factor,0))
            #Scale factor is multiplied by two as formatter has a built in scale factor of a half
            ax.xaxis.set_major_formatter(mpl.ticker.\
                                               FuncFormatter(pts.LonAxisFormatter(0,
                                                            axis_tick_label_scale_factor*2)))
            ax.yaxis.set_major_formatter(mpl.ticker.\
                                               FuncFormatter(pts.LatAxisFormatter(0,
                                                            axis_tick_label_scale_factor*2)))
    num_colors = 10 if use_single_color_for_discrepancies else 11
    if plot_glaciers:
        num_colors += 1
    if use_only_one_color_for_flowmap:
        num_colors -= 2
    if second_ls_mask:
        num_colors += 1 if no_extra_sea else 2
    ax.format_coord = pts.OrogCoordFormatter(0,0)
    mappable = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)
    mappable.set_array(image)
    dvdr = make_axes_locatable(ax)
    cax = dvdr.append_axes("right", size=0.2, pad=0.05)
    if use_only_one_common_catchment_label:
        if plot_glaciers:
            adjusted_bounds = (list(range(num_colors))[:-4]
                                + [num_colors-4-2.0/3,num_colors-4-1.0/3,
                                   num_colors-4,num_colors-3])
        else:
            adjusted_bounds = (list(range(num_colors))[:-3]
                               + [num_colors-3-2.0/3,num_colors-3-1.0/3,num_colors-3])
        norm = mpl.colors.BoundaryNorm(adjusted_bounds,cmap.N)
        cb = mpl.colorbar.ColorbarBase(cax,cmap=cmap,norm=norm,boundaries=adjusted_bounds,
                                       spacing='proportional')
    else:
        cb = plt.colorbar(mappable,cax=cax)
        if second_ls_mask:
          raise UserWarning("Use of secondary ls mask is not compatible with mutliple common"
                            "catchments label mode. Color bar will not be correct.")
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    tic_loc = (np.arange(num_colors) + 0.5)
    if use_only_one_color_for_flowmap:
        tic_labels = ['Sea', 'Minor catchments','{} river path'.format(second_datasource_name)]
    else:
        tic_labels = ['Sea', 'Minor catchments','{} river path'.format(first_datasource_name),
                      'Common river path','{} river path'.format(second_datasource_name)]
    if use_single_color_for_discrepancies:
        tic_labels.extend(["{} in catchments".format(difference_in_catchment_label),
                           "Common catchment"])
        if not use_only_one_common_catchment_label:
            tic_labels.extend(["Common catchment","Common catchment"])
    else:
        tic_labels.extend(["Model 1 catchment","Common catchment","Model 2 catchment",
                           "Common catchment","Common catchment"])
    if plot_glaciers:
        tic_labels.extend(["Glacier"])
    if second_ls_mask:
        tic_labels[2:2] = ["Shelves exposed at {}".format(second_datasource_name)]
        if not no_extra_sea:
          tic_labels[1:1] = ["Additional Sea at {}".format(second_datasource_name)]
    cb.set_ticks(tic_loc)
    cb.set_ticklabels(tic_labels)
    cb.ax.tick_params(labelsize=20)

