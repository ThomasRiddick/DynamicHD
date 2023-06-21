'''
Routines to plot a comparison of rivers and catchments between an evaluated dataset and a
reference dataset.

Created on Jul 21, 2016

@author: thomasriddick
'''

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
from HD_Plots.utilities import plotting_tools as pts

def find_catchment_numbers(ref_catchment_field,data_catchment_field,
                           data_catchment_field_original_scale,
                           data_original_scale_flowtocellfield,pair,
                           catchment_grid_changed,swap_ref_and_data_when_finding_labels=False,
                           use_original_scale_field_for_determining_data_and_ref_labels=False,
                           ref_original_scale_flowtocellfield=None,
                           ref_catchment_field_original_scale=None,
                           data_original_scale_grid_type='HD',
                           ref_original_scale_grid_type='HD',
                           data_original_scale_grid_kwargs={},
                           ref_original_scale_grid_kwargs={},
                           grid_type='HD',**grid_kwargs):
    """Find the catchment number of a reference and a data catchment"""
    if use_original_scale_field_for_determining_data_and_ref_labels:
        ref_coarse_coords=pair[0].get_coords()
        ref_catchment_num =\
            pts.find_data_catchment_number(ref_catchment_field,
                                           ref_catchment_field_original_scale,
                                           ref_original_scale_flowtocellfield,ref_coarse_coords,
                                           catchment_grid_changed,ref_original_scale_grid_type,
                                           ref_original_scale_grid_kwargs,grid_type,**grid_kwargs)[0]
        data_coarse_coords=pair[1].get_coords()
        data_catchment_num,scale_factor =\
            pts.find_data_catchment_number(data_catchment_field,
                                           data_catchment_field_original_scale,
                                           data_original_scale_flowtocellfield,data_coarse_coords,
                                           catchment_grid_changed,data_original_scale_grid_type,
                                           data_original_scale_grid_kwargs,grid_type,**grid_kwargs)
    elif swap_ref_and_data_when_finding_labels:
        data_catchment_num = data_catchment_field[pair[0].get_coords()]
        ref_coarse_coords=pair[1].get_coords()
        ref_catchment_num,scale_factor =\
            pts.find_data_catchment_number(ref_catchment_field,
                                           data_catchment_field_original_scale,
                                           data_original_scale_flowtocellfield,ref_coarse_coords,
                                           catchment_grid_changed,data_original_scale_grid_type,
                                           data_original_scale_grid_kwargs,grid_type,**grid_kwargs)
    else:
        ref_catchment_num = ref_catchment_field[pair[0].get_coords()]
        data_coarse_coords=pair[1].get_coords()
        data_catchment_num,scale_factor =\
            pts.find_data_catchment_number(data_catchment_field,
                                           data_catchment_field_original_scale,
                                           data_original_scale_flowtocellfield,data_coarse_coords,
                                           catchment_grid_changed,data_original_scale_grid_type,
                                           data_original_scale_grid_kwargs,grid_type,**grid_kwargs)
    return ref_catchment_num,data_catchment_num,scale_factor

def select_bounds_around_rivermouth(pair,border=10):
    """Select a set of bounds for a plot of a river mouth"""
    imin = max(min(pair[0].get_lat(),pair[1].get_lat()) - border,0)
    imax = max(pair[0].get_lat(),pair[1].get_lat()) + border
    jmin = max(min(pair[0].get_lon(),pair[1].get_lon()) - border,0)
    jmax = max(pair[0].get_lon(),pair[1].get_lon()) + border
    return imin,imax,jmin,jmax

def plot_orography_section(ax,cax,section,min_height,max_height,outflow_lat,outflow_lon,new_cb=False,
                           num_levels=50,plot_type='Image',alpha=1.0,cmap_name="terrain",
                           lat_offset=0,lon_offset=0,
                           point_label_coords_scaling=1):
    """Plot a section from an orography field"""
    if plot_type=='Filled Contour' or plot_type=='Contour':
        levels = np.linspace(min_height,max_height,num_levels)
        if plot_type=='Filled Contour':
            cs = ax.contourf(section,levels=levels,alpha=alpha,cmap=cm.get_cmap('viridis'))
        else:
            cs = ax.contour(section,levels=levels,alpha=alpha)
    elif plot_type=='Image':
        im = ax.imshow(section,interpolation='nearest',cmap=cm.get_cmap(cmap_name),
                       norm = cm.colors.Normalize(vmax=max_height,vmin=min_height),rasterized=True)
    ax.format_coord = pts.OrogCoordFormatter(lon_offset,lat_offset,add_latlon=True,
                                             scale_factor=point_label_coords_scaling)
    ax.set_title(" Outflow"
                 " Lat: " + (lambda x: "{:.2f}".format((0.5*x - 90)*(-1 if x<=180 else 1))
                             + r'$^{\circ}$' + ('N' if x <=180 else 'S'))(outflow_lat)
                 + ' Lon: ' + (lambda y: "{:.2f}".format((0.5*y-180)*(-1 if y<= 360 else 1))
                               + r'$^{\circ}$' + ('W' if y <= 360 else 'E'))(outflow_lon))
    ax.axis('image')
    if not new_cb:
        cax.clear()
    if plot_type=='Filled Contour' or plot_type=='Contour':
        cb = plt.colorbar(cs,cax=cax)
    elif plot_type=='Image':
        cb = plt.colorbar(im,cax=cax)
    cb.ax.set_title("Height\n Above\n Sea\n Level\n (m)",fontsize=8)

def select_rivermaps_section(ref_flowtocellfield,data_flowtocellfield,
                             rdirs_field,imin,imax,jmin,jmax,threshold,
                             points_to_mark,mark_true_sinks=False,
                             data_true_sinks = None,
                             allow_new_sink_points=False):
    """Select a section of a flow to cell field"""
    rmap_field = np.copy(ref_flowtocellfield)
    rmap_field[rdirs_field <= 0] = 0
    rmap_field[np.logical_and(threshold > rmap_field,rmap_field > 0)] = 1
    rmap_field[rmap_field >= threshold] = 2
    rmap_field[np.logical_and(rmap_field == 2,data_flowtocellfield >= threshold)] = 3
    rmap_field[np.logical_and(rmap_field != 3,data_flowtocellfield >= threshold)] = 4
    if mark_true_sinks:
        rmap_field[rdirs_field == 5] = 0
    if points_to_mark:
        for i,point in enumerate(points_to_mark,start=5):
            rmap_field[point] = i
    if data_true_sinks is not None:
        rmap_field[rdirs_field == 5] = 7
        rmap_field[np.logical_and(rdirs_field == 5,data_true_sinks)] = 8
        if np.any(np.logical_and(rdirs_field != 5,data_true_sinks)):
            if allow_new_sink_points:
                rmap_field[np.logical_and(rdirs_field != 5,data_true_sinks)] = 9
            else:
                raise RuntimeWarning("New true sink point has appeared in data")
    return rmap_field[imin:imax,jmin:jmax]

def plot_whole_river_flowmap(ax,pair,ref_flowtocellfield,data_flowtocellfield,rdirs_field,data_rdirs_field,
                        catchment_bounds,allow_new_sink_points=False,simplified_flowmap_plot=False,colors=None):
    points_to_mark = [pair[0].get_coords(),pair[1].get_coords()]
    rmap_threshold_wholecatch = 25*9
    whole_catchment_rmap_section = select_rivermaps_section(ref_flowtocellfield,data_flowtocellfield,
                                                            rdirs_field,*catchment_bounds,
                                                            threshold=rmap_threshold_wholecatch,
                                                            points_to_mark=points_to_mark,
                                                            mark_true_sinks=True,
                                                            data_true_sinks=(data_rdirs_field == 5),
                                                            allow_new_sink_points=allow_new_sink_points)
    plot_flowmap(ax, section=whole_catchment_rmap_section,colors=colors,reduced_map=simplified_flowmap_plot)
    plt.subplots_adjust(hspace=0.25,left=0.1)

def plot_flowmap(ax,section,reduced_map=False,cax=None,
                 interpolation='none',alternative_colors=False,
                 remove_ticks_flag=True,colors=None):
    if reduced_map:
        num_colors = 5
    else:
        num_colors = 9
    cmap_wholec,norm_wholec = create_colormap(section,num_colors=num_colors,
                                              alternative_colors=alternative_colors,
                                              colors=colors)
    ax.imshow(section,interpolation=interpolation,cmap=cmap_wholec,norm=norm_wholec,rasterized=True)
    if remove_ticks_flag:
        pts.remove_ticks(ax)
    if reduced_map:
        mappable_wc = mpl.cm.ScalarMappable(norm=norm_wholec,cmap=cmap_wholec)
        mappable_wc.set_array(section)
        cb_wc = plt.colorbar(mappable_wc,ax=ax,cax=cax)
        tic_labels_wc = ['Sea', 'Land','Reference River Path','Common River Path','Data River Path']
        tic_loc_wc = np.arange(5) + 0.5
        cb_wc.set_ticks(tic_loc_wc)
        cb_wc.set_ticklabels(tic_labels_wc)
    else:
        mappable_wc = mpl.cm.ScalarMappable(norm=norm_wholec,cmap=cmap_wholec)
        mappable_wc.set_array(section)
        cb_wc = plt.colorbar(mappable_wc,ax=ax,cax=cax)
        tic_labels_wc = ['Sea', 'Land','Reference River Path','Common River Path','Data River Path',
                         'Reference River Mouth','Data River Mouth','Reference True Sink',
                         'Common True Sink']
        tic_loc_wc = np.arange(9) + 0.5
        cb_wc.set_ticks(tic_loc_wc)
        cb_wc.set_ticklabels(tic_labels_wc)

def plot_river_rmouth_flowmap(ax,ref_flowtocellfield,data_flowtocellfield,rdirs_field,pair,colors,
                              point_label_coords_scaling=1):
    points_to_mark = [pair[0].get_coords(),pair[1].get_coords()]
    bounds = select_bounds_around_rivermouth(pair,border=20)
    rmap_threshold = 0.5*min(pair[0].get_outflow(),pair[1].get_outflow())
    rmap_section = select_rivermaps_section(ref_flowtocellfield,data_flowtocellfield,
                                            rdirs_field,*bounds,
                                            threshold=rmap_threshold,
                                            points_to_mark=points_to_mark,
                                            allow_new_sink_points=False)
    cmap,norm = create_colormap(rmap_section,colors=colors)
    ax.imshow(rmap_section,interpolation='none',cmap=cmap,norm=norm,rasterized=True)
    plt.title(" Lat: " + (lambda x: "{:.2f}".format((0.5*x - 90)*(-1 if x<=180 else 1))
                          + r'$^{\circ}$' + ('N' if x <=180 else 'S'))(pair[0].get_lat()/
                                                                       point_label_coords_scaling)
              + ' Lon: ' + (lambda y: "{:.2f}".format((0.5*y-180)*(-1 if y<= 360 else 1))
                            + r'$^{\circ}$' + ('W' if y <= 360 else 'E'))(pair[0].get_lon()/
                                                                          point_label_coords_scaling))
    mappable = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)
    mappable.set_array(rmap_section)
    pts.remove_ticks(ax)
    cb = plt.colorbar(mappable,ax=ax)
    tic_loc = np.arange(7) + 0.5
    tic_labels = ['Sea', 'Land','Reference River Path','Common River Path','Data River Path',
                  'Reference River Mouth','Data River Mouth']
    cb.set_ticks(tic_loc)
    cb.set_ticklabels(tic_labels)

def plot_catchment_and_histogram_for_river(ax_hist,ax_catch,ref_catchment_field,
                                           data_catchment_field,
                                           data_catchment_field_original_scale,
                                           data_original_scale_flowtocellfield,
                                           rdirs_field,data_rdirs_field,pair,
                                           catchment_grid_changed,colors,
                                           ref_grid,grid_type,
                                           swap_ref_and_data_when_finding_labels=False,
                                           alternative_catchment_bounds=None,
                                           use_simplified_catchment_colorscheme=False,
                                           use_upscaling_labels=False,
                                           ref_original_scale_flowtocellfield=None,
                                           ref_catchment_field_original_scale=None,
                                           allow_new_sink_points=False,
                                           use_original_scale_field_for_determining_data_and_ref_labels=False,
                                           external_landsea_mask = None,
                                           return_catchment_plotter=False,
                                           data_original_scale_grid_type='HD',
                                           ref_original_scale_grid_type='HD',
                                           data_original_scale_grid_kwargs={},
                                           ref_original_scale_grid_kwargs={},
                                           **grid_kwargs):
    points_to_mark = [pair[0].get_coords(),pair[1].get_coords()]
    indices = np.arange(2)
    width = 0.45
    offset = 0.1
    bar_labels = ('Reference','Data')
    bar_heights = (pair[0].get_outflow(),pair[1].get_outflow())
    ax_hist.bar(indices+offset,bar_heights,2*width)
    ax_hist.set_xticks(indices+width+offset)
    ax_hist.set_xticklabels(bar_labels)
    ax_hist.set_ylabel("Cumulative Flow to Cell (Number of Cells)")
    ax_hist.set_xlim(0,indices[-1]+2*width+2*offset)
    ax_hist.set_title("Change in Flow: {0:.1f}%".format(pair[1].get_outflow()
                                                        *100.0/pair[0].get_outflow() - 100.0))
    ref_catchment_num,data_catchment_num,scale_factor = find_catchment_numbers(ref_catchment_field,data_catchment_field,
                                                                               data_catchment_field_original_scale,
                                                                               data_original_scale_flowtocellfield,pair,
                                                                               catchment_grid_changed,
                                                                               swap_ref_and_data_when_finding_labels=\
                                                                               swap_ref_and_data_when_finding_labels,
                                                                               use_original_scale_field_for_determining_data_and_ref_labels=\
                                                                               use_original_scale_field_for_determining_data_and_ref_labels,
                                                                               ref_original_scale_flowtocellfield=\
                                                                               ref_original_scale_flowtocellfield,
                                                                               ref_catchment_field_original_scale=\
                                                                               ref_catchment_field_original_scale,
                                                                               data_original_scale_grid_type=data_original_scale_grid_type,
                                                                               data_original_scale_grid_kwargs=data_original_scale_grid_kwargs,
                                                                               ref_original_scale_grid_type=ref_original_scale_grid_type,
                                                                               ref_original_scale_grid_kwargs=ref_original_scale_grid_kwargs,
                                                                               grid_type=grid_type,**grid_kwargs)
    catchment_section,catchment_bounds = select_catchment(ref_catchment_field,
                                                          data_catchment_field,
                                                          rdirs_field,
                                                          ref_catchment_num,
                                                          data_catchment_num,
                                                          points_to_mark,
                                                          data_true_sinks=(data_rdirs_field == 5),
                                                          allow_new_sink_points=allow_new_sink_points,
                                                          alternative_catchment_bounds=\
                                                          alternative_catchment_bounds,
                                                          use_simplified_catchment_colorscheme=\
                                                          use_simplified_catchment_colorscheme,
                                                          external_landsea_mask=external_landsea_mask)
    plot_catchment(ax_catch,catchment_section,colors,
                   lat_offset=catchment_bounds[0],
                   lon_offset=catchment_bounds[2],
                   simplified_colorscheme=use_simplified_catchment_colorscheme,
                   cax=None,use_upscaling_labels=use_upscaling_labels,
                   format_coords=True,
                   remove_ticks_flag=False)
    if return_catchment_plotter:
        catchment_plotter = CatchmentPlotter(catchment_section,colors,
                                             lat_offset=catchment_bounds[0],
                                             lon_offset=catchment_bounds[2],
                                             simplified_colorscheme=use_simplified_catchment_colorscheme,
                                             cax=None,use_upscaling_labels=use_upscaling_labels,
                                             format_coords=True,
                                             remove_ticks_flag=False)
    axis_tick_label_scale_factor=ref_grid.get_scale_factor_for_geographic_coords()
    catch_x_axis_major_locator = mpl.ticker.IndexLocator(5//axis_tick_label_scale_factor,
                                                             -catchment_bounds[2])
    ax_catch.xaxis.set_major_locator(catch_x_axis_major_locator)
    catch_y_axis_major_locator = mpl.ticker.IndexLocator(5//axis_tick_label_scale_factor,
                                                             -catchment_bounds[0])
    ax_catch.yaxis.set_major_locator(catch_y_axis_major_locator)
    #Scale factor is multiplied by two as formatter has a built in scale factor of a half
    catch_x_axis_major_formatter = mpl.ticker.\
        FuncFormatter(pts.LonAxisFormatter((catchment_bounds[2] +
                                            ref_grid.get_longitude_offset_adjustment())*
                                           axis_tick_label_scale_factor,
                                           2/axis_tick_label_scale_factor))
    ax_catch.xaxis.set_major_formatter(catch_x_axis_major_formatter)
    catch_y_axis_major_formatter = mpl.ticker.\
        FuncFormatter(pts.LatAxisFormatter(catchment_bounds[0]*axis_tick_label_scale_factor,
                                           2/axis_tick_label_scale_factor))
    ax_catch.yaxis.set_major_formatter(catch_y_axis_major_formatter)
    if return_catchment_plotter:
        catchment_plotter.add_axis_locators_and_formatters(major_x_axis_formatter=\
                                                           catch_x_axis_major_formatter,
                                                           major_y_axis_formatter=\
                                                           catch_y_axis_major_formatter,
                                                           major_x_axis_locator=\
                                                           catch_x_axis_major_locator,
                                                           major_y_axis_locator=\
                                                           catch_y_axis_major_locator)
    plt.tight_layout()
    if return_catchment_plotter:
        return catchment_section,catchment_bounds,scale_factor,catchment_plotter
    else:
        return catchment_section,catchment_bounds,scale_factor

class CatchmentPlotter:

    def __init__(self,catchment_section,colors,simplified_colorscheme=False,
                 cax=None,legend=True,remove_ticks_flag=True,
                 format_coords=False,lat_offset=0,lon_offset=0,
                 use_upscaling_labels=False):
        self.catchment_section=catchment_section
        self.colors = colors
        self.simplified_colorscheme = simplified_colorscheme
        self.cax = cax
        self.legend = legend
        self.remove_ticks_flag = remove_ticks_flag
        self.format_coords = format_coords
        self.lat_offset = lat_offset
        self.lon_offset = lon_offset
        self.use_upscaling_labels = use_upscaling_labels

    def set_legend(self,legend):
        self.legend=legend

    def set_cax(self,cax):
        self.cax = cax

    def add_axis_locators_and_formatters(self,major_x_axis_formatter,major_y_axis_formatter,
                                         major_x_axis_locator,major_y_axis_locator):
        self.major_x_axis_formatter = major_x_axis_formatter
        self.major_y_axis_formatter = major_y_axis_formatter
        self.major_x_axis_locator = major_x_axis_locator
        self.major_y_axis_locator = major_y_axis_locator

    def apply_axis_locators_and_formatters(self,ax):
        ax.xaxis.set_major_locator(self.major_x_axis_locator)
        ax.yaxis.set_major_locator(self.major_y_axis_locator)
        ax.xaxis.set_major_formatter(self.major_x_axis_formatter)
        ax.yaxis.set_major_formatter(self.major_y_axis_formatter)

    def __call__(self,ax):
        plot_catchment(ax,catchment_section=self.catchment_section,
                       colors=self.colors,
                       simplified_colorscheme=self.colors,
                       cax=self.cax,
                       legend=self.legend,
                       remove_ticks_flag=self.remove_ticks_flag,
                       format_coords=self.format_coords,
                       lat_offset=self.lat_offset,
                       lon_offset=self.lon_offset,
                       use_upscaling_labels=self.use_upscaling_labels)

def plot_catchment(ax,catchment_section,colors,simplified_colorscheme=False,
                   cax=None,legend=True,remove_ticks_flag=True,
                   format_coords=False,lat_offset=0,lon_offset=0,
                   use_upscaling_labels=False,
                   point_label_coords_scaling=1):
    if simplified_colorscheme:
        cmap_catch,norm_catch = create_colormap(catchment_section,5,colors=colors)
    else:
        cmap_catch,norm_catch = create_colormap(catchment_section,9,colors=colors)
    ax.imshow(catchment_section,interpolation='none',cmap=cmap_catch,norm=norm_catch,rasterized=True)
    if format_coords:
        ax.format_coord = pts.OrogCoordFormatter(lon_offset,lat_offset,add_latlon=True,
                                                 scale_factor=point_label_coords_scaling)
    if remove_ticks_flag:
        pts.remove_ticks(ax)
    if legend:
        mappable_catch = mpl.cm.ScalarMappable(norm=norm_catch,cmap=cmap_catch)
        mappable_catch.set_array(catchment_section)
        cb_wc = plt.colorbar(mappable_catch,ax=ax,cax=cax)
        if use_upscaling_labels:
            tic_labels_catch = ['Sea','Land','Fine river directions\ncatchment','Common catchment',
                                'Upscaled river directions\ncatchment','Fine river directions river mouth',
                                'Upscaled river directions river mouth','Fine river directions true sink',
                                'Common true sink']
        else:
            tic_labels_catch = ['Sea','Land','Default HD catchment','Common catchment','Dynamic HD catchment',
                                'Model 1 river mouth','Model 2 river mouth','Model 1 true sink',
                                'Common true sink']
        tic_loc_catch = np.arange(9) + 0.5
        cb_wc.set_ticks(tic_loc_catch)
        cb_wc.set_ticklabels(tic_labels_catch)

def select_catchment(ref_catchment_field,data_catchment_field,
                     rdirs_field,ref_catchment_num,
                     data_catchment_num,points_to_mark=None,
                     data_true_sinks=None,
                     allow_new_sink_points=False,
                     alternative_catchment_bounds=None,
                     use_simplified_catchment_colorscheme=False,
                     external_landsea_mask=None):
    """Prepare a catchment field that combines both the reference and data catchments"""
    catchment_field = np.copy(ref_catchment_field)
    catchment_field[np.logical_and(data_catchment_field == data_catchment_num,
                                   ref_catchment_field == ref_catchment_num)] = 3
    catchment_field[np.logical_and(data_catchment_field == data_catchment_num,
                                   ref_catchment_field != ref_catchment_num)] = 4
    catchment_field[np.logical_and(data_catchment_field != data_catchment_num,
                                   ref_catchment_field == ref_catchment_num)] = 2
    catchment_field[np.logical_and(data_catchment_field != data_catchment_num,
                                   ref_catchment_field != ref_catchment_num)] = 1
    if external_landsea_mask is None:
        catchment_field[rdirs_field <= 0] = 0
    else:
        catchment_field[external_landsea_mask == 1] = 0
    if not use_simplified_catchment_colorscheme:
        catchment_field[rdirs_field == 5] = 7
    if not use_simplified_catchment_colorscheme:
        for i,point in enumerate(points_to_mark,start=5):
            catchment_field[point] = i
            if data_true_sinks is not None:
                catchment_field[np.logical_and(rdirs_field == 5,data_true_sinks)] = 8
                if np.any(np.logical_and(rdirs_field != 5,data_true_sinks)):
                    if allow_new_sink_points:
                        catchment_field[np.logical_and(rdirs_field != 5,data_true_sinks)] = 9
                    else:
                        raise RuntimeWarning("New true sink point has appeared in data")
    if alternative_catchment_bounds is None:
        imin,imax,jmin,jmax = find_catchment_edges(catchment_field)
    else:
        imin,imax,jmin,jmax = alternative_catchment_bounds
    return catchment_field[imin:imax,jmin:jmax], (imin,imax,jmin,jmax)

def find_catchment_edges(catchment_field,border = 5):
    """Find the edges of a given catchment and add a border before returning them"""
    is_catchment = np.logical_not(np.logical_or(np.logical_or(np.logical_or(catchment_field == 0,
                                                                            catchment_field == 1),
                                                              catchment_field == 7),
                                                np.logical_or(catchment_field == 8,
                                                              catchment_field == 9)))
    ivalues,jvalues = is_catchment.nonzero()
    imin = np.amin(ivalues) - border
    imax = np.amax(ivalues) + border
    jmin = np.amin(jvalues) - border
    jmax = np.amax(jvalues) + border
    if imin < 0:
        imin = 0
    if imax >= is_catchment.shape[0]:
        imax = is_catchment.shape[0] - 1
    if jmin < 0:
        jmin = 0
    if jmax >= is_catchment.shape[1]:
        jmax = is_catchment.shape[1] - 1
    return imin,imax,jmin,jmax

def create_colormap(field,num_colors=7,alternative_colors=False,colors=None):
    """Generate a discrete colormap"""
    if colors is not None:
        color_list = colors.create_colormap_colors
        alternative_color_list = colors.create_colormap_alternative_colors
    else:
        color_list = ['blue','peru','yellow','white','gray','red','black',
                      'indigo','deepskyblue']
        alternative_color_list = ['blue','peru','blueviolet','black','red','gray','green',
                                  'yellow','deepskyblue']
    cmap = mpl.colors.ListedColormap(color_list[:num_colors])
    if alternative_colors:
        cmap = mpl.colors.ListedColormap(alternative_color_list[:num_colors])
    bounds = list(range(num_colors+1))
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    return cmap,norm

def simple_catchment_and_flowmap_plots(fig,ref_ax,data_ax,ref_catchment_field,data_catchment_field,
                                       data_catchment_field_original_scale,ref_flowtocellfield,
                                       data_flowtocellfield,data_original_scale_flowtocellfield,pair,
                                       catchment_bounds,flowtocell_threshold,
                                       catchment_grid_changed,colors,grid_type,
                                       data_original_scale_grid_type,data_original_scale_grid_kwargs={},
                                       external_ls_mask=None,**grid_kwargs):
    if external_ls_mask is not None:
        external_ls_mask = np.logical_not(external_ls_mask.astype(np.bool_)).astype(type(external_ls_mask))
    fig.suptitle('River catchment plus cells with a cumulative flow greater than {0}'.format(flowtocell_threshold))
    ref_ax.set_title('Reference')
    data_ax.set_title('Data')
    ref_catchment_num,data_catchment_num = find_catchment_numbers(ref_catchment_field,
                                                                  data_catchment_field,
                                                                  data_catchment_field_original_scale,
                                                                  data_original_scale_flowtocellfield,
                                                                  pair, catchment_grid_changed,
                                                                  data_original_scale_grid_type=\
                                                                  data_original_scale_grid_type,
                                                                  data_original_scale_grid_kwargs=\
                                                                  data_original_scale_grid_kwargs,
                                                                  grid_type=grid_type)[0:2]
    simple_catchment_and_flowmap_plot_object_ref = SimpleCatchmentAndFlowMapPlt(catchment_field=\
                                                                                ref_catchment_field,
                                                                                catchment_field_for_lsmask=\
                                                                                data_catchment_field if \
                                                                                external_ls_mask is None else
                                                                                external_ls_mask,
                                                                                flowtocell=ref_flowtocellfield,
                                                                                catchment_bounds=\
                                                                                catchment_bounds,
                                                                                catchment_num=\
                                                                                ref_catchment_num,
                                                                                flowtocell_threshold=\
                                                                                flowtocell_threshold,
                                                                                colors=colors)
    simple_catchment_and_flowmap_plot_object_ref(ref_ax)
    simple_catchment_and_flowmap_plot_object_data = SimpleCatchmentAndFlowMapPlt(catchment_field=\
                                                                                 data_catchment_field,
                                                                                 catchment_field_for_lsmask=\
                                                                                 data_catchment_field if\
                                                                                 external_ls_mask is None else
                                                                                 external_ls_mask,
                                                                                 flowtocell=data_flowtocellfield,
                                                                                 catchment_bounds=\
                                                                                 catchment_bounds,
                                                                                 catchment_num=\
                                                                                 data_catchment_num,
                                                                                 flowtocell_threshold=\
                                                                                 flowtocell_threshold,
                                                                                 colors=colors)
    simple_catchment_and_flowmap_plot_object_data(data_ax)
    return simple_catchment_and_flowmap_plot_object_ref,simple_catchment_and_flowmap_plot_object_data

class SimpleCatchmentAndFlowMapPlt:

    def __init__(self,catchment_field,catchment_field_for_lsmask,flowtocell,
                 catchment_bounds,catchment_num,flowtocell_threshold,colors):
        self.catchment_field = catchment_field
        self.catchment_field_for_lsmask = catchment_field_for_lsmask
        self.flowtocell = flowtocell
        self.catchment_bounds = catchment_bounds
        self.catchment_num = catchment_num
        self.flowtocell_threshold = flowtocell_threshold
        self.colors = colors

    def __call__(self,ax):
        simple_catchment_and_flowmap_plot(ax,self.catchment_field,self.catchment_field_for_lsmask,
                                          self.flowtocell,self.catchment_bounds,self.catchment_num,
                                          self.flowtocell_threshold,self.colors)

    def get_flowtocell_threshold(self):
        return self.flowtocell_threshold

def simple_catchment_and_flowmap_plot(ax,catchment_field,catchment_field_for_lsmask,flowtocell,
                                      catchment_bounds,catchment_num,flowtocell_threshold,colors,
                                      remove_ticks=False):
    """Simple version of overlaid catchment and flowmaps plot"""
    working_catchment = np.copy(catchment_field)
    imin,imax,jmin,jmax = catchment_bounds
    working_catchment[catchment_field == catchment_num] = 2
    working_catchment[catchment_field != catchment_num] = 1
    working_catchment[catchment_field_for_lsmask <= 0] = 0
    working_catchment[flowtocell > flowtocell_threshold] = 4
    working_catchment_section = working_catchment[imin:imax,jmin:jmax]
    cmap_catch = mpl.colors.ListedColormap(colors.simple_catchment_and_flowmap_colors)
    norm_catch = mpl.colors.BoundaryNorm(list(range(5)),cmap_catch.N)
    ax.imshow(working_catchment_section,interpolation='none',cmap=cmap_catch,norm=norm_catch,rasterized=True)
    if remove_ticks:
        pts.remove_ticks(ax)
    else:
            #Assume HD Grid
            axis_tick_label_scale_factor=0.5
            ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(10//axis_tick_label_scale_factor,
                                                               -catchment_bounds[2]))
            ax.yaxis.set_major_locator(mpl.ticker.IndexLocator(10//axis_tick_label_scale_factor,
                                                               -catchment_bounds[0]))
            #Scale factor is multiplied by two as formatter has a built in scale factor of a half
            ax.xaxis.set_major_formatter(mpl.ticker.\
                                               FuncFormatter(pts.LonAxisFormatter(catchment_bounds[2],
                                                            axis_tick_label_scale_factor*2)))
            ax.yaxis.set_major_formatter(mpl.ticker.\
                                               FuncFormatter(pts.LatAxisFormatter(catchment_bounds[0],
                                                            axis_tick_label_scale_factor*2)))
            plt.tight_layout()


def add_catchment_and_outflow_to_river(catchments,outflows,sink_outflow_to_remap,processing_mod_type,
                                       flowmap=None,rdirs=None,original_scale_catchment=None,
                                       original_scale_flowmap=None,catchment_grid_changed=False,
                                       original_scale_grid_type=None,original_scale_grid_kwargs={},
                                       grid_type=None,**grid_kwargs):
    """Moves given sink or river catchment and outflow to a given river outflow and relabel catchment accordingly"""
    original_outflow_coords = sink_outflow_to_remap[0]
    new_outflow_coords = sink_outflow_to_remap[1]
    print("Moving outflow from point lat={0}, lon={1} to point lat={2}, lon={3}"\
          " and relabeling catchment accordingly".format(original_outflow_coords[0],original_outflow_coords[1],
                                                         new_outflow_coords[0],new_outflow_coords[1]))

    if processing_mod_type == 'Sink':
        outflows.set_data(pts.move_outflow(outflows.get_data(), original_outflow_coords, new_outflow_coords,
                                           flowmap,None))
        catchments = pts.relabel_outflow_catchment(catchments, original_outflow_coords,
                                                   new_outflow_coords)
    elif processing_mod_type == 'River Mouth':
        outflows.set_data(pts.move_outflow(outflows.get_data(), original_outflow_coords, new_outflow_coords,
                                           None,None))
        catchments = pts.relabel_outflow_catchment(catchments.get_data(),
                                                   original_outflow_coords,
                                                   new_outflow_coords,
                                                   original_scale_catchment.get_data()
                                                   if original_scale_catchment is not None else
                                                   None,
                                                   original_scale_flowmap.get_data()
                                                   if original_scale_flowmap is not None else
                                                   None,
                                                   catchment_grid_changed,False,
                                                   original_scale_grid_type,
                                                   original_scale_grid_kwargs,
                                                   grid_type,**grid_kwargs)
    elif processing_mod_type == 'Ref River Mouth':
        outflows.set_data(pts.move_outflow(outflows.get_data(), original_outflow_coords, new_outflow_coords,
                                           None,rdirs.get_data()))
        catchments = pts.relabel_outflow_catchment(catchments.get_data(),original_outflow_coords,new_outflow_coords,
                                                   None)
    elif processing_mod_type == 'Ref Join Catchments':
        catchments = pts.relabel_outflow_catchment(catchments, original_outflow_coords,
                                                   new_outflow_coords,
                                                   original_scale_catchment,
                                                   original_scale_flowmap,
                                                   catchment_grid_changed,True,
                                                   original_scale_grid_type,
                                                   original_scale_grid_kwargs,
                                                   grid_type,**grid_kwargs)
    else:
        raise RuntimeError('Trying to process unknown modification type')
    return catchments,outflows

def modify_catchments_and_outflows(ref_catchments,ref_outflows,ref_flowmap,ref_rdirs,
                                   data_catchments,data_outflows,
                                   catchment_and_outflows_modifications_list_filename,
                                   original_scale_catchment=None,original_scale_flowmap=None,
                                   catchment_grid_changed=False,
                                   swap_ref_and_data_when_finding_labels=False,
                                   original_scale_grid_type=None,original_scale_grid_kwargs={},
                                   grid_type=None,**grid_kwargs):
    """Load various temporary modifications to make to the catchment and outflows to aid comparison

    Modifications are applied after fields have been orientated
    """

    first_line_pattern = re.compile(r"^ *Catchment and Outflow Modifications *$")
    sinks_first_line_pattern = re.compile(r"^ *Sinks to Move *$")
    mouths_first_line_pattern = re.compile(r"^ *Mouths to Move *$")
    ref_mouths_first_line_pattern = re.compile(r"^ *Ref Mouths to Move *$")
    ref_join_catchments_first_line_pattern = re.compile(r"^ *Ref Catchments to Join *$")
    mod_second_line_pattern = re.compile(r"^ *grid_type= *(.*) *$")
    mod_third_line_pattern = re.compile(r"^ *original_lat *, *original_lon *, *new_lat *, *new_lon *$")
    line_of_input_pattern = re.compile(r"^ *[0-9]* *, *[0-9]* *, *[0-9]* *, *[0-9]* *$")
    def process_modification_generator(catchments,outflows,processing_mod_type,
                                       flowmap=None,rdirs=None,original_scale_catchment=None,
                                       original_scale_flowmap=None,catchment_grid_changed=False,
                                       original_scale_grid_type=None,original_scale_grid_kwargs={},
                                       grid_type=None,**grid_kwargs):
        mod_second_line_pattern_match = mod_second_line_pattern.match(line.strip())
        if not mod_second_line_pattern_match:
            raise RuntimeError("Mistake in file containing list of modifications"
                                   " at line {0}".format(i))
        grid_type_from_file = mod_second_line_pattern_match.group(0)
        if grid_type != grid_type_from_file:
            RuntimeError("File containing list of modifications is for wrong grid type")
        yield catchments,outflows
        if not mod_third_line_pattern.match(line.strip()):
            raise RuntimeError("Mistake in file containing list of modifications"
                                   " at line {0}".format(i))
        yield catchments,outflows
        while True:
            if not line_of_input_pattern.match(line):
                raise RuntimeError("Mistake in file containing list of modifications"
                                   " at line {0}".format(i))
            original_lat,original_lon,new_lat,new_lon=\
                [int(coord) for coord in line.strip().split(",")]
            add_catchment_and_outflow_to_river(catchments, outflows,
                                               [(original_lat,original_lon),
                                                (new_lat,new_lon)],
                                               processing_mod_type,
                                               flowmap,ref_rdirs,
                                               original_scale_catchment,
                                               original_scale_flowmap,
                                               catchment_grid_changed,
                                               original_scale_grid_type,
                                               original_scale_grid_kwargs,
                                               grid_type,**grid_kwargs)
            yield catchments,outflows
    comment_line_pattern = re.compile(r"^ *#.*$")
    proc_mod_gen = None
    process_mod_type = ""
    with open(catchment_and_outflows_modifications_list_filename) as f:
        if not first_line_pattern.match(f.readline().strip()):
            raise RuntimeError("List of modifications being loaded has incorrect format the first line")
        for i,line in enumerate(f,start=2):
            if comment_line_pattern.match(line):
                continue
            if sinks_first_line_pattern.match(line):
                process_mod_type = 'Sink'
                proc_mod_gen = process_modification_generator(ref_catchments,ref_outflows,
                                                              process_mod_type,
                                                              flowmap=ref_flowmap,
                                                              rdirs=None,
                                                              grid_type=grid_type)
            elif mouths_first_line_pattern.match(line):
                process_mod_type = 'River Mouth'
                proc_mod_gen = process_modification_generator(data_catchments,data_outflows,
                                                              process_mod_type,None,None,
                                                              original_scale_catchment,
                                                              original_scale_flowmap,
                                                              catchment_grid_changed,
                                                              original_scale_grid_type,
                                                              original_scale_grid_kwargs,
                                                              grid_type,**grid_kwargs)
            elif ref_mouths_first_line_pattern.match(line):
                process_mod_type = 'Ref River Mouth'
                proc_mod_gen = process_modification_generator(ref_catchments,ref_outflows,
                                                              process_mod_type,None,ref_rdirs,
                                                              grid_type=grid_type)
            elif ref_join_catchments_first_line_pattern.match(line):
                process_mod_type = 'Ref Join Catchments'
                if swap_ref_and_data_when_finding_labels:
                    proc_mod_gen = process_modification_generator(ref_catchments,ref_outflows,
                                                                  process_mod_type,None,None,
                                                                  original_scale_catchment,
                                                                  original_scale_flowmap,
                                                                  catchment_grid_changed,
                                                                  original_scale_grid_type,
                                                                  original_scale_grid_kwargs,
                                                                  grid_type=grid_type)
                else:
                    proc_mod_gen = process_modification_generator(ref_catchments,ref_outflows,
                                                                  process_mod_type,None,ref_rdirs,
                                                                  grid_type=grid_type)
            elif (process_mod_type == 'Sink' or process_mod_type == 'Ref River Mouth' or
                  process_mod_type == 'Ref Join Catchments'):
                ref_catchments,ref_outflows = next(proc_mod_gen)
            elif process_mod_type == 'River Mouth':
                data_catchments,data_outflows = next(proc_mod_gen)
            else:
                raise RuntimeError("Mistake in file containing list of modifications:"
                                   "Have reached line {0} without first reading "
                                   "necessary headers".format(i))
    return ref_catchments,ref_outflows,data_catchments,data_outflows

def simple_thresholded_data_only_flowmap(ax,data_flowmap,lsmask,threshold=75,glacier_mask=None,
                                         catchments=None,catchnumone=-1,catchnumtwo=-1,
                                         catchnumthree=-1,bounds=None,cax=None,
                                         colors=None):
    """Plot cumulative flow over a threshold using single input field"""
    if bounds is None:
      data_flowmap_section = data_flowmap
      lsmask_section = lsmask
      if glacier_mask is not None:
        glacier_mask_section = glacier_mask
      if catchments is not None:
        catchments_section = catchments
    else:
      data_flowmap_section = data_flowmap[bounds[0]:bounds[1],bounds[2]:bounds[3]]
      lsmask_section = lsmask[bounds[0]:bounds[1],bounds[2]:bounds[3]]
      if glacier_mask is not None:
        glacier_mask_section = glacier_mask[bounds[0]:bounds[1],bounds[2]:bounds[3]]
      if catchments is not None:
        catchments_section = catchments[bounds[0]:bounds[1],bounds[2]:bounds[3]]
    thresholded_data_flowmap_section = np.zeros(data_flowmap_section.shape)
    thresholded_data_flowmap_section[np.logical_and(data_flowmap_section < threshold,lsmask_section)]  = 1
    thresholded_data_flowmap_section[np.logical_and(data_flowmap_section >= threshold,lsmask_section)] = 2
    if glacier_mask is not None:
      thresholded_data_flowmap_section[np.logical_and(glacier_mask_section >= 0.5,
                                                      thresholded_data_flowmap_section != 2)] = 3
    if catchments is not None:
      thresholded_data_flowmap_section[np.logical_and(catchments_section == catchnumone,
                                                      thresholded_data_flowmap_section == 1)] = 4
      thresholded_data_flowmap_section[np.logical_and(catchments_section == catchnumtwo,
                                                      thresholded_data_flowmap_section == 1)] = 5
      thresholded_data_flowmap_section[np.logical_and(catchments_section == catchnumthree,
                                                      thresholded_data_flowmap_section == 1)] = 6
    num_colors = 7
    color_list =  (colors.simple_thresholded_data_flowmap_and_catchments_colors
                   if colors is not None else
                  ['blue','peru','yellow','white','gray','red','black','indigo','deepskyblue'])
    cmap = mpl.colors.ListedColormap(color_list[:num_colors])
    colors_bounds = list(range(num_colors+1))
    norm = mpl.colors.BoundaryNorm(colors_bounds,cmap.N)
    ax.imshow(thresholded_data_flowmap_section,interpolation='none',cmap=cmap,norm=norm)
    #Assume HD Grid
    axis_tick_label_scale_factor=0.5
    ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(20//axis_tick_label_scale_factor,
                                                       -bounds[2]))
    ax.yaxis.set_major_locator(mpl.ticker.IndexLocator(20//axis_tick_label_scale_factor,
                                                       -bounds[0]))
    #Scale factor is multiplied by two as formatter has a built in scale factor of a half
    catch_x_axis_major_formatter = mpl.ticker.\
        FuncFormatter(pts.LonAxisFormatter(bounds[2] ,#+
                                           #ref_grid.get_longitude_offset_adjustment(),
                                           axis_tick_label_scale_factor*2))
    ax.xaxis.set_major_formatter(catch_x_axis_major_formatter)
    catch_y_axis_major_formatter = mpl.ticker.\
        FuncFormatter(pts.LatAxisFormatter(bounds[0],
                                           axis_tick_label_scale_factor*2))
    ax.yaxis.set_major_formatter(catch_y_axis_major_formatter)
    if cax is not None:
        mappable_wc = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)
        mappable_wc.set_array(thresholded_data_flowmap_section)
        cb_wc = plt.colorbar(mappable_wc,cax=cax)
        tic_labels_wc = ['Sea', 'Land','River Path','Glacier','Mississippi Catchment',"St Lawrence Catchment",
                         'Mackenzie Catchment']
        tic_loc_wc = np.arange(7) + 0.5
        cb_wc.set_ticks(tic_loc_wc)
        cb_wc.set_ticklabels(tic_labels_wc)
