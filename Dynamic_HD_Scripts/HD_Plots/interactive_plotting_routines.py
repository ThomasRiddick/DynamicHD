'''
Contains classes for generating interactive orography and riverflow comparison plots for 
large range of individual catchments

Created on Jul 21, 2016

@author: thomasriddick
'''
from matplotlib.widgets import Slider,Button,RadioButtons,CheckButtons
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import river_comparison_plotting_routines as rc_pts
import plotting_tools as pts
import numpy as np
import math

class Interactive_Plots(object):
    """The main class for the interactive orography and cumulative flow/river flow comparison plots"""

    numbers_max_size = 200

    def __init__(self):
        """Class constructor. Initialise permenant class members as required and set constants.
        
        Arguments:
        None
        Returns:
        Nothing
        """

        self.min_heights = []
        self.max_heights = []
        self.num_levels_sliders = []
        self.fmap_threshold_sliders = []
        self.reset_height_range_buttons = []
        self.linear_colors_buttons_list = []
        self.overlay_flowmap_buttons_list = []
        self.overlay_flowmap_on_super_fine_orog_buttons_list = []
        self.plot_type_radio_buttons_list = []
        self.secondary_orog_radio_buttons_list = []
        self.display_height_buttons_list = []
        self.match_left_scaling_callback_functions = []
        self.match_right_scaling_callback_functions = []
        self.match_catch_scaling_callback_functions = []
        self.ref_flowtocellfield_sections = []
        self.data_flowtocellfield_sections = []
        self.super_fine_data_flowtocellfield_sections = []
        self.rdirs_field_sections = []
        self.catchment_sections = []
        self.super_fine_orog_field_sections = []
        self.x_formatter_funcs = []
        self.y_formatter_funcs = []
        self.x_scaled_formatter_funcs = []
        self.y_scaled_formatter_funcs = []
        self.y_super_fine_scaled_formatter_funcs = []
        self.x_super_fine_scaled_formatter_funcs = []
        self.update_funcs = []
        self.using_numbers_ref = []
        self.using_numbers_data = []
        self.ref_orog_field_sections = []
        self.data_orog_original_scale_field_sections = []
        self.orog_gridspecs = []
        self.orog_rmouth_coords = []
        self.use_super_fine_orog_flags = []

        self.scale_factor = 1
        self.ref_to_super_fine_scale_factor = 1
        self.catchment_bounds = None
        self.catchment_section = None
        self.ref_orog_field = None
        self.data_orog_original_scale_field = None
        self.ref_flowtocellfield = None
        self.data_flowtocellfield = None
        self.rdirs_field = None
        self.super_fine_orog_field = None
        self.pair = None
    
    def setup_plots(self,catchment_section,ref_orog_field,data_orog_original_scale_field,
                    ref_flowtocellfield,data_flowtocellfield,rdirs_field,super_fine_orog_field,
                    super_fine_data_flowtocellfield,pair,catchment_bounds,scale_factor,
                    ref_to_super_fine_scale_factor):
        """Setup the set of interactive plots and bind on events to callbacks
        
        Arguments:
        catchment_section
        ref_orog_field
        data_orog_original_scale_field
        ref_flowtocellfield
        data_flowtocellfield
        rdirs_field
        super_fine_orog_field
        super_fine_data_flowtocellfield
        pair
        catchment_bounds
        scale_factor
        ref_to_super_fine_scale_facto
        Returns: nothing
        """
        self.scale_factor = scale_factor
        self.ref_to_super_fine_scale_factor = ref_to_super_fine_scale_factor
        self.catchment_bounds = catchment_bounds
        self.catchment_section = catchment_section
        self.ref_orog_field = ref_orog_field
        self.data_orog_original_scale_field = data_orog_original_scale_field
        self.ref_flowtocellfield = ref_flowtocellfield
        self.data_flowtocellfield = data_flowtocellfield
        self.rdirs_field = rdirs_field 
        self.super_fine_orog_field = super_fine_orog_field
        self.super_fine_data_flowtocellfield = super_fine_data_flowtocellfield
        self.pair = pair
        plt.figure(figsize=(25,14))
        gs=gridspec.GridSpec(6,9,width_ratios=[8,1,1.5,8,1,1.5,8,1,1],height_ratios=[12,1,1,1,1,1])
        ax = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[0,3])
        ax_catch = plt.subplot(gs[0,6])
        cax_cct = plt.subplot(gs[0,7])
        self.using_numbers_ref.append(False)
        self.using_numbers_data.append(False)
        self.use_super_fine_orog_flags.append(False)
        rc_pts.plot_catchment(ax_catch, self.catchment_section, cax=cax_cct,remove_ticks_flag=False,
                              format_coords=True,
                              lat_offset=self.catchment_bounds[0],
                              lon_offset=self.catchment_bounds[2],
                              colors=None)
        def format_fn_y(tick_val,tick_pos,offset=self.catchment_bounds[0]):
            return pts.calculate_lat_label(tick_val,offset)
        self.y_formatter_funcs.append(format_fn_y)
        def format_fn_x(tick_val,tick_pos,offset=self.catchment_bounds[2]):
            return pts.calculate_lon_label(tick_val,offset)
        self.x_formatter_funcs.append(format_fn_x)
        ax_catch.yaxis.set_major_formatter(FuncFormatter(format_fn_y))
        ax_catch.xaxis.set_major_formatter((FuncFormatter(format_fn_x)))
        plt.setp(ax_catch.xaxis.get_ticklabels(),rotation='vertical')
        ref_orog_field_section = ref_orog_field[self.catchment_bounds[0]:self.catchment_bounds[1],
                                                self.catchment_bounds[2]:self.catchment_bounds[3]]
        data_orog_original_scale_field_section = \
            data_orog_original_scale_field[self.catchment_bounds[0]*self.scale_factor:
                                           self.catchment_bounds[1]*self.scale_factor,
                                           self.catchment_bounds[2]*self.scale_factor:
                                           self.catchment_bounds[3]*self.scale_factor]
        ref_flowtocellfield_section = \
            self.ref_flowtocellfield[self.catchment_bounds[0]:self.catchment_bounds[1],
                                     self.catchment_bounds[2]:self.catchment_bounds[3]]
        data_flowtocellfield_section = \
            data_flowtocellfield[self.catchment_bounds[0]:self.catchment_bounds[1],
                                 self.catchment_bounds[2]:self.catchment_bounds[3]]
        rdirs_field_section = \
            rdirs_field[self.catchment_bounds[0]:self.catchment_bounds[1],
                        self.catchment_bounds[2]:self.catchment_bounds[3]]
        if self.super_fine_orog_field is not None:
            super_fine_orog_field_section = self.super_fine_orog_field[self.catchment_bounds[0]
                                                                       *self.ref_to_super_fine_scale_factor:
                                                                       self.catchment_bounds[1]
                                                                       *self.ref_to_super_fine_scale_factor,
                                                                       self.catchment_bounds[2]
                                                                       *self.ref_to_super_fine_scale_factor:
                                                                       self.catchment_bounds[3]
                                                                       *self.ref_to_super_fine_scale_factor]
            self.super_fine_orog_field_sections.append(super_fine_orog_field_section)
            if self.super_fine_data_flowtocellfield is not None:
                super_fine_data_flowtocellfield_section = self.super_fine_data_flowtocellfield[self.catchment_bounds[0]
                                                                                               *self.\
                                                                                               ref_to_super_fine_scale_factor:
                                                                                               self.catchment_bounds[1]
                                                                                               *self.\
                                                                                               ref_to_super_fine_scale_factor,
                                                                                               self.catchment_bounds[2]
                                                                                               *self.\
                                                                                               ref_to_super_fine_scale_factor:
                                                                                               self.catchment_bounds[3]
                                                                                               *self.\
                                                                                               ref_to_super_fine_scale_factor]
                self.super_fine_data_flowtocellfield_sections.append(super_fine_data_flowtocellfield_section)
        cax1 = plt.subplot(gs[0,1])
        cax2 = plt.subplot(gs[0,4])
        ref_field_min = np.min(ref_orog_field_section)
        ref_field_max = np.max(ref_orog_field_section)
        def_num_levels = 50
        rc_pts.plot_orography_section(ax,cax1,ref_orog_field_section,
                                      ref_field_min,ref_field_max,
                                      pair[0].get_lat(),pair[0].get_lon(),
                                      new_cb=True,num_levels=def_num_levels,
                                      lat_offset=self.catchment_bounds[0],
                                      lon_offset=self.catchment_bounds[2])
        ax.yaxis.set_major_formatter(FuncFormatter(format_fn_y))
        ax.xaxis.set_major_formatter(FuncFormatter(format_fn_x))
        plt.setp(ax.xaxis.get_ticklabels(),rotation='vertical')
        rc_pts.plot_orography_section(ax2,cax2,data_orog_original_scale_field_section,
                                      ref_field_min,ref_field_max,
                                      pair[0].get_lat(),pair[0].get_lon(),
                                      new_cb=True,num_levels=def_num_levels,
                                      lat_offset=self.catchment_bounds[0]*self.scale_factor,
                                      lon_offset=self.catchment_bounds[2]*self.scale_factor)
        def format_fn_y_scaled(tick_val,tick_pos,offset=self.catchment_bounds[0]):
            return pts.calculate_lat_label(tick_val,offset,scale_factor=self.scale_factor)
        self.y_scaled_formatter_funcs.append(format_fn_y_scaled)
        ax2.yaxis.set_major_formatter(FuncFormatter(format_fn_y_scaled))
        def format_fn_x_scaled(tick_val,tick_pos,offset=self.catchment_bounds[2]):
            return pts.calculate_lon_label(tick_val,offset,scale_factor=self.scale_factor)
        self.x_scaled_formatter_funcs.append(format_fn_x_scaled)
        ax2.xaxis.set_major_formatter(FuncFormatter(format_fn_x_scaled))
        def format_fn_y_super_fine_scaled(tick_val,tick_pos,offset=self.catchment_bounds[0]):
            return pts.calculate_lat_label(tick_val,offset,
                                           scale_factor=self.ref_to_super_fine_scale_factor)
        self.y_super_fine_scaled_formatter_funcs.append(format_fn_y_super_fine_scaled)
        def format_fn_x_super_fine_scaled(tick_val,tick_pos,offset=self.catchment_bounds[2]):
            return pts.calculate_lon_label(tick_val,offset,
                                           scale_factor=self.ref_to_super_fine_scale_factor) 
        self.x_super_fine_scaled_formatter_funcs.append(format_fn_x_super_fine_scaled)
        plt.setp(ax2.xaxis.get_ticklabels(),rotation='vertical')
        ax3 = plt.subplot(gs[2,0])
        ax4 = plt.subplot(gs[3,0])
        ax5 = plt.subplot(gs[2,2])
        ax6 = plt.subplot(gs[2:4,3])
        ax7 = plt.subplot(gs[4,0])
        ax8 = plt.subplot(gs[4,3])
        ax9 = plt.subplot(gs[2,6])
        ax10 = plt.subplot(gs[3,6])
        ax11 = plt.subplot(gs[5,0])
        ax12 = plt.subplot(gs[4:6,6])
        ax13 = plt.subplot(gs[5,3])
        min_height = Slider(ax3,'Minimum Height (m)',ref_field_min,ref_field_max,valinit=ref_field_min)
        max_height = Slider(ax4,'Maximum Height (m)',ref_field_min,ref_field_max,valinit=ref_field_max,
                            slidermin=min_height)
        num_levels_slider = Slider(ax7,'Number of\nContour Levels',2,100,valinit=def_num_levels)
        fmap_threshold_slider = Slider(ax9,"Flowmap Threshold",1,min(np.max(ref_flowtocellfield_section),200),valinit=50)
        reset_height_range_button = Button(ax5,"Reset",color='lightgrey',hovercolor='grey')
        linear_colors_buttons = CheckButtons(ax8,["Uniform Linear Colors"],[False])
        overlay_flowmap_buttons = CheckButtons(ax10,['Flowmap Overlay'],[False])
        overlay_flowmap_on_super_fine_orog_buttons = CheckButtons(ax13,['Overlay Flowmaps on\n'
                                                                        'Super Fine Orography'],[False])
        display_height_buttons = CheckButtons(ax11,['Display Height Values'],[False])
        plot_type_radio_buttons = RadioButtons(ax6,('Contour','Filled Contour','Image'))
        plot_type_radio_buttons.set_active(2)
        secondary_orog_radio_buttons = RadioButtons(ax12,('Reference Scale Orography',
                                                          'Super Fine Scale Orography'))
        secondary_orog_radio_buttons.set_active(0)
        if self.super_fine_orog_field is None:
            ax12.set_visible(False)
        else:
            ax12.set_visible(True)
        ax13.set_visible(False)
        self.ref_orog_field_sections.append(ref_orog_field_section)
        self.data_orog_original_scale_field_sections.append(data_orog_original_scale_field_section)
        self.ref_flowtocellfield_sections.append(ref_flowtocellfield_section)
        self.data_flowtocellfield_sections.append(data_flowtocellfield_section)
        self.rdirs_field_sections.append(rdirs_field_section)
        self.catchment_sections.append(self.catchment_section)
        self.orog_gridspecs.append(gs)
        self.orog_rmouth_coords.append(pair[0].get_coords())
        #better to rewrite this as a class with the __call__ property
        update = Update(orog_plot_num=len(self.orog_gridspecs)-1,
                        lat_offset=self.catchment_bounds[0],
                        lon_offset=self.catchment_bounds[2],
                        plots_object=self)
        min_height.on_changed(update)
        max_height.on_changed(update)
        self.update_funcs.append(update)
        self.min_heights.append(min_height)
        self.max_heights.append(max_height)
        num_levels_slider.on_changed(update)
        self.num_levels_sliders.append(num_levels_slider)
        fmap_threshold_slider.on_changed(update)
        self.fmap_threshold_sliders.append(fmap_threshold_slider) 
        overlay_flowmap_buttons.on_clicked(update)
        self.overlay_flowmap_buttons_list.append(overlay_flowmap_buttons)
        overlay_flowmap_on_super_fine_orog_buttons.on_clicked(update)
        self.overlay_flowmap_on_super_fine_orog_buttons_list.append(overlay_flowmap_on_super_fine_orog_buttons)
        linear_colors_buttons.on_clicked(update)
        self.linear_colors_buttons_list.append(linear_colors_buttons)
        display_height_buttons.on_clicked(update)
        self.display_height_buttons_list.append(display_height_buttons)
        if self.super_fine_orog_field is not None:
            secondary_orog_radio_buttons.on_clicked(update)
        self.secondary_orog_radio_buttons_list.append(secondary_orog_radio_buttons)
        reset = Reset(orog_plot_num=len(self.orog_gridspecs)-1,
                      plots_object=self)
        reset_height_range_button.on_clicked(reset)
        self.reset_height_range_buttons.append(reset_height_range_button)
        plot_type_radio_buttons.on_clicked(update)
        self.plot_type_radio_buttons_list.append(plot_type_radio_buttons)
        match_right_scaling = Match_Right_Scaling(orog_plot_num=len(self.orog_gridspecs)-1,
                                                  plots_object=self)
        match_left_scaling = Match_Left_Scaling(orog_plot_num=len(self.orog_gridspecs)-1,
                                                plots_object=self)
        match_catch_scaling = Match_Catch_Scaling(orog_plot_num=len(self.orog_gridspecs)-1,
                                                  plots_object=self)
        ax.callbacks.connect('xlim_changed',match_left_scaling)
        ax.callbacks.connect('ylim_changed',match_left_scaling)
        ax2.callbacks.connect('xlim_changed',match_right_scaling)
        ax2.callbacks.connect('ylim_changed',match_right_scaling)
        ax_catch.callbacks.connect('xlim_changed',match_catch_scaling)
        ax_catch.callbacks.connect('ylim_changed',match_catch_scaling)
        self.match_left_scaling_callback_functions.append(match_left_scaling)
        self.match_right_scaling_callback_functions.append(match_right_scaling)
        self.match_catch_scaling_callback_functions.append(match_catch_scaling)
        
class Update(object):
        
    def __init__(self,orog_plot_num,lat_offset,lon_offset,plots_object):
        self.orog_plot_num = orog_plot_num
        self.lat_offset = lat_offset
        self.lon_offset = lon_offset
        #give this a short name as it will be used very frequently
        self.po = plots_object
            
    def __call__(self,dummy_input):
        if self.po.min_heights[self.orog_plot_num].val >= self.po.max_heights[self.orog_plot_num].val:
            self.po.max_heights[self.orog_plot_num].set_val(self.po.min_heights[self.orog_plot_num].val+0.01)
        ax_catch = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,6])
        ref_xlim = ax_catch.get_xlim()
        ref_ylim = ax_catch.get_ylim()
        plt.cla()
        rc_pts.plot_catchment(ax_catch, self.po.catchment_sections[self.orog_plot_num], 
                              cax=None,legend=False,remove_ticks_flag=False,
                              format_coords=True,
                              lat_offset=self.lat_offset,
                              lon_offset=self.lon_offset)
        ax_catch.yaxis.set_major_formatter(FuncFormatter(self.po.y_formatter_funcs[self.orog_plot_num]))
        ax_catch.xaxis.set_major_formatter(FuncFormatter(self.po.x_formatter_funcs[self.orog_plot_num]))
        plt.setp(ax_catch.xaxis.get_ticklabels(),rotation='vertical')
        ax = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,0])
        if self.po.use_super_fine_orog_flags[self.orog_plot_num]:
            super_fine_xlim = ax.get_xlim()
            super_fine_ylim = ax.get_ylim()
            use_super_fine_orog_old = True
        else:
            use_super_fine_orog_old = False
        fig = plt.gcf()
        plt.cla()
        cax1 = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,1])
        alpha = 0.4 if self.po.overlay_flowmap_buttons_list[self.orog_plot_num].lines[0][0].get_visible() and\
            (self.po.secondary_orog_radio_buttons_list[self.orog_plot_num].value_selected == 
            'Reference Scale Orography') else 1.0
        cmap_name = 'viridis' if self.po.linear_colors_buttons_list[self.orog_plot_num].lines[0][0].get_visible()\
                    else 'terrain'
        if (self.po.secondary_orog_radio_buttons_list[self.orog_plot_num].value_selected == 
            'Reference Scale Orography'):
            self.po.use_super_fine_orog_flags[self.orog_plot_num] = False
            rc_pts.plot_orography_section(ax,cax1,
                                          section=self.po.ref_orog_field_sections[self.orog_plot_num],
                                          min_height=self.po.min_heights[self.orog_plot_num].val, 
                                          max_height=self.po.max_heights[self.orog_plot_num].val, 
                                          outflow_lat=self.po.orog_rmouth_coords[self.orog_plot_num][0], 
                                          outflow_lon=self.po.orog_rmouth_coords[self.orog_plot_num][1],
                                          new_cb=False,num_levels=self.po.num_levels_sliders[self.orog_plot_num].val,
                                          plot_type=self.po.plot_type_radio_buttons_list[self.orog_plot_num].value_selected,
                                          alpha=alpha,cmap_name=cmap_name,
                                          lat_offset=self.lat_offset, lon_offset=self.lon_offset)
            ax.yaxis.set_major_formatter(FuncFormatter(self.po.y_formatter_funcs[self.orog_plot_num]))
            ax.xaxis.set_major_formatter(FuncFormatter(self.po.x_formatter_funcs[self.orog_plot_num]))
        elif (self.po.secondary_orog_radio_buttons_list[self.orog_plot_num].value_selected == 
            'Super Fine Scale Orography'):
            self.po.use_super_fine_orog_flags[self.orog_plot_num] = True
            rc_pts.plot_orography_section(ax,cax1,
                                          section=self.po.super_fine_orog_field_sections[self.orog_plot_num],
                                          min_height=self.po.min_heights[self.orog_plot_num].val, 
                                          max_height=self.po.max_heights[self.orog_plot_num].val, 
                                          outflow_lat=self.po.orog_rmouth_coords[self.orog_plot_num][0], 
                                          outflow_lon=self.po.orog_rmouth_coords[self.orog_plot_num][1],
                                          new_cb=False,num_levels=self.po.num_levels_sliders[self.orog_plot_num].val,
                                          plot_type=self.po.plot_type_radio_buttons_list[self.orog_plot_num].value_selected,
                                          alpha=alpha,cmap_name=cmap_name,
                                          lat_offset=self.lat_offset*self.po.ref_to_super_fine_scale_factor, 
                                          lon_offset=self.lon_offset*self.po.ref_to_super_fine_scale_factor)
            ax.yaxis.set_major_formatter(FuncFormatter(self.po.y_super_fine_scaled_formatter_funcs[self.orog_plot_num]))
            ax.xaxis.set_major_formatter(FuncFormatter(self.po.x_super_fine_scaled_formatter_funcs[self.orog_plot_num]))
        else:
            raise RuntimeError('Unknown radio button value selected!')
        plt.setp(ax.xaxis.get_ticklabels(),rotation='vertical')
        ax2 = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,3])
        data_xlim = ax2.get_xlim()
        data_ylim = ax2.get_ylim()
        plt.cla()
        cax2 = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,4]) 
        rc_pts.plot_orography_section(ax2,cax2,
                                      section=self.po.data_orog_original_scale_field_sections[self.orog_plot_num],
                                      min_height=self.po.min_heights[self.orog_plot_num].val, 
                                      max_height=self.po.max_heights[self.orog_plot_num].val, 
                                      outflow_lat=self.po.orog_rmouth_coords[self.orog_plot_num][0], 
                                      outflow_lon=self.po.orog_rmouth_coords[self.orog_plot_num][1],
                                      new_cb=False,num_levels=self.po.num_levels_sliders[self.orog_plot_num].val,
                                      plot_type=self.po.plot_type_radio_buttons_list[self.orog_plot_num].value_selected,
                                      cmap_name=cmap_name,lat_offset=self.lat_offset*self.po.scale_factor,
                                      lon_offset=self.lon_offset*self.po.scale_factor)
        ax2.yaxis.set_major_formatter(FuncFormatter(self.po.y_scaled_formatter_funcs[self.orog_plot_num]))
        ax2.xaxis.set_major_formatter(FuncFormatter(self.po.x_scaled_formatter_funcs[self.orog_plot_num]))
        plt.setp(ax2.xaxis.get_ticklabels(),rotation='vertical')
        cax_fmap = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][3:5,7])
        ax13 = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][5,3])
        if self.po.overlay_flowmap_buttons_list[self.orog_plot_num].lines[0][0].get_visible():
            cax_fmap.set_visible(True)
            ax13.set_visible(True) if len(self.po.super_fine_data_flowtocellfield_sections) else None
            catchment_rmap_section = rc_pts.select_rivermaps_section(self.po.ref_flowtocellfield_sections[self.orog_plot_num],
                                                                     self.po.data_flowtocellfield_sections[self.orog_plot_num],
                                                                     self.po.rdirs_field_sections[self.orog_plot_num],
                                                                     0,self.po.ref_flowtocellfield_sections[self.orog_plot_num].shape[0],
                                                                     0,self.po.ref_flowtocellfield_sections[self.orog_plot_num].shape[1],
                                                                     threshold=self.po.fmap_threshold_sliders[self.orog_plot_num].val,
                                                                     points_to_mark=None,
                                                                     mark_true_sinks=False,
                                                                     data_true_sinks=None)
            catchment_rmap_section = np.ma.masked_where(np.logical_or(catchment_rmap_section == 0,
                                                                      catchment_rmap_section == 1)
                                                        ,catchment_rmap_section)
            if not self.po.use_super_fine_orog_flags[self.orog_plot_num]:
                rc_pts.plot_flowmap(ax, section=catchment_rmap_section,reduced_map=True,cax=cax_fmap,interpolation='none',
                                    alternative_colors=True,remove_ticks_flag=False)
            elif (len(self.po.super_fine_data_flowtocellfield_sections) != 0 and 
                  self.po.overlay_flowmap_on_super_fine_orog_buttons_list[self.orog_plot_num].lines[0][0].get_visible()):
                super_fine_catchment_rmap_section = rc_pts.select_rivermaps_section(self.po.super_fine_data_flowtocellfield_sections[self.orog_plot_num],
                                                                                    self.po.super_fine_data_flowtocellfield_sections[self.orog_plot_num],
                                                                                    self.po.super_fine_data_flowtocellfield_sections[self.orog_plot_num],
                                                                                    0,self.po.super_fine_data_flowtocellfield_sections[self.orog_plot_num].shape[0],
                                                                                    0,self.po.super_fine_data_flowtocellfield_sections[self.orog_plot_num].shape[1],
                                                                                    threshold=self.po.fmap_threshold_sliders[self.orog_plot_num].val*\
                                                                                    math.pow(self.po.ref_to_super_fine_scale_factor,2),
                                                                                    points_to_mark=None,
                                                                                    mark_true_sinks=False,
                                                                                    data_true_sinks=None)
                super_fine_catchment_rmap_section = np.ma.masked_where(np.logical_or(super_fine_catchment_rmap_section == 0,
                                                                                     super_fine_catchment_rmap_section == 1)
                                                                       ,super_fine_catchment_rmap_section)
                super_fine_catchment_rmap_section[super_fine_catchment_rmap_section == 3] = 4
                rc_pts.plot_flowmap(ax, section=super_fine_catchment_rmap_section,reduced_map=True,cax=cax_fmap,interpolation='none',
                                    alternative_colors=True,remove_ticks_flag=False)
            rc_pts.plot_flowmap(ax_catch, section=catchment_rmap_section,reduced_map=True,cax=cax_fmap,interpolation='none',
                                alternative_colors=True,remove_ticks_flag=False)
        else:
            cax_fmap.set_visible(False)
            ax13.set_visible(False)
            
        fig.canvas.toolbar.push_current()
        if self.po.use_super_fine_orog_flags[self.orog_plot_num]:
            if not use_super_fine_orog_old:
                super_fine_xlim = tuple(self.po.ref_to_super_fine_scale_factor*x for x in ref_xlim)
                super_fine_ylim = tuple(self.po.ref_to_super_fine_scale_factor*y for y in ref_ylim)
                super_fine_xlim = (max(super_fine_xlim[0],-0.5),
                                   min(super_fine_xlim[1],self.po.super_fine_orog_field_sections[self.orog_plot_num].shape[1] - 0.5))
                super_fine_ylim  = (min(super_fine_ylim[0],
                                        self.po.super_fine_orog_field_sections[self.orog_plot_num].shape[0] - 0.5),
                                    max(super_fine_ylim[1],-0.5))
            ax.set_xlim(super_fine_xlim,emit=False)
            ax.set_ylim(super_fine_ylim,emit=False)
        else:
            ax.set_xlim(ref_xlim,emit=False)
            ax.set_ylim(ref_ylim,emit=False)
        if  (pts.calc_displayed_plot_size(ref_xlim,ref_ylim) < self.po.numbers_max_size and
             self.po.display_height_buttons_list[self.orog_plot_num].lines[0][0].get_visible()
             and not self.po.use_super_fine_orog_flags[self.orog_plot_num]):
            pts.print_nums(ax,self.po.ref_orog_field_sections[self.orog_plot_num], 
                           ref_xlim,ref_ylim)
            self.po.using_numbers_ref[self.orog_plot_num] = True
        else:
            self.po.using_numbers_ref[self.orog_plot_num] = False
        ax2.set_xlim(data_xlim,emit=False)
        ax2.set_ylim(data_ylim,emit=False)
        if (pts.calc_displayed_plot_size(data_xlim,data_ylim) < self.po.numbers_max_size and
            self.po.display_height_buttons_list[self.orog_plot_num].lines[0][0].get_visible()):
            pts.print_nums(ax2,self.po.data_orog_original_scale_field_sections[self.orog_plot_num], 
                           data_xlim,data_ylim)
            self.po.using_numbers_data[self.orog_plot_num] = True
        else:
            self.po.using_numbers_data[self.orog_plot_num] = False
        ax_catch.set_xlim(ref_xlim,emit=False)
        ax_catch.set_ylim(ref_ylim,emit=False)
        fig.canvas.toolbar.push_current()
        ax.callbacks.connect('xlim_changed',self.po.match_left_scaling_callback_functions[self.orog_plot_num])
        ax.callbacks.connect('ylim_changed',self.po.match_left_scaling_callback_functions[self.orog_plot_num])
        ax2.callbacks.connect('xlim_changed',self.po.match_right_scaling_callback_functions[self.orog_plot_num])
        ax2.callbacks.connect('ylim_changed',self.po.match_right_scaling_callback_functions[self.orog_plot_num])
        ax_catch.callbacks.connect('xlim_changed',self.po.match_catch_scaling_callback_functions[self.orog_plot_num])
        ax_catch.callbacks.connect('ylim_changed',self.po.match_catch_scaling_callback_functions[self.orog_plot_num])
        fig.canvas.draw()
        
class Reset(object):
    
    def __init__(self,orog_plot_num,plots_object):
        self.orog_plot_num = orog_plot_num
        self.po = plots_object
                         
    def __call__(self,event):
        self.po.min_heights[self.orog_plot_num].reset()
        self.po.max_heights[self.orog_plot_num].reset()
        self.po.num_levels_sliders[self.orog_plot_num].reset()
        self.po.fmap_threshold_sliders[self.orog_plot_num].reset()
        ax_catch = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,6])
        ax = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,0])
        ax2 = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,3])
        ref_section = self.po.ref_orog_field_sections[self.orog_plot_num]
        data_section = self.po.data_orog_original_scale_field_sections[self.orog_plot_num]
        if (self.po.using_numbers_data[self.orog_plot_num] or 
            self.po.using_numbers_ref[self.orog_plot_num]):
            self.po.update_funcs[self.orog_plot_num](None)
        ax_catch.set_xlim(-0.5,ref_section.shape[1]-0.5,emit=False)
        ax_catch.set_ylim(ref_section.shape[0]-0.5,-0.5,emit=False)
        ax2.set_xlim(-0.5,data_section.shape[1]-0.5,emit=False)
        ax2.set_ylim(data_section.shape[0]-0.5,-0.5,emit=False)
        if self.po.use_super_fine_orog_flags[self.orog_plot_num]:
            super_fine_data_section = self.po.super_fine_orog_field_sections[self.orog_plot_num]
            ax.set_xlim(-0.5,super_fine_data_section.shape[1]-0.5,emit=False)
            ax.set_ylim(super_fine_data_section.shape[0]-0.5,-0.5,emit=False)
        else:
            ax.set_xlim(-0.5,ref_section.shape[1]-0.5,emit=False)
            ax.set_ylim(ref_section.shape[0]-0.5,-0.5,emit=False)
        fig = plt.gcf()
        fig.canvas.draw()
        
class Match_Right_Scaling(object):
        
    def __init__(self,orog_plot_num,plots_object):
        self.orog_plot_num = orog_plot_num
        self.po = plots_object
                 
    def __call__(self,event):
        fig = plt.gcf()
        ax_catch = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,6])
        ax = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,0])
        ax2 = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,3])
        xlim = ax2.get_xlim()
        ylim = ax2.get_ylim()
        xlim_scaled = tuple(round(i/self.po.scale_factor) for i in xlim)
        ylim_scaled = tuple(round(i/self.po.scale_factor) for i in ylim)
        xlim_scaled = (max(xlim_scaled[0],-0.5),
                       min(xlim_scaled[1],
                           self.po.ref_orog_field_sections[self.orog_plot_num].shape[1] - 0.5))
        ylim_scaled = (min(ylim_scaled[0],
                           self.po.ref_orog_field_sections[self.orog_plot_num].shape[0] - 0.5),
                       max(ylim_scaled[1],-0.5))
        fig.canvas.toolbar.push_current()
        if (self.po.secondary_orog_radio_buttons_list[self.orog_plot_num].value_selected == 
            'Reference Scale Orography'):
            ax.set_xlim(xlim_scaled,emit=False)
            ax.set_ylim(ylim_scaled,emit=False)
        elif (self.po.secondary_orog_radio_buttons_list[self.orog_plot_num].value_selected == 
            'Super Fine Scale Orography'):
            xlim_scaled_to_super_fine = tuple(i*self.po.ref_to_super_fine_scale_factor/self.po.scale_factor 
                                              for i in xlim)
            ylim_scaled_to_super_fine = tuple(i*self.po.ref_to_super_fine_scale_factor/self.po.scale_factor 
                                              for i in ylim) 
            ax.set_xlim(xlim_scaled_to_super_fine,emit=False)
            ax.set_ylim(ylim_scaled_to_super_fine,emit=False)
        else:
            raise RuntimeError('Unknown radio button value selected!')
        ax_catch.set_xlim(xlim_scaled,emit=False)
        ax_catch.set_ylim(ylim_scaled,emit=False)
        if (self.po.using_numbers_data[self.orog_plot_num] or 
            self.po.using_numbers_ref[self.orog_plot_num]  or
            (pts.calc_displayed_plot_size(xlim_scaled,ylim_scaled) < self.po.numbers_max_size and 
            self.po.display_height_buttons_list[self.orog_plot_num].lines[0][0].get_visible())):
            self.po.update_funcs[self.orog_plot_num](None)
        fig.canvas.toolbar.push_current()
        fig.canvas.draw()

class Match_Left_Scaling(object):
    
    def __init__(self,orog_plot_num,plots_object):
        self.orog_plot_num = orog_plot_num
        self.po = plots_object

    def __call__(self,event):
        fig = plt.gcf()
        ax_catch = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,6])
        ax = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,0])
        ax2 = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,3])
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if (self.po.secondary_orog_radio_buttons_list[self.orog_plot_num].value_selected == 
            'Super Fine Scale Orography'):
            xlim = tuple(round(i/self.po.ref_to_super_fine_scale_factor) for i in xlim)
            ylim = tuple(round(i/self.po.ref_to_super_fine_scale_factor) for i in ylim)
            xlim = (max(xlim[0],-0.5),
                       min(xlim[1],
                           self.po.ref_orog_field_sections[self.orog_plot_num].shape[1] - 0.5))
            ylim = (min(ylim[0],
                           self.po.ref_orog_field_sections[self.orog_plot_num].shape[0] - 0.5),
                       max(ylim[1],-0.5))
        xlim_scaled = tuple(i*self.po.scale_factor for i in xlim)
        ylim_scaled = tuple(i*self.po.scale_factor for i in ylim)
        fig.canvas.toolbar.push_current()
        ax_catch.set_xlim(xlim,emit=False)
        ax_catch.set_ylim(ylim,emit=False)
        ax2.set_xlim(xlim_scaled,emit=False)
        ax2.set_ylim(ylim_scaled,emit=False)
        if (self.po.using_numbers_data[self.orog_plot_num] or 
            self.po.using_numbers_ref[self.orog_plot_num]  or
            (pts.calc_displayed_plot_size(xlim,ylim) < self.po.numbers_max_size and
             self.po.display_height_buttons_list[self.orog_plot_num].lines[0][0].get_visible())):
            self.po.update_funcs[self.orog_plot_num](None)
        fig.canvas.toolbar.push_current()
        fig.canvas.draw()

class Match_Catch_Scaling(object):
            
    def __init__(self,orog_plot_num,plots_object):
        self.orog_plot_num = orog_plot_num
        self.po = plots_object

    def __call__(self,event):
        fig = plt.gcf()
        ax_catch = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,6])
        ax = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,0])
        ax2 = plt.subplot(self.po.orog_gridspecs[self.orog_plot_num][0,3])
        xlim = ax_catch.get_xlim()
        ylim = ax_catch.get_ylim()
        xlim_scaled = tuple(i*self.po.scale_factor for i in xlim)
        ylim_scaled = tuple(i*self.po.scale_factor for i in ylim)
        fig.canvas.toolbar.push_current()
        if (self.po.secondary_orog_radio_buttons_list[self.orog_plot_num].value_selected == 
            'Reference Scale Orography'):
            ax.set_xlim(xlim,emit=False)
            ax.set_ylim(ylim,emit=False)
        elif (self.po.secondary_orog_radio_buttons_list[self.orog_plot_num].value_selected == 
            'Super Fine Scale Orography'):
            xlim_scaled_to_super_fine = tuple(i*self.po.ref_to_super_fine_scale_factor for i in xlim)
            ylim_scaled_to_super_fine = tuple(i*self.po.ref_to_super_fine_scale_factor for i in ylim) 
            ax.set_xlim(xlim_scaled_to_super_fine,emit=False)
            ax.set_ylim(ylim_scaled_to_super_fine,emit=False)
        else:
            raise RuntimeError('Unknown radio button value selected!')
        ax2.set_xlim(xlim_scaled,emit=False)
        ax2.set_ylim(ylim_scaled,emit=False)
        if (self.po.using_numbers_data[self.orog_plot_num] or 
            self.po.using_numbers_ref[self.orog_plot_num]  or
            (pts.calc_displayed_plot_size(xlim,ylim) < self.po.numbers_max_size and 
             self.po.display_height_buttons_list[self.orog_plot_num].lines[0][0].get_visible())):
            self.po.update_funcs[self.orog_plot_num](None)
        fig.canvas.toolbar.push_current()
        fig.canvas.draw()