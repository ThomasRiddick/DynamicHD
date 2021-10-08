'''
A module containing routines to assisting in making plots for
analysing the product of dynamic lake model dry runs

Created on Jun 4, 2021

@author: thomasriddick
'''
import math
import glob
import re
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, TextBox, RectangleSelector, RangeSlider
from matplotlib.widgets import Slider
from matplotlib.colors import LogNorm
from HD_Plots.utilities.match_river_mouths import generate_matches
from HD_Plots.utilities import match_river_mouths
from HD_Plots.utilities import plotting_tools as pts
from Dynamic_HD_Scripts.utilities import utilities
from Dynamic_HD_Scripts.base.field import Field
from enum import Enum

class PlotScales(Enum):
    NORMAL = 1
    FINE = 2
    SUPERFINE = 3

class TimeslicePlot():

    def __init__(self,ax):
        self.ax = ax
        self.plot = None
        self.selector = None
        self.scale = PlotScales.NORMAL

class ZoomSettings():

    fine_scale_factor = 3
    super_fine_scale_factor = 60
    def __init__(self,zoomed,zoomed_section_bounds):
        self.zoomed = zoomed
        self.zoomed_section_bounds = zoomed_section_bounds
        if not self.zoomed:
            self.zoomed_section_bounds = {"min_lat":0,
                                          "min_lon":0}

    def copy(self):
        return ZoomSettings(self.zoomed,dict(self.zoomed_section_bounds))

class InteractiveTimeslicePlots:

    zoom_button_colors = itertools.cycle(['grey','lightgray'])
    height_range_text_match = re.compile("^\s*([0-9]*)\s*,\s*([0-9]*)\s*$")

    def __init__(self,colors,
                 configuration,
                 lsmask_sequence,
                 glacier_mask_sequence,
                 catchment_nums_one_sequence,
                 catchment_nums_two_sequence,
                 river_flow_one_sequence,
                 river_flow_two_sequence,
                 river_mouths_one_sequence,
                 river_mouths_two_sequence,
                 lake_volume_one_sequence,
                 lake_volume_two_sequence,
                 lake_basin_numbers_one_sequence,
                 lake_basin_numbers_two_sequence,
                 fine_river_flow_one_sequence,
                 fine_river_flow_two_sequence,
                 orography_one_sequence,
                 orography_two_sequence,
                 super_fine_orography,
                 first_corrected_orography,
                 second_corrected_orography,
                 third_corrected_orography,
                 fourth_corrected_orography,
                 true_sinks,
                 date_sequence,
                 minflowcutoff,
                 use_glacier_mask,
                 zoomed,
                 zoomed_section_bounds,
                 zero_slice_only_one=False,
                 zero_slice_only_two=False):
        self.plot_types = {"comp":self.catchment_and_cflow_comp_plot,
                           "none":self.no_plot,
                           "cflow1":self.cflow_plot_one,
                           "cflow2":self.cflow_plot_two,
                           "catch1":self.catchments_plot_one,
                           "catch2":self.catchments_plot_two,
                           "fcflow1":self.fine_cflow_plot_one,
                           "fcflow2":self.fine_cflow_plot_two,
                           "fcflowcomp":self.fine_cflow_comp_plot,
                           "orog1":self.orography_plot_one,
                           "orog2":self.orography_plot_one,
                           "orogcomp":self.orography_comp_plot,
                           "sforog":self.super_fine_orography_plot,
                           "firstcorrorog":self.first_corrected_orography_plot,
                           "secondcorrorog":self.second_corrected_orography_plot,
                           "thirdcorrorog":self.third_corrected_orography_plot,
                           "fourthcorrorog":self.fourth_corrected_orography_plot,
                           "corrorog12comp":self.first_vs_second_corrected_orography_plot,
                           "corrorog23comp":self.second_vs_third_corrected_orography_plot,
                           "corrorog34comp":self.third_vs_fourth_corrected_orography_plot,
                           "truesinks":self.true_sinks_plot,
                           "lakev1":self.lake_volume_plot_one,
                           "lakev2":self.lake_volume_plot_two,
                           "loglakev1":self.log_lake_volume_plot_one,
                           "loglakev2":self.log_lake_volume_plot_two,
                           "cflowandlake1":self.cflow_and_lake_plot_one,
                           "cflowandlake2":self.cflow_and_lake_plot_two,
                           "lakevcomp":self.lake_volume_comp_plot,
                           "lakebasinnums1":self.lake_basin_numbers_plot_one,
                           "lakebasinnums2":self.lake_basin_numbers_plot_two}
        self.orog_plot_types  = ["orog1","orog2","orogcomp","sforog",
                                 "firstcorrorog","secondcorrorog",
                                 "thirdcorrorog","fourthcorrorog",
                                 "corrorog12comp","corrorog23comp",
                                 "corrorog34comp"]
        self.cflow_plot_types = ["fcflow1","fcflow1","fcflow1comp",
                                 "cflow1","cflow2","comp"]
        self.configuration = configuration
        self.next_command_to_send = None
        self.timeslice_plots = []
        self.minflowcutoff = minflowcutoff
        self.use_glacier_mask = use_glacier_mask
        self.cmap = mpl.colors.ListedColormap(['blue','peru','black','green','red','white','yellow'])
        self.bounds = list(range(8))
        self.norm = mpl.colors.BoundaryNorm(self.bounds,self.cmap.N)
        N_catch = 50
        self.cmap_catch = 'jet'
        self.orog_min = -9999.0
        self.orog_max = 9999.0
        self.bounds_catch = np.linspace(0,N_catch,N_catch+1)
        self.norm_catch = mpl.colors.BoundaryNorm(boundaries=self.bounds_catch,
                                                  ncolors=256)
        self.zoom_settings = ZoomSettings(zoomed=zoomed,
                                          zoomed_section_bounds=zoomed_section_bounds)
        self.original_zoom_settings = self.zoom_settings.copy()
        self.fine_cutoff_scaling = self.zoom_settings.fine_scale_factor*self.zoom_settings.fine_scale_factor
        self.date_sequence = date_sequence
        self.widgets = []
        self.replot_required = False
        self.fig = plt.figure(figsize=(16,10))
        if len(self.configuration) == 2 or len(self.configuration) == 4:
            if len(self.configuration) == 2:
                gs=gridspec.GridSpec(nrows=3,ncols=6,width_ratios=[1,1,1,1,1,1],
                                     height_ratios=[12,1,1])
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[0,0:3])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[0,3:6])))
                wrows_offset = 1
            elif len(self.configuration) == 4:
                gs=gridspec.GridSpec(nrows=4,ncols=6,width_ratios=[1,1,1,1,1,1],
                                     height_ratios=[6,6,1,1])
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[0,0:3])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[0,3:6])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[1,0:3])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[1,3:6])))
                wrows_offset = 2
        else:
            if len(self.configuration) == 1:
                gs=gridspec.GridSpec(nrows=3,ncols=6,width_ratios=[1,1,1,1,1,1],height_ratios=[12,1,1])
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[0,0:3])))
                wrows_offset = 1
            elif len(self.configuration) == 3:
                gs=gridspec.GridSpec(nrows=3,ncols=6,width_ratios=[1,1,1,1,1,1],height_ratios=[12,1,1])
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[0,0:2])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[0,2:4])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[0,4:6])))
                wrows_offset = 1
            elif len(self.configuration) == 6:
                gs=gridspec.GridSpec(nrows=4,ncols=6,width_ratios=[1,1,1,1,1,1],height_ratios=[6,6,1,1])
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[0,0:2])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[0,2:4])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[0,4:6])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[1,0:2])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[1,2:4])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[1,4:6])))
                wrows_offset = 2
            elif len(self.configuration) == 9:
                gs=gridspec.GridSpec(nrows=5,ncols=6,width_ratios=[1,1,1,1,1,1],
                                     height_ratios=[4,4,4,1,1])
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[0,0:2])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[0,2:4])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[0,4:6])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[1,0:2])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[1,2:4])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[1,4:6])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[2,0:2])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[2,2:4])))
                self.timeslice_plots.append(TimeslicePlot(self.fig.add_subplot(gs[2,4:6])))
                wrows_offset = 3
        self.wax1 = self.fig.add_subplot(gs[wrows_offset,0])
        self.wax2 = self.fig.add_subplot(gs[wrows_offset,1])
        self.wax3 = self.fig.add_subplot(gs[wrows_offset,2])
        self.wax4 = self.fig.add_subplot(gs[wrows_offset,4:6])
        self.wax5 = self.fig.add_subplot(gs[wrows_offset+1,0])
        self.wax6 = self.fig.add_subplot(gs[wrows_offset+1,1])
        self.wax7 = self.fig.add_subplot(gs[wrows_offset+1,2])
        self.wax8 = self.fig.add_subplot(gs[wrows_offset+1,4:6])
        previous_button = Button(self.wax1,"Previous",color='lightgrey',hovercolor='grey')
        next_button = Button(self.wax2,"Next",color='lightgrey',hovercolor='grey')
        text_box = TextBox(self.wax3,"YBP:")
        if not set(self.configuration).isdisjoint(self.cflow_plot_types):
            #add zero to list to prevent errors if the two sequences are Nones
            maxflow =   max([np.max(slice) if slice is not None else 0
                            for slice in itertools.chain(river_flow_one_sequence if
                                                         river_flow_one_sequence is
                                                         not None else [],
                                                         river_flow_two_sequence if
                                                         river_flow_two_sequence is
                                                         not None else [])]+[0])
            #add zero to list to prevent errors if the two sequences are Nones
            maxflow_fine =   max([np.max(slice) if slice is not None else 0
                            for slice in itertools.chain(fine_river_flow_one_sequence if
                                                         fine_river_flow_one_sequence is
                                                         not None else [],
                                                         fine_river_flow_two_sequence if
                                                         fine_river_flow_two_sequence is
                                                         not None else [])]+[0])
            maxflow = max(maxflow,math.ceil(maxflow_fine/self.fine_cutoff_scaling))
            #Prevent slider from becoming dominated by very high values in anatartica
            #by capping it at 1500
            maxflow = min(1000,maxflow)
            minflowcutoff_slider = Slider(self.wax4,"Minimum Flow Threshold",valmin=0,
                                          valinit=self.minflowcutoff,valmax=maxflow)
        else:
            self.wax4.set_visible(False)
        zoom_button = Button(self.wax5,"Zoom",color='lightgrey',hovercolor='grey')
        zoom_reset_button = Button(self.wax6,"Reset Zoom",color='lightgrey',hovercolor='grey')
        if not set(self.configuration).isdisjoint(self.orog_plot_types):
            height_text_box = TextBox(self.wax7,"HR:")
            height_slider = RangeSlider(self.wax8,"Height (m)",self.orog_min,self.orog_max)
        else:
            self.wax7.set_visible(False)
            self.wax8.set_visible(False)
        lake_and_river_plots_required=(not set(self.configuration).isdisjoint(["cflowandlake1",
                                                                               "cflowandlake1"]))
        combined_sequence_tuples =  prep_combined_sequence_tuples(lsmask_sequence,
                                                                  glacier_mask_sequence,
                                                                  catchment_nums_one_sequence,
                                                                  catchment_nums_two_sequence,
                                                                  river_flow_one_sequence,
                                                                  river_flow_two_sequence,
                                                                  river_mouths_one_sequence,
                                                                  river_mouths_two_sequence,
                                                                  lake_volume_one_sequence,
                                                                  lake_volume_two_sequence,
                                                                  lake_basin_numbers_one_sequence,
                                                                  lake_basin_numbers_two_sequence,
                                                                  fine_river_flow_one_sequence,
                                                                  fine_river_flow_two_sequence,
                                                                  orography_one_sequence,
                                                                  orography_two_sequence,
                                                                  ["{} YBP".format(date) for date
                                                                   in self.date_sequence],
                                                                   lake_and_river_plots_required,
                                                                   self.minflowcutoff)

        self.gen = generate_catchment_and_cflow_sequence_tuple(combined_sequence_tuples,
                                                               super_fine_orography,
                                                               first_corrected_orography,
                                                               second_corrected_orography,
                                                               third_corrected_orography,
                                                               fourth_corrected_orography,
                                                               true_sinks,
                                                               bidirectional=True,
                                                               zoom_settings=self.zoom_settings,
                                                               return_zero_slice_only_one=
                                                               zero_slice_only_one,
                                                               return_zero_slice_only_two=
                                                               zero_slice_only_two)
        self.step()
        next_button.on_clicked(self.step_forward)
        self.widgets.append(next_button)
        previous_button.on_clicked(self.step_back)
        self.widgets.append(previous_button)
        text_box.on_submit(self.step_to_date)
        self.widgets.append(text_box)
        if not set(self.configuration).isdisjoint(self.cflow_plot_types):
            minflowcutoff_slider.on_changed(self.update_minflowcutoff)
            self.widgets.append(minflowcutoff_slider)
        zoom_button.on_clicked(self.toggle_zoom)
        self.widgets.append(zoom_button)
        zoom_reset_button.on_clicked(self.reset_zoom)
        self.widgets.append(zoom_reset_button)
        if not set(self.configuration).isdisjoint(self.orog_plot_types):
            height_text_box.on_submit(self.change_height_range_text_box)
            self.widgets.append(height_text_box)
            height_slider.on_changed(self.change_height_range_slider)
            self.widgets.append(height_slider)
        self.fig.canvas.mpl_connect('key_press_event',self.toggle_zoom_keyboard)
        zoom_in_funcs = {PlotScales.NORMAL:self.zoom_in_normal,
                         PlotScales.FINE:self.zoom_in_fine,
                         PlotScales.SUPERFINE:self.zoom_in_super_fine}
        for plot in self.timeslice_plots:
            plot.selector = RectangleSelector(plot.ax,zoom_in_funcs[plot.scale],
                                              drawtype='box',useblit=True,
                                              button=[1,3],minspanx=3,
                                              minspany=3,spancoords='pixels')
            plot.selector.set_active(False)

    def step(self):
        (self.lsmask_slice_zoomed,self.glacier_mask_slice_zoomed,
         self.matched_catchment_nums_one,self.matched_catchment_nums_two,
         self.river_flow_one_slice_zoomed,self.river_flow_two_slice_zoomed,
         self.lake_volume_one_slice_zoomed,self.lake_volume_two_slice_zoomed,
         self.lake_basin_numbers_one_slice_zoomed,
         self.lake_basin_numbers_two_slice_zoomed,
         self.fine_river_flow_one_slice_zoomed,self.fine_river_flow_two_slice_zoomed,
         self.orography_one_slice_zoomed,self.orography_two_slice_zoomed,
         self.super_fine_orography_slice_zoomed,
         self.first_corrected_orography_slice_zoomed,
         self.second_corrected_orography_slice_zoomed,
         self.third_corrected_orography_slice_zoomed,
         self.fourth_corrected_orography_slice_zoomed,
         self.true_sinks_slice_zoomed,
         self.lake_and_river_colour_codes_one_slice_zoomed,
         self.lake_and_river_colour_codes_two_slice_zoomed,
         self.date_text) = self.gen.send(self.next_command_to_send)
        for index,plot in enumerate(self.configuration):
            self.plot_types[plot](index)
        self.fig.suptitle("Timeslice: " + self.date_text,fontsize=16)
        self.fig.canvas.draw()
        plt.pause(0.001)

    def step_back(self,event):
        self.next_command_to_send = True
        self.step()

    def step_forward(self,event):
        self.next_command_to_send = False
        self.step()

    def step_to_date(self,event):
        self.next_command_to_send = self.date_sequence.index(int(event))
        self.step()

    def toggle_zoom(self,event):
        self.widgets[3].color = next(self.zoom_button_colors)
        for plot in self.timeslice_plots:
            if plot.selector.active:
                plot.selector.set_active(False)
            else:
                plot.selector.set_active(True)

    def toggle_zoom_keyboard(self,event):
        if event.key == 'z':
            self.toggle_zoom(event)

    def plot_from_colour_codes(self,colour_codes,index):
        if not self.timeslice_plots[index].plot or self.replot_required:
            if self.replot_required:
                self.timeslice_plots[index].ax.clear()
            self.timeslice_plots[index].plot = \
                self.timeslice_plots[index].ax.imshow(colour_codes,cmap=self.cmap,
                                                         norm=self.norm,interpolation="none")
            self.timeslice_plots[index].ax.tick_params(axis="x",which='both',bottom=False,
                                                       top=False,labelbottom=False)
            self.timeslice_plots[index].ax.tick_params(axis="y",which='both',left=False,
                                                       right=False,labelleft=False)
            self.set_format_coord(self.timeslice_plots[index].ax,
                                  self.timeslice_plots[index].scale)
        else:
            self.timeslice_plots[index].plot.set_data(colour_codes)

    def catchment_and_cflow_comp_plot(self,index):
        colour_codes = generate_colour_codes_comp(self.lsmask_slice_zoomed,
                                                  self.glacier_mask_slice_zoomed,
                                                  self.matched_catchment_nums_one,
                                                  self.matched_catchment_nums_two,
                                                  self.river_flow_one_slice_zoomed,
                                                  self.river_flow_two_slice_zoomed,
                                                  minflowcutoff=
                                                  self.minflowcutoff,
                                                  use_glacier_mask=
                                                  self.use_glacier_mask)
        self.plot_from_colour_codes(colour_codes,index)

    def cflow_and_lake_plot_one(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.plot_from_colour_codes(self.lake_and_river_colour_codes_one_slice_zoomed,
                                    index)

    def cflow_and_lake_plot_two(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.plot_from_colour_codes(self.lake_and_river_colour_codes_two_slice_zoomed,
                                    index)

    def no_plot(self,index):
        self.timeslice_plots[index].ax.set_visible(False)


    def catchments_plot_one(self,index):
        self.catchments_plot_base(index,self.matched_catchment_nums_one)

    def catchments_plot_two(self,index):
        self.catchments_plot_base(index,self.matched_catchment_nums_two)

    def catchments_plot_base(self,index,catchment_slice):
        if not self.timeslice_plots[index].plot or self.replot_required:
            if self.replot_required:
                self.timeslice_plots[index].ax.clear()
            self.timeslice_plots[index].plot = \
                self.timeslice_plots[index].ax.imshow(catchment_slice,cmap=self.cmap_catch,
                                                         norm=self.norm_catch,interpolation="none")

            self.timeslice_plots[index].ax.tick_params(axis="x",which='both',bottom=False,
                                                       top=False,labelbottom=False)
            self.timeslice_plots[index].ax.tick_params(axis="y",which='both',left=False,
                                                       right=False,labelleft=False)
            self.set_format_coord(self.timeslice_plots[index].ax,
                                  self.timeslice_plots[index].scale)
        else:
            self.timeslice_plots[index].plot.set_data(catchment_slice)

    def cflow_plot_one(self,index):
        self.cflow_plot_base(index,True)

    def cflow_plot_two(self,index):
        self.cflow_plot_base(index,False)

    def cflow_plot_base(self,index,use_first_sequence_set):
        colour_codes_cflow = np.zeros(self.lsmask_slice_zoomed.shape)
        colour_codes_cflow[self.lsmask_slice_zoomed == 0] = 1
        river_flow_slice = (self.river_flow_one_slice_zoomed if use_first_sequence_set
                            else self.river_flow_two_slice_zoomed)
        colour_codes_cflow[river_flow_slice >= self.minflowcutoff] = 2
        if self.use_glacier_mask:
            colour_codes_cflow[self.glacier_mask_slice_zoomed == 1] = 3
        colour_codes_cflow[self.lsmask_slice_zoomed == 1] = 0
        self.plot_from_colour_codes(colour_codes_cflow,index)

    def fine_cflow_plot_one(self,index):
        self.fine_cflow_plot_base(index,True)

    def fine_cflow_plot_two(self,index):
        self.fine_cflow_plot_base(index,False)

    def fine_cflow_plot_base(self,index,use_first_sequence_set):
        self.timeslice_plots[index].scale = PlotScales.FINE
        colour_codes_cflow = np.zeros(self.fine_river_flow_one_slice_zoomed.shape if use_first_sequence_set
                                      else self.fine_river_flow_two_slice_zoomed.shape)
        colour_codes_cflow[:,:] = 1
        river_flow_slice = (self.fine_river_flow_one_slice_zoomed if use_first_sequence_set
                            else self.fine_river_flow_two_slice_zoomed)
        colour_codes_cflow[river_flow_slice >= self.minflowcutoff*self.fine_cutoff_scaling] = 2
        self.plot_from_colour_codes(colour_codes_cflow,index)

    def fine_cflow_comp_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        colour_codes_cflow = np.zeros(self.fine_river_flow_one_slice_zoomed.shape)
        colour_codes_cflow[:,:] = 1
        colour_codes_cflow[np.logical_and(self.fine_river_flow_one_slice_zoomed >= self.minflowcutoff*self.fine_cutoff_scaling,
                                          self.fine_river_flow_two_slice_zoomed >= self.minflowcutoff*self.fine_cutoff_scaling)] = 2
        colour_codes_cflow[np.logical_and(self.fine_river_flow_one_slice_zoomed >= self.minflowcutoff*self.fine_cutoff_scaling,
                                          self.fine_river_flow_two_slice_zoomed < self.minflowcutoff*self.fine_cutoff_scaling)] = 0
        colour_codes_cfow[np.logical_and(self.fine_river_flow_one_slice_zoomed < self.minflowcutoff*self.fine_cutoff_scaling,
                                          self.fine_river_flow_two_slice_zoomed >= self.minflowcutoff*self.fine_cutoff_scaling)] = 0
        self.plot_from_colour_codes(colour_codes_cflow,index)

    def orography_plot_one(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.orography_one_slice_zoomed)

    def orography_plot_two(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.orography_two_slice_zoomed)

    def orography_comp_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.orography_one_slice_zoomed -
                                          self.orography_two_slice_zoomed)

    def super_fine_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.SUPERFINE
        self.orography_plot_base(index,self.super_fine_orography_slice_zoomed)

    def first_corrected_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.first_corrected_orography_slice_zoomed)

    def second_corrected_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.second_corrected_orography_slice_zoomed)

    def third_corrected_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.third_corrected_orography_slice_zoomed)

    def fourth_corrected_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.fourth_corrected_orography_slice_zoomed)

    def first_vs_second_corrected_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.second_corrected_orography_slice_zoomed -
                                       self.first_corrected_orography_slice_zoomed)

    def second_vs_third_corrected_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.third_corrected_orography_slice_zoomed -
                                       self.second_corrected_orography_slice_zoomed)

    def third_vs_fourth_corrected_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.fourth_corrected_orography_slice_zoomed -
                                       self.third_corrected_orography_slice_zoomed)

    def true_sinks_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.plot_from_colour_codes(self.true_sinks_slice_zoomed,index)

    def orography_plot_base(self,index,orography):
        if not self.timeslice_plots[index].plot or self.replot_required:
            if self.replot_required:
                self.timeslice_plots[index].ax.clear()
            self.timeslice_plots[index].plot = \
                self.timeslice_plots[index].ax.imshow(orography,vmin=self.orog_min,
                                                      vmax=self.orog_max)
            self.timeslice_plots[index].ax.tick_params(axis="x",which='both',bottom=False,
                                                       top=False,labelbottom=False)
            self.timeslice_plots[index].ax.tick_params(axis="y",which='both',left=False,
                                                       right=False,labelleft=False)
            self.set_format_coord(self.timeslice_plots[index].ax,
                                  self.timeslice_plots[index].scale)
        elif self.timeslice_plots[index].scale == PlotScales.SUPERFINE:
            pass
        else:
            self.timeslice_plots[index].plot.set_data(orography)

    def lake_volume_plot_one(self,index):
        self.lake_volume_plot_base(index,self.lake_volume_one_slice_zoomed)

    def lake_volume_plot_two(self,index):
        self.lake_volume_plot_base(index,self.lake_volume_two_slice_zoomed)

    def log_lake_volume_plot_one(self,index):
        self.lake_volume_plot_base(index,self.lake_volume_one_slice_zoomed,True)

    def log_lake_volume_plot_two(self,index):
        self.lake_volume_plot_base(index,self.lake_volume_two_slice_zoomed,True)

    def lake_volume_comp_plot(self,index):
        self.lake_volume_plot_base(index,self.lake_volume_one_slice_zoomed -
                                            self.lake_volume_two_slice_zoomed)

    def lake_volume_plot_base(self,index,lake_volumes,use_log_scale=False):
        self.timeslice_plots[index].scale = PlotScales.FINE
        if (not self.timeslice_plots[index].plot or
            self.replot_required or np.all(lake_volumes<=0)):
            self.timeslice_plots[index].ax.clear()
            if use_log_scale and not np.all(lake_volumes<=0):
                self.timeslice_plots[index].plot = \
                    self.timeslice_plots[index].ax.imshow(lake_volumes,
                                                          norm=LogNorm(clip=True))
            else:
                self.timeslice_plots[index].plot = \
                    self.timeslice_plots[index].ax.imshow(lake_volumes)
            self.timeslice_plots[index].ax.tick_params(axis="x",which='both',bottom=False,
                                                       top=False,labelbottom=False)
            self.timeslice_plots[index].ax.tick_params(axis="y",which='both',left=False,
                                                       right=False,labelleft=False)
            self.set_format_coord(self.timeslice_plots[index].ax,
                                  self.timeslice_plots[index].scale)
        else:
            self.timeslice_plots[index].plot.set_data(lake_volumes)

    def lake_basin_numbers_plot_one(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.catchments_plot_base(index,self.lake_basin_numbers_one_slice_zoomed)

    def lake_basin_numbers_plot_two(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.catchments_plot_base(index,self.lake_basin_numbers_two_slice_zoomed)

    def update_minflowcutoff(self,val):
        self.minflowcutoff = val
        self.next_command_to_send = "zoom"
        self.step()

    def reset_zoom(self,event):
        self.zoom_settings.zoomed = self.original_zoom_settings.zoomed
        self.zoom_settings.zoomed_section_bounds = dict(self.original_zoom_settings.zoomed_section_bounds)
        self.next_command_to_send = "zoom"
        self.replot_required = True
        self.step()
        self.replot_required = False

    def zoom_in_normal(self,eclick,erelease):
        self.zoom_in_base(eclick,erelease,scale_factor=1)

    def zoom_in_fine(self,eclick,erelease):
        self.zoom_in_base(eclick,erelease,scale_factor=
                          self.zoom_settings.fine_scale_factor)

    def zoom_in_super_fine(self,eclick,erelease):
        self.zoom_in_base(eclick,erelease,scale_factor=
                          self.zoom_settings.super_fine_scale_factor)

    def zoom_in_base(self,eclick,erelease,scale_factor):
        if self.zoom_settings.zoomed:
            old_min_lat = self.zoom_settings.zoomed_section_bounds["min_lat"]
            old_min_lon = self.zoom_settings.zoomed_section_bounds["min_lon"]
        else:
            old_min_lat = 0
            old_min_lon = 0
        min_lat = round(min(eclick.ydata,erelease.ydata)/scale_factor) + old_min_lat
        max_lat = round(max(eclick.ydata,erelease.ydata)/scale_factor) + old_min_lat
        min_lon = round(min(eclick.xdata,erelease.xdata)/scale_factor) + old_min_lon
        max_lon = round(max(eclick.xdata,erelease.xdata)/scale_factor) + old_min_lon
        if min_lat == max_lat or min_lon == max_lon:
            return
        self.zoom_settings.zoomed = True
        self.zoom_settings.zoomed_section_bounds = {"min_lat":min_lat,
                                                    "min_lon":min_lon,
                                                    "max_lat":max_lat,
                                                    "max_lon":max_lon}
        self.next_command_to_send = "zoom"
        self.replot_required = True
        self.step()
        self.replot_required = False

    def change_height_range_slider(self,val):
        self.change_height_range(val[0],val[1])

    def change_height_range_text_box(self,event):
        match = self.height_range_text_match.match(event)
        if match:
            self.change_height_range(int(match.group(1)),int(match.group(2)))

    def change_height_range(self,new_min,new_max):
        self.orog_min=new_min
        self.orog_max=new_max
        for index,plot in enumerate(self.configuration):
            if plot in self.orog_plot_types:
                self.replot_required = True
                self.plot_types[plot](index)
                self.replot_required = False
        self.fig.canvas.draw()
        plt.pause(0.001)

    def set_format_coord(self,ax,scale):
        scale_factor = {PlotScales.NORMAL:1,
                        PlotScales.FINE:self.zoom_settings.fine_scale_factor,
                        PlotScales.SUPERFINE:self.zoom_settings.super_fine_scale_factor}[scale]
        ax.format_coord = \
            pts.OrogCoordFormatter(xoffset=
                                   self.zoom_settings.zoomed_section_bounds["min_lon"]*scale_factor,
                                   yoffset=
                                   self.zoom_settings.zoomed_section_bounds["min_lat"]*scale_factor)

def find_highest_version(base_dir_template):
    split_string = base_dir_template.rsplit("VERSION_NUMBER",1)
    wildcarded_base_dir_template = "*".join(split_string)
    versions = glob.glob(wildcarded_base_dir_template)
    version_match = re.compile(r"_([0-9]*)(_|\.)")
    version_numbers = [ int(version_match.match(version.rsplit("_version",1)[1]).group(1))
                        for version in versions]
    return max(version_numbers)

def prepare_matched_catchment_numbers(catchments_one,
                                      catchments_two,
                                      river_mouths_one,
                                      river_mouths_two):
    if ((catchments_one is None) or (catchments_two is None) or
       (river_mouths_one is None) or (river_mouths_two is None)):
       return catchments_one,catchments_two
    params = match_river_mouths.Params("very_extensive")
    conflict_free_pairs, pairs_from_unresolved_conflicts = generate_matches(river_mouths_one,
                                                                            river_mouths_two,params)
    matched_catchments_one = np.zeros(catchments_one.shape)
    matched_catchments_two = np.zeros(catchments_one.shape)
    for pair in conflict_free_pairs:
        catchment_num_one = catchments_one[pair[0].get_coords()]
        catchment_num_two = catchments_two[pair[1].get_coords()]
        matched_catchments_one[catchments_one == catchment_num_one] = catchment_num_one
        matched_catchments_two[catchments_two == catchment_num_two] = catchment_num_one
    return matched_catchments_one, matched_catchments_two

def generate_colour_codes_comp(lsmask,
                               glacier_mask,
                               matched_catchment_nums_one,
                               matched_catchment_nums_two,
                               river_flow_one,
                               river_flow_two,
                               minflowcutoff,
                               use_glacier_mask=True):
    colour_codes = np.zeros(lsmask.shape)
    colour_codes[lsmask == 0] = 1
    colour_codes[np.logical_and(matched_catchment_nums_one != 0,
                                matched_catchment_nums_one ==
                                matched_catchment_nums_two) ] = 5
    colour_codes[np.logical_and(np.logical_or(matched_catchment_nums_one != 0,
                                              matched_catchment_nums_two != 0),
                                matched_catchment_nums_one !=
                                matched_catchment_nums_two) ] = 6
    colour_codes[np.logical_and(river_flow_one >= minflowcutoff,
                                river_flow_two >= minflowcutoff)] = 2
    colour_codes[np.logical_and(river_flow_one >= minflowcutoff,
                                river_flow_two < minflowcutoff)]  = 3
    colour_codes[np.logical_and(river_flow_one < minflowcutoff,
                                river_flow_two >= minflowcutoff)] = 4
    if use_glacier_mask:
        colour_codes[glacier_mask == 1] = 7
    colour_codes[lsmask == 1] = 0
    return colour_codes

def generate_colour_codes_lake_and_river_sequence(cumulative_flow_sequence,
                                                  lake_volume_sequence,
                                                  glacier_mask_sequence,
                                                  landsea_mask_sequence,
                                                  minflowcutoff):
    col_codes_lake_and_river_sequence = []
    for cumulative_flow,lake_volume,glacier_mask,landsea_mask in zip(cumulative_flow_sequence,
                                                                     lake_volume_sequence,
                                                                      glacier_mask_sequence,
                                                                      landsea_mask_sequence):
        col_codes_lake_and_river_sequence.\
            append(generate_colour_codes_lake_and_river(cumulative_flow,
                                                        lake_volume,
                                                        glacier_mask,
                                                        landsea_mask,
                                                        minflowcutoff))
    return col_codes_lake_and_river_sequence

def generate_colour_codes_lake_and_river(cumulative_flow,
                                         lake_volume,
                                         glacier_mask,
                                         landsea_mask,
                                         minflowcutoff):
    rivers_and_lakes = np.zeros(cumulative_flow.shape)
    rivers_and_lakes[cumulative_flow < minflowcutoff] = 1
    rivers_and_lakes[cumulative_flow >= minflowcutoff] = 2
    rivers_and_lakes_fine = utilities.downscale_ls_mask(Field(rivers_and_lakes,grid="HD"),
                                                              "LatLong10min").get_data()
    landsea_mask_fine = utilities.downscale_ls_mask(Field(landsea_mask,grid="HD"),
                                                    "LatLong10min").get_data()
    rivers_and_lakes_fine[lake_volume > 0] = 3
    rivers_and_lakes_fine[glacier_mask == 1] = 4
    rivers_and_lakes_fine[landsea_mask_fine == 1] = 0
    return rivers_and_lakes_fine

def generate_catchment_and_cflow_comp_slice(colors,
                                            lsmask,
                                            glacier_mask,
                                            matched_catchment_nums_one,
                                            matched_catchment_nums_two,
                                            river_flow_one,
                                            river_flow_two,
                                            minflowcutoff,
                                            use_glacier_mask=True):
    colour_codes = generate_colour_codes_comp(lsmask,
                                              glacier_mask,
                                              matched_catchment_nums_one,
                                              matched_catchment_nums_two,
                                              river_flow_one,
                                              river_flow_two,
                                              minflowcutoff,
                                              use_glacier_mask=use_glacier_mask)
    cmap = mpl.colors.ListedColormap(['blue','peru','black','green','red','white','yellow'])
    bounds = list(range(8))
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    im = plt.imshow(colour_codes,cmap=cmap,norm=norm,interpolation="none")
    return im

def extract_zoomed_section(data_in,zoomed_section_bounds,scale_factor=1):
    if data_in is None:
        return None
    return data_in[zoomed_section_bounds["min_lat"]*scale_factor:
                   zoomed_section_bounds["max_lat"]*scale_factor+1,
                   zoomed_section_bounds["min_lon"]*scale_factor:
                   zoomed_section_bounds["max_lon"]*scale_factor+1]

def change_none_to_list(value):
    if value is None:
        return [None]
    else:
        return value

def prep_combined_sequence_tuples(lsmask_sequence,
                                  glacier_mask_sequence,
                                  catchment_nums_one_sequence,
                                  catchment_nums_two_sequence,
                                  river_flow_one_sequence,
                                  river_flow_two_sequence,
                                  river_mouths_one_sequence,
                                  river_mouths_two_sequence,
                                  lake_volume_one_sequence,
                                  lake_volume_two_sequence,
                                  lake_basin_numbers_one_sequence,
                                  lake_basin_numbers_two_sequence,
                                  fine_river_flow_one_sequence,
                                  fine_river_flow_two_sequence,
                                  orography_one_sequence,
                                  orography_two_sequence,
                                  date_text_sequence,
                                  lake_and_river_plots_required=False,
                                  minflowcutoff=100):
    lake_and_river_colour_codes_sequence_one=None
    lake_and_river_colour_codes_sequence_two=None
    if lake_and_river_plots_required:
        if(river_flow_one_sequence is not None and
           lake_volume_one_sequence is not None):
            lake_and_river_colour_codes_sequence_one = \
                generate_colour_codes_lake_and_river_sequence(river_flow_one_sequence,
                                                              lake_volume_one_sequence,
                                                              glacier_mask_sequence,
                                                              lsmask_sequence,
                                                              minflowcutoff)
        if (river_flow_two_sequence is not None and
            lake_volume_two_sequence is not None):
            lake_and_river_colour_codes_sequence_two = \
                generate_colour_codes_lake_and_river_sequence(river_flow_two_sequence,
                                                              lake_volume_two_sequence,
                                                              glacier_mask_sequence,
                                                              lsmask_sequence,
                                                              minflowcutoff)
    return list(itertools.zip_longest(change_none_to_list(lsmask_sequence),
                                      change_none_to_list(glacier_mask_sequence),
                                      change_none_to_list(catchment_nums_one_sequence),
                                      change_none_to_list(catchment_nums_two_sequence),
                                      change_none_to_list(river_flow_one_sequence),
                                      change_none_to_list(river_flow_two_sequence),
                                      change_none_to_list(river_mouths_one_sequence),
                                      change_none_to_list(river_mouths_two_sequence),
                                      change_none_to_list(lake_volume_one_sequence),
                                      change_none_to_list(lake_volume_two_sequence),
                                      change_none_to_list(lake_basin_numbers_one_sequence),
                                      change_none_to_list(lake_basin_numbers_two_sequence),
                                      change_none_to_list(fine_river_flow_one_sequence),
                                      change_none_to_list(fine_river_flow_two_sequence),
                                      change_none_to_list(orography_one_sequence),
                                      change_none_to_list(orography_two_sequence),
                                      change_none_to_list(lake_and_river_colour_codes_sequence_one),
                                      change_none_to_list(lake_and_river_colour_codes_sequence_two),
                                      change_none_to_list(date_text_sequence),
                                      fillvalue=None))

def generate_catchment_and_cflow_sequence_tuple(combined_sequence_tuples,
                                                super_fine_orography=None,
                                                first_corrected_orography=None,
                                                second_corrected_orography=None,
                                                third_corrected_orography=None,
                                                fourth_corrected_orography=None,
                                                true_sinks=None,
                                                bidirectional=False,
                                                zoom_settings=None,
                                                return_zero_slice_only_one=False,
                                                return_zero_slice_only_two=False):
    run_backwards = False
    skip_to_index = False
    i = 0
    while i < len(combined_sequence_tuples) or bidirectional:
        (lsmask_slice,glacier_mask_slice,catchment_nums_one_slice,
         catchment_nums_two_slice,river_flow_one_slice,
         river_flow_two_slice,river_mouths_one_slice,
         river_mouths_two_slice,lake_volume_one_slice,
         lake_volume_two_slice,lake_basin_numbers_one_slice,
         lake_basin_numbers_two_slice,fine_river_flow_one_slice,
         fine_river_flow_two_slice,orography_one_slice,
         orography_two_slice,lake_and_river_colour_codes_one_slice,
         lake_and_river_colour_codes_two_slice,date_text) = combined_sequence_tuples[i]
        if return_zero_slice_only_one:
            if return_zero_slice_only_two:
                (lsmask_slice,glacier_mask_slice,catchment_nums_one_slice,
                 catchment_nums_two_slice,river_flow_one_slice,
                 river_flow_two_slice,river_mouths_one_slice,
                 river_mouths_two_slice,lake_volume_one_slice,
                 lake_volume_two_slice,lake_basin_numbers_one_slice,
                 lake_basin_numbers_two_slice,fine_river_flow_one_slice,
                 fine_river_flow_two_slice,lake_and_river_colour_codes_one_slice,
                 lake_and_river_colour_codes_two_slice,date_text) = combined_sequence_tuples[0]
            else:
                (_,_,catchment_nums_one_slice,
                 _,river_flow_one_slice,
                 _,river_mouths_one_slice,
                 _,lake_volume_one_slice,
                 _,lake_basin_numbers_one_slice,
                 _,fine_river_flow_one_slice,
                 _,lake_and_river_colour_codes_one_slice,
                 _,date_text) = combined_sequence_tuples[0]
        elif return_zero_slice_only_two:
             (_,_,_,
             catchment_nums_two_slice,_,
             river_flow_two_slice,_,
             river_mouths_two_slice,_,
             lake_volume_two_slice,_,
             lake_basin_numbers_two_slice,_,
             fine_river_flow_two_slice,_,
             lake_and_river_colour_codes_two_slice,date_text) = combined_sequence_tuples[0]
        if zoom_settings.zoomed:
            lsmask_slice_zoomed=extract_zoomed_section(lsmask_slice,zoom_settings.zoomed_section_bounds)
            glacier_mask_slice_zoomed=extract_zoomed_section(glacier_mask_slice,
                                                             zoom_settings.zoomed_section_bounds)
            catchment_nums_one_slice_zoomed=extract_zoomed_section(catchment_nums_one_slice,
                                                                   zoom_settings.zoomed_section_bounds)
            catchment_nums_two_slice_zoomed=extract_zoomed_section(catchment_nums_two_slice,
                                                                   zoom_settings.zoomed_section_bounds)
            river_flow_one_slice_zoomed=extract_zoomed_section(river_flow_one_slice,
                                                               zoom_settings.zoomed_section_bounds)
            river_flow_two_slice_zoomed=extract_zoomed_section(river_flow_two_slice,
                                                               zoom_settings.zoomed_section_bounds)
            river_mouths_one_slice_zoomed=extract_zoomed_section(river_mouths_one_slice,
                                                                 zoom_settings.zoomed_section_bounds)
            river_mouths_two_slice_zoomed=extract_zoomed_section(river_mouths_two_slice,
                                                                 zoom_settings.zoomed_section_bounds)
            lake_volume_one_slice_zoomed=extract_zoomed_section(lake_volume_one_slice,
                                                                zoom_settings.zoomed_section_bounds,
                                                                zoom_settings.fine_scale_factor)
            lake_volume_two_slice_zoomed=extract_zoomed_section(lake_volume_two_slice,
                                                                zoom_settings.zoomed_section_bounds,
                                                                zoom_settings.fine_scale_factor)
            lake_basin_numbers_one_slice_zoomed=extract_zoomed_section(lake_basin_numbers_one_slice,
                                                                       zoom_settings.zoomed_section_bounds,
                                                                       zoom_settings.fine_scale_factor)
            lake_basin_numbers_two_slice_zoomed=extract_zoomed_section(lake_basin_numbers_two_slice,
                                                                       zoom_settings.zoomed_section_bounds,
                                                                       zoom_settings.fine_scale_factor)
            fine_river_flow_one_slice_zoomed=extract_zoomed_section(fine_river_flow_one_slice,
                                                                    zoom_settings.zoomed_section_bounds,
                                                                    zoom_settings.fine_scale_factor)
            fine_river_flow_two_slice_zoomed=extract_zoomed_section(fine_river_flow_two_slice,
                                                                    zoom_settings.zoomed_section_bounds,
                                                                    zoom_settings.fine_scale_factor)
            orography_one_slice_zoomed=extract_zoomed_section(orography_one_slice,
                                                              zoom_settings.zoomed_section_bounds,
                                                              zoom_settings.fine_scale_factor)
            orography_two_slice_zoomed=extract_zoomed_section(orography_two_slice,
                                                              zoom_settings.zoomed_section_bounds,
                                                              zoom_settings.fine_scale_factor)
            super_fine_orography_slice_zoomed=extract_zoomed_section(super_fine_orography,
                                                                     zoom_settings.zoomed_section_bounds,
                                                                     zoom_settings.super_fine_scale_factor)
            first_corrected_orography_slice_zoomed=extract_zoomed_section(first_corrected_orography,
                                                                          zoom_settings.zoomed_section_bounds,
                                                                          zoom_settings.fine_scale_factor)
            second_corrected_orography_slice_zoomed=extract_zoomed_section(second_corrected_orography,
                                                                           zoom_settings.zoomed_section_bounds,
                                                                           zoom_settings.fine_scale_factor)
            third_corrected_orography_slice_zoomed=extract_zoomed_section(third_corrected_orography,
                                                                          zoom_settings.zoomed_section_bounds,
                                                                          zoom_settings.fine_scale_factor)
            fourth_corrected_orography_slice_zoomed=extract_zoomed_section(fourth_corrected_orography,
                                                                           zoom_settings.zoomed_section_bounds,
                                                                           zoom_settings.fine_scale_factor)
            true_sinks_slice_zoomed=extract_zoomed_section(true_sinks,
                                                           zoom_settings.zoomed_section_bounds,
                                                           zoom_settings.fine_scale_factor)
            lake_and_river_colour_codes_one_slice_zoomed=extract_zoomed_section(lake_and_river_colour_codes_one_slice,
                                                                                zoom_settings.zoomed_section_bounds,
                                                                                zoom_settings.fine_scale_factor)
            lake_and_river_colour_codes_two_slice_zoomed=extract_zoomed_section(lake_and_river_colour_codes_two_slice,
                                                                                zoom_settings.zoomed_section_bounds,
                                                                                zoom_settings.fine_scale_factor)
        else:
            lsmask_slice_zoomed=lsmask_slice
            glacier_mask_slice_zoomed=glacier_mask_slice
            catchment_nums_one_slice_zoomed=catchment_nums_one_slice
            catchment_nums_two_slice_zoomed=catchment_nums_two_slice
            river_flow_one_slice_zoomed=river_flow_one_slice
            river_flow_two_slice_zoomed=river_flow_two_slice
            river_mouths_one_slice_zoomed=river_mouths_one_slice
            river_mouths_two_slice_zoomed=river_mouths_two_slice
            lake_volume_one_slice_zoomed=lake_volume_one_slice
            lake_volume_two_slice_zoomed=lake_volume_two_slice
            lake_basin_numbers_one_slice_zoomed=lake_basin_numbers_one_slice
            lake_basin_numbers_two_slice_zoomed=lake_basin_numbers_two_slice
            fine_river_flow_one_slice_zoomed=fine_river_flow_one_slice
            fine_river_flow_two_slice_zoomed=fine_river_flow_two_slice
            orography_one_slice_zoomed=orography_one_slice
            orography_two_slice_zoomed=orography_two_slice
            super_fine_orography_slice_zoomed=super_fine_orography
            first_corrected_orography_slice_zoomed=first_corrected_orography
            second_corrected_orography_slice_zoomed=second_corrected_orography
            third_corrected_orography_slice_zoomed=third_corrected_orography
            fourth_corrected_orography_slice_zoomed=fourth_corrected_orography
            true_sinks_slice_zoomed=true_sinks
            lake_and_river_colour_codes_one_slice_zoomed=lake_and_river_colour_codes_one_slice
            lake_and_river_colour_codes_two_slice_zoomed=lake_and_river_colour_codes_two_slice
        matched_catchment_nums_one,matched_catchment_nums_two =\
             prepare_matched_catchment_numbers(catchment_nums_one_slice_zoomed,
                                               catchment_nums_two_slice_zoomed,
                                               river_mouths_one_slice_zoomed,
                                               river_mouths_two_slice_zoomed)
        input_from_send = yield (lsmask_slice_zoomed,
                                 glacier_mask_slice_zoomed,
                                 matched_catchment_nums_one,
                                 matched_catchment_nums_two,
                                 river_flow_one_slice_zoomed,
                                 river_flow_two_slice_zoomed,
                                 lake_volume_one_slice_zoomed,
                                 lake_volume_two_slice_zoomed,
                                 lake_basin_numbers_one_slice_zoomed,
                                 lake_basin_numbers_two_slice_zoomed,
                                 fine_river_flow_one_slice_zoomed,
                                 fine_river_flow_two_slice_zoomed,
                                 orography_one_slice_zoomed,
                                 orography_two_slice_zoomed,
                                 super_fine_orography_slice_zoomed,
                                 first_corrected_orography_slice_zoomed,
                                 second_corrected_orography_slice_zoomed,
                                 third_corrected_orography_slice_zoomed,
                                 fourth_corrected_orography_slice_zoomed,
                                 true_sinks_slice_zoomed,
                                 lake_and_river_colour_codes_one_slice_zoomed,
                                 lake_and_river_colour_codes_two_slice_zoomed,
                                 date_text)
        if bidirectional:
            if not isinstance(input_from_send,bool):
                if input_from_send == "zoom":
                    continue
                elif isinstance(input_from_send,int):
                    skip_to_index = True
                    new_index = input_from_send
                else:
                    raise RuntimeError("Generator recieved unknown command")
            else:
                skip_to_index = False
                run_backwards = input_from_send
        if skip_to_index:
            i = new_index
        else:
            i += (1 if not run_backwards else -1)
        if bidirectional:
            if i < 0:
                i = 0
                UserWarning("Already at first slice")
            elif i >= len(combined_sequence_tuples):
                i = len(combined_sequence_tuples) - 1
                UserWarning("Already at last slice")

def generate_catchment_and_cflow_comp_sequence(colors,
                                               lsmask_sequence,
                                               glacier_mask_sequence,
                                               catchment_nums_one_sequence,
                                               catchment_nums_two_sequence,
                                               river_flow_one_sequence,
                                               river_flow_two_sequence,
                                               river_mouths_one_sequence,
                                               river_mouths_two_sequence,
                                               date_text_sequence,
                                               minflowcutoff,
                                               use_glacier_mask=True,
                                               zoomed=False,
                                               zoomed_section_bounds={}):
    im_list = []
    combined_sequence_tuples = prep_combined_sequence_tuples(lsmask_sequence,
                                                             glacier_mask_sequence,
                                                             catchment_nums_one_sequence,
                                                             catchment_nums_two_sequence,
                                                             river_flow_one_sequence,
                                                             river_flow_two_sequence,
                                                             river_mouths_one_sequence,
                                                             river_mouths_two_sequence,
                                                             None,None,None,None,
                                                             date_text_sequence,False)
    for (lsmask_slice_zoomed,glacier_mask_slice_zoomed,
         matched_catchment_nums_one,matched_catchment_nums_two,
         river_flow_one_slice_zoomed,river_flow_two_slice_zoomed,_,_,_,_,_,_,_,_,_,
         date_text) in generate_catchment_and_cflow_sequence_tuple(combined_sequence_tuples,
                                                                   ZoomSettings(zoomed,
                                                                                zoomed_section_bounds)):
        im_list.append([generate_catchment_and_cflow_comp_slice(colors,
                                                                lsmask_slice_zoomed,
                                                                glacier_mask_slice_zoomed,
                                                                matched_catchment_nums_one,
                                                                matched_catchment_nums_two,
                                                                river_flow_one_slice_zoomed,
                                                                river_flow_two_slice_zoomed,
                                                                minflowcutoff=minflowcutoff,
                                                                use_glacier_mask=
                                                                use_glacier_mask),date_text])
    return im_list
