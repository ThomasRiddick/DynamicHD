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
import matplotlib.backends
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from plotting_utilities.match_river_mouths import generate_matches
from plotting_utilities import match_river_mouths
from plotting_utilities import plotting_tools as pts
from plotting_utilities.lake_analysis_tools import SpillwayProfiler
from Dynamic_HD_Scripts import utilities
from Dynamic_HD_Scripts.base.iodriver import advanced_field_loader
from Dynamic_HD_Scripts.base.field import Field
from os.path import join
from enum import Enum

class PlotScales(Enum):
    NORMAL = 1
    FINE = 2
    SUPERFINE = 3

class TimeSlicePlot():

    def __init__(self,ax):
        self.ax = ax
        self.plot = None
        self.selector = None
        self.scale = PlotScales.NORMAL

class TimeSeriesPlot:

    def __init__(self,ax):
        self.ax = ax

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

class TimeSequences:
    def __init__(self,dates,
                 sequence_one_base_dir,
                 sequence_two_base_dir,
                 glacier_mask_file_template,
                 super_fine_orography_filepath,
                 use_connected_catchments=True,
                 missing_fields=[],
                 use_latest_version_for_sequence_one=True,
                 sequence_one_fixed_version=-1,
                 use_latest_version_for_sequence_two=True,
                 sequence_two_fixed_version=-1,
                 **kwargs):
        self.lsmask_sequence = []
        self.glacier_mask_sequence = []
        self.catchment_nums_one_sequence = []
        self.river_flow_one_sequence= []
        self.river_mouths_one_sequence = []
        if not "lake_volumes_one" in missing_fields:
            self.lake_volumes_one_sequence = []
        else:
            self.lake_volumes_one_sequence = None
        if not "lake_basin_numbers_one" in missing_fields:
            self.lake_basin_numbers_one_sequence = []
        else:
            self.lake_basin_numbers_one_sequence = None
        self.fine_river_flow_one_sequence= []
        if not "orography_one" in missing_fields:
            self.orography_one_sequence = []
        else:
            self.orography_one_sequence = None
        self.catchment_nums_two_sequence = []
        self.river_flow_two_sequence= []
        self.river_mouths_two_sequence = []
        if not "lake_volumes_two" in missing_fields:
            self.lake_volumes_two_sequence = []
        else:
            self.lake_volumes_two_sequence = None
        if not "lake_basin_numbers_two" in missing_fields:
            self.lake_basin_numbers_two_sequence = []
        else:
            self.lake_basin_numbers_two_sequence = None
        self.fine_river_flow_two_sequence= []
        if not "orography_two" in missing_fields:
            self.orography_two_sequence = []
        else:
            self.orography_two_sequence = None
        if not "filled_orography_one" in missing_fields:
            self.filled_orography_one_sequence = []
        else:
            self.filled_orography_one_sequence = None
        if not "filled_orography_two" in missing_fields:
            self.filled_orography_two_sequence = []
        else:
            self.filled_orography_two_sequence = None
        if not "sinkless_rdirs_one" in missing_fields:
            self.sinkless_rdirs_one_sequence = []
        else:
            self.sinkless_rdirs_one_sequence = None,
        if not "sinkless_rdirs_two" in missing_fields:
            self.sinkless_rdirs_two_sequence = []
        else:
            self.sinkless_rdirs_two_sequence = None,
        self.date_sequence = dates
        for date in self.date_sequence:
            #Note latest version may differ between dates hence calculate this
            #on a date by date basis
            if use_latest_version_for_sequence_one:
                sequence_one_lakes_version = \
                    find_highest_version(sequence_one_base_dir +
                                        "lakes/results/"
                                        "diag_version_VERSION_NUMBER_date_{}".format(date))
            else:
                sequence_one_lakes_version = sequence_one_fixed_version
            if use_latest_version_for_sequence_two:
                sequence_two_lakes_version = \
                    find_highest_version(sequence_two_base_dir +
                                        "lakes/results/"
                                        "diag_version_VERSION_NUMBER_date_{}".format(date))
            else:
                sequence_two_lakes_version = sequence_two_fixed_version
            sequence_one_results_base_dir = (sequence_one_base_dir +
                                             "lakes/results/diag_version_{}_date_{}".\
                                               format(sequence_one_lakes_version,date))
            sequence_two_results_base_dir = (sequence_two_base_dir +
                                                  "lakes/results/diag_version_{}_date_{}".\
                                                  format(sequence_two_lakes_version,date))
            rdirs = advanced_field_loader(filename=join(sequence_one_results_base_dir,"30min_rdirs.nc"),
                                          time_slice=None,
                                          field_type="RiverDirections",
                                          fieldname="rdirs",
                                          adjust_orientation=True)
            lsmask_data = rdirs.get_lsmask()
            self.lsmask_sequence.append(lsmask_data)
            glacier_mask = advanced_field_loader(filename=glacier_mask_file_template.replace("DATE",str(date)),
                                                 time_slice=None,
                                                 fieldname="glac",
                                                 adjust_orientation=True)
            self.glacier_mask_sequence.append(glacier_mask.get_data())
            catchment_nums_one = advanced_field_loader(filename=join(sequence_one_results_base_dir,
                                                                     "30min_connected_catchments.nc"
                                                                     if use_connected_catchments else
                                                                     "30min_catchments.nc"),
                                                          time_slice=None,
                                                          fieldname="catchments",
                                                          adjust_orientation=True)
            self.catchment_nums_one_sequence.append(catchment_nums_one.get_data())
            river_flow_one = advanced_field_loader(filename=join(sequence_one_results_base_dir,
                                                                 "30min_flowtocell_connected.nc"
                                                                 if use_connected_catchments else
                                                                 "30min_flowtocell.nc"),
                                                      time_slice=None,
                                                      fieldname="cumulative_flow",
                                                      adjust_orientation=True)
            self.river_flow_one_sequence.append(river_flow_one.get_data())
            river_mouths_one = advanced_field_loader(filename=join(sequence_one_results_base_dir,
                                                                      "30min_flowtorivermouths_connected.nc"),
                                                        time_slice=None,
                                                        fieldname="cumulative_flow_to_ocean",
                                                        adjust_orientation=True)
            self.river_mouths_one_sequence.append(river_mouths_one.get_data())
            if not "lake_volumes_one" in missing_fields:
                lake_volumes_one = advanced_field_loader(filename=join(sequence_one_results_base_dir,
                                                                         "10min_lake_volumes.nc"),
                                                           time_slice=None,
                                                           fieldname="lake_volume",
                                                           adjust_orientation=True)
                self.lake_volumes_one_sequence.append(lake_volumes_one.get_data())
            if not "lake_basin_numbers_one" in missing_fields:
                lake_basin_numbers_one = advanced_field_loader(filename=join(sequence_one_results_base_dir,
                                                                           "basin_catchment_numbers.nc"),
                                                                           time_slice=None,
                                                                           fieldname="basin_catchment_numbers",
                                                                           adjust_orientation=True)
                self.lake_basin_numbers_one_sequence.append(lake_basin_numbers_one.get_data())
            if not "fine_river_flow_one" in missing_fields:
                fine_river_flow_one = advanced_field_loader(filename=join(sequence_one_results_base_dir,
                                                                             "10min_flowtocell.nc"),
                                                               time_slice=None,
                                                               fieldname="cumulative_flow",
                                                               adjust_orientation=True)
                self.fine_river_flow_one_sequence.append(fine_river_flow_one.get_data())
            else:
                self.fine_river_flow_one_sequence = None
            if not "orography_one" in missing_fields:
                orography_one = advanced_field_loader(filename=join(sequence_one_results_base_dir,
                                                                    "10min_corrected_orog.nc"),
                                                         time_slice=None,
                                                         fieldname="corrected_orog",
                                                         adjust_orientation=True)
                self.orography_one_sequence.append(orography_one.get_data())
            if not "orography_two" in missing_fields:
                orography_two = advanced_field_loader(filename=join(sequence_two_results_base_dir,
                                                                    "10min_corrected_orog.nc"),
                                                         time_slice=None,
                                                         fieldname="corrected_orog",
                                                         adjust_orientation=True)
                self.orography_two_sequence.append(orography_two.get_data())
            catchment_nums_two = advanced_field_loader(filename=join(sequence_two_results_base_dir,
                                                                      "30min_connected_catchments.nc"),
                                                        time_slice=None,
                                                        fieldname="catchments",
                                                        adjust_orientation=True)
            self.catchment_nums_two_sequence.append(catchment_nums_two.get_data())
            river_flow_two = advanced_field_loader(filename=join(sequence_two_results_base_dir,
                                                                  "30min_flowtocell_connected.nc"),
                                                    time_slice=None,
                                                    fieldname="cumulative_flow",
                                                    adjust_orientation=True)
            self.river_flow_two_sequence.append(river_flow_two.get_data())
            river_mouths_two = advanced_field_loader(filename=join(sequence_two_results_base_dir,
                                                                    "30min_flowtorivermouths_connected.nc"),
                                                      time_slice=None,
                                                      fieldname="cumulative_flow_to_ocean",
                                                      adjust_orientation=True)
            self.river_mouths_two_sequence.append(river_mouths_two.get_data())
            if not "lake_volumes_two" in missing_fields:
                lake_volumes_two = advanced_field_loader(filename=join(sequence_two_results_base_dir,
                                                                       "10min_lake_volumes.nc"),
                                                         time_slice=None,
                                                         fieldname="lake_volume",
                                                         adjust_orientation=True)
                self.lake_volumes_two_sequence.append(lake_volumes_two.get_data())
            if not "lake_basin_numbers_two" in missing_fields:
                lake_basin_numbers_two = advanced_field_loader(filename=join(sequence_two_results_base_dir,
                                                                         "basin_catchment_numbers.nc"),
                                                                         time_slice=None,
                                                                         fieldname="basin_catchment_numbers",
                                                                         adjust_orientation=True)
                self.lake_basin_numbers_two_sequence.append(lake_basin_numbers_two.get_data())
            if not "fine_river_flow_two" in missing_fields:
                fine_river_flow_two = advanced_field_loader(filename=join(sequence_two_results_base_dir,
                                                            "10min_flowtocell.nc"),
                                                            time_slice=None,
                                                            fieldname="cumulative_flow",
                                                            adjust_orientation=True)
                self.fine_river_flow_two_sequence.append(fine_river_flow_two.get_data())
            else:
                self.fine_river_flow_two_sequence = None
            if not "filled_orography_one" in missing_fields:
                filled_orography_one = advanced_field_loader(filename=join(sequence_one_results_base_dir,
                                                                           "10min_filled_orog.nc"),
                                                             time_slice=None,
                                                             fieldname="filled_orog",
                                                             adjust_orientation=True)
                self.filled_orography_one_sequence.append(filled_orography_one.get_data())
            if not "filled_orography_two" in missing_fields:
                filled_orography_two = advanced_field_loader(filename=join(sequence_two_results_base_dir,
                                                                           "10min_filled_orog.nc"),
                                                             time_slice=None,
                                                             fieldname="filled_orog",
                                                             adjust_orientation=True)
                self.filled_orography_two_sequence.append(filled_orography_two.get_data())
            if not "sinkless_rdirs_one" in missing_fields:
                sinkless_rdirs_one = advanced_field_loader(filename=join(sequence_one_results_base_dir,
                                                                           "10min_sinkless_rdirs.nc"),
                                                           time_slice=None,
                                                           fieldname="rdirs",
                                                           adjust_orientation=True)
                self.sinkless_rdirs_one_sequence.append(sinkless_rdirs_one.get_data())
            if not "sinkless_rdirs_two" in missing_fields:
                sinkless_rdirs_two = advanced_field_loader(filename=join(sequence_two_results_base_dir,
                                                                           "10min_sinkless_rdirs.nc"),
                                                           time_slice=None,
                                                           fieldname="rdirs",
                                                           adjust_orientation=True)
                self.sinkless_rdirs_two_sequence.append(sinkless_rdirs_two.get_data())
        if not "super_fine_orography" in missing_fields:
            self.super_fine_orography = advanced_field_loader(filename=join(super_fine_orography_filepath),
                                                         time_slice=None,
                                                         fieldname="topo",
                                                         adjust_orientation=True).get_data()
        else:
            self.super_fine_orography = None
        if not "corrected_orographies" in missing_fields:
            self.first_corrected_orography = advanced_field_loader(filename=join(sequence_one_base_dir,
                                                                            "corrections","work",
                                                              "pre_preliminary_tweak_orography.nc"),
                                                              time_slice=None,
                                                              fieldname="orog",
                                                              adjust_orientation=True).get_data()
            self.second_corrected_orography = advanced_field_loader(filename=join(sequence_one_base_dir,
                                                                             "corrections","work",
                                                               "post_preliminary_tweak_orography.nc"),
                                                               time_slice=None,
                                                               fieldname="orog",
                                                               adjust_orientation=True).get_data()
            self.third_corrected_orography = advanced_field_loader(filename=join(sequence_one_base_dir,
                                                                            "corrections","work",
                                                              "pre_final_tweak_orography.nc"),
                                                             time_slice=None,
                                                             fieldname="orog",
                                                             adjust_orientation=True).get_data()
            self.fourth_corrected_orography = advanced_field_loader(filename=join(sequence_one_base_dir,
                                                                            "corrections","work",
                                                               "post_final_tweak_orography.nc"),
                                                               time_slice=None,
                                                               fieldname="orog",
                                                               adjust_orientation=True).get_data()
        else:
            self.first_corrected_orography = None
            self.second_corrected_orography = None
            self.third_corrected_orography = None
            self.fourth_corrected_orography = None
        if not "true_sinks" in missing_fields:
            highest_true_sinks_version = find_highest_version(join(sequence_one_base_dir,
                                                             "corrections","true_sinks_fields",
                                                             "true_sinks_field_version_"
                                                             "VERSION_NUMBER.nc"))
            self.true_sinks = advanced_field_loader(filename=join(sequence_one_base_dir,
                                                                  "corrections","true_sinks_fields",
                                                                  "true_sinks_field_version_{}.nc".\
                                                                  format(highest_true_sinks_version)),
                                                                  time_slice=None,
                                                                  fieldname="true_sinks",
                                                                  adjust_orientation=True).get_data()
        else:
            self.true_sinks = None

class InteractiveSpillwayPlots:

    def __init__(self,colors,
                 **kwargs):
        mpl.use('TkAgg')
        self.show_plot_one = True
        self.setup_sequences(**kwargs)
        self.spillway_plot_axes = []
        self.figs = []
        fig = plt.figure(figsize=(9,4),facecolor="#E3F2FD",dpi=200)
        self.figs.append(fig)
        gs=gridspec.GridSpec(nrows=1,ncols=1,width_ratios=[1],
                             height_ratios=[1],
                             hspace=0.1,
                             wspace=0.1)
        self.spillway_plot_axes.append(fig.add_subplot(gs[0,0]))
        fig = plt.figure(figsize=(9,4),facecolor="#E3F2FD",dpi=200)
        self.figs.append(fig)
        gs=gridspec.GridSpec(nrows=2,ncols=1,width_ratios=[1],
                             height_ratios=[1,1],
                             hspace=0.1,
                             wspace=0.1)
        self.spillway_plot_axes.append(fig.add_subplot(gs[0,0]))
        self.spillway_plot_axes.append(fig.add_subplot(gs[1,0]))
        self.step()

    def replot(self,**kwargs):
        self.setup_sequences(**kwargs)
        self.step()

    def step(self):
        spillway_profile_one = SpillwayProfiler.\
            extract_spillway_profile(lake_center=
                                     self.lake_center_one_sequence[self.current_index],
                                     sinkless_rdirs=
                                     self.sinkless_rdirs_one_sequence[self.current_index],
                                     corrected_heights=
                                     self.orography_one_sequence[self.current_index])
        spillway_profile_two = SpillwayProfiler.\
            extract_spillway_profile(lake_center=
                                     self.lake_center_two_sequence[self.current_index],
                                     sinkless_rdirs=
                                     self.sinkless_rdirs_two_sequence[self.current_index],
                                     corrected_heights=
                                     self.orography_two_sequence[self.current_index])
        self.plot_spillway_profile(spillway_profile_one if self.show_plot_one else
                                   spillway_profile_two,0)
        self.plot_spillway_profile(spillway_profile_one,1)
        self.plot_spillway_profile(spillway_profile_two,2)

    def step_back(self):
        if self.current_index > 0:
            self.current_index -= 1
        self.step()

    def step_forward(self):
        if self.current_index < self.max_index - 1:
            self.current_index += 1
        self.step()

    def step_to_date(self,event):
        self.current_index = self.date_sequence.index(int(event))
        self.step()

    def toggle_plot_one(self):
        self.show_plot_one = True
        self.step()

    def toggle_plot_two(self):
        self.show_plot_one = False
        self.step()

    def plot_spillway_profile(self,profile,index):
        self.spillway_plot_axes[index].clear()
        self.spillway_plot_axes[index].plot(profile[:-2])

    def get_current_date(self):
        return self.date_sequence[self.current_index]

    def setup_sequences(self,date_sequence,
                        lake_center_one_sequence,
                        lake_center_two_sequence,
                        sinkless_rdirs_one_sequence,
                        sinkless_rdirs_two_sequence,
                        orography_one_sequence,
                        orography_two_sequence,
                        **kwargs):
        self.date_sequence = date_sequence
        self.lake_center_one_sequence=lake_center_one_sequence
        self.lake_center_two_sequence=lake_center_two_sequence
        self.sinkless_rdirs_one_sequence=sinkless_rdirs_one_sequence
        self.sinkless_rdirs_two_sequence=sinkless_rdirs_two_sequence
        self.orography_one_sequence = orography_one_sequence
        self.orography_two_sequence = orography_two_sequence
        self.current_index = 0
        self.max_index = min(len(self.orography_one_sequence),
                             len(self.orography_two_sequence))

class InteractiveTimeSeriesPlots():

    def __init__(self,colors,
                 **kwargs):
        mpl.use('TkAgg')
        self.setup_sequences(**kwargs)
        self.plot_types = {"none":self.no_plot,
                           "height1":self.lake_height_plot_one,
                           "height2":self.lake_height_plot_two,
                           "volume1":self.lake_volume_plot_one,
                           "volume2":self.lake_volume_plot_two,
                           "outflow1":self.lake_outflow_plot_one,
                           "outflow2":self.lake_outflow_plot_two }
        self.plot_configuration = ["height1" for _ in range(10)]
        self.figs = []
        self.timeseries_plots = []
        fig = plt.figure(figsize=(9,4),facecolor="#E3F2FD",dpi=200)
        self.figs.append(fig)
        gs=gridspec.GridSpec(nrows=1,ncols=1,width_ratios=[1],
                             height_ratios=[1],
                             hspace=0.1,
                             wspace=0.1)
        self.timeseries_plots.append(TimeSeriesPlot(fig.add_subplot(gs[0,0])))
        fig = plt.figure(figsize=(9,4),facecolor="#E3F2FD",dpi=200)
        self.figs.append(fig)
        gs=gridspec.GridSpec(nrows=2,ncols=1,width_ratios=[1],
                             height_ratios=[1,1],
                             hspace=0.1,
                             wspace=0.1)
        self.timeseries_plots.append(TimeSeriesPlot(fig.add_subplot(gs[0,0])))
        self.timeseries_plots.append(TimeSeriesPlot(fig.add_subplot(gs[1,0])))
        fig = plt.figure(figsize=(9,4),facecolor="#E3F2FD",dpi=200)
        self.figs.append(fig)
        gs=gridspec.GridSpec(nrows=3,ncols=1,width_ratios=[1],
                             height_ratios=[1,1,1],
                             hspace=0.1,
                             wspace=0.1)
        self.timeseries_plots.append(TimeSeriesPlot(fig.add_subplot(gs[0,0])))
        self.timeseries_plots.append(TimeSeriesPlot(fig.add_subplot(gs[1,0])))
        self.timeseries_plots.append(TimeSeriesPlot(fig.add_subplot(gs[2,0])))
        fig = plt.figure(figsize=(9,4),facecolor="#E3F2FD",dpi=200)
        self.figs.append(fig)
        gs=gridspec.GridSpec(nrows=4,ncols=1,width_ratios=[1],
                             height_ratios=[1,1,1,1],
                             hspace=0.1,
                             wspace=0.1)
        self.timeseries_plots.append(TimeSeriesPlot(fig.add_subplot(gs[0,0])))
        self.timeseries_plots.append(TimeSeriesPlot(fig.add_subplot(gs[1,0])))
        self.timeseries_plots.append(TimeSeriesPlot(fig.add_subplot(gs[2,0])))
        self.timeseries_plots.append(TimeSeriesPlot(fig.add_subplot(gs[3,0])))
        self.step()

    def setup_sequences(self,date_sequence,
                        lake_heights_one_sequence,
                        lake_heights_two_sequence,
                        lake_volume_one_sequence,
                        lake_volume_two_sequence,
                        lake_outflow_basin_one_sequence,
                        lake_outflow_basin_two_sequence,
                        **kwargs):
        self.date_sequence = date_sequence
        self.lake_heights_one_sequence = lake_heights_one_sequence
        self.lake_heights_two_sequence = lake_heights_two_sequence
        self.lake_volume_one_sequence = lake_volume_one_sequence
        self.lake_volume_two_sequence = lake_volume_two_sequence
        self.lake_outflow_basin_one_sequence = lake_outflow_basin_one_sequence
        self.lake_outflow_basin_two_sequence = lake_outflow_basin_two_sequence

    def replot(self,**kwargs):
        self.setup_sequences(**kwargs)
        self.step()

    def step(self):
        for index,plot in enumerate(self.plot_configuration):
            if plot is not None:
                self.plot_types[plot](index)

    def set_plot_type(self,plot_index,plot_type):
        self.plot_configuration[plot_index] = plot_type
        self.plot_types[plot_type](plot_index)

    def no_plot(self,index):
        self.timeslice_plots[index].ax.set_visible(False)

    def lake_height_plot_base(self,index,lake_height_sequence):
        self.timeseries_plots[index].ax.clear()
        self.timeseries_plots[index].ax.plot(lake_height_sequence)

    def lake_height_plot_one(self,index):
        self.lake_height_plot_base(index,self.lake_heights_one_sequence)

    def lake_height_plot_two(self,index):
        self.lake_height_plot_base(index,self.lake_heights_two_sequence)

    def lake_volume_plot_base(self,index,lake_volume_sequence):
        self.timeseries_plots[index].ax.clear()
        self.timeseries_plots[index].ax.plot(lake_volume_sequence)

    def lake_volume_plot_one(self,index):
        self.lake_volume_plot_base(index,self.lake_volume_one_sequence)

    def lake_volume_plot_two(self,index):
        self.lake_volume_plot_base(index,self.lake_volume_two_sequence)

    def lake_outflow_plot_base(self,index,lake_outflow_sequence):
        self.timeseries_plots[index].ax.clear()
        carib = np.zeros((len(lake_outflow_sequence)),dtype=np.int32)
        artic = np.zeros((len(lake_outflow_sequence)),dtype=np.int32)
        saintlawrence = np.zeros((len(lake_outflow_sequence)),dtype=np.int32)
        for i,item in enumerate(lake_outflow_sequence):
            if item == "Carib":
                carib[i] = 1
            if item == "Artic":
                artic[i] = 1
            if item == "St Lawrence":
                saintlawrence[i] = 1
        self.timeseries_plots[index].ax.plot(carib,color="blue")
        self.timeseries_plots[index].ax.plot(artic,color="red")
        self.timeseries_plots[index].ax.plot(saintlawrence,color="green")

    def lake_outflow_plot_one(self,index):
        self.lake_outflow_plot_base(index,self.lake_outflow_basin_one_sequence)

    def lake_outflow_plot_two(self,index):
        self.lake_outflow_plot_base(index,self.lake_outflow_basin_two_sequence)

class InteractiveTimeSlicePlots:

    def __init__(self,colors,
                 initial_plot_configuration,
                 minflowcutoff,
                 use_glacier_mask,
                 zoomed,
                 zoomed_section_bounds,
                 zero_slice_only_one=False,
                 zero_slice_only_two=False,
                 dynamic_configuration=False,
                 **kwargs):
        mpl.use('TkAgg')
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
        self.plot_configuration = [initial_plot_configuration[0] for _ in range(13)]
        offset = 0
        for j in [1,2,4,6]:
            for i,plot in enumerate(initial_plot_configuration):
                self.plot_configuration[i + offset] = initial_plot_configuration[i]
                if i == j-1:
                    break
            offset += j
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
        self.zero_slice_only_one = zero_slice_only_one
        self.zero_slice_only_two = zero_slice_only_two
        self.original_zoom_settings = self.zoom_settings.copy()
        self.fine_cutoff_scaling = self.zoom_settings.fine_scale_factor*self.zoom_settings.fine_scale_factor
        self.replot_required = False
        self.figs = []
        fig = plt.figure(figsize=(9,4),facecolor="#E3F2FD",dpi=200)
        self.figs.append(fig)
        gs=gridspec.GridSpec(nrows=1,ncols=1,width_ratios=[1],
                             height_ratios=[1],
                             hspace=0.1,
                             wspace=0.1)
        self.timeslice_plots.append(TimeSlicePlot(fig.add_subplot(gs[0,0])))
        fig = plt.figure(figsize=(9,4),facecolor="#E3F2FD",dpi=200)
        self.figs.append(fig)
        gs=gridspec.GridSpec(nrows=2,ncols=1,width_ratios=[1],
                             height_ratios=[1,1],
                             hspace=0.1,
                             wspace=0.1)
        self.timeslice_plots.append(TimeSlicePlot(fig.add_subplot(gs[0,0])))
        self.timeslice_plots.append(TimeSlicePlot(fig.add_subplot(gs[1,0])))
        fig = plt.figure(figsize=(9,4),facecolor="#E3F2FD",dpi=200)
        self.figs.append(fig)
        gs=gridspec.GridSpec(nrows=2,ncols=2,width_ratios=[1,1],
                             height_ratios=[1,1],
                             hspace=0.1,
                             wspace=0.1)
        self.timeslice_plots.append(TimeSlicePlot(fig.add_subplot(gs[0,0])))
        self.timeslice_plots.append(TimeSlicePlot(fig.add_subplot(gs[0,1])))
        self.timeslice_plots.append(TimeSlicePlot(fig.add_subplot(gs[1,0])))
        self.timeslice_plots.append(TimeSlicePlot(fig.add_subplot(gs[1,1])))
        fig = plt.figure(figsize=(9,4),facecolor="#E3F2FD",dpi=200)
        self.figs.append(fig)
        gs=gridspec.GridSpec(nrows=3,ncols=2,width_ratios=[1,1],
                             height_ratios=[1,1,1],
                             hspace=0.1,
                             wspace=0.1)
        self.timeslice_plots.append(TimeSlicePlot(fig.add_subplot(gs[0,0])))
        self.timeslice_plots.append(TimeSlicePlot(fig.add_subplot(gs[0,1])))
        self.timeslice_plots.append(TimeSlicePlot(fig.add_subplot(gs[1,0])))
        self.timeslice_plots.append(TimeSlicePlot(fig.add_subplot(gs[1,1])))
        self.timeslice_plots.append(TimeSlicePlot(fig.add_subplot(gs[2,0])))
        self.timeslice_plots.append(TimeSlicePlot(fig.add_subplot(gs[2,1])))
        # if (not set(self.plot_configuration).isdisjoint(self.cflow_plot_types)):
        #     #add zero to list to prevent errors if the two sequences are Nones
        #     maxflow =   max([np.max(slice) if slice is not None else 0
        #                     for slice in itertools.chain(river_flow_one_sequence if
        #                                                  river_flow_one_sequence is
        #                                                  not None else [],
        #                                                  river_flow_two_sequence if
        #                                                  river_flow_two_sequence is
        #                                                  not None else [])]+[0])
        #     #add zero to list to prevent errors if the two sequences are Nones
        #     maxflow_fine =   max([np.max(slice) if slice is not None else 0
        #                     for slice in itertools.chain(fine_river_flow_one_sequence if
        #                                                  fine_river_flow_one_sequence is
        #                                                  not None else [],
        #                                                  fine_river_flow_two_sequence if
        #                                                  fine_river_flow_two_sequence is
        #                                                  not None else [])]+[0])
        #     maxflow = max(maxflow,math.ceil(maxflow_fine/self.fine_cutoff_scaling))
        #     #Prevent slider from becoming dominated by very high values in anatartica
        #     #by capping it at 1500
        #     maxflow = min(1000,maxflow)
        self.lake_and_river_plots_required=((not set(self.plot_configuration).isdisjoint(["cflowandlake1",
                                                                               "cflowandlake1"]))
                                             or dynamic_configuration)
        self.lake_and_river_plots_required=False
        self.setup_generator(**kwargs)
        self.step()
        zoom_in_funcs = {PlotScales.NORMAL:self.zoom_in_normal,
                         PlotScales.FINE:self.zoom_in_fine,
                         PlotScales.SUPERFINE:self.zoom_in_super_fine}

    def setup_generator(self,
                        lsmask_sequence,
                        glacier_mask_sequence,
                        catchment_nums_one_sequence,
                        catchment_nums_two_sequence,
                        river_flow_one_sequence,
                        river_flow_two_sequence,
                        river_mouths_one_sequence,
                        river_mouths_two_sequence,
                        lake_volumes_one_sequence,
                        lake_volumes_two_sequence,
                        lake_basin_numbers_one_sequence,
                        lake_basin_numbers_two_sequence,
                        fine_river_flow_one_sequence,
                        fine_river_flow_two_sequence,
                        orography_one_sequence,
                        orography_two_sequence,
                        filled_orography_one_sequence,
                        filled_orography_two_sequence,
                        sinkless_rdirs_one_sequence,
                        sinkless_rdirs_two_sequence,
                        super_fine_orography,
                        first_corrected_orography,
                        second_corrected_orography,
                        third_corrected_orography,
                        fourth_corrected_orography,
                        true_sinks,
                        date_sequence):
        combined_sequence_tuples =  prep_combined_sequence_tuples(lsmask_sequence,
                                                                  glacier_mask_sequence,
                                                                  catchment_nums_one_sequence,
                                                                  catchment_nums_two_sequence,
                                                                  river_flow_one_sequence,
                                                                  river_flow_two_sequence,
                                                                  river_mouths_one_sequence,
                                                                  river_mouths_two_sequence,
                                                                  lake_volumes_one_sequence,
                                                                  lake_volumes_two_sequence,
                                                                  lake_basin_numbers_one_sequence,
                                                                  lake_basin_numbers_two_sequence,
                                                                  fine_river_flow_one_sequence,
                                                                  fine_river_flow_two_sequence,
                                                                  orography_one_sequence,
                                                                  orography_two_sequence,
                                                                  ["{} YBP".format(date) for date
                                                                   in date_sequence],
                                                                   self.lake_and_river_plots_required,
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
                                                               self.zero_slice_only_one,
                                                               return_zero_slice_only_two=
                                                               self.zero_slice_only_two)

    def step(self):
        (self.lsmask_slice_zoomed,self.glacier_mask_slice_zoomed,
         self.matched_catchment_nums_one,self.matched_catchment_nums_two,
         self.river_flow_one_slice_zoomed,self.river_flow_two_slice_zoomed,
         self.lake_volumes_one_slice_zoomed,self.lake_volumes_two_slice_zoomed,
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
        for index,plot in enumerate(self.plot_configuration):
            if plot is not None:
                self.plot_types[plot](index)

    def replot(self,**kwargs):
        self.setup_generator(**kwargs)
        self.next_command_to_send = None
        self.step()

    def set_plot_type(self,plot_index,plot_type):
        self.plot_configuration[plot_index] = plot_type
        self.plot_types[plot_type](plot_index)

    def step_back(self):
        self.next_command_to_send = True
        self.step()

    def step_forward(self):
        self.next_command_to_send = False
        self.step()

    def step_to_date(self,event):
        self.next_command_to_send = self.date_sequence.index(int(event))
        self.step()

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
        self.lake_volume_plot_base(index,self.lake_volumes_one_slice_zoomed)

    def lake_volume_plot_two(self,index):
        self.lake_volume_plot_base(index,self.lake_volumes_two_slice_zoomed)

    def log_lake_volume_plot_one(self,index):
        self.lake_volume_plot_base(index,self.lake_volumes_one_slice_zoomed,True)

    def log_lake_volume_plot_two(self,index):
        self.lake_volume_plot_base(index,self.lake_volumes_two_slice_zoomed,True)

    def lake_volume_comp_plot(self,index):
        self.lake_volume_plot_base(index,self.lake_volumes_one_slice_zoomed -
                                            self.lake_volumes_two_slice_zoomed)

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

    # def reset_zoom(self,event):
    #     self.zoom_settings.zoomed = self.original_zoom_settings.zoomed
    #     self.zoom_settings.zoomed_section_bounds = dict(self.original_zoom_settings.zoomed_section_bounds)
    #     self.next_command_to_send = "zoom"
    #     self.replot_required = True
    #     self.step()
    #     self.replot_required = False

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

    def change_height_range(self,new_min,new_max):
        if new_min > new_max:
            new_min = new_max
        self.orog_min=new_min
        self.orog_max=new_max
        for index,plot in enumerate(self.plot_configuration):
            if plot in self.orog_plot_types:
                self.replot_required = True
                self.plot_types[plot](index)
                self.replot_required = False

    def set_format_coord(self,ax,scale):
        scale_factor = {PlotScales.NORMAL:1,
                        PlotScales.FINE:self.zoom_settings.fine_scale_factor,
                        PlotScales.SUPERFINE:self.zoom_settings.super_fine_scale_factor}[scale]
        ax.format_coord = \
            pts.OrogCoordFormatter(xoffset=
                                   self.zoom_settings.zoomed_section_bounds["min_lon"]*scale_factor,
                                   yoffset=
                                   self.zoom_settings.zoomed_section_bounds["min_lat"]*scale_factor,
                                   add_latlon=True,
                                   scale_factor=scale_factor)

    def get_current_date(self):
        return self.date_text

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
                                                  lake_volumes_sequence,
                                                  glacier_mask_sequence,
                                                  landsea_mask_sequence,
                                                  minflowcutoff):
    col_codes_lake_and_river_sequence = []
    for cumulative_flow,lake_volumes,glacier_mask,landsea_mask in zip(cumulative_flow_sequence,
                                                                     lake_volumes_sequence,
                                                                      glacier_mask_sequence,
                                                                      landsea_mask_sequence):
        col_codes_lake_and_river_sequence.\
            append(generate_colour_codes_lake_and_river(cumulative_flow,
                                                        lake_volumes,
                                                        glacier_mask,
                                                        landsea_mask,
                                                        minflowcutoff))
    return col_codes_lake_and_river_sequence

def generate_colour_codes_lake_and_river(cumulative_flow,
                                         lake_volumes,
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
    rivers_and_lakes_fine[lake_volumes > 0] = 3
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
                                  lake_volumes_one_sequence,
                                  lake_volumes_two_sequence,
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
           lake_volumes_one_sequence is not None):
            lake_and_river_colour_codes_sequence_one = \
                generate_colour_codes_lake_and_river_sequence(river_flow_one_sequence,
                                                              lake_volumes_one_sequence,
                                                              glacier_mask_sequence,
                                                              lsmask_sequence,
                                                              minflowcutoff)
        if (river_flow_two_sequence is not None and
            lake_volumes_two_sequence is not None):
            lake_and_river_colour_codes_sequence_two = \
                generate_colour_codes_lake_and_river_sequence(river_flow_two_sequence,
                                                              lake_volumes_two_sequence,
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
                                      change_none_to_list(lake_volumes_one_sequence),
                                      change_none_to_list(lake_volumes_two_sequence),
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
         river_mouths_two_slice,lake_volumes_one_slice,
         lake_volumes_two_slice,lake_basin_numbers_one_slice,
         lake_basin_numbers_two_slice,fine_river_flow_one_slice,
         fine_river_flow_two_slice,orography_one_slice,
         orography_two_slice,lake_and_river_colour_codes_one_slice,
         lake_and_river_colour_codes_two_slice,date_text) = combined_sequence_tuples[i]
        if return_zero_slice_only_one:
            if return_zero_slice_only_two:
                (lsmask_slice,glacier_mask_slice,catchment_nums_one_slice,
                 catchment_nums_two_slice,river_flow_one_slice,
                 river_flow_two_slice,river_mouths_one_slice,
                 river_mouths_two_slice,lake_volumes_one_slice,
                 lake_volumes_two_slice,lake_basin_numbers_one_slice,
                 lake_basin_numbers_two_slice,fine_river_flow_one_slice,
                 fine_river_flow_two_slice,lake_and_river_colour_codes_one_slice,
                 lake_and_river_colour_codes_two_slice,date_text) = combined_sequence_tuples[0]
            else:
                (_,_,catchment_nums_one_slice,
                 _,river_flow_one_slice,
                 _,river_mouths_one_slice,
                 _,lake_volumes_one_slice,
                 _,lake_basin_numbers_one_slice,
                 _,fine_river_flow_one_slice,
                 _,lake_and_river_colour_codes_one_slice,
                 _,date_text) = combined_sequence_tuples[0]
        elif return_zero_slice_only_two:
             (_,_,_,
             catchment_nums_two_slice,_,
             river_flow_two_slice,_,
             river_mouths_two_slice,_,
             lake_volumes_two_slice,_,
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
            lake_volumes_one_slice_zoomed=extract_zoomed_section(lake_volumes_one_slice,
                                                                zoom_settings.zoomed_section_bounds,
                                                                zoom_settings.fine_scale_factor)
            lake_volumes_two_slice_zoomed=extract_zoomed_section(lake_volumes_two_slice,
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
            lake_volumes_one_slice_zoomed=lake_volumes_one_slice
            lake_volumes_two_slice_zoomed=lake_volumes_two_slice
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
                                 lake_volumes_one_slice_zoomed,
                                 lake_volumes_two_slice_zoomed,
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
                    raise RuntimeError("Generator received unknown command")
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
            elif i >= len(combined_sequence_tuples):
                i = len(combined_sequence_tuples) - 1

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
