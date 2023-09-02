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
from plotting_utilities.lake_analysis_tools import LakeTracker
from plotting_utilities.lake_analysis_tools import FlowPathExtractor
from plotting_utilities.lake_analysis_tools import Basins
from Dynamic_HD_Scripts.utilities import utilities
from Dynamic_HD_Scripts.base.iodriver import advanced_field_loader
from Dynamic_HD_Scripts.base.field import Field
from os.path import join, isfile
from enum import Enum

class PlotScales(Enum):
    NORMAL = 1
    FINE = 2
    SUPERFINE = 3

class TimeSlicePlot():

    def __init__(self,ax):
        self.ax = ax
        self.plot = None
        self.scale = PlotScales.NORMAL

class TimeSeriesPlot:

    def __init__(self,ax):
        self.ax = ax

class ZoomSettings():

    fine_scale_factor = 3
    super_fine_scale_factor = 60
    scale_factors = {PlotScales.NORMAL:1,
                     PlotScales.FINE:fine_scale_factor,
                     PlotScales.SUPERFINE:super_fine_scale_factor}

    def __init__(self,zoomed,zoomed_section_bounds):
        self.zoomed = zoomed
        self.zoomed_section_bounds = zoomed_section_bounds
        if not self.zoomed:
            self.zoomed_section_bounds = {"min_lat":0,
                                          "min_lon":0}

    def get_scale_factor(self,plot_scale):
        return self.scale_factors[plot_scale]

    def copy(self):
        return ZoomSettings(self.zoomed,dict(self.zoomed_section_bounds))

    def translate_point_to_zoomed_coords(self,point,scale_factor):
        if point is None:
            return None
        if not self.zoomed:
            return tuple(point)
        #Because of array offset need to subtract min_lat/lon ...
        #Calculation is old_coords - (min_lat_new_coords - min_lat_old_coords)
        #and min_lat_old_coord is 0
        return (point[0] - (self.zoomed_section_bounds["min_lat"]*scale_factor),
                point[1] - (self.zoomed_section_bounds["min_lon"]*scale_factor))

    def translate_jumps_to_zoomed_coords(self,jumps,scale_factor,dim_name):
        if jumps is None:
            return None
        #See comment above on array index
        if not self.zoomed:
            return jumps
        translated_jumps = jumps.copy()
        translated_jumps[jumps >= 0] -= (self.zoomed_section_bounds[f'min_{dim_name}']*scale_factor)
        return translated_jumps

    def calculate_scale_factor_adjustment(self,old_plot_scale,new_plot_scale):
        if old_plot_scale == new_plot_scale:
            return 1
        else:
            return self.scale_factors[new_plot_scale]/self.scale_factors[old_plot_scale]

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
        self.rdirs_one_sequence = []
        self.rdirs_two_sequence = []
        self.river_flow_one_sequence= []
        self.river_mouths_one_sequence = []
        if not "lake_volumes_one" in missing_fields:
            self.lake_volumes_one_sequence = []
        else:
            self.lake_volumes_one_sequence = None
        if not "lake_basin_numbers_one" in missing_fields:
            self.lake_basin_numbers_one_sequence = []
            self.connected_lake_basin_numbers_one_sequence = []
        else:
            self.lake_basin_numbers_one_sequence = None
            self.connected_lake_basin_numbers_one_sequence = None
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
            self.connected_lake_basin_numbers_two_sequence = []
        else:
            self.lake_basin_numbers_two_sequence = None
            self.connected_lake_basin_numbers_two_sequence = None
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
            self.sinkless_rdirs_two_sequence = Non
        if not "rdirs_jump_next_cell_indices_one" in missing_fields:
            self.rdirs_jump_next_cell_lat_one_sequence = []
            self.rdirs_jump_next_cell_lon_one_sequence = []
            self.coarse_lake_outflows_one_sequence = []
        else:
            self.rdirs_jump_next_cell_lat_one_sequence = None
            self.rdirs_jump_next_cell_lon_one_sequence = None
            self.coarse_lake_outflows_one_sequence = None
        if not "rdirs_jump_next_cell_indices_two" in missing_fields:
            self.rdirs_jump_next_cell_lat_two_sequence = []
            self.rdirs_jump_next_cell_lon_two_sequence = []
            self.coarse_lake_outflows_two_sequence = []
        else:
            self.rdirs_jump_next_cell_lat_two_sequence = None
            self.rdirs_jump_next_cell_lon_two_sequence = None
            self.coarse_lake_outflows_two_sequence = None
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
            rdirs_one = advanced_field_loader(filename=join(sequence_one_results_base_dir,"30min_rdirs.nc"),
                                          time_slice=None,
                                          field_type="RiverDirections",
                                          fieldname="rdirs",
                                          adjust_orientation=True)
            self.rdirs_one_sequence.append(rdirs_one.get_data())
            rdirs_two = advanced_field_loader(filename=join(sequence_two_results_base_dir,"30min_rdirs.nc"),
                                          time_slice=None,
                                          field_type="RiverDirections",
                                          fieldname="rdirs",
                                          adjust_orientation=True)
            self.rdirs_two_sequence.append(rdirs_two.get_data())
            lsmask_data = rdirs_one.get_lsmask()
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
                                                                           "10min_basin_catchment_numbers.nc"),
                                                                           time_slice=None,
                                                                           fieldname="basin_catchment_numbers",
                                                                           adjust_orientation=True)
                self.lake_basin_numbers_one_sequence.append(lake_basin_numbers_one.get_data())
                self.connected_lake_basin_numbers_one_sequence.append(LakeTracker.\
                    number_lakes(lake_basin_numbers_one.get_data() > 0))
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
                                                                         "10min_basin_catchment_numbers.nc"),
                                                                         time_slice=None,
                                                                         fieldname="basin_catchment_numbers",
                                                                         adjust_orientation=True)
                self.lake_basin_numbers_two_sequence.append(lake_basin_numbers_two.get_data())
                self.connected_lake_basin_numbers_two_sequence.append(LakeTracker.\
                    number_lakes(lake_basin_numbers_two.get_data() > 0))
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
            if not "rdirs_jump_next_cell_indices_one" in missing_fields:
                rdirs_jump_next_cell_lat_one = \
                    advanced_field_loader(filename=join(sequence_one_results_base_dir,
                                                        "30min_rdirs_jump_next_cell_indices.nc"),
                                                        time_slice=None,
                                                        fieldname="rdirs_jump_lat",
                                                        adjust_orientation=True)
                rdirs_jump_next_cell_lat_one.change_dtype(np.int64)
                self.rdirs_jump_next_cell_lat_one_sequence.\
                    append(rdirs_jump_next_cell_lat_one.get_data())
                rdirs_jump_next_cell_lon_one = \
                    advanced_field_loader(filename=join(sequence_one_results_base_dir,
                                                        "30min_rdirs_jump_next_cell_indices.nc"),
                                                        time_slice=None,
                                                        fieldname="rdirs_jump_lon",
                                                        adjust_orientation=True)
                rdirs_jump_next_cell_lon_one.change_dtype(np.int64)
                self.rdirs_jump_next_cell_lon_one_sequence.\
                append(rdirs_jump_next_cell_lon_one.get_data())
                coarse_lake_outflows_one = \
                    advanced_field_loader(filename=join(sequence_one_results_base_dir,
                                                        "30min_rdirs_jump_next_cell_indices.nc"),
                                                        time_slice=None,
                                                        fieldname="outflow_points",
                                                        adjust_orientation=True)
                coarse_lake_outflows_one.change_dtype(bool)
                self.coarse_lake_outflows_one_sequence.\
                append(coarse_lake_outflows_one.get_data())
            if not "rdirs_jump_next_cell_indices_two" in missing_fields:
                rdirs_jump_next_cell_lat_two = \
                    advanced_field_loader(filename=join(sequence_two_results_base_dir,
                                                        "30min_rdirs_jump_next_cell_indices.nc"),
                                                        time_slice=None,
                                                        fieldname="rdirs_jump_lat",
                                                        adjust_orientation=True)
                rdirs_jump_next_cell_lat_two.change_dtype(np.int64)
                self.rdirs_jump_next_cell_lat_two_sequence.\
                    append(rdirs_jump_next_cell_lat_two.get_data())
                rdirs_jump_next_cell_lon_two = \
                    advanced_field_loader(filename=join(sequence_two_results_base_dir,
                                                        "30min_rdirs_jump_next_cell_indices.nc"),
                                                        time_slice=None,
                                                        fieldname="rdirs_jump_lon",
                                                        adjust_orientation=True)
                rdirs_jump_next_cell_lon_two.change_dtype(np.int64)
                self.rdirs_jump_next_cell_lon_two_sequence.\
                    append(rdirs_jump_next_cell_lon_two.get_data())
                coarse_lake_outflows_two = \
                    advanced_field_loader(filename=join(sequence_two_results_base_dir,
                                                        "30min_rdirs_jump_next_cell_indices.nc"),
                                                        time_slice=None,
                                                        fieldname="outflow_points",
                                                        adjust_orientation=True)
                coarse_lake_outflows_two.change_dtype(bool)
                self.coarse_lake_outflows_two_sequence.\
                append(coarse_lake_outflows_two.get_data())
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
        if int(event) in self.date_sequence:
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
        self.spillway_plot_axes[index].set_title("Spillway Profile")
        self.spillway_plot_axes[index].set_ylabel("Spillway Height (m)")

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
        if index == 0:
            self.timeseries_plots[index].ax.set_title("Lake Height Evolution")
        else:
            self.timeseries_plots[index].ax.text(0.05,0.05,"Lake Height Evolution",
                                                 transform=
                                                 self.timeseries_plots[index].ax.transAxes)

    def lake_height_plot_one(self,index):
        self.lake_height_plot_base(index,self.lake_heights_one_sequence)

    def lake_height_plot_two(self,index):
        self.lake_height_plot_base(index,self.lake_heights_two_sequence)

    def lake_volume_plot_base(self,index,lake_volume_sequence):
        self.timeseries_plots[index].ax.clear()
        self.timeseries_plots[index].ax.plot(lake_volume_sequence)
        if index == 0:
            self.timeseries_plots[index].ax.set_title("Lake Volume Evolution")
        else:
            self.timeseries_plots[index].ax.text(0.05,0.85,"Lake Volume Evolution",
                                                 transform=
                                                 self.timeseries_plots[index].ax.transAxes)

    def lake_volume_plot_one(self,index):
        self.lake_volume_plot_base(index,self.lake_volume_one_sequence)

    def lake_volume_plot_two(self,index):
        self.lake_volume_plot_base(index,self.lake_volume_two_sequence)

    def lake_outflow_plot_base(self,index,lake_outflow_sequence):
        self.timeseries_plots[index].ax.clear()
        basins = {Basins.CAR:"Carib.",Basins.ART:"Artic",Basins.NATL:"N. Atl."}
        bars = { name:[] for name in basins.keys() }
        current_basin=None
        for i,item in enumerate(lake_outflow_sequence):
            for basin in basins:
                if item == basin:
                    if current_basin == basin:
                        bar_section_length += 1
                    else:
                        if current_basin is not None:
                            bars[current_basin].append(tuple([bar_section_start,
                                                              bar_section_length]))
                        current_basin = basin
                        bar_section_start = i
                        bar_section_length = 1
        if current_basin is not None:
            bars[current_basin].append(tuple([bar_section_start,
                                              bar_section_length]))
        self.timeseries_plots[index].ax.set_ylim(0,30)
        self.timeseries_plots[index].ax.set_xlim(0,len(lake_outflow_sequence))
        self.timeseries_plots[index].ax.broken_barh(bars[Basins.NATL],(20,9),facecolor="tab:green")
        self.timeseries_plots[index].ax.broken_barh(bars[Basins.ART],(10,9),facecolor="tab:blue")
        self.timeseries_plots[index].ax.broken_barh(bars[Basins.CAR],(0,9),facecolor="tab:red")
        ypos = (4.5,15.5,24.5)
        self.timeseries_plots[index].ax.set_yticks(ypos,tuple(basins.values()))
        self.timeseries_plots[index].ax.set_xticks(range(len(lake_outflow_sequence))[0::10],
                                                   labels=self.date_sequence[0::10])

        if index == 0:
            self.timeseries_plots[index].ax.set_title("Lake Agassiz Outflow Ocean")

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
                 lake_points_one=None,
                 lake_points_two=None,
                 corrections=None,
                 **kwargs):
        mpl.use('TkAgg')
        self.plot_types = {"comp":self.catchment_and_cflow_comp_plot,
                           "none":self.no_plot,
                           "cflow1":self.cflow_plot_one,
                           "cflow2":self.cflow_plot_two,
                           "catch1":self.catchments_plot_one,
                           "catch2":self.catchments_plot_two,
                           #"fcflow1":self.fine_cflow_plot_one,
                           #"fcflow2":self.fine_cflow_plot_two,
                           #"fcflowcomp":self.fine_cflow_comp_plot,
                           "orog1":self.orography_plot_one,
                           "orog2":self.orography_plot_one,
                           "morog1":self.modified_orography_plot_one,
                           "morog2":self.modified_orography_plot_two,
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
                           "cflowandlakeselhigh1":self.cflow_and_lake_plot_sel_highlight_one,
                           "cflowandlakeselhigh2":self.cflow_and_lake_plot_sel_highlight_two,
                           "lakevcomp":self.lake_volume_comp_plot,
                           "lakebasinnums1":self.lake_basin_numbers_plot_one,
                           "lakebasinnums2":self.lake_basin_numbers_plot_two,
                           "sellake1":self.selected_lake_plot_one,
                           "sellake2":self.selected_lake_plot_two,
                           "selflowpath1":self.selected_lake_flowpath_one,
                           "selflowpath2":self.selected_lake_flowpath_two,
                           "selspill1":self.selected_lake_spillway_one,
                           "selspill2":self.selected_lake_spillway_two,
                           "debuglakepoints1":self.debug_lake_points_one}
        self.orog_plot_types  = ["orog1","orog2","orogcomp","sforog",
                                 "morog1","morog2",
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
        self.cmap_comp_hl = mpl.colors.ListedColormap(['darkblue','peru','black','lightblue','white','blue','slateblue'])
        self.bounds_comp_hl = list(range(8))
        self.norm_comp_hl = mpl.colors.BoundaryNorm(self.bounds_comp_hl,self.cmap_comp_hl.N)
        N_catch = 50
        self.cmap_catch = 'jet'
        self.orog_min = -9999.0
        self.orog_max = 9999.0
        self.bounds_catch = np.linspace(0,N_catch,N_catch+1)
        self.norm_catch = mpl.colors.BoundaryNorm(boundaries=self.bounds_catch,
                                                  ncolors=256)
        N_lakeoutline = 3
        self.cmap_lakeoutline = 'jet'
        self.bounds_lakeoutline = np.linspace(0,N_lakeoutline,N_lakeoutline+1)
        self.norm_lakeoutline = mpl.colors.BoundaryNorm(boundaries=self.bounds_lakeoutline,
                                                  ncolors=256)
        self.zoom_settings = ZoomSettings(zoomed=zoomed,
                                          zoomed_section_bounds=zoomed_section_bounds)
        self.zero_slice_only_one = zero_slice_only_one
        self.zero_slice_only_two = zero_slice_only_two
        self.original_zoom_settings = self.zoom_settings.copy()
        self.fine_cutoff_scaling = self.zoom_settings.fine_scale_factor*self.zoom_settings.fine_scale_factor
        self.select_coords = False
        self.corrections_file = None
        self.specify_coords_and_height_callback = None
        self.use_orog_one_for_original_height = True
        self.replot_required = False
        self.corrections = corrections
        self.include_date_itself_in_corrected_slices = True
        self.lake_points_one = lake_points_one
        self.lake_points_two = lake_points_two
        if self.lake_points_one is not None:
            self.lake_spillway_masks_one = len(self.lake_points_one)*[None]
            self.lake_flowpath_masks_one = len(self.lake_points_one)*[None]
        if self.lake_points_two is not None:
            self.lake_spillway_masks_two = len(self.lake_points_two)*[None]
            self.lake_flowpath_masks_two = len(self.lake_points_two)*[None]
        self.time_index = 0
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
        for fig in self.figs:
            fig.canvas.mpl_connect('button_release_event',
                                   lambda event: self.set_coords_and_height(event))

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
        self.setup_generator(**kwargs)
        self.date_sequence = kwargs["date_sequence"]
        self.step()

    def setup_generator(self,
                        filled_orography_one_sequence,
                        filled_orography_two_sequence,
                        super_fine_orography,
                        first_corrected_orography,
                        second_corrected_orography,
                        third_corrected_orography,
                        fourth_corrected_orography,
                        true_sinks,
                        date_sequence,
                        **kwargs):
        combined_sequences =  prep_combined_sequences(["{} YBP".format(date) for date in date_sequence],
                                                      date_sequence,
                                                      self.lake_and_river_plots_required,
                                                      self.minflowcutoff,
                                                      **kwargs)
        self.gen = generate_catchment_and_cflow_sequence_tuple(combined_sequences,
                                                               super_fine_orography=super_fine_orography,
                                                               first_corrected_orography=first_corrected_orography,
                                                               second_corrected_orography=second_corrected_orography,
                                                               third_corrected_orography=third_corrected_orography,
                                                               fourth_corrected_orography=fourth_corrected_orography,
                                                               true_sinks=true_sinks,
                                                               bidirectional=True,
                                                               zoom_settings=self.zoom_settings,
                                                               return_zero_slice_only_one=
                                                               self.zero_slice_only_one,
                                                               return_zero_slice_only_two=
                                                               self.zero_slice_only_two)

    def step(self):
        self.slice_data = self.gen.send(self.next_command_to_send)
        self.date = self.slice_data["date"]
        for index,plot in enumerate(self.plot_configuration):
            if plot is not None:
                if self.timeslice_plots[index].plot is not None:
                    plot_limits = {"ylims":self.timeslice_plots[index].ax.get_ylim(),
                                   "xlims":self.timeslice_plots[index].ax.get_xlim()}
                    old_plot_scale = self.timeslice_plots[index].scale
                else:
                    plot_limits = None
                self.timeslice_plots[index].ax.set_visible(True)
                self.plot_types[plot](index)
                if plot_limits is not None:
                    new_plot_scale = self.timeslice_plots[index].scale
                    scale_factor = self.zoom_settings.\
                                   calculate_scale_factor_adjustment(old_plot_scale,
                                                                     new_plot_scale)
                    self.timeslice_plots[index].ax.set_ylim([ scale_factor*val for val in
                                                              plot_limits["ylims"]])
                    self.timeslice_plots[index].ax.set_xlim([ scale_factor*val for val in
                                                              plot_limits["xlims"]])


    def replot(self,lake_points_one,lake_points_two,**kwargs):
        self.lake_points_one = lake_points_one
        self.lake_points_two = lake_points_two
        if self.lake_points_one is not None:
            self.lake_spillway_masks_one = len(self.lake_points_one)*[None]
            self.lake_flowpath_masks_one = len(self.lake_points_one)*[None]
        if self.lake_points_two is not None:
            self.lake_spillway_masks_two = len(self.lake_points_two)*[None]
            self.lake_flowpath_masks_two = len(self.lake_points_two)*[None]
        self.time_index = 0
        self.setup_generator(**kwargs)
        self.date_sequence = kwargs["date_sequence"]
        self.next_command_to_send = None
        self.step()

    def set_plot_type(self,plot_index,plot_type):
        self.plot_configuration[plot_index] = plot_type
        self.timeslice_plots[plot_index].ax.set_visible(True)
        self.replot_required = True
        self.plot_types[plot_type](plot_index)
        self.replot_required = False

    def step_back(self):
        if self.time_index > 0:
            self.time_index -= 1
        self.next_command_to_send = True
        self.step()

    def step_forward(self):
        if self.time_index < len(self.date_sequence) - 1:
            self.time_index += 1
        self.next_command_to_send = False
        self.step()

    def step_to_date(self,event):
        if int(event) in self.date_sequence:
            step_to_index = self.date_sequence.index(int(event))
            self.time_index = step_to_index
            self.next_command_to_send = step_to_index
            self.step()

    def plot_from_color_codes(self,color_codes,index,
                              cmap=None,norm=None):
        if cmap is None:
            cmap = self.cmap
        if norm is None:
            norm = self.norm
        if not self.timeslice_plots[index].plot or self.replot_required:
            if self.replot_required:
                self.timeslice_plots[index].ax.clear()
            self.timeslice_plots[index].plot = \
                self.timeslice_plots[index].ax.imshow(color_codes,cmap=cmap,
                                                      norm=norm,interpolation="none")
            pts.set_ticks_to_zero(self.timeslice_plots[index].ax)
            self.set_format_coord(self.timeslice_plots[index].ax,
                                  self.timeslice_plots[index].scale)
        else:
            self.timeslice_plots[index].plot.set_data(color_codes)

    def catchment_and_cflow_comp_plot(self,index):
        color_codes = generate_color_codes_comp(self.slice_data["lsmask_slice_zoomed"],
                                                self.slice_data["glacier_mask_slice_zoomed"],
                                                self.slice_data["matched_catchment_nums_one"],
                                                self.slice_data["matched_catchment_nums_two"],
                                                self.slice_data["river_flow_one_slice_zoomed"],
                                                self.slice_data["river_flow_two_slice_zoomed"],
                                                minflowcutoff=
                                                self.minflowcutoff,
                                                use_glacier_mask=
                                                self.use_glacier_mask)
        self.plot_from_color_codes(color_codes,index)

    def cflow_and_lake_plot_one(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.plot_from_color_codes(self.slice_data["lake_and_river_color_codes_one_slice_zoomed"],
                                    index)

    def cflow_and_lake_plot_two(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.plot_from_color_codes(self.slice_data["lake_and_river_color_codes_two_slice_zoomed"],
                                    index)

    def cflow_and_lake_plot_sel_highlight_one(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        color_codes = self.slice_data["lake_and_river_color_codes_one_slice_zoomed"]
        if self.lake_points_one and self.lake_points_one[self.time_index]:
            lake_mask = self.generate_selected_lake_mask(
                            self.lake_points_one,
                            self.slice_data["connected_lake_basin_numbers_one_slice_zoomed"],
                            self.slice_data["coarse_lake_outflows_one_slice_zoomed"])
            color_codes[lake_mask == 1] = 5
            color_codes[lake_mask == 2] = 6
            if self.lake_flowpath_masks_one[self.time_index] is None:
                self.generate_flowpaths(self.lake_points_one,
                                        flowpath_masks=
                                        self.lake_flowpath_masks_one,
                                        rdirs=
                                        self.slice_data["rdirs_one_slice_zoomed"],
                                        rdirs_jumps_lat=
                                        self.slice_data["rdirs_jump_next_cell_lat_one_slice_zoomed"],
                                        rdirs_jumps_lon=
                                        self.slice_data["rdirs_jump_next_cell_lon_one_slice_zoomed"])
            for coords in np.argwhere(self.lake_flowpath_masks_one[self.time_index]):
                fine_coords = [coord*self.zoom_settings.fine_scale_factor
                               for coord in coords]
                color_codes[fine_coords[0]-1:fine_coords[0]+2,
                            fine_coords[1]-1:fine_coords[1]+2] = 7
        self.plot_from_color_codes(color_codes,
                                   index,
                                   cmap=self.cmap_comp_hl,
                                   norm=self.norm_comp_hl)

    def cflow_and_lake_plot_sel_highlight_two(self,index):
        pass

    def no_plot(self,index):
        self.timeslice_plots[index].ax.set_visible(False)

    def catchments_plot_one(self,index):
        self.timeslice_plots[index].scale = PlotScales.NORMAL
        self.catchments_plot_base(index,self.slice_data["matched_catchment_nums_one"])

    def catchments_plot_two(self,index):
        self.timeslice_plots[index].scale = PlotScales.NORMAL
        self.catchments_plot_base(index,self.slice_data["matched_catchment_nums_two"])

    def catchments_plot_base(self,index,catchment_slice):
        if not self.timeslice_plots[index].plot or self.replot_required:
            if self.replot_required:
                self.timeslice_plots[index].ax.clear()
            self.timeslice_plots[index].plot = \
                self.timeslice_plots[index].ax.imshow(catchment_slice,
                                                      cmap=self.cmap_catch,
                                                      norm=self.norm_catch,
                                                      interpolation="none")
            pts.set_ticks_to_zero(self.timeslice_plots[index].ax)
            self.set_format_coord(self.timeslice_plots[index].ax,
                                  self.timeslice_plots[index].scale)
        else:
            self.timeslice_plots[index].plot.set_data(catchment_slice)

    def cflow_plot_one(self,index):
        self.cflow_plot_base(index,True)

    def cflow_plot_two(self,index):
        self.cflow_plot_base(index,False)

    def cflow_plot_base(self,index,use_first_sequence_set):
        color_codes_cflow = np.zeros(self.slice_data["lsmask_slice_zoomed"].shape)
        color_codes_cflow[self.slice_data["lsmask_slice_zoomed"] == 0] = 1
        river_flow_slice = (self.slice_data["river_flow_one_slice_zoomed"]
                            if use_first_sequence_set
                            else self.slice_data["river_flow_two_slice_zoomed"])
        color_codes_cflow[river_flow_slice >= self.minflowcutoff] = 2
        if self.use_glacier_mask:
            color_codes_cflow[self.slice_data["glacier_mask_slice_zoomed"] == 1] = 3
        color_codes_cflow[self.slice_data["lsmask_slice_zoomed"] == 1] = 0
        self.plot_from_color_codes(color_codes_cflow,index)

    def fine_cflow_plot_one(self,index):
        self.fine_cflow_plot_base(index,True)

    def fine_cflow_plot_two(self,index):
        self.fine_cflow_plot_base(index,False)

    def fine_cflow_plot_base(self,index,use_first_sequence_set):
        self.timeslice_plots[index].scale = PlotScales.FINE
        color_codes_cflow = np.zeros(self.slice_data["fine_river_flow_one_slice_zoomed"].shape
                                     if use_first_sequence_set
                                     else self.slice_data["fine_river_flow_two_slice_zoomed"].shape)
        color_codes_cflow[:,:] = 1
        river_flow_slice = (self.slice_data["self.fine_river_flow_one_slice_zoomed"]
                            if use_first_sequence_set
                            else self.slice_data["fine_river_flow_two_slice_zoomed"])
        color_codes_cflow[river_flow_slice >= self.minflowcutoff*self.fine_cutoff_scaling] = 2
        self.plot_from_color_codes(color_codes_cflow,index)

    def fine_cflow_comp_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        color_codes_cflow = np.zeros(self.slice_data["fine_river_flow_one_slice_zoomed"].shape)
        color_codes_cflow[:,:] = 1
        color_codes_cflow[np.logical_and(self.slice_data["fine_river_flow_one_slice_zoomed"]
                                         >= self.minflowcutoff*self.fine_cutoff_scaling,
                                         self.slice_data["fine_river_flow_two_slice_zoomed"]
                                         >= self.minflowcutoff*self.fine_cutoff_scaling)] = 2
        color_codes_cflow[np.logical_and(self.slice_data["fine_river_flow_one_slice_zoomed"]
                                         >= self.minflowcutoff*self.fine_cutoff_scaling,
                                         self.slice_data["fine_river_flow_two_slice_zoomed"]
                                         < self.minflowcutoff*self.fine_cutoff_scaling)] = 0
        color_codes_cflow[np.logical_and(self.slice_data["fine_river_flow_one_slice_zoomed"]
                                         < self.minflowcutoff*self.fine_cutoff_scaling,
                                         self.slice_data["fine_river_flow_two_slice_zoomed"]
                                         >= self.minflowcutoff*self.fine_cutoff_scaling)] = 0
        self.plot_from_color_codes(color_codes_cflow,index)

    def orography_plot_one(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.slice_data["orography_one_slice_zoomed"])

    def orography_plot_two(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.slice_data["orography_two_slice_zoomed"])

    def modified_orography_plot_one(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.modified_orography_plot_base(index,self.slice_data["orography_one_slice_zoomed"])

    def modified_orography_plot_two(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.modified_orography_plot_base(index,self.slice_data["orography_two_slice_zoomed"])

    def orography_comp_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.slice_data["orography_one_slice_zoomed"] -
                                       self.slice_data["self.orography_two_slice_zoomed"])

    def super_fine_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.SUPERFINE
        self.orography_plot_base(index,self.slice_data["super_fine_orography_slice_zoomed"])

    def first_corrected_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.slice_data["first_corrected_orography_slice_zoomed"])

    def second_corrected_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.slice_data["second_corrected_orography_slice_zoomed"])

    def third_corrected_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.slice_data["third_corrected_orography_slice_zoomed"])

    def fourth_corrected_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.slice_data["fourth_corrected_orography_slice_zoomed"])

    def first_vs_second_corrected_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.slice_data["second_corrected_orography_slice_zoomed"] -
                                       self.slice_data["first_corrected_orography_slice_zoomed"])

    def second_vs_third_corrected_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.slice_data["third_corrected_orography_slice_zoomed"] -
                                       self.slice_data["second_corrected_orography_slice_zoomed"])

    def third_vs_fourth_corrected_orography_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.orography_plot_base(index,self.slice_data["fourth_corrected_orography_slice_zoomed"] -
                                       self.slice_data["third_corrected_orography_slice_zoomed"])

    def true_sinks_plot(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.plot_from_color_codes(self.slice_data["true_sinks_slice_zoomed"],index)

    def modified_orography_plot_base(self,index,orography):
        modified_orography = np.copy(orography)
        for correction in self.corrections:
            if (self.date > correction["date"] or
                (self.include_date_itself_in_corrected_slices and
                 self.date == correction["date"])):
                modified_orography[correction["lat"],correction["lon"]] = \
                    (orography[correction["lat"],correction["lon"]] +
                     correction["height_change"])
        self.orography_plot_base(index,modified_orography)


    def orography_plot_base(self,index,orography):
        if not self.timeslice_plots[index].plot or self.replot_required:
            if self.replot_required:
                self.timeslice_plots[index].ax.clear()
            self.timeslice_plots[index].plot = \
                self.timeslice_plots[index].ax.imshow(orography,vmin=self.orog_min,
                                                      vmax=self.orog_max,
                                                      interpolation="none")
            pts.set_ticks_to_zero(self.timeslice_plots[index].ax)
            self.set_format_coord(self.timeslice_plots[index].ax,
                                  self.timeslice_plots[index].scale)
        elif self.timeslice_plots[index].scale == PlotScales.SUPERFINE:
            pass
        else:
            self.timeslice_plots[index].plot.set_data(orography)

    def lake_volume_plot_one(self,index):
        self.lake_volume_plot_base(index,self.slice_data["lake_volumes_one_slice_zoomed"])

    def lake_volume_plot_two(self,index):
        self.lake_volume_plot_base(index,self.slice_data["lake_volumes_two_slice_zoomed"])

    def log_lake_volume_plot_one(self,index):
        self.lake_volume_plot_base(index,self.slice_data["lake_volumes_one_slice_zoomed"],True)

    def log_lake_volume_plot_two(self,index):
        self.lake_volume_plot_base(index,self.slice_data["lake_volumes_two_slice_zoomed"],True)

    def lake_volume_comp_plot(self,index):
        self.lake_volume_plot_base(index,self.slice_data["lake_volumes_one_slice_zoomed"] -
                                         self.slice_data["lake_volumes_two_slice_zoomed"])

    def lake_volume_plot_base(self,index,lake_volumes,use_log_scale=False):
        self.timeslice_plots[index].scale = PlotScales.FINE
        if (not self.timeslice_plots[index].plot or
            self.replot_required or np.all(lake_volumes<=0)):
            self.timeslice_plots[index].ax.clear()
            if use_log_scale and not np.all(lake_volumes<=0):
                self.timeslice_plots[index].plot = \
                    self.timeslice_plots[index].ax.imshow(lake_volumes,
                                                          norm=LogNorm(clip=True),
                                                          interpolation="none")
            else:
                self.timeslice_plots[index].plot = \
                    self.timeslice_plots[index].ax.imshow(lake_volumes,interpolation="none")
            pts.set_ticks_to_zero(self.timeslice_plots[index].ax)
            self.set_format_coord(self.timeslice_plots[index].ax,
                                  self.timeslice_plots[index].scale)
        else:
            self.timeslice_plots[index].plot.set_data(lake_volumes)

    def lake_basin_numbers_plot_one(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.catchments_plot_base(index,self.slice_data["lake_basin_numbers_one_slice_zoomed"])

    def lake_basin_numbers_plot_two(self,index):
        self.timeslice_plots[index].scale = PlotScales.FINE
        self.catchments_plot_base(index,self.slice_data["lake_basin_numbers_two_slice_zoomed"])

    def generate_selected_lake_mask(self,lake_points,
                                    connected_lake_basin_numbers,
                                    coarse_lake_outflows):
        lake_number = connected_lake_basin_numbers\
                [self.zoom_settings.\
                 translate_point_to_zoomed_coords(
                    lake_points[self.time_index],
                    self.zoom_settings.fine_scale_factor)]
        lake_mask = np.array(connected_lake_basin_numbers ==
                             lake_number,dtype=np.int32)
        for coords in np.argwhere(coarse_lake_outflows):
            fine_coords = [coord*self.zoom_settings.fine_scale_factor
                                   for coord in coords]
            #Assuming scale factor is 3
            lake_mask[tuple(fine_coords)] = (2 if
                (np.any(lake_mask[fine_coords[0]-1:fine_coords[0]+2,
                 fine_coords[1]-1:fine_coords[1]+2] == 1)) else 0)
        return lake_mask

    def selected_lake_plot_base(self,index,lake_points,
                                connected_lake_basin_numbers,
                                coarse_lake_outflows):
        if (lake_points is not None and lake_points[self.time_index] is not None):
            self.timeslice_plots[index].scale = PlotScales.FINE
            lake_mask = self.generate_selected_lake_mask(lake_points,
                                                         connected_lake_basin_numbers,
                                                         coarse_lake_outflows)
            self.plot_from_color_codes(lake_mask,index,
                                       cmap=self.cmap_lakeoutline,
                                       norm=self.norm_lakeoutline)
        else:
            self.timeslice_plots[index].ax.set_visible(False)

    def selected_lake_plot_one(self,index):
        if self.lake_points_one is not None:
            self.selected_lake_plot_base(index,
                                         lake_points=
                                         self.lake_points_one,
                                         connected_lake_basin_numbers=
                                         self.slice_data["connected_lake_basin_numbers_one_slice_zoomed"],
                                         coarse_lake_outflows=
                                         self.slice_data["coarse_lake_outflows_one_slice_zoomed"])

    def selected_lake_plot_two(self,index):
        if self.lake_points_two is not None:
            self.selected_lake_plot_base(index,
                                         lake_points=
                                         self.lake_points_two,
                                         connected_lake_basin_numbers=
                                         self.slice_data["connected_lake_basin_numbers_two_slice_zoomed"],
                                         coarse_lake_outflows=
                                         self.slice_data["coarse_lake_outflows_two_slice_zoomed"])


    def generate_flowpaths(self,lake_points,flowpath_masks,rdirs,
                           rdirs_jumps_lat,rdirs_jumps_lon):
        lake_point_coarse = [round(coord/self.zoom_settings.fine_scale_factor)
                             for coord in lake_points[self.time_index]]
        flowpath_masks[self.time_index] = \
            FlowPathExtractor.extract_flowpath(lake_center=
                                               self.zoom_settings.\
                                               translate_point_to_zoomed_coords(
                                                lake_point_coarse,1),
                                               rdirs=rdirs,
                                               rdirs_jumps_lat=
                                               self.zoom_settings.\
                                               translate_jumps_to_zoomed_coords(
                                               rdirs_jumps_lat,1,"lat"),
                                               rdirs_jumps_lon=
                                               self.zoom_settings.\
                                               translate_jumps_to_zoomed_coords(
                                               rdirs_jumps_lon,1,"lon"))

    def selected_lake_flowpath_base(self,index,lake_points,flowpath_masks,rdirs,
                                    rdirs_jumps_lat,rdirs_jumps_lon):
        if lake_points is not None:
            self.timeslice_plots[index].scale = PlotScales.FINE
            self.timeslice_plots[index].ax.clear()
            if lake_points[self.time_index] is not None:
                if flowpath_masks[self.time_index] is None:
                    self.generate_flowpaths(lake_points,flowpath_masks,rdirs,
                                            rdirs_jumps_lat,rdirs_jumps_lon)
                self.timeslice_plots[index].plot = \
                    self.timeslice_plots[index].ax.imshow(flowpath_masks[self.time_index],
                                                          interpolation="none")
                pts.set_ticks_to_zero(self.timeslice_plots[index].ax)
            else:
                self.timeslice_plots[index].ax.set_visible(False)
        else:
            self.timeslice_plots[index].ax.set_visible(False)

    def selected_lake_flowpath_one(self,index):
        self.selected_lake_flowpath_base(index,
                                         lake_points=
                                         self.lake_points_one,
                                         flowpath_masks=
                                         self.lake_flowpath_masks_one,
                                         rdirs=
                                         self.slice_data["rdirs_one_slice_zoomed"],
                                         rdirs_jumps_lat=
                                         self.slice_data["rdirs_jump_next_cell_lat_one_slice_zoomed"],
                                         rdirs_jumps_lon=
                                         self.slice_data["rdirs_jump_next_cell_lon_one_slice_zoomed"])

    def selected_lake_flowpath_two(self,index):
        self.selected_lake_flowpath_base(index,
                                         lake_points=
                                         self.lake_points_two,
                                         flowpath_masks=
                                         self.lake_flowpath_masks_two,
                                         rdirs=
                                         self.slice_data["rdirs_two_slice_zoomed"],
                                         rdirs_jumps_lat=
                                         self.slice_data["rdirs_jump_next_cell_lat_two_slice_zoomed"],
                                         rdirs_jumps_lon=
                                         self.slice_data["rdirs_jump_next_cell_lon_two_slice_zoomed"])

    def selected_lake_spillway_base(self,index,lake_points,spillway_masks,sinkless_rdirs):
        if lake_points is not None:
            self.timeslice_plots[index].ax.clear()
            if lake_points[self.time_index] is not None:
                if spillway_masks[self.time_index] is None:
                    spillway_masks[self.time_index] = \
                        SpillwayProfiler.extract_spillway_mask(lake_center=
                                                               self.zoom_settings.\
                                                               translate_point_to_zoomed_coords(
                                                               lake_points[self.time_index],
                                                               self.zoom_settings.fine_scale_factor),
                                                               sinkless_rdirs=
                                                               sinkless_rdirs)
                self.timeslice_plots[index].plot = \
                    self.timeslice_plots[index].ax.imshow(spillway_masks[self.time_index],
                                                          interpolation="none")
                pts.set_ticks_to_zero(self.timeslice_plots[index].ax)
            else:
                self.timeslice_plots[index].ax.set_visible(False)
        else:
            self.timeslice_plots[index].ax.set_visible(False)

    def selected_lake_spillway_one(self,index):
        self.selected_lake_spillway_base(index,
                                         lake_points=
                                         self.lake_points_one,
                                         spillway_masks=
                                         self.lake_spillway_masks_one,
                                         sinkless_rdirs=
                                         self.slice_data["sinkless_rdirs_one_slice_zoomed"])

    def selected_lake_spillway_two(self,index):
        self.selected_lake_spillway_base(index,
                                         lake_points=
                                         self.lake_points_two,
                                         spillway_masks=
                                         self.lake_spillway_masks_two,
                                         sinkless_rdirs=
                                         self.slice_data["sinkless_rdirs_two_slice_zoomed"])

    def debug_lake_points_one(self,index):
        if (self.lake_points_one is not None and
            self.lake_points_one[self.time_index] is not None):
            self.timeslice_plots[index].scale = PlotScales.FINE
            array = self.slice_data["connected_lake_basin_numbers_one_slice_zoomed"].copy()
            array[self.zoom_settings.translate_point_to_zoomed_coords(
                  self.lake_points_one[self.time_index],
                  self.zoom_settings.fine_scale_factor)] = -5
            self.timeslice_plots[index].ax.clear()
            self.timeslice_plots[index].plot = \
                self.timeslice_plots[index].ax.imshow(array,interpolation="none")
            pts.set_ticks_to_zero(self.timeslice_plots[index].ax)
        else:
            self.timeslice_plots[index].ax.set_visible(False)

    def update_minflowcutoff(self,val):
        self.minflowcutoff = val
        self.next_command_to_send = "zoom"
        self.step()

    def set_coords_and_height(self,eclick):
        if eclick.ydata is None or eclick.xdata is None:
            return
        if self.select_coords:
            if self.zoom_settings.zoomed:
                min_lat = self.zoom_settings.zoomed_section_bounds["min_lat"]
                min_lon = self.zoom_settings.zoomed_section_bounds["min_lon"]
            else:
                min_lat = 0
                min_lon = 0
            orography = (self.slice_data["orography_one_slice_zoomed"]
                         if self.use_orog_one_for_original_height else
                         self.slice_data["orography_two_slice_zoomed"])
            self.specify_coords_and_height_callback(lat=
                                                    round(eclick.ydata)+min_lat,
                                                    lon=
                                                    round(eclick.xdata)+min_lon,
                                                    date=self.date,
                                                    original_height=
                                                    orography[round(eclick.ydata),
                                                              round(eclick.xdata)])

    def write_correction(self,new_height):
        lat,lon,date,original_height = self.specify_coords_and_height_callback.get_stored_values()
        height_change = float(new_height)-original_height
        with open(self.corrections_file,'a') as f:
            f.write("{}, {}, {}, {} \n".format(lat,lon,date,height_change))
        self.corrections.append({"lat":lat,"lon":lon,"date":date,"height_change":height_change})
        self.next_command_to_send = "zoom"
        self.replot_required = True
        self.step()
        self.replot_required = False

    def read_corrections(self):
        self.corrections.clear()
        if isfile(self.corrections_file):
            with open(self.corrections_file,'r') as f:
                for line in f:
                    data = line.strip().split(",")
                    self.corrections.append({"lat":int(data[0]),
                                             "lon":int(data[1]),
                                             "date":int(data[2]),
                                             "height_change":float(data[3])})
        self.next_command_to_send = "zoom"
        self.replot_required = True
        self.step()
        self.replot_required = False


    def toggle_include_date_itself_in_corrected_slices(self,value):
        self.include_date_itself_in_corrected_slices = value
        self.next_command_to_send = "zoom"
        self.replot_required = True
        self.step()
        self.replot_required = False

    def toggle_use_orog_one_for_original_height(self,value):
        self.use_orog_one_for_original_height = value

    def toggle_select_coords(self,value):
        self.select_coords = value

    def set_specify_coords_and_height_callback(self,callback):
        self.specify_coords_and_height_callback = callback

    def set_corrections_file(self,filepath):
        self.corrections_file = filepath
        self.read_corrections()

    def match_zoom(self,index):
        zoomed_plot = self.timeslice_plots[index]
        if zoomed_plot.plot is not None:
            scale_factor = self.zoom_settings.get_scale_factor(zoomed_plot.scale)
            min_lat,max_lat = [round(value/scale_factor) for
                               value in zoomed_plot.ax.get_ylim()]
            min_lon,max_lon = [round(value/scale_factor) for
                               value in zoomed_plot.ax.get_xlim()]
        else:
            return
        for plot in self.timeslice_plots:
            if plot.plot is not None and plot != zoomed_plot:
                scale_factor = self.zoom_settings.get_scale_factor(plot.scale)
                plot.ax.set_ylim(min_lat*scale_factor,max_lat*scale_factor)
                plot.ax.set_xlim(min_lon*scale_factor,max_lon*scale_factor)

    def change_height_range(self,new_min,new_max):
        if new_min > new_max:
            new_min = new_max
        self.orog_min=new_min
        self.orog_max=new_max
        for index,plot in enumerate(self.plot_configuration):
            if plot in self.orog_plot_types:
                plot_limits = {"ylims":self.timeslice_plots[index].ax.get_ylim(),
                               "xlims":self.timeslice_plots[index].ax.get_xlim()}
                self.timeslice_plots[index].ax.set_visible(True)
                self.replot_required = True
                self.plot_types[plot](index)
                self.replot_required = False
                self.timeslice_plots[index].ax.set_ylim(plot_limits["ylims"])
                self.timeslice_plots[index].ax.set_xlim(plot_limits["xlims"])

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
        return self.date

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

def generate_color_codes_comp(lsmask,
                              glacier_mask,
                              matched_catchment_nums_one,
                              matched_catchment_nums_two,
                              river_flow_one,
                              river_flow_two,
                              minflowcutoff,
                              use_glacier_mask=True):
    color_codes = np.zeros(lsmask.shape)
    color_codes[lsmask == 0] = 1
    color_codes[np.logical_and(matched_catchment_nums_one != 0,
                               matched_catchment_nums_one ==
                               matched_catchment_nums_two) ] = 5
    color_codes[np.logical_and(np.logical_or(matched_catchment_nums_one != 0,
                                             matched_catchment_nums_two != 0),
                               matched_catchment_nums_one !=
                               matched_catchment_nums_two) ] = 6
    color_codes[np.logical_and(river_flow_one >= minflowcutoff,
                               river_flow_two >= minflowcutoff)] = 2
    color_codes[np.logical_and(river_flow_one >= minflowcutoff,
                               river_flow_two < minflowcutoff)]  = 3
    color_codes[np.logical_and(river_flow_one < minflowcutoff,
                               river_flow_two >= minflowcutoff)] = 4
    if use_glacier_mask:
        color_codes[glacier_mask == 1] = 7
    color_codes[lsmask == 1] = 0
    return color_codes

def generate_color_codes_lake_and_river_sequence(cumulative_flow_sequence,
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
            append(generate_color_codes_lake_and_river(cumulative_flow,
                                                       lake_volumes,
                                                       glacier_mask,
                                                       landsea_mask,
                                                       minflowcutoff))
    return col_codes_lake_and_river_sequence

def generate_color_codes_lake_and_river(cumulative_flow,
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
    color_codes = generate_color_codes_comp(lsmask,
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
    im = plt.imshow(color_codes,cmap=cmap,norm=norm,interpolation="none")
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

def prep_combined_sequences(date_text_sequence,
                            date_sequence,
                            lake_and_river_plots_required=False,
                            minflowcutoff=100,
                            **kwargs):
    lake_and_river_color_codes_sequence_one=[None]
    lake_and_river_color_codes_sequence_two=[None]
    if lake_and_river_plots_required:
        if(kwargs["river_flow_one_sequence"] is not None and
           kwargs["lake_volumes_one_sequence"] is not None):
            lake_and_river_color_codes_sequence_one = \
                generate_color_codes_lake_and_river_sequence(kwargs["river_flow_one_sequence"],
                                                             kwargs["lake_volumes_one_sequence"],
                                                             kwargs["glacier_mask_sequence"],
                                                             kwargs["lsmask_sequence"],
                                                             minflowcutoff)
        if (kwargs["river_flow_two_sequence"] is not None and
            kwargs["lake_volumes_two_sequence"] is not None):
            lake_and_river_color_codes_sequence_two = \
                generate_color_codes_lake_and_river_sequence(kwargs["river_flow_two_sequence"],
                                                             kwargs["lake_volumes_two_sequence"],
                                                             kwargs["glacier_mask_sequence"],
                                                             kwargs["lsmask_sequence"],
                                                             minflowcutoff)
    combined_sequences = {re.sub("_sequence","",name):change_none_to_list(sequence) for name,sequence in kwargs.items()}
    combined_sequences['lake_and_river_color_codes_one'] = lake_and_river_color_codes_sequence_one
    combined_sequences['lake_and_river_color_codes_two'] = lake_and_river_color_codes_sequence_two
    combined_sequences['date_text'] = change_none_to_list(date_text_sequence)
    combined_sequences['date'] = change_none_to_list(date_sequence)
    longest_sequence = 0
    for sequence in combined_sequences.values():
        if len(sequence) > longest_sequence:
            longest_sequence = len(sequence)
    for sequence in combined_sequences.values():
        if len(sequence) < longest_sequence:
            sequence.extend((longest_sequence-len(sequence))*[None])
    return combined_sequences

def generate_catchment_and_cflow_sequence_tuple(combined_sequences,
                                                bidirectional=False,
                                                zoom_settings=None,
                                                return_zero_slice_only_one=False,
                                                return_zero_slice_only_two=False,
                                                **kwargs):
    run_backwards = False
    skip_to_index = False
    i = 0
    sequence_names = ["rdirs_one","rdirs_two",
                      "lsmask","glacier_mask","catchment_nums_one",
                      "catchment_nums_two","river_flow_one",
                      "river_flow_two","river_mouths_one",
                      "river_mouths_two","lake_volumes_one",
                      "lake_volumes_two","lake_basin_numbers_one",
                      "lake_basin_numbers_two","connected_lake_basin_numbers_one",
                      "connected_lake_basin_numbers_two","fine_river_flow_one",
                      "fine_river_flow_two","orography_one","orography_two",
                      "rdirs_jump_next_cell_lat_one",
                      "rdirs_jump_next_cell_lon_one",
                      "rdirs_jump_next_cell_lat_two",
                      "rdirs_jump_next_cell_lon_two",
                      "coarse_lake_outflows_one",
                      "coarse_lake_outflows_two",
                      "sinkless_rdirs_one",
                      "sinkless_rdirs_two",
                      "lake_and_river_color_codes_one",
                      "lake_and_river_color_codes_two"]
    while i < len(combined_sequences) or bidirectional:
        slice_data = {f'{name}_slice':combined_sequences[name][i] for name in sequence_names }
        slice_data["date_text"] = combined_sequences["date_text"][i]
        slice_data["date"] = combined_sequences["date"][i]
        if return_zero_slice_only_one:
            if return_zero_slice_only_two:
                slice_data = {f'{name}_slice':combined_sequences[name][0] for name in sequence_names }
                slice_data["date_text"] = combined_sequences["date_text"][0]
                slice_data["date"] = combined_sequences["date"][0]
            else:
                slice_data = {f'{name}_slice':combined_sequences[name][0] for name in sequence_names if
                                  name.endswith('_one_slice')}
                slice_data["date_text"] = combined_sequences["date_text"][0]
                slice_data["date"] = combined_sequences["date"][0]
        elif return_zero_slice_only_two:
                slice_data = {f'{name}_slice':combined_sequences[name][0] for name in sequence_names if
                                  name.endswith('_two_slice')}
                slice_data["date_text"] = combined_sequences["date_text"][0]
                slice_data["date"] = combined_sequences["date"][0]
        zoomed_slice_data = {}
        slices_to_zoom_normal_scale = ["rdirs_one_slice","rdirs_two_slice",
                                       "lsmask_slice","glacier_mask_slice",
                                       "catchment_nums_one_slice",
                                       "catchment_nums_two_slice",
                                       "river_flow_one_slice",
                                       "river_flow_two_slice",
                                       "river_mouths_one_slice",
                                       "river_mouths_two_slice",
                                       "rdirs_jump_next_cell_lat_one_slice",
                                       "rdirs_jump_next_cell_lon_one_slice",
                                       "rdirs_jump_next_cell_lat_two_slice",
                                       "rdirs_jump_next_cell_lon_two_slice",
                                       "coarse_lake_outflows_one_slice",
                                       "coarse_lake_outflows_two_slice"]
        for slice_name in slices_to_zoom_normal_scale:
            zoomed_slice_data[f"{slice_name}_zoomed"] = \
                extract_zoomed_section(slice_data[slice_name],
                                       zoom_settings.zoomed_section_bounds) \
                if zoom_settings.zoomed else slice_data[slice_name]
        slices_to_zoom_fine_scale = ["lake_volumes_one_slice","lake_volumes_two_slice",
                                     "lake_basin_numbers_one_slice",
                                     "lake_basin_numbers_two_slice",
                                     "connected_lake_basin_numbers_one_slice",
                                     "connected_lake_basin_numbers_two_slice",
                                     "fine_river_flow_one_slice",
                                     "fine_river_flow_two_slice",
                                     "orography_one_slice",
                                     "orography_two_slice",
                                     "sinkless_rdirs_one_slice",
                                     "sinkless_rdirs_two_slice",
                                     "lake_and_river_color_codes_one_slice",
                                     "lake_and_river_color_codes_two_slice"]
        for slice_name in slices_to_zoom_fine_scale:
            zoomed_slice_data[f"{slice_name}_zoomed"] = \
                extract_zoomed_section(slice_data[slice_name],
                                       zoom_settings.zoomed_section_bounds,
                                       zoom_settings.fine_scale_factor) \
                if zoom_settings.zoomed else slice_data[slice_name]
        fixed_fields_to_zoom_fine_scale = ["first_corrected_orography",
                                           "second_corrected_orography",
                                           "third_corrected_orography",
                                           "fourth_corrected_orography",
                                           "true_sinks"]
        for slice_name in fixed_fields_to_zoom_fine_scale:
            zoomed_slice_data[f"{slice_name}_slice_zoomed"] = \
                extract_zoomed_section(kwargs[slice_name],
                                       zoom_settings.zoomed_section_bounds,
                                       zoom_settings.fine_scale_factor) \
                if zoom_settings.zoomed else kwargs[slice_name]
        zoomed_slice_data["super_fine_orography_slice_zoomed"] = \
            extract_zoomed_section(kwargs["super_fine_orography"],
                                   zoom_settings.zoomed_section_bounds,
                                   zoom_settings.super_fine_scale_factor) \
                                   if zoom_settings.zoomed else kwargs["super_fine_orography"]
        zoomed_slice_data["matched_catchment_nums_one"],zoomed_slice_data["matched_catchment_nums_two"] = \
             prepare_matched_catchment_numbers(zoomed_slice_data["catchment_nums_one_slice_zoomed"],
                                               zoomed_slice_data["catchment_nums_two_slice_zoomed"],
                                               zoomed_slice_data["river_mouths_one_slice_zoomed"],
                                               zoomed_slice_data["river_mouths_two_slice_zoomed"])
        zoomed_slice_data["date_text"] = slice_data["date_text"]
        zoomed_slice_data["date"] = slice_data["date"]
        input_from_send = yield zoomed_slice_data
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
            elif i >= len(combined_sequences["lsmask"]):
                i = len(combined_sequences["lsmask"]) - 1

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
    combined_sequences = prep_combined_sequences(lsmask_slice=lsmask_sequence,
                                                 glacier_mask_slice=glacier_mask_sequence,
                                                 catchment_nums_one_slice=catchment_nums_one_sequence,
                                                 catchment_nums_two_slice=catchment_nums_two_sequence,
                                                 river_flow_one_slice=river_flow_one_sequence,
                                                 river_flow_two_slice=river_flow_two_sequence,
                                                 river_mouths_one_slice=river_mouths_one_sequence,
                                                 river_mouths_two_slice=river_mouths_two_sequence,
                                                 date_text_sequence=date_text_sequence,
                                                 date_sequence=None,
                                                 lake_and_river_plots_required=False)
    for slice_data in generate_catchment_and_cflow_sequence_tuple(combined_sequences,
                                                                    ZoomSettings(zoomed,
                                                                                 zoomed_section_bounds)):
        im_list.append([generate_catchment_and_cflow_comp_slice(colors,
                                                                slice_data["lsmask_slice_zoomed"],
                                                                slice_data["glacier_mask_slice_zoomed"],
                                                                slice_data["matched_catchment_nums_one"],
                                                                slice_data["matched_catchment_nums_two"],
                                                                slice_data["river_flow_one_slice_zoomed"],
                                                                slice_data["river_flow_two_slice_zoomed"],
                                                                minflowcutoff=minflowcutoff,
                                                                use_glacier_mask=
                                                                use_glacier_mask),date_text])
    return im_list
