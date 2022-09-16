'''
A module containing a library of methods and classes to generate plots
needed for dynamic HD work. Which plots are created is controlled in
the main function.

Created on Jan 29, 2016

@author: thomasriddick
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
import datetime
import textwrap
import os.path
import math
import copy
from HD_Plots.utilities import plotting_tools as pts
from netCDF4 import Dataset
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.base.iodriver import advanced_field_loader
from Dynamic_HD_Scripts.base import iohelper as iohlpr
from Dynamic_HD_Scripts.base import field
from Dynamic_HD_Scripts.base import grid
from Dynamic_HD_Scripts.utilities import utilities
from HD_Plots.utilities import plotting_tools as pts
from HD_Plots.utilities import match_river_mouths as mtch_rm
from HD_Plots.utilities import river_comparison_plotting_routines as rc_pts
from HD_Plots.utilities import flowmap_plotting_routines as fmp_pts #@UnresolvedImport
from HD_Plots.utilities.interactive_plotting_routines import Interactive_Plots
from HD_Plots.utilities.color_palette import ColorPalette #@UnresolvedImport

global interactive_plots

class Plots(object):
    """A general base class for plots"""

    hd_data_path = '/Users/thomasriddick/Documents/data/HDdata/'
    scratch_dir = '/Users/thomasriddick/Documents/data/temp/'

    def __init__(self,save=False,color_palette_to_use='default'):
        """Class constructor."""
        self.colors = ColorPalette(color_palette_to_use)
        self.save = save

class HDparameterPlots(Plots):

    hdfile_extension = "hdfiles"

    def __init__(self,save=False,color_palette_to_use='default'):
        super(HDparameterPlots,self).__init__(save,color_palette_to_use)
        self.hdfile_path = os.path.join(self.hd_data_path,self.hdfile_extension)

    def flow_parameter_distribution_for_non_lake_cells_for_current_HD_model(self):
        """Calculate the distribution of flow parameter values for the current HD model in non lake cells"""
        current_HD_model_hd_file = os.path.join(self.hdfile_path,"hdpara_file_from_current_model.nc")
        self._flow_parameter_distribution_helper(current_HD_model_hd_file)

    def flow_parameter_distribution_current_HD_model_for_current_HD_model_reprocessed_without_lakes_and_wetlands(self):
        """Calculate the distribution of flow parameter values for the current model reprocessed without lakes/wetlands"""
        reprocessed_current_HD_model_hd_file = os.path.join(self.hdfile_path,"generated",
                                                            "hd_file_regenerate_hd_file_without_lakes_"
                                                            "and_wetlands_20170113_173241.nc")
        self._flow_parameter_distribution_helper(reprocessed_current_HD_model_hd_file)

    def flow_parameter_distribution_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs(self):
        """Calculate the distribution of flow parameter values for the current model reprocessed without lakes/wetlands"""
        hd_file = os.path.join(self.hdfile_path,"generated",
                               "hd_file_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_"
                               "sinks_oceans_lsmask_plus_upscale_rdirs_20170123_165707.nc")
        self._flow_parameter_distribution_helper(hd_file)


    def flow_parameter_distribution_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_no_tuning(self):
        """Calculate the distribution of flow parameter values for the current model reprocessed without lakes/wetlands"""
        hd_file = os.path.join(self.hdfile_path,"generated",
                               "hd_file_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_"
                               "oceans_lsmask_plus_upscale_rdirs_20170112_161226.nc")
        self._flow_parameter_distribution_helper(hd_file)

    def _flow_parameter_distribution_helper(self,hd_file):
        river_flow_k_param = iodriver.load_field(filename=hd_file,
                                                 file_type=iodriver.\
                                                 get_file_extension(hd_file),
                                                 field_type="Generic", unmask=False,
                                                 fieldname="ARF_K", grid_type="HD")
        river_flow_n_param = iodriver.load_field(filename=hd_file,
                                                 file_type=iodriver.\
                                                 get_file_extension(hd_file),
                                                 field_type="Generic", unmask=False,
                                                 fieldname="ARF_N", grid_type="HD")
        river_flow_k_param.mask_field_with_external_mask(river_flow_n_param.get_data() < 5)
        values = river_flow_k_param.get_data()[np.ma.nonzero(river_flow_k_param.get_data())]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        num_bins = 150
        ax.hist(values,num_bins,range=(0.0,1.5))
        ax.set_ylim(0.1,100000)
        plt.yscale('log', nonposy='clip')

class HDOutputPlots(Plots):
    """A class for plotting HD offline (or online?) model output"""

    rdirs_path_extension = 'rdirs'
    jsbach_restart_file_path_extension = 'jsbachrestartfiles'

    def __init__(self,save=False,color_palette_to_use='default'):
        """Class constructor."""
        super(HDOutputPlots,self).__init__(save,color_palette_to_use)
        self.rdirs_data_directory = os.path.join(self.hd_data_path,self.rdirs_path_extension)
        self.upscaled_rdirs_data_directory = os.path.join(self.rdirs_data_directory,
                                                          'generated','upscaled')
        self.jsbach_restart_file_directory = os.path.join(self.hd_data_path,
                                                          self.jsbach_restart_file_path_extension)
        self.generated_jsbach_restart_file_directory = os.path.join(self.jsbach_restart_file_directory,
                                                             'generated')
        self.hdinput_data_directory = os.path.join(self.hd_data_path,'hdinputdata')
        self.cell_areas_data_directory = os.path.join(self.hd_data_path,'gridareasandspacings')
        self.river_discharge_output_data_path = '/Users/thomasriddick/Documents/data/HDoutput'

    def check_water_balance_of_1978_for_constant_forcing_of_0_01(self):
        lsmask = iodriver.load_field("/Users/thomasriddick/Documents/data/HDdata/lsmasks/generated/"
                                     "ls_mask_ten_minute_data_from_virna_0k_ALG4_sinkless"
                                     "_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170123_165707_HD_transf.nc",
                                     ".nc",field_type='Generic',grid_type='HD')
        cell_areas = iodriver.load_field("/Users/thomasriddick/Documents/data/HDdata/"
                                         "gridareasandspacings/hdcellareas.nc",".nc",
                                         field_type='Generic',fieldname="cell_area",grid_type='HD')
        #stage summation to reduce rounding errors
        five_day_discharges = []
        for j in range(73):
            for i in range(j*5,(j+1)*5):
                discharge = iodriver.load_field("/Users/thomasriddick/Documents/data/HDoutput/hd_N01_1978-01-02_hd_"
                                                "discharge_05__ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_"
                                                "sinks_oceans_lsmask_plus_upscale_rdirs_20170123_165707.nc",
                                                ".nc",field_type='Generic',fieldname="disch",timeslice=i,grid_type='HD')
                discharge_times_area = discharge.get_data()*cell_areas.get_data()
                if i == j*5:
                    five_day_discharges.append(np.sum(discharge_times_area,dtype=np.float128))
                else:
                    five_day_discharges[-1] += np.sum(discharge_times_area,dtype=np.float128)
        total_discharge = np.sum(five_day_discharges,dtype=np.float128)
        lsmask_times_area = lsmask.get_data()*cell_areas.get_data()
        change_in_water = self._calculate_total_water_in_restart("/Users/thomasriddick/Documents/data/temp/hdrestart_1978.nc") -\
                          self._calculate_total_water_in_restart("/Users/thomasriddick/Documents/data/temp/hdrestart_1977.nc")
        days_in_year = 365.0
        inflow_per_meter_squared = 0.02
        print("Total water entering HD model: {0}".format(np.sum(lsmask_times_area,dtype=np.float128)*days_in_year*inflow_per_meter_squared))
        print("Total discharge into oceans: {0}".format(total_discharge))
        print("Total change in water in reservoirs: {0}".format(change_in_water))
        print("Total discharge - total inflow: {0}: ".format((total_discharge - days_in_year*\
                                                             np.sum(lsmask_times_area,dtype=np.float128)*inflow_per_meter_squared)))
        print("(Total discharge - total inflow) + change in reservoirs: {0}".format((total_discharge - days_in_year*np.sum(lsmask_times_area,dtype=np.float128)*inflow_per_meter_squared)+ change_in_water))
        print("(Total discharge - total inflow) + change in reservoirs/Change in Reservoirs: {0}".format(((total_discharge - days_in_year*np.sum(lsmask_times_area,dtype=np.float128)*inflow_per_meter_squared)+ change_in_water)/change_in_water))

    def _calculate_total_water_in_restart(self,restart_filename):
        total_water = 0.0
        fgmem_field = iodriver.load_field(restart_filename,
                                          file_type=iodriver.get_file_extension(restart_filename),
                                          field_type="Generic",
                                          unmask=False,
                                          timeslice=None,
                                          fieldname="FGMEM",
                                          grid_type="HD")
        total_water += np.sum(fgmem_field.get_data(),dtype=np.float128)
        finfl_field = iodriver.load_field(restart_filename,
                                          file_type=iodriver.get_file_extension(restart_filename),
                                          field_type="Generic",
                                          unmask=False,
                                          timeslice=None,
                                          fieldname="FINFL",
                                          grid_type="HD")
        total_water += np.sum(finfl_field.get_data(),dtype=np.float128)
        flfmem_field = iodriver.load_field(restart_filename,
                                           file_type=iodriver.get_file_extension(restart_filename),
                                           field_type="Generic",
                                           unmask=False,
                                           timeslice=None,
                                           fieldname="FLFMEM",
                                           grid_type="HD")
        total_water += np.sum(flfmem_field.get_data(),dtype=np.float128)
        frfmem_fields = []
        for i in range(5):
            frfmem_fields.append(iodriver.load_field(restart_filename,
                                                     file_type=iodriver.get_file_extension(restart_filename),
                                                     field_type="Generic",
                                                     unmask=False,
                                                     timeslice=None,
                                                     fieldname="FRFMEM{0}".format(i+1),
                                                     grid_type="HD"))
            total_water += np.sum(frfmem_fields[-1].get_data(),dtype=np.float128)
        return total_water

    def _calculate_discharge_lost_to_changes_in_lsmask(self,lsmask_source_ref_filepath,lsmask_source_data_filepath,
                                                       run_off_filepath,discharge_filepath,
                                                       cell_areas_filepath,num_timeslices,grid_type="HD"):
        if grid_type == "HD":
            rdirs_ref = iodriver.load_field(lsmask_source_ref_filepath,iodriver.get_file_extension(lsmask_source_ref_filepath),
                                            field_type='RiverDirections', unmask=True, grid_type='HD').get_data()
            lsmask_ref = (rdirs_ref <= 0).astype(np.int32)
            rdirs_data = iodriver.load_field(lsmask_source_data_filepath,iodriver.get_file_extension(lsmask_source_data_filepath),
                                             field_type='RiverDirections', unmask=True, grid_type='HD').get_data()
            lsmask_data = (rdirs_data <= 0).astype(np.int32)
        else:
            lsmask_ref = iodriver.load_field(lsmask_source_ref_filepath,iodriver.get_file_extension(lsmask_source_ref_filepath),
                                             field_type='RiverDirections', unmask=True, fieldname='slm',grid_type=grid_type).get_data()
            lsmask_data = iodriver.load_field(lsmask_source_data_filepath,iodriver.get_file_extension(lsmask_source_data_filepath),
                                              field_type='RiverDirections', unmask=True, fieldname='slm',grid_type=grid_type).get_data()
        cell_areas = iodriver.load_field(cell_areas_filepath,iodriver.get_file_extension(cell_areas_filepath),
                                         field_type='Generic',unmask=True, fieldname='cell_area',grid_type=grid_type).get_data()
        lost_discharge = []
        for timeslice in range(num_timeslices):
            run_off_field = iodriver.load_field(run_off_filepath, iodriver.get_file_extension(run_off_filepath),
                                                field_type='Generic', unmask=True, timeslice=timeslice ,fieldname="var501", grid_type=grid_type).get_data()*\
                                                cell_areas
            discharge_field = iodriver.load_field(discharge_filepath,iodriver.get_file_extension(discharge_filepath),
                                                  field_type='Generic', unmask=True, timeslice=timeslice, fieldname="var502", grid_type=grid_type).get_data()*\
                                                  cell_areas
            mask_difference = lsmask_ref - lsmask_data
            run_off_field = run_off_field*mask_difference
            discharge_field = discharge_field*mask_difference
            lost_discharge.append(np.sum(run_off_field,dtype=np.float128) + np.sum(discharge_field,dtype=np.float128))
        return lost_discharge

    def _river_discharge_outflow_comparison_helper(self,ax,river_discharge_output_filepath,
                                                   rdirs_filepath,num_timeslices,lost_discharge=None,
                                                   label=None):
        rdirs = iodriver.load_field(rdirs_filepath,
                                    file_type=\
                                    iodriver.get_file_extension(rdirs_filepath),
                                    field_type='RiverDirections',
                                    unmask=True,
                                    grid_type='HD')
        daily_global_river_discharge_outflow = np.zeros((num_timeslices))
        for i in range(num_timeslices):
            river_discharge = iodriver.load_field(river_discharge_output_filepath,
                                                  file_type=\
                                                  iodriver.get_file_extension(river_discharge_output_filepath),
                                                  field_type='RiverDischarge',
                                                  unmask=True,
                                                  timeslice=i,
                                                  fieldname='friv',
                                                  grid_type='HD')
            river_discharge.set_non_outflow_points_to_zero(rdirs)
            daily_global_river_discharge_outflow[i] =river_discharge.sum_river_outflow()
        if lost_discharge is not None:
            daily_global_river_discharge_outflow += lost_discharge
        total_discharge_info = "Total discharge over year for {0}: {1} \n".format(label,np.sum(daily_global_river_discharge_outflow))
        days = np.linspace(1,365,365)
        ax.plot(days,daily_global_river_discharge_outflow,label=label)
        return total_discharge_info

    def plot_comparison_using_1990_rainfall_data(self):
        ax = plt.subplots(1, 1, figsize=(12, 9))[1]
        plt.ylim(0,7000000)
        plt.xlim(1,365)
        plt.xlabel("Time/days")
        plt.ylabel("Discharge Rate/m^3/s")
        total_discharge_info=""

        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output__ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.upscaled_rdirs_data_directory,
                                                                                             "upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_upscaled_updated_transf.nc"),
                                                                                num_timeslices=365,label="Dynamic Model")
        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output_from_current_model.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.rdirs_data_directory,
                                                                                             "rdirs_from_current_hdparas.nc"),
                                                                                num_timeslices=365,label="Current JSBACH Model")
        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output_from_current_model_after_100_cycles.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.rdirs_data_directory,
                                                                                             "rdirs_from_current_hdparas.nc"),
                                                                                num_timeslices=365,label="Current Model HD Run using 100 cycle spin-up ")
        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output__ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_after_one_years_running.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.upscaled_rdirs_data_directory,
                                                                                             "upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_upscaled_updated_transf.nc"),
                                                                                num_timeslices=365,label="Dynamic HD using 1 cycle spin-up")
        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output__ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170116_235534.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.upscaled_rdirs_data_directory,
                                                                                             "upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_upscaled_updated_transf.nc"),
                                                                                num_timeslices=365,label="Dynamic HD using 1 cycle spin-up as basis")
        days = np.linspace(1,365,365)
        lost_discharge =  self._calculate_discharge_lost_to_changes_in_lsmask(lsmask_source_ref_filepath=\
                                                                              os.path.join(self.jsbach_restart_file_directory,
                                                                                           "jsbach_T106_11tiles_5layers_1976.nc"),
                                                                              lsmask_source_data_filepath=\
                                                                              os.path.join(self.generated_jsbach_restart_file_directory,
                                                                                           "updated_jsbach_T106_11tiles_5layers_1976_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170123_165707.nc"),
                                                                              run_off_filepath=os.path.join(self.hdinput_data_directory,'runoff_T106_1990.nc'),
                                                                              discharge_filepath=os.path.join(self.hdinput_data_directory,'drainage_T106_1990.nc'),
                                                                              cell_areas_filepath=os.path.join(self.cell_areas_data_directory,'T106_grid_cell_areas.nc'),
                                                                              num_timeslices=365,grid_type="T106")
        ax.plot(days,lost_discharge,label="Lost discharge")
        ax.legend()
        print(total_discharge_info)

    def plot_comparison_using_1990_rainfall_data_adding_back_to_discharge(self):
        ax = plt.subplots(1, 1, figsize=(12, 9))[1]
        plt.ylim(0,7000000)
        plt.xlim(1,365)
        plt.xlabel("Time/days")
        plt.ylabel("Discharge Rate/m^3/s")
        total_discharge_info=""

        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output_from_current_model_after_100_cycles.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.rdirs_data_directory,
                                                                                             "rdirs_from_current_hdparas.nc"),
                                                                                num_timeslices=365,label="Current Model HD Run using 100 cycle spin-up")
        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output__ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_after_one_years_running.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.upscaled_rdirs_data_directory,
                                                                                             "upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_upscaled_updated_transf.nc"),
                                                                                num_timeslices=365,label="Dynamic HD using 1 cycle spin-up")
        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output__ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170116_235534.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.upscaled_rdirs_data_directory,
                                                                                             "upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_upscaled_updated_transf.nc"),
                                                                                num_timeslices=365,label="Dynamic HD using 1 cycle spin-up as basis")
        lost_discharge =  self._calculate_discharge_lost_to_changes_in_lsmask(lsmask_source_ref_filepath=\
                                                                              os.path.join(self.jsbach_restart_file_directory,
                                                                                           "jsbach_T106_11tiles_5layers_1976.nc"),
                                                                              lsmask_source_data_filepath=\
                                                                              os.path.join(self.generated_jsbach_restart_file_directory,
                                                                                           "updated_jsbach_T106_11tiles_5layers_1976_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170123_165707.nc"),
                                                                              run_off_filepath=os.path.join(self.hdinput_data_directory,'runoff_T106_1990.nc'),
                                                                              discharge_filepath=os.path.join(self.hdinput_data_directory,'drainage_T106_1990.nc'),
                                                                              cell_areas_filepath=os.path.join(self.cell_areas_data_directory,'T106_grid_cell_areas.nc'),
                                                                              num_timeslices=365,grid_type="T106")
        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output__ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_after_one_years_running.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.upscaled_rdirs_data_directory,
                                                                                             "upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_upscaled_updated_transf.nc"),
                                                                                num_timeslices=365,lost_discharge=lost_discharge,label="Dynamic HD using 1 cycle spin-up + lost discharge")
        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output__ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_after_thirty_years_running.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.upscaled_rdirs_data_directory,
                                                                                             "upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_upscaled_updated_transf.nc"),
                                                                                num_timeslices=365,lost_discharge=lost_discharge,label="Dynamic HD using 30 cycle spin-up + lost discharge")
        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output__ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170116_235534.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.upscaled_rdirs_data_directory,
                                                                                             "upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_upscaled_updated_transf.nc"),
                                                                                num_timeslices=365,lost_discharge=lost_discharge,label="Dynamic HD using 1 cycle spin-up as basis+ lost discharge")
        ax.legend()
        print(total_discharge_info)

class CoupledRunOutputPlots(HDOutputPlots):
    """A class for plotting the output of coupled runs"""

    def __init__(self,save=False,color_palette_to_use="default"):
        """Class constructor."""
        super(CoupledRunOutputPlots,self).__init__(save,color_palette_to_use)

    def ice6g_rdirs_lgm_run_discharge_plot(self):
        """ """
        cell_areas = iodriver.load_field("/Users/thomasriddick/Documents/data/HDdata/"
                                         "gridareasandspacings/hdcellareas.nc",".nc",
                                         field_type='Generic',fieldname="cell_area",
                                         grid_type='HD')
        rdirs = iodriver.load_field(os.path.join(self.rdirs_data_directory,
                                                 "generated","upscaled",
                                                 "upscaled_rdirs_ICE5G_21k_ALG4_sinkless"
                                                 "_no_true_sinks_oceans_lsmask_plus_upscale"
                                                 "_rdirs_tarasov_orog_corrs_generation_and_"
                                                 "upscaling_20170615_174943_upscaled_updated"
                                                 "_transf.nc"),
                                      ".nc",field_type='Generic',grid_type='HD')
        outdata_data = None
        for time in range(120):
            discharge = iodriver.load_field(os.path.join(self.river_discharge_output_data_path,
                                                         "rid0004_hd_higres_mon_79900101_79991231.nc"),
                                            ".nc",field_type='Generic',fieldname="friv",timeslice=time,
                                            grid_type='HD')
            if not outdata_data:
                outflow_data = discharge.get_data()
            else:
                outflow_data = outflow_data + discharge.get_data()
        outflow_data[ rdirs.get_data() != 0 ] = 0.0
        outflow_times_area = outflow_data*cell_areas.get_data()
        plt.figure()
        plt.imshow(outflow_times_area,norm=mpl.colors.LogNorm(),interpolation='none')
        plt.colorbar()

    def extended_present_day_rdirs_lgm_run_discharge_plot(self):
        """ """
        cell_areas = iodriver.load_field("/Users/thomasriddick/Documents/data/HDdata/"
                                         "gridareasandspacings/hdcellareas.nc",".nc",
                                         field_type='Generic',fieldname="cell_area",
                                         grid_type='HD')
        rdirs = iodriver.load_field(os.path.join(self.rdirs_data_directory,
                                                 "rivdir_vs_1_9_data_from_stefan.nc"),
                                    ".nc",field_type='Generic',grid_type='HD')
        outdata_data = None
        for time in range(120):
            discharge = iodriver.load_field(os.path.join(self.river_discharge_output_data_path,
                                                         "rid0003_hd_higres_mon_79900101_79991231.nc"),
                                            ".nc",field_type='Generic',fieldname="friv",timeslice=time,
                                            grid_type='HD')
            if not outdata_data:
                outflow_data = discharge.get_data()
            else:
                outflow_data = outflow_data + discharge.get_data()
        outflow_data[ rdirs.get_data() != 0 ] = 0.0
        outflow_times_area = outflow_data*cell_areas.get_data()
        plt.figure()
        plt.imshow(outflow_times_area,norm=mpl.colors.LogNorm(),interpolation='none')
        plt.colorbar()

    def extended_present_day_rdirs_vs_ice6g_rdirs_lgm_echam(self):
        difference_in_lgm_data_filename=os.path.join(self.river_discharge_output_data_path,
                                                     "rid0004_minus_rid0003_jsbach_jsbach"
                                                     "_mm_last_100_year_mean_times_area.nc")
        lgm_lsmask_file = os.path.join(self.river_discharge_output_data_path,
                                       "rid0003_jsbach_jsbach_tm_7900-7999.nc")
        with Dataset(difference_in_lgm_data_filename,
                     mode='r',format='NETCDF4') as dataset:
            fields = dataset.get_variables_by_attributes(name="var218")
            difference_field = np.asarray(fields[0])[0,:,:]
        with Dataset(lgm_lsmask_file,mode='r',format='NETCDF4'):
            fields = dataset.get_variables_by_attributes(name="land_fract")
            lgm_lsmask_file = np.asarray(fields[0])[0,:,:]
        difference_field_masked = np.ma.array(difference_field,
                                              mask=lgm_lsmask_file)
        plt.figure()
        plt.imshow(difference_field_masked,interpolation='none')
        cb = plt.colorbar()
        cb.set_label(r"River Discharge ($m^{3}s^{-1}$)")
        nlat=48
        nlon=96
        difference_field_total_horizontal_slice =\
            np.mean(difference_field,axis=0)*nlat
        difference_field_total_vertical_slice =\
            np.mean(difference_field,axis=1)*nlon
        xvalues1 = np.linspace(0,360,num=96)
        ax1 = plt.subplots(1, 1, figsize=(12, 9))[1]
        ax1.plot(xvalues1,difference_field_total_horizontal_slice)
        ax1.set_xlabel("Longitude (Degrees East)")
        ax1.set_ylabel(r'River Discharge ($m^{3}s^{-1}$)')
        ax1.set_title("Latitudal Totals")
        ax1.set_xlim(0,360)
        xvalues2 = np.linspace(-90,90,num=48)
        ax2 = plt.subplots(1, 1, figsize=(12, 9))[1]
        ax2.plot(xvalues2,np.flipud(difference_field_total_vertical_slice))
        ax2.set_xlabel("Latitude (Degrees North)")
        ax2.set_ylabel(r'River Discharge ($m^{3}s^{-1}$)')
        ax2.set_title("Longitude Totals")
        ax2.set_xlim(-90,90)
        pacific_unmasked = np.zeros((48,96),dtype=np.bool)
        pacific_unmasked[0:7,:] = True
        pacific_unmasked[:,0:7] = True
        pacific_unmasked[:18,70:] = True
        pacific_unmasked[18:21,72:] = True
        pacific_unmasked[21:,77:] = True
        atlantic_unmasked = np.invert(pacific_unmasked)
        difference_field_pacific_unmasked =\
            np.ma.array(difference_field,mask=pacific_unmasked)
        difference_field_pacific_unmasked.filled(0)
        difference_field_pacific_unmasked_total_vertical_slice =\
            np.mean(difference_field_pacific_unmasked,axis=1)*nlon
        ax3 = plt.subplots(1, 1, figsize=(12, 9))[1]
        ax3.plot(xvalues2,
                 np.flipud(difference_field_pacific_unmasked_total_vertical_slice))
        ax3.set_xlim(-90,90)
        ax3.set_xlabel("Latitude (Degrees North)")
        ax3.set_ylabel(r'River Discharge Difference ($m^{3}s^{-1}$)')
        ax3.set_title("Indo-Pacific")
        difference_field_atlantic_unmasked =\
            np.ma.array(difference_field,mask=atlantic_unmasked)
        difference_field_atlantic_unmasked.filled(0)
        difference_field_atlantic_unmasked_total_vertical_slice =\
            np.mean(difference_field_atlantic_unmasked,axis=1)*nlon
        ax4 = plt.subplots(1, 1, figsize=(12, 9))[1]
        ax4.plot(xvalues2,
                 np.flipud(difference_field_atlantic_unmasked_total_vertical_slice))
        ax4.set_xlim(-90,90)
        ax4.set_xlabel("Latitude (Degrees North)")
        ax4.set_ylabel(r'River Discharge Difference ($m^{3}s^{-1}$)')
        ax4.set_title("Atlantic")
        difference_field_atlantic_unmasked_total_vertical_slice_summed = \
            np.zeros(len(difference_field_atlantic_unmasked_total_vertical_slice))
        it = np.nditer([difference_field_atlantic_unmasked_total_vertical_slice,
                        difference_field_atlantic_unmasked_total_vertical_slice_summed],
                       op_flags=['readwrite'])
        cumulative_sum = 0
        for x,y in it:
            cumulative_sum += x
            y[...] = cumulative_sum
        ax5 = plt.subplots(1, 1, figsize=(12, 9))[1]
        ax5.plot(xvalues2,
                 np.flipud(difference_field_atlantic_unmasked_total_vertical_slice_summed))
        ax5.set_xlim(-90,90)
        ax5.set_xlabel("Latitude (Degrees North)")
        ax5.set_ylabel(r'Cumulative River Discharge Difference ($m^{3}s^{-1}$)')
        ax5.set_title("Integrated Atlantic Discharge Starting from the North Pole")
        difference_field_pacific_unmasked_total_vertical_slice_summed = \
            np.zeros(len(difference_field_pacific_unmasked_total_vertical_slice))
        it = np.nditer([difference_field_pacific_unmasked_total_vertical_slice,
                        difference_field_pacific_unmasked_total_vertical_slice_summed],
                       op_flags=['readwrite'])
        cumulative_sum = 0
        for x,y in it:
            cumulative_sum += x
            y[...] = cumulative_sum
        ax6 = plt.subplots(1, 1, figsize=(12, 9))[1]
        ax6.plot(xvalues2,
                 np.flipud(difference_field_pacific_unmasked_total_vertical_slice_summed))
        ax6.set_xlim(-90,90)
        ax6.set_xlabel("Latitude (Degrees North)")
        ax6.set_ylabel(r'Cumulative River Discharge Difference ($m^{3}s^{-1}$)')
        ax6.set_title("Integrated Indo-Pacific Discharge Starting from the North Pole")

#         difference_field_positive = np.copy(difference_field)
#         difference_field_negative = -1.0*np.copy(difference_field)
#         difference_field_positive[difference_field < 0] = 0
#         difference_field_negative[difference_field > 0] = 0
#         plt.imshow(difference_field_positive,norm=mpl.colors.LogNorm(),interpolation='none') #         plt.imshow(difference_field_negative,norm=mpl.colors.LogNorm(),interpolation='none')
#         plt.colorbar()


    def extended_present_day_rdirs_vs_ice6g_rdirs_lgm_mpiom_pem(self):
        difference_in_lgm_data_filename=os.path.join(self.river_discharge_output_data_path,
                                                     "rid0004_minus_rid0003_mpim_data_2d_mm"
                                                     "_last_100_years_premeaned_non_nan.nc")
        mpiom_lgm_mask_filename=os.path.join(self.river_discharge_output_data_path,
                                             "rid0004landseamask.np")
        with Dataset(difference_in_lgm_data_filename,
                     mode='r',format='NETCDF4') as dataset:
            fields = dataset.get_variables_by_attributes(name="pem")
            difference_field = np.asarray(fields[0])[0,0,:,:]
        with Dataset(mpiom_lgm_mask_filename,
                     mode='r',format='NETCDF4') as dataset:
            fields = dataset.get_variables_by_attributes(name="pem")
            lsmask = np.asarray(fields[0])[0,0,:,:]
        difference_field_masked = np.ma.array(difference_field,mask=lsmask)
        plt.figure()
        plt.imshow(difference_field_masked,interpolation='none')
        cb = plt.colorbar()
        cb.set_label(r"Water Flux Into Ocean $(m^{3}s^{-1})$")

    def ocean_grid_extended_present_day_rdirs_vs_ice6g_rdirs_lgm_run_discharge_plot(self):
        extended_present_day_rdirs_data_filename=os.path.join(self.river_discharge_output_data_path,
                                                              "rid0003_mpiom_data_moc_mm_last_100_years.nc")
        ice6g_rdirs_data_filename=os.path.join(self.river_discharge_output_data_path,
                                               "rid0004_mpiom_data_moc_mm_last_100_years.nc")
        difference_on_ocean_grid_filename=os.path.join(self.river_discharge_output_data_path,
                                                       "rid0004_minus_0003_mpiom_data_moc_mm"
                                                       "_last_100_years.nc")
        with Dataset(extended_present_day_rdirs_data_filename,
                     mode='r',format='NETCDF4') as dataset:
            fields = dataset.get_variables_by_attributes(name="atlantic_wfl")
            atlantic_wfl_ext = np.asarray(fields[0])
            fields = dataset.get_variables_by_attributes(name="indopacific_wfl")
            indopacific_wfl_ext = np.asarray(fields[0])
        with Dataset(ice6g_rdirs_data_filename,
                     mode='r',format='NETCDF4') as dataset:
            fields = dataset.get_variables_by_attributes(name="atlantic_wfl")
            atlantic_wfl_ice6g = np.asarray(fields[0])
            fields = dataset.get_variables_by_attributes(name="indopacific_wfl")
            indopacific_wfl_ice6g = np.asarray(fields[0])
        with Dataset(difference_on_ocean_grid_filename,mode='r',format='NETCDF4') as dataset:
            fields = dataset.get_variables_by_attributes(name="atlantic_wfl")
            atlantic_wfl_diff = np.asarray(fields[0])
            fields = dataset.get_variables_by_attributes(name="indopacific_wfl")
            indopacific_wfl_diff = np.asarray(fields[0])
        x = np.linspace(-90,90,num=180)
        atlantic_wfl_temporalmean_diff = np.mean(atlantic_wfl_diff,axis=0)[0,:,0]
        indopacific_wfl_temporalmean_diff = np.mean(indopacific_wfl_diff,axis=0)[0,:,0]
        atlantic_wfl_temporalmean_ext = np.mean(atlantic_wfl_ext,axis=0)[0,:,0]
        indopacific_wfl_temporalmean_ext = np.mean(indopacific_wfl_ext,axis=0)[0,:,0]
        atlantic_wfl_temporalmean_ice6g = np.mean(atlantic_wfl_ice6g,axis=0)[0,:,0]
        indopacific_wfl_temporalmean_ice6g = np.mean(indopacific_wfl_ice6g,axis=0)[0,:,0]
        atlantic_bins = [-90,-55,-12,0,31,65,90]
        pacific_bins  = [-90,-56,65,90]
        def bin_values(bins,values):
            binned_values = np.zeros((np.size(bins)))
            for i,cell in zip(x,values):
                for j,bin_edge in enumerate(bins):
                    if i > bin_edge:
                        binned_values[j] += cell
                        continue
            return binned_values
        atlantic_bin_wfl_temporalmean_diff = \
            bin_values(atlantic_bins,
                       atlantic_wfl_temporalmean_diff)
        indopacific_bin_wfl_temporalmean_diff = \
            bin_values(pacific_bins,
                       indopacific_wfl_temporalmean_diff)
        ax1 = plt.subplots(1, 1, figsize=(12, 9))[1]
        ax1_step = plt.subplots(1, 1, figsize=(12, 9))[1]
        ax1.plot(x,atlantic_wfl_temporalmean_ext,'.',
                 label='Extended Present Day River Directions')
        ax1.plot(x,atlantic_wfl_temporalmean_ice6g,'.',
                 label='ICE6G River Directions')
        ax1.set_xlabel("Latitude (Degrees North)")
        ax1.set_ylabel("Implied Freshwater Transport ($m^{3}s^{-1}$)")
        ax1.set_xlim(-90,90)
        ax1.set_xticks([-90,-60,-30,0,30,60,90])
        ax1.legend()
        ax1.set_title("Atlantic")
        ax1_step.step(atlantic_bins,
                      atlantic_bin_wfl_temporalmean_diff,
                      where='post',
                      label='Atlantic')
        ax1_step.step(pacific_bins,
                      indopacific_bin_wfl_temporalmean_diff,
                      where='post',
                      label='Indo-Pacific')
        ax1_step.set_xlim(-90,90)
        ax1_step.set_xticks([-90,-60,-30,0,30,60,90])
        ax1_step.set_xlabel("Latitude (Degrees North)")
        ax1_step.set_ylabel("Implied Freshwater Transport ($m^{3}s^{-1}$)")
        ax1_step.legend()
        ax2 = plt.subplots(1, 1, figsize=(12, 9))[1]
        ax2.plot(x,indopacific_wfl_temporalmean_ext,'.',
                 label='Extended Present Day River Directions')
        ax2.plot(x,indopacific_wfl_temporalmean_ice6g,'.',
                 label='ICE6G River Directions')
        ax2.set_xlabel("Latitude (Degrees North)")
        ax2.set_ylabel("Implied Freshwater Transport ($m^{3}s^{-1}$)")
        ax2.set_xlim(-90,90)
        ax2.set_xticks([-90,-60,-30,0,30,60,90])
        ax2.set_title("Indo-Pacific")
        ax2.legend()
        ax3 = plt.subplots(1, 1, figsize=(12, 9))[1]
        ax3.plot(x,atlantic_wfl_temporalmean_diff,'.',
                 label='Atlantic')
        ax3.plot(x,indopacific_wfl_temporalmean_diff,'.',
                 label='Indo-Pacific')
        ax3.set_xlabel("Latitude (Degrees North)")
        ax3.set_ylabel(r'Change in Implied Freshwater Transport ($m^{3}s^{-1}$)')
        ax3.set_xlim(-90,90)
        ax3.set_xticks([-90,-60,-30,0,30,60,90])
        ax3.legend()

class OutflowPlots(Plots):
    """A class for river mouth outflow plots"""

    rmouth_outflow_path_extension = 'rmouthflow'
    flow_maps_path_extension = 'flowmaps'
    rdirs_path_extension = 'rdirs'
    catchments_path_extension = 'catchmentmaps'
    orog_path_extension = 'orographys'
    additional_matches_list_extension = 'addmatches'
    catchment_and_outflows_mods_list_extension = 'catchmods'
    ls_mask_path_extension="lsmasks"

    def __init__(self,save,color_palette_to_use='default'):
        super(OutflowPlots,self).__init__(save,color_palette_to_use)
        self.rmouth_outflow_data_directory = os.path.join(self.hd_data_path,self.rmouth_outflow_path_extension)
        self.flow_maps_data_directory = os.path.join(self.hd_data_path,self.flow_maps_path_extension)
        self.rdirs_data_directory = os.path.join(self.hd_data_path,self.rdirs_path_extension)
        self.catchments_data_directory = os.path.join(self.hd_data_path,self.catchments_path_extension)
        self.orog_data_directory = os.path.join(self.hd_data_path,self.orog_path_extension)
        self.ls_mask_data_directory = os.path.join(self.hd_data_path,self.ls_mask_path_extension)
        self.additional_matches_list_directory = os.path.join(self.hd_data_path,
                                                              self.additional_matches_list_extension)
        self.catchment_and_outflows_mods_list_directory = os.path.join(self.hd_data_path,
                                                                       self.catchment_and_outflows_mods_list_extension)
        self.temp_label = 'temp_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + "_"

    def AdvancedOutFlowComparisonPlotHelpers(self,
                                             reference_rmouth_outflows_filename,
                                             data_rmouth_outflows_filename,
                                             ref_flowmaps_filename,
                                             data_flowmaps_filename,
                                             ref_rdirs_filename,
                                             data_rdirs_filename,
                                             ref_catchment_filename,
                                             data_catchment_filename,
                                             ref_orog_filename,
                                             data_orog_filename,
                                             super_fine_orog_filename=None,
                                             super_fine_data_flowmap_filename=None,
                                             external_ls_mask_filename=None,
                                             additional_matches_list_filename=None,
                                             catchment_and_outflows_mods_list_filename=None,
                                             rivers_to_plot=None,
                                             matching_parameter_set='default',
                                             select_only_rivers_in=None,
                                             allow_new_true_sinks=False,
                                             alternative_catchment_bounds=None,
                                             use_simplified_catchment_colorscheme=False,
                                             use_simplified_flowmap_colorscheme=False,
                                             use_upscaling_labels=False,
                                             swap_ref_and_data_when_finding_labels=False,
                                             super_fine_orog_grid_type='LatLong1min',
                                             grid_type='HD',
                                             super_fine_orog_grid_kwargs={},
                                             **grid_kwargs):
        """Help produce a comparison of two fields of river outflow data"""
        ref_catchment_field = advanced_field_loader(ref_catchment_filename,
                                                    time_slice=None,
                                                    fieldname="catch",
                                                    adjust_orientation=True)
        data_catchment_field = advanced_field_loader(data_catchment_filename,
                                                     time_slice=None,
                                                     fieldname="catch",
                                                     adjust_orientation=True)
        ref_flowtocellfield = advanced_field_loader(ref_flowmaps_filename,
                                                    time_slice=None,
                                                    fieldname="acc",
                                                    adjust_orientation=True)
        data_flowtocellfield = advanced_field_loader(data_flowmaps_filename,
                                                     time_slice=None,
                                                     fieldname="acc",
                                                     adjust_orientation=True)
        ref_rdirs_field = advanced_field_loader(ref_rdirs_filename,
                                                time_slice=None,
                                                fieldname="rdir",
                                                adjust_orientation=True)
        data_rdirs_field = advanced_field_loader(data_rdirs_filename,
                                                 time_slice=None,
                                                 fieldname="rdir",
                                                 adjust_orientation=True)
        ref_orog_field = advanced_field_loader(ref_orog_filename,
                                               time_slice=None,
                                               fieldname="z",
                                               adjust_orientation=True)
        data_orog_field = advanced_field_loader(data_orog_filename,
                                                time_slice=None,
                                                fieldname="z",
                                                adjust_orientation=True)
        super_fine_orog_field = advanced_field_loader(super_fine_orog_filename,
                                                      time_slice=None,
                                                      fieldname="z",
                                                      adjust_orientation=True)

        super_fine_flowtocellfield = None
        if external_ls_mask_filename is not None:
            external_ls_mask = advanced_field_loader(external_ls_mask_filename,
                                                     time_slice=None,
                                                     fieldname="lsmask",
                                                     adjust_orientation=True)
        else:
            external_ls_mask = None
        scale_factor = 1.0
        catchment_grid_changed = False
        temp_file_list = []
        if catchment_and_outflows_mods_list_filename:
            ref_outflow_field = advanced_field_loader(reference_rmouth_outflows_filename,
                                                      time_slice=None,
                                                      fieldname="acc",
                                                      adjust_orientation=True)
            data_outflow_field = advanced_field_loader(data_rmouth_outflows_filename,
                                                       time_slice=None,
                                                       fieldname="acc",
                                                       adjust_orientation=True)
            ref_catchment_field, ref_outflow_field, data_catchment_field, data_outflow_field =\
                rc_pts.modify_catchments_and_outflows(ref_catchments=ref_catchment_field,
                                                      ref_outflows=ref_outflow_field,
                                                      ref_flowmap=ref_flowtocellfield,
                                                      ref_rdirs = ref_rdirs_field,
                                                      data_catchments=data_catchment_field,
                                                      data_outflows=data_outflow_field,
                                                      catchment_and_outflows_modifications_list_filename=\
                                                      catchment_and_outflows_mods_list_filename,
                                                      original_scale_catchment=\
                                                      None,
                                                      original_scale_flowmap=\
                                                      None,
                                                      catchment_grid_changed=False,
                                                      swap_ref_and_data_when_finding_labels=\
                                                      swap_ref_and_data_when_finding_labels,
                                                      original_scale_grid_type=\
                                                      None,
                                                      original_scale_grid_kwargs=\
                                                      None,
                                                      grid_type=grid_type,**grid_kwargs)
            reference_rmouth_outflows_filename=os.path.join(self.scratch_dir,
                                                            self.temp_label + os.path.\
                                                            basename(reference_rmouth_outflows_filename))
            data_rmouth_outflows_filename=os.path.join(self.scratch_dir,
                                                       self.temp_label + os.path.\
                                                       basename(reference_rmouth_outflows_filename))
            temp_file_list.append(reference_rmouth_outflows_filename)
            temp_file_list.append(data_rmouth_outflows_filename)
            iodriver.advanced_field_writer(reference_rmouth_outflows_filename,
                                           ref_outflow_field,
                                           fieldname="acc")
            iodriver.advanced_field_writer(data_rmouth_outflows_filename,
                                           data_outflow_field,
                                           fieldname="acc")
        matchedpairs, unresolved_conflicts  = mtch_rm.advanced_main(reference_rmouth_outflows_filename,
                                                                    data_rmouth_outflows_filename,
                                                                    'acc',"acc",
                                                                    param_set=matching_parameter_set)
        if additional_matches_list_filename:
            additional_matches = mtch_rm.load_additional_manual_matches(additional_matches_list_filename,
                                                                        reference_rmouth_outflows_filename,
                                                                        data_rmouth_outflows_filename,
                                                                        grid_type=grid_type,
                                                                        **grid_kwargs)
            matchedpairs.extend(additional_matches)
        interactive_plots = Interactive_Plots()
        super_fine_grid = grid.makeGrid(super_fine_orog_grid_type,
                                        **super_fine_orog_grid_kwargs)
        ref_grid = grid.makeGrid(grid_type,**grid_kwargs)
        fine_grid = grid.makeGrid(grid_type,**grid_kwargs)
        river_names = []
        for pair in matchedpairs:
            if pair[0].get_lat() > 310*3:
                continue
            if select_only_rivers_in == "North America":
                if(pair[0].get_lat() > 156*3 or pair[0].get_lon() > 260*3):
                    continue
            print("Ref Point: " + str(pair[0]) + "Matches: " + str(pair[1]))
            if rivers_to_plot is not None:
                if not (pair[0].get_lat(),pair[0].get_lon()) in rivers_to_plot:
                    continue
            formatted_river_name = pts.generate_label_for_river(pair,scale_factor=6,
                                                                river_names=river_names)
            fig = plt.figure(formatted_river_name + " - Overview",figsize=(25,12.5))
            fig.suptitle(formatted_river_name)
            ax = plt.subplot(222)
            rc_pts.plot_river_rmouth_flowmap(ax=ax,
                                             ref_flowtocellfield=ref_flowtocellfield.get_data(),
                                             data_flowtocellfield=data_flowtocellfield.get_data(),
                                             rdirs_field=ref_rdirs_field.get_data(),
                                             pair=pair,colors=self.colors,
                                             point_label_coords_scaling=3)
            ax_hist = plt.subplot(221)
            ax_catch = plt.subplot(223)
            catchment_section,catchment_bounds,scale_factor = \
                rc_pts.plot_catchment_and_histogram_for_river(ax_hist=ax_hist,ax_catch=ax_catch,
                                                              ref_catchment_field=ref_catchment_field.get_data(),
                                                              data_catchment_field=data_catchment_field.get_data(),
                                                              data_catchment_field_original_scale=\
                                                              data_catchment_field.get_data(),
                                                              data_original_scale_flowtocellfield=\
                                                              data_flowtocellfield.get_data(),
                                                              rdirs_field=ref_rdirs_field.get_data(),
                                                              data_rdirs_field=data_rdirs_field.get_data(),
                                                              pair=pair,
                                                              catchment_grid_changed=catchment_grid_changed,
                                                              swap_ref_and_data_when_finding_labels=\
                                                              swap_ref_and_data_when_finding_labels,
                                                              colors=self.colors,
                                                              ref_grid=ref_grid,
                                                              grid_type=grid_type,
                                                              alternative_catchment_bounds=\
                                                              alternative_catchment_bounds,
                                                              use_simplified_catchment_colorscheme=\
                                                              use_simplified_catchment_colorscheme,
                                                              use_upscaling_labels=\
                                                              use_upscaling_labels,
                                                              allow_new_sink_points=\
                                                              allow_new_true_sinks,
                                                              external_landsea_mask=\
                                                              external_ls_mask,
                                                              ref_original_scale_flowtocellfield=\
                                                              ref_flowtocellfield,
                                                              ref_catchment_field_original_scale=\
                                                              ref_catchment_field,
                                                              use_original_scale_field_for_determining_data_and_ref_labels=\
                                                              False,
                                                              return_catchment_plotter=\
                                                              False,
                                                              data_original_scale_grid_type=\
                                                              grid_type,
                                                              ref_original_scale_grid_type=\
                                                              grid_type,
                                                              data_original_scale_grid_kwargs=\
                                                              grid_kwargs,
                                                              ref_original_scale_grid_kwargs=\
                                                              grid_kwargs,
                                                              **grid_kwargs)
            ax = plt.subplot(224)
            rc_pts.plot_whole_river_flowmap(ax=ax,pair=pair,
                                            ref_flowtocellfield=ref_flowtocellfield.get_data(),
                                            data_flowtocellfield=data_flowtocellfield.get_data(),
                                            rdirs_field=ref_rdirs_field.get_data(),
                                            data_rdirs_field=data_rdirs_field.get_data(),
                                            catchment_bounds=catchment_bounds,colors=self.colors,
                                            simplified_flowmap_plot=use_simplified_flowmap_colorscheme,
                                            allow_new_sink_points=allow_new_true_sinks)
            if super_fine_orog_filename:
                ref_to_super_fine_scale_factor = \
                    pts.calculate_scale_factor(coarse_grid_type=grid_type,
                                               coarse_grid_kwargs=grid_kwargs,
                                               fine_grid_type=super_fine_orog_grid_type,
                                               fine_grid_kwargs=super_fine_orog_grid_kwargs)
            else:
                ref_to_super_fine_scale_factor=None
            interactive_plots.setup_plots(catchment_section,
                                          ref_orog_field.get_data(),
                                          data_orog_field.get_data(),
                                          ref_flowtocellfield.get_data(),
                                          data_flowtocellfield.get_data(),
                                          ref_rdirs_field.get_data(),
                                          super_fine_orog_field.get_data(),
                                          super_fine_flowtocellfield.get_data()
                                          if super_fine_flowtocellfield is not None else None,
                                          pair, catchment_bounds,
                                          scale_factor,
                                          ref_to_super_fine_scale_factor,
                                          ref_grid_offset_adjustment=ref_grid.\
                                          get_longitude_offset_adjustment(),
                                          fine_grid_offset_adjustment=fine_grid.\
                                          get_longitude_offset_adjustment(),
                                          super_fine_grid_offset_adjustment=super_fine_grid.\
                                          get_longitude_offset_adjustment(),
                                          point_label_coords_scaling=3,
                                          river_name=formatted_river_name)
        print("Unresolved Conflicts: ")
        for conflict in unresolved_conflicts:
            print(" Conflict:")
            for pair in conflict:
                print("  Ref Point" + str(pair[0]) + "Matches" + str(pair[1]))
        for temp_file in temp_file_list:
            if os.path.basename(temp_file).startswith("temp_"):
                print("Deleting File: {0}".format(temp_file))
                os.remove(temp_file)

    def OutFlowComparisonPlotHelpers(self,reference_rmouth_outflows_filename,
                                     data_rmouth_outflows_filename,
                                     ref_flowmaps_filename,data_flowmaps_filename,
                                     rdirs_filename,flip_data_field=False,rotate_data_field=False,
                                     flip_ref_field=False,rotate_ref_field=False,
                                     ref_catchment_filename=None,data_catchment_filename=None,
                                     data_catchment_original_scale_filename=None,
                                     data_rdirs_filename=None,
                                     data_original_scale_flow_map_filename=None,
                                     ref_orog_filename=None,
                                     data_orog_original_scale_filename=None,
                                     flip_orog_original_scale_relative_to_data=False,
                                     super_fine_orog_filename=None,
                                     super_fine_data_flowmap_filename=None,
                                     flip_super_fine_orog=False,
                                     rotate_super_fine_orog=False,
                                     additional_matches_list_filename=None,
                                     catchment_and_outflows_mods_list_filename=None,
                                     plot_simple_catchment_and_flowmap_plots=False,
                                     return_simple_catchment_and_flowmap_plotters=False,
                                     return_catchment_plotters=False,
                                     swap_ref_and_data_when_finding_labels=False,
                                     rivers_to_plot=None,
                                     alternative_catchment_bounds=None,
                                     matching_parameter_set='default',
                                     split_comparison_plots_across_multiple_canvases=False,
                                     use_simplified_catchment_colorscheme=False,
                                     use_simplified_flowmap_colorscheme=False,
                                     use_upscaling_labels=False,
                                     select_only_rivers_in=None,
                                     allow_new_true_sinks=False,
                                     ref_original_scale_flow_map_filename=None,
                                     ref_catchment_original_scale_filename=None,
                                     use_original_scale_field_for_determining_data_and_ref_labels=False,
                                     external_ls_mask_filename=None,
                                     flip_external_ls_mask=False,
                                     rotate_external_ls_mask=False,
                                     ref_original_scale_grid_type='HD',
                                     grid_type='HD',data_original_scale_grid_type='HD',
                                     super_fine_orog_grid_type='HD',
                                     no_data_path_prefixes=False,
                                     data_original_scale_grid_kwargs={},
                                     ref_original_scale_grid_kwargs={},
                                     super_fine_orog_grid_kwargs={},
                                     **grid_kwargs):
        """Help produce a comparison of two fields of river outflow data"""
        ref_flowmaps_filepath = os.path.join(self.flow_maps_data_directory,ref_flowmaps_filename)
        data_flowmaps_filepath = os.path.join(self.flow_maps_data_directory,data_flowmaps_filename)
        rdirs_filepath = os.path.join(self.rdirs_data_directory,rdirs_filename)
        if ref_catchment_filename:
            ref_catchments_filepath = os.path.join(self.catchments_data_directory,
                                                   ref_catchment_filename)
        if data_catchment_filename:
            if no_data_path_prefixes:
                data_catchment_filepath = data_catchment_filename
            else:
                data_catchment_filepath = os.path.join(self.catchments_data_directory,
                                                       data_catchment_filename)
        if data_rdirs_filename:
            if no_data_path_prefixes:
                data_rdirs_filepath =  data_rdirs_filename
            else:
                data_rdirs_filepath =  os.path.join(self.rdirs_data_directory,
                                                    data_rdirs_filename)
        if ref_orog_filename:
            ref_orog_filepath = os.path.join(self.orog_data_directory,
                                             ref_orog_filename)
        if data_orog_original_scale_filename:
            if no_data_path_prefixes:
                data_orog_original_scale_filepath = data_orog_original_scale_filename
            else:
                data_orog_original_scale_filepath = os.path.join(self.orog_data_directory,
                                                                 data_orog_original_scale_filename)
        if data_catchment_original_scale_filename:
            if no_data_path_prefixes:
                data_catchment_original_scale_filepath = data_catchment_original_scale_filename
            else:
                data_catchment_original_scale_filepath = os.path.join(self.catchments_data_directory,
                                                                      data_catchment_original_scale_filename)
        if ref_catchment_original_scale_filename:
            ref_catchment_original_scale_filepath = os.path.join(self.catchments_data_directory,
                                                                 ref_catchment_original_scale_filename)
        if catchment_and_outflows_mods_list_filename:
            catchment_and_outflows_mods_list_filepath = os.path.join(self.catchment_and_outflows_mods_list_directory,
                                                                     catchment_and_outflows_mods_list_filename)
        if additional_matches_list_filename:
            additional_matches_list_filepath = os.path.join(self.additional_matches_list_directory,
                                                            additional_matches_list_filename)
        if external_ls_mask_filename:
            external_ls_mask_filepath = os.path.join(self.ls_mask_data_directory,
                                                     external_ls_mask_filename)
        if super_fine_orog_filename:
            super_fine_orog_filepath = os.path.join(self.orog_data_directory,
                                                    super_fine_orog_filename)
            if super_fine_data_flowmap_filename:
                super_fine_data_flowmap_filepath = os.path.join(self.flow_maps_data_directory,
                                                                super_fine_data_flowmap_filename)
        if ref_catchment_filename:
            ref_catchment_field = iohlpr.NetCDF4FileIOHelper.load_field(ref_catchments_filepath,
                                                                       grid_type,**grid_kwargs)
        if data_catchment_filename:
            data_catchment_field =\
                iohlpr.NetCDF4FileIOHelper.load_field(data_catchment_filepath,
                                                      grid_type,**grid_kwargs)
            if grid_type == data_original_scale_grid_type and grid_kwargs == data_original_scale_grid_kwargs:
                catchment_grid_changed = False
                data_catchment_field_original_scale = data_catchment_field
            else:
                catchment_grid_changed = True
                if data_catchment_original_scale_filepath is None:
                    raise RuntimeError('require original scale catchment to use upscaled catchments')
                data_catchment_field_original_scale =\
                    iohlpr.NetCDF4FileIOHelper.load_field(data_catchment_original_scale_filepath,
                                                          grid_type=data_original_scale_grid_type,
                                                          **data_original_scale_grid_kwargs)
                if data_original_scale_flow_map_filename is None:
                    raise RuntimeError('require original flow to cell data to use upscaled catchments')
                else:
                    data_original_scale_flow_map_filepath = os.path.join(self.flow_maps_data_directory,
                                                                         data_original_scale_flow_map_filename)
                    data_original_scale_flowtocellfield =  iohlpr.NetCDF4FileIOHelper.\
                        load_field(data_original_scale_flow_map_filepath,grid_type=data_original_scale_grid_type,
                                    **data_original_scale_grid_kwargs)
        if use_original_scale_field_for_determining_data_and_ref_labels:
            if ref_original_scale_flow_map_filename is None:
                raise RuntimeError('require original flow to cell field to use upscaled catchments for ref')
            elif ref_catchment_original_scale_filename is None:
                    raise RuntimeError('require original scale catchment to use upscaled catchments for ref')
            else:
                ref_original_scale_flow_map_filepath = os.path.join(self.flow_maps_data_directory,
                                                                    ref_original_scale_flow_map_filename)
                ref_original_scale_flowtocellfield =  iohlpr.NetCDF4FileIOHelper.\
                    load_field(ref_original_scale_flow_map_filepath,grid_type=data_original_scale_grid_type,
                                    **data_original_scale_grid_kwargs)
                ref_catchment_field_original_scale =\
                    iohlpr.NetCDF4FileIOHelper.load_field(ref_catchment_original_scale_filepath,
                                                          grid_type=ref_original_scale_grid_type,
                                                          **ref_original_scale_grid_kwargs)
        else:
            ref_original_scale_flowtocellfield = None
            ref_catchment_field_original_scale = None
        ref_flowtocellfield = iohlpr.NetCDF4FileIOHelper.load_field(ref_flowmaps_filepath,grid_type,**grid_kwargs)
        data_flowtocellfield = iohlpr.NetCDF4FileIOHelper.load_field(data_flowmaps_filepath,grid_type,**grid_kwargs)
        rdirs_field = iohlpr.NetCDF4FileIOHelper.load_field(rdirs_filepath,grid_type,**grid_kwargs)
        ref_grid = grid.makeGrid(grid_type,**grid_kwargs)
        if data_rdirs_filename:
            if catchment_grid_changed:
                data_rdirs_field = iohlpr.NetCDF4FileIOHelper.load_field(data_rdirs_filepath,
                                                                         grid_type=data_original_scale_grid_type,
                                                                         **data_original_scale_grid_kwargs)
                data_rdirs_field = utilities.upscale_field(input_field=field.\
                                                           Field(data_rdirs_field,
                                                                 grid=data_original_scale_grid_type,
                                                                 **data_original_scale_grid_kwargs),
                                                           output_grid_type=grid_type,
                                                           method='CheckValue',
                                                           output_grid_kwargs=grid_kwargs,
                                                           scalenumbers=False).get_data()
            else:
                data_rdirs_field = iohlpr.NetCDF4FileIOHelper.load_field(data_rdirs_filepath,grid_type,
                                                                         **grid_kwargs)
        else:
            data_rdirs_field = None
        if ref_orog_filename:
            ref_orog_field = iohlpr.NetCDF4FileIOHelper.load_field(ref_orog_filepath,grid_type,
                                                                   **grid_kwargs)
            ref_orog_field = np.ma.array(ref_orog_field)

        if data_orog_original_scale_filename:
            data_orog_original_scale_field = iohlpr.NetCDF4FileIOHelper.\
                load_field(data_orog_original_scale_filepath,
                           grid_type=data_original_scale_grid_type,
                           **data_original_scale_grid_kwargs)
            fine_grid = grid.makeGrid(data_original_scale_grid_type,
                                      **data_original_scale_grid_kwargs)
            if flip_orog_original_scale_relative_to_data:
                #This is an extra flip along with the flip applied below
                data_orog_original_scale_field = np.flipud(data_orog_original_scale_field)
        else:
            fine_grid = ref_grid
        if super_fine_orog_filename:
            super_fine_orog_field = iohlpr.NetCDF4FileIOHelper.\
                load_field(super_fine_orog_filepath,
                           grid_type=super_fine_orog_grid_type,
                           **super_fine_orog_grid_kwargs)
            super_fine_grid = grid.makeGrid(super_fine_orog_grid_type,
                                            **super_fine_orog_grid_kwargs)
            if super_fine_data_flowmap_filename:
                super_fine_data_flowmap = iohlpr.NetCDF4FileIOHelper.\
                    load_field(super_fine_data_flowmap_filepath,
                               grid_type=super_fine_orog_grid_type,
                               **super_fine_orog_grid_kwargs)
            else:
                super_fine_data_flowmap = None
            if flip_super_fine_orog:
                super_fine_orog_field = np.flipud(super_fine_orog_field)
                if super_fine_data_flowmap is not None:
                    super_fine_data_flowmap = np.flipud(super_fine_data_flowmap)
            if rotate_super_fine_orog:
                super_fine_orog_field = np.roll(super_fine_orog_field,
                                                np.size(super_fine_orog_field,
                                                        axis=1)//2,
                                                axis=1)
                if super_fine_data_flowmap is not None:
                    super_fine_data_flowmap = np.roll(super_fine_data_flowmap,
                                                      np.size(super_fine_data_flowmap,
                                                              axis=1)//2,
                                                      axis=1)
        else:
            super_fine_orog_field = None
            super_fine_data_flowmap = None
            super_fine_grid = ref_grid
        if external_ls_mask_filename:
            external_ls_mask = iohlpr.NetCDF4FileIOHelper.\
                load_field(external_ls_mask_filepath,
                           grid_type=grid_type,
                           **grid_kwargs)
        else:
            external_ls_mask = None
        if flip_ref_field:
            ref_flowtocellfield = np.flipud(ref_flowtocellfield)
            rdirs_field = np.flipud(rdirs_field)
            if ref_catchment_filename:
                ref_catchment_field = np.flipud(ref_catchment_field)
            if use_original_scale_field_for_determining_data_and_ref_labels:
                    ref_original_scale_flowtocellfield = np.flipud(ref_original_scale_flowtocellfield)
                    ref_catchment_field_original_scale = np.flipud(ref_catchment_field_original_scale)
        if flip_data_field:
            data_flowtocellfield = np.flipud(data_flowtocellfield)
            if data_rdirs_filename:
                data_rdirs_field = np.flipud(data_rdirs_field)
            if data_catchment_filename:
                data_catchment_field = np.flipud(data_catchment_field)
                if catchment_grid_changed:
                    data_original_scale_flowtocellfield = np.flipud(data_original_scale_flowtocellfield)
                    data_catchment_field_original_scale = np.flipud(data_catchment_field_original_scale)
            if data_orog_original_scale_filename:
                data_orog_original_scale_field = np.flipud(data_orog_original_scale_field)
        if rotate_ref_field:
            ref_flowtocellfield = np.roll(ref_flowtocellfield,
                                          np.size(ref_flowtocellfield,axis=1)//2,
                                          axis=1)
            rdirs_field = np.roll(rdirs_field,
                                  np.size(rdirs_field,axis=1)//2,
                                  axis=1)
            if ref_catchment_filename:
                ref_catchment_field = np.roll(ref_catchment_field,
                                              np.size(ref_catchment_field,axis=1)//2,
                                              axis=1)
                if use_original_scale_field_for_determining_data_and_ref_labels:
                    ref_original_scale_flowtocellfield = np.roll(ref_original_scale_flowtocellfield,
                                                                 np.size(ref_original_scale_flowtocellfield,
                                                                         axis=1)//2,
                                                                 axis=1)
                    ref_catchment_field_original_scale = np.roll(ref_catchment_field_original_scale,
                                                                 np.size(ref_catchment_field_original_scale,
                                                                         axis=1)//2,
                                                                 axis=1)
        if rotate_data_field:
            data_flowtocellfield = np.roll(data_flowtocellfield,
                                           np.size(data_flowtocellfield,axis=1)//2,
                                           axis=1)
            if data_rdirs_filename:
                data_rdirs_field = np.roll(data_rdirs_field,
                                           np.size(data_rdirs_field,axis=1)//2,
                                           axis=1)
            if data_catchment_filename:
                data_catchment_field = np.roll(data_catchment_field,
                                              np.size(data_catchment_field,axis=1)//2,
                                              axis=1)
                if catchment_grid_changed:
                    data_original_scale_flowtocellfield = np.roll(data_original_scale_flowtocellfield,
                                                                  np.size(data_original_scale_flowtocellfield,
                                                                          axis=1)//2,
                                                                  axis=1)
                    data_catchment_field_original_scale = np.roll(data_catchment_field_original_scale,
                                                                  np.size(data_catchment_field_original_scale,
                                                                          axis=1)//2,
                                                                  axis=1)
            if data_orog_original_scale_filename:
                data_orog_original_scale_field = np.roll(data_orog_original_scale_field,
                                                         np.size(data_orog_original_scale_field,
                                                                 axis=1)//2,
                                                         axis=1)
            else:
                data_orog_original_scale_field = None
        if flip_external_ls_mask:
            external_ls_mask = np.flipud(external_ls_mask)
        if rotate_external_ls_mask:
            external_ls_mask = np.roll(external_ls_mask,
                                       np.size(external_ls_mask,
                                               axis=1)//2,
                                       axis=1)
        temp_file_list = []
        if catchment_and_outflows_mods_list_filename:
            ref_outflow_field = iodriver.load_field(reference_rmouth_outflows_filename,
                                                    file_type=iodriver.\
                                                    get_file_extension(reference_rmouth_outflows_filename),
                                                    field_type='Generic', grid_type=grid_type,**grid_kwargs)
            data_outflow_field = iodriver.load_field(data_rmouth_outflows_filename,
                                                     file_type=iodriver.\
                                                     get_file_extension(data_rmouth_outflows_filename),
                                                     field_type='Generic', grid_type=grid_type,**grid_kwargs)
            if flip_ref_field:
                ref_outflow_field.flip_data_ud()
            if rotate_ref_field:
                ref_outflow_field.rotate_field_by_a_hundred_and_eighty_degrees()
            if flip_data_field:
                data_outflow_field.flip_data_ud()
            if rotate_data_field:
                data_outflow_field.rotate_field_by_a_hundred_and_eighty_degrees()
            ref_catchment_field, ref_outflow_field, data_catchment_field, data_outflow_field =\
                rc_pts.modify_catchments_and_outflows(ref_catchments=ref_catchment_field,
                                                      ref_outflows=ref_outflow_field,
                                                      ref_flowmap=ref_flowtocellfield,
                                                      ref_rdirs = rdirs_field,
                                                      data_catchments=data_catchment_field,
                                                      data_outflows=data_outflow_field,
                                                      catchment_and_outflows_modifications_list_filename=\
                                                      catchment_and_outflows_mods_list_filepath,
                                                      original_scale_catchment=\
                                                      data_catchment_field_original_scale.get_data(),
                                                      original_scale_flowmap=\
                                                      data_original_scale_flowtocellfield.get_data(),
                                                      catchment_grid_changed=catchment_grid_changed,
                                                      swap_ref_and_data_when_finding_labels=\
                                                      swap_ref_and_data_when_finding_labels,
                                                      original_scale_grid_type=\
                                                      data_original_scale_grid_type,
                                                      original_scale_grid_kwargs=\
                                                      data_original_scale_grid_kwargs,
                                                      grid_type=grid_type,**grid_kwargs)
            if flip_data_field:
                data_outflow_field.flip_data_ud()
            if rotate_data_field:
                data_outflow_field.rotate_field_by_a_hundred_and_eighty_degrees()
            if flip_ref_field:
                ref_outflow_field.flip_data_ud()
            if rotate_ref_field:
                ref_outflow_field.rotate_field_by_a_hundred_and_eighty_degrees()
            reference_rmouth_outflows_filename=os.path.join(self.scratch_dir,
                                                            self.temp_label + os.path.\
                                                            basename(reference_rmouth_outflows_filename))
            data_rmouth_outflows_filename=os.path.join(self.scratch_dir,
                                                       self.temp_label + os.path.\
                                                       basename(reference_rmouth_outflows_filename))
            temp_file_list.append(reference_rmouth_outflows_filename)
            temp_file_list.append(data_rmouth_outflows_filename)
            iodriver.write_field(reference_rmouth_outflows_filename,
                                 field=ref_outflow_field,
                                 file_type=iodriver.\
                                 get_file_extension(reference_rmouth_outflows_filename))
            iodriver.write_field(data_rmouth_outflows_filename,
                                 field=data_outflow_field,
                                 file_type=iodriver.\
                                 get_file_extension(data_rmouth_outflows_filename))
        matchedpairs, unresolved_conflicts  = mtch_rm.main(reference_rmouth_outflows_filename=\
                                                           reference_rmouth_outflows_filename,
                                                           data_rmouth_outflows_filename=\
                                                           data_rmouth_outflows_filename,
                                                           grid_type=grid_type,
                                                           flip_data_field=flip_data_field,
                                                           rotate_data_field=rotate_data_field,
                                                           flip_ref_field=flip_ref_field,
                                                           rotate_ref_field=rotate_ref_field,
                                                           param_set=matching_parameter_set,
                                                           **grid_kwargs)
        if additional_matches_list_filename:
            additional_matches = mtch_rm.load_additional_manual_matches(additional_matches_list_filepath,
                                                                        reference_rmouth_outflows_filename,
                                                                        data_rmouth_outflows_filename,
                                                                        flip_data_field=flip_data_field,
                                                                        rotate_data_field=rotate_data_field,
                                                                        grid_type='HD',**grid_kwargs)
            matchedpairs.extend(additional_matches)
        if ref_orog_filename:
            ref_orog_field[rdirs_field <= 0] = np.ma.masked
        interactive_plots = Interactive_Plots()
        if return_simple_catchment_and_flowmap_plotters and plot_simple_catchment_and_flowmap_plots:
            simple_catchment_and_flowmap_plotters = []
        if return_catchment_plotters:
            catchment_plotters = []
        for pair in matchedpairs:
            if pair[0].get_lat() > 310:
                continue
            if select_only_rivers_in == "North America":
                if(pair[0].get_lat() > 156 or pair[0].get_lon() > 260):
                    continue
            print("Ref Point: " + str(pair[0]) + "Matches: " + str(pair[1]))
            if rivers_to_plot is not None:
                if not (pair[0].get_lat(),pair[0].get_lon()) in rivers_to_plot:
                    continue
            if split_comparison_plots_across_multiple_canvases:
                plt.figure(figsize=(25,6.25))
                ax = plt.subplot(121)
                plt.tight_layout()
            else:
                plt.figure(figsize=(25,12.5))
                ax = plt.subplot(222)
            rc_pts.plot_river_rmouth_flowmap(ax=ax,
                                             ref_flowtocellfield=ref_flowtocellfield,
                                             data_flowtocellfield=data_flowtocellfield,
                                             rdirs_field=rdirs_field,
                                             pair=pair,colors=self.colors)
            if split_comparison_plots_across_multiple_canvases:
                ax_hist = plt.subplot(122)
                plt.figure(figsize=(12.5,12.5))
                ax_catch = plt.subplot(111)
                plt.tight_layout(rect=(0,0,0.9,1))
            else:
                ax_hist = plt.subplot(221)
                ax_catch = plt.subplot(223)
            plot_catchment_and_histogram_output = \
                rc_pts.plot_catchment_and_histogram_for_river(ax_hist=ax_hist,ax_catch=ax_catch,
                                                              ref_catchment_field=ref_catchment_field,
                                                              data_catchment_field=data_catchment_field,
                                                              data_catchment_field_original_scale=\
                                                              data_catchment_field_original_scale,
                                                              data_original_scale_flowtocellfield=\
                                                              data_original_scale_flowtocellfield,
                                                              rdirs_field=rdirs_field,
                                                              data_rdirs_field=data_rdirs_field,pair=pair,
                                                              catchment_grid_changed=catchment_grid_changed,
                                                              swap_ref_and_data_when_finding_labels=\
                                                              swap_ref_and_data_when_finding_labels,
                                                              colors=self.colors,
                                                              ref_grid=ref_grid,
                                                              grid_type=grid_type,
                                                              alternative_catchment_bounds=\
                                                              alternative_catchment_bounds,
                                                              use_simplified_catchment_colorscheme=\
                                                              use_simplified_catchment_colorscheme,
                                                              use_upscaling_labels=\
                                                              use_upscaling_labels,
                                                              allow_new_sink_points=\
                                                              allow_new_true_sinks,
                                                              external_landsea_mask=\
                                                              external_ls_mask,
                                                              ref_original_scale_flowtocellfield=\
                                                              ref_original_scale_flowtocellfield,
                                                              ref_catchment_field_original_scale=\
                                                              ref_catchment_field_original_scale,
                                                              use_original_scale_field_for_determining_data_and_ref_labels=\
                                                              use_original_scale_field_for_determining_data_and_ref_labels,
                                                              return_catchment_plotter=\
                                                              return_catchment_plotters,
                                                              data_original_scale_grid_type=\
                                                              data_original_scale_grid_type,
                                                              ref_original_scale_grid_type=\
                                                              ref_original_scale_grid_type,
                                                              data_original_scale_grid_kwargs=\
                                                              data_original_scale_grid_kwargs,
                                                              ref_original_scale_grid_kwargs=\
                                                              ref_original_scale_grid_kwargs,
                                                              **grid_kwargs)
            if return_catchment_plotters:
                catchment_section,catchment_bounds,scale_factor,catchment_plotter =\
                    plot_catchment_and_histogram_output
                catchment_plotters.append(catchment_plotter)
            else:
                catchment_section,catchment_bounds,scale_factor = plot_catchment_and_histogram_output
            if split_comparison_plots_across_multiple_canvases:
                plt.figure(figsize=(12.5,12.5))
                ax = plt.subplot(111)
                plt.tight_layout(rect=(0,0,0.9,1))
            else:
                ax = plt.subplot(224)
            rc_pts.plot_whole_river_flowmap(ax=ax,pair=pair,ref_flowtocellfield=ref_flowtocellfield,
                                            data_flowtocellfield=data_flowtocellfield,
                                            rdirs_field=rdirs_field,data_rdirs_field=data_rdirs_field,
                                            catchment_bounds=catchment_bounds,colors=self.colors,
                                            simplified_flowmap_plot=use_simplified_flowmap_colorscheme,
                                            allow_new_sink_points=allow_new_true_sinks)
            if plot_simple_catchment_and_flowmap_plots:
                simple_candf_plt = plt.figure(figsize=(10,6))
                simple_ref_ax  = plt.subplot(121)
                simple_data_ax = plt.subplot(122)
                flowtocell_threshold = 75
                plotters = rc_pts.simple_catchment_and_flowmap_plots(fig=simple_candf_plt,
                                                                     ref_ax=simple_ref_ax,
                                                                     data_ax=simple_data_ax,
                                                                     ref_catchment_field=ref_catchment_field,
                                                                     data_catchment_field=data_catchment_field,
                                                                     data_catchment_field_original_scale=\
                                                                     data_catchment_field_original_scale,
                                                                     ref_flowtocellfield=ref_flowtocellfield,
                                                                     data_flowtocellfield=data_flowtocellfield,
                                                                     data_original_scale_flowtocellfield=\
                                                                     data_original_scale_flowtocellfield,
                                                                     pair=pair,catchment_bounds=catchment_bounds,
                                                                     flowtocell_threshold=flowtocell_threshold,
                                                                     catchment_grid_changed=catchment_grid_changed,
                                                                     colors=self.colors,
                                                                     external_ls_mask=external_ls_mask,
                                                                     grid_type=grid_type,
                                                                     data_original_scale_grid_type=\
                                                                     data_original_scale_grid_type,
                                                                     data_original_scale_grid_kwargs=\
                                                                     data_original_scale_grid_kwargs,**grid_kwargs)
                if return_simple_catchment_and_flowmap_plotters:
                    simple_catchment_and_flowmap_plotters.append(plotters)
            if ref_orog_filename and data_orog_original_scale_filename:
                if super_fine_orog_filename:
                            data_to_super_fine_scale_factor = \
                                pts.calculate_scale_factor(coarse_grid_type=data_original_scale_grid_type,
                                                           coarse_grid_kwargs=data_original_scale_grid_kwargs,
                                                           fine_grid_type=super_fine_orog_grid_type,
                                                           fine_grid_kwargs=super_fine_orog_grid_kwargs)
                            ref_to_super_fine_scale_factor = data_to_super_fine_scale_factor*scale_factor
                else:
                    ref_to_super_fine_scale_factor=None
                interactive_plots.setup_plots(catchment_section,
                                              ref_orog_field,
                                              data_orog_original_scale_field,
                                              ref_flowtocellfield,
                                              data_flowtocellfield,
                                              rdirs_field,
                                              super_fine_orog_field,
                                              super_fine_data_flowmap,
                                              pair, catchment_bounds,
                                              scale_factor,
                                              ref_to_super_fine_scale_factor,
                                              ref_grid_offset_adjustment=ref_grid.\
                                              get_longitude_offset_adjustment(),
                                              fine_grid_offset_adjustment=fine_grid.\
                                              get_longitude_offset_adjustment(),
                                              super_fine_grid_offset_adjustment=super_fine_grid.\
                                              get_longitude_offset_adjustment())
            elif ref_orog_filename or data_orog_original_scale_filename:
                raise UserWarning("No orography plot generated, require both a reference orography"
                                  " and a data orography to generate an orography plot")
        print("Unresolved Conflicts: ")
        for conflict in unresolved_conflicts:
            print(" Conflict:")
            for pair in conflict:
                print("  Ref Point" + str(pair[0]) + "Matches" + str(pair[1]))
        for temp_file in temp_file_list:
            if os.path.basename(temp_file).startswith("temp_"):
                print("Deleting File: {0}".format(temp_file))
                os.remove(temp_file)
        if return_simple_catchment_and_flowmap_plotters and plot_simple_catchment_and_flowmap_plots:
            return simple_catchment_and_flowmap_plotters
        if return_catchment_plotters:
            return catchment_plotters

    def Compare_Corrected_HD_Rdirs_And_ICE5G_as_HD_data_ALG4_sinkless_all_points_0k(self):
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        ice5g_as_HD_data_ALG4_sinkless_all_points_0k = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_ICE5G_as_HD_data_ALG4_sinkless_all_points_0k_20160427_134237.nc")
        self.OutFlowComparisonPlotHelpers(corrected_hd_rdirs_rmouthoutflow_file,
                                                ice5g_as_HD_data_ALG4_sinkless_all_points_0k,
                                                "flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                                "flowmap_ICE5G_as_HD_data_ALG4_sinkless_all_points_0k_20160427_134237.nc",
                                                "rivdir_vs_1_9_data_from_stefan.nc",
                                                grid_type='HD')

    def Compare_Corrected_HD_Rdirs_And_ICE5G_as_HD_data_ALG4_true_sinks_all_points_0k(self):
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        ice5g_as_HD_data_ALG4_sinkless_all_points_0k = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_ICE5G_as_HD_data_ALG4_sinkless_all_points_0k_20160608_184931.nc")
        self.OutFlowComparisonPlotHelpers(corrected_hd_rdirs_rmouthoutflow_file,
                                                ice5g_as_HD_data_ALG4_sinkless_all_points_0k,
                                                "flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                                "flowmap_ICE5G_as_HD_data_ALG4_sinkless_all_points_0k_20160608_184931.nc",
                                                "rivdir_vs_1_9_data_from_stefan.nc",
                                                ref_catchment_filename=\
                                                "catchmentmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                                data_catchment_filename=\
                                                "catchmentmap_ICE5G_as_HD_data_ALG4_sinkless_all_points_0k_20160608_184931.nc",
                                                grid_type='HD')

    def Compare_Corrected_HD_Rdirs_And_ICE5G_ALG4_sinkless_all_points_0k_directly_upscaled_fields(self):
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.rmouth_outflow_data_directory,
            "upscaled/rmouthflows_ICE5G_data_ALG4_sinkless_0k_upscale_riverflows_and_river_mouth_flows_20160502_163323.nc")
        self.OutFlowComparisonPlotHelpers(corrected_hd_rdirs_rmouthoutflow_file,
                                                ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field,
                                                "flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                                "upscaled/flowmap_ICE5G_data_ALG4_sinkless_0k_upscale_riverflows"
                                                "_and_river_mouth_flows_20160502_163323.nc",
                                                "rivdir_vs_1_9_data_from_stefan.nc",
                                                flip_data_field=True,
                                                rotate_data_field=True,
                                                super_fine_orog_filename="ETOPO1_Ice_c_gmt4.nc",
                                                flip_super_fine_orog=True,
                                                rotate_super_fine_orog=False,
                                                super_fine_orog_grid_type='LatLong1min',
                                                grid_type='HD')

    def Compare_Corrected_HD_Rdirs_And_ICE5G_ALG4_true_sinks_all_points_0k_directly_upscaled_fields(self):
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.rmouth_outflow_data_directory,
            "upscaled/rmouthflows__ICE5G_data_ALG4_sinkless_0k_upscale_riverflows_and_river_mouth_flows_20160603_112520.nc")
        self.OutFlowComparisonPlotHelpers(corrected_hd_rdirs_rmouthoutflow_file,
                                                ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field,
                                                "flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                                "upscaled/flowmap__ICE5G_data_ALG4_sinkless_0k_upscale_riverflows"
                                                "_and_river_mouth_flows_20160603_112520.nc",
                                                "rivdir_vs_1_9_data_from_stefan.nc",
                                                flip_data_field=True,
                                                rotate_data_field=True,
                                                data_rdirs_filename="generated/"
                                                "updated_RFDs_ICE5G_data_ALG4_sinkless_0k_20160603_112512.nc",
                                                ref_catchment_filename=\
                                                "catchmentmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                                data_catchment_filename=\
                                                "upscaled/catchmentmap_unsorted__ICE5G_data_ALG4_sinkless_0k_upscale_riverflows"
                                                "_and_river_mouth_flows_20160704_152025.nc",
                                                data_catchment_original_scale_filename=\
                                                "catchmentmap_unsorted_ICE5G_data_ALG4_sinkless_0k_20160603_112512.nc",
                                                data_original_scale_flow_map_filename=\
                                                "flowmap_ICE5G_data_ALG4_sinkless_0k_20160603_112512.nc",
                                                ref_orog_filename="topo_hd_vs1_9_data_from_stefan.nc",
                                                data_orog_original_scale_filename=
                                                "ice5g_v1_2_00_0k_10min.nc",
                                                additional_matches_list_filename=\
                                                'additional_matches_ice5g_10min.txt',
                                                super_fine_orog_filename="ETOPO1_Ice_c_gmt4.nc",
                                                flip_super_fine_orog=True,
                                                rotate_super_fine_orog=False,
                                                super_fine_orog_grid_type='LatLong1min',
                                                grid_type='HD',data_original_scale_grid_type='LatLong10min')

    def Compare_Corrected_HD_Rdirs_And_ICE5G_ALG4_corr_orog_all_points_0k_directly_upscaled_fields(self):
        data_creation_datetime="20160802_112138"
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.rmouth_outflow_data_directory,
            "upscaled/rmouthflows_ICE5G_data_ALG4_sinkless_0k_{0}.nc".format(data_creation_datetime))
        self.OutFlowComparisonPlotHelpers(corrected_hd_rdirs_rmouthoutflow_file,
                                          ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field,
                                          "flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                          "upscaled/flowmap_ICE5G_data_ALG4_sinkless_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          "rivdir_vs_1_9_data_from_stefan.nc",
                                          flip_data_field=True,
                                          rotate_data_field=True,
                                          data_rdirs_filename="generated/"
                                          "updated_RFDs_ICE5G_data_ALG4_sinkless_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          ref_catchment_filename=\
                                          "catchmentmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                          data_catchment_filename=\
                                          "upscaled/catchmentmap_unsorted_ICE5G_data_ALG4_sinkless_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          data_catchment_original_scale_filename=\
                                          "catchmentmap_unsorted_ICE5G_data_ALG4_sinkless_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          data_original_scale_flow_map_filename=\
                                          "flowmap_ICE5G_data_ALG4_sinkless_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          ref_orog_filename="topo_hd_vs1_9_data_from_stefan.nc",
                                          data_orog_original_scale_filename=
                                          "generated/corrected/"
                                          "corrected_orog_ICE5G_data_ALG4_sinkless_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          additional_matches_list_filename=\
                                          'additional_matches_ice5g_10min.txt',
                                          super_fine_orog_filename="ETOPO1_Ice_c_gmt4.nc",
                                          super_fine_data_flowmap_filename=
                                            "flowmap_etopo1_data_ALG4_sinkless_20160603_112520.nc",
                                          flip_super_fine_orog=True,
                                          rotate_super_fine_orog=False,
                                          super_fine_orog_grid_type='LatLong1min',
                                          grid_type='HD',data_original_scale_grid_type='LatLong10min')

    def Compare_Corrected_HD_Rdirs_And_ICE5G_ALG4_corr_orog_downscaled_ls_mask_all_points_0k_directly_upscaled_fields(self):
        #data_creation_datetime="20160930_001057" #original rdirs from the original complete corrected orography
        data_creation_datetime="20170514_104220" #Version with Amu Darya added
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.rmouth_outflow_data_directory,
            "upscaled/rmouthflows_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".format(data_creation_datetime))
        self.OutFlowComparisonPlotHelpers(corrected_hd_rdirs_rmouthoutflow_file,
                                          ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field,
                                          "flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                          "upscaled/flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          "rivdir_vs_1_9_data_from_stefan.nc",
                                          #It is no longer required to flip data when using 2017 or later data files
                                          flip_data_field=False,
                                          rotate_data_field=True,
                                          data_rdirs_filename="generated/"
                                          "updated_RFDs_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          ref_catchment_filename=\
                                          "catchmentmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                          data_catchment_filename=\
                                          "upscaled/catchmentmap_unsorted_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          data_catchment_original_scale_filename=\
                                          "catchmentmap_unsorted_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          data_original_scale_flow_map_filename=\
                                          "flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          ref_orog_filename="topo_hd_vs1_9_data_from_stefan.nc",
                                          data_orog_original_scale_filename=
                                          "generated/corrected/"
                                          "corrected_orog_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          additional_matches_list_filename=\
                                          'additional_matches_ice5g_10min.txt',
                                          catchment_and_outflows_mods_list_filename='catch_and_outflow_mods_ice5g_10min.txt',
                                          super_fine_orog_filename="ETOPO1_Ice_c_gmt4.nc",
                                          super_fine_data_flowmap_filename=
                                          "flowmap_etopo1_data_ALG4_sinkless_20160603_112520.nc",
                                          flip_super_fine_orog=True,
                                          rotate_super_fine_orog=False,
                                          super_fine_orog_grid_type='LatLong1min',
                                          grid_type='HD',data_original_scale_grid_type='LatLong10min')

    def Compare_ICE5G_with_and_without_tarasov_upscaled_srtm30_ALG4_corr_orog_0k_directly_upscaled_fields(self):
        data_creation_datetime_with_tarasov="20170511_121440"
        data_creation_datetime_ICE5G_alone="20170505_144847"
        data_creation_datetime_ICE5G_alone_upscaled="20170507_135726"
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field_without_tarasov_ups_data =\
        os.path.join(self.rmouth_outflow_data_directory,
                     "upscaled/rmouthflows_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".format(data_creation_datetime_ICE5G_alone))
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field_with_tarasov_ups_data = \
        os.path.join(self.rmouth_outflow_data_directory,
                     "upscaled/rmouthflows_ICE5G_and_tarasov_upscaled_srtm30plus_data_ALG4_sinkless_"
                     "downscaled_ls_mask_0k_{0}.nc".format(data_creation_datetime_with_tarasov))
        self.OutFlowComparisonPlotHelpers(ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field_without_tarasov_ups_data,
                                          ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field_with_tarasov_ups_data,
                                          "upscaled/flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime_ICE5G_alone),
                                          "upscaled/flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_data_"
                                          "ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime_with_tarasov),
                                          "generated/upscaled/upscaled_rdirs_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k"
                                          "_upscale_rdirs_{0}_updated.nc".\
                                          format(data_creation_datetime_ICE5G_alone_upscaled),
                                          flip_data_field=False,rotate_data_field=True,
                                          flip_ref_field=False,rotate_ref_field=True,
                                          data_rdirs_filename="generated/"
                                          "updated_RFDs_ICE5G_and_tarasov_upscaled_srtm30plus_data"
                                          "_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime_with_tarasov),
                                          ref_catchment_filename=\
                                          "upscaled/catchmentmap_unsorted_ICE5G_data_ALG4"
                                          "_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime_ICE5G_alone),
                                          data_catchment_filename=\
                                          "upscaled/catchmentmap_unsorted_ICE5G_and_tarasov_upscaled_srtm30plus_"
                                          "data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime_with_tarasov),
                                          data_catchment_original_scale_filename=\
                                          "catchmentmap_unsorted_ICE5G_and_tarasov_upscaled_srtm30plus_data_"
                                          "ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime_with_tarasov),
                                          data_original_scale_flow_map_filename=\
                                          "flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_data_ALG4_sinkless"
                                          "_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime_with_tarasov),
                                          flip_orog_original_scale_relative_to_data=True,
                                          ref_orog_filename="topo_hd_vs1_9_data_from_stefan.nc",
                                          data_orog_original_scale_filename=
                                          "generated/corrected/"
                                          "corrected_orog_ICE5G_and_tarasov_upscaled_srtm30plus_data_ALG4_sinkless_"
                                          "downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime_with_tarasov),
                                          super_fine_orog_filename="ETOPO1_Ice_c_gmt4.nc",
                                          super_fine_data_flowmap_filename=
                                          "flowmap_etopo1_data_ALG4_sinkless_20160603_112520.nc",
                                          flip_super_fine_orog=True,
                                          rotate_super_fine_orog=False,
                                          select_only_rivers_in="North America",
                                          allow_new_true_sinks=True,
                                          use_original_scale_field_for_determining_data_and_ref_labels=True,
                                          ref_original_scale_flow_map_filename="flowmap_ICE5G_data_ALG4_sinkless_"
                                          "downscaled_ls_mask_0k_{0}.nc".format(data_creation_datetime_ICE5G_alone),
                                          ref_catchment_original_scale_filename=
                                          "catchmentmap_unsorted_ICE5G_data_ALG4_"
                                          "sinkless_downscaled_ls_mask_0k_{0}.nc".format(data_creation_datetime_ICE5G_alone),
                                          matching_parameter_set='magnitude_extensive',
                                          super_fine_orog_grid_type='LatLong1min',
                                          grid_type='HD',data_original_scale_grid_type='LatLong10min',
                                          ref_original_scale_grid_type='LatLong10min')


    def Compare_Corrected_HD_Rdirs_And_ICE5G_plus_tarasov_upscaled_srtm30_ALG4_corr_orog_0k_directly_upscaled_fields(self):
        data_creation_datetime="20170506_105104"
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.rmouth_outflow_data_directory,
            "upscaled/rmouthflows_ICE5G_and_tarasov_upscaled_srtm30plus_data_ALG4_sinkless_"
            "downscaled_ls_mask_0k_{0}.nc".format(data_creation_datetime))
        self.OutFlowComparisonPlotHelpers(corrected_hd_rdirs_rmouthoutflow_file,
                                          ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field,
                                          "flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                          "upscaled/flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_data_"
                                          "ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          "rivdir_vs_1_9_data_from_stefan.nc",
                                          flip_data_field=False,
                                          rotate_data_field=True,
                                          data_rdirs_filename="generated/"
                                          "updated_RFDs_ICE5G_and_tarasov_upscaled_srtm30plus_data"
                                          "_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          ref_catchment_filename=\
                                          "catchmentmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                          data_catchment_filename=\
                                          "upscaled/catchmentmap_unsorted_ICE5G_and_tarasov_upscaled_srtm30plus_"
                                          "data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          data_catchment_original_scale_filename=\
                                          "catchmentmap_unsorted_ICE5G_and_tarasov_upscaled_srtm30plus_data_"
                                          "ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          data_original_scale_flow_map_filename=\
                                          "flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_data_ALG4_sinkless"
                                          "_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          flip_orog_original_scale_relative_to_data=True,
                                          ref_orog_filename="topo_hd_vs1_9_data_from_stefan.nc",
                                          data_orog_original_scale_filename=
                                          "generated/corrected/"
                                          "corrected_orog_ICE5G_and_tarasov_upscaled_srtm30plus_data_ALG4_sinkless_"
                                          "downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime),
                                          additional_matches_list_filename=\
                                          'additional_matches_ice5g_10min.txt',
                                          catchment_and_outflows_mods_list_filename='catch_and_outflow_mods_ice5g_10min.txt',
                                          super_fine_orog_filename="ETOPO1_Ice_c_gmt4.nc",
                                          super_fine_data_flowmap_filename=
                                          "flowmap_etopo1_data_ALG4_sinkless_20160603_112520.nc",
                                          flip_super_fine_orog=True,
                                          rotate_super_fine_orog=False,
                                          select_only_rivers_in="North America",
                                          super_fine_orog_grid_type='LatLong1min',
                                          grid_type='HD',data_original_scale_grid_type='LatLong10min')

    def Compare_Original_Corrections_vs_Upscaled_MERIT_DEM_0k(self):
        reference_rmouth_outflows_filename= ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                                             "lake_analysis_one_21_Jun_2021/rivers/results/default_orog_corrs/"
                                             "diag_version_29_date_0_original_truesinks/"
                                             "10min_rmouth_flowtocell.nc")
        data_rmouth_outflows_filename= ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                                        "lake_analysis_two_26_Mar_2022/"
                                        "rivers/results/diag_version_0_date_0_original_truesinks/"
                                        "10min_rmouth_flowtocell.nc")
        ref_flowmaps_filename = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                                 "lake_analysis_one_21_Jun_2021/rivers/results/default_orog_corrs/"
                                 "diag_version_29_date_0_original_truesinks/"
                                 "10min_flowtocell.nc")
        data_flowmaps_filename = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                                  "lake_analysis_two_26_Mar_2022/"
                                  "rivers/results/diag_version_0_date_0_original_truesinks/"
                                  "10min_flowtocell.nc")
        ref_rdirs_filename = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                              "lake_analysis_one_21_Jun_2021/rivers/results/default_orog_corrs/"
                              "diag_version_29_date_0_original_truesinks/"
                              "10min_rdirs.nc")
        data_rdirs_filename = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                               "lake_analysis_two_26_Mar_2022/"
                               "rivers/results/diag_version_0_date_0_original_truesinks/10min_rdirs.nc")
        ref_catchments_filename = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                                   "lake_analysis_one_21_Jun_2021/rivers/results/default_orog_corrs/"
                                   "diag_version_29_date_0_original_truesinks/"
                                   "10min_catchments.nc")
        data_catchments_filename = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                                    "lake_analysis_two_26_Mar_2022/"
                                    "rivers/results/diag_version_0_date_0_original_truesinks/"
                                    "10min_catchments.nc")
        ref_orog_filename = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                             "lake_analysis_one_21_Jun_2021/rivers/results/default_orog_corrs/"
                             "diag_version_29_date_0_original_truesinks/"
                             "10min_corrected_orog_two.nc")
        data_orog_filename = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                              "lake_analysis_two_26_Mar_2022/"
                              "rivers/results/diag_version_0_date_0_original_truesinks/"
                              "10min_corrected_orog_two.nc")
        super_fine_orog_filename = ("/Users/thomasriddick/Documents/data/HDdata/orographys/"
                                    "ETOPO1_Ice_g_gmt4_grid_registered.nc")
        super_fine_data_flowmap_filename = ("/Users/thomasriddick/Documents/data/HDdata/flowmaps/"
                                            "flowmap_etopo1_data_ALG4_sinkless_20160603_112520.nc")
        extra_matches_filename = None
        catchment_and_outflow_mods_filename = None

        self.AdvancedOutFlowComparisonPlotHelpers(reference_rmouth_outflows_filename,
                                                  data_rmouth_outflows_filename,
                                                  ref_flowmaps_filename,
                                                  data_flowmaps_filename,
                                                  ref_rdirs_filename,
                                                  data_rdirs_filename,
                                                  ref_catchments_filename,
                                                  data_catchments_filename,
                                                  ref_orog_filename,
                                                  data_orog_filename,
                                                  super_fine_orog_filename=super_fine_orog_filename,
                                                  super_fine_data_flowmap_filename=super_fine_data_flowmap_filename,
                                                  external_ls_mask_filename=None,
                                                  additional_matches_list_filename=extra_matches_filename,
                                                  catchment_and_outflows_mods_list_filename=
                                                  catchment_and_outflow_mods_filename,
                                                  rivers_to_plot=None,
                                                  matching_parameter_set='minimal',
                                                  select_only_rivers_in=None,
                                                  allow_new_true_sinks=True,
                                                  alternative_catchment_bounds=None,
                                                  use_simplified_catchment_colorscheme=False,
                                                  use_simplified_flowmap_colorscheme=False,
                                                  use_upscaling_labels=False,
                                                  swap_ref_and_data_when_finding_labels=False,
                                                  super_fine_orog_grid_type='LatLong1min',
                                                  grid_type='LatLong10min')

    def Compare_Original_Corrections_vs_Upscaled_MERIT_DEM_0k_new_truesinks(self):
        ref_base_dir = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                        "lake_analysis_one_21_Jun_2021/rivers/results/"
                        "default_orog_corrs/"
                        "diag_version_29_date_0_original_truesinks")
        data_base_dir = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                         "lake_analysis_two_26_Mar_2022/"
                         "rivers/results/diag_version_32_date_0_with_truesinks")
        reference_rmouth_outflows_filename= os.path.join(ref_base_dir,
                                                         "10min_rmouth_flowtocell.nc")
        data_rmouth_outflows_filename= os.path.join(data_base_dir,
                                                    "10min_rmouth_flowtocell.nc")
        ref_flowmaps_filename = os.path.join(ref_base_dir,
                                             "10min_flowtocell.nc")
        data_flowmaps_filename = os.path.join(data_base_dir,
                                              "10min_flowtocell.nc")
        ref_rdirs_filename = os.path.join(ref_base_dir,
                                          "10min_rdirs.nc")
        data_rdirs_filename = os.path.join(data_base_dir,
                                           "10min_rdirs.nc")
        ref_catchments_filename = os.path.join(ref_base_dir,
                                               "10min_catchments_ext.nc")
        data_catchments_filename = os.path.join(data_base_dir,
                                                "10min_catchments_ext.nc")
        ref_orog_filename = os.path.join(ref_base_dir,
                                         "10min_corrected_orog_two.nc")
        data_orog_filename = os.path.join(data_base_dir,
                                          "10min_corrected_orog_two.nc")
        super_fine_orog_filename = ("/Users/thomasriddick/Documents/data/HDdata/orographys/"
                                    "ETOPO1_Ice_g_gmt4_grid_registered.nc")
        super_fine_data_flowmap_filename = ("/Users/thomasriddick/Documents/data/HDdata/flowmaps/"
                                            "flowmap_etopo1_data_ALG4_sinkless_20160603_112520.nc")
        extra_matches_filename = ("/Users/thomasriddick/Documents/data/HDdata/addmatches/"
                                  "additional_matches_10min_upscaled_MERIT_rdirs_vs_modern_day.txt")
        catchment_and_outflow_mods_filename = ("/Users/thomasriddick/Documents/data/HDdata/"
                                               "catchmods/catch_and_outflow_mods_10min_upscaled_MERIT_rdirs_vs_modern_day.txt")
        self.AdvancedOutFlowComparisonPlotHelpers(reference_rmouth_outflows_filename,
                                                  data_rmouth_outflows_filename,
                                                  ref_flowmaps_filename,
                                                  data_flowmaps_filename,
                                                  ref_rdirs_filename,
                                                  data_rdirs_filename,
                                                  ref_catchments_filename,
                                                  data_catchments_filename,
                                                  ref_orog_filename,
                                                  data_orog_filename,
                                                  super_fine_orog_filename=super_fine_orog_filename,
                                                  super_fine_data_flowmap_filename=super_fine_data_flowmap_filename,
                                                  external_ls_mask_filename=None,
                                                  additional_matches_list_filename=extra_matches_filename,
                                                  catchment_and_outflows_mods_list_filename=
                                                  catchment_and_outflow_mods_filename,
                                                  rivers_to_plot=None,
                                                  matching_parameter_set='minimal',
                                                  select_only_rivers_in=None,
                                                  allow_new_true_sinks=True,
                                                  alternative_catchment_bounds=None,
                                                  use_simplified_catchment_colorscheme=False,
                                                  use_simplified_flowmap_colorscheme=False,
                                                  use_upscaling_labels=False,
                                                  swap_ref_and_data_when_finding_labels=False,
                                                  super_fine_orog_grid_type='LatLong1min',
                                                  grid_type='LatLong10min')

    def Compare_Original_Corrections_vs_Upscaled_MERIT_DEM_0k_new_truesinks_individual_rivers(self):
        ref_base_dir = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                        "lake_analysis_one_21_Jun_2021/rivers/results/"
                        "default_orog_corrs/"
                        "diag_version_29_date_0_original_truesinks")
        data_base_dir = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                         "lake_analysis_two_26_Mar_2022/"
                         "rivers/results/diag_version_32_date_0_with_truesinks")
        reference_rmouth_outflows_filename= os.path.join(ref_base_dir,
                                                         "10min_rmouth_flowtocell.nc")
        data_rmouth_outflows_filename= os.path.join(data_base_dir,
                                                    "10min_rmouth_flowtocell.nc")
        ref_flowmaps_filename = os.path.join(ref_base_dir,
                                             "10min_flowtocell.nc")
        data_flowmaps_filename = os.path.join(data_base_dir,
                                              "10min_flowtocell.nc")
        ref_rdirs_filename = os.path.join(ref_base_dir,
                                          "10min_rdirs.nc")
        data_rdirs_filename = os.path.join(data_base_dir,
                                           "10min_rdirs.nc")
        ref_catchments_filename = os.path.join(ref_base_dir,
                                               "10min_catchments_ext.nc")
        data_catchments_filename = os.path.join(data_base_dir,
                                                "10min_catchments_ext.nc")
        ref_orog_filename = os.path.join(ref_base_dir,
                                         "10min_corrected_orog_two.nc")
        data_orog_filename = os.path.join(data_base_dir,
                                          "10min_corrected_orog_two.nc")
        super_fine_orog_filename = ("/Users/thomasriddick/Documents/data/HDdata/orographys/"
                                    "ETOPO1_Ice_g_gmt4_grid_registered.nc")
        super_fine_data_flowmap_filename = ("/Users/thomasriddick/Documents/data/HDdata/flowmaps/"
                                            "flowmap_etopo1_data_ALG4_sinkless_20160603_112520.nc")
        extra_matches_filename = ("/Users/thomasriddick/Documents/data/HDdata/addmatches/"
                                  "additional_matches_10min_upscaled_MERIT_rdirs_vs_modern_day_selected_catchments.txt")
        catchment_and_outflow_mods_filename = ("/Users/thomasriddick/Documents/data/HDdata/"
                                               "catchmods/catch_and_outflow_mods_10min_upscaled_MERIT_rdirs_vs_modern_day.txt")
        self.AdvancedOutFlowComparisonPlotHelpers(reference_rmouth_outflows_filename,
                                                  data_rmouth_outflows_filename,
                                                  ref_flowmaps_filename,
                                                  data_flowmaps_filename,
                                                  ref_rdirs_filename,
                                                  data_rdirs_filename,
                                                  ref_catchments_filename,
                                                  data_catchments_filename,
                                                  ref_orog_filename,
                                                  data_orog_filename,
                                                  super_fine_orog_filename=super_fine_orog_filename,
                                                  super_fine_data_flowmap_filename=super_fine_data_flowmap_filename,
                                                  external_ls_mask_filename=None,
                                                  additional_matches_list_filename=extra_matches_filename,
                                                  catchment_and_outflows_mods_list_filename=
                                                  catchment_and_outflow_mods_filename,
                                                  rivers_to_plot=None,
                                                  matching_parameter_set='no_matches',
                                                  select_only_rivers_in=None,
                                                  allow_new_true_sinks=True,
                                                  alternative_catchment_bounds=None,
                                                  use_simplified_catchment_colorscheme=False,
                                                  use_simplified_flowmap_colorscheme=False,
                                                  use_upscaling_labels=False,
                                                  swap_ref_and_data_when_finding_labels=False,
                                                  super_fine_orog_grid_type='LatLong1min',
                                                  grid_type='LatLong10min')

    def Compare_Corrected_HD_Rdirs_And_Etopo1_ALG4_sinkless_directly_upscaled_fields(self):
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.rmouth_outflow_data_directory,
            "upscaled/rmouthflows_etopo1_data_ALG4_sinkless_upscale_riverflows_and_river_mouth_flows_20160503_231022.nc")
        self.OutFlowComparisonPlotHelpers(corrected_hd_rdirs_rmouthoutflow_file,
                                                ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field,
                                                "flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                                "upscaled/flowmap_etopo1_data_ALG4_sinkless_upscale_riverflows_"
                                                "and_river_mouth_flows_20160503_231022.nc",
                                                "rivdir_vs_1_9_data_from_stefan.nc",
                                                flip_data_field=True,
                                                grid_type='HD')

    def Compare_Corrected_HD_Rdirs_And_Etopo1_ALG4_true_sinks_directly_upscaled_fields(self):
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.rmouth_outflow_data_directory,
            "upscaled/rmouthflows__etopo1_data_ALG4_sinkless_upscale_riverflows_and_river_mouth_flows_20160603_114215.nc")
        self.OutFlowComparisonPlotHelpers(corrected_hd_rdirs_rmouthoutflow_file,
                                                ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field,
                                                "flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                                "upscaled/flowmap__etopo1_data_ALG4_sinkless_upscale_riverflows_"
                                                "and_river_mouth_flows_20160603_114215.nc",
                                                "rivdir_vs_1_9_data_from_stefan.nc",
                                                flip_data_field=True,
                                                ref_catchment_filename=\
                                                "catchmentmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                                data_catchment_filename=\
                                                "catchmentmap_unsorted_etopo1_data_ALG4_sinkless_20160603_112520.nc",
                                                data_original_scale_flow_map_filename=\
                                                "flowmap_etopo1_data_ALG4_sinkless_20160603_112520.nc",
                                                grid_type='HD',data_original_scale_grid_type='LatLong1min')

    def Compare_Upscaled_Rdirs_vs_Directly_Upscaled_fields_ICE5G_data_ALG4_corr_orog_downscaled_ls_mask_0k(self):
        data_creation_datetime_directly_upscaled="20160930_001057"
        data_creation_datetime_rdirs_upscaled = "20161031_113238"
        ice5g_ALG4_sinkless_all_points_0k_river_flow_dir_upsc_field = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs_{0}_updated.nc"\
                .format(data_creation_datetime_rdirs_upscaled))
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.rmouth_outflow_data_directory,
            "upscaled/rmouthflows_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc"\
                .format(data_creation_datetime_directly_upscaled))
        self.OutFlowComparisonPlotHelpers(ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field,
                                          ice5g_ALG4_sinkless_all_points_0k_river_flow_dir_upsc_field,
                                          "upscaled/flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime_directly_upscaled),
                                          "flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs_{0}_updated.nc".\
                                          format(data_creation_datetime_rdirs_upscaled),
                                          "generated/upscaled/"
                                          "upscaled_rdirs_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs_{0}_updated.nc".\
                                          format(data_creation_datetime_rdirs_upscaled),
                                          flip_ref_field=True,
                                          rotate_ref_field=True,
                                          flip_data_field=True,
                                          rotate_data_field=True,
                                          ref_orog_filename=\
                                          "topo_hd_vs1_9_data_from_stefan.nc",
                                          data_orog_original_scale_filename=\
                                          "generated/corrected/"
                                          "corrected_orog_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".format(data_creation_datetime_directly_upscaled),
                                          data_catchment_filename=\
                                          "catchmentmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs_{0}_updated.nc".\
                                          format(data_creation_datetime_rdirs_upscaled),
                                          ref_catchment_filename=
                                          "upscaled/catchmentmap_unsorted_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime_directly_upscaled),
                                          data_catchment_original_scale_filename=\
                                          "catchmentmap_unsorted_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime_directly_upscaled),
                                          data_original_scale_flow_map_filename=\
                                          "flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime_directly_upscaled),
                                          swap_ref_and_data_when_finding_labels=True,
                                          catchment_and_outflows_mods_list_filename=\
                                          "catch_and_outflow_mods_ice5g_10min_directly_upscaled_rdirs_vs_indirectly_upscaled_data.txt",
                                          matching_parameter_set='area',
                                          grid_type='HD',data_original_scale_grid_type='LatLong10min')

class FlowMapPlots(Plots):
    """A general base class for flow maps"""

    flow_maps_path_extension = 'flowmaps'
    ls_masks_extension       = 'lsmasks'
    hdpara_extension         = 'hdfiles'
    orography_extension      = 'orographys'
    catchments_extension     = 'catchmentmaps'

    def __init__(self,save,color_palette_to_use='default'):
        """Class Constructor"""
        super(FlowMapPlots,self).__init__(save,color_palette_to_use)
        self.flow_maps_data_directory = os.path.join(self.hd_data_path,self.flow_maps_path_extension)
        self.ls_masks_data_directory= os.path.join(self.hd_data_path,self.ls_masks_extension)
        self.hdpara_directory = os.path.join(self.hd_data_path,self.hdpara_extension)
        self.orography_directory = os.path.join(self.hd_data_path,self.orography_extension)
        self.catchments_directory = os.path.join(self.hd_data_path,self.catchments_extension)


    def FourFlowMapSectionsFromDeglaciation(self,time_one=14000,time_two=13600,time_three=12700,time_four=12660):
        """ """
        flowmap_one_filename = os.path.join(self.flow_maps_data_directory,
                                "30min_flowtocell_pmu0171a_{}.nc".format(time_one))
        flowmap_two_filename = os.path.join(self.flow_maps_data_directory,
                                "30min_flowtocell_pmu0171b_{}.nc".format(time_two))
        flowmap_three_filename = os.path.join(self.flow_maps_data_directory,
                                  "30min_flowtocell_pmu0171b_{}.nc".format(time_three))
        flowmap_four_filename = os.path.join(self.flow_maps_data_directory,
                                  "30min_flowtocell_pmu0171b_{}.nc".format(time_four))
        catchments_one_filename = os.path.join(self.catchments_directory,
                                               "30min_catchments_pmu0171a_{}.nc".format(time_one))
        catchments_two_filename = os.path.join(self.catchments_directory,
                                               "30min_catchments_pmu0171b_{}.nc".format(time_two))
        catchments_three_filename = os.path.join(self.catchments_directory,
                                               "30min_catchments_pmu0171b_{}.nc".format(time_three))
        catchments_four_filename = os.path.join(self.catchments_directory,
                                               "30min_catchments_pmu0171b_{}.nc".format(time_four))
        lsmask_one_filename = os.path.join(self.hdpara_directory,
                                  "hdpara_{}k.nc".format(time_one))
        lsmask_two_filename = os.path.join(self.hdpara_directory,
                                  "hdpara_{}k.nc".format(time_two))
        lsmask_three_filename = os.path.join(self.hdpara_directory,
                                    "hdpara_{}k.nc".format(time_three))
        lsmask_four_filename = os.path.join(self.hdpara_directory,
                                   "hdpara_{}k.nc".format(time_four))
        glac_mask_one_filename = os.path.join(self.orography_directory,
                                              "glac01_{}.nc".format(time_one))
        glac_mask_two_filename = os.path.join(self.orography_directory,
                                              "glac01_{}.nc".format(time_two))
        glac_mask_three_filename = os.path.join(self.orography_directory,
                                              "glac01_{}.nc".format(time_three))
        glac_mask_four_filename = os.path.join(self.orography_directory,
                                              "glac01_{}.nc".format(time_four))
        flowmap_one = iodriver.load_field(flowmap_one_filename,
                                          file_type=iodriver.get_file_extension(flowmap_one_filename),
                                          field_type='Generic',
                                          grid_type='HD').get_data()
        lsmask_one = iodriver.load_field(lsmask_one_filename,
                                         file_type=iodriver.get_file_extension(lsmask_one_filename),
                                         field_type='Generic',
                                         fieldname='FLAG',
                                         grid_type='HD').get_data().astype(np.int32)
        glac_mask_one = iodriver.load_field(glac_mask_one_filename,
                                            file_type=iodriver.get_file_extension(glac_mask_one_filename),
                                            field_type='Generic',
                                            fieldname='glac',
                                            grid_type='LatLong10min')
        glac_mask_hd_one = utilities.upscale_field(glac_mask_one,"HD",'Sum',
                                                   output_grid_kwargs={},
                                                   scalenumbers=True)
        glac_mask_hd_one.flip_data_ud()
        glac_mask_hd_one.rotate_field_by_a_hundred_and_eighty_degrees()
        glac_mask_hd_one = glac_mask_hd_one.get_data()
        catchments_one = iodriver.load_field(catchments_one_filename,
                                             file_type=iodriver.get_file_extension(catchments_one_filename),
                                             field_type='Generic',
                                             grid_type='HD').get_data()
        flowmap_two = iodriver.load_field(flowmap_two_filename,
                                          file_type=iodriver.get_file_extension(flowmap_two_filename),
                                          field_type='Generic',
                                          grid_type='HD').get_data()
        lsmask_two = iodriver.load_field(lsmask_two_filename,
                                         file_type=iodriver.get_file_extension(lsmask_two_filename),
                                         field_type='Generic',
                                         fieldname='FLAG',
                                         grid_type='HD').get_data().astype(np.int32)
        glac_mask_two = iodriver.load_field(glac_mask_two_filename,
                                            file_type=iodriver.get_file_extension(glac_mask_two_filename),
                                            field_type='Generic',
                                            fieldname='glac',
                                            grid_type='LatLong10min')
        glac_mask_hd_two = utilities.upscale_field(glac_mask_two,"HD",'Sum',
                                                   output_grid_kwargs={},
                                                   scalenumbers=True)
        glac_mask_hd_two.flip_data_ud()
        glac_mask_hd_two.rotate_field_by_a_hundred_and_eighty_degrees()
        glac_mask_hd_two = glac_mask_hd_two.get_data()
        catchments_two = iodriver.load_field(catchments_two_filename,
                                             file_type=iodriver.get_file_extension(catchments_two_filename),
                                             field_type='Generic',
                                             grid_type='HD').get_data()
        flowmap_three = iodriver.load_field(flowmap_three_filename,
                                            file_type=iodriver.get_file_extension(flowmap_three_filename),
                                            field_type='Generic',
                                            grid_type='HD').get_data()
        lsmask_three = iodriver.load_field(lsmask_three_filename,
                                           file_type=iodriver.get_file_extension(lsmask_three_filename),
                                           field_type='Generic',
                                           fieldname='FLAG',
                                           grid_type='HD').get_data().astype(np.int32)
        glac_mask_three = iodriver.load_field(glac_mask_three_filename,
                                              file_type=iodriver.get_file_extension(glac_mask_three_filename),
                                              field_type='Generic',
                                              fieldname='glac',
                                              grid_type='LatLong10min')
        glac_mask_hd_three = utilities.upscale_field(glac_mask_three,"HD",'Sum',
                                                   output_grid_kwargs={},
                                                   scalenumbers=True)
        glac_mask_hd_three.flip_data_ud()
        glac_mask_hd_three.rotate_field_by_a_hundred_and_eighty_degrees()
        glac_mask_hd_three = glac_mask_hd_three.get_data()
        catchments_three = iodriver.load_field(catchments_three_filename,
                                               file_type=iodriver.get_file_extension(catchments_three_filename),
                                               field_type='Generic',
                                               grid_type='HD').get_data()
        flowmap_four = iodriver.load_field(flowmap_four_filename,
                                           file_type=iodriver.get_file_extension(flowmap_four_filename),
                                           field_type='Generic',
                                           grid_type='HD').get_data()
        lsmask_four = iodriver.load_field(lsmask_four_filename,
                                          file_type=iodriver.get_file_extension(lsmask_four_filename),
                                          field_type='Generic',
                                          fieldname='FLAG',
                                          grid_type='HD').get_data().astype(np.int32)
        glac_mask_four = iodriver.load_field(glac_mask_four_filename,
                                             file_type=iodriver.get_file_extension(glac_mask_four_filename),
                                             field_type='Generic',
                                             fieldname='glac',
                                             grid_type='LatLong10min')
        glac_mask_hd_four = utilities.upscale_field(glac_mask_four,"HD",'Sum',
                                                    output_grid_kwargs={},
                                                    scalenumbers=True)
        catchments_four = iodriver.load_field(catchments_four_filename,
                                              file_type=iodriver.get_file_extension(catchments_four_filename),
                                              field_type='Generic',
                                              grid_type='HD').get_data()
        glac_mask_hd_four.flip_data_ud()
        glac_mask_hd_four.rotate_field_by_a_hundred_and_eighty_degrees()
        glac_mask_hd_four = glac_mask_hd_four.get_data()
        bounds=[0,150,60,265]
        fig = plt.figure(figsize=(14,10))
        gs = gridspec.GridSpec(2,3,width_ratios=[4,4,1])
        ax1 = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[0,1])
        ax3 = plt.subplot(gs[1,0])
        ax4 = plt.subplot(gs[1,1])
        cax = plt.subplot(gs[:,2])
        rc_pts.simple_thresholded_data_only_flowmap(ax1,flowmap_one,lsmask_one,threshold=75,
                                                    glacier_mask=glac_mask_hd_one,
                                                    catchments=catchments_one,
                                                    catchnumone=4,
                                                    catchnumtwo=30,
                                                    catchnumthree=20,
                                                    bounds=bounds,
                                                    cax = cax,
                                                    colors=self.colors)
        ax1.set_title("{} BP".format(time_one))
        rc_pts.simple_thresholded_data_only_flowmap(ax2,flowmap_two,lsmask_two,threshold=75,
                                                    glacier_mask=glac_mask_hd_two,
                                                    catchments=catchments_two,
                                                    catchnumone=4,
                                                    catchnumtwo=30,
                                                    catchnumthree=51,
                                                    bounds=bounds,
                                                    colors=self.colors)
        ax2.set_title("{} BP".format(time_two))
        rc_pts.simple_thresholded_data_only_flowmap(ax3,flowmap_three,lsmask_three,threshold=75,
                                                    glacier_mask=glac_mask_hd_three,
                                                    catchments=catchments_three,
                                                    catchnumone=3,
                                                    catchnumtwo=21,
                                                    catchnumthree=8,
                                                    bounds=bounds,
                                                    colors=self.colors)
        ax3.set_title("{} BP".format(time_three))
        rc_pts.simple_thresholded_data_only_flowmap(ax4,flowmap_four,lsmask_four,threshold=75,
                                                    glacier_mask=glac_mask_hd_four,
                                                    catchments=catchments_four,
                                                    catchnumone=11,
                                                    catchnumtwo=7,
                                                    catchnumthree=8,
                                                    bounds=bounds,
                                                    colors=self.colors)
        ax4.set_title("{} BP".format(time_four))
        gs.tight_layout(fig,rect=(0,0.1,1,1))

    def SimpleFlowMapPlotHelper(self,filename,grid_type,log_max=4):
        """Help produce simple flow maps"""
        flowmap_object = iodriver.load_field(filename,
                                             file_type=iodriver.get_file_extension(filename),
                                             field_type='Generic',
                                             grid_type=grid_type)
        flowmap = flowmap_object.get_data()
        plt.figure()
        plt.subplot(111)
        if log_max == 0:
            log_max = math.log(np.amax(flowmap))
        levels = np.logspace(0,log_max,num=50)
        #ctrs = plt.contourf(flowmap,levels=levels,norm=colors.LogNorm())
        #plt.contourf(flowmap,levels=levels,norm=colors.LogNorm())
        plt.contourf(flowmap,levels=levels)
        #cbar = plt.colorbar(ctrs)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Number of cells flowing to cell')
        pts.remove_ticks()
        if self.save:
            #plt.savefig('')
            pass

    def FlowMapTwoColourComparisonHelper(self,ref_filename,data_filename,lsmask_filename=None,
                                         grid_type='HD',minflowcutoff=100,flip_data=False,
                                         rotate_data=False,flip_ref=False,rotate_ref=False,
                                         lsmask_has_same_orientation_as_ref=True,
                                         invert_ls_mask=False,
                                         first_datasource_name="Reference",
                                         second_datasource_name="Data",
                                         add_title=True,**kwargs):
        """Help compare two two-colour flow maps"""
        flowmap_ref_field = iodriver.load_field(ref_filename,
                                                file_type=iodriver.get_file_extension(ref_filename),
                                                field_type='Generic',
                                                grid_type=grid_type,**kwargs)
        flowmap_data_field = iodriver.load_field(data_filename,
                                                 file_type=iodriver.get_file_extension(data_filename),
                                                 field_type='Generic',
                                                 grid_type=grid_type,**kwargs)
        if lsmask_filename:
            lsmask_field = iodriver.load_field(lsmask_filename,
                                               file_type=iodriver.get_file_extension(lsmask_filename),
                                               field_type='Generic', grid_type=grid_type,**kwargs)
        if flip_data:
            flowmap_data_field.flip_data_ud()
        if rotate_data:
            flowmap_data_field.rotate_field_by_a_hundred_and_eighty_degrees()
        if flip_ref:
            flowmap_ref_field.flip_data_ud()
            if lsmask_filename and lsmask_has_same_orientation_as_ref:
                lsmask_field.flip_data_ud()
        if rotate_ref:
            flowmap_ref_field.rotate_field_by_a_hundred_and_eighty_degrees()
            if lsmask_filename and lsmask_has_same_orientation_as_ref:
                lsmask_field.rotate_field_by_a_hundred_and_eighty_degrees()
        if invert_ls_mask:
            lsmask_field.invert_data()
        if lsmask_filename:
            lsmask = lsmask_field.get_data()
        flowmap_ref_field = flowmap_ref_field.get_data()
        flowmap_data_field = flowmap_data_field.get_data()
        plt.figure(figsize=(20,8))
        ax = plt.subplot(111)
        fmp_pts.make_basic_flowmap_comparison_plot(ax,flowmap_ref_field,flowmap_data_field,minflowcutoff,
                                                   first_datasource_name,second_datasource_name,lsmask,
                                                   colors=self.colors,add_title=add_title)

    def FlowMapTwoColourPlotHelper(self,filename,lsmask_filename=None,grid_type='HD',
                                   minflowcutoff=100,flip_data=False,flip_mask=False,
                                   **kwargs):
        """Help produce two colour flow maps"""
        flowmap_object = iodriver.load_field(filename,
                                             file_type=iodriver.get_file_extension(filename),
                                             field_type='Generic',
                                             grid_type=grid_type,**kwargs)
        lsmask_field = iodriver.load_field(lsmask_filename,
                                           file_type=iodriver.get_file_extension(lsmask_filename),
                                           field_type='Generic', grid_type=grid_type,**kwargs)
        if flip_data:
            flowmap_object.flip_data_ud()
        if flip_mask:
            lsmask_field.flip_data_ud()
        lsmask = lsmask_field.get_data()
        flowmap = flowmap_object.get_data()
        plt.figure()
        plt.subplot(111)
        flowmap[flowmap < minflowcutoff] = 1
        flowmap[flowmap >= minflowcutoff] = 2
        if lsmask is not None:
            flowmap[lsmask == 1] = 0
        cmap = mpl.colors.ListedColormap(['blue','peru','black'])
        bounds = list(range(4))
        norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
        plt.imshow(flowmap,cmap=cmap,norm=norm,interpolation="none")
        plt.title('Cells with cumulative flow greater than or equal to {0}'.format(minflowcutoff))

    def Etopo1FlowMap(self):
        filename=os.path.join(self.flow_maps_data_directory,
                              'flowmap_etopo1_data_ALG4_sinkless_20160603_114215.nc')
        self.SimpleFlowMapPlotHelper(filename,'LatLong1min')

    def Etopo1FlowMap_two_colour(self):
        filename=os.path.join(self.flow_maps_data_directory,
                              'flowmap_etopo1_data_ALG4_sinkless_20160603_112520.nc')
        lsmask_filename = os.path.join(self.ls_masks_data_directory,'generated',
                                       'ls_mask_etopo1_data_ALG4_sinkless_20160603_112520.nc')
        self.FlowMapTwoColourPlotHelper(filename,lsmask_filename=lsmask_filename,
                                        grid_type='LatLong1min',
                                        minflowcutoff=25000,flip_data=True,flip_mask=True)

    def Etopo1FlowMap_two_colour_directly_upscaled_fields(self):
        filename=os.path.join(self.flow_maps_data_directory,'upscaled',
                              'flowmap__etopo1_data_ALG4_sinkless_upscale_riverflows'
                              '_and_river_mouth_flows_20160603_114215.nc')
        lsmask_filename = os.path.join(self.ls_masks_data_directory,'generated',
                                       'ls_mask_extract_ls_mask_from_corrected_'
                                       'HD_rdirs_20160504_142435.nc')
        self.FlowMapTwoColourPlotHelper(filename,lsmask_filename=lsmask_filename,
                                        grid_type='HD',
                                        minflowcutoff=50,flip_data=True,flip_mask=False)

    def Corrected_HD_Rdirs_FlowMap_two_colour(self):
        filename=os.path.join(self.flow_maps_data_directory,
                              'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        lsmask_filename = os.path.join(self.ls_masks_data_directory,'generated',
                                       'ls_mask_extract_ls_mask_from_corrected_'
                                       'HD_rdirs_20160504_142435.nc')
        self.FlowMapTwoColourPlotHelper(filename,lsmask_filename=lsmask_filename,
                                        grid_type='HD',
                                        minflowcutoff=25,flip_data=False,flip_mask=False)


    def Corrected_HD_Rdirs_And_Etopo1_ALG4_sinkless_directly_upscaled_fields_FlowMap_comparison(self):
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,'upscaled',
                                   'flowmap_etopo1_data_ALG4_sinkless_upscale_riverflows'
                                   '_and_river_mouth_flows_20160503_231022.nc')
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_"
                                     "HD_rdirs_20160504_142435.nc")
        self.FlowMapTwoColourComparisonHelper(ref_filename=ref_filename,
                                              data_filename=data_filename,
                                              lsmask_filename=lsmask_filename,
                                              grid_type='HD',
                                              minflowcutoff=75,
                                              flip_data=True)

    def Corrected_HD_Rdirs_And_Etopo1_ALG4_true_sinks_directly_upscaled_fields_FlowMap_comparison(self):
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,'upscaled',
                                   'flowmap__etopo1_data_ALG4_sinkless_upscale_riverflows'
                                    '_and_river_mouth_flows_20160603_114215.nc')
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_"
                                     "HD_rdirs_20160504_142435.nc")
        self.FlowMapTwoColourComparisonHelper(ref_filename=ref_filename,
                                              data_filename=data_filename,
                                              lsmask_filename=lsmask_filename,
                                              grid_type='HD',
                                              minflowcutoff=50,
                                              flip_data=True)

    def Corrected_HD_Rdirs_And_ICE5G_data_ALG4_sinkless_0k_directly_upscaled_fields_FlowMap_comparison(self):
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,'upscaled',
                                    "flowmap_ICE5G_data_ALG4_sinkless_0k_upscale_riverflows"
                                   "_and_river_mouth_flows_20160502_163323.nc")
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_"
                                     "HD_rdirs_20160504_142435.nc")
        self.FlowMapTwoColourComparisonHelper(ref_filename=ref_filename,
                                              data_filename=data_filename,
                                              lsmask_filename=lsmask_filename,
                                              grid_type='HD',
                                              minflowcutoff=75,
                                              flip_data=True,
                                              rotate_data=True)

    def Corrected_HD_Rdirs_And_ICE5G_data_ALG4_true_sinks_0k_directly_upscaled_fields_FlowMap_comparison(self):
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,'upscaled',
                                    "flowmap__ICE5G_data_ALG4_sinkless_0k_upscale_riverflows"
                                   "_and_river_mouth_flows_20160603_112520.nc")
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_"
                                     "HD_rdirs_20160504_142435.nc")
        self.FlowMapTwoColourComparisonHelper(ref_filename=ref_filename,
                                              data_filename=data_filename,
                                              lsmask_filename=lsmask_filename,
                                              grid_type='HD',
                                              minflowcutoff=75,
                                              flip_data=True,
                                              rotate_data=True)

    def Corrected_HD_Rdirs_And_ICE5G_data_ALG4_corr_orog_0k_directly_upscaled_fields_FlowMap_comparison(self):
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,'upscaled',
                                   "flowmap_ICE5G_data_ALG4_sinkless_0k_20160802_112138.nc")
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_"
                                     "HD_rdirs_20160504_142435.nc")
        self.FlowMapTwoColourComparisonHelper(ref_filename=ref_filename,
                                              data_filename=data_filename,
                                              lsmask_filename=lsmask_filename,
                                              grid_type='HD',
                                              minflowcutoff=75,
                                              flip_data=True,
                                              rotate_data=True)

    def Corrected_HD_Rdirs_And_ICE5G_data_ALG4_corr_orog_downscaled_ls_mask_0k_directly_upscaled_fields_FlowMap_comparison(self):
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,'upscaled',
                                   "flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_20160919_090154.nc")
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_"
                                     "HD_rdirs_20160504_142435.nc")
        self.FlowMapTwoColourComparisonHelper(ref_filename=ref_filename,
                                              data_filename=data_filename,
                                              lsmask_filename=lsmask_filename,
                                              grid_type='HD',
                                              minflowcutoff=75,
                                              flip_data=True,
                                              rotate_data=True)

    def Corrected_HD_Rdirs_And_ICE5G_data_ALG4_no_true_sinks_corr_orog_0k_directly_upscaled_fields_FlowMap_comparison(self):
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,'upscaled',
                                   "flowmap_ICE5G_data_ALG4_sinkless_no_true_sinks_0k_20160718_114758.nc")
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_"
                                     "HD_rdirs_20160504_142435.nc")
        self.FlowMapTwoColourComparisonHelper(ref_filename=ref_filename,
                                              data_filename=data_filename,
                                              lsmask_filename=lsmask_filename,
                                              grid_type='HD',
                                              minflowcutoff=75,
                                              flip_data=True,
                                              rotate_data=True)

    def Corrected_HD_Rdirs_And_ICE5G_HD_as_data_ALG4_true_sinks_0k_directly_upscaled_fields_FlowMap_comparison(self):
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,
                                   "flowmap_ICE5G_as_HD_data_ALG4_sinkless_all_points_0k_20160608_184931.nc")
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_"
                                     "HD_rdirs_20160504_142435.nc")
        self.FlowMapTwoColourComparisonHelper(ref_filename=ref_filename,
                                              data_filename=data_filename,
                                              lsmask_filename=lsmask_filename,
                                              grid_type='HD',
                                              minflowcutoff=75,
                                              flip_data=False,
                                              rotate_data=False)

    def Upscaled_Rdirs_vs_Directly_Upscaled_fields_ICE5G_data_ALG4_corr_orog_downscaled_ls_mask_0k_FlowMap_comparison(self):
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs_20161031_113238_updated.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,'upscaled',
                                   "flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_20160930_001057.nc")
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_"
                                     "HD_rdirs_20160504_142435.nc")
        self.FlowMapTwoColourComparisonHelper(ref_filename=ref_filename,
                                              data_filename=data_filename,
                                              lsmask_filename=lsmask_filename,
                                              grid_type='HD',
                                              minflowcutoff=50,
                                              flip_data=True,
                                              rotate_data=True,
                                              flip_ref=True,
                                              rotate_ref=True,
                                              lsmask_has_same_orientation_as_ref=False)

    def Upscaled_Rdirs_vs_Corrected_HD_Rdirs_ICE5G_data_ALG4_corr_orog_downscaled_ls_mask_0k_FlowMap_comparison(self):
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs_20161031_113238_updated.nc')
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_"
                                     "HD_rdirs_20160504_142435.nc")
        self.FlowMapTwoColourComparisonHelper(ref_filename=ref_filename,
                                              data_filename=data_filename,
                                              lsmask_filename=lsmask_filename,
                                              grid_type='HD',
                                              minflowcutoff=60,
                                              flip_data=True,
                                              rotate_data=True,
                                              flip_ref=False,
                                              rotate_ref=False,
                                              lsmask_has_same_orientation_as_ref=False)

    def ICE5G_data_ALG4_true_sinks_21k_And_ICE5G_data_ALG4_true_sinks_0k_FlowMap_comparison(self):
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_ICE5G_data_ALG4_sinkless_21k_20160603_132009.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,
                                    "flowmap_ICE5G_data_ALG4_sinkless_0k_20160603_112512.nc")
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_connected_ICE5G_data_ALG4_sinkless_21k_20160603_132009.nc")
        self.FlowMapTwoColourComparisonHelper(ref_filename=ref_filename,
                                              data_filename=data_filename,
                                              lsmask_filename=lsmask_filename,
                                              grid_type='LatLong10min',
                                              minflowcutoff=75,
                                              flip_data=True,
                                              rotate_data=True,
                                              flip_ref=True,
                                              rotate_ref=True)

    def ICE5G_data_all_points_0k_alg4_two_colour(self):
        filename=os.path.join(self.flow_maps_data_directory,
                              'flowmap_ICE5G_data_ALG4_sinkless_0k_20160603_112512.nc')
        lsmask_filename = os.path.join(self.ls_masks_data_directory,'generated',
                                       'ls_mask_ICE5G_data_ALG4_sinkless_0k_20160603_112512.nc')
        self.FlowMapTwoColourPlotHelper(filename,lsmask_filename=lsmask_filename,
                                        grid_type='LatLong10min',
                                        minflowcutoff=250,flip_data=True,
                                        flip_mask=True)

    def ICE5G_data_all_points_21k_alg4_two_colour(self):
        filename=os.path.join(self.flow_maps_data_directory,
                              'flowmap_ICE5G_data_ALG4_sinkless_21k_20160603_132009.nc')
        lsmask_filename = os.path.join(self.ls_masks_data_directory,'generated',
                                       'ls_mask_ICE5G_data_ALG4_sinkless_21k_20160603_132009.nc')
        self.FlowMapTwoColourPlotHelper(filename,lsmask_filename=lsmask_filename,
                                        grid_type='LatLong10min',
                                        minflowcutoff=250,flip_data=True,
                                        flip_mask=True)

    def ICE5G_data_all_points_0k_alg4(self):
        filename=os.path.join(self.flow_maps_data_directory,
                              'flowmap_ICE5G_data_ALG4_sinkless_0k_20160603_112520.nc')
        self.SimpleFlowMapPlotHelper(filename,'LatLong10min')

    def ICE5G_data_all_points_0k_no_sink_filling(self):
        filename=os.path.join(self.flow_maps_data_directory,
                              'flowmap_ICE5G_data_all_points_0k_20160229_133433.nc')
        self.SimpleFlowMapPlotHelper(filename,'LatLong10min',log_max=3.0)

    def Ten_Minute_Data_from_Virna_data_ALG4_corr_orog_downscaled_lsmask_no_sinks_21k_vs_0k_FlowMap_comparison(self):
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  "flowmap_ten_minute_data_from_virna_0k_ALG4_sinkless"
                                   "_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170123"
                                   "_165707_upscaled_updated.nc")
        data_filename=os.path.join(self.flow_maps_data_directory,
                                   "flowmap_ten_minute_data_from_virna_lgm_ALG4_sinkless"
                                   "_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170127"
                                   "_163957_upscaled_updated.nc")
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_ten_minute_data_from_virna_lgm_"
                                     "ALG4_sinkless_no_true_sinks_oceans_lsmask"
                                     "_plus_upscale_rdirs_20170127_163957_HD_transf.nc")
        self.FlowMapTwoColourComparisonHelper(ref_filename=ref_filename,
                                              data_filename=data_filename,
                                              lsmask_filename=lsmask_filename,
                                              grid_type='HD',
                                              minflowcutoff=35,
                                              flip_data=False,
                                              rotate_data=True,
                                              flip_ref=False,
                                              rotate_ref=True,
                                              lsmask_has_same_orientation_as_ref=False,
                                              invert_ls_mask=True,
                                              first_datasource_name="Present day",
                                              second_datasource_name="LGM")

class FlowMapPlotsWithCatchments(FlowMapPlots):
    """Flow map plots with selected catchments areas overlaid"""

    catchments_path_extension = 'catchmentmaps'
    rdirs_path_extension = 'rdirs'
    rmouth_outflow_path_extension = 'rmouthflow'
    catchment_and_outflows_mods_list_extension = 'catchmods'
    additional_matches_list_extension = 'addmatches'
    additional_truesink_matches_list_extension = 'addmatches_truesinks'
    orog_path_extension = 'orographys'

    def __init__(self,save,color_palette_to_use='default'):
        """Class constructor"""
        super(FlowMapPlotsWithCatchments,self).__init__(save,color_palette_to_use)
        self.catchments_data_directory = os.path.join(self.hd_data_path,self.catchments_path_extension)
        self.rdirs_data_directory = os.path.join(self.hd_data_path,self.rdirs_path_extension)
        self.rmouth_outflow_data_directory = os.path.join(self.hd_data_path,self.rmouth_outflow_path_extension)
        self.temp_label = 'temp_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + "_"
        self.additional_matches_list_directory = os.path.join(self.hd_data_path,
                                                              self.additional_matches_list_extension)
        self.additional_truesink_matches_list_directory = os.path.join(self.additional_matches_list_directory,
                                                                       self.additional_truesink_matches_list_extension)
        self.catchment_and_outflows_mods_list_directory = os.path.join(self.hd_data_path,
                                                                       self.catchment_and_outflows_mods_list_extension)
        self.orog_data_directory = os.path.join(self.hd_data_path,self.orog_path_extension)

    def FlowMapTwoColourComparisonWithCatchmentsHelper(self,ref_flowmap_filename,data_flowmap_filename,
                                                       ref_catchment_filename,data_catchment_filename,
                                                       ref_rdirs_filename,data_rdirs_filename,
                                                       reference_rmouth_outflows_filename,
                                                       data_rmouth_outflows_filename,
                                                       lsmask_filename=None,minflowcutoff=100,flip_data=False,
                                                       rotate_data=False,flip_ref=False,rotate_ref=False,
                                                       lsmask_has_same_orientation_as_ref=True,
                                                       flip_lsmask=False,rotate_lsmask=False,
                                                       invert_ls_mask=False,matching_parameter_set='default',
                                                       rivers_to_plot=None,
                                                       rivers_to_plot_alt_color=None,
                                                       rivers_to_plot_secondary_alt_color=None,
                                                       use_single_color_for_discrepancies=True,
                                                       use_only_one_color_for_flowmap=False,
                                                       additional_matches_list_filename=None,
                                                       additional_truesink_matches_list_filename=None,
                                                       catchment_and_outflows_mods_list_filename=None,
                                                       first_datasource_name="Reference",
                                                       second_datasource_name="Data",use_title=True,
                                                       remove_antartica=False,
                                                       difference_in_catchment_label="Discrepancy",
                                                       glacier_mask_filename=None,
                                                       extra_lsmask_filename=None,
                                                       show_true_sinks=False,
                                                       fig_size=(12,5),
                                                       grid_type='HD',
                                                       glacier_mask_grid_type='LatLong10min',
                                                       glacier_mask_grid_kwargs={},
                                                       flip_glacier_mask=False,
                                                       rotate_glacier_mask=False,
                                                       **grid_kwargs):
        """Help compare two two-colour flow maps"""
        if grid_type == "LatLong10min":
            scale_factor = 3
        else:
            scale_factor = 1
        if (rivers_to_plot_secondary_alt_color is not None):
            if (rivers_to_plot is None) or (rivers_to_plot_alt_color is None):
                raise RuntimeError("Invalid options - Secondary alternative color set when primary and/or"
                                   "secondary colors unused")
            else:
                rivers_to_plot_alt_color.extend(rivers_to_plot_secondary_alt_color)
        else:
            rivers_to_plot_secondary_alt_color = []
        flowmap_grid=grid.makeGrid(grid_type)
        ref_flowmaps_filepath = os.path.join(self.flow_maps_data_directory,ref_flowmap_filename)
        data_flowmaps_filepath = os.path.join(self.flow_maps_data_directory,data_flowmap_filename)
        ref_catchment_filepath = os.path.join(self.catchments_data_directory,
                                               ref_catchment_filename)
        data_catchment_filepath = os.path.join(self.catchments_data_directory,
                                               data_catchment_filename)
        flowmap_ref_field = iodriver.load_field(ref_flowmaps_filepath,
                                                file_type=iodriver.get_file_extension(ref_flowmaps_filepath),
                                                field_type='Generic',
                                                grid_type=grid_type,**grid_kwargs)
        flowmap_data_field = iodriver.load_field(data_flowmaps_filepath,
                                                 file_type=iodriver.get_file_extension(data_flowmaps_filepath),
                                                 field_type='Generic',
                                                 grid_type=grid_type,**grid_kwargs)
        data_catchment_field = iodriver.load_field(data_catchment_filepath,
                                                   file_type=iodriver.get_file_extension(data_catchment_filepath),
                                                   field_type='Generic',
                                                   grid_type=grid_type,**grid_kwargs)
        ref_catchment_field = iodriver.load_field(ref_catchment_filepath,
                                                  file_type=iodriver.get_file_extension(ref_catchment_filepath),
                                                  field_type='Generic',
                                                  grid_type=grid_type,**grid_kwargs)
        if data_rdirs_filename:
            data_rdirs_filepath =  os.path.join(self.rdirs_data_directory,
                                                data_rdirs_filename)
        ref_rdirs_filepath = os.path.join(self.rdirs_data_directory,ref_rdirs_filename)
        if data_rdirs_filename:
            data_rdirs_field = iodriver.load_field(data_rdirs_filepath,
                                                   file_type=iodriver.get_file_extension(data_rdirs_filepath),
                                                   field_type='Generic',
                                                   grid_type=grid_type,**grid_kwargs)
        else:
            data_rdirs_field = None
        ref_rdirs_field = iodriver.load_field(ref_rdirs_filepath,
                                              file_type=iodriver.get_file_extension(ref_rdirs_filepath),
                                              field_type='RiverDirections',
                                              grid_type=grid_type,**grid_kwargs)
        if lsmask_filename:
            lsmask_field = iodriver.load_field(lsmask_filename,
                                               file_type=iodriver.get_file_extension(lsmask_filename),
                                               field_type='Generic', grid_type=grid_type,**grid_kwargs)
        else:
            lsmask_field = field.Field(ref_rdirs_field.get_lsmask(),grid="LatLong10min")
        if extra_lsmask_filename:
            extra_lsmask_field = iodriver.load_field(extra_lsmask_filename,
                                                     file_type=iodriver.
                                                     get_file_extension(extra_lsmask_filename),
                                                     field_type='Generic',
                                                     grid_type=grid_type,**grid_kwargs)
        if catchment_and_outflows_mods_list_filename:
            catchment_and_outflows_mods_list_filepath = os.path.join(self.catchment_and_outflows_mods_list_directory,
                                                                     catchment_and_outflows_mods_list_filename)
        if additional_matches_list_filename:
            additional_matches_list_filepath = os.path.join(self.additional_matches_list_directory,
                                                            additional_matches_list_filename)
        if additional_truesink_matches_list_filename:
            additional_truesink_matches_list_filepath = os.path.join(self.additional_truesink_matches_list_directory,
                                                                     additional_truesink_matches_list_filename)
        if glacier_mask_filename:
            glacier_mask_field = iodriver.load_field(glacier_mask_filename,
                                                     file_type=iodriver.\
                                                     get_file_extension(glacier_mask_filename),
                                                     fieldname='sftgif',
                                                     field_type='Generic',
                                                     grid_type=glacier_mask_grid_type,
                                                     **glacier_mask_grid_kwargs)
            if glacier_mask_grid_type != grid_type:
                glacier_mask_field = utilities.upscale_field(glacier_mask_field,
                                                             output_grid_type=grid_type,
                                                             method="Mode",
                                                             output_grid_kwargs=grid_kwargs,
                                                             scalenumbers=False)
        else:
            glacier_mask_field=None
        if flip_data:
            flowmap_data_field.flip_data_ud()
            data_catchment_field.flip_data_ud()
            if data_rdirs_filename:
                data_rdirs_field.flip_data_ud()
        if rotate_data:
            flowmap_data_field.rotate_field_by_a_hundred_and_eighty_degrees()
            data_catchment_field.rotate_field_by_a_hundred_and_eighty_degrees()
            if data_rdirs_filename:
                data_rdirs_field.rotate_field_by_a_hundred_and_eighty_degrees()
        if flip_ref:
            flowmap_ref_field.flip_data_ud()
            ref_catchment_field.flip_data_ud()
            ref_rdirs_field.flip_data_ud()
            if lsmask_filename and lsmask_has_same_orientation_as_ref:
                lsmask_field.flip_data_ud()
        if rotate_ref:
            flowmap_ref_field.rotate_field_by_a_hundred_and_eighty_degrees()
            ref_catchment_field.rotate_field_by_a_hundred_and_eighty_degrees()
            ref_rdirs_field.rotate_field_by_a_hundred_and_eighty_degrees()
            if lsmask_filename and lsmask_has_same_orientation_as_ref:
                lsmask_field.rotate_field_by_a_hundred_and_eighty_degrees()
        if invert_ls_mask:
            lsmask_field.invert_data()
            if extra_lsmask_filename:
                extra_lsmask_field.invert_data()
        if flip_lsmask and not lsmask_has_same_orientation_as_ref:
            lsmask_field.flip_data_ud()
        if rotate_lsmask and not lsmask_has_same_orientation_as_ref:
            lsmask_field.rotate_field_by_a_hundred_and_eighty_degrees()
        if glacier_mask_filename:
            if flip_glacier_mask:
                glacier_mask_field.flip_data_ud()
            if rotate_glacier_mask:
                glacier_mask_field.rotate_field_by_a_hundred_and_eighty_degrees()
        plt.figure(figsize=fig_size)
        ax = plt.subplot(111)
        if extra_lsmask_filename:
            image_array,extra_lsmask =fmp_pts.\
              make_basic_flowmap_comparison_plot(ax,flowmap_ref_field.get_data(),
                                                flowmap_data_field.get_data(),
                                                minflowcutoff,
                                                first_datasource_name,
                                                second_datasource_name,
                                                lsmask_field.get_data(),
                                                return_image_array_instead_of_plotting=True,
                                                glacier_mask=glacier_mask_field,
                                                second_lsmask = extra_lsmask,
                                                scale_factor=scale_factor)
        else:
            image_array =fmp_pts.\
              make_basic_flowmap_comparison_plot(ax,flowmap_ref_field.get_data(),
                                                flowmap_data_field.get_data(),
                                                minflowcutoff,
                                                first_datasource_name,
                                                second_datasource_name,
                                                lsmask_field.get_data(),
                                                return_image_array_instead_of_plotting=True,
                                                glacier_mask=glacier_mask_field.get_data()
                                                             if glacier_mask_field is not None else None,
                                                scale_factor=scale_factor)
        temp_file_list = []
        if catchment_and_outflows_mods_list_filename:
            ref_outflow_field = iodriver.load_field(reference_rmouth_outflows_filename,
                                                    file_type=iodriver.\
                                                    get_file_extension(reference_rmouth_outflows_filename),
                                                    field_type='Generic', grid_type=grid_type,**grid_kwargs)
            data_outflow_field = iodriver.load_field(data_rmouth_outflows_filename,
                                                     file_type=iodriver.\
                                                     get_file_extension(data_rmouth_outflows_filename),
                                                     field_type='Generic', grid_type=grid_type,**grid_kwargs)
            if flip_data:
                data_outflow_field.flip_data_ud()
            if rotate_data:
                data_outflow_field.rotate_field_by_a_hundred_and_eighty_degrees()
            ref_catchment_field, ref_outflow_field, data_catchment_field, data_outflow_field =\
                rc_pts.modify_catchments_and_outflows(ref_catchment_field,ref_outflow_field,flowmap_ref_field,
                                                      ref_rdirs_field,data_catchment_field,data_outflow_field,
                                                      catchment_and_outflows_modifications_list_filename=\
                                                      catchment_and_outflows_mods_list_filepath,
                                                      grid_type=grid_type)
            if flip_data:
                data_outflow_field.flip_data_ud()
            if rotate_data:
                data_outflow_field.rotate_field_by_a_hundred_and_eighty_degrees()
            reference_rmouth_outflows_filename=os.path.join(self.scratch_dir,
                                                            self.temp_label + os.path.\
                                                            basename(reference_rmouth_outflows_filename))
            data_rmouth_outflows_filename=os.path.join(self.scratch_dir,
                                                       self.temp_label + os.path.\
                                                       basename(reference_rmouth_outflows_filename))
            temp_file_list.append(reference_rmouth_outflows_filename)
            temp_file_list.append(data_rmouth_outflows_filename)
            iodriver.write_field(reference_rmouth_outflows_filename,
                                 field=ref_outflow_field,
                                 file_type=iodriver.\
                                 get_file_extension(reference_rmouth_outflows_filename))
            iodriver.write_field(data_rmouth_outflows_filename,
                                 field=data_outflow_field,
                                 file_type=iodriver.\
                                 get_file_extension(data_rmouth_outflows_filename))
        #Using get data to convert field type causes confusion... possibly rewrite
        if lsmask_filename:
            lsmask = lsmask_field.get_data()
        else:
            lsmask = None
        if extra_lsmask_filename:
            extra_lsmask = extra_lsmask_field.get_data()
        flowmap_ref_field = flowmap_ref_field.get_data()
        flowmap_data_field = flowmap_data_field.get_data()
        data_catchment_field = data_catchment_field.get_data()
        ref_catchment_field = ref_catchment_field.get_data()
        if data_rdirs_filename:
            data_rdirs_field = data_rdirs_field.get_data()
        ref_rdirs_field = ref_rdirs_field.get_data()
        if glacier_mask_filename:
            glacier_mask_field = glacier_mask_field.get_data()
        matchedpairs,_  = mtch_rm.main(reference_rmouth_outflows_filename=\
                                       reference_rmouth_outflows_filename,
                                       data_rmouth_outflows_filename=\
                                       data_rmouth_outflows_filename,
                                       flip_data_field=flip_data,
                                       rotate_data_field=rotate_data,
                                       flip_ref_field=flip_ref,
                                       rotate_ref_field=rotate_ref,
                                       param_set=matching_parameter_set,
                                       grid_type=grid_type,**grid_kwargs)
        if additional_matches_list_filename:
            additional_matches = mtch_rm.load_additional_manual_matches(additional_matches_list_filepath,
                                                                        reference_rmouth_outflows_filename,
                                                                        data_rmouth_outflows_filename,
                                                                        flip_data_field=flip_data,
                                                                        rotate_data_field=rotate_data,
                                                                        grid_type=grid_type,**grid_kwargs)
            matchedpairs.extend(additional_matches)
        if additional_truesink_matches_list_filename:
            additional_matches = mtch_rm.load_additional_manual_truesink_matches(additional_truesink_matches_list_filepath,
                                                                                 reference_rmouth_outflows_filename,
                                                                                 data_rmouth_outflows_filename,
                                                                                 ref_flowmap_filename,
                                                                                 data_flowmap_filename,
                                                                                 flip_data_rmouth_outflow_field=\
                                                                                 flip_data,
                                                                                 rotate_data_rmouth_outflow_field=\
                                                                                 rotate_data,
                                                                                 flip_data_flowmap_field=\
                                                                                 flip_data,
                                                                                 rotate_data_flowmap_field=\
                                                                                 rotate_data,
                                                                                 grid_type=grid_type,
                                                                                 **grid_kwargs)
            matchedpairs.extend(additional_matches)
        for pair in matchedpairs:
            if pair[0].get_lat() > 310*scale_factor:
                continue
            alt_color_num = 8
            if (rivers_to_plot is not None) and (rivers_to_plot_alt_color is not None):
                if ((not (pair[0].get_lat(),pair[0].get_lon()) in rivers_to_plot) and
                    (not (pair[0].get_lat(),pair[0].get_lon()) in rivers_to_plot_alt_color)):
                    continue
                elif (((pair[0].get_lat(),pair[0].get_lon()) in rivers_to_plot) and
                      ((pair[0].get_lat(),pair[0].get_lon()) in rivers_to_plot_alt_color)):
                    raise RuntimeError("Cannot plot a catchment in both original and alternative colors - check for duplicate")
                elif ((pair[0].get_lat(),pair[0].get_lon()) in rivers_to_plot):
                    alt_color=False
                elif ((pair[0].get_lat(),pair[0].get_lon()) in rivers_to_plot_secondary_alt_color):
                    alt_color=True
                    alt_color_num = 9
                else:
                    alt_color=True
            elif  rivers_to_plot is not None:
                alt_color = False
                if not (pair[0].get_lat(),pair[0].get_lon()) in rivers_to_plot:
                    continue
            elif rivers_to_plot_alt_color is not None:
                alt_color = True
                if not (pair[0].get_lat(),pair[0].get_lon()) in rivers_to_plot_alt_color:
                    continue
            else:
                alt_color = False
            print("Ref Point: " + str(pair[0]) + "Matches: " + str(pair[1]))
            image_array = fmp_pts.add_selected_catchment_to_existing_plot(image_array,data_catchment_field,
                                                                          ref_catchment_field,data_catchment_field,
                                                                          flowmap_data_field, ref_rdirs_field,
                                                                          data_rdirs_field, pair=pair,
                                                                          catchment_grid_changed=False,
                                                                          use_alt_color=alt_color,
                                                                          alt_color_num=alt_color_num,
                                                                          use_single_color_for_discrepancies=\
                                                                          use_single_color_for_discrepancies,
                                                                          use_only_one_color_for_flowmap=\
                                                                          use_only_one_color_for_flowmap,
                                                                          allow_new_sink_points=show_true_sinks,
                                                                          grid_type=grid_type,
                                                                          data_original_scale_grid_type=grid_type)
        if extra_lsmask_filename:
            image_array = fmp_pts.add_extra_flowmap(image_array,extra_lsmask)
        if show_true_sinks:
            image_array[np.logical_and(ref_rdirs_field == 5,
                                       data_rdirs_field == 5)] = -4
            image_array[np.logical_and(ref_rdirs_field == 5,
                                       data_rdirs_field != 5)] = -5
            image_array[np.logical_and(ref_rdirs_field != 5,
                                       data_rdirs_field == 5)] = -6
        if remove_antartica:
            image_array = image_array[:320*scale_factor]
        fmp_pts.plot_composite_image(ax,image_array,minflowcutoff,first_datasource_name,second_datasource_name,
                                     use_single_color_for_discrepancies,use_only_one_color_for_flowmap,use_title,
                                     colors=self.colors,difference_in_catchment_label=difference_in_catchment_label,
                                     flowmap_grid=flowmap_grid,plot_glaciers=True if glacier_mask_filename else False,
                                     second_ls_mask=True if extra_lsmask_filename else False,
                                     show_true_sinks=show_true_sinks)
        for temp_file in temp_file_list:
            if os.path.basename(temp_file).startswith("temp_"):
                print("Deleting File: {0}".format(temp_file))
                os.remove(temp_file)

    def Upscaled_Rdirs_vs_Corrected_HD_Rdirs_ICE5G_data_ALG4_corr_orog_downscaled_ls_mask_0k_FlowMap_comparison(self):
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale'
                                  '_rdirs_20161031_113238_updated.nc')
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_"
                                     "HD_rdirs_20160504_142435.nc")
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
                                                             "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        upscaled_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
                                                         "rmouthflows_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs"
                                                         "_20161031_113238_updated.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            "catchmentmap_corrected_HD_rdirs_"
                                                            "post_processing_20160427_141158.nc",
                                                            data_catchment_filename="catchmentmap_ICE5G_data_ALG4_"
                                                            "sinkless_downscaled_ls_mask_0k_upscale_rdirs_20161031_113238_updated.nc",
                                                            ref_rdirs_filename="rivdir_vs_1_9_data_from_stefan.nc",
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            corrected_hd_rdirs_rmouthoutflow_file,
                                                            data_rmouth_outflows_filename=\
                                                            upscaled_rdirs_rmouthoutflow_file,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=100,flip_data=True,
                                                            rotate_data=True,flip_ref=False,rotate_ref=False,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            invert_ls_mask=False,
                                                            first_datasource_name="Reference",
                                                            matching_parameter_set='extensive',
                                                            rivers_to_plot=[(117,424),(121,176),(179,260),
                                                                            (160,573),(40,90),(217,432),
                                                                            (104,598),(46,504),(252,638),
                                                                            (32,612),(132,494),(171,371),
                                                                            (50,439),(121,456),(40,682),
                                                                            (88,430)],
                                                            rivers_to_plot_alt_color=[(192,384),(82,223),
                                                                                      (249,244),(117,603),
                                                                                      (35,521),(144,548),
                                                                                      (72,641),(54,29),
                                                                                      (88,457),(62,173),
                                                                                      (91,111),(125,165),
                                                                                      (159,235),(237,392),
                                                                                      (36,660),(51,717),
                                                                                      (33,603),(90,418),
                                                                                      (89,482),(111,380)],
                                                            rivers_to_plot_secondary_alt_color=[(64,175),
                                                                                                (42,468),
                                                                                                (32,577),
                                                                                                (43,508),
                                                                                                (117,130),
                                                                                                (230,427),
                                                                                                (36,631),
                                                                                                (86,436),
                                                                                                (55,174),
                                                                                                (82,113),
                                                                                                (60,416),
                                                                                                (154,388),
                                                                                                (136,536),
                                                                                                (201,286)],
                                                            use_single_color_for_discrepancies=True,
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "catch_and_outflow_mods_ice5g_10min_upscaled_"
                                                            "rdirs_vs_modern_day.txt",
                                                            second_datasource_name="Data",grid_type='HD')

    def compare_present_day_and_lgm_river_directions_with_catchments_virna_data_plus_tarasov_style_orog_corrs_for_both(self):
        """Compare LGM to present using Virna's data plus tarasov style orography corrections for both times"""
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  "flowmap_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans"
                                  "_lsmask_plus_upscale_rdirs_tarasov_orog_corrs_20170422_195301_upscaled_updated.nc")
        data_filename=os.path.join(self.flow_maps_data_directory,
                                   "flowmap_ten_minute_data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask"
                                   "_plus_upscale_rdirs_tarasov_orog_corrs_20170422_195436_upscaled_updated.nc")
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_ten_minute_data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask"
                                     "_plus_upscale_rdirs_tarasov_orog_corrs_20170422_195436_HD_transf.dat")
        ref_catchment_filename=("catchmentmap_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask"
                                "_plus_upscale_rdirs_tarasov_orog_corrs_20170422_195301_upscaled_updated.nc")
        data_catchment_filename=("catchmentmap_ten_minute_data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_"
                                 "lsmask_plus_upscale_rdirs_tarasov_orog_corrs_20170422_195436_upscaled_updated.nc")
        ref_rdirs_filename=("generated/upscaled/upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_"
                            "sinks_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs_20170422_195301_upscaled_"
                            "updated_transf.dat")
        reference_rmouth_outflows_filename=("/Users/thomasriddick/Documents/data/HDdata/rmouths/rmouthmap_ten_minute_"
                                            "data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale"
                                            "_rdirs_tarasov_orog_corrs_20170422_195301_upscaled_updated.nc")
        data_rmouth_outflows_filename=("/Users/thomasriddick/Documents/data/HDdata/rmouths/rmouthmap_ten_minute_"
                                       "data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale"
                                       "_rdirs_tarasov_orog_corrs_20170422_195436_upscaled_updated.nc")
        glacier_mask_filename=os.path.join(self.orog_data_directory,"ice5g_v1_2_21_0k_10min.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            ref_catchment_filename,
                                                            data_catchment_filename=\
                                                            data_catchment_filename,
                                                            ref_rdirs_filename=\
                                                            ref_rdirs_filename,
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            reference_rmouth_outflows_filename,
                                                            data_rmouth_outflows_filename=\
                                                            data_rmouth_outflows_filename,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=100,
                                                            flip_data=False,
                                                            rotate_data=True,
                                                            flip_ref=False,
                                                            rotate_ref=True,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            invert_ls_mask=True,
                                                            first_datasource_name="Present day",
                                                            second_datasource_name="LGM",
                                                            matching_parameter_set='extensive',
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "catch_and_outflow_mods_lgm_vs_present_day.txt",
                                                            additional_matches_list_filename=\
                                                            "additional_matches_10min_upscaled_lgm_vs_present.txt",
                                                            use_single_color_for_discrepancies=True,
                                                            use_only_one_color_for_flowmap=False,
                                                            use_title=False,remove_antartica=True,
                                                            difference_in_catchment_label="Difference",
                                                            glacier_mask_filename=glacier_mask_filename,
                                                            glacier_mask_grid_type='LatLong10min',
                                                            flip_glacier_mask=True,
                                                            rotate_glacier_mask=True,
                                                            grid_type='HD')

    def compare_present_day_river_directions_with_catchments_virna_data_with_vs_without_tarasov_style_orog_corrs(self):
        """Compare present day data with and without tarasov upscaling using virna's data"""
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  "flowmap_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans"
                                  "_lsmask_plus_upscale_rdirs_tarasov_orog_corrs_20170422_195301_upscaled_updated.nc")
        data_filename=os.path.join(self.flow_maps_data_directory,
                                   "flowmap_ten_minute_data_from_virna_0k_ALG4_sinkless"
                                   "_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170123"
                                   "_165707_upscaled_updated.nc")
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask"
                                     "_plus_upscale_rdirs_tarasov_orog_corrs_20170422_195301_HD_transf.nc")
        ref_catchment_filename=("catchmentmap_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask"
                                "_plus_upscale_rdirs_tarasov_orog_corrs_20170422_195301_upscaled_updated.nc")
        data_catchment_filename=("catchmentmap_ten_minute_data_from_virna_0k_ALG4_sinkless"
                                 "_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170123_165707_upscaled_updated.nc")
        ref_rdirs_filename=("generated/upscaled/upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_"
                            "sinks_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs_20170422_195301_upscaled_"
                            "updated_transf.dat")
        reference_rmouth_outflows_filename=("/Users/thomasriddick/Documents/data/HDdata/rmouths/rmouthmap_ten_minute_"
                                            "data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale"
                                            "_rdirs_tarasov_orog_corrs_20170422_195301_upscaled_updated.nc")
        data_rmouth_outflows_filename=("/Users/thomasriddick/Documents/data/HDdata/rmouths/rmouthmap_ten_"
                                       "minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus"
                                       "_upscale_rdirs_20170123_165707_upscaled_updated.nc")
        glacier_mask_filename=os.path.join(self.orog_data_directory,"ice5g_v1_2_21_0k_10min.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            ref_catchment_filename,
                                                            data_catchment_filename=\
                                                            data_catchment_filename,
                                                            ref_rdirs_filename=\
                                                            ref_rdirs_filename,
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            reference_rmouth_outflows_filename,
                                                            data_rmouth_outflows_filename=\
                                                            data_rmouth_outflows_filename,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=100,
                                                            flip_data=False,
                                                            rotate_data=True,
                                                            flip_ref=False,
                                                            rotate_ref=True,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            invert_ls_mask=True,
                                                            first_datasource_name="Present day",
                                                            second_datasource_name="LGM",
                                                            matching_parameter_set='extensive',
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "catch_and_outflow_mods_lgm_vs_present_day.txt",
                                                            additional_matches_list_filename=\
                                                            "additional_matches_10min_upscaled_lgm_vs_present.txt",
                                                            use_single_color_for_discrepancies=True,
                                                            use_only_one_color_for_flowmap=False,
                                                            use_title=False,remove_antartica=True,
                                                            difference_in_catchment_label="Difference",
                                                            glacier_mask_filename=glacier_mask_filename,
                                                            glacier_mask_grid_type='LatLong10min',
                                                            flip_glacier_mask=True,
                                                            rotate_glacier_mask=True,
                                                            grid_type='HD')

    def compare_lgm_river_directions_with_catchments_virna_data_with_vs_without_tarasov_style_orog_corrs(self):
        """Compare lgm data with and without tarasov upscaling using virna's data"""
        tarasov_upscaled_data_datetime="20170518_193949"
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  "flowmap_ten_minute_data_from_virna_lgm_ALG4_sinkless"
                                   "_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170127"
                                   "_163957_upscaled_updated.nc")
        data_filename=os.path.join(self.flow_maps_data_directory,
                                   "flowmap_ten_minute_data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask"
                                   "_plus_upscale_rdirs_tarasov_orog_corrs_{0}_upscaled_updated.nc".\
                                   format(tarasov_upscaled_data_datetime))
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_ten_minute_data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask"
                                     "_plus_upscale_rdirs_tarasov_orog_corrs_{0}_HD_transf.dat".\
                                     format(tarasov_upscaled_data_datetime))
        ref_catchment_filename=("catchmentmap_ten_minute_data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_"
                                 "lsmask_plus_upscale_rdirs_20170127_163957_upscaled_updated.nc")
        data_catchment_filename=("catchmentmap_ten_minute_data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_"
                                 "lsmask_plus_upscale_rdirs_tarasov_orog_corrs_{0}_upscaled_updated.nc".\
                                 format(tarasov_upscaled_data_datetime))
        ref_rdirs_filename=("generated/upscaled/upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true"
                            "_sinks_oceans_lsmask_plus_upscale_rdirs_20170123_165707.nc")
        reference_rmouth_outflows_filename=("/Users/thomasriddick/Documents/data/HDdata/rmouths/rmouthmap_ten_"
                                            "minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus"
                                            "_upscale_rdirs_20170123_165707_upscaled_updated.nc")
        data_rmouth_outflows_filename=("/Users/thomasriddick/Documents/data/HDdata/rmouths/rmouthmap_ten_minute_"
                                       "data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale"
                                       "_rdirs_tarasov_orog_corrs_{0}_upscaled_updated.nc".\
                                       format(tarasov_upscaled_data_datetime))
        glacier_mask_filename=os.path.join(self.orog_data_directory,"ice5g_v1_2_21_0k_10min.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            ref_catchment_filename,
                                                            data_catchment_filename=\
                                                            data_catchment_filename,
                                                            ref_rdirs_filename=\
                                                            ref_rdirs_filename,
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            reference_rmouth_outflows_filename,
                                                            data_rmouth_outflows_filename=\
                                                            data_rmouth_outflows_filename,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=100,
                                                            flip_data=False,
                                                            rotate_data=True,
                                                            flip_ref=False,
                                                            rotate_ref=True,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            invert_ls_mask=True,
                                                            first_datasource_name="Present day",
                                                            second_datasource_name="LGM",
                                                            matching_parameter_set='extensive',
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "catch_and_outflow_mods_lgm_vs_present_day.txt",
                                                            additional_matches_list_filename=\
                                                            "additional_matches_10min_upscaled_lgm_vs_present.txt",
                                                            use_single_color_for_discrepancies=True,
                                                            use_only_one_color_for_flowmap=False,
                                                            use_title=False,remove_antartica=True,
                                                            difference_in_catchment_label="Difference",
                                                            glacier_mask_filename=glacier_mask_filename,
                                                            glacier_mask_grid_type='LatLong10min',
                                                            flip_glacier_mask=True,
                                                            rotate_glacier_mask=True,
                                                            grid_type='HD')

    def upscaled_rdirs_with_and_without_tarasov_upscaled_data_ALG4_corr_orog_downscaled_ls_mask_0k_FlowMap_comparison(self):
        """

        Note this was adapted from previous code... not all variable names are accurate
        """

        data_label="20170511_163955"
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale'
                                  '_rdirs_20161031_113238_updated.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_data_ALG4_sinkless'
                                  '_downscaled_ls_mask_0k_upscale_rdirs_' + data_label + '_updated.nc')
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_"
                                     "HD_rdirs_20160504_142435.nc")
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
                                                         "rmouthflows_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs"
                                                         "_20161031_113238_updated.nc")
        upscaled_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
                                                         "rmouthflows_ICE5G_and_tarasov_upscaled_srtm30plus_data_ALG4_sinkless_"
                                                         "downscaled_ls_mask_0k_upscale_rdirs_" + data_label + "_updated.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            "catchmentmap_ICE5G_data_ALG4_"
                                                            "sinkless_downscaled_ls_mask_0k_upscale_rdirs_20161031_113238_updated.nc",
                                                            data_catchment_filename="catchmentmap_ICE5G_and_tarasov_upscaled_"
                                                            "srtm30plus_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_"
                                                            "rdirs_" + data_label + "_updated.nc",
                                                            ref_rdirs_filename="generated/upscaled/upscaled_rdirs_ICE5G_data_ALG4_sinkless_downscaled_"
                                                            "ls_mask_0k_upscale_rdirs_20161031_113238_updated.nc",
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            corrected_hd_rdirs_rmouthoutflow_file,
                                                            data_rmouth_outflows_filename=\
                                                            upscaled_rdirs_rmouthoutflow_file,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=50,flip_data=False,
                                                            rotate_data=True,flip_ref=True,rotate_ref=True,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            invert_ls_mask=False,
                                                            first_datasource_name="Reference",
                                                            matching_parameter_set='magnitude_extensive',
                                                            use_single_color_for_discrepancies=True,
                                                            second_datasource_name="Data",grid_type='HD')

    def upscaled_rdirs_with_and_without_tarasov_upscaled_north_america_only_data_ALG4_corr_orog_downscaled_ls_mask_0k_FlowMap_comparison(self):
        """

        Note this was adapted from previous code... not all variable names are accurate
        """

        data_label="20170511_230901"
        ref_label="20170507_135726"
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale'
                                  '_rdirs_{0}_updated.nc'.format(ref_label))
        data_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless'
                                  '_downscaled_ls_mask_0k_upscale_rdirs_' + data_label + '_updated.nc')
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_"
                                     "HD_rdirs_20160504_142435.nc")
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
                                                         "rmouthflows_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs"
                                                         "_{0}_updated.nc".format(ref_label))
        upscaled_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
                                                         "rmouthflows_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_"
                                                         "data_ALG4_sinkless_"
                                                         "downscaled_ls_mask_0k_upscale_rdirs_" + data_label + "_updated.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            "catchmentmap_ICE5G_data_ALG4_"
                                                            "sinkless_downscaled_ls_mask_0k_upscale_rdirs_{0}_updated.nc"\
                                                            .format(ref_label),
                                                            data_catchment_filename="catchmentmap_ICE5G_and_tarasov_upscaled_"
                                                            "srtm30plus_north_america_only_data_ALG4_sinkless_downscaled_ls_"
                                                            "mask_0k_upscale_rdirs_" + data_label + "_updated.nc",
                                                            ref_rdirs_filename="generated/upscaled/upscaled_rdirs_ICE5G_data"
                                                            "_ALG4_sinkless_downscaled_"
                                                            "ls_mask_0k_upscale_rdirs_{0}_updated.nc".format(ref_label),
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            corrected_hd_rdirs_rmouthoutflow_file,
                                                            data_rmouth_outflows_filename=\
                                                            upscaled_rdirs_rmouthoutflow_file,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=50,flip_data=False,
                                                            rotate_data=True,flip_ref=False,rotate_ref=True,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            invert_ls_mask=False,
                                                            first_datasource_name="Reference",
                                                            matching_parameter_set='magnitude_extensive',
                                                            use_single_color_for_discrepancies=True,
                                                            second_datasource_name="Data",grid_type='HD')

    def Upscaled_Rdirs_vs_Corrected_HD_Rdirs_tarasov_upscaled_data_ALG4_corr_orog_downscaled_ls_mask_0k_FlowMap_comparison(self):
        data_label="20170508_021105"
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_data_ALG4_sinkless'
                                  '_downscaled_ls_mask_0k_upscale_rdirs_' + data_label + '_updated.nc')
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_"
                                     "HD_rdirs_20160504_142435.nc")
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
                                                             "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        upscaled_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
                                                         "rmouthflows_ICE5G_and_tarasov_upscaled_srtm30plus_data_ALG4_sinkless_"
                                                         "downscaled_ls_mask_0k_upscale_rdirs_" + data_label + "_updated.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            "catchmentmap_corrected_HD_rdirs_"
                                                            "post_processing_20160427_141158.nc",
                                                            data_catchment_filename="catchmentmap_ICE5G_and_tarasov_upscaled_"
                                                            "srtm30plus_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_"
                                                            "rdirs_" + data_label + "_updated.nc",
                                                            ref_rdirs_filename="rivdir_vs_1_9_data_from_stefan.nc",
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            corrected_hd_rdirs_rmouthoutflow_file,
                                                            data_rmouth_outflows_filename=\
                                                            upscaled_rdirs_rmouthoutflow_file,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=100,flip_data=False,
                                                            rotate_data=True,flip_ref=False,rotate_ref=False,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            invert_ls_mask=False,
                                                            first_datasource_name="Reference",
                                                            matching_parameter_set='extensive',
                                                            rivers_to_plot=[(117,424),(121,176),(179,260),
                                                                            (160,573),(40,90),(217,432),
                                                                            (104,598),(46,504),(252,638),
                                                                            (32,612),(132,494),(171,371),
                                                                            (50,439),(121,456),(40,682),
                                                                            (88,430)],
                                                            rivers_to_plot_alt_color=[(192,384),(82,223),
                                                                                      (249,244),(117,603),
                                                                                      (35,521),(144,548),
                                                                                      (72,641),(54,29),
                                                                                      (88,457),(62,173),
                                                                                      (91,111),(125,165),
                                                                                      (159,235),(237,392),
                                                                                      (36,660),(51,717),
                                                                                      (33,603),(90,418),
                                                                                      (89,482),(111,380)],
                                                            rivers_to_plot_secondary_alt_color=[(64,175),
                                                                                                (42,468),
                                                                                                (32,577),
                                                                                                (43,508),
                                                                                                (117,130),
                                                                                                (230,427),
                                                                                                (36,631),
                                                                                                (86,436),
                                                                                                (55,174),
                                                                                                (82,113),
                                                                                                (60,416),
                                                                                                (154,388),
                                                                                                (136,536),
                                                                                                (201,286)],
                                                            use_single_color_for_discrepancies=True,
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "catch_and_outflow_mods_ice5g_10min_upscaled_"
                                                            "rdirs_vs_modern_day.txt",
                                                            second_datasource_name="Data",grid_type='HD')


    def Upscaled_Rdirs_vs_Corrected_HD_Rdirs_tarasov_upscaled_north_america_only_data_ALG4_corr_orog_downscaled_ls_mask_0k_FlowMap_comparison(self):
        data_label="20170511_230901"
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless'
                                  '_downscaled_ls_mask_0k_upscale_rdirs_' + data_label + '_updated.nc')
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_"
                                     "HD_rdirs_20160504_142435.nc")
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
                                                             "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        upscaled_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
                                                         "rmouthflows_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only"
                                                         "_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs_"
                                                         + data_label + "_updated.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            "catchmentmap_corrected_HD_rdirs_"
                                                            "post_processing_20160427_141158.nc",
                                                            data_catchment_filename="catchmentmap_ICE5G_and_tarasov_upscaled_"
                                                            "srtm30plus_north_america_only_data_ALG4_sinkless_downscaled_ls_"
                                                            "mask_0k_upscale_rdirs_" + data_label + "_updated.nc",
                                                            ref_rdirs_filename="rivdir_vs_1_9_data_from_stefan.nc",
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            corrected_hd_rdirs_rmouthoutflow_file,
                                                            data_rmouth_outflows_filename=\
                                                            upscaled_rdirs_rmouthoutflow_file,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=100,flip_data=False,
                                                            rotate_data=True,flip_ref=False,rotate_ref=False,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            invert_ls_mask=False,
                                                            first_datasource_name="Reference",
                                                            matching_parameter_set='extensive',
                                                            rivers_to_plot=[(117,424),(121,176),(179,260),
                                                                            (160,573),(40,90),(217,432),
                                                                            (104,598),(46,504),(252,638),
                                                                            (32,612),(132,494),(171,371),
                                                                            (50,439),(121,456),(40,682),
                                                                            (88,430)],
                                                            rivers_to_plot_alt_color=[(192,384),(82,223),
                                                                                      (249,244),(117,603),
                                                                                      (35,521),(144,548),
                                                                                      (72,641),(54,29),
                                                                                      (88,457),(62,173),
                                                                                      (91,111),(125,165),
                                                                                      (159,235),(237,392),
                                                                                      (36,660),(51,717),
                                                                                      (33,603),(90,418),
                                                                                      (89,482),(111,380)],
                                                            rivers_to_plot_secondary_alt_color=[(64,175),
                                                                                                (42,468),
                                                                                                (32,577),
                                                                                                (43,508),
                                                                                                (117,130),
                                                                                                (230,427),
                                                                                                (36,631),
                                                                                                (86,436),
                                                                                                (55,174),
                                                                                                (82,113),
                                                                                                (60,416),
                                                                                                (154,388),
                                                                                                (136,536),
                                                                                                (201,286)],
                                                            use_single_color_for_discrepancies=True,
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "catch_and_outflow_mods_ice5g_10min_upscaled_"
                                                            "rdirs_vs_modern_day.txt",
                                                            second_datasource_name="Data",grid_type='HD')

    def Upscaled_Rdirs_vs_Corrected_HD_Rdirs_tarasov_upscaled_north_america_only_data_ALG4_corr_orog_glcc_olson_lsmask_0k_FlowMap_comparison(self):
        data_label="20170517_004128"
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless'
                                  '_glcc_olson_lsmask_0k_upscale_rdirs_' + data_label + '_updated.nc')
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_recreate_connected_HD_lsmask_"
                                     "from_glcc_olson_data_20170513_195421.nc")
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
                                                             "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        upscaled_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
                                                         "rmouthflows_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only"
                                                         "_data_ALG4_sinkless_glcc_olson_lsmask_0k_upscale_rdirs_"
                                                         + data_label + "_updated.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            "catchmentmap_corrected_HD_rdirs_"
                                                            "post_processing_20160427_141158.nc",
                                                            data_catchment_filename="catchmentmap_ICE5G_and_tarasov_upscaled_"
                                                            "srtm30plus_north_america_only_data_ALG4_sinkless_glcc"
                                                            "_olson_lsmask_0k_upscale_rdirs_" + data_label + "_updated.nc",
                                                            ref_rdirs_filename="rivdir_vs_1_9_data_from_stefan.nc",
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            corrected_hd_rdirs_rmouthoutflow_file,
                                                            data_rmouth_outflows_filename=\
                                                            upscaled_rdirs_rmouthoutflow_file,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=100,flip_data=False,
                                                            rotate_data=True,flip_ref=False,rotate_ref=False,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            flip_lsmask=False,rotate_lsmask=True,
                                                            invert_ls_mask=False,
                                                            first_datasource_name="Reference",
                                                            matching_parameter_set='magnitude_extensive',
                                                            additional_truesink_matches_list_filename=\
                                                            "additional_truesink_matches_ice5g_upscaled_"
                                                            "present_with_glcc_lsmask_vs_manual_HD_rdirs.txt",
                                                            rivers_to_plot=[(117,424),(121,176),(179,260),
                                                                            (160,573),(40,90),(217,432),
                                                                            (104,598),(46,504),(252,638),
                                                                            (32,612),(132,494),(171,371),
                                                                            (50,439),(121,456),(40,682),
                                                                            (88,430)],
                                                            rivers_to_plot_alt_color=[(192,384),(82,223),
                                                                                      (249,244),(117,603),
                                                                                      (35,521),(144,548),
                                                                                      (72,641),(54,29),
                                                                                      (88,457),(62,173),
                                                                                      (91,111),(125,165),
                                                                                      (159,235),(237,392),
                                                                                      (36,660),(51,717),
                                                                                      (33,603),(90,418),
                                                                                      (89,482),(111,380)],
                                                            rivers_to_plot_secondary_alt_color=[(64,175),
                                                                                                (42,468),
                                                                                                (32,577),
                                                                                                (43,508),
                                                                                                (117,130),
                                                                                                (230,427),
                                                                                                (36,631),
                                                                                                (86,436),
                                                                                                (55,174),
                                                                                                (82,113),
                                                                                                (60,416),
                                                                                                (154,388),
                                                                                                (136,536),
                                                                                                (201,286)],
                                                            use_single_color_for_discrepancies=True,
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "catch_and_outflow_mods_ice5g_10min_upscaled_"
                                                            "rdirs_vs_modern_day_glcc_olson_lsmask.txt",
                                                            second_datasource_name="Data",grid_type='HD')

    def compare_present_day_and_lgm_river_directions_with_catchments_ICE5G_plus_tarasov_style_orog_corrs_for_both(self):
        """Compare LGM to present using ICE5G data plus tarasov style orography corrections for both times"""
        present_day_data_datetime = "20170521_002051"
        lgm_data_datetime = "20170521_151723"
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  "flowmap_ICE5G_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov"
                                  "_orog_corrs_generation_and_upscaling_{0}_upscaled_updated.nc".\
                                  format(present_day_data_datetime))
        data_filename=os.path.join(self.flow_maps_data_directory,
                                   "flowmap_ICE5G_21k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_"
                                   "tarasov_orog_corrs_generation_and_upscaling_{0}_upscaled_updated.nc".\
                                   format(lgm_data_datetime))
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_ICE5G_21k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs"
                                     "_tarasov_orog_corrs_generation_and_upscaling_{0}_HD_transf.dat".\
                                     format(lgm_data_datetime))
        ref_catchment_filename=("catchmentmap_ICE5G_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_"
                                "tarasov_orog_corrs_generation_and_upscaling_{0}_upscaled_updated.nc".\
                                format(present_day_data_datetime))
        data_catchment_filename=("catchmentmap_ICE5G_21k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs"
                                 "_tarasov_orog_corrs_generation_and_upscaling_{0}_upscaled_updated.nc".\
                                 format(lgm_data_datetime))
        ref_rdirs_filename=("generated/upscaled/upscaled_rdirs_ICE5G_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_"
                            "plus_upscale_rdirs_tarasov_orog_corrs_generation_and_upscaling_{0}_upscaled_"
                            "updated_transf.dat".format(present_day_data_datetime))
        reference_rmouth_outflows_filename=os.path.join(self.rmouth_outflow_data_directory,
                                                        "rmouthflows_ICE5G_0k_ALG4_sinkless_no_true_sinks_oceans_"
                                                        "lsmask_plus_upscale_rdirs_tarasov_orog_corrs"
                                                        "_generation_and_upscaling_{0}_upscaled_updated.nc".\
                                                        format(present_day_data_datetime))
        data_rmouth_outflows_filename=os.path.join(self.rmouth_outflow_data_directory,
                                                   "rmouthflows_ICE5G_21k_ALG4_sinkless_no_true_sinks_oceans_"
                                                   "lsmask_plus_upscale_rdirs_tarasov_orog_corrs_"
                                                   "generation_and_upscaling_{0}_upscaled_updated.nc".\
                                                   format(lgm_data_datetime))
        glacier_mask_filename=os.path.join(self.orog_data_directory,"ice5g_v1_2_21_0k_10min.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            ref_catchment_filename,
                                                            data_catchment_filename=\
                                                            data_catchment_filename,
                                                            ref_rdirs_filename=\
                                                            ref_rdirs_filename,
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            reference_rmouth_outflows_filename,
                                                            data_rmouth_outflows_filename=\
                                                            data_rmouth_outflows_filename,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=100,
                                                            flip_data=False,
                                                            rotate_data=True,
                                                            flip_ref=False,
                                                            rotate_ref=True,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            invert_ls_mask=True,
                                                            first_datasource_name="Present day",
                                                            second_datasource_name="LGM",
                                                            matching_parameter_set='extensive',
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "catch_and_outflow_mods_lgm_vs_present_day.txt",
                                                            additional_matches_list_filename=\
                                                            "additional_matches_10min_upscaled_lgm_vs_present.txt",
                                                            use_single_color_for_discrepancies=True,
                                                            use_only_one_color_for_flowmap=False,
                                                            use_title=False,remove_antartica=True,
                                                            difference_in_catchment_label="Difference",
                                                            glacier_mask_filename=glacier_mask_filename,
                                                            glacier_mask_grid_type='LatLong10min',
                                                            flip_glacier_mask=True,
                                                            rotate_glacier_mask=True,
                                                            grid_type='HD')

    def compare_present_day_and_lgm_river_directions_with_catchments_ICE6G_plus_tarasov_style_orog_corrs_for_both(self):
        """Compare LGM to present using ICE6G data plus tarasov style orography corrections for both times"""
        present_day_data_datetime = "20170612_202721"
        lgm_data_datetime = "20170612_202559"
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  "flowmap_ICE6g_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov"
                                  "_orog_corrs_{0}_upscaled_updated.nc".\
                                  format(present_day_data_datetime))
        data_filename=os.path.join(self.flow_maps_data_directory,
                                   "flowmap_ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_"
                                   "tarasov_orog_corrs_{0}_upscaled_updated.nc".\
                                   format(lgm_data_datetime))
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs"
                                     "_tarasov_orog_corrs_{0}_HD_transf.nc".\
                                     format(lgm_data_datetime))
        extra_lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                           "ls_mask_ICE6g_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale"
                                           "_rdirs_tarasov_orog_corrs_{0}_HD_transf.nc".\
                                           format(present_day_data_datetime))
        ref_catchment_filename=("catchmentmap_ICE6g_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_"
                                "tarasov_orog_corrs_{0}_upscaled_updated.nc".\
                                format(present_day_data_datetime))
        data_catchment_filename=("catchmentmap_ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs"
                                 "_tarasov_orog_corrs_{0}_upscaled_updated.nc".\
                                 format(lgm_data_datetime))
        ref_rdirs_filename=("generated/upscaled/upscaled_rdirs_ICE6g_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_"
                            "plus_upscale_rdirs_tarasov_orog_corrs_{0}_upscaled_"
                            "updated.nc".format(present_day_data_datetime))
        reference_rmouth_outflows_filename=os.path.join(self.rmouth_outflow_data_directory,
                                                        "rmouthflows_ICE6g_0k_ALG4_sinkless_no_true_sinks_oceans_"
                                                        "lsmask_plus_upscale_rdirs_tarasov_orog_corrs"
                                                        "_{0}_upscaled_updated.nc".\
                                            format(present_day_data_datetime))
        data_rmouth_outflows_filename=os.path.join(self.rmouth_outflow_data_directory,
                                                   "rmouthflows_ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_"
                                                   "lsmask_plus_upscale_rdirs_tarasov_orog_corrs_"
                                                   "{0}_upscaled_updated.nc".\
                                                   format(lgm_data_datetime))
        glacier_mask_filename=os.path.join(self.orog_data_directory,"Ice6g_c_VM5a_10min_21k.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            ref_catchment_filename,
                                                            data_catchment_filename=\
                                                            data_catchment_filename,
                                                            ref_rdirs_filename=\
                                                            ref_rdirs_filename,
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            reference_rmouth_outflows_filename,
                                                            data_rmouth_outflows_filename=\
                                                            data_rmouth_outflows_filename,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=75,
                                                            flip_data=False,
                                                            rotate_data=True,
                                                            flip_ref=False,
                                                            rotate_ref=True,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            invert_ls_mask=True,
                                                            first_datasource_name="Present day",
                                                            second_datasource_name="LGM",
                                                            matching_parameter_set='extensive',
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "ice6g_catch_and_outflow_mods_lgm_vs_present_day.txt",
                                                            additional_matches_list_filename=\
                                                            "ice6g_additional_matches_10min_upscaled_lgm_vs_present.txt",
                                                            use_single_color_for_discrepancies=True,
                                                            use_only_one_color_for_flowmap=False,
                                                            use_title=False,remove_antartica=True,
                                                            difference_in_catchment_label="Difference",
                                                            rivers_to_plot=[(216,433),(117,424),(112,380),(146,327),
                                                                            (132,496),(120,176),(251,638),(115,603),
                                                                            (33,571),(34,571),(36,660),(181,256),
                                                                            (120,457),(77,365),(258,235),(167,361),
                                                                            (219,598)],
                                                            rivers_to_plot_alt_color=[(237,393),(192,384),(169,371),
                                                                                      (119,399),(72,640),(126,165),
                                                                                      (87,112),(88,419),(160,237),
                                                                                      (60,35),(147,552),(245,635),
                                                                                      (86,460),(33,603),
                                                                                      (247,243),(41,682),(185,276),
                                                                                      (147,522),(244,612)],
                                                            rivers_to_plot_secondary_alt_color=[(230,427),(170,376),
                                                                                                (180,446),(143,327),
                                                                                                (201,287),(136,538),
                                                                                                (100,467),(116,130),
                                                                                                (160,572),(32,614),
                                                                                                (50,712),(210,619),
                                                                                                (179,445),(212,384),
                                                                                                (261,230),(85,438)],
                                                            glacier_mask_filename=glacier_mask_filename,
                                                            extra_lsmask_filename=extra_lsmask_filename,
                                                            glacier_mask_grid_type='LatLong10min',
                                                            flip_glacier_mask=True,
                                                            rotate_glacier_mask=True,
                                                            grid_type='HD')

    def compare_ICE5G_and_ICE6G_with_catchments_tarasov_style_orog_corrs_for_both(self):
        """Compare LGM to present using ICE6G data plus tarasov style orography corrections for both times"""
        ice5g_datetime = "20170615_174943"
        ice6g_datetime = "20170612_202559"
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  "flowmap_ICE5G_21k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov"
                                  "_orog_corrs_generation_and_upscaling_{0}_upscaled_updated.nc".\
                                  format(ice5g_datetime))
        data_filename=os.path.join(self.flow_maps_data_directory,
                                   "flowmap_ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_"
                                   "tarasov_orog_corrs_{0}_upscaled_updated.nc".\
                                   format(ice6g_datetime))
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs"
                                     "_tarasov_orog_corrs_{0}_HD_transf.nc".\
                                     format(ice6g_datetime))
        ref_catchment_filename=("catchmentmap_ICE5G_21k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_"
                                "tarasov_orog_corrs_generation_and_upscaling_{0}_upscaled_updated.nc".\
                                format(ice5g_datetime))
        data_catchment_filename=("catchmentmap_ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs"
                                 "_tarasov_orog_corrs_{0}_upscaled_updated.nc".\
                                 format(ice6g_datetime))
        ref_rdirs_filename=("generated/upscaled/upscaled_rdirs_ICE5G_21k_ALG4_sinkless_no_true_sinks_oceans_lsmask_"
                            "plus_upscale_rdirs_tarasov_orog_corrs_generation_and_upscaling_{0}_upscaled_"
                            "updated.nc".format(ice5g_datetime))
        reference_rmouth_outflows_filename=os.path.join(self.rmouth_outflow_data_directory,
                                                        "rmouthflows_ICE5G_21k_ALG4_sinkless_no_true_sinks_oceans_"
                                                        "lsmask_plus_upscale_rdirs_tarasov_orog_corrs"
                                                        "_generation_and_upscaling_{0}_upscaled_updated.nc".\
                                                        format(ice5g_datetime))
        data_rmouth_outflows_filename=os.path.join(self.rmouth_outflow_data_directory,
                                                   "rmouthflows_ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_"
                                                   "plus_upscale_rdirs_tarasov_orog_corrs_{0}_upscaled_updated.nc".\
                                                   format(ice6g_datetime))
        #glacier_mask_filename=os.path.join(self.orog_data_directory,"ice5g_v1_2_21_0k_10min.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            ref_catchment_filename,
                                                            data_catchment_filename=\
                                                            data_catchment_filename,
                                                            ref_rdirs_filename=\
                                                            ref_rdirs_filename,
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            reference_rmouth_outflows_filename,
                                                            data_rmouth_outflows_filename=\
                                                            data_rmouth_outflows_filename,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=100,
                                                            flip_data=False,
                                                            rotate_data=True,
                                                            flip_ref=False,
                                                            rotate_ref=True,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            flip_lsmask=False,rotate_lsmask=False,
                                                            invert_ls_mask=True,
                                                            first_datasource_name="ICE5G",
                                                            second_datasource_name="ICE6G",
                                                            matching_parameter_set='extensive',
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "catch_and_outflow_mods_ice6g_vs_ice5g_lgm.txt",
                                                            #additional_matches_list_filename=\
                                                            #"additional_matches_ice6g_vs_ice5g_lgm.txt",
                                                            use_single_color_for_discrepancies=True,
                                                            use_only_one_color_for_flowmap=False,
                                                            use_title=False,remove_antartica=True,
                                                            difference_in_catchment_label="Difference",
                                                            rivers_to_plot=[(216,434),(116,426),(111,385),
                                                                            (127,473),(71,641),(123,175),
                                                                            (147,327),(146,554),(194,590),
                                                                            (25,605),(26,691),(69,31),
                                                                            (178,267),(88,455),(67,369),
                                                                            (253,637),(204,609),(270,238),
                                                                            (39,315),(42,252),
                                                                            (16,444),(56,232)],
                                                            rivers_to_plot_alt_color=[(193,383),(237,392),(170,369),
                                                                                      (118,398),(24,591),(114,612),
                                                                                      (126,167),(87,111),(88,246),
                                                                                      (160,240),(249,255),(91,421),
                                                                                      (197,626),(217,598),(135,496),
                                                                                      (31,112),(14,263),(54,2)],
                                                            rivers_to_plot_secondary_alt_color=[(230,428),(171,377),
                                                                                                (154,446),(100,467),
                                                                                                (142,327),(180,445),
                                                                                                (170,578),(117,131),
                                                                                                (24,622),(34,8),
                                                                                                (75,99),(201,287),
                                                                                                (90,433),(250,632),
                                                                                                (82,349),(138,536),
                                                                                                (29,392),(32,200),
                                                                                                (35,104),(261,246),
                                                                                                (14,320)],
                                                            grid_type='HD')

    def compare_river_directions_with_dynriver_corrs_and_MERIThydro_derived_corrs(self):
        """Compare the orog corrs from the dynamic river papers with the MERIT hydro derived ones"""
        ref_base_dir = ("/Users/thomasriddick/Documents/data/lake_analysis_runs"
                        "/lake_analysis_one_21_Jun_2021")
        data_base_dir = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                         "lake_analysis_two_26_Mar_2022")
        ref_filename=os.path.join(ref_base_dir,
                                  "rivers/results/diag_version_29_date_0",
                                  "30min_flowtocell.nc")
        data_filename=os.path.join(data_base_dir,
                                  "rivers/results/diag_version_0_date_0",
                                  "30min_flowtocell.nc")
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_HD_rdirs_20160504_142435.nc")
        ref_catchment_filename=os.path.join(ref_base_dir,
                                            "rivers/results/diag_version_29_date_0",
                                            "30min_catchments.nc")
        data_catchment_filename=os.path.join(data_base_dir,
                                             "rivers/results/diag_version_0_date_0",
                                             "30min_catchments.nc")
        ref_rdirs_filename=os.path.join(ref_base_dir,
                                        "rivers/results/diag_version_29_date_0",
                                        "30min_rdirs.nc")
        reference_rmouth_outflows_filename=os.path.join(ref_base_dir,
                                                        "rivers/results/diag_version_29_date_0",
                                                        "30min_flowtorivermouths.nc")
        data_rmouth_outflows_filename=os.path.join(data_base_dir,
                                                   "rivers/results/diag_version_0_date_0",
                                                   "30min_flowtorivermouths.nc")
        #glacier_mask_filename=os.path.join(self.orog_data_directory,"ice5g_v1_2_21_0k_10min.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            ref_catchment_filename,
                                                            data_catchment_filename=\
                                                            data_catchment_filename,
                                                            ref_rdirs_filename=\
                                                            ref_rdirs_filename,
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            reference_rmouth_outflows_filename,
                                                            data_rmouth_outflows_filename=\
                                                            data_rmouth_outflows_filename,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=80,
                                                            flip_data=False,
                                                            rotate_data=False,
                                                            flip_ref=False,
                                                            rotate_ref=False,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            flip_lsmask=False,rotate_lsmask=False,
                                                            invert_ls_mask=False,
                                                            first_datasource_name="old",
                                                            second_datasource_name="MERIT derived",
                                                            matching_parameter_set='extensive',
                                                            catchment_and_outflows_mods_list_filename=\
                                                            None,
                                                            #"catch_and_outflow_mods_ice6g_vs_ice5g_lgm.txt",
                                                            #additional_matches_list_filename=\
                                                            #"additional_matches_ice6g_vs_ice5g_lgm.txt",
                                                            use_single_color_for_discrepancies=True,
                                                            use_only_one_color_for_flowmap=False,
                                                            use_title=False,remove_antartica=True,
                                                            difference_in_catchment_label="Difference",
                                                            grid_type='HD')

    def compare_river_directions_with_dynriver_corrs_and_MERIThydro_derived_corrs_original_ts(self):
        """Compare the orog corrs from the paper with the ones derived from MERIThydro including original true sinks"""
        ref_base_dir = ("/Users/thomasriddick/Documents/data/lake_analysis_runs"
                        "/lake_analysis_one_21_Jun_2021")
        data_base_dir = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                         "lake_analysis_two_26_Mar_2022")
        ref_filename=os.path.join(ref_base_dir,
                                  "rivers/results/default_orog_corrs/diag_version_29_date_0_original_truesinks",
                                  "30min_flowtocell.nc")
        data_filename=os.path.join(data_base_dir,
                                  "rivers/results/diag_version_0_date_0_original_truesinks",
                                  "30min_flowtocell.nc")
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_extract_ls_mask_from_corrected_HD_rdirs_20160504_142435.nc")
        ref_catchment_filename=os.path.join(ref_base_dir,
                                            "rivers/results/default_orog_corrs/diag_version_29_date_0_original_truesinks",
                                            "30min_catchments.nc")
        data_catchment_filename=os.path.join(data_base_dir,
                                             "rivers/results/diag_version_0_date_0_original_truesinks",
                                             "30min_catchments.nc")
        ref_rdirs_filename=os.path.join(ref_base_dir,
                                        "rivers/results/default_orog_corrs/"
                                        "diag_version_29_date_0_original_truesinks",
                                        "30min_rdirs.nc")
        reference_rmouth_outflows_filename=os.path.join(ref_base_dir,
                                                        "rivers/results/default_orog_corrs/"
                                                        "diag_version_29_date_0_original_truesinks",
                                                        "30min_rmouth_flowtocell.nc")
        data_rmouth_outflows_filename=os.path.join(data_base_dir,
                                                   "rivers/results/diag_version_0_date_0_original_truesinks",
                                                   "30min_rmouth_flowtocell.nc")
        #glacier_mask_filename=os.path.join(self.orog_data_directory,"ice5g_v1_2_21_0k_10min.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            ref_catchment_filename,
                                                            data_catchment_filename=\
                                                            data_catchment_filename,
                                                            ref_rdirs_filename=\
                                                            ref_rdirs_filename,
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            reference_rmouth_outflows_filename,
                                                            data_rmouth_outflows_filename=\
                                                            data_rmouth_outflows_filename,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=80,
                                                            flip_data=False,
                                                            rotate_data=False,
                                                            flip_ref=False,
                                                            rotate_ref=False,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            flip_lsmask=False,rotate_lsmask=False,
                                                            invert_ls_mask=False,
                                                            first_datasource_name="old",
                                                            second_datasource_name="MERIT derived",
                                                            matching_parameter_set='default',
                                                            catchment_and_outflows_mods_list_filename=\
                                                            None,
                                                            #additional_matches_list_filename=\
                                                            #"additional_matches_ice6g_vs_ice5g_lgm.txt",
                                                            use_single_color_for_discrepancies=True,
                                                            use_only_one_color_for_flowmap=False,
                                                            use_title=False,remove_antartica=True,
                                                            difference_in_catchment_label="Difference",
                                                            grid_type='HD')

    def compare_river_directions_with_dynriver_corrs_and_MERIThydro_derived_corrs_new_ts_10min(self):
        """Compare the orog corrs from the paper with the ones derived from MERIThydro including original true sinks"""
        ref_base_dir = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                        "lake_analysis_one_21_Jun_2021/rivers/results/"
                        "default_orog_corrs/"
                        "diag_version_29_date_0_original_truesinks")
        data_base_dir = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                         "lake_analysis_two_26_Mar_2022/"
                         "rivers/results/diag_version_32_date_0_with_truesinks")
        ref_filename=os.path.join(ref_base_dir,"10min_flowtocell.nc")
        data_filename=os.path.join(data_base_dir,"10min_flowtocell.nc")
        #lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
        #                             "ls_mask_extract_ls_mask_from_corrected_HD_rdirs_20160504_142435.nc")
        lsmask_filename=None
        ref_catchment_filename=os.path.join(ref_base_dir,
                                            "10min_catchments_ext.nc")
        data_catchment_filename=os.path.join(data_base_dir,
                                             "10min_catchments_ext.nc")
        ref_rdirs_filename=os.path.join(ref_base_dir,
                                        "10min_rdirs.nc")
        data_rdirs_filename=os.path.join(data_base_dir,
                                        "10min_rdirs.nc")
        reference_rmouth_outflows_filename=os.path.join(ref_base_dir,
                                                        "10min_rmouth_flowtocell.nc")
        data_rmouth_outflows_filename=os.path.join(data_base_dir,
                                                   "10min_rmouth_flowtocell.nc")
        #glacier_mask_filename=os.path.join(self.orog_data_directory,"ice5g_v1_2_21_0k_10min.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            ref_catchment_filename,
                                                            data_catchment_filename=\
                                                            data_catchment_filename,
                                                            ref_rdirs_filename=\
                                                            ref_rdirs_filename,
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            reference_rmouth_outflows_filename,
                                                            data_rmouth_outflows_filename=\
                                                            data_rmouth_outflows_filename,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=80*9/3,
                                                            flip_data=False,
                                                            rotate_data=False,
                                                            flip_ref=False,
                                                            rotate_ref=False,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            flip_lsmask=False,rotate_lsmask=False,
                                                            invert_ls_mask=False,
                                                            first_datasource_name="old",
                                                            second_datasource_name="MERIT derived",
                                                            matching_parameter_set='minimal',
                                                            additional_matches_list_filename=
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "addmatches/additional_matches_10min_upscaled_"
                                                            "MERIT_rdirs_vs_modern_day_ext.txt",
                                                            additional_truesink_matches_list_filename=
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "addmatches/addmatches_truesinks/"
                                                            "additional_truesinks_matches_10min_upscaled_"
                                                            "MERIT_rdirs_vs_modern_day.txt",
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "catchmods/catch_and_outflow_mods_10min_upscaled_MERIT_rdirs_vs_modern_day.txt",
                                                            use_single_color_for_discrepancies=True,
                                                            use_only_one_color_for_flowmap=False,
                                                            use_title=False,remove_antartica=False,
                                                            difference_in_catchment_label="Difference",
                                                            grid_type='LatLong10min')
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            ref_catchment_filename,
                                                            data_catchment_filename=\
                                                            data_catchment_filename,
                                                            ref_rdirs_filename=\
                                                            ref_rdirs_filename,
                                                            data_rdirs_filename=None,
                                                            reference_rmouth_outflows_filename=\
                                                            reference_rmouth_outflows_filename,
                                                            data_rmouth_outflows_filename=\
                                                            data_rmouth_outflows_filename,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=20*9/3,
                                                            flip_data=False,
                                                            rotate_data=False,
                                                            flip_ref=False,
                                                            rotate_ref=False,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            flip_lsmask=False,rotate_lsmask=False,
                                                            invert_ls_mask=False,
                                                            first_datasource_name="old",
                                                            second_datasource_name="MERIT derived",
                                                            matching_parameter_set='minimal',
                                                            additional_matches_list_filename=
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "addmatches/additional_matches_10min_upscaled_"
                                                            "MERIT_rdirs_vs_modern_day_ext.txt",
                                                            additional_truesink_matches_list_filename=
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "addmatches/addmatches_truesinks/"
                                                            "additional_truesinks_matches_10min_upscaled_"
                                                            "MERIT_rdirs_vs_modern_day.txt",
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "catchmods/catch_and_outflow_mods_10min_upscaled_MERIT_rdirs_vs_modern_day.txt",
                                                            use_single_color_for_discrepancies=True,
                                                            use_only_one_color_for_flowmap=False,
                                                            use_title=False,remove_antartica=False,
                                                            difference_in_catchment_label="Difference",
                                                            grid_type='LatLong10min')
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            ref_catchment_filename,
                                                            data_catchment_filename=\
                                                            data_catchment_filename,
                                                            ref_rdirs_filename=\
                                                            ref_rdirs_filename,
                                                            data_rdirs_filename=\
                                                            data_rdirs_filename,
                                                            reference_rmouth_outflows_filename=\
                                                            reference_rmouth_outflows_filename,
                                                            data_rmouth_outflows_filename=\
                                                            data_rmouth_outflows_filename,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=20*9/3,
                                                            flip_data=False,
                                                            rotate_data=False,
                                                            flip_ref=False,
                                                            rotate_ref=False,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            flip_lsmask=False,rotate_lsmask=False,
                                                            invert_ls_mask=False,
                                                            first_datasource_name="old",
                                                            second_datasource_name="MERIT derived",
                                                            matching_parameter_set='minimal',
                                                            additional_matches_list_filename=
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "addmatches/additional_matches_10min_upscaled_"
                                                            "MERIT_rdirs_vs_modern_day_ext.txt",
                                                            additional_truesink_matches_list_filename=
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "addmatches/addmatches_truesinks/"
                                                            "additional_truesinks_matches_10min_upscaled_"
                                                            "MERIT_rdirs_vs_modern_day.txt",
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "catchmods/catch_and_outflow_mods_10min_upscaled_MERIT_rdirs_vs_modern_day.txt",
                                                            use_single_color_for_discrepancies=True,
                                                            use_only_one_color_for_flowmap=False,
                                                            use_title=False,remove_antartica=False,
                                                            show_true_sinks=True,
                                                            difference_in_catchment_label="Difference",
                                                            grid_type='LatLong10min')
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            ref_catchment_filename,
                                                            data_catchment_filename=\
                                                            data_catchment_filename,
                                                            ref_rdirs_filename=\
                                                            ref_rdirs_filename,
                                                            data_rdirs_filename=\
                                                            None,
                                                            reference_rmouth_outflows_filename=\
                                                            reference_rmouth_outflows_filename,
                                                            data_rmouth_outflows_filename=\
                                                            data_rmouth_outflows_filename,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=5*9/3,
                                                            flip_data=False,
                                                            rotate_data=False,
                                                            flip_ref=False,
                                                            rotate_ref=False,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            flip_lsmask=False,rotate_lsmask=False,
                                                            invert_ls_mask=False,
                                                            first_datasource_name="old",
                                                            second_datasource_name="MERIT derived",
                                                            matching_parameter_set='minimal',
                                                            additional_matches_list_filename=
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "addmatches/additional_matches_10min_upscaled_"
                                                            "MERIT_rdirs_vs_modern_day_ext.txt",
                                                            additional_truesink_matches_list_filename=
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "addmatches/addmatches_truesinks/"
                                                            "additional_truesinks_matches_10min_upscaled_"
                                                            "MERIT_rdirs_vs_modern_day.txt",
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "catchmods/catch_and_outflow_mods_10min_upscaled_MERIT_rdirs_vs_modern_day.txt",
                                                            use_single_color_for_discrepancies=True,
                                                            use_only_one_color_for_flowmap=False,
                                                            use_title=False,remove_antartica=False,
                                                            show_true_sinks=False,
                                                            difference_in_catchment_label="Difference",
                                                            grid_type='LatLong10min')
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            ref_catchment_filename,
                                                            data_catchment_filename=\
                                                            data_catchment_filename,
                                                            ref_rdirs_filename=\
                                                            ref_rdirs_filename,
                                                            data_rdirs_filename=\
                                                            data_rdirs_filename,
                                                            reference_rmouth_outflows_filename=\
                                                            reference_rmouth_outflows_filename,
                                                            data_rmouth_outflows_filename=\
                                                            data_rmouth_outflows_filename,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=5*9/3,
                                                            flip_data=False,
                                                            rotate_data=False,
                                                            flip_ref=False,
                                                            rotate_ref=False,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            flip_lsmask=False,rotate_lsmask=False,
                                                            invert_ls_mask=False,
                                                            first_datasource_name="old",
                                                            second_datasource_name="MERIT derived",
                                                            matching_parameter_set='minimal',
                                                            additional_matches_list_filename=
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "addmatches/additional_matches_10min_upscaled_"
                                                            "MERIT_rdirs_vs_modern_day_ext.txt",
                                                            additional_truesink_matches_list_filename=
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "addmatches/addmatches_truesinks/"
                                                            "additional_truesinks_matches_10min_upscaled_"
                                                            "MERIT_rdirs_vs_modern_day.txt",
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "catchmods/catch_and_outflow_mods_10min_upscaled_MERIT_rdirs_vs_modern_day.txt",
                                                            use_single_color_for_discrepancies=True,
                                                            use_only_one_color_for_flowmap=False,
                                                            use_title=False,remove_antartica=False,
                                                            show_true_sinks=True,
                                                            difference_in_catchment_label="Difference",
                                                            grid_type='LatLong10min')
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            ref_catchment_filename,
                                                            data_catchment_filename=\
                                                            data_catchment_filename,
                                                            ref_rdirs_filename=\
                                                            ref_rdirs_filename,
                                                            data_rdirs_filename=\
                                                            data_rdirs_filename,
                                                            reference_rmouth_outflows_filename=\
                                                            reference_rmouth_outflows_filename,
                                                            data_rmouth_outflows_filename=\
                                                            data_rmouth_outflows_filename,
                                                            lsmask_filename=lsmask_filename,
                                                            minflowcutoff=2*9/3,
                                                            flip_data=False,
                                                            rotate_data=False,
                                                            flip_ref=False,
                                                            rotate_ref=False,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            flip_lsmask=False,rotate_lsmask=False,
                                                            invert_ls_mask=False,
                                                            first_datasource_name="old",
                                                            second_datasource_name="MERIT derived",
                                                            matching_parameter_set='minimal',
                                                            additional_matches_list_filename=
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "addmatches/additional_matches_10min_upscaled_"
                                                            "MERIT_rdirs_vs_modern_day_ext.txt",
                                                            additional_truesink_matches_list_filename=
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "addmatches/addmatches_truesinks/"
                                                            "additional_truesinks_matches_10min_upscaled_"
                                                            "MERIT_rdirs_vs_modern_day.txt",
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "/Users/thomasriddick/Documents/data/HDdata/"
                                                            "catchmods/catch_and_outflow_mods_10min_upscaled_MERIT_rdirs_vs_modern_day.txt",
                                                            use_single_color_for_discrepancies=True,
                                                            use_only_one_color_for_flowmap=False,
                                                            use_title=False,remove_antartica=False,
                                                            show_true_sinks=True,
                                                            difference_in_catchment_label="Difference",
                                                            grid_type='LatLong10min')
class OrographyPlots(Plots):
    """A general base class for orography plots"""

    orography_path_extension = 'orographys'

    def __init__(self,save,color_palette_to_use='default'):
        """Class constructor"""
        super(OrographyPlots,self).__init__(save,color_palette_to_use)
        self.orography_data_directory = os.path.join(self.hd_data_path,self.orography_path_extension)

class SimpleOrographyPlots(OrographyPlots):

    def __init__(self,save,color_palette_to_use='default'):
        """Class constructor."""
        super(SimpleOrographyPlots,self).__init__(save,color_palette_to_use)

    def SimpleArrayPlotHelper(self,filename):
        """Assists the creation of simple array plots"""
        #levels = np.linspace(-100.0, 9900.0, 100, endpoint=True)
        plt.figure()
        #plt.contourf(orography_field,levels)
        plt.colorbar()
        pts.invert_y_axis()

class Ice5GComparisonPlots(OrographyPlots):
    """Handles generation Ice5G data comparison plots"""

    def __init__(self,save,use_old_data=False,color_palette_to_use='default'):
        """Class constructor. Sets filename (to point to either old or new data)"""
        super(Ice5GComparisonPlots,self).__init__(save,color_palette_to_use)
        print("Comparing the Modern and LGM Ice-5G 5-minute resolution orography datasets")

        if use_old_data:
            #The data Uwe gave me; this is possibly an older version
            modern_ice_5g_filename = self.orography_data_directory +"/ice5g_0k_5min.nc"
            lgm_ice_5g_filename = self.orography_data_directory + "/ice5g_21k_5min.nc"
            modern_ice_5g_field = iohlpr.NetCDF4FileIOHelper.load_field(modern_ice_5g_filename, 'LatLong5min')
            lgm_ice_5g_field = iohlpr.NetCDF4FileIOHelper.load_field(lgm_ice_5g_filename, 'LatLong5min')
        else:
            #The latest version of the data from the ICE5G website
            modern_ice_5g_filename = self.orography_data_directory +"/ice5g_v1_2_00_0k_10min.nc"
            lgm_ice_5g_filename = self.orography_data_directory + "/ice5g_v1_2_21_0k_10min.nc"
            modern_ice_5g_field = iohlpr.NetCDF4FileIOHelper.load_field(modern_ice_5g_filename, 'LatLong10min')
            lgm_ice_5g_field = iohlpr.NetCDF4FileIOHelper.load_field(lgm_ice_5g_filename, 'LatLong10min')

        self.difference_in_ice_5g_orography = lgm_ice_5g_field - modern_ice_5g_field
        if use_old_data:
            #Remove antartica
            self.difference_in_ice_5g_orography = self.difference_in_ice_5g_orography[275:,:]

    def plotLine(self):
        """Contour plot comparing the Modern and LGM Ice-5G 5-minute resolution orography datasets"""
        minc = 0
        maxc = 500
        num = 500
        levels = np.linspace(minc,maxc,num+1)
        title = textwrap.dedent("""\
        Orography difference between LGM and Modern ICE-5G data
        using {0} meter contour interval""").format((maxc-minc)/num)
        plt.figure()
        plt.contour(self.difference_in_ice_5g_orography,levels=levels)
        plt.title(title)
        pts.remove_ticks()
        #if self.save:
            #plt.savefig('something')
        print("Line contour plot created")

    def plotFilled(self):
        """Filled contour plot comparing the Modern and LGM Ice-5G 5-minute resolution orography datasets"""
        minc = 70
        maxc = 120
        num  = 25
        levels = np.linspace(minc,maxc,num+1)
        title = "Orography difference between LGM and Modern ICE-5G data"
        plt.figure()
        plt.contourf(self.difference_in_ice_5g_orography,levels=levels)
        plt.title(title)
        pts.remove_ticks()
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Orography difference in meters')
        #if self.save:
            #plt.savefig('something')
        print("Filled contour plot created")

    def plotCombined(self):
        """Basic combined plot"""
        self.CombinedPlotHelper()

    def plotCombinedIncludingOceanFloors(self):
        """Combined plot with extended range of levels to include the ocean floor"""
        self.CombinedPlotHelper(minc=70,maxc=170,num=50)

    def CombinedPlotHelper(self,minc=70,maxc=120,num=25):
        """Combined filled and line contour plots of orography difference between LGM and Modern ICE-5G data"""
        levels = np.linspace(minc,maxc,num+1)
        title = textwrap.dedent("""\
        Orography difference between LGM and Modern ICE-5G data
        using {0} meter contour interval""").format((maxc-minc)/num)
        plt.figure()
        ax = plt.subplot(111)
        contourset = plt.contourf(self.difference_in_ice_5g_orography,
                                  levels=levels,hold=True)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Orography difference in meters')
        plt.contour(self.difference_in_ice_5g_orography,levels=contourset.levels,
                    colors='black',hold=True)
        ufcntr = plt.contourf(self.difference_in_ice_5g_orography,
                     levels=[np.min(self.difference_in_ice_5g_orography),minc],
                     colors='white',
                     hatches=['/'],hold=True)
        ofcntr = plt.contourf(self.difference_in_ice_5g_orography,
                     levels=[maxc,np.max(self.difference_in_ice_5g_orography)],
                     colors='white',
                     hatches=['\\'],hold=True)
        ufartists,uflabels = ufcntr.legend_elements() #@UnusedVariable
        ofartists,oflabels = ofcntr.legend_elements() #@UnusedVariable
        uflabels=['Difference $\\leq {0}$'.format(minc)]
        oflabels=['${0} <$ Difference'.format(maxc)]
        artists = ufartists + ofartists
        labels  = uflabels + oflabels
        plt.title(title)
        pts.remove_ticks()
        axbounds = ax.get_position()
        #Shrink box by 5%
        ax.set_position([axbounds.x0,axbounds.y0 + axbounds.height*0.05,
                         axbounds.width,axbounds.height*0.95])
        ax.legend(artists,labels,loc='upper center',
                  bbox_to_anchor=(0.5,-0.025),fancybox=True,ncol=2)
        #if self.save:
            #plt.savefig('something')
        print("Combined plot created")

class LakePlots(Plots):

  ls_masks_extension       = 'lsmasks'
  basin_catchment_nums_extension = "basin_catchment_numbers"
  glacier_data_extension= 'orographys'

  def __init__(self,save=False,color_palette_to_use='default'):
      self.ls_masks_data_directory= os.path.join(self.hd_data_path,
                                                 self.ls_masks_extension)
      self.basin_catchment_nums_directory = os.path.join(self.hd_data_path,
                                                         self.basin_catchment_nums_extension)
      self.glacier_data_directory = os.path.join(self.hd_data_path,
                                                 self.glacier_data_extension)

  def plotLakeDepth(self,ax,timeslice):
    timestamps = {1850:"20190211_131542",
                  1800:"20190211_131517",
                  1750:"20190211_131429",
                  1700:"20190211_131345",
                  1650:"20190211_131301",
                  1600:"20190211_131212",
                  1550:"20190211_131126",
                  1500:"20190211_131041",
                  1450:"20190211_130957",
                  1400:"20190211_130918"}
    times = {1850:7500,
             1800:8000,
             1750:8500,
             1700:9000,
             1650:9500,
             1600:10000,
             1550:10500,
             1500:11000,
             1450:11500,
             1400:12000}
    depth = \
      iodriver.advanced_field_loader(filename=
                                     "/Users/thomasriddick/Downloads/updated_orog_{}_lake_"
                                     "basins_prepare_orography_{}.nc".format(timeslice,timestamps[timeslice]),
                                     time_slice=None,
                                     fieldname="depth",
                                     adjust_orientation=True)
    lsmask = field.Field(iohlpr.NetCDF4FileIOHelper.
                         load_field("/Users/thomasriddick/Documents/data/HDdata/lsmasks/10min_lsmask_pmu0178_merged.nc",
                                    grid_type='LatLong10min',
                                    timeslice=timeslice),grid='LatLong10min')
    lsmask.rotate_field_by_a_hundred_and_eighty_degrees()
    lsmask.flip_data_ud()
    depth.flip_data_ud()
    depth_masked = np.ma.MaskedArray(data=depth.get_data(),
                                     mask=lsmask.get_data())
    ax.set_title("{} BP".format(times[timeslice]))
    cs = ax.contourf(depth_masked[800:-170,450:650],
                levels=np.linspace(0,700,25),
                cmap="Blues")
    ax.set_xticks(np.array([90]))
    ax.set_xticklabels(["90$^{\circ}$ E"])
    ax.set_yticks(np.array([10]))
    ax.set_yticklabels(["45$^{\circ}$ N"])
    return cs

  def plotLakeDepths(self):
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(3, 2,height_ratios=[4,4, 1])
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    ax_cb = plt.subplot(gs[2, :])
    self.plotLakeDepth(ax1,1450)
    self.plotLakeDepth(ax2,1550)
    self.plotLakeDepth(ax3,1650)
    cs = self.plotLakeDepth(ax4,1750)
    fig.colorbar(cs,cax=ax_cb,orientation="horizontal")
    adddaf_cb.set_xlabel("Potential Lake Depth (m)")
    fig.tight_layout()


  def TwoColourRiverAndLakePlotHelper(self,river_flow_filename,
                                      lsmask_filename,
                                      basin_catchment_num_filename,
                                      lake_data_filename,
                                      glacier_data_filename,
                                      flip_river_data=False,
                                      flip_mask=False,
                                      flip_catchment_nums=False,
                                      flip_lake_data=False,
                                      flip_glacier_mask=False,
                                      rotate_lsmask=False,
                                      rotate_catchment_nums=False,
                                      rotate_lake_data=False,
                                      rotate_glacier_mask=False,
                                      minflowcutoff=1000000000000.0,
                                      lake_grid_type='LatLong10min',
                                      lake_kwargs={},
                                      river_grid_type='HD',**river_kwargs):
    """Help produce a map of river flow, lakes and potential lakes"""
    river_flow_object = iodriver.load_field(river_flow_filename,
                                            file_type=iodriver.\
                                              get_file_extension(river_flow_filename),
                                            field_type='Generic',
                                            grid_type=river_grid_type,**river_kwargs)
    lsmask_field = iodriver.load_field(lsmask_filename,
                                       file_type=iodriver.get_file_extension(lsmask_filename),
                                       field_type='Generic',grid_type=lake_grid_type,
                                       **lake_kwargs)
    basin_catchment_nums_field = iodriver.load_field(basin_catchment_num_filename,
                                                     file_type=iodriver.\
                                                     get_file_extension\
                                                     (basin_catchment_num_filename),
                                                     field_type='Generic',
                                                     grid_type=lake_grid_type,
                                                     **lake_kwargs)
    lake_data_field = iodriver.load_field(lake_data_filename,
                                          file_type=iodriver.\
                                          get_file_extension(lake_data_filename),
                                          field_type='Generic',
                                          grid_type=lake_grid_type,
                                          **lake_kwargs)
    glacier_mask_field = iodriver.load_field(glacier_data_filename,
                                             file_type=iodriver.\
                                             get_file_extension(glacier_data_filename),
                                             fieldname="ICEM",
                                             field_type='Generic',
                                             grid_type=lake_grid_type,
                                             **lake_kwargs)
    if flip_river_data:
        river_flow_object.flip_data_ud()
    if flip_mask:
        lsmask_field.flip_data_ud()
    if flip_catchment_nums:
        basin_catchment_nums_field.flip_data_ud()
    if flip_lake_data:
        lake_data_field.flip_data_ud()
    if flip_glacier_mask:
        glacier_mask_field.flip_data_ud()
    if rotate_catchment_nums:
        basin_catchment_nums_field.rotate_field_by_a_hundred_and_eighty_degrees()
    if rotate_lake_data:
        lake_data_field.rotate_field_by_a_hundred_and_eighty_degrees()
    if rotate_lsmask:
        lsmask_field.rotate_field_by_a_hundred_and_eighty_degrees()
    if rotate_glacier_mask:
        glacier_mask_field.rotate_field_by_a_hundred_and_eighty_degrees()
    lsmask = lsmask_field.get_data()
    basin_catchment_nums = basin_catchment_nums_field.get_data()
    river_flow = river_flow_object.get_data()
    rivers_and_lakes_field = copy.deepcopy(river_flow_object)
    rivers_and_lakes = rivers_and_lakes_field.get_data()
    lake_data = lake_data_field.get_data()
    glacier_mask = glacier_mask_field.get_data()
    plt.figure()
    plt.subplot(111)
    rivers_and_lakes[river_flow < minflowcutoff] = 1
    rivers_and_lakes[river_flow >= minflowcutoff] = 2
    fine_rivers_and_lakes_field = utilities.downscale_ls_mask(rivers_and_lakes_field,
                                                              lake_grid_type,
                                                              **lake_kwargs)
    fine_rivers_and_lakes = fine_rivers_and_lakes_field.get_data()
    fine_rivers_and_lakes[basin_catchment_nums > 0 ] = 3
    fine_rivers_and_lakes[lake_data > 0] = 4
    fine_rivers_and_lakes[glacier_mask == 1] = 5
    fine_rivers_and_lakes[lsmask == 1] = 0
    cmap = mpl.colors.ListedColormap(['blue','peru','black','green','red','white'])
    bounds = list(range(7))
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    plt.imshow(fine_rivers_and_lakes,cmap=cmap,norm=norm,interpolation="none")
    plt.title('Cells with cumulative flow greater than or equal to {0}'.format(minflowcutoff))

  def TwoColourRiverAndLakeAnimationHelper(self,
                                           river_flow_file_basename,
                                           lsmask_file_basename,
                                           basin_catchment_num_file_basename,
                                           lake_data_file_basename,
                                           glacier_data_file_basename,
                                           river_flow_fieldname,
                                           lsmask_fieldname,
                                           basin_catchment_num_fieldname,
                                           lake_data_fieldname,
                                           glacier_fieldname,
                                           catchment_nums_file_basename=None,
                                           catchment_nums_fieldname=None,
                                           rdirs_file_basename=None,
                                           rdirs_fieldname=None,
                                           minflowcutoff=1000000000000.0,
                                           zoomed=False,
                                           zoom_section_bounds={}):
    """Help produce a map of river flow, lakes and potential lakes"""
    fig = plt.figure()
    plt.subplot(111)
    cmap = mpl.colors.ListedColormap(['darkblue','peru','black','cyan','blue','white','purple','darkred','slategray'])
    bounds = list(range(10))
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    plt.title('Lakes and rivers with flow greater than {0} m3/s'.format(minflowcutoff))
    ims = []
    show_slices = [14600,13500,12800,12330,11500,11300]
    #show_slices = [15990]
    for time in range(15990,11000,-10):
      mpiesm_time = 3000 + 16000 - time
      show_snapshot = True if time in show_slices else False
      date_text = fig.text(0.4,0.075,"{} YBP".format(time))
      ims.append([self.TwoColourRiverAndLakeAnimationHelperSliceGenerator(cmap=cmap,norm=norm,
                                                                          river_flow_filename=
                                                                          river_flow_file_basename.replace("DATETIME",str(mpiesm_time)),
                                                                          lsmask_filename=
                                                                          lsmask_file_basename.replace("DATETIME",str(time)),
                                                                          basin_catchment_num_filename=
                                                                          basin_catchment_num_file_basename.replace("DATETIME",str(time)),
                                                                          lake_data_filename=
                                                                          lake_data_file_basename.replace("DATETIME",str(mpiesm_time)),
                                                                          glacier_data_filename=
                                                                          glacier_data_file_basename.replace("DATETIME",str(time)),
                                                                          river_flow_fieldname=river_flow_fieldname,
                                                                          lsmask_fieldname=lsmask_fieldname,
                                                                          basin_catchment_num_fieldname=
                                                                          basin_catchment_num_fieldname,
                                                                          lake_data_fieldname=
                                                                          lake_data_fieldname,
                                                                          glacier_fieldname=
                                                                          glacier_fieldname,
                                                                          catchment_nums_filename=
                                                                          catchment_nums_file_basename.replace("DATETIME",str(time)) if
                                                                          catchment_nums_file_basename is not None else None,
                                                                          catchment_nums_fieldname=
                                                                          catchment_nums_fieldname,
                                                                          rdirs_filename=
                                                                          rdirs_file_basename.replace("DATETIME",str(time)) if
                                                                          rdirs_file_basename is not None else None,
                                                                          rdirs_fieldname=
                                                                          rdirs_fieldname,
                                                                          minflowcutoff=minflowcutoff,
                                                                          zoomed=zoomed,
                                                                          zoom_section_bounds=
                                                                          zoom_section_bounds,
                                                                          show_snapshot=show_snapshot),
                                                                          date_text])
    anim = animation.ArtistAnimation(fig,ims,interval=200,blit=False,repeat_delay=500)
    plt.show()
    #writer = animation.writers['ffmpeg'](fps=7,bitrate=1800)
    #anim.save('/Users/thomasriddick/Desktop/deglac.mp4',writer=writer,dpi=1000)

  def TwoColourRiverAndLakeAnimationHelperSliceGenerator(self,cmap,norm,
                                                         river_flow_filename,
                                                         lsmask_filename,
                                                         basin_catchment_num_filename,
                                                         lake_data_filename,
                                                         glacier_data_filename,
                                                         river_flow_fieldname,
                                                         lsmask_fieldname,
                                                         basin_catchment_num_fieldname,
                                                         lake_data_fieldname,
                                                         glacier_fieldname,
                                                         catchment_nums_filename=None,
                                                         catchment_nums_fieldname=None,
                                                         rdirs_filename=None,
                                                         rdirs_fieldname=None,
                                                         minflowcutoff=1000000000000.0,
                                                         zoomed=False,
                                                         zoom_section_bounds={},
                                                         show_snapshot=False):
    river_flow_object = iodriver.advanced_field_loader(river_flow_filename,
                                                       fieldname=river_flow_fieldname,
                                                       field_type='Generic',
                                                       time_slice=5)
    lsmask_field = iodriver.advanced_field_loader(lsmask_filename,
                                                  fieldname=lsmask_fieldname,
                                                  field_type='Generic')
    basin_catchment_nums_field = iodriver.advanced_field_loader(basin_catchment_num_filename,
                                                  fieldname=basin_catchment_num_fieldname,
                                                  field_type='Generic')
    lake_data_field = iodriver.advanced_field_loader(lake_data_filename,
                                                     fieldname=lake_data_fieldname,
                                                     field_type='Generic',
                                                     time_slice=5)
    glacier_mask_field = iodriver.advanced_field_loader(glacier_data_filename,
                                                        fieldname=glacier_fieldname,
                                                        field_type='Generic')
    if catchment_nums_filename is not None:
      catchment_nums_field = iodriver.advanced_field_loader(catchment_nums_filename,
                                                            fieldname=catchment_nums_fieldname,
                                                            field_type='Generic')
    if rdirs_filename is not None:
      rdirs_field = iodriver.advanced_field_loader(rdirs_filename,
                                                   fieldname=rdirs_fieldname,
                                                   field_type='Generic')
    lsmask = lsmask_field.get_data()
    basin_catchment_nums = basin_catchment_nums_field.get_data()
    river_flow = river_flow_object.get_data()
    rivers_and_lakes_field = copy.deepcopy(river_flow_object)
    rivers_and_lakes = rivers_and_lakes_field.get_data()
    lake_data = lake_data_field.get_data()
    glacier_mask = glacier_mask_field.get_data()
    rivers_and_lakes[river_flow < minflowcutoff] = 1
    rivers_and_lakes[river_flow >= minflowcutoff] = 2
    fine_rivers_and_lakes_field = utilities.downscale_ls_mask(rivers_and_lakes_field,
                                                              "LatLong10min")
    fine_rivers_and_lakes = fine_rivers_and_lakes_field.get_data()
    if (rdirs_filename is not None) and (catchment_nums_filename is not None):
        ocean_basin_catchments = pts.find_ocean_basin_catchments(rdirs_field,catchment_nums_field,areas=[{'min_lat':70,
                                                                                                          'max_lat':115,
                                                                                                          'min_lon':195,
                                                                                                          'max_lon':265},
                                                                                                          {'min_lat':110,
                                                                                                          'max_lat':135,
                                                                                                          'min_lon':162,
                                                                                                          'max_lon':196},
                                                                                                          {'min_lat':15,
                                                                                                          'max_lat':50,
                                                                                                          'min_lon':0,
                                                                                                          'max_lon':175}])
        fine_ocean_basin_catchments_field = utilities.downscale_ls_mask(ocean_basin_catchments,
                                                                        "LatLong10min")
        fine_ocean_basin_catchments= fine_ocean_basin_catchments_field.get_data()
        fine_rivers_and_lakes[np.logical_and(fine_rivers_and_lakes == 1,
                                             fine_ocean_basin_catchments == 1)] = 6
        fine_rivers_and_lakes[np.logical_and(fine_rivers_and_lakes == 1,
                                             fine_ocean_basin_catchments == 2)] = 7
        fine_rivers_and_lakes[np.logical_and(fine_rivers_and_lakes == 1,
                                             fine_ocean_basin_catchments == 3)] = 8
    fine_rivers_and_lakes[basin_catchment_nums > 0 ] = 3
    fine_rivers_and_lakes[lake_data > 0] = 4
    fine_rivers_and_lakes[glacier_mask == 1] = 5
    fine_rivers_and_lakes[lsmask == 1] = 0
    if show_snapshot:
        plt.clf()
    if zoomed:
        im = plt.imshow(fine_rivers_and_lakes[zoom_section_bounds["min_lat"]:
                                              zoom_section_bounds["max_lat"]+1,
                                              zoom_section_bounds["min_lon"]:
                                              zoom_section_bounds["max_lon"]+1],
                        cmap=cmap,norm=norm,interpolation="none")
    else:
        im = plt.imshow(fine_rivers_and_lakes,cmap=cmap,norm=norm,interpolation="none")
    if show_snapshot:
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
        plt.show()
    return im

  def LakeAndRiverMap(self):
    timeinslice="18250"
    timeslice="1400"
    timeslice_creation_date_time="20190925_225029"
    river_flow_filename="/Users/thomasriddick/Documents/data/temp/transient_sim_1/river_model_results_3650.nc"
    lsmask_filename=os.path.join(self.ls_masks_data_directory,'generated',
                                 "ls_mask_prepare_basins_from_glac1D_"
                                 + timeslice_creation_date_time+ "_"
                                 + timeslice + "_orig.nc")
    basin_catchment_num_filename=os.path.join(self.basin_catchment_nums_directory,
                                              "basin_catchment_numbers_prepare_basins"
                                              "_from_glac1D_" + timeslice_creation_date_time +
                                              "_" + timeslice + ".nc")
    lake_data_filename="/Users/thomasriddick/Documents/data/temp/transient_sim_1/lake_model_results_3650.nc"
    glacier_filename=os.path.join(self.glacier_data_directory,
                                  "GLAC1D_ICEM_10min_ts"+ timeslice+ ".nc")
    self.TwoColourRiverAndLakePlotHelper(river_flow_filename,
                                         lsmask_filename,
                                         basin_catchment_num_filename,
                                         lake_data_filename,
                                         glacier_filename,
                                         river_flow_fieldname,
                                         lsmask_fieldname,
                                         basin_catchment_num_fieldname,
                                         lake_data_fieldname,
                                         glacier_fieldname,
                                         minflowcutoff=500000000.0,
                                         lake_grid_type='LatLong10min',
                                         river_grid_type='HD')

  def LakeAndRiverMaps(self):
    river_flow_file_basename=("/Users/thomasriddick/Documents/data/lake_transient_data/run_1/"
                              "pmt0531_Tom_lake_16k_DATETIME01.01_hd_higres_ym.nc")
    lsmask_file_basename=("/Users/thomasriddick/Documents/data/lake_transient_data/run_1/"
                          "10min_slm_DATETIME.nc")
    basin_catchment_num_file_basename=("/Users/thomasriddick/Documents/data/lake_transient_data/run_1/"
                                       "lake_numbers_DATETIME.nc")
    lake_data_file_basename=("/Users/thomasriddick/Documents/data/lake_transient_data/run_1/"
                             "pmt0531_Tom_lake_16k_DATETIME01.01_diagnostic_lake_volumes.nc")
    glacier_file_basename=("/Users/thomasriddick/Documents/data/lake_transient_data/run_1/"
                           "10min_glac_DATETIMEk.nc")
    rdirs_file_basename=("/Users/thomasriddick/Documents/data/lake_transient_data/run_1/"
                         "hdpara_DATETIMEk.nc")
    catchment_nums_file_basename=("/Users/thomasriddick/Documents/data/lake_transient_data/run_1/"
                                  "connected_catchments_DATETIME.nc")
    river_flow_fieldname = "friv"
    lsmask_fieldname = "slm"
    basin_catchment_num_fieldname = "lake_number"
    lake_data_fieldname = "diagnostic_lake_vol"
    glacier_fieldname = "glac"
    rdirs_fieldname="FDIR"
    catchment_nums_fieldname="catchments"
    self.TwoColourRiverAndLakeAnimationHelper(river_flow_file_basename,
                                              lsmask_file_basename,
                                              basin_catchment_num_file_basename,
                                              lake_data_file_basename,
                                              glacier_file_basename,
                                              river_flow_fieldname,
                                              lsmask_fieldname,
                                              basin_catchment_num_fieldname,
                                              lake_data_fieldname,
                                              glacier_fieldname,
                                              catchment_nums_file_basename=
                                              catchment_nums_file_basename,
                                              catchment_nums_fieldname=
                                              catchment_nums_fieldname,
                                              rdirs_file_basename=
                                              rdirs_file_basename,
                                              rdirs_fieldname=
                                              rdirs_fieldname,
                                              minflowcutoff=1000.0,
                                              zoomed=True,
                                              zoom_section_bounds={"min_lat":50,
                                                                   "max_lat":500,
                                                                   "min_lon":100,
                                                                   "max_lon":800})

def main():
    """Top level function; define some overarching options and which plots to create"""
    save = False
    show = True

    #hd_parameter_plots =  HDparameterPlots(save=save)
    #hd_parameter_plots.flow_parameter_distribution_for_non_lake_cells_for_current_HD_model()
    #hd_parameter_plots.flow_parameter_distribution_current_HD_model_for_current_HD_model_reprocessed_without_lakes_and_wetlands()
    #hd_parameter_plots.flow_parameter_distribution_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs()
    #hd_parameter_plots.flow_parameter_distribution_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_no_tuning()
    #ice5g_comparison_plots = Ice5GComparisonPlots(save=save)
    #ice5g_comparison_plots.plotLine()
    #ice5g_comparison_plots.plotFilled()
    #ice5g_comparison_plots.plotCombined()
    #ice5g_comparison_plots.plotCombinedIncludingOceanFloors()
    #flowmapplot = FlowMapPlots(save)
    #flowmapplot.FourFlowMapSectionsFromDeglaciation()
    #flowmapplot.Etopo1FlowMap()
    #flowmapplot.ICE5G_data_all_points_0k()
    #flowmapplot.ICE5G_data_all_points_0k_no_sink_filling()
    #flowmapplot.ICE5G_data_all_points_0k_alg4_two_colour()
    #flowmapplot.ICE5G_data_all_points_21k_alg4_two_colour()
    #flowmapplot.Etopo1FlowMap_two_colour()
    #flowmapplot.Etopo1FlowMap_two_colour_directly_upscaled_fields()
    #flowmapplot.Corrected_HD_Rdirs_FlowMap_two_colour()
    #flowmapplot.ICE5G_data_ALG4_true_sinks_21k_And_ICE5G_data_ALG4_true_sinks_0k_FlowMap_comparison()
    #flowmapplot.Corrected_HD_Rdirs_And_Etopo1_ALG4_sinkless_directly_upscaled_fields_FlowMap_comparison()
    #flowmapplot.Corrected_HD_Rdirs_And_Etopo1_ALG4_true_sinks_directly_upscaled_fields_FlowMap_comparison()
    #flowmapplot.Corrected_HD_Rdirs_And_ICE5G_data_ALG4_sinkless_0k_directly_upscaled_fields_FlowMap_comparison()
    #flowmapplot.Corrected_HD_Rdirs_And_ICE5G_data_ALG4_true_sinks_0k_directly_upscaled_fields_FlowMap_comparison()
    #flowmapplot.Corrected_HD_Rdirs_And_ICE5G_data_ALG4_corr_orog_0k_directly_upscaled_fields_FlowMap_comparison()
    #flowmapplot.Corrected_HD_Rdirs_And_ICE5G_data_ALG4_corr_orog_downscaled_ls_mask_0k_directly_upscaled_fields_FlowMap_comparison()
    #flowmapplot.Corrected_HD_Rdirs_And_ICE5G_data_ALG4_no_true_sinks_corr_orog_0k_directly_upscaled_fields_FlowMap_comparison()
    #flowmapplot.Corrected_HD_Rdirs_And_ICE5G_HD_as_data_ALG4_true_sinks_0k_directly_upscaled_fields_FlowMap_comparison()
    #flowmapplot.Upscaled_Rdirs_vs_Directly_Upscaled_fields_ICE5G_data_ALG4_corr_orog_downscaled_ls_mask_0k_FlowMap_comparison()
    #flowmapplot.Ten_Minute_Data_from_Virna_data_ALG4_corr_orog_downscaled_lsmask_no_sinks_21k_vs_0k_FlowMap_comparison()
    #flowmapplot.Upscaled_Rdirs_vs_Corrected_HD_Rdirs_ICE5G_data_ALG4_corr_orog_downscaled_ls_mask_0k_FlowMap_comparison()
    flowmapplotwithcatchment = FlowMapPlotsWithCatchments(save)
    #flowmapplotwithcatchment.Upscaled_Rdirs_vs_Corrected_HD_Rdirs_ICE5G_data_ALG4_corr_orog_downscaled_ls_mask_0k_FlowMap_comparison()
    #flowmapplotwithcatchment.compare_present_day_and_lgm_river_directions_with_catchments_virna_data_plus_tarasov_style_orog_corrs_for_both()
    #flowmapplotwithcatchment.compare_present_day_river_directions_with_catchments_virna_data_with_vs_without_tarasov_style_orog_corrs()
    #flowmapplotwithcatchment.compare_lgm_river_directions_with_catchments_virna_data_with_vs_without_tarasov_style_orog_corrs()
    #flowmapplotwithcatchment.Upscaled_Rdirs_vs_Corrected_HD_Rdirs_tarasov_upscaled_data_ALG4_corr_orog_downscaled_ls_mask_0k_FlowMap_comparison()
    #flowmapplotwithcatchment.upscaled_rdirs_with_and_without_tarasov_upscaled_data_ALG4_corr_orog_downscaled_ls_mask_0k_FlowMap_comparison()
    #flowmapplotwithcatchment.\
    #upscaled_rdirs_with_and_without_tarasov_upscaled_north_america_only_data_ALG4_corr_orog_downscaled_ls_mask_0k_FlowMap_comparison()
    #flowmapplotwithcatchment.\
    #Upscaled_Rdirs_vs_Corrected_HD_Rdirs_tarasov_upscaled_north_america_only_data_ALG4_corr_orog_downscaled_ls_mask_0k_FlowMap_comparison()
    #flowmapplotwithcatchment.\
    #Upscaled_Rdirs_vs_Corrected_HD_Rdirs_tarasov_upscaled_north_america_only_data_ALG4_corr_orog_glcc_olson_lsmask_0k_FlowMap_comparison()
    #flowmapplotwithcatchment.compare_present_day_and_lgm_river_directions_with_catchments_ICE5G_plus_tarasov_style_orog_corrs_for_both()
    #flowmapplotwithcatchment.compare_present_day_and_lgm_river_directions_with_catchments_ICE6G_plus_tarasov_style_orog_corrs_for_both()
    #flowmapplotwithcatchment.compare_ICE5G_and_ICE6G_with_catchments_tarasov_style_orog_corrs_for_both()
    #flowmapplotwithcatchment.compare_river_directions_with_dynriver_corrs_and_MERIThydro_derived_corrs()
    #flowmapplotwithcatchment.compare_river_directions_with_dynriver_corrs_and_MERIThydro_derived_corrs_original_ts()
    flowmapplotwithcatchment.compare_river_directions_with_dynriver_corrs_and_MERIThydro_derived_corrs_new_ts_10min()
    outflowplots = OutflowPlots(save)
    #outflowplots.Compare_Upscaled_Rdirs_vs_Directly_Upscaled_fields_ICE5G_data_ALG4_corr_orog_downscaled_ls_mask_0k()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_ICE5G_as_HD_data_ALG4_sinkless_all_points_0k()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_ICE5G_as_HD_data_ALG4_true_sinks_all_points_0k()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_ICE5G_ALG4_sinkless_all_points_0k_directly_upscaled_fields()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_ICE5G_ALG4_true_sinks_all_points_0k_directly_upscaled_fields()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_ICE5G_ALG4_corr_orog_all_points_0k_directly_upscaled_fields()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_ICE5G_ALG4_corr_orog_downscaled_ls_mask_all_points_0k_directly_upscaled_fields()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_Etopo1_ALG4_sinkless_directly_upscaled_fields()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_Etopo1_ALG4_true_sinks_directly_upscaled_fields()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_ICE5G_plus_tarasov_upscaled_srtm30_ALG4_corr_orog_0k_directly_upscaled_fields()
    #outflowplots.Compare_Original_Corrections_vs_Upscaled_MERIT_DEM_0k()
    outflowplots.Compare_Original_Corrections_vs_Upscaled_MERIT_DEM_0k_new_truesinks()
    #outflowplots.Compare_Original_Corrections_vs_Upscaled_MERIT_DEM_0k_new_truesinks_individual_rivers()
    #outflowplots.Compare_ICE5G_with_and_without_tarasov_upscaled_srtm30_ALG4_corr_orog_0k_directly_upscaled_fields()
    #hd_output_plots = HDOutputPlots()
    #hd_output_plots.check_water_balance_of_1978_for_constant_forcing_of_0_01()
    #hd_output_plots.plot_comparison_using_1990_rainfall_data()
    #hd_output_plots.plot_comparison_using_1990_rainfall_data_adding_back_to_discharge()
    #coupledrunoutputplots = CoupledRunOutputPlots(save=save)
    #coupledrunoutputplots.ice6g_rdirs_lgm_run_discharge_plot()
    #coupledrunoutputplots.extended_present_day_rdirs_lgm_run_discharge_plot()
    #coupledrunoutputplots.ocean_grid_extended_present_day_rdirs_vs_ice6g_rdirs_lgm_run_discharge_plot()
    #coupledrunoutputplots.extended_present_day_rdirs_vs_ice6g_rdirs_lgm_echam()
    #coupledrunoutputplots.extended_present_day_rdirs_vs_ice6g_rdirs_lgm_mpiom_pem()
    #lake_plots = LakePlots()
    #lake_plots.plotLakeDepths()
    #lake_plots.LakeAndRiverMap()
    #lake_plots.LakeAndRiverMaps()
    if show:
        plt.show()

if __name__ == '__main__':
    main()
