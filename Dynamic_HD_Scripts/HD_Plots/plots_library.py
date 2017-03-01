'''
A module containing a library of methods and classes to generate plots 
needed for dynamic HD work. Which plots are created is controlled in
the main function.

Created on Jan 29, 2016

@author: thomasriddick
'''
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import Dynamic_HD_Scripts.iohelper as iohlpr
import textwrap
import os.path
import math
import plotting_tools as pts
import match_river_mouths as mtch_rm
from Dynamic_HD_Scripts import dynamic_hd
from Dynamic_HD_Scripts import utilities
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Dynamic_HD_Scripts import field
import river_comparison_plotting_routines as rc_pts
from interactive_plotting_routines import Interactive_Plots
from __builtin__ import True

global interactive_plots

class Plots(object):
    """A general base class for plots"""
   
    hd_data_path = '/Users/thomasriddick/Documents/data/HDdata/'
    scratch_dir = '/Users/thomasriddick/Documents/data/temp/' 
    
    def __init__(self,save=False):
        """Class constructor."""
        self.save = save
        
class HDparameterPlots(Plots):
    
    hdfile_extension = "hdfiles"
    
    def __init__(self,save=False):
        super(HDparameterPlots,self).__init__(save)
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
        river_flow_k_param = dynamic_hd.load_field(filename=hd_file,
                                                   file_type=dynamic_hd.\
                                                   get_file_extension(hd_file),
                                                   field_type="Generic", unmask=False,
                                                   fieldname="ARF_K", grid_type="HD")
        river_flow_n_param = dynamic_hd.load_field(filename=hd_file,
                                                   file_type=dynamic_hd.\
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
    
    def __init__(self,save=False):
        """Class constructor."""
        super(HDOutputPlots,self).__init__(save)
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
        lsmask = dynamic_hd.load_field("/Users/thomasriddick/Documents/data/HDdata/lsmasks/generated/"
                                       "ls_mask_ten_minute_data_from_virna_0k_ALG4_sinkless"
                                       "_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170123_165707_HD_transf.nc",
                                       ".nc",field_type='Generic',grid_type='HD')
        cell_areas = dynamic_hd.load_field("/Users/thomasriddick/Documents/data/HDdata/"
                                           "gridareasandspacings/hdcellareas.nc",".nc",
                                           field_type='Generic',fieldname="cell_area",grid_type='HD')
        #stage summation to reduce rounding errors
        five_day_discharges = []
        for j in range(73):
            for i in range(j*5,(j+1)*5):
                discharge = dynamic_hd.load_field("/Users/thomasriddick/Documents/data/HDoutput/hd_N01_1978-01-02_hd_"
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
        print "Total water entering HD model: {0}".format(np.sum(lsmask_times_area,dtype=np.float128)*days_in_year*inflow_per_meter_squared)
        print "Total discharge into oceans: {0}".format(total_discharge)
        print "Total change in water in reservoirs: {0}".format(change_in_water)
        print "Total discharge - total inflow: {0}: ".format((total_discharge - days_in_year*\
                                                             np.sum(lsmask_times_area,dtype=np.float128)*inflow_per_meter_squared))
        print "(Total discharge - total inflow) + change in reservoirs: {0}".format((total_discharge - days_in_year*np.sum(lsmask_times_area,dtype=np.float128)*inflow_per_meter_squared)+ change_in_water)
        print "(Total discharge - total inflow) + change in reservoirs/Change in Reservoirs: {0}".format(((total_discharge - days_in_year*np.sum(lsmask_times_area,dtype=np.float128)*inflow_per_meter_squared)+ change_in_water)/change_in_water)

    def _calculate_total_water_in_restart(self,restart_filename):
        total_water = 0.0
        fgmem_field = dynamic_hd.load_field(restart_filename, 
                                            file_type=dynamic_hd.get_file_extension(restart_filename), 
                                            field_type="Generic",
                                            unmask=False,
                                            timeslice=None,
                                            fieldname="FGMEM",
                                            grid_type="HD")
        total_water += np.sum(fgmem_field.get_data(),dtype=np.float128)
        finfl_field = dynamic_hd.load_field(restart_filename, 
                                            file_type=dynamic_hd.get_file_extension(restart_filename), 
                                            field_type="Generic",
                                            unmask=False,
                                            timeslice=None,
                                            fieldname="FINFL",
                                            grid_type="HD")
        total_water += np.sum(finfl_field.get_data(),dtype=np.float128)
        flfmem_field = dynamic_hd.load_field(restart_filename, 
                                             file_type=dynamic_hd.get_file_extension(restart_filename), 
                                             field_type="Generic",
                                             unmask=False,
                                             timeslice=None,
                                             fieldname="FLFMEM",
                                             grid_type="HD")
        total_water += np.sum(flfmem_field.get_data(),dtype=np.float128)
        frfmem_fields = []
        for i in range(5):
            frfmem_fields.append(dynamic_hd.load_field(restart_filename, 
                                                       file_type=dynamic_hd.get_file_extension(restart_filename), 
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
            rdirs_ref = dynamic_hd.load_field(lsmask_source_ref_filepath,dynamic_hd.get_file_extension(lsmask_source_ref_filepath),
                                              field_type='RiverDirections', unmask=True, grid_type='HD').get_data()
            lsmask_ref = (rdirs_ref <= 0).astype(np.int32)
            rdirs_data = dynamic_hd.load_field(lsmask_source_data_filepath,dynamic_hd.get_file_extension(lsmask_source_data_filepath),
                                               field_type='RiverDirections', unmask=True, grid_type='HD').get_data()
            lsmask_data = (rdirs_data <= 0).astype(np.int32)
        else:
            lsmask_ref = dynamic_hd.load_field(lsmask_source_ref_filepath,dynamic_hd.get_file_extension(lsmask_source_ref_filepath),
                                              field_type='RiverDirections', unmask=True, fieldname='slm',grid_type=grid_type).get_data()
            lsmask_data = dynamic_hd.load_field(lsmask_source_data_filepath,dynamic_hd.get_file_extension(lsmask_source_data_filepath),
                                               field_type='RiverDirections', unmask=True, fieldname='slm',grid_type=grid_type).get_data()
        cell_areas = dynamic_hd.load_field(cell_areas_filepath,dynamic_hd.get_file_extension(cell_areas_filepath),
                                           field_type='Generic',unmask=True, fieldname='cell_area',grid_type=grid_type).get_data()
        lost_discharge = []
        for timeslice in range(num_timeslices):
            run_off_field = dynamic_hd.load_field(run_off_filepath, dynamic_hd.get_file_extension(run_off_filepath),
                                                  field_type='Generic', unmask=True, timeslice=timeslice ,fieldname="var501", grid_type=grid_type).get_data()*\
                                                  cell_areas
            discharge_field = dynamic_hd.load_field(discharge_filepath,dynamic_hd.get_file_extension(discharge_filepath),
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
        rdirs = dynamic_hd.load_field(rdirs_filepath,
                                      file_type=\
                                      dynamic_hd.get_file_extension(rdirs_filepath),
                                      field_type='RiverDirections',
                                      unmask=True,
                                      grid_type='HD')
        daily_global_river_discharge_outflow = np.zeros((num_timeslices))
        for i in range(num_timeslices):
            river_discharge = dynamic_hd.load_field(river_discharge_output_filepath,
                                                    file_type=\
                                                    dynamic_hd.get_file_extension(river_discharge_output_filepath),
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
        print total_discharge_info
        
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
        print total_discharge_info
        
class OutflowPlots(Plots):
    """A class for river mouth outflow plots"""
    
    rmouth_outflow_path_extension = 'rmouthflow'
    flow_maps_path_extension = 'flowmaps'
    rdirs_path_extension = 'rdirs'
    catchments_path_extension = 'catchmentmaps'
    orog_path_extension = 'orographys'
    additional_matches_list_extension = 'addmatches'
    catchment_and_outflows_mods_list_extension = 'catchmods'
    
    def __init__(self,save):
        super(OutflowPlots,self).__init__(save)
        self.rmouth_outflow_data_directory = os.path.join(self.hd_data_path,self.rmouth_outflow_path_extension)
        self.flow_maps_data_directory = os.path.join(self.hd_data_path,self.flow_maps_path_extension)
        self.rdirs_data_directory = os.path.join(self.hd_data_path,self.rdirs_path_extension)
        self.catchments_data_directory = os.path.join(self.hd_data_path,self.catchments_path_extension)
        self.orog_data_directory = os.path.join(self.hd_data_path,self.orog_path_extension)
        self.additional_matches_list_directory = os.path.join(self.hd_data_path,
                                                              self.additional_matches_list_extension)
        self.catchment_and_outflows_mods_list_directory = os.path.join(self.hd_data_path,
                                                                       self.catchment_and_outflows_mods_list_extension)
        self.temp_label = 'temp_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + "_"
        
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
                                     super_fine_orog_filename=None,
                                     super_fine_data_flowmap_filename=None,
                                     flip_super_fine_orog=False,
                                     rotate_super_fine_orog=False,
                                     additional_matches_list_filename=None,
                                     catchment_and_outflows_mods_list_filename=None,
                                     plot_simple_catchment_and_flowmap_plots=False,
                                     swap_ref_and_data_when_finding_labels=False,
                                     matching_parameter_set='default',
                                     grid_type='HD',data_original_scale_grid_type='HD',
                                     super_fine_orog_grid_type='HD',
                                     data_original_scale_grid_kwargs={},
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
            data_catchment_filepath = os.path.join(self.catchments_data_directory,
                                                   data_catchment_filename)
        if data_rdirs_filename:
            data_rdirs_filepath =  os.path.join(self.rdirs_data_directory,
                                                data_rdirs_filename)
        if ref_orog_filename:
            ref_orog_filepath = os.path.join(self.orog_data_directory,
                                             ref_orog_filename)
        if data_orog_original_scale_filename:
            data_orog_original_scale_filepath = os.path.join(self.orog_data_directory,
                                                             data_orog_original_scale_filename)
        if data_catchment_original_scale_filename:
            data_catchment_original_scale_filepath = os.path.join(self.catchments_data_directory,
                                                                  data_catchment_original_scale_filename)
        if catchment_and_outflows_mods_list_filename:
            catchment_and_outflows_mods_list_filepath = os.path.join(self.catchment_and_outflows_mods_list_directory,
                                                                     catchment_and_outflows_mods_list_filename)
        if additional_matches_list_filename:
            additional_matches_list_filepath = os.path.join(self.additional_matches_list_directory,
                                                            additional_matches_list_filename)
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
        ref_flowtocellfield = iohlpr.NetCDF4FileIOHelper.load_field(ref_flowmaps_filepath,grid_type,**grid_kwargs)
        data_flowtocellfield = iohlpr.NetCDF4FileIOHelper.load_field(data_flowmaps_filepath,grid_type,**grid_kwargs)
        rdirs_field = iohlpr.NetCDF4FileIOHelper.load_field(rdirs_filepath,grid_type,**grid_kwargs)
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
        if super_fine_orog_filename:
            super_fine_orog_field = iohlpr.NetCDF4FileIOHelper.\
                load_field(super_fine_orog_filepath,
                           grid_type=super_fine_orog_grid_type,
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
                                                        axis=1)/2,
                                                axis=1)
                if super_fine_data_flowmap is not None:
                    super_fine_data_flowmap = np.roll(super_fine_data_flowmap,
                                                      np.size(super_fine_data_flowmap,
                                                              axis=1)/2,
                                                      axis=1)
        else:  
            super_fine_orog_field = None
            super_fine_data_flowmap = None
        if flip_ref_field:
            ref_flowtocellfield = np.flipud(ref_flowtocellfield)
            rdirs_field = np.flipud(rdirs_field)
            if ref_catchment_filename:
                ref_catchment_field = np.flipud(ref_catchment_field)
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
                                          np.size(ref_flowtocellfield,axis=1)/2,
                                          axis=1)
            rdirs_field = np.roll(rdirs_field,
                                  np.size(rdirs_field,axis=1)/2,
                                  axis=1)
            if ref_catchment_filename:
                ref_catchment_field = np.roll(ref_catchment_field,
                                              np.size(ref_catchment_field,axis=1)/2,
                                              axis=1)
        if rotate_data_field:
            data_flowtocellfield = np.roll(data_flowtocellfield,
                                           np.size(data_flowtocellfield,axis=1)/2,
                                           axis=1)
            if data_rdirs_filename:
                data_rdirs_field = np.roll(data_rdirs_field,
                                           np.size(data_rdirs_field,axis=1)/2,
                                           axis=1)
            if data_catchment_filename:
                data_catchment_field = np.roll(data_catchment_field,
                                              np.size(data_catchment_field,axis=1)/2,
                                              axis=1)
                if catchment_grid_changed:
                    data_original_scale_flowtocellfield = np.roll(data_original_scale_flowtocellfield,
                                                                  np.size(data_original_scale_flowtocellfield,
                                                                          axis=1)/2,
                                                                  axis=1)
                    data_catchment_field_original_scale = np.roll(data_catchment_field_original_scale,
                                                                  np.size(data_catchment_field_original_scale,
                                                                          axis=1)/2,
                                                                  axis=1)
            if data_orog_original_scale_filename:
                data_orog_original_scale_field = np.roll(data_orog_original_scale_field,
                                                         np.size(data_orog_original_scale_field,
                                                                 axis=1)/2,
                                                         axis=1)  
            else:
                data_orog_original_scale_field = None   
        temp_file_list = []
        if catchment_and_outflows_mods_list_filename:
            ref_outflow_field = dynamic_hd.load_field(reference_rmouth_outflows_filename,
                                                      file_type=dynamic_hd.\
                                                      get_file_extension(reference_rmouth_outflows_filename), 
                                                      field_type='Generic', grid_type=grid_type,**grid_kwargs)
            data_outflow_field = dynamic_hd.load_field(data_rmouth_outflows_filename,
                                                       file_type=dynamic_hd.\
                                                       get_file_extension(data_rmouth_outflows_filename), 
                                                       field_type='Generic', grid_type=grid_type,**grid_kwargs)
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
                                                      data_catchment_field_original_scale,
                                                      original_scale_flowmap=\
                                                      data_original_scale_flowtocellfield,
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
            reference_rmouth_outflows_filename=os.path.join(self.scratch_dir,
                                                            self.temp_label + os.path.\
                                                            basename(reference_rmouth_outflows_filename))
            data_rmouth_outflows_filename=os.path.join(self.scratch_dir,
                                                       self.temp_label + os.path.\
                                                       basename(reference_rmouth_outflows_filename))
            temp_file_list.append(reference_rmouth_outflows_filename)
            temp_file_list.append(data_rmouth_outflows_filename)
            dynamic_hd.write_field(reference_rmouth_outflows_filename, 
                                   field=ref_outflow_field, 
                                   file_type=dynamic_hd.\
                                   get_file_extension(reference_rmouth_outflows_filename))
            dynamic_hd.write_field(data_rmouth_outflows_filename, 
                                   field=data_outflow_field, 
                                   file_type=dynamic_hd.\
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
        for pair in matchedpairs:
            if pair[0].get_lat() > 312:
                continue
            print "Ref Point: " + str(pair[0]) + "Matches: " + str(pair[1])
            plt.figure(figsize=(25,12.5))
            ax = plt.subplot(222)
            rc_pts.plot_river_rmouth_flowmap(ax=ax, 
                                             ref_flowtocellfield=ref_flowtocellfield, 
                                             data_flowtocellfield=data_flowtocellfield,
                                             rdirs_field=rdirs_field, 
                                             pair=pair)
            ax_hist = plt.subplot(221)
            ax_catch = plt.subplot(223)
            catchment_section,catchment_bounds,scale_factor = \
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
                                                              grid_type=grid_type,
                                                              data_original_scale_grid_type=\
                                                              data_original_scale_grid_type,
                                                              data_original_scale_grid_kwargs=\
                                                              data_original_scale_grid_kwargs,
                                                              **grid_kwargs)
            ax = plt.subplot(224)
            rc_pts.plot_whole_river_flowmap(ax=ax,pair=pair,ref_flowtocellfield=ref_flowtocellfield,
                                            data_flowtocellfield=data_flowtocellfield,
                                            rdirs_field=rdirs_field,data_rdirs_field=data_rdirs_field,
                                            catchment_bounds=catchment_bounds)
            if plot_simple_catchment_and_flowmap_plots:
                simple_candf_plt = plt.figure(figsize=(10,6))
                simple_ref_ax  = plt.subplot(121)
                simple_data_ax = plt.subplot(122)
                flowtocell_threshold = 75
                rc_pts.simple_catchment_and_flowmap_plots(fig=simple_candf_plt,
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
                                                          grid_type=grid_type,
                                                          data_original_scale_grid_type=\
                                                          data_original_scale_grid_type,
                                                          data_original_scale_grid_kwargs=\
                                                          data_original_scale_grid_kwargs,**grid_kwargs)
            if ref_orog_filename and data_orog_original_scale_filename:
                if super_fine_orog_filename:
                            data_to_super_fine_scale_factor = \
                                pts.calculate_scale_factor(course_grid_type=data_original_scale_grid_type,
                                                           course_grid_kwargs=data_original_scale_grid_kwargs,
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
                                              ref_to_super_fine_scale_factor) 
            elif ref_orog_filename or data_orog_original_scale_filename:
                raise UserWarning("No orography plot generated, require both a reference orography"
                                  " and a data orography to generate an orography plot")
        print "Unresolved Conflicts: "
        for conflict in unresolved_conflicts:
            print " Conflict:"
            for pair in conflict:
                print "  Ref Point" + str(pair[0]) + "Matches" + str(pair[1])
        for temp_file in temp_file_list:
            if os.path.basename(temp_file).startswith("temp_"):
                print "Deleting File: {0}".format(temp_file)
                os.remove(temp_file)
        
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
        data_creation_datetime="20160930_001057"
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
                                          flip_data_field=True,
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
    ls_masks_extension        = 'lsmasks'
    
    def __init__(self,save):
        """Class Constructor"""
        super(FlowMapPlots,self).__init__(save)
        self.flow_maps_data_directory = os.path.join(self.hd_data_path,self.flow_maps_path_extension)
        self.ls_masks_data_directory= os.path.join(self.hd_data_path,self.ls_masks_extension)
        
    def SimpleFlowMapPlotHelper(self,filename,grid_type,log_max=4):
        """Help produce simple flow maps"""
        flowmap_object = dynamic_hd.load_field(filename,
                                               file_type=dynamic_hd.get_file_extension(filename), 
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
                                         second_datasource_name="Data",**kwargs):
        """Help compare two two-colour flow maps"""
        flowmap_ref_field = dynamic_hd.load_field(ref_filename,
                                                  file_type=dynamic_hd.get_file_extension(ref_filename), 
                                                  field_type='Generic', 
                                                  grid_type=grid_type,**kwargs)
        flowmap_data_field = dynamic_hd.load_field(data_filename,
                                                   file_type=dynamic_hd.get_file_extension(data_filename), 
                                                   field_type='Generic', 
                                                   grid_type=grid_type,**kwargs)
        if lsmask_filename:
            lsmask_field = dynamic_hd.load_field(lsmask_filename, 
                                                 file_type=dynamic_hd.get_file_extension(lsmask_filename), 
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
        plt.figure()
        ax = plt.subplot(111)
        flowmap_ref_field[flowmap_ref_field < minflowcutoff] = 1
        flowmap_ref_field[flowmap_ref_field >= minflowcutoff] = 2
        flowmap_ref_field[np.logical_and(flowmap_data_field >= minflowcutoff,
                                         flowmap_ref_field == 2)] = 3
        flowmap_ref_field[np.logical_and(flowmap_data_field >= minflowcutoff,
                                         flowmap_ref_field != 3)] = 4                                
        if lsmask_filename:
            flowmap_ref_field[lsmask == 1] = 0
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
        
    def FlowMapTwoColourPlotHelper(self,filename,lsmask_filename=None,grid_type='HD',
                                   minflowcutoff=100,flip_data=False,flip_mask=False,
                                   **kwargs):
        """Help produce two colour flow maps"""
        flowmap_object = dynamic_hd.load_field(filename,
                                               file_type=dynamic_hd.get_file_extension(filename), 
                                               field_type='Generic', 
                                               grid_type=grid_type,**kwargs)
        lsmask_field = dynamic_hd.load_field(lsmask_filename, 
                                             file_type=dynamic_hd.get_file_extension(lsmask_filename), 
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
        bounds = range(4)
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
        data_filename=os.path.join(self.flow_maps_data_directory,'upscaled',
                                   "flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_20160930_001057.nc")
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
                                              first_datasource_name="Present Day",
                                              second_datasource_name="LGM")
    
class OrographyPlots(Plots):
    """A general base class for orography plots"""
    
    orography_path_extension = 'orographys'

    def __init__(self,save):
        """Class constructor"""
        super(OrographyPlots,self).__init__(save)
        self.orography_data_directory = os.path.join(self.hd_data_path,self.orography_path_extension)
        
class SimpleOrographyPlots(OrographyPlots):
    
    def __init__(self,save):
        """Class constructor."""
        super(SimpleOrographyPlots,self).__init__(save)
        
    def SimpleArrayPlotHelper(self,filename):
        """Assists the creation of simple array plots"""
        #levels = np.linspace(-100.0, 9900.0, 100, endpoint=True)
        plt.figure()
        #plt.contourf(orography_field,levels)
        plt.colorbar()
        pts.invert_y_axis()

class Ice5GComparisonPlots(OrographyPlots):
    """Handles generation Ice5G data comparison plots"""

    def __init__(self,save,use_old_data=False):
        """Class constructor. Sets filename (to point to either old or new data)"""
        super(Ice5GComparisonPlots,self).__init__(save)
        print "Comparing the Modern and LGM Ice-5G 5-minute resolution orography datasets"
        
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
        print "Line contour plot created"
            
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
        print "Filled contour plot created"
    
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
        uflabels=[u'Difference $\\leq {0}$'.format(minc)]
        oflabels=[u'${0} <$ Difference'.format(maxc)]
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
        print "Combined plot created"
         
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
    flowmapplot = FlowMapPlots(save)
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
    flowmapplot.Upscaled_Rdirs_vs_Corrected_HD_Rdirs_ICE5G_data_ALG4_corr_orog_downscaled_ls_mask_0k_FlowMap_comparison()
    #outflowplots = OutflowPlots(save)
    #outflowplots.Compare_Upscaled_Rdirs_vs_Directly_Upscaled_fields_ICE5G_data_ALG4_corr_orog_downscaled_ls_mask_0k()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_ICE5G_as_HD_data_ALG4_sinkless_all_points_0k()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_ICE5G_as_HD_data_ALG4_true_sinks_all_points_0k()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_ICE5G_ALG4_sinkless_all_points_0k_directly_upscaled_fields()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_ICE5G_ALG4_true_sinks_all_points_0k_directly_upscaled_fields()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_ICE5G_ALG4_corr_orog_all_points_0k_directly_upscaled_fields()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_ICE5G_ALG4_corr_orog_downscaled_ls_mask_all_points_0k_directly_upscaled_fields()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_Etopo1_ALG4_sinkless_directly_upscaled_fields()
    #outflowplots.Compare_Corrected_HD_Rdirs_And_Etopo1_ALG4_true_sinks_directly_upscaled_fields()
    #hd_output_plots = HDOutputPlots()
    #hd_output_plots.check_water_balance_of_1978_for_constant_forcing_of_0_01()
    #hd_output_plots.plot_comparison_using_1990_rainfall_data()
    #hd_output_plots.plot_comparison_using_1990_rainfall_data_adding_back_to_discharge()
    if show:
        plt.show()

if __name__ == '__main__':
    main()