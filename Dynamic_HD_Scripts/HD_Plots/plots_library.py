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

global interactive_plots

class Plots(object):
    """A general base class for plots"""
   
    hd_data_path = '/Users/thomasriddick/Documents/data/HDdata/'
    scratch_dir = '/Users/thomasriddick/Documents/data/temp/' 
    
    def __init__(self,save=False):
        """Class constructor."""
        self.save = save
        
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
                                     rivers_to_plot=None,
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
                                                                         data_original_scale_grid_type,
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
                           data_original_scale_grid_type,
                           **data_original_scale_grid_kwargs)
        if super_fine_orog_filename:
            super_fine_orog_field = iohlpr.NetCDF4FileIOHelper.\
                load_field(super_fine_orog_filepath,
                           super_fine_orog_grid_type,
                           **super_fine_orog_grid_kwargs)
            if super_fine_data_flowmap_filename:
                super_fine_data_flowmap = iohlpr.NetCDF4FileIOHelper.\
                    load_field(super_fine_data_flowmap_filepath,
                               super_fine_orog_grid_type,
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
                                                           param_set='default',**grid_kwargs)
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
            if rivers_to_plot is not None:
                if not (pair[0].get_lat(),pair[0].get_lon()) in rivers_to_plot:
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
                                          "generated/upscaled/upscaled_rdirs_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs_{0}_updated.nc".\
                                          format(data_creation_datetime_rdirs_upscaled),
                                          flip_ref_field=True,
                                          rotate_ref_field=True,
                                          flip_data_field=True,
                                          rotate_data_field=True,
                                          #data_rdirs_filename="generated/upscaled/"
                                          #"upscaled_rdirs_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs_{0}_updated.nc".\
                                          #format(data_creation_datetime_rdirs_upscaled),
                                          ref_catchment_filename=\
                                          "upscaled/catchmentmap_unsorted_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime_directly_upscaled),
                                          data_catchment_filename=\
                                          "catchmentmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs_{0}_updated.nc".\
                                          format(data_creation_datetime_rdirs_upscaled),
                                          data_catchment_original_scale_filename=\
                                          "catchmentmap_unsorted_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime_directly_upscaled),
                                          data_original_scale_flow_map_filename=\
                                          "flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".\
                                          format(data_creation_datetime_directly_upscaled),
                                          swap_ref_and_data_when_finding_labels=True,
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
                                         lsmask_has_same_orientation_as_ref=True,**kwargs):
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
        ax.imshow(flowmap_ref_field,cmap=cmap,norm=norm)
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
        tic_labels = ['Sea', 'Land','Reference River Path','Common River Path','Data River Path']
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
        plt.imshow(flowmap,cmap=cmap,norm=norm)
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
    
    if show:
        plt.show()

if __name__ == '__main__':
    main()