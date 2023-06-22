'''
Created on Mar 2, 2017

@author: thomasriddick
'''


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec
from matplotlib import rcParams
import os
import os.path as path
import numpy as np
from netCDF4 import Dataset
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.utilities import utilities
from plotting_utilities import river_comparison_plotting_routines as rc_pts
from plotting_utilities import plotting_tools as pts
from HD_Plots.plots.plots_library import (Plots,HDparameterPlots,HDOutputPlots,OutflowPlots, #@UnusedImport
                                          FlowMapPlots,FlowMapPlotsWithCatchments,OrographyPlots, #@UnusedImport
                                          SimpleOrographyPlots, Ice5GComparisonPlots) #@UnusedImport

class PlotsForGMDPaper(OutflowPlots,FlowMapPlotsWithCatchments,HDOutputPlots):
    """Plots for GMD Paper"""

    save_path = "/Users/thomasriddick/Documents/plots/Dynamic HD/plots_generated_for_paper"

    def __init__(self, save):
        """Class constructor"""
        color_palette_to_use="gmd_paper"
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Helvetica']
        super(PlotsForGMDPaper,self).__init__(save,color_palette_to_use)

    def danube_catchment_correction_plots(self):
        """Three plots of the Danube catchment showing the effect of applying correction an orography."""
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.plots_data_dir,
            "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.plots_data_dir,
            "upscaled/rmouthflows__ICE5G_data_ALG4_sinkless_0k_upscale_riverflows_and_river_mouth_flows_20160603_112520.nc")
        plotters = self.OutFlowComparisonPlotHelpers(corrected_hd_rdirs_rmouthoutflow_file,
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
                                                     'additional_matches_ice5g_10min_uncorrected.txt',
                                                     super_fine_orog_filename="ETOPO1_Ice_c_gmt4.nc",
                                                     flip_super_fine_orog=True,
                                                     rotate_super_fine_orog=False,
                                                     super_fine_orog_grid_type='LatLong1min',
                                                     external_ls_mask_filename="generated/ls_mask_recreate_connected_HD_lsmask_"
                                                                               "from_glcc_olson_data_20170513_195421.nc",
                                                     flip_external_ls_mask=False,
                                                     rotate_external_ls_mask=True,
                                                     rivers_to_plot=[(90,418)],
                                                     alternative_catchment_bounds=[67,99,373,422],
                                                     plot_simple_catchment_and_flowmap_plots=True,
                                                     return_simple_catchment_and_flowmap_plotters=True,
                                                     grid_type='HD',data_original_scale_grid_type='LatLong10min')
        ref_plotter = plotters[0][0]
        uncorr_data_plotter = plotters[0][1]
        data_creation_datetime="20160930_001057"
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.plots_data_dir,
            "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.plots_data_dir,
            "upscaled/rmouthflows_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_{0}.nc".format(data_creation_datetime))
        plotters = self.OutFlowComparisonPlotHelpers(corrected_hd_rdirs_rmouthoutflow_file,
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
                                                     external_ls_mask_filename="generated/ls_mask_recreate_connected_HD_lsmask_"
                                                     "from_glcc_olson_data_20170513_195421.nc",
                                                     flip_external_ls_mask=False,
                                                     rotate_external_ls_mask=True,
                                                     rivers_to_plot=[(90,418)],
                                                     alternative_catchment_bounds=[67,99,373,422],
                                                     super_fine_orog_grid_type='LatLong1min',
                                                     plot_simple_catchment_and_flowmap_plots=True,
                                                     return_simple_catchment_and_flowmap_plotters=True,
                                                     grid_type='HD',data_original_scale_grid_type='LatLong10min')
        for fig_num in plt.get_fignums()[-6:]:
            plt.close(fig_num)
        corr_data_plotter = plotters[0][1]
        plt.figure(figsize=(12,3))
        gs = gridspec.GridSpec(1,3)
        print("Total cumulative flow threshold used for danube plot {0}".\
            format(corr_data_plotter.get_flowtocell_threshold()))
        #fig.suptitle('River catchment plus cells with a cumulative flow greater than {0}'.\
        #             format(corr_data_plotter.get_flowtocell_threshold()),fontsize=30)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax2 = plt.subplot(gs[2])
        ax0.annotate(xy=(0, -34),xycoords="axes points",s="(a)",fontsize=20)
        ax1.annotate(xy=(0, -34),xycoords="axes points",s="(b)",fontsize=20)
        ax2.annotate(xy=(0, -34),xycoords="axes points",s="(c)",fontsize=20)
        ref_plotter(ax0)
        uncorr_data_plotter(ax1)
        corr_data_plotter(ax2)
        plt.tight_layout(rect=(0,0.1,1,1))
        if self.save:
            plt.savefig(path.join(self.save_path,"danube_comparison.pdf"),dpi=300)

    def comparison_of_manually_corrected_HD_rdirs_vs_automatically_generated_10min_rdirs(self):
        data_creation_datetime="20170517_003802"
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.plots_data_dir,
            "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.plots_data_dir,
            "upscaled/rmouthflows_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson_lsmask_0k_{0}.nc".format(data_creation_datetime))
        catchment_plotters =\
            self.OutFlowComparisonPlotHelpers(corrected_hd_rdirs_rmouthoutflow_file,
                                              ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field,
                                              "flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                              "upscaled/flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson_lsmask_0k_{0}.nc".\
                                              format(data_creation_datetime),
                                              "rivdir_vs_1_9_data_from_stefan.nc",
                                              flip_data_field=False,
                                              rotate_data_field=True,
                                              data_rdirs_filename="generated/"
                                              "updated_RFDs_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson_lsmask_0k_{0}.nc".\
                                              format(data_creation_datetime),
                                              ref_catchment_filename=\
                                              "catchmentmap_corrected_HD_rdirs_post_processing_20160427_141158.nc",
                                              data_catchment_filename=\
                                              "upscaled/catchmentmap_unsorted_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson_lsmask_0k_{0}.nc".\
                                              format(data_creation_datetime),
                                              data_catchment_original_scale_filename=\
                                              "catchmentmap_unsorted_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson_lsmask_0k_{0}.nc".\
                                              format(data_creation_datetime),
                                              data_original_scale_flow_map_filename=\
                                              "flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson_lsmask_0k_{0}.nc".\
                                              format(data_creation_datetime),
                                              ref_orog_filename="topo_hd_vs1_9_data_from_stefan.nc",
                                              data_orog_original_scale_filename=
                                              "generated/corrected/"
                                              "corrected_orog_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson_lsmask_0k_{0}.nc".\
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
                                              external_ls_mask_filename="generated/ls_mask_recreate_connected_HD_lsmask_"
                                                                        "from_glcc_olson_data_20170513_195421.nc",
                                              flip_external_ls_mask=False,
                                              rotate_external_ls_mask=True,
                                              rivers_to_plot=[(160,573),(117,424),(121,176)],
                                              allow_new_true_sinks=True,
                                              alternative_catchment_bounds=None,
                                              split_comparison_plots_across_multiple_canvases=True,
                                              use_simplified_catchment_colorscheme=True,
                                              use_simplified_flowmap_colorscheme=True,
                                              return_catchment_plotters=True,
                                              grid_type='HD',data_original_scale_grid_type='LatLong10min')
        highest_fig_num = plt.get_fignums()[-1]
        for i in range(highest_fig_num-11,highest_fig_num+1):
            plt.close(i)
        plt.figure(figsize=(12.4,9))
        gs = gridspec.GridSpec(4,5,width_ratios=[22,1,22,2,12],
                                   height_ratios=[16,1,3,8])
        ax0 = plt.subplot(gs[:-1,0])
        ax1 = plt.subplot(gs[2:,2:])
        ax2 = plt.subplot(gs[0,2])
        ax3 = plt.subplot(gs[0,3])
        ax0.set_title("Nile",fontsize=14)
        ax1.set_title("Mississippi",fontsize=14)
        ax2.set_title("Mekong",fontsize=14)
        catchment_plotters[0].set_legend(False)
        catchment_plotters[1].set_legend(False)
        catchment_plotters[2].set_cax(ax3)
        ax3.tick_params(labelsize=20)
        catchment_plotters[0](ax0)
        catchment_plotters[0].apply_axis_locators_and_formatters(ax0)
        catchment_plotters[1](ax1)
        catchment_plotters[1].apply_axis_locators_and_formatters(ax1)
        catchment_plotters[2](ax2)
        catchment_plotters[2].apply_axis_locators_and_formatters(ax2)
        if self.save:
            plt.savefig(path.join(self.save_path,"me_mi_ni_10min_vs_man_corr.pdf"),dpi=300)

    def comparison_of_modern_river_directions_10_minute_original_vs_HD_upscaled(self):
        data_creation_datetime_directly_upscaled="20170517_003802"
        data_creation_datetime_rdirs_upscaled = "20170517_004128"
        ice5g_ALG4_sinkless_all_points_0k_river_flow_dir_upsc_field = os.path.join(self.plots_data_dir,
            "rmouthflows_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson"
            "_lsmask_0k_upscale_rdirs_{0}_updated.nc"\
                .format(data_creation_datetime_rdirs_upscaled))
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.plots_data_dir,
            "upscaled/rmouthflows_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_"
            "sinkless_glcc_olson_lsmask_0k_{0}.nc"\
                .format(data_creation_datetime_directly_upscaled))
        catchment_plotters =\
            self.OutFlowComparisonPlotHelpers(ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field,
                                              ice5g_ALG4_sinkless_all_points_0k_river_flow_dir_upsc_field,
                                              "upscaled/flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_"
                                              "ALG4_sinkless_glcc_olson_lsmask_0k_{0}.nc".\
                                              format(data_creation_datetime_directly_upscaled),
                                              "flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_"
                                              "olson_lsmask_0k_upscale_rdirs_{0}_updated.nc".\
                                              format(data_creation_datetime_rdirs_upscaled),
                                              "generated/upscaled/"
                                              "upscaled_rdirs_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless"
                                              "_glcc_olson_lsmask_0k_upscale_rdirs_{0}_updated.nc".\
                                              format(data_creation_datetime_rdirs_upscaled),
                                              flip_ref_field=False,
                                              rotate_ref_field=True,
                                              flip_data_field=False,
                                              rotate_data_field=True,
                                              ref_orog_filename=\
                                              "topo_hd_vs1_9_data_from_stefan.nc",
                                              data_orog_original_scale_filename=\
                                              "generated/corrected/"
                                              "corrected_orog_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless"
                                              "_glcc_olson_lsmask_0k_{0}.nc".format(data_creation_datetime_directly_upscaled),
                                              data_catchment_filename=\
                                              "catchmentmap_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless"
                                              "_glcc_olson_lsmask_0k_upscale_rdirs_{0}_updated.nc".\
                                              format(data_creation_datetime_rdirs_upscaled),
                                              ref_catchment_filename=
                                              "upscaled/catchmentmap_unsorted_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_"
                                              "data_ALG4_sinkless_glcc_olson_lsmask_0k_{0}.nc".\
                                              format(data_creation_datetime_directly_upscaled),
                                              data_catchment_original_scale_filename=\
                                              "catchmentmap_unsorted_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4"
                                              "_sinkless_glcc_olson_lsmask_0k_{0}.nc".\
                                              format(data_creation_datetime_directly_upscaled),
                                              data_original_scale_flow_map_filename=\
                                              "flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_"
                                              "glcc_olson_lsmask_0k_{0}.nc".\
                                              format(data_creation_datetime_directly_upscaled),
                                              swap_ref_and_data_when_finding_labels=True,
                                              catchment_and_outflows_mods_list_filename=\
                                              "catch_and_outflow_mods_ice5g_10min_directly_upscaled_rdirs_vs_indirectly_upscaled_data.txt",
                                              matching_parameter_set='area',
                                              rivers_to_plot=[(161,573),(120,176)],
                                              external_ls_mask_filename="generated/ls_mask_recreate_connected_HD_lsmask_"
                                                                        "from_glcc_olson_data_20170513_195421.nc",
                                              flip_external_ls_mask=False,
                                              rotate_external_ls_mask=True,
                                              split_comparison_plots_across_multiple_canvases=True,
                                              use_simplified_catchment_colorscheme=True,
                                              use_simplified_flowmap_colorscheme=True,
                                              use_upscaling_labels=True,
                                              return_catchment_plotters=True,
                                              grid_type='HD',data_original_scale_grid_type='LatLong10min')
        highest_fig_num = plt.get_fignums()[-1]
        for i in range(highest_fig_num-7,highest_fig_num+1):
            plt.close(i)
        plt.figure(figsize=(9,8))
        gs = gridspec.GridSpec(2,3,width_ratios=[25,3,20],
                               height_ratios=[4,5])
        ax0 = plt.subplot(gs[0,:])
        ax1 = plt.subplot(gs[1,0])
        ax2 = plt.subplot(gs[1,1])
        ax0.set_title("Mississippi",fontsize=14)
        ax1.set_title("Mekong",fontsize=14)
        catchment_plotters[0].set_legend(False)
        catchment_plotters[1].set_cax(ax2)
        ax2.tick_params(labelsize=20)
        catchment_plotters[0](ax0)
        catchment_plotters[1](ax1)
        catchment_plotters[0].apply_axis_locators_and_formatters(ax0)
        catchment_plotters[1].apply_axis_locators_and_formatters(ax1)
        if self.save:
            plt.savefig(path.join(self.save_path,"me_mi_upscaling_comp.pdf"),dpi=300)

    def compare_upscaled_automatically_generated_rdirs_to_HD_manually_corrected_rdirs(self):
        ref_filename=os.path.join(self.plots_data_dir,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.plots_data_dir,
                                  'flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale'
                                  '_rdirs_20161031_113238_updated.nc')
        lsmask_filename=os.path.join(self.plots_data_dir,
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

    def compare_upscaled_automatically_generated_rdirs_to_HD_manually_corrected_rdirs_with_catchments(self):
        ref_filename=os.path.join(self.plots_data_dir,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.plots_data_dir,
                                  'flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless'
                                  '_glcc_olson_lsmask_0k_upscale_rdirs_20170517_004128_updated.nc')
        lsmask_filename=os.path.join(self.plots_data_dir,"generated",
                                     "ls_mask_recreate_connected_HD_lsmask_"
                                     "from_glcc_olson_data_20170513_195421.nc")
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.plots_data_dir,
                                                             "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        upscaled_rdirs_rmouthoutflow_file = os.path.join(self.plots_data_dir,
                                                         "rmouthflows_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only"
                                                         "_data_ALG4_sinkless_glcc_olson_lsmask_0k_upscale_rdirs_20170517_004128"
                                                         "_updated.nc")
        glacier_mask_filename=os.path.join(self.orog_data_directory,"ice5g_v1_2_00_0k_10min.nc")
        self.FlowMapTwoColourComparisonWithCatchmentsHelper(ref_flowmap_filename=ref_filename,
                                                            data_flowmap_filename=data_filename,
                                                            ref_catchment_filename=\
                                                            "catchmentmap_corrected_HD_rdirs_"
                                                            "post_processing_20160427_141158.nc",
                                                            data_catchment_filename="catchmentmap_ICE5G_and_tarasov_upscaled_"
                                                            "srtm30plus_north_america_only_data_ALG4_sinkless_glcc"
                                                            "_olson_lsmask_0k_upscale_rdirs_20170517_004128_updated.nc",
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
                                                            first_datasource_name="Default HD",
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
                                                            use_only_one_color_for_flowmap=True,
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "catch_and_outflow_mods_ice5g_10min_upscaled_"
                                                            "rdirs_vs_modern_day_glcc_olson_lsmask.txt",
                                                            second_datasource_name="Dynamic HD",use_title=False,
                                                            remove_antartica=True,
                                                            glacier_mask_filename=glacier_mask_filename,
                                                            glacier_mask_grid_type='LatLong10min',
                                                            flip_glacier_mask=True,
                                                            rotate_glacier_mask=True,
                                                            grid_type='HD')
        plt.tight_layout()
        if self.save:
            plt.savefig(path.join(self.save_path,"global_man_corr_vs_auto_gen_upscaled_global.pdf"),dpi=300)

    def compare_present_day_and_lgm_river_directions(self):
        ref_filename=os.path.join(self.plots_data_dir,
                                  "flowmap_ten_minute_data_from_virna_0k_ALG4_sinkless"
                                   "_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170123"
                                   "_165707_upscaled_updated.nc")
        data_filename=os.path.join(self.plots_data_dir,
                                   "flowmap_ten_minute_data_from_virna_lgm_ALG4_sinkless"
                                   "_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170127"
                                   "_163957_upscaled_updated.nc")
        lsmask_filename=os.path.join(self.plots_data_dir,
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
                                              second_datasource_name="LGM",
                                              add_title=False)
        plt.tight_layout()
        if self.save:
            plt.savefig(path.join(self.save_path,"lgm_vs_present_global_comp.pdf"),dpi=300)

    def compare_present_day_and_lgm_river_directions_with_catchments(self):
        present_day_data_datetime = "20170612_202721"
        lgm_data_datetime = "20170612_202559"
        ref_filename=os.path.join(self.plots_data_dir,
                                  "flowmap_ICE6g_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov"
                                  "_orog_corrs_{0}_upscaled_updated.nc".\
                                  format(present_day_data_datetime))
        data_filename=os.path.join(self.plots_data_dir,
                                   "flowmap_ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_"
                                   "tarasov_orog_corrs_{0}_upscaled_updated.nc".\
                                   format(lgm_data_datetime))
        lsmask_filename=os.path.join(self.plots_data_dir,"generated",
                                     "ls_mask_ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs"
                                     "_tarasov_orog_corrs_{0}_HD_transf.nc".\
                                     format(lgm_data_datetime))
        extra_lsmask_filename=os.path.join(self.plots_data_dir,"generated",
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
        reference_rmouth_outflows_filename=os.path.join(self.plots_data_dir,
                                                        "rmouthflows_ICE6g_0k_ALG4_sinkless_no_true_sinks_oceans_"
                                                        "lsmask_plus_upscale_rdirs_tarasov_orog_corrs"
                                                        "_{0}_upscaled_updated.nc".\
                                            format(present_day_data_datetime))
        data_rmouth_outflows_filename=os.path.join(self.plots_data_dir,
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
                                                            catchment_and_outflows_mods_list_filename=\
                                                            "ice6g_catch_and_outflow_mods_lgm_vs_"
                                                            "present_day.txt",
                                                            additional_matches_list_filename=\
                                                            "ice6g_additional_matches_10min_upscaled"
                                                            "_lgm_vs_present.txt",
                                                            use_single_color_for_discrepancies=True,
                                                            use_only_one_color_for_flowmap=False,
                                                            use_title=False,remove_antartica=True,
                                                            difference_in_catchment_label="Difference",
                                                            glacier_mask_filename=glacier_mask_filename,
                                                            extra_lsmask_filename=extra_lsmask_filename,
                                                            glacier_mask_grid_type='LatLong10min',
                                                            flip_glacier_mask=True,
                                                            rotate_glacier_mask=True,
                                                            grid_type='HD')
        plt.tight_layout()
        if self.save:
            plt.savefig(path.join(self.save_path,"lgm_vs_present_global_comp.pdf"),dpi=300)

    def discharge_plot(self):
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
                                                                                os.path.join(self.plots_data_dir,
                                                                                             "rdirs_from_current_hdparas.nc"),
                                                                                num_timeslices=365,label="Current Model HD Run using 100 cycle spin-up")
        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output__ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_after_one_years_running.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.plots_data_dir,
                                                                                             "upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_upscaled_updated_transf.nc"),
                                                                                num_timeslices=365,label="Dynamic HD using 1 cycle spin-up")
        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output__ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170116_235534.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.plots_data_dir,
                                                                                             "upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_upscaled_updated_transf.nc"),
                                                                                num_timeslices=365,label="Dynamic HD using 1 cycle spin-up as basis")
        lost_discharge =  self._calculate_discharge_lost_to_changes_in_lsmask(lsmask_source_ref_filepath=\
                                                                              os.path.join(self.plots_data_dir,
                                                                                           "jsbach_T106_11tiles_5layers_1976.nc"),
                                                                              lsmask_source_data_filepath=\
                                                                              os.path.join(self.plots_data_dir,
                                                                                           "updated_jsbach_T106_11tiles_5layers_1976_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170123_165707.nc"),
                                                                              run_off_filepath=os.path.join(self.hdinput_data_directory,'runoff_T106_1990.nc'),
                                                                              discharge_filepath=os.path.join(self.hdinput_data_directory,'drainage_T106_1990.nc'),
                                                                              cell_areas_filepath=os.path.join(self.cell_areas_data_directory,'T106_grid_cell_areas.nc'),
                                                                              num_timeslices=365,grid_type="T106")
        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output__ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_after_one_years_running.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.plots_data_dir,
                                                                                             "upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_upscaled_updated_transf.nc"),
                                                                                num_timeslices=365,lost_discharge=lost_discharge,label="Dynamic HD using 1 cycle spin-up + lost discharge")
        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output__ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_after_thirty_years_running.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.plots_data_dir,
                                                                                             "upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_upscaled_updated_transf.nc"),
                                                                                num_timeslices=365,lost_discharge=lost_discharge,label="Dynamic HD using 30 cycle spin-up + lost discharge")
        total_discharge_info += self._river_discharge_outflow_comparison_helper(ax,river_discharge_output_filepath=\
                                                                                os.path.join(self.river_discharge_output_data_path,
                                                                                "hd_1990-01-2_hd_higres_output__ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170116_235534.nc"),
                                                                                rdirs_filepath=\
                                                                                os.path.join(self.plots_data_dir,
                                                                                             "upscaled_rdirs_ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_135934_upscaled_updated_transf.nc"),
                                                                                num_timeslices=365,lost_discharge=lost_discharge,label="Dynamic HD using 1 cycle spin-up as basis+ lost discharge")
        ax.legend()
        print(total_discharge_info)

    def ocean_pem_plots_extended_present_day_vs_ice6g_rdirs(self):
        extended_present_day_rdirs_data_filename=os.path.join(self.river_discharge_output_data_path,
                                                              "rid0003_mpiom_data_moc_mm_7500-7999_mean.nc")
        ice6g_rdirs_data_filename=os.path.join(self.river_discharge_output_data_path,
                                               "rid0004_mpiom_data_moc_mm_7500-7999_mean.nc")
        difference_on_ocean_grid_filename=os.path.join(self.river_discharge_output_data_path,
                                                       "rid0004meanminus0003mean_mpiom_data"
                                                       "_moc_mm_7500-7999.nc")
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
        x = np.linspace(-89.5,89.5,num=180)
        atlantic_wfl_temporalmean_diff = -np.mean(atlantic_wfl_diff,axis=0)[0,:,0]
        indopacific_wfl_temporalmean_diff = -np.mean(indopacific_wfl_diff,axis=0)[0,:,0]
        atlantic_wfl_temporalmean_ext = -np.mean(atlantic_wfl_ext,axis=0)[0,:,0]
        indopacific_wfl_temporalmean_ext = -np.mean(indopacific_wfl_ext,axis=0)[0,:,0]
        atlantic_wfl_temporalmean_ice6g = -np.mean(atlantic_wfl_ice6g,axis=0)[0,:,0]
        indopacific_wfl_temporalmean_ice6g = -np.mean(indopacific_wfl_ice6g,axis=0)[0,:,0]
        #Using small figsize to increase line thickness and font size
        scale_factor = 0.70
        plt.figure(figsize=(12*scale_factor,18*scale_factor))
        gs = gridspec.GridSpec(3,1)
        ax1 = plt.subplot(gs[0])
        ax1.plot(atlantic_wfl_temporalmean_ext,x,
                 label='Extended present day\nriver directions',color='#af8dc3',
                 linewidth=1.5)
        ax1.plot(atlantic_wfl_temporalmean_ice6g,x,
                 label='ICE6G river directions',color='#7fbf7b',linewidth=1.5)
        ax1.set_ylabel("Latitude",size=15)
        ax1.set_xlabel("Implied southward ocean freshwater transport ($m^{3}s^{-1}$)",
                       size=15)
        ax1.set_ylim(-90,90)
        ax1.set_xlim(-700000,700000)
        ax1.set_yticks([-90,-60,-30,0,30,60,90])
        ax1.yaxis.set_major_formatter(ticker.\
                                      FuncFormatter(pts.LatAxisFormatter(yoffset=181,
                                                                         scale_factor=-0.5,
                                                                         precision=0)))
        ax1.legend(numpoints=1,loc=4,prop={'size': 15})
        ax1.set_title("Atlantic",size=15)
        ax1.annotate(xy=(0, -34),xycoords="axes points",s="(a)",fontsize=20)
        ax2 = plt.subplot(gs[1])
        ax2.plot(indopacific_wfl_temporalmean_ext,x,
                 label='Extended present day\nriver directions',color='#af8dc3',
                 linewidth=1.5)
        ax2.plot(indopacific_wfl_temporalmean_ice6g,x,
                 label='ICE6G river directions',color='#7fbf7b',linewidth=1.5)
        ax2.set_ylabel("Latitude",size=15)
        ax2.set_xlabel("Implied southward ocean freshwater transport ($m^{3}s^{-1}$)",
                       size=15)
        ax2.set_ylim(-90,90)
        ax2.set_xlim(-700000,700000)
        ax2.set_yticks([-90,-60,-30,0,30,60,90])
        ax2.yaxis.set_major_formatter(ticker.\
                                      FuncFormatter(pts.LatAxisFormatter(yoffset=181,
                                                                         scale_factor=-0.5,
                                                                         precision=0)))
        ax2.set_title("Indo-Pacific",size=15)
        ax2.legend(numpoints=1,loc=2,prop={'size': 15})
        ax2.annotate(xy=(0, -34),xycoords="axes points",s="(b)",fontsize=20)
        ax3 = plt.subplot(gs[2])
        ax3.plot(atlantic_wfl_temporalmean_diff,x,
                 label='Atlantic',color='#ef8a62',linewidth=1.5)
        ax3.plot(indopacific_wfl_temporalmean_diff,x,
                 label='Indo-Pacific',color='#67a9cf',linewidth=1.5)
        ax3.set_ylabel("Latitude",size=15)
        ax3.set_xlabel(r'Change in implied southward ocean freshwater transport ($m^{3}s^{-1}$)',
                       size=15)
        ax3.set_ylim(-90,90)
        ax3.set_xlim(-50000,70000)
        ax3.set_yticks([-90,-60,-30,0,30,60,90])
        ax3.yaxis.set_major_formatter(ticker.\
                                      FuncFormatter(pts.LatAxisFormatter(yoffset=181,
                                                                         scale_factor=-0.5,
                                                                         precision=0)))
        ax3.legend(loc=4,numpoints=1,prop={'size': 15})
        ax3.set_title('$($Value using ICE6G river directions$)-$\n$($Value using extended present day river directions$)$',
                      size=15)
        ax3.annotate(xy=(0, -34),xycoords="axes points",s="(c)",fontsize=20)
        plt.tight_layout()
        if self.save:
            plt.savefig(path.join(self.save_path,"implied_freshwater_latitudinal_sums.pdf"),dpi=300)

    def ocean_fresh_water_input_plots_extended_present_day_vs_ice6g_rdirs(self):
        extended_present_day_rdirs_data_filename=os.path.join(self.river_discharge_output_data_path,
                                                              "rid0003_mpiom_data_moc_mm_7500-7999_mean.nc")
        ice6g_rdirs_data_filename=os.path.join(self.river_discharge_output_data_path,
                                               "rid0004_mpiom_data_moc_mm_7500-7999_mean.nc")
        difference_on_ocean_grid_filename=os.path.join(self.river_discharge_output_data_path,
                                                       "rid0004meanminus0003mean_mpiom_data"
                                                       "_moc_mm_7500-7999.nc")
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
        x = np.linspace(-89.5,89.5,num=180)
        atlantic_wfl_temporalmean_diff = -np.mean(atlantic_wfl_diff,axis=0)[0,:,0]
        #Note that the version of numpy we are using does not support suppling an array as
        #a varargs argument to gradient - doing so will give erroneous results
        atlantic_fwf_temporalmean_diff = np.gradient(atlantic_wfl_temporalmean_diff,-1)
        indopacific_wfl_temporalmean_diff = -np.mean(indopacific_wfl_diff,axis=0)[0,:,0]
        #Note that the version of numpy we are using does not support suppling an array as
        #a varargs argument to gradient - doing so will give erroneous results
        indopacific_fwf_temporalmean_diff = np.gradient(indopacific_wfl_temporalmean_diff,-1)
        atlantic_wfl_temporalmean_ext = -np.mean(atlantic_wfl_ext,axis=0)[0,:,0]
        #Note that the version of numpy we are using does not support suppling an array as
        #a varargs argument to gradient - doing so will give erroneous results
        atlantic_fwf_temporalmean_ext = np.gradient(atlantic_wfl_temporalmean_ext,-1)
        indopacific_wfl_temporalmean_ext = -np.mean(indopacific_wfl_ext,axis=0)[0,:,0]
        #Note that the version of numpy we are using does not support suppling an array as
        #a varargs argument to gradient - doing so will give erroneous results
        indopacific_fwf_temporalmean_ext = np.gradient(indopacific_wfl_temporalmean_ext,-1)
        atlantic_wfl_temporalmean_ice6g = -np.mean(atlantic_wfl_ice6g,axis=0)[0,:,0]
        #Note that the version of numpy we are using does not support suppling an array as
        #a varargs argument to gradient - doing so will give erroneous results
        atlantic_fwf_temporalmean_ice6g = np.gradient(atlantic_wfl_temporalmean_ice6g,-1)
        indopacific_wfl_temporalmean_ice6g = -np.mean(indopacific_wfl_ice6g,axis=0)[0,:,0]
        #Note that the version of numpy we are using does not support suppling an array as
        #a varargs argument to gradient - doing so will give erroneous results
        indopacific_fwf_temporalmean_ice6g = np.gradient(indopacific_wfl_temporalmean_ice6g,-1)
        #Using small figsize to increase line thickness and font size
        scale_factor = 0.70
        plt.figure(figsize=(12*scale_factor,18*scale_factor))
        gs = gridspec.GridSpec(3,1)
        ax1 = plt.subplot(gs[0])
        ax1.plot(atlantic_fwf_temporalmean_ext,x,
                 label='Extended present day\nriver directions',color='#af8dc3',linewidth=1.5)
        ax1.plot(atlantic_fwf_temporalmean_ice6g,x,
                 label='ICE6G river directions',color='#7fbf7b',linewidth=1.5)
        ax1.set_ylabel("Latitude",size=15)
        ax1.set_xlabel("Freshwater flux ($m^{3}s^{-1}\mathrm{deg}^{-1}$)",
                       size=15)
        ax1.set_ylim(-90,90)
        ax1.set_xlim(-90000,60000)
        ax1.set_yticks([-90,-60,-30,0,30,60,90])
        ax1.yaxis.set_major_formatter(ticker.\
                                      FuncFormatter(pts.LatAxisFormatter(yoffset=181,
                                                                         scale_factor=-0.5,
                                                                         precision=0)))
        ax1.legend(numpoints=1,loc=3,prop={'size': 15})
        ax1.set_title("Atlantic",size=15)
        ax1.annotate(xy=(0, -34),xycoords="axes points",s="(a)",fontsize=20)
        ax2 = plt.subplot(gs[1])
        ax2.plot(indopacific_fwf_temporalmean_ext,x,
                 label='Extended\npresent\nday river\ndirections',color='#af8dc3',linewidth=1.5)
        ax2.plot(indopacific_fwf_temporalmean_ice6g,x,
                 label='ICE6G river\ndirections',color='#7fbf7b',linewidth=1.5)
        ax2.set_ylabel("Latitude",size=15)
        ax2.set_xlabel("Freshwater flux ($m^{3}s^{-1}\mathrm{deg}^{-1}$)",
                       size=15)
        ax2.set_ylim(-90,90)
        ax2.set_xlim(-90000,60000)
        ax2.set_yticks([-90,-60,-30,0,30,60,90])
        ax2.yaxis.set_major_formatter(ticker.\
                                      FuncFormatter(pts.LatAxisFormatter(yoffset=181,
                                                                         scale_factor=-0.5,
                                                                         precision=0)))
        ax2.set_title("Indo-Pacific",size=15)
        ax2.legend(numpoints=1,loc=2,prop={'size': 12})
        ax2.annotate(xy=(0, -34),xycoords="axes points",s="(b)",fontsize=20)
        ax3 = plt.subplot(gs[2])
        ax3.plot(atlantic_fwf_temporalmean_diff,x,
                 label='Atlantic',color='#ef8a62',linewidth=1.5)
        ax3.plot(indopacific_fwf_temporalmean_diff,x,
                 label='Indo-Pacific',color='#67a9cf',linewidth=1.5)
        ax3.set_ylabel("Latitude",size=15)
        ax3.set_xlabel(r'Change in freshwater flux ($m^{3}s^{-1}\mathrm{deg}^{-1}$)',
                       size=15)
        ax3.set_ylim(-90,90)
        ax3.set_xlim(-16000,24000)
        ax3.set_yticks([-90,-60,-30,0,30,60,90])
        ax3.yaxis.set_major_formatter(ticker.\
                                      FuncFormatter(pts.LatAxisFormatter(yoffset=181,
                                                                         scale_factor=-0.5,
                                                                         precision=0)))
        ax3.legend(loc=3,numpoints=1,prop={'size': 15})
        ax3.set_title('$($Value using ICE6G river directions$)-$\n$($Value using extended present day river directions$)$',
                      size=15)
        ax3.annotate(xy=(0, -34),xycoords="axes points",s="(c)",fontsize=20)
        plt.tight_layout()
        if self.save:
            plt.savefig(path.join(self.save_path,"freshwater_flux_latitudinal_sums.pdf"),dpi=300)

    def four_flowmap_sections_from_deglaciation(self):
        """ """
        time_one=14000
        time_two=13600
        time_three=12700
        time_four=12630
        flowmap_one_filename = os.path.join(self.plots_data_dir,
                                "30min_flowtocell_pmu0171a_{}.nc".format(time_one))
        flowmap_two_filename = os.path.join(self.plots_data_dir,
                                "30min_flowtocell_pmu0171b_{}.nc".format(time_two))
        flowmap_three_filename = os.path.join(self.plots_data_dir,
                                  "30min_flowtocell_pmu0171b_{}.nc".format(time_three))
        flowmap_four_filename = os.path.join(self.plots_data_dir,
                                  "30min_flowtocell_pmu0171b_{}.nc".format(time_four))
        catchments_one_filename = os.path.join(self.plots_data_dir,
                                               "30min_catchments_pmu0171a_{}.nc".format(time_one))
        catchments_two_filename = os.path.join(self.plots_data_dir,
                                               "30min_catchments_pmu0171b_{}.nc".format(time_two))
        catchments_three_filename = os.path.join(self.plots_data_dir,
                                               "30min_catchments_pmu0171b_{}.nc".format(time_three))
        catchments_four_filename = os.path.join(self.plots_data_dir,
                                               "30min_catchments_pmu0171b_{}.nc".format(time_four))
        lsmask_one_filename = os.path.join(self.plots_data_dir,
                                  "hdpara_{}k.nc".format(time_one))
        lsmask_two_filename = os.path.join(self.plots_data_dir,
                                  "hdpara_{}k.nc".format(time_two))
        lsmask_three_filename = os.path.join(self.plots_data_dir,
                                    "hdpara_{}k.nc".format(time_three))
        lsmask_four_filename = os.path.join(self.plots_data_dir,
                                   "hdpara_{}k.nc".format(time_four))
        glac_mask_one_filename = os.path.join(self.plots_data_dir,
                                              "glac01_{}.nc".format(time_one))
        glac_mask_two_filename = os.path.join(self.plots_data_dir,
                                              "glac01_{}.nc".format(time_two))
        glac_mask_three_filename = os.path.join(self.plots_data_dir,
                                              "glac01_{}.nc".format(time_three))
        glac_mask_four_filename = os.path.join(self.plots_data_dir,
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
        fig = plt.figure(figsize=(12.4,9))
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
        ax1.set_title("{} years before present".format(time_one))
        rc_pts.simple_thresholded_data_only_flowmap(ax2,flowmap_two,lsmask_two,threshold=75,
                                                    glacier_mask=glac_mask_hd_two,
                                                    catchments=catchments_two,
                                                    catchnumone=4,
                                                    catchnumtwo=30,
                                                    catchnumthree=51,
                                                    bounds=bounds,
                                                    colors=self.colors)
        ax2.set_title("{} years before present".format(time_two))
        rc_pts.simple_thresholded_data_only_flowmap(ax3,flowmap_three,lsmask_three,threshold=75,
                                                    glacier_mask=glac_mask_hd_three,
                                                    catchments=catchments_three,
                                                    catchnumone=3,
                                                    catchnumtwo=21,
                                                    catchnumthree=8,
                                                    bounds=bounds,
                                                    colors=self.colors)
        ax3.set_title("{} years before present".format(time_three))
        rc_pts.simple_thresholded_data_only_flowmap(ax4,flowmap_four,lsmask_four,threshold=75,
                                                    glacier_mask=glac_mask_hd_four,
                                                    catchments=catchments_four,
                                                    catchnumone=20,
                                                    catchnumtwo=8,
                                                    catchnumthree=4,
                                                    bounds=bounds,
                                                    colors=self.colors)
        ax4.set_title("{} years before present".format(time_four))
        gs.tight_layout(fig,rect=(0,0.1,1,1))
        if self.save:
          plt.savefig(path.join(self.save_path,"four_timeslice_catchment_and_flowmap_plot.pdf"),dpi=300)

def main():
    plots_for_GMD_paper =  PlotsForGMDPaper(False)
    #plots_for_GMD_paper.danube_catchment_correction_plots()
    #plots_for_GMD_paper.comparison_of_manually_corrected_HD_rdirs_vs_automatically_generated_10min_rdirs()
    #plots_for_GMD_paper.comparison_of_modern_river_directions_10_minute_original_vs_HD_upscaled()
        #With catchments version is better
        #plots_for_GMD_paper.compare_upscaled_automatically_generated_rdirs_to_HD_manually_corrected_rdirs()
    #plots_for_GMD_paper.compare_upscaled_automatically_generated_rdirs_to_HD_manually_corrected_rdirs_with_catchments()
        #With catchments version is better
        #plots_for_GMD_paper.compare_present_day_and_lgm_river_directions()
    plots_for_GMD_paper.compare_present_day_and_lgm_river_directions_with_catchments()
        #Only for supplementary material
        #plots_for_GMD_paper.discharge_plot()
    #plots_for_GMD_paper.ocean_pem_plots_extended_present_day_vs_ice6g_rdirs()
    #plots_for_GMD_paper.ocean_fresh_water_input_plots_extended_present_day_vs_ice6g_rdirs()
    #plots_for_GMD_paper.four_flowmap_sections_from_deglaciation()
    plt.show()

if __name__ == '__main__':
    main()
