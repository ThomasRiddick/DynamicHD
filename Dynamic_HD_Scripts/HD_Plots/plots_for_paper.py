'''
Created on Mar 2, 2017

@author: thomasriddick
'''
from plots_library import (Plots,HDparameterPlots,HDOutputPlots,OutflowPlots, #@UnusedImport
                           FlowMapPlots,FlowMapPlotsWithCatchments,OrographyPlots, #@UnusedImport  
                           SimpleOrographyPlots, Ice5GComparisonPlots) #@UnusedImport
import os
import os.path as path
import matplotlib.pyplot as plt
from matplotlib import gridspec 

class PlotsForGMDPaper(OutflowPlots,FlowMapPlotsWithCatchments,HDOutputPlots):
    """Plots for GMD Paper"""
    
    save_path = "/Users/thomasriddick/Documents/plots/Dynamic HD/plots_generated_for_paper"

    def __init__(self, save):
        color_palette_to_use="gmd_paper"
        super(PlotsForGMDPaper,self).__init__(save,color_palette_to_use)
        """Class constructor"""
        
    def danube_catchment_correction_plots(self):
        """Three plots of the Danube catchment showing the effect of applying correction an orography."""

        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.rmouth_outflow_data_directory,
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
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.rmouth_outflow_data_directory,
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
        print "Total cumulative flow threshold used for danube plot {0}".\
            format(corr_data_plotter.get_flowtocell_threshold())
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
            plt.savefig(path.join(self.save_path,"danube_comparison.png"))

    def comparison_of_manually_corrected_HD_rdirs_vs_automatically_generated_10min_rdirs(self):
        data_creation_datetime="20170517_003802"
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.rmouth_outflow_data_directory,
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
        plt.figure(figsize=(12,9))
        gs = gridspec.GridSpec(3,5,width_ratios=[22,1,22,2,12],
                                   height_ratios=[4,1,2])
        ax0 = plt.subplot(gs[:-1,0])
        ax1 = plt.subplot(gs[1:,2:])
        ax2 = plt.subplot(gs[0,2])
        ax3 = plt.subplot(gs[0,3])
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
            plt.savefig(path.join(self.save_path,"me_mi_ni_10min_vs_man_corr.png"))
        
    def comparison_of_modern_river_directions_10_minute_original_vs_HD_upscaled(self):
        data_creation_datetime_directly_upscaled="20170517_003802"
        data_creation_datetime_rdirs_upscaled = "20170517_004128"
        ice5g_ALG4_sinkless_all_points_0k_river_flow_dir_upsc_field = os.path.join(self.rmouth_outflow_data_directory,
            "rmouthflows_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson"
            "_lsmask_0k_upscale_rdirs_{0}_updated.nc"\
                .format(data_creation_datetime_rdirs_upscaled))
        ice5g_ALG4_sinkless_all_points_0k_dir_upsc_field = os.path.join(self.rmouth_outflow_data_directory,
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
        catchment_plotters[0].set_legend(False)
        catchment_plotters[1].set_cax(ax2)
        ax2.tick_params(labelsize=20) 
        catchment_plotters[0](ax0)
        catchment_plotters[1](ax1)
        catchment_plotters[0].apply_axis_locators_and_formatters(ax0)
        catchment_plotters[1].apply_axis_locators_and_formatters(ax1)
        if self.save:
            plt.savefig(path.join(self.save_path,"me_mi_upscaling_comp.png"))
        
    def compare_upscaled_automatically_generated_rdirs_to_HD_manually_corrected_rdirs(self):
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale'
                                  '_rdirs_20161031_113238_updated.nc')
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
        
    def compare_upscaled_automatically_generated_rdirs_to_HD_manually_corrected_rdirs_with_catchments(self):
        ref_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_corrected_HD_rdirs_post_processing_20160427_141158.nc')
        data_filename=os.path.join(self.flow_maps_data_directory,
                                  'flowmap_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless'
                                  '_glcc_olson_lsmask_0k_upscale_rdirs_20170517_004128_updated.nc')
        lsmask_filename=os.path.join(self.ls_masks_data_directory,"generated",
                                     "ls_mask_recreate_connected_HD_lsmask_"
                                     "from_glcc_olson_data_20170513_195421.nc")
        corrected_hd_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory,
                                                             "rmouthflows_corrected_HD_rdirs_post_processing_20160427_141158.nc")
        upscaled_rdirs_rmouthoutflow_file = os.path.join(self.rmouth_outflow_data_directory, 
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
                                                            first_datasource_name="Model 1",
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
                                                            second_datasource_name="Model 2",use_title=False,
                                                            remove_antartica=True,
                                                            glacier_mask_filename=glacier_mask_filename,
                                                            glacier_mask_grid_type='LatLong10min',
                                                            flip_glacier_mask=True,
                                                            rotate_glacier_mask=True,
                                                            grid_type='HD')
        plt.tight_layout()
        if self.save:
            plt.savefig(path.join(self.save_path,"global_man_corr_vs_auto_gen_upscaled_global.png"))
    
    def compare_present_day_and_lgm_river_directions(self):
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
                                              second_datasource_name="LGM",
                                              add_title=False)
        plt.tight_layout()
        if self.save:
            plt.savefig(path.join(self.save_path,"lgm_vs_present_global_comp.png"))
            
    def compare_present_day_and_lgm_river_directions_with_catchments(self):
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
                                                            minflowcutoff=100,
                                                            flip_data=False,
                                                            rotate_data=True,
                                                            flip_ref=False,
                                                            rotate_ref=True,
                                                            lsmask_has_same_orientation_as_ref=False,
                                                            invert_ls_mask=True,
                                                            first_datasource_name="Present Day",
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
                                                            glacier_mask_grid_type='LatLong10min',
                                                            flip_glacier_mask=True,
                                                            rotate_glacier_mask=True,
                                                            grid_type='HD') 
        plt.tight_layout()
        if self.save:
            plt.savefig(path.join(self.save_path,"lgm_vs_present_global_comp.png"))
        
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
        
def main():
    plots_for_GMD_paper =  PlotsForGMDPaper(True)
    plots_for_GMD_paper.danube_catchment_correction_plots()
    plots_for_GMD_paper.comparison_of_manually_corrected_HD_rdirs_vs_automatically_generated_10min_rdirs()
    plots_for_GMD_paper.comparison_of_modern_river_directions_10_minute_original_vs_HD_upscaled()
        #With catchments version is better
        #plots_for_GMD_paper.compare_upscaled_automatically_generated_rdirs_to_HD_manually_corrected_rdirs()
    plots_for_GMD_paper.compare_upscaled_automatically_generated_rdirs_to_HD_manually_corrected_rdirs_with_catchments()
        #With catchments version is better
        #plots_for_GMD_paper.compare_present_day_and_lgm_river_directions()
    plots_for_GMD_paper.compare_present_day_and_lgm_river_directions_with_catchments()
        #Only for supplementary material
        #plots_for_GMD_paper.discharge_plot()
    plt.show()

if __name__ == '__main__':
    main()