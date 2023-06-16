'''
Driving routines for a wide range of specific dynamic HD file generation runs
Created on Feb 24, 2016

@author: thomasriddick
'''

import inspect
import datetime
import subprocess
import numpy as np
import os.path as path
import shutil
import netCDF4
from subprocess import CalledProcessError
from Dynamic_HD_Scripts.tools import flow_to_grid_cell
from Dynamic_HD_Scripts.tools import compute_catchments
from Dynamic_HD_Scripts.tools import fill_sinks_driver
from Dynamic_HD_Scripts.tools import upscale_orography_driver
from Dynamic_HD_Scripts.tools import river_mouth_marking_driver
from Dynamic_HD_Scripts.tools import create_connected_lsmask_driver as cc_lsmask_driver
from Dynamic_HD_Scripts.tools import cotat_plus_driver
from Dynamic_HD_Scripts.tools import loop_breaker_driver
from Dynamic_HD_Scripts.utilities import utilities
from Dynamic_HD_Scripts.base import grid
from Dynamic_HD_Scripts.base import field
from Dynamic_HD_Scripts.base import iohelper
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.context import bash_scripts_path
from Dynamic_HD_Scripts.context import private_bash_scripts_path

class Dynamic_HD_Drivers(object):
    """Class that drives a wide variety of dynamic HD related scripts and programs

    Public Members:
    """

    def __init__(self):
        """Setup paths to various input and output data directories

        Arguments: None
        """

        data_dir = "/Users/thomasriddick/Documents/data/HDdata"
        rdirs_path_extension = "rdirs"
        rmouth_path_extension = "rmouths"
        orog_path_extension  = "orographys"
        weights_path_extension = 'remapweights'
        ls_masks_path_extension = 'lsmasks'
        update_masks_extension = 'updatemasks'
        rmouth_cumulative_flow_path_extension = 'rmouthflow'
        grid_path_extension = 'grids'
        flowmaps_path_extension = 'flowmaps'
        catchments_path_extension = 'catchmentmaps'
        truesinks_path_extension = 'truesinks'
        ls_seed_points_path_extension = 'lsseedpoints'
        orography_corrections_path_extension = 'orogcorrs'
        truesinks_modifications_path_extension = 'truesinksmods'
        intelligent_burning_regions_extension = 'intburnregions'
        orography_corrections_fields_path_extension = 'orogcorrsfields'
        null_fields_path_extension= 'nullfields'
        grid_areas_and_spacings_path_extension = 'gridareasandspacings'
        base_RFD_filename = "rivdir_vs_1_9_data_from_stefan.txt"
        parameter_path_extension = "params"
        flow_params_dirs_path_extension = "flowparams"
        hd_file_path_extension = 'hdfiles'
        hd_restart_file_path_extension = 'hdrestartfiles'
        js_bach_restart_file_path_extension = 'jsbachrestartfiles'
        paragen_code_copies_path_extension = 'paragencopies'
        minima_path_extension = 'minima'
        lakemask_path_extension= 'lakemasks'
        lake_parameter_file_extension = 'lakeparafiles'
        basin_catchment_numbers_file_extension = 'basin_catchment_numbers'
        lake_bathymetry_file_extension = 'lakebath'
        cotat_plus_parameters_path_extension = path.join(parameter_path_extension,'cotat_plus')
        orography_upscaling_parameters_path_extension = path.join(parameter_path_extension,
                                                                  'orography_upscaling')
        lake_and_hd_params_log_file_extension = "paramsfilepathlogs"
        self.base_RFD_filepath = path.join(data_dir,rdirs_path_extension,
                                           base_RFD_filename)
        self.orography_path = path.join(data_dir,orog_path_extension)
        self.upscaled_orography_filepath = path.join(self.orography_path,'upscaled','upscaled_orog_')
        self.tarasov_upscaled_orography_filepath = path.join(self.orography_path,'tarasov_upscaled','upscaled_orog_')
        self.generated_orography_filepath = path.join(self.orography_path,'generated','updated_orog_')
        self.corrected_orography_filepath = path.join(self.orography_path,'generated','corrected',
                                                      'corrected_orog_')
        self.rdir_path = path.join(data_dir,rdirs_path_extension)
        self.generated_rdir_filepath = path.join(self.rdir_path,'generated','updated_RFDs_')
        self.upscaled_generated_rdir_filepath = path.join(self.rdir_path,'generated','upscaled',
                                                          'upscaled_rdirs_')
        self.generated_rdir_with_outflows_marked_filepath = path.join(self.rdir_path,
                                                                      'generated_outflows_marked',
                                                                      'updated_RFDs_')
        self.update_masks_path = path.join(data_dir,update_masks_extension)
        self.generated_update_masks_filepath = path.join(self.update_masks_path,'update_mask_')
        self.weights_path = path.join(data_dir,weights_path_extension)
        self.grids_path = path.join(data_dir,grid_path_extension)
        self.ls_masks_path = path.join(data_dir,ls_masks_path_extension)
        self.flowmaps_path = path.join(data_dir,flowmaps_path_extension)
        self.generated_flowmaps_filepath = path.join(self.flowmaps_path,'flowmap_')
        self.upscaled_flowmaps_filepath = path.join(self.flowmaps_path,'upscaled','flowmap_')
        self.catchments_path = path.join(data_dir,catchments_path_extension)
        self.generated_catchments_path = path.join(self.catchments_path,'catchmentmap_')
        self.upscaled_catchments_path = path.join(self.catchments_path,'upscaled','catchmentmap_')
        self.generated_ls_mask_filepath = path.join(self.ls_masks_path,'generated','ls_mask_')
        self.generated_gaussian_ls_mask_filepath = path.join(self.ls_masks_path,'generated','gaussian',
                                                             'ls_mask_')
        self.rmouth_path =  path.join(data_dir,rmouth_path_extension)
        self.generated_rmouth_path = path.join(self.rmouth_path,'rmouthmap_')
        self.rmouth_cumulative_flow_path = path.join(data_dir,rmouth_cumulative_flow_path_extension)
        self.generated_rmouth_cumulative_flow_path = path.join(self.rmouth_cumulative_flow_path,
                                                               'rmouthflows_')
        self.upscaled_rmouth_cumulative_flow_path = path.join(self.rmouth_cumulative_flow_path,
                                                              'upscaled','rmouthflows_')
        self.truesinks_path = path.join(data_dir,truesinks_path_extension)
        self.generated_truesinks_path = path.join(self.truesinks_path,'truesinks_')
        self.ls_seed_points_path = path.join(data_dir,ls_seed_points_path_extension)
        self.generated_ls_seed_points_path = path.join(self.ls_seed_points_path,
                                                       'lsseedpoints_')
        self.orography_corrections_path = path.join(data_dir,orography_corrections_path_extension)
        self.copied_orography_corrections_filepath = path.join(self.orography_corrections_path,
                                                               'copies','orog_corr_')
        self.truesinks_modifications_filepath = path.join(data_dir,
                                                          truesinks_modifications_path_extension)
        self.intelligent_burning_regions_path = path.join(data_dir,
                                                          intelligent_burning_regions_extension)
        self.copied_intelligent_burning_regions_path = path.join(self.intelligent_burning_regions_path,
                                                                 'copies','int_burn_regions_')
        self.cotat_plus_parameters_path = path.join(data_dir,
                                                    cotat_plus_parameters_path_extension)
        self.copied_cotat_plus_parameters_path = path.join(self.cotat_plus_parameters_path,
                                                           'copies','cotat_plus_params_')
        self.orography_upscaling_parameters_path = path.join(data_dir,
                                                             orography_upscaling_parameters_path_extension)
        self.copied_orography_upscaling_parameters_path = path.join(self.orography_upscaling_parameters_path,
                                                                    'copies','orography_upscaling_params_')
        self.orography_corrections_fields_path = path.join(data_dir,
                                                           orography_corrections_fields_path_extension)
        self.generated_orography_corrections_fields_path = path.join(self.orography_corrections_fields_path,
                                                                     'orog_corrs_field_')
        self.null_fields_filepath = path.join(data_dir,null_fields_path_extension)
        self.flow_params_dirs_path = path.join(data_dir,flow_params_dirs_path_extension)
        self.grid_areas_and_spacings_filepath = path.join(data_dir,
                                                          grid_areas_and_spacings_path_extension)
        self.hd_file_path = path.join(data_dir,hd_file_path_extension)
        self.generated_hd_file_path= path.join(self.hd_file_path,'generated','hd_file_')
        self.hd_restart_file_path = path.join(data_dir,hd_restart_file_path_extension)
        self.generated_hd_restart_file_path = path.join(self.hd_restart_file_path,
                                                        'generated','hd_restart_file_')
        self.js_bach_restart_filepath = path.join(data_dir,js_bach_restart_file_path_extension)
        self.generated_js_bach_restart_filepath = path.join(self.js_bach_restart_filepath,
                                                            'generated','updated_')
        self.paragen_code_copies_path = path.join(data_dir,paragen_code_copies_path_extension)
        self.generated_paragen_code_copies_path = path.join(self.paragen_code_copies_path,
                                                            "paragen_copy_")
        self.minima_filepath = path.join(data_dir,minima_path_extension)
        self.generated_minima_filepath = path.join(self.minima_filepath,'minima_')
        self.lakemask_filepath = path.join(data_dir,lakemask_path_extension)
        self.lake_parameter_file_path = path.join(data_dir,
                                                  lake_parameter_file_extension)
        self.basin_catchment_numbers_path = path.join(data_dir,
                                                      basin_catchment_numbers_file_extension)
        self.lake_and_hd_params_log_path = path.join(data_dir,
                                                     lake_and_hd_params_log_file_extension)
        self.generated_lake_and_hd_params_log_path = path.join(self.lake_and_hd_params_log_path,
                                                               'generated','lake_and_hd_params_')
        self.lake_bathymetry_filepath = path.join(data_dir,lake_bathymetry_file_extension)
        self.hd_grid_filepath = path.join(self.grids_path,"hdmodel2d_griddes")
        self.half_degree_grid_filepath = path.join(self.grids_path,"grid_0_5.txt")
        self.ten_minute_grid_filepath = path.join(self.grids_path,"grid_10min.txt")
        self.thirty_second_grid_filepath= path.join(self.grids_path,"grid_30sec.txt")
        self.hd_grid_ls_mask_filepath = path.join(self.ls_masks_path,
                                                  "lsmmaskvonGR30.srv")
        self.hd_truesinks_filepath = path.join(self.truesinks_path,
                                               "truesinks_extract_true_sinks_from_"
                                               "corrected_HD_rdirs_20160527_105218.nc")
        #Would only need to revert to the old value if existing file was deleted and need to recreated
        #by running HD model for one year using ref file from current model
        #self.base_hd_restart_file = path.join(self.hd_restart_file_path,"hd_restart_file_from_current_model.nc")
        self.base_hd_restart_file = path.join(self.hd_restart_file_path,"hd_restart_from_hd_file_ten_minute_data_from_virna_"
                                           "0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_"
                                           "135934_after_one_year_running.nc")
        #Would only need to revert to the old value if existing file was deleted and need to recreated
        #by running HD model for one year using ref file from current model
        #self.ref_hd_paras_file = path.join(self.hd_file_path,"hdpara_file_from_current_model.nc")
        self.ref_hd_paras_file = path.join(self.hd_file_path,"hd_file_ten_minute_data_from_virna_0k_ALG4_sinkless_no_"
                                           "true_sinks_oceans_lsmask_plus_upscale_rdirs_20170116_123858_to_use_as_"
                                           "hdparas_ref.nc")
        self.base_js_bach_restart_file_T106 = path.join(self.js_bach_restart_filepath,
                                                        "jsbach_T106_11tiles_5layers_1976.nc")
        self.base_js_bach_restart_file_T63 = path.join(self.js_bach_restart_filepath,
                                                       "jsbach_T63GR15_11tiles_5layers_1976.nc")

    @staticmethod
    def _generate_file_label():
        """Generate a label for files based on the name of routine they are generated in and the date/time it is called

        Arguments: None
        Returns: a string containing the name of the routine that called this routine and the date end time this
        routine was called at
        """

        return "_".join([str(inspect.stack()[1][3]),datetime.datetime.now().strftime("%Y%m%d_%H%M%S")])

    @classmethod
    def generate_present_day_river_directions_with_original_true_sink_set(cls,
                                                                          orography_filepath,
                                                                          landsea_mask_filepath,
                                                                          glacier_mask_filepath,
                                                                          orography_corrections_filepath,
                                                                          output_dir,
                                                                          orography_fieldname="Topo",
                                                                          landsea_mask_fieldname="slm",
                                                                          glacier_mask_fieldname="glac",
                                                                          orography_corrections_fieldname="orog",
                                                                          truesinks_fieldname="true_sinks"):
        truesinks_filename = ("/Users/thomasriddick/Documents/data/HDdata/truesinks/"
                              "truesinks_ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson_lsmask_0k_20191014_173825_with_grid.nc")
        cls.generate_present_day_river_directions_with_true_sinks(orography_filepath,
                                                                  landsea_mask_filepath,
                                                                  glacier_mask_filepath,
                                                                  orography_corrections_filepath,
                                                                  output_dir,
                                                                  truesinks_filepath=
                                                                  truesinks_filename,
                                                                  orography_fieldname=orography_fieldname,
                                                                  landsea_mask_fieldname=landsea_mask_fieldname,
                                                                  glacier_mask_fieldname=glacier_mask_fieldname,
                                                                  orography_corrections_fieldname=orography_corrections_fieldname,
                                                                  truesinks_fieldname=truesinks_fieldname)

    @staticmethod
    def generate_present_day_river_directions_with_true_sinks(orography_filepath,
                                                              landsea_mask_filepath,
                                                              glacier_mask_filepath,
                                                              orography_corrections_filepath,
                                                              output_dir,
                                                              truesinks_filepath,
                                                              orography_fieldname="Topo",
                                                              landsea_mask_fieldname="slm",
                                                              glacier_mask_fieldname="glac",
                                                              orography_corrections_fieldname="orog",
                                                              truesinks_fieldname="true_sinks"):
        cotat_plus_parameters_filepath = ("/Users/thomasriddick/Documents/workspace/"
                                          "Dynamic_HD_Code/Dynamic_HD_Resources/cotat_plus_standard_params.nl")
        present_day_reference_orography_filepath = ("/Users/thomasriddick/Documents/data/hd_ancillary_data_dirs/"
                                                    "HDancillarydata/ice5g_v1_2_00_0k_10min.nc")
        rebased_orography_filename = path.join(output_dir,"10min_rebased_orog.nc")
        first_corrected_orography_filename = path.join(output_dir,"10min_corrected_orog_one.nc")
        second_corrected_orography_filename = path.join(output_dir,"10min_corrected_orog_two.nc")
        fine_rdirs_filename = path.join(output_dir,"10min_rdirs.nc")
        fine_catchments_filename = path.join(output_dir,"10min_catchments.nc")
        fine_cflow_filename = path.join(output_dir,"10min_flowtocell.nc")
        coarse_catchments_filename = path.join(output_dir,"30min_pre_loop_removal_catchments.nc")
        loops_log_filename = path.join(output_dir,"30min_pre_loop_removal_loops_log.txt")
        coarse_cflow_filename = path.join(output_dir,"30min_pre_loop_removal_flowtocell.nc")
        coarse_rdirs_filename = path.join(output_dir,"30min_pre_loop_removal_rdirs.nc")
        updated_coarse_rdirs_filename = path.join(output_dir,"30min_rdirs.nc")
        updated_coarse_catchments_filename = path.join(output_dir,"30min_catchments.nc")
        updated_loops_log_filename = path.join(output_dir,"30min_loops_log.txt")
        updated_coarse_cflow_filename = path.join(output_dir,"30min_flowtocell.nc")
        updated_coarse_rmouth_cflow_filename = path.join(output_dir,"30min_rmouth_flowtocell.nc")
        original_orography_fieldname = orography_fieldname
        present_day_reference_orography_fieldname = "orog"
        utilities.advanced_rebase_orography_driver(orography_filepath,
                                                   present_day_base_orography_filename=orography_filepath,
                                                   present_day_reference_orography_filename=
                                                   present_day_reference_orography_filepath,
                                                   rebased_orography_filename=
                                                   rebased_orography_filename ,
                                                   orography_fieldname=original_orography_fieldname,
                                                   present_day_base_orography_fieldname=
                                                   original_orography_fieldname,
                                                   present_day_reference_orography_fieldname=
                                                   present_day_reference_orography_fieldname,
                                                   rebased_orography_fieldname="z")
        utilities.advanced_apply_orog_correction_field(rebased_orography_filename ,
                                                       orography_corrections_filepath,
                                                       first_corrected_orography_filename,
                                                       original_orography_fieldname="z",
                                                       orography_corrections_fieldname=
                                                       orography_corrections_fieldname,
                                                       corrected_orography_fieldname="z")
        utilities.\
            advanced_replace_corrected_orog_with_orig_for_glcted_grid_points_drivers(first_corrected_orography_filename,
                rebased_orography_filename,
                glacier_mask_filepath,
                second_corrected_orography_filename,
                input_corrected_orography_fieldname="z",
                input_original_orography_fieldname="z",
                input_glacier_mask_fieldname=glacier_mask_fieldname,
                out_orography_fieldname="z")
        fill_sinks_driver.advanced_sinkless_flow_directions_generator(second_corrected_orography_filename,
                                                                      fine_rdirs_filename,
                                                                      fieldname="z",
                                                                      output_fieldname="rdir",
                                                                      ls_mask_filename=
                                                                      landsea_mask_filepath,
                                                                      truesinks_filename=
                                                                      truesinks_filepath,
                                                                      catchment_nums_filename=
                                                                      fine_catchments_filename,
                                                                      ls_mask_fieldname=landsea_mask_fieldname,
                                                                      truesinks_fieldname=truesinks_fieldname,
                                                                      catchment_fieldname="catch")
        flow_to_grid_cell.advanced_main(rdirs_filename=fine_rdirs_filename,
                                        output_filename=fine_cflow_filename,
                                        rdirs_fieldname="rdir",
                                        output_fieldname="acc")
        cotat_plus_driver.advanced_cotat_plus_driver(fine_rdirs_filename,
                                                     fine_cflow_filename,
                                                     coarse_rdirs_filename,
                                                     input_fine_rdirs_fieldname="rdir",
                                                     input_fine_total_cumulative_flow_fieldname="acc",
                                                     output_coarse_rdirs_fieldname="rdir",
                                                     cotat_plus_parameters_filepath=
                                                     cotat_plus_parameters_filepath,
                                                     scaling_factor=3)

        compute_catchments.advanced_main(coarse_rdirs_filename,"rdir",
                         coarse_catchments_filename,"catch",
                         loop_logfile=loops_log_filename,
                         use_cpp_alg=True)
        flow_to_grid_cell.advanced_main(rdirs_filename=coarse_rdirs_filename,
                                        output_filename=coarse_cflow_filename,
                                        rdirs_fieldname='rdir',
                                        output_fieldname='acc')
        loop_breaker_driver.advanced_loop_breaker_driver(
            input_coarse_rdirs_filepath=coarse_rdirs_filename,
            input_coarse_cumulative_flow_filepath=coarse_cflow_filename,
            input_coarse_catchments_filepath=coarse_catchments_filename,
            input_fine_rdirs_filepath=fine_rdirs_filename,
            input_fine_cumulative_flow_filepath=fine_cflow_filename,
            output_updated_coarse_rdirs_filepath=updated_coarse_rdirs_filename,
            input_coarse_rdirs_fieldname="rdir",
            input_coarse_cumulative_flow_fieldname="acc",
            input_coarse_catchments_fieldname="catch",
            input_fine_rdirs_fieldname="rdir",
            input_fine_cumulative_flow_fieldname="acc",
            output_updated_coarse_rdirs_fieldname="rdir",
            loop_nums_list_filepath=loops_log_filename,
            scaling_factor=3)
        compute_catchments.advanced_main(updated_coarse_rdirs_filename,"rdir",
                         updated_coarse_catchments_filename,"catch",
                         loop_logfile=updated_loops_log_filename,
                         use_cpp_alg=True)
        flow_to_grid_cell.advanced_main(rdirs_filename=updated_coarse_rdirs_filename,
                                        output_filename=updated_coarse_cflow_filename,
                                        rdirs_fieldname='rdir',
                                        output_fieldname='acc')
        river_mouth_marking_driver.\
        advanced_flow_to_rivermouth_calculation_driver(input_river_directions_filename=
                                                       updated_coarse_rdirs_filename,
                                                       input_flow_to_cell_filename=
                                                       updated_coarse_cflow_filename,
                                                       output_flow_to_river_mouths_filename=
                                                       updated_coarse_rmouth_cflow_filename,
                                                       input_river_directions_fieldname=
                                                       "rdir",
                                                       input_flow_to_cell_fieldname=
                                                       "acc",
                                                       output_flow_to_river_mouths_fieldname=
                                                       "acc")


    def _generate_hd_file(self,rdir_file,lsmask_file,null_file,area_spacing_file,
                          hd_grid_specs_file,output_file,paras_dir,production_run=False):
        """Generate an hdpara.nc file to be used as input to the standalone HD model or JSBACH

        Arguments:
        rdir_file: string; full path to the file containing the river direction to put in the hd file
        lsmask_file: string; full path to the file containing the land-sea mask (on the HD grid) to
            put in the hd file
        null_file: string; full path to a file containing a field set entirely to zero (on a HD grid)
        area_spacing_file: string; full path to a file containing the areas of grid boxes within the HD grid
        hd_grid_specs_file: string; full path to a file containing the grid specification for the HD grid
        output_file: string; full target path to write the output hd file to
        paras_dir: string; full path to a directory of srv parameters files produced by parameter generation
        production_run: bool; is this a production run (in which case don't compile paragen) or not?

        Returns: nothing

        Converts input file to revelant format and acts as a wrapper for the generate_output_file.sh script
        """

        if path.splitext(rdir_file)[1] != '.dat':
            self._convert_data_file_type(rdir_file,'.dat','HD')
        if path.splitext(lsmask_file)[1] != '.dat':
            self._convert_data_file_type(lsmask_file,'.dat','HD')
        if path.splitext(null_file)[1] != '.dat':
            self._convert_data_file_type(null_file,'.dat','HD')
        if path.splitext(area_spacing_file)[1] != '.dat':
            self._convert_data_file_type(area_spacing_file,'.dat','HD')
        if path.splitext(output_file)[1] != '.nc':
            raise UserWarning("Output filename doesn't have a netCDF extension as expected")
        try:
            print(subprocess.check_output([path.join(private_bash_scripts_path,
                                                     "generate_output_file.sh"),
                                           path.join(bash_scripts_path,
                                                     "bin"),
                                           path.join(private_bash_scripts_path,
                                                     "fortran"),
                                           path.splitext(rdir_file)[0] + ".dat",
                                           path.splitext(lsmask_file)[0] + ".dat",
                                           path.splitext(null_file)[0] + ".dat",
                                           path.splitext(area_spacing_file)[0] + ".dat",
                                           hd_grid_specs_file,output_file,paras_dir,
                                           "true" if production_run else "false"]))
        except CalledProcessError as cperror:
            raise RuntimeError("Failure in called process {0}; return code {1}; output:\n{2}".format(cperror.cmd,
                                                                                                     cperror.returncode,
                                                                                                     cperror.output))

    def _generate_flow_parameters(self,rdir_file,topography_file,inner_slope_file,lsmask_file,
                                  null_file,area_spacing_file,orography_variance_file,
                                  output_dir,paragen_source_label=None,production_run=False,
                                  grid_type="HD",**grid_kwargs):
        """Generate flow parameters files in a specified directory from given input

        Arguments:
        rdir_file: string; full path to the file containing the river direction to put in the hd file
        topography_file: string; full path to the HD orography to use to generate the parameters
        inner_slope_file: string; full path to the inner slopes values to use to generate the overland flow
            parameter
        lsmask_file: string; full path to the file containing the land-sea mask (on the HD grid) to
            put in the hd file
        null_file: string; full path to a file containing a field set entirely to zero (on a HD grid)
        area_spacing_file: string; full path to a file containing the areas of grid boxes within the HD grid
        orography_variance_file: string; full path to a file containing the variance of the orography
        output_dir: string; full path to directory to place the various srv output files from this script in
        paragen_source_label: string; a label for modified source files if not using an HD grid (optional)
        production_run: bool; is this a production run (in which case don't compile paragen) or not?
        grid_type: string; code for the grid type of the grid (optional)
        grid_kwargs: dictionary; key word dictionary specifying parameters of the grid (if required)
        Returns: nothing

        Converts input file to revelant format and acts as a wrapper for the parameter_generation_driver.sh
        script
        """

        parameter_generation_grid = grid.makeGrid(grid_type,**grid_kwargs)
        nlat,nlon = parameter_generation_grid.get_grid_dimensions()
        original_paragen_source_filepath = path.join(private_bash_scripts_path,"fortran",
                                                     "paragen.f")
        if (nlat != 360 or nlon != 720):
            paragen_source_filepath = self.generated_paragen_code_copies_path  + paragen_source_label
            with open(original_paragen_source_filepath,"r") as f:
                source = f.readlines()
            source.replace(360,str(nlat))
            source.replace(720,str(nlon))
            with open(paragen_source_filepath,"w") as f:
                f.writelines(source)
            paragen_bin_file = "paragen_nlat{0}_nlon{1}".format(nlat,nlon)
        else:
            paragen_source_filepath = original_paragen_source_filepath
            paragen_bin_file = "paragen"
        if path.splitext(rdir_file)[1] != '.dat':
            self._convert_data_file_type(rdir_file,'.dat','HD')
        if path.splitext(topography_file)[1] != '.dat':
            self._convert_data_file_type(topography_file,'.dat','HD')
        if path.splitext(inner_slope_file)[1] != '.dat':
            self._convert_data_file_type(inner_slope_file,'.dat','HD')
        if path.splitext(lsmask_file)[1] != '.dat':
            self._convert_data_file_type(lsmask_file,'.dat','HD')
        if path.splitext(null_file)[1] != '.dat':
            self._convert_data_file_type(null_file,'.dat','HD')
        if path.splitext(area_spacing_file)[1] != '.dat':
            self._convert_data_file_type(area_spacing_file,'.dat','HD')
        if path.splitext(orography_variance_file)[1] != '.dat':
            self._convert_data_file_type(orography_variance_file,'.dat','HD')
        try:
            print(subprocess.check_output([path.join(private_bash_scripts_path,
                                                     "parameter_generation_driver.sh"),
                                           path.join(bash_scripts_path,
                                                     "bin"),
                                           path.join(private_bash_scripts_path,
                                                     "fortran"),
                                           path.splitext(rdir_file)[0] + ".dat",
                                           path.splitext(topography_file)[0] + ".dat",
                                           path.splitext(inner_slope_file)[0] + ".dat",
                                           path.splitext(lsmask_file)[0] + ".dat",
                                           path.splitext(null_file)[0] + ".dat",
                                           path.splitext(area_spacing_file)[0] + ".dat",
                                           path.splitext(orography_variance_file)[0] + ".dat",
                                           paragen_source_filepath,paragen_bin_file,output_dir,
                                           "true" if production_run else "false"],
                                          stderr=subprocess.STDOUT))
        except CalledProcessError as cperror:
            raise RuntimeError("Failure in called process {0}; return code {1}; output:\n{2}".format(cperror.cmd,
                                                                                                     cperror.returncode,
                                                                                                     cperror.output))


    def compile_paragen_and_hdfile(self):
        """Compile the paragen and hdfile executables when testing the production run code

        Arguments: None
        Returns: Nothing

        Not used for actual production runs.
        """

        try:
            print(subprocess.check_output([path.join(bash_scripts_path,
                                                     "compile_paragen_and_hdfile.sh"),
                                           path.join(bash_scripts_path,
                                                     "bin"),
                                           path.join(private_bash_scripts_path,
                                                     "fortran"),
                                           path.join(private_bash_scripts_path,"fortran",
                                                     "paragen.f"),"paragen"]))
        except CalledProcessError as cperror:
            raise RuntimeError("Failure in called process {0}; return code {1}; output:\n{2}".format(cperror.cmd,
                                                                                                     cperror.returncode,
                                                                                                     cperror.output))


    def _run_postprocessing(self,rdirs_filename,output_file_label,ls_mask_filename = None,
                            skip_marking_mouths=False,compute_catchments=True,flip_mask_ud=False,
                            grid_type='HD',**grid_kwargs):
        """Run post processing scripts for a given set of river directions

        Arguments:
        rdirs_filename: string; full path to the file containing the river directions to use
        output_file_label: string; label to add to output files
        ls_mask_filename: string; full path to the file containing the land-sea mask to use
        skip_marking_mouths: boolean; if true then don't mark river mouths but still run
            mark river mouth driver to produce river mouth and flow to river mouth files
        compute_catchments: boolean; if true then compute the catchments for this set of river
            directions
        flip_mask_ud: boolean; flip the landsea mask upside down before processing
        grid_type: string; code for the grid type of the grid
        grid_kwargs: dictionary; key word dictionary specifying parameters of the grid (if required)

        Run the flow to grid cell preparation routine, the compute catchment routine (if required) and
        the river mouth marking routine (that also produces a file of rivermouths and a file of flow to
        river mouths in addition to marking them).
        """

        self._run_flow_to_grid_cell(rdirs_filename,output_file_label,grid_type,**grid_kwargs)
        if compute_catchments:
            self._run_compute_catchments(rdirs_filename, output_file_label,
                                         grid_type,**grid_kwargs)
        self._run_river_mouth_marking(rdirs_filename, output_file_label, ls_mask_filename,
                                      flowtocell_filename=self.generated_flowmaps_filepath
                                      + output_file_label + '.nc',
                                      skip_marking_mouths=skip_marking_mouths,
                                      flip_mask_ud=flip_mask_ud,grid_type=grid_type,**grid_kwargs)

    def _run_compute_catchments(self,rdirs_filename,output_file_label,grid_type,**grid_kwargs):
        """Run the catchment computing code placing the results in an appropriate location

        Arguments:
        rdirs_filename: string; full path to the file containing the river direction to use
        output_file_label: string; file label to use on the output file
        grid_type: string; code for the grid type of the grid
        grid_kwargs: dictionary; key word dictionary specifying parameters of the grid (if required)

        Returns: nothing
        """

        compute_catchments.main(filename=rdirs_filename,
                                output_filename=self.generated_catchments_path +
                                output_file_label + '.nc',
                                loop_logfile=self.generated_catchments_path +
                                output_file_label + '_loops.log',
                                grid_type=grid_type,**grid_kwargs)

    def _run_flow_to_grid_cell(self,rdirs_filename,output_file_label,grid_type,**grid_kwargs):
        """Run the cumulative flow generation code placing the results in an appropriate location

        Arguments:
        rdirs_filename: string; full path to the file containing the river direction to use
        output_file_label: string; file label to use on the output file
        grid_type: string; code for the grid type of the grid
        grid_kwargs: dictionary; key word dictionary specifying parameters of the grid (if required)

        Returns: nothing
        """

        flow_to_grid_cell.main(rdirs_filename=rdirs_filename,
                               output_filename=self.generated_flowmaps_filepath
                               + output_file_label + '.nc',
                               grid_type=grid_type,**grid_kwargs)

    def _run_river_mouth_marking(self,rdirs_filename,output_file_label,ls_mask_filename,
                                 flowtocell_filename,skip_marking_mouths,flip_mask_ud=False,
                                 grid_type='HD',**grid_kwargs):
        """Mark river mouths in the river directions and also create two additional river mouth related files

        Arguments:
        rdirs_filename: string; full path to the file containing the river direction to use
        output_file_label: string; file label to use on the output file
        ls_mask_filename: string; full path to the file containing the land-sea mask to use
        flowtocell_filename: string; file name of the cumulative flow file generated from the river
            directions used
        skip_marking_mouths: boolean; if true then don't mark river mouths but still run
            mark river mouth driver to produce river mouth and flow to river mouth files
        flip_mask_ud:boolean; flip the landsea mask upside down before processing
        grid_type: string; code for the grid type of the grid
        grid_kwargs: dictionary; key word dictionary specifying parameters of the grid (if required)

        Return: nothing

        Along with marking the river mouth in the river directions and writing these updated river
        directions to a new file (unless this fucntions is swicthed off) this routine can also
        create a file of just river mouths and a file of the cumulative flow at the river mouths
        if desired.
        """

        river_mouth_marking_driver.main(rdirs_filepath=rdirs_filename,
                                        updatedrdirs_filepath = \
                                            self.generated_rdir_with_outflows_marked_filepath +
                                            output_file_label + '.nc',
                                        lsmask_filepath=ls_mask_filename,
                                        flowtocell_filepath = flowtocell_filename,
                                        rivermouths_filepath = self.generated_rmouth_path +
                                            output_file_label + '.nc',
                                        flowtorivermouths_filepath = \
                                            self.generated_rmouth_cumulative_flow_path +
                                            output_file_label + '.nc',
                                        skip_marking_mouths=skip_marking_mouths,
                                        flip_mask_ud=flip_mask_ud,
                                        grid_type=grid_type,**grid_kwargs)

    def _convert_data_file_type(self,filename,new_file_type,grid_type,**grid_kwargs):
        """Convert the type of a given input file and write to an output file with the same basename

        Arguments:
        filename: string; full path to the input file
        new_file_type: string; extension/type for new file
        grid_type: string; code for the grid type of the grid
        grid_kwargs: dictionary; key word dictionary specifying parameters of the grid (if required)
        Returns:nothing

        The filename of the new file is the basename of the input file with the extension of the new
        filetype
        """

        if new_file_type==(path.splitext(filename)[1]):
            raise UserWarning('File {0} is already of type {1}'.format(filename,new_file_type))
            return
        field_to_convert = iodriver.load_field(filename,
                                               file_type=iodriver.get_file_extension(filename),
                                               field_type='Generic',
                                               grid_type=grid_type,**grid_kwargs)
        iodriver.write_field(filename=path.splitext(filename)[0] + new_file_type,
                             field=field_to_convert,
                             file_type=new_file_type)

    def _correct_orography(self,input_orography_filename,input_corrections_list_filename,
                           output_orography_filename,output_file_label,grid_type,**grid_kwargs):
        """Apply a set of absolute corrections to an input orography and write it to an output file

        Arguments:
        input_orography_filename: string, full path to the orography to apply the corrections to
        input_corrections_list_filename: string, full path to the file with the list of corrections
            to apply, see inside function for format of header and comment lines
        output_orography_filename: string, full path of target file to write the corrected orography
            to
        output_file_label: string; label to use for copy of the correction list file that is made
        grid_type: string; the code for the type of the grid used
        grid_kwargs: dictionary; key word arguments specifying parameters of
            the grid type used
        Returns: nothing

        Makes a copy of the correction list file as a record of which corrections where applied
        (as original version will likely often change after run).
        """

        shutil.copy2(input_corrections_list_filename,self.copied_orography_corrections_filepath +
                     output_file_label + '.txt')
        utilities.apply_orography_corrections(input_orography_filename,
                                              input_corrections_list_filename,
                                              output_orography_filename,
                                              grid_type,**grid_kwargs)

    def _apply_intelligent_burning(self,input_orography_filename,input_superfine_orography_filename,
                                   input_superfine_flowmap_filename,
                                   input_intelligent_burning_regions_list,output_orography_filename,
                                   output_file_label,grid_type,super_fine_grid_type,
                                   super_fine_grid_kwargs={},**grid_kwargs):
        """Apply intelligent burning to selected regions.

        Arguments:
        input_fine_orography_filename: string; full path to the fine orography field file to use as a reference
        input_coarse_orography_filename: string; full path to the coarse orography field file to intelligently burn
        input_fine_fmap_filename: string; full path to the fine cumulative flow field file to use as reference
        output_coarse_orography_filename: string; full path to target file to write the output intelligently burned
            orogrpahy to
        regions_to_burn_list_filename: string; full path of list of regions to burn and the burning thershold to use
            for each region. See inside function the necessary format for the header and the necessary format to
            specifying each region to burn
        change_print_out_limit: integer; limit on the number of changes to the orography to individually print out
        output_file_label: string; label to use for copy of the regions to burn list file that is made
        fine_grid_type: string; code for the grid type of the fine grid
        coarse_grid_type: string; code for teh grid type of the coarse grid
        fine_grid_kwargs: dictionary; key word dictionary specifying parameters of the fine grid (if required)
        coarse_grid_kwargs: dictionary; key word dictionary specifying parameters of the coarse grid (if required)
        Returns: nothing

        Makes a copy of the intelligent burning region list file as a record of which intelligent burnings where
        applied (as original version will likely often change after run).
        """

        shutil.copy2(input_intelligent_burning_regions_list,self.copied_intelligent_burning_regions_path
                     + output_file_label + '.txt')
        utilities.intelligent_orography_burning_driver(input_fine_orography_filename=\
                                                       input_superfine_orography_filename,
                                                       input_coarse_orography_filename=\
                                                       input_orography_filename,
                                                       input_fine_fmap_filename=\
                                                       input_superfine_flowmap_filename,
                                                       output_coarse_orography_filename=\
                                                       output_orography_filename,
                                                       regions_to_burn_list_filename=
                                                       input_intelligent_burning_regions_list,
                                                       fine_grid_type=super_fine_grid_type,
                                                       coarse_grid_type=grid_type,
                                                       fine_grid_kwargs=super_fine_grid_kwargs,
                                                       **grid_kwargs)

    def _run_orography_upscaling(self,input_fine_orography_file,output_coarse_orography_file,
                                 output_file_label,landsea_file=None,true_sinks_file=None,
                                 upscaling_parameters_filename=None,
                                 fine_grid_type='LatLong10min',coarse_grid_type='HD',
                                 input_orography_field_name=None,flip_landsea=False,
                                 rotate_landsea=False,flip_true_sinks=False,rotate_true_sinks=False,
                                 fine_grid_kwargs={},**coarse_grid_kwargs):
        """Drive the C++ sink filling code base to make a tarasov-like orography upscaling

        Arguments:
        input_fine_orography_file: string; full path to input fine orography file
        output_coarse_orography_file: string; full path of target output coarse orography file
        output_file_label: string; label to use for copy of the parameters file that is made
        landsea_file: string; full path to input fine landsea mask file (optional)
        true_sinks_file: string; full path to input fine true sinks file (optional)
        upscaling_parameters_filename: string; full path to the orography upscaling parameter
            file (optional)
        fine_grid_type: string; code for the fine grid type to be upscaled from  (optional)
        coarse_grid_type: string; code for the coarse grid type to be upscaled to (optional)
        input_orography_field_name: string; name of field in the input orography file (optional)
        flip_landsea: bool; flip the input landsea mask upside down
        rotate_landsea: bool; rotate the input landsea mask by 180 degrees along the horizontal axis
        flip_true_sinks: bool; flip the input true sinks field upside down
        rotate_true_sinks: bool; rotate the input true sinks field by 180 degrees along the
            horizontal axis
        fine_grid_kwargs:  keyword dictionary; the parameter of the fine grid to upscale
            from (if required)
        **coarse_grid_kwargs: keyword dictionary; the parameters of the coarse grid to upscale
            to (if required)
        Returns: Nothing.
        """

        shutil.copy2(upscaling_parameters_filename,self.copied_orography_upscaling_parameters_path
                     + output_file_label + '.cfg')
        upscale_orography_driver.drive_orography_upscaling(input_fine_orography_file,output_coarse_orography_file,
                                                           landsea_file,true_sinks_file,
                                                           upscaling_parameters_filename,
                                                           fine_grid_type,coarse_grid_type,
                                                           input_orography_field_name,flip_landsea,
                                                           rotate_landsea,flip_true_sinks,rotate_true_sinks,
                                                           fine_grid_kwargs,**coarse_grid_kwargs)

    def _run_cotat_plus_upscaling(self,input_fine_rdirs_filename,input_fine_cumulative_flow_filename,
                                  cotat_plus_parameters_filename,output_coarse_rdirs_filename,
                                  output_file_label,fine_grid_type,fine_grid_kwargs={},
                                  coarse_grid_type='HD',**coarse_grid_kwargs):
        """Run the cotat plus upscaling routine

        Arguments:
        input_fine_rdirs_filepath: string; path to the file with fine river directions to upscale
        input_fine_total_cumulative_flow_path: string; path to the file with the fine scale cumulative
            flow from the fine river directions
        output_coarse_rdirs_filepath: string; path to the file to write the upscaled coarse river directions to
        cotat_plus_parameters_filepath: string; the file path containing the namelist with the parameters
            for the cotat plus upscaling algorithm
        output_file_label: string; label to use for copy of the parameters file that is made
        fine_grid_type: string; code for the fine grid type to upscale from
        **fine_grid_kwargs(optional): keyword dictionary; the parameter of the fine grid to
            upscale from
        coarse_grid_type: string; code for the coarse grid type to be upscaled to
        **coarse_grid_kwargs(optional): keyword dictionary; the parameter of the coarse grid to
            upscale to (if required)
        Returns: Nothing
        """

        shutil.copy2(cotat_plus_parameters_filename,self.copied_cotat_plus_parameters_path
                     + output_file_label + '.nl')
        cotat_plus_driver.cotat_plus_driver(input_fine_rdirs_filepath=input_fine_rdirs_filename,
                                            input_fine_total_cumulative_flow_path=\
                                            input_fine_cumulative_flow_filename,
                                            output_coarse_rdirs_filepath=output_coarse_rdirs_filename,
                                            cotat_plus_parameters_filepath=\
                                            cotat_plus_parameters_filename,
                                            fine_grid_type=fine_grid_type,
                                            fine_grid_kwargs={},
                                            coarse_grid_type=coarse_grid_type,
                                            **coarse_grid_kwargs)

    def _run_advanced_cotat_plus_upscaling(self,input_fine_rdirs_filename,
                                           input_fine_cumulative_flow_filename,
                                           output_coarse_rdirs_filename,
                                           input_fine_rdirs_fieldname,
                                           input_fine_cumulative_flow_fieldname,
                                           output_coarse_rdirs_fieldname,
                                           cotat_plus_parameters_filename,
                                           output_file_label,
                                           scaling_factor):
      shutil.copy2(cotat_plus_parameters_filename,self.copied_cotat_plus_parameters_path
                   + output_file_label + '.nl')
      cotat_plus_driver.advanced_cotat_plus_driver(input_fine_rdirs_filename,
                                                   input_fine_cumulative_flow_filename,
                                                   output_coarse_rdirs_filename,
                                                   input_fine_rdirs_fieldname,
                                                   input_fine_cumulative_flow_fieldname,
                                                   output_coarse_rdirs_fieldname,
                                                   cotat_plus_parameters_filename,scaling_factor)

    def _apply_transforms_to_field(self,input_filename,output_filename,flip_ud=False,
                                   rotate180lr=False,invert_data=False,
                                   timeslice=None,griddescfile=None,
                                   grid_type='HD',**grid_kwargs):
        """Apply various transformation to a field and optionally add grid information

        Arguments:
        input_filename: string; full path to the input file
        output_filename: string; full path to the target output file to write the
            transformed field to
        flip_ud: boolean; flip the field upside down
        rotate180lr: boolean; rotate the field 180 around the pole, ie move between the
            greenwich meridan and the international dateline as the fields edge
        invert_data: boolean; swap the polarity of boolean data, switch 1's to zeros and
            visa versa
        timeslice: the time slice to select out of the input file (default is None)
        griddescfile: string; full path the file with a description to the grid to
            use to add grid information to this file
        grid_type: string; the code for the type of the grid used
        grid_kwargs: dictionary; key word arguments specifying parameters of
            the grid type used

        Returns: nothing
        """

        if (not flip_ud) and (not rotate180lr) and (not invert_data):
            print("Note: no transform specified, just adding grid parameters and then resaving file")
        field = iodriver.load_field(input_filename,
                                    file_type=iodriver.get_file_extension(input_filename),
                                    field_type='Generic',unmask=False,timeslice=timeslice,
                                    grid_type=grid_type,**grid_kwargs)
        if flip_ud:
            field.flip_data_ud()
        if rotate180lr:
            field.rotate_field_by_a_hundred_and_eighty_degrees()
        if invert_data:
            field.invert_data()
        iodriver.write_field(output_filename,
                             field=field,
                             file_type=iodriver.get_file_extension(output_filename),
                             griddescfile=griddescfile)

    def _add_timeslice_to_combined_dataset(self,first_timeslice,slicetime,
                                           timeslice_hdfile_label,combined_dataset_filename):
        """Add a timeslice to a netcdf 4 dataset combining/that will combine multiple timeslices

        Arguments:
        first_timeslice: boolean; is this the first timeslice? (yes=true)
        slicetime: string; time of the timeslice being added
        timeslice_hdfile_label: string; full path of file containing timeslice to be added
        combined_dataset_filename: string; full path to the (target) file containing/that will contain
            the mutliple slice dataset
        Returns: nothing
        """

        if first_timeslice:
            with netCDF4.Dataset(timeslice_hdfile_label,mode='r',format='NETCDF4') as dataset_in:
                with netCDF4.Dataset(combined_dataset_filename,mode='w',format='NETCDF4') as dataset_out:
                    iohelper.NetCDF4FileIOHelper.\
                        copy_and_append_time_dimension_to_netcdf_dataset(dataset_in,
                                                                         dataset_out)
        else:
            with netCDF4.Dataset(timeslice_hdfile_label,mode='r',format='NETCDF4') as dataset_to_append:
                with netCDF4.Dataset(combined_dataset_filename,mode='a',format='NETCDF4') as main_dataset:
                    iohelper.NetCDF4FileIOHelper.\
                        append_earlier_timeslice_to_dataset(main_dataset, dataset_to_append, slicetime)

class Utilities_Drivers(Dynamic_HD_Drivers):
    """Drive miscellaneous utility processes"""

    def upscale_srtm30_plus_orog_to_10min(self):
        """Upscale a srtm30plus orography to a 10 minute orography"""
        file_label = self._generate_file_label()
        orography_upscaling_parameters_file = path.join(self.orography_upscaling_parameters_path,
                                                        "default_orography_upscaling_"
                                                        "params_for_fac_20.cfg")
        input_srtm30_orography = path.join(self.orography_path,"srtm30plus_v6.nc")
        input_30sec_landsea_mask = path.join(self.ls_masks_path,"glcc_olson_land_cover_data",
                                             "glcc_olson-2.0_lsmask_with_bacseas.nc")
        output_coarse_orography_file = self.tarasov_upscaled_orography_filepath + file_label + '.nc'
        self._run_orography_upscaling(input_srtm30_orography,
                                      output_coarse_orography_file,
                                      output_file_label=file_label,
                                      landsea_file=input_30sec_landsea_mask,
                                      true_sinks_file=None,
                                      upscaling_parameters_filename=\
                                      orography_upscaling_parameters_file,
                                      fine_grid_type="LatLong30sec",
                                      coarse_grid_type="LatLong10min")

    def upscale_srtm30_plus_orog_to_10min_no_lsmask(self):
        """Upscale a srtm30plus orography to a 10 minute orography without using land sea mask"""
        file_label = self._generate_file_label()
        orography_upscaling_parameters_file = path.join(self.orography_upscaling_parameters_path,
                                                        "default_orography_upscaling_"
                                                        "params_for_fac_20.cfg")
        input_srtm30_orography = path.join(self.orography_path,"srtm30plus_v6.nc")
        output_coarse_orography_file = self.tarasov_upscaled_orography_filepath + file_label + '.nc'
        self._run_orography_upscaling(input_srtm30_orography,
                                      output_coarse_orography_file,
                                      output_file_label=file_label,
                                      landsea_file=None,
                                      true_sinks_file=None,
                                      upscaling_parameters_filename=\
                                      orography_upscaling_parameters_file,
                                      fine_grid_type="LatLong30sec",
                                      coarse_grid_type="LatLong10min")

    def upscale_srtm30_plus_orog_to_10min_no_lsmask_tarasov_style_params(self):
        """Upscale a srtm30plus orography to a 10 minute orography without using land sea mask"""
        file_label = self._generate_file_label()
        orography_upscaling_parameters_file = path.join(self.orography_upscaling_parameters_path,
                                                        "tarasov_style_params_orography_upscaling_"
                                                        "params_for_fac_20.cfg")
        input_srtm30_orography = path.join(self.orography_path,"srtm30plus_v6.nc")
        output_coarse_orography_file = self.tarasov_upscaled_orography_filepath + file_label + '.nc'
        self._run_orography_upscaling(input_srtm30_orography,
                                      output_coarse_orography_file,
                                      output_file_label=file_label,
                                      landsea_file=None,
                                      true_sinks_file=None,
                                      upscaling_parameters_filename=\
                                      orography_upscaling_parameters_file,
                                      fine_grid_type="LatLong30sec",
                                      coarse_grid_type="LatLong10min")


    def upscale_srtm30_plus_orog_to_10min_no_lsmask_half_cell_upscaling_params(self):
        """Upscale a srtm30plus orography to a 10 minute orography without using land sea mask"""
        file_label = self._generate_file_label()
        orography_upscaling_parameters_file = path.join(self.orography_upscaling_parameters_path,
                                                        "half_cell_min_upscaling_params"
                                                        "_for_fac_20.cfg")
        input_srtm30_orography = path.join(self.orography_path,"srtm30plus_v6.nc")
        output_coarse_orography_file = self.tarasov_upscaled_orography_filepath + file_label + '.nc'
        self._run_orography_upscaling(input_srtm30_orography,
                                      output_coarse_orography_file,
                                      output_file_label=file_label,
                                      landsea_file=None,
                                      true_sinks_file=None,
                                      upscaling_parameters_filename=\
                                      orography_upscaling_parameters_file,
                                      fine_grid_type="LatLong30sec",
                                      coarse_grid_type="LatLong10min")


    def upscale_srtm30_plus_orog_to_10min_no_lsmask_reduced_back_looping(self):
        """Upscale a srtm30plus orography to a 10 minute orography without using land sea mask"""
        file_label = self._generate_file_label()
        orography_upscaling_parameters_file = path.join(self.orography_upscaling_parameters_path,
                                                        "reduced_back_looping_orography_upscaling"
                                                        "_params_for_fac_20.cfg")
        input_srtm30_orography = path.join(self.orography_path,"srtm30plus_v6.nc")
        output_coarse_orography_file = self.tarasov_upscaled_orography_filepath + file_label + '.nc'
        self._run_orography_upscaling(input_srtm30_orography,
                                      output_coarse_orography_file,
                                      output_file_label=file_label,
                                      landsea_file=None,
                                      true_sinks_file=None,
                                      upscaling_parameters_filename=\
                                      orography_upscaling_parameters_file,
                                      fine_grid_type="LatLong30sec",
                                      coarse_grid_type="LatLong10min")

    def generate_rdirs_from_srtm30_plus(self):
        """Generate river directions on a 30 second grid from the strm30plus orography"""
        file_label = self._generate_file_label()
        input_srtm30_orography = path.join(self.orography_path,"srtm30plus_v6.nc")
        lsmask = path.join(self.ls_masks_path,"glcc_olson_land_cover_data",
                           "glcc_olson-2.0_lsmask_with_bacseas.nc")
        output_rdirs_file = self.generated_orography_filepath + file_label + '.nc'
        output_catch_file = self.generated_catchments_path + file_label + '.nc'
        fill_sinks_driver.advanced_sinkless_flow_directions_generator(filename=
                                                                      input_srtm30_orography,
                                                                      output_filename=
                                                                      output_rdirs_file,
                                                                      fieldname="topo",
                                                                      output_fieldname="rdirs",
                                                                      ls_mask_filename=
                                                                      lsmask,
                                                                      ls_mask_fieldname=
                                                                      "field_value",
                                                                      catchment_nums_filename=
                                                                      output_catch_file,
                                                                      catchment_fieldname="catch")

    def renumber_catchments_from_strm30_plus(self):
      catchment_file_label = "generate_rdirs_from_srtm30_plus_20180802_202027"
      catchment_filename = \
          "/Users/thomasriddick/Documents/data/temp/30sec_catch_test{}.nc".format(catchment_file_label)
      reordered_catchment_file_label = output_rdirs_file = self.generated_orography_filepath + file_label + '.nc'
      unordered_catchments = iodriver.advanced_field_loader(catchment_filename,
                                                            fieldname='catch')
      ordered_catchments = compute_catchments.renumber_catchments_by_size(unordered_catchments.get_data())
      iodriver.advanced_field_writer(reordered_catchment_file_label,
                                     field.Field(ordered_catchments,grid=unordered_catchments.get_grid()),
                                     fieldname='catch')

    def create_lgm_orography_from_strm30_plus_and_ice_6g(self):
        file_label = self._generate_file_label()
        input_srtm30_orography_filename = path.join(self.orography_path,"srtm30plus_v6.nc")
        ice6g_0k_filename = path.join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        ice6g_21k_filename = path.join(self.orography_path,"Ice6g_c_VM5a_10min_21k.nc")
        output_lgm_orography_filename = self.generated_orography_filepath + file_label + '.nc'
        utilities.\
        create_30s_lgm_orog_from_hr_present_day_and_lr_pair_driver(input_lgm_low_res_orog_filename=
                                                                   ice6g_21k_filename,
                                                                   input_present_day_low_res_orog_filename=
                                                                   ice6g_0k_filename,
                                                                   input_present_day_high_res_orog_filename=
                                                                   input_srtm30_orography_filename,
                                                                   output_lgm_high_res_orog_filename=
                                                                   output_lgm_orography_filename,
                                                                   input_lgm_low_res_orog_fieldname="Topo",
                                                                   input_present_day_low_res_orog_fieldname=
                                                                   "Topo",
                                                                   input_present_day_high_res_orog_fieldname=
                                                                   "topo",
                                                                   output_lgm_high_res_orog_fieldname="topo")

    def generate_rdirs_from_srtm30_plus_iceg6_30sec_lgm(self):
        """Generate river directions on a 30 second grid from the strm30plus orography"""
        file_label = self._generate_file_label()
        input_srtm30_orography = path.join(self.orography_path,"generated",
            "updated_orog_create_lgm_orography_from_strm30_plus_and_ice_6g_20180803_080552.nc")
        ls_mask = path.join(self.ls_masks_path,"generated",
                            "ls_mask_generate_rdirs_from_srtm30_plus_iceg6_30sec_lgm_20180803_091544_with_grid.nc")
        output_rdirs_file = self.generated_rdir_filepath + file_label + '.nc'
        output_catch_file = self.generated_catchments_path + file_label + '.nc'
        fill_sinks_driver.advanced_sinkless_flow_directions_generator(filename=
                                                                      input_srtm30_orography,
                                                                      output_filename=
                                                                      output_rdirs_file,
                                                                      fieldname="topo",
                                                                      output_fieldname="rdirs",
                                                                      ls_mask_filename=
                                                                      ls_mask,
                                                                      ls_mask_fieldname=
                                                                      "slm",
                                                                      catchment_nums_filename=
                                                                      output_catch_file,
                                                                      catchment_fieldname="catch")

    def renumber_catchments_from_strm30_plus_ice6g_30sec_lgm(self):
      file_label = "generate_rdirs_from_srtm30_plus_iceg6_30sec_lgm_20180803_100943"
      catchment_filename = path.join(self.catchments_path,
                                     "catchmentmap_generate_rdirs_from_srtm30"
                                     "_plus_iceg6_30sec_lgm_20180803_100943.nc")
      reordered_catchment_file_label = output_rdirs_file = self.generated_catchments_path + file_label + '.nc'
      unordered_catchments = iodriver.advanced_field_loader(catchment_filename,
                                                            fieldname='catch')
      ordered_catchments = compute_catchments.renumber_catchments_by_size(unordered_catchments.get_data())
      iodriver.advanced_field_writer(reordered_catchment_file_label,
                                     field.Field(ordered_catchments,grid=unordered_catchments.get_grid()),
                                     fieldname='catch')

class UpscaledMERIThydroDrivers(Dynamic_HD_Drivers):

    def generate_corrections_for_upscaled_MeritHydro_0k(self):
        """Using the MERIT hydro 3 second data upscaled to 10 minutes
        """

        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        upscaled_MERIT_orography_filename = path.join(self.orography_path,"tarasov_upscaled",
                                                      "MERITdem_hydroupscaled_to_10min_global_g.nc")
        orography_filename = self.corrected_orography_filepath + file_label + '.nc'
        orography_corrections_field_filename = self.generated_orography_corrections_fields_path +\
                                                file_label + '.nc'
        utilities.advanced_orog_correction_field_generator(original_orography_filename,
                                                           upscaled_MERIT_orography_filename,
                                                           orography_corrections_field_filename,
                                                           original_orography_fieldname="orog",
                                                           corrected_orography_fieldname="z",
                                                           orography_corrections_fieldname="orog_corrections")

def main():
    """Select the revelant runs to make

    Select runs by uncommenting them and also the revelant object instantation.
    """

    #utilities_drivers = Utilities_Drivers()
    #utilities_drivers.upscale_srtm30_plus_orog_to_10min()
    #utilities_drivers.upscale_srtm30_plus_orog_to_10min_no_lsmask()
    #utilities_drivers.upscale_srtm30_plus_orog_to_10min_no_lsmask_tarasov_style_params()
    #utilities_drivers.upscale_srtm30_plus_orog_to_10min_no_lsmask_reduced_back_looping()
    #upscaled_MERIThydro_drivers = UpscaledMERIThydroDrivers()
    #upscaled_MERIThydro_drivers.generate_corrections_for_upscaled_MeritHydro_0k()
    pass

if __name__ == '__main__':
    main()
