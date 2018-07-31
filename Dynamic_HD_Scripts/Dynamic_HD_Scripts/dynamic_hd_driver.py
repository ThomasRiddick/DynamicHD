'''
Driving routines for a wide range of specific dynamic HD file generation runs
Created on Feb 24, 2016

@author: thomasriddick
'''

import inspect
import datetime
import subprocess
import dynamic_hd
import os.path as path
from matplotlib.compat.subprocess import CalledProcessError
import flow_to_grid_cell
import compute_catchments
import fill_sinks_driver
import upscale_orography_driver
import utilities
import grid
import river_mouth_marking_driver
import create_connected_lsmask_driver as cc_lsmask_driver
from context import bash_scripts_path
from context import private_bash_scripts_path
import shutil
import cotat_plus_driver
import loop_breaker_driver
import numpy as np
import iohelper
import netCDF4

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
        cotat_plus_parameters_path_extension = path.join(parameter_path_extension,'cotat_plus')
        orography_upscaling_parameters_path_extension = path.join(parameter_path_extension,
                                                                  'orography_upscaling')
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

    def _prepare_topography(self,orog_nc_file,grid_file,weights_file,output_file,lsmask_file):
        """Run the prepare topography script

        Arguments:
        orog_nc_file,grid_file,weights_file,output_file,lsmask_file: string, full path to various input
        files. The exact format required in these files is unknown; check inside prepare_topography.sh
        for more details
        Returns: nothing

        Run an old script for preparing a topography; not currently used, this is kept only for archival
        purposes.
        """

        try:
            print subprocess.check_output([path.join(bash_scripts_path,
                                                     "prepare_topography.sh"),
                                           orog_nc_file,
                                           grid_file,
                                           weights_file,
                                           output_file,
                                           lsmask_file],stderr=subprocess.STDOUT)
        except CalledProcessError as cperror:
            raise RuntimeError("Failure in called process {0}; return code {1}; output:\n{2}".format(cperror.cmd,
                                                                                                     cperror.returncode,
                                                                                                     cperror.output))

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
            print subprocess.check_output([path.join(private_bash_scripts_path,
                                                     "generate_output_file.sh"),
                                           path.join(bash_scripts_path,
                                                     "bin"),
                                           path.join(bash_scripts_path,
                                                     "fortran"),
                                           path.splitext(rdir_file)[0] + ".dat",
                                           path.splitext(lsmask_file)[0] + ".dat",
                                           path.splitext(null_file)[0] + ".dat",
                                           path.splitext(area_spacing_file)[0] + ".dat",
                                           hd_grid_specs_file,output_file,paras_dir,
                                           "true" if production_run else "false"])
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
        original_paragen_source_filepath = path.join(bash_scripts_path,"fortran",
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
            print subprocess.check_output([path.join(private_bash_scripts_path,
                                                     "parameter_generation_driver.sh"),
                                           path.join(bash_scripts_path,
                                                     "bin"),
                                           path.join(bash_scripts_path,
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
                                          stderr=subprocess.STDOUT)
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
            print subprocess.check_output([path.join(bash_scripts_path,
                                                     "compile_paragen_and_hdfile.sh"),
                                           path.join(bash_scripts_path,
                                                     "bin"),
                                           path.join(private_bash_scripts_path,
                                                     "fortran"),
                                           path.join(private_bash_scripts_path,"fortran",
                                                     "paragen.f"),"paragen"])
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
        field_to_convert = dynamic_hd.load_field(filename,
                                                 file_type=dynamic_hd.get_file_extension(filename),
                                                 field_type='Generic',
                                                 grid_type=grid_type,**grid_kwargs)
        dynamic_hd.write_field(filename=path.splitext(filename)[0] + new_file_type,
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
        input_course_orography_filename: string; full path to the course orography field file to intelligently burn
        input_fine_fmap_filename: string; full path to the fine cumulative flow field file to use as reference
        output_course_orography_filename: string; full path to target file to write the output intelligently burned
            orogrpahy to
        regions_to_burn_list_filename: string; full path of list of regions to burn and the burning thershold to use
            for each region. See inside function the necessary format for the header and the necessary format to
            specifying each region to burn
        change_print_out_limit: integer; limit on the number of changes to the orography to individually print out
        output_file_label: string; label to use for copy of the regions to burn list file that is made
        fine_grid_type: string; code for the grid type of the fine grid
        course_grid_type: string; code for teh grid type of the course grid
        fine_grid_kwargs: dictionary; key word dictionary specifying parameters of the fine grid (if required)
        course_grid_kwargs: dictionary; key word dictionary specifying parameters of the course grid (if required)
        Returns: nothing

        Makes a copy of the intelligent burning region list file as a record of which intelligent burnings where
        applied (as original version will likely often change after run).
        """

        shutil.copy2(input_intelligent_burning_regions_list,self.copied_intelligent_burning_regions_path
                     + output_file_label + '.txt')
        utilities.intelligent_orography_burning_driver(input_fine_orography_filename=\
                                                       input_superfine_orography_filename,
                                                       input_course_orography_filename=\
                                                       input_orography_filename,
                                                       input_fine_fmap_filename=\
                                                       input_superfine_flowmap_filename,
                                                       output_course_orography_filename=\
                                                       output_orography_filename,
                                                       regions_to_burn_list_filename=
                                                       input_intelligent_burning_regions_list,
                                                       fine_grid_type=super_fine_grid_type,
                                                       course_grid_type=grid_type,
                                                       fine_grid_kwargs=super_fine_grid_kwargs,
                                                       **grid_kwargs)

    def _run_orography_upscaling(self,input_fine_orography_file,output_course_orography_file,
                                 output_file_label,landsea_file=None,true_sinks_file=None,
                                 upscaling_parameters_filename=None,
                                 fine_grid_type='LatLong10min',course_grid_type='HD',
                                 input_orography_field_name=None,flip_landsea=False,
                                 rotate_landsea=False,flip_true_sinks=False,rotate_true_sinks=False,
                                 fine_grid_kwargs={},**course_grid_kwargs):
        """Drive the C++ sink filling code base to make a tarasov-like orography upscaling

        Arguments:
        input_fine_orography_file: string; full path to input fine orography file
        output_course_orography_file: string; full path of target output course orography file
        output_file_label: string; label to use for copy of the parameters file that is made
        landsea_file: string; full path to input fine landsea mask file (optional)
        true_sinks_file: string; full path to input fine true sinks file (optional)
        upscaling_parameters_filename: string; full path to the orography upscaling parameter
            file (optional)
        fine_grid_type: string; code for the fine grid type to be upscaled from  (optional)
        course_grid_type: string; code for the course grid type to be upscaled to (optional)
        input_orography_field_name: string; name of field in the input orography file (optional)
        flip_landsea: bool; flip the input landsea mask upside down
        rotate_landsea: bool; rotate the input landsea mask by 180 degrees along the horizontal axis
        flip_true_sinks: bool; flip the input true sinks field upside down
        rotate_true_sinks: bool; rotate the input true sinks field by 180 degrees along the
            horizontal axis
        fine_grid_kwargs:  keyword dictionary; the parameter of the fine grid to upscale
            from (if required)
        **course_grid_kwargs: keyword dictionary; the parameters of the course grid to upscale
            to (if required)
        Returns: Nothing.
        """

        shutil.copy2(upscaling_parameters_filename,self.copied_orography_upscaling_parameters_path
                     + output_file_label + '.cfg')
        upscale_orography_driver.drive_orography_upscaling(input_fine_orography_file,output_course_orography_file,
                                                           landsea_file,true_sinks_file,
                                                           upscaling_parameters_filename,
                                                           fine_grid_type,course_grid_type,
                                                           input_orography_field_name,flip_landsea,
                                                           rotate_landsea,flip_true_sinks,rotate_true_sinks,
                                                           fine_grid_kwargs,**course_grid_kwargs)

    def _run_cotat_plus_upscaling(self,input_fine_rdirs_filename,input_fine_cumulative_flow_filename,
                                  cotat_plus_parameters_filename,output_course_rdirs_filename,
                                  output_file_label,fine_grid_type,fine_grid_kwargs={},
                                  course_grid_type='HD',**course_grid_kwargs):
        """Run the cotat plus upscaling routine

        Arguments:
        input_fine_rdirs_filepath: string; path to the file with fine river directions to upscale
        input_fine_total_cumulative_flow_path: string; path to the file with the fine scale cumulative
            flow from the fine river directions
        output_course_rdirs_filepath: string; path to the file to write the upscaled course river directions to
        cotat_plus_parameters_filepath: string; the file path containing the namelist with the parameters
            for the cotat plus upscaling algorithm
        output_file_label: string; label to use for copy of the parameters file that is made
        fine_grid_type: string; code for the fine grid type to upscale from
        **fine_grid_kwargs(optional): keyword dictionary; the parameter of the fine grid to
            upscale from
        course_grid_type: string; code for the course grid type to be upscaled to
        **course_grid_kwargs(optional): keyword dictionary; the parameter of the course grid to
            upscale to (if required)
        Returns: Nothing
        """

        shutil.copy2(cotat_plus_parameters_filename,self.copied_cotat_plus_parameters_path
                     + output_file_label + '.nl')
        cotat_plus_driver.cotat_plus_driver(input_fine_rdirs_filepath=input_fine_rdirs_filename,
                                            input_fine_total_cumulative_flow_path=\
                                            input_fine_cumulative_flow_filename,
                                            output_course_rdirs_filepath=output_course_rdirs_filename,
                                            cotat_plus_parameters_filepath=\
                                            cotat_plus_parameters_filename,
                                            fine_grid_type=fine_grid_type,
                                            fine_grid_kwargs={},
                                            course_grid_type=course_grid_type,
                                            **course_grid_kwargs)

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
            print "Note: no transform specified, just adding grid parameters and then resaving file"
        field = dynamic_hd.load_field(input_filename,
                                      file_type=dynamic_hd.get_file_extension(input_filename),
                                      field_type='Generic',unmask=False,timeslice=timeslice,
                                      grid_type=grid_type,**grid_kwargs)
        if flip_ud:
            field.flip_data_ud()
        if rotate180lr:
            field.rotate_field_by_a_hundred_and_eighty_degrees()
        if invert_data:
            field.invert_data()
        dynamic_hd.write_field(output_filename,
                               field=field,
                               file_type=dynamic_hd.get_file_extension(output_filename),
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

    def create_catchments_from_hdpara_file_from_swati(self):
        """Create catchments from the hdpara file that Swati gave me"""
        file_label = self._generate_file_label()
        hdpara_filepath = path.join(self.rdir_path,"rdirs_hdpara_from_swati.nc")
        self._run_compute_catchments(rdirs_filename=hdpara_filepath,output_file_label=file_label,
                                     grid_type='HD')

    def convert_corrected_HD_hydrology_dat_files_to_nc(self):
        """Convert original river directiosn from dat to nc"""
        corrected_RFD_filepath = path.join(self.rdir_path,'rivdir_vs_1_9_data_from_stefan.dat')
        corrected_orography_filepath = path.join(self.orography_path,'topo_hd_vs1_9_data_from_stefan.dat')
        for filename in [corrected_RFD_filepath,corrected_orography_filepath]:
            self._convert_data_file_type(filename, new_file_type='.nc', grid_type='HD')

    def recreate_connected_HD_lsmask(self):
        """Regenerate a connected version of the landsea mask extracted from the original river directions"""
        file_label = self._generate_file_label()
        hd_lsmask_seed_points = path.join(self.ls_seed_points_path,'lsseedpoints_HD_160530_0001900.txt')
        cc_lsmask_driver.drive_connected_lsmask_creation(input_lsmask_filename=\
                                                            self.generated_ls_mask_filepath +\
                                                            "extract_ls_mask_from_corrected_"
                                                            "HD_rdirs_20160504_142435.nc",
                                                         output_lsmask_filename=\
                                                            self.generated_ls_mask_filepath +
                                                            file_label + '.nc',
                                                         input_ls_seed_points_filename=None,
                                                         input_ls_seed_points_list_filename=\
                                                            hd_lsmask_seed_points,
                                                         use_diagonals_in=True, grid_type='HD')

    def recreate_connected_HD_lsmask_from_glcc_olson_data(self):
        """Regenerate a connected version of the landsea mask extracted from upscaled glcc olson data"""
        file_label = self._generate_file_label()
        hd_lsmask_seed_points = path.join(self.ls_seed_points_path,'lsseedpoints_HD_true_seas_inc'
                                                                   '_casp_only_160718_105600.txt')
        cc_lsmask_driver.drive_connected_lsmask_creation(input_lsmask_filename=\
                                                         path.join(self.ls_masks_path,
                                                         "glcc_olson_land_cover_data",
                                                         "glcc_olson-2.0_lsmask_with_bacseas_upscaled_30min.nc"),
                                                         output_lsmask_filename=\
                                                            self.generated_ls_mask_filepath +
                                                            file_label + '.nc',
                                                         rotate_seeds_about_polar_axis=True,
                                                         input_ls_seed_points_filename=None,
                                                         input_ls_seed_points_list_filename=\
                                                            hd_lsmask_seed_points,
                                                         flip_input_mask_ud=True,
                                                         use_diagonals_in=True, grid_type='HD')

    def recreate_connected_10min_lsmask_from_glcc_olson_data(self):
        """Regenerate a connected version of the landsea mask extracted from upscaled glcc olson data"""
        file_label = self._generate_file_label()
        _10min_lsmask_seed_points = path.join(self.ls_seed_points_path,'lsseedpoints_downscale_HD_ls_seed_points_to_'
                                                                   '10min_lat_lon_true_seas_inc_casp_only_20160718_114402.txt')
        cc_lsmask_driver.drive_connected_lsmask_creation(input_lsmask_filename=\
                                                         path.join(self.ls_masks_path,
                                                         "glcc_olson_land_cover_data",
                                                         "glcc_olson-2.0_lsmask_with_bacseas_upscaled_10min.nc"),
                                                         output_lsmask_filename=\
                                                            self.generated_ls_mask_filepath +
                                                            file_label + '.nc',
                                                         rotate_seeds_about_polar_axis=True,
                                                         flip_seeds_ud=True,
                                                         input_ls_seed_points_filename=None,
                                                         input_ls_seed_points_list_filename=\
                                                            _10min_lsmask_seed_points,
                                                         flip_input_mask_ud=True,
                                                         use_diagonals_in=True, grid_type='LatLong10min')

    def recreate_connected_HD_lsmask_true_seas_inc_casp_only(self):
        """Recreate a connected version of the landsea mask of the original river directions with only Caspian included

        So this has only the main oceans plus the Caspian and no other inland seas
        """

        file_label = self._generate_file_label()
        hd_lsmask_seed_points = path.join(self.ls_seed_points_path,"lsseedpoints_HD_true_seas_"
                                                                   "inc_casp_only_160718_105600.txt")
        cc_lsmask_driver.drive_connected_lsmask_creation(input_lsmask_filename=\
                                                            self.generated_ls_mask_filepath +\
                                                            "extract_ls_mask_from_corrected_"
                                                            "HD_rdirs_20160504_142435.nc",
                                                         output_lsmask_filename=\
                                                            self.generated_ls_mask_filepath +
                                                            file_label + '.nc',
                                                         input_ls_seed_points_filename=None,
                                                         input_ls_seed_points_list_filename=\
                                                            hd_lsmask_seed_points,
                                                         use_diagonals_in=True, grid_type='HD')

    def recreate_connected_lsmask_for_black_azov_and_caspian_seas_from_glcc_olson_data(self):
        """Create an lsmask for the black,azov and caspian seas from a lake mask on a 30 second resolution"""
        file_label = self._generate_file_label()
        glcc_olson_lake_mask = path.join(self.ls_masks_path,"glcc_olson_land_cover_data",
                                         "glcc_olson-2.0_lakemask.nc")
        thirty_minute_black_azov_caspian_lsmask_seed_points = path.join(self.ls_seed_points_path,
                                                                        "30sec_black_azov_caspian"
                                                                        "_lsmask_seed_points.txt")
        cc_lsmask_driver.drive_connected_lsmask_creation(input_lsmask_filename=glcc_olson_lake_mask,
                                                         output_lsmask_filename=\
                                                         self.generated_ls_mask_filepath +
                                                         file_label + '.nc',
                                                         input_ls_seed_points_list_filename=\
                                                         thirty_minute_black_azov_caspian_lsmask_seed_points,
                                                         use_diagonals_in=True,
                                                         rotate_seeds_about_polar_axis=False,
                                                         grid_type='LatLong30sec')
        self._apply_transforms_to_field(self.generated_ls_mask_filepath +
                                        file_label + '.nc',
                                        output_filename = self.generated_ls_mask_filepath +
                                            file_label + '_with_grid_info.nc',
                                        flip_ud=False,
                                        rotate180lr=False,invert_data=False,
                                        griddescfile=self.thirty_second_grid_filepath,
                                        grid_type='LatLong30sec')

    def downscale_HD_ls_seed_points_to_1min_lat_lon(self):
        """Downscale the set of sea seed points to a 1 minute latlon resolution"""
        file_label = self._generate_file_label()
        hd_lsmask_seed_points = path.join(self.ls_seed_points_path,'lsseedpoints_HD_160530_0001900.txt')
        utilities.downscale_ls_seed_points_list_driver(hd_lsmask_seed_points,
                                                       self.generated_ls_seed_points_path +
                                                       file_label + '.txt',
                                                       factor=30,
                                                       nlat_fine=10800,
                                                       nlon_fine=21600,
                                                       input_grid_type='HD',
                                                       output_grid_type='LatLong1min')

    def downscale_HD_ls_seed_points_to_10min_lat_lon(self):
        """Downscale the set of sea seed points to a 10 minute latlon resolution"""
        file_label = self._generate_file_label()
        hd_lsmask_seed_points = path.join(self.ls_seed_points_path,'lsseedpoints_HD_160530_0001900.txt')
        utilities.downscale_ls_seed_points_list_driver(hd_lsmask_seed_points,
                                                       self.generated_ls_seed_points_path +
                                                       file_label + '.txt',
                                                       factor=3,
                                                       nlat_fine=1080,
                                                       nlon_fine=2160,
                                                       input_grid_type='HD',
                                                       output_grid_type='LatLong10min')

    def downscale_HD_ls_seed_points_to_10min_lat_lon_true_seas_inc_casp_only(self):
        """Downscale the set of sea seed points to a 10 minute latlon resolution including Caspian only

        So this has only the main oceans plus the Caspian and no other inland seas
        """

        file_label = self._generate_file_label()
        hd_lsmask_seed_points = path.join(self.ls_seed_points_path,
                                          "lsseedpoints_HD_true_seas_inc_casp_only_160718_105600.txt")
        utilities.downscale_ls_seed_points_list_driver(hd_lsmask_seed_points,
                                                       self.generated_ls_seed_points_path +
                                                       file_label + '.txt',
                                                       factor=3,
                                                       nlat_fine=1080,
                                                       nlon_fine=2160,
                                                       input_grid_type='HD',
                                                       output_grid_type='LatLong10min')

    def upscale_srtm30_plus_orog_to_10min(self):
        """Upscale a srtm30plus orography to a 10 minute orography"""
        file_label = self._generate_file_label()
        orography_upscaling_parameters_file = path.join(self.orography_upscaling_parameters_path,
                                                        "default_orography_upscaling_"
                                                        "params_for_fac_20.cfg")
        input_srtm30_orography = path.join(self.orography_path,"srtm30plus_v6.nc")
        input_30sec_landsea_mask = path.join(self.ls_masks_path,"glcc_olson_land_cover_data",
                                             "glcc_olson-2.0_lsmask_with_bacseas.nc")
        output_course_orography_file = self.tarasov_upscaled_orography_filepath + file_label + '.nc'
        self._run_orography_upscaling(input_srtm30_orography,
                                      output_course_orography_file,
                                      output_file_label=file_label,
                                      landsea_file=input_30sec_landsea_mask,
                                      true_sinks_file=None,
                                      upscaling_parameters_filename=\
                                      orography_upscaling_parameters_file,
                                      fine_grid_type="LatLong30sec",
                                      course_grid_type="LatLong10min")

    def upscale_srtm30_plus_orog_to_10min_no_lsmask(self):
        """Upscale a srtm30plus orography to a 10 minute orography without using land sea mask"""
        file_label = self._generate_file_label()
        orography_upscaling_parameters_file = path.join(self.orography_upscaling_parameters_path,
                                                        "default_orography_upscaling_"
                                                        "params_for_fac_20.cfg")
        input_srtm30_orography = path.join(self.orography_path,"srtm30plus_v6.nc")
        output_course_orography_file = self.tarasov_upscaled_orography_filepath + file_label + '.nc'
        self._run_orography_upscaling(input_srtm30_orography,
                                      output_course_orography_file,
                                      output_file_label=file_label,
                                      landsea_file=None,
                                      true_sinks_file=None,
                                      upscaling_parameters_filename=\
                                      orography_upscaling_parameters_file,
                                      fine_grid_type="LatLong30sec",
                                      course_grid_type="LatLong10min")

    def upscale_srtm30_plus_orog_to_10min_no_lsmask_tarasov_style_params(self):
        """Upscale a srtm30plus orography to a 10 minute orography without using land sea mask"""
        file_label = self._generate_file_label()
        orography_upscaling_parameters_file = path.join(self.orography_upscaling_parameters_path,
                                                        "tarasov_style_params_orography_upscaling_"
                                                        "params_for_fac_20.cfg")
        input_srtm30_orography = path.join(self.orography_path,"srtm30plus_v6.nc")
        output_course_orography_file = self.tarasov_upscaled_orography_filepath + file_label + '.nc'
        self._run_orography_upscaling(input_srtm30_orography,
                                      output_course_orography_file,
                                      output_file_label=file_label,
                                      landsea_file=None,
                                      true_sinks_file=None,
                                      upscaling_parameters_filename=\
                                      orography_upscaling_parameters_file,
                                      fine_grid_type="LatLong30sec",
                                      course_grid_type="LatLong10min")


    def upscale_srtm30_plus_orog_to_10min_no_lsmask_half_cell_upscaling_params(self):
        """Upscale a srtm30plus orography to a 10 minute orography without using land sea mask"""
        file_label = self._generate_file_label()
        orography_upscaling_parameters_file = path.join(self.orography_upscaling_parameters_path,
                                                        "half_cell_min_upscaling_params"
                                                        "_for_fac_20.cfg")
        input_srtm30_orography = path.join(self.orography_path,"srtm30plus_v6.nc")
        output_course_orography_file = self.tarasov_upscaled_orography_filepath + file_label + '.nc'
        self._run_orography_upscaling(input_srtm30_orography,
                                      output_course_orography_file,
                                      output_file_label=file_label,
                                      landsea_file=None,
                                      true_sinks_file=None,
                                      upscaling_parameters_filename=\
                                      orography_upscaling_parameters_file,
                                      fine_grid_type="LatLong30sec",
                                      course_grid_type="LatLong10min")


    def upscale_srtm30_plus_orog_to_10min_no_lsmask_reduced_back_looping(self):
        """Upscale a srtm30plus orography to a 10 minute orography without using land sea mask"""
        file_label = self._generate_file_label()
        orography_upscaling_parameters_file = path.join(self.orography_upscaling_parameters_path,
                                                        "reduced_back_looping_orography_upscaling"
                                                        "_params_for_fac_20.cfg")
        input_srtm30_orography = path.join(self.orography_path,"srtm30plus_v6.nc")
        output_course_orography_file = self.tarasov_upscaled_orography_filepath + file_label + '.nc'
        self._run_orography_upscaling(input_srtm30_orography,
                                      output_course_orography_file,
                                      output_file_label=file_label,
                                      landsea_file=None,
                                      true_sinks_file=None,
                                      upscaling_parameters_filename=\
                                      orography_upscaling_parameters_file,
                                      fine_grid_type="LatLong30sec",
                                      course_grid_type="LatLong10min")

    def generate_rdirs_from_srtm30_plus(self):
        """Generate river directions on a 30 second grid from the strm30plus orography"""
        file_label = self._generate_file_label()
        input_srtm30_orography = path.join(self.orography_path,"srtm30plus_v6.nc")
        lsmask = path.join(self.ls_masks_path,"glcc_olson_land_cover_data",
                           "glcc_olson-2.0_lsmask_with_bacseas.nc")
        output_rdirs_file =\
          "/Users/thomasriddick/Documents/data/temp/30sec_rdirs_test{}.nc".format(file_label)
        fill_sinks_driver.advanced_sinkless_flow_directions_generator(filename=
                                                                      input_srtm30_orography,
                                                                      output_filename=
                                                                      output_rdirs_file,
                                                                      fieldname="topo",
                                                                      output_fieldname="rdirs",
                                                                      ls_mask_filename=
                                                                      lsmask,
                                                                      ls_mask_fieldname=
                                                                      "field_value")
    def generate_rdirs_from_ice5g_21k(self):
        """Generate river directions on a 30 second grid from the strm30plus orography"""
        file_label = self._generate_file_label()
        input_ice5g_orography = path.join(self.orography_path,"ice5g_v1_2_21_0k_10min.nc")
        land_sea_mask_file = path.join(self.ls_masks_path,
                                             "10min-mask-lgm-from-virna_with_gridinfo.nc")
        output_rdirs_file =\
          "/Users/thomasriddick/Documents/data/temp/10min_rdirs_test{}.nc".format(file_label)
        fill_sinks_driver.advanced_sinkless_flow_directions_generator(filename=
                                                                      input_ice5g_orography,
                                                                      output_filename=
                                                                      output_rdirs_file,
                                                                      fieldname="orog",
                                                                      output_fieldname="rdirs",
                                                                      ls_mask_filename=
                                                                      land_sea_mask_file,
                                                                      ls_mask_fieldname="field_value")

    def upscale_1min_orography_to_30min(self):
        """Upscale the ETOPO 1min orography to a 30 minute orography"""
        file_label = self._generate_file_label()
        orography_upscaling_parameters_file = path.join(self.orography_upscaling_parameters_path,
                                                        "default_orography_upscaling_"
                                                        "params_for_fac_30.cfg")
        input_orography = path.join(self.orography_path,"ETOPO1_Ice_c_gmt4.nc")
        output_course_orography_file = self.tarasov_upscaled_orography_filepath + file_label + '.nc'
        self._run_orography_upscaling(input_orography,
                                      output_course_orography_file,
                                      output_file_label=file_label,
                                      landsea_file=None,
                                      true_sinks_file=None,
                                      upscaling_parameters_filename=\
                                      orography_upscaling_parameters_file,
                                      fine_grid_type="LatLong1min",
                                      course_grid_type="HD")

    def downscale_ICE6G_21k_landsea_mask_and_remove_disconnected_points(self):
        """Downscale a 1 degree ICE6G landsea mask"""
        file_label = self._generate_file_label()
        ice6g_land_sea_mask_file = path.join(self.orography_path,"ice6g_VM5a_1deg_21_0k.nc")
        present_day_10min_mask_file = path.join(self.ls_masks_path,"glcc_olson_land_cover_data",
                                                "glcc_olson-2.0_lsmask_with_bacseas_upscaled_10min.nc")
        intermediary_land_sea_mask_file = (self.generated_ls_mask_filepath +
                                           "intermediary_" + file_label + '.nc')
        second_intermediary_land_sea_mask_file = (self.generated_ls_mask_filepath +
                                                  "2nd_intermediary_" + file_label + '.nc')
        third_intermediary_land_sea_mask_file = (self.generated_ls_mask_filepath +
                                                 "3rd_intermediary_" + file_label + '.nc')
        landsea_mask = dynamic_hd.load_field(filename=ice6g_land_sea_mask_file,
                                             file_type=".nc",field_type="Generic",
                                             fieldname="sftlf", grid_type="LatLong1deg")
        landsea_mask.convert_to_binary_mask()
        landsea_mask.invert_data()
        dynamic_hd.write_field(filename=intermediary_land_sea_mask_file,
                               field=landsea_mask,file_type=".nc")
        utilities.downscale_ls_mask_driver(input_course_ls_mask_filename=intermediary_land_sea_mask_file,
                                           output_fine_ls_mask_filename=\
                                           second_intermediary_land_sea_mask_file,
                                           input_flipud=False,
                                           input_rotate180lr=False,
                                           course_grid_type='LatLong1deg',
                                           fine_grid_type='LatLong10min')
        cc_lsmask_driver.drive_connected_lsmask_creation(input_lsmask_filename=\
                                                         second_intermediary_land_sea_mask_file,
                                                         output_lsmask_filename=\
                                                         third_intermediary_land_sea_mask_file,
                                                         input_ls_seed_points_filename=None,
                                                         input_ls_seed_points_list_filename=\
                                                         path.join(self.ls_seed_points_path,
                                                                   "lsseedpoints_downscale_HD_ls_seed_points"
                                                                   "_to_10min_lat_lon_true_seas_inc_casp_only"
                                                                   "_20160718_114402.txt"),
                                                         rotate_seeds_about_polar_axis=True,
                                                         use_diagonals_in=True, grid_type='LatLong10min')
        landsea_mask_present = dynamic_hd.load_field(filename=present_day_10min_mask_file,
                                                     file_type=".nc",field_type="Generic",
                                                     grid_type="LatLong10min")
        landsea_mask = dynamic_hd.load_field(filename=third_intermediary_land_sea_mask_file,
                                             file_type=".nc",field_type="Generic",
                                             grid_type="LatLong10min")
        #Copy present day Caspian
        landsea_mask.data[756:842,272:328] = landsea_mask_present.data[756:842,272:328]
        dynamic_hd.write_field(filename=self.generated_ls_mask_filepath +
                               file_label + '.nc',
                               field=landsea_mask,file_type=".nc")

    def remove_disconnected_points_from_ICE6G_21k_landsea_mask_and_add_caspian(self):
        """Remove disconnected points from ICE6G landsea mask on 10 minute resolution"""
        file_label = self._generate_file_label()
        ice6g_land_sea_mask_file = path.join(self.orography_path,"Ice6g_c_VM5a_10min_21k.nc")
        present_day_10min_mask_file = path.join(self.ls_masks_path,"glcc_olson_land_cover_data",
                                                "glcc_olson-2.0_lsmask_with_bacseas_upscaled_10min.nc")
        intermediary_land_sea_mask_file = (self.generated_ls_mask_filepath +
                                           "intermediary_" + file_label + '.nc')
        second_intermediary_land_sea_mask_file = (self.generated_ls_mask_filepath +
                                                  "2nd_intermediary_" + file_label + '.nc')
        landsea_mask = dynamic_hd.load_field(filename=ice6g_land_sea_mask_file,
                                             file_type=".nc",field_type="Generic",
                                             fieldname="sftlf", grid_type="LatLong10min")
        landsea_mask.convert_to_binary_mask()
        landsea_mask.invert_data()
        dynamic_hd.write_field(filename=intermediary_land_sea_mask_file,
                               field=landsea_mask,file_type=".nc")
        cc_lsmask_driver.drive_connected_lsmask_creation(input_lsmask_filename=\
                                                         intermediary_land_sea_mask_file,
                                                         output_lsmask_filename=\
                                                         second_intermediary_land_sea_mask_file,
                                                         input_ls_seed_points_filename=None,
                                                         input_ls_seed_points_list_filename=\
                                                         path.join(self.ls_seed_points_path,
                                                                   "lsseedpoints_downscale_HD_ls_seed_points"
                                                                   "_to_10min_lat_lon_true_seas_inc_casp_only"
                                                                   "_20160718_114402.txt"),
                                                         rotate_seeds_about_polar_axis=True,
                                                         use_diagonals_in=True, grid_type='LatLong10min')
        landsea_mask_present = dynamic_hd.load_field(filename=present_day_10min_mask_file,
                                                     file_type=".nc",field_type="Generic",
                                                     grid_type="LatLong10min")
        landsea_mask = dynamic_hd.load_field(filename=second_intermediary_land_sea_mask_file,
                                             file_type=".nc",field_type="Generic",
                                             grid_type="LatLong10min")
        #Copy present day Caspian
        landsea_mask.data[756:842,272:328] = landsea_mask_present.data[756:842,272:328]
        dynamic_hd.write_field(filename=self.generated_ls_mask_filepath +
                               file_label + '.nc',
                               field=landsea_mask,file_type=".nc")


    def remove_disconnected_points_from_ICE6G_0k_landsea_mask_and_add_caspian(self):
        """Remove disconnected points from ICE6G landsea mask on 10 minute resolution"""
        file_label = self._generate_file_label()
        ice6g_land_sea_mask_file = path.join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        present_day_10min_mask_file = path.join(self.ls_masks_path,"glcc_olson_land_cover_data",
                                                "glcc_olson-2.0_lsmask_with_bacseas_upscaled_10min.nc")
        intermediary_land_sea_mask_file = (self.generated_ls_mask_filepath +
                                           "intermediary_" + file_label + '.nc')
        second_intermediary_land_sea_mask_file = (self.generated_ls_mask_filepath +
                                                  "2nd_intermediary_" + file_label + '.nc')
        landsea_mask = dynamic_hd.load_field(filename=ice6g_land_sea_mask_file,
                                             file_type=".nc",field_type="Generic",
                                             fieldname="sftlf", grid_type="LatLong10min")
        landsea_mask.convert_to_binary_mask()
        landsea_mask.invert_data()
        dynamic_hd.write_field(filename=intermediary_land_sea_mask_file,
                               field=landsea_mask,file_type=".nc")
        cc_lsmask_driver.drive_connected_lsmask_creation(input_lsmask_filename=\
                                                         intermediary_land_sea_mask_file,
                                                         output_lsmask_filename=\
                                                         second_intermediary_land_sea_mask_file,
                                                         input_ls_seed_points_filename=None,
                                                         input_ls_seed_points_list_filename=\
                                                         path.join(self.ls_seed_points_path,
                                                                   "lsseedpoints_downscale_HD_ls_seed_"
                                                                   "points_to_10min_lat_lon_true_seas_"
                                                                   "exc_casp_20170608_140500.txt"),
                                                         rotate_seeds_about_polar_axis=True,
                                                         use_diagonals_in=True, grid_type='LatLong10min')
        landsea_mask_present = dynamic_hd.load_field(filename=present_day_10min_mask_file,
                                                     file_type=".nc",field_type="Generic",
                                                     grid_type="LatLong10min")
        landsea_mask = dynamic_hd.load_field(filename=second_intermediary_land_sea_mask_file,
                                             file_type=".nc",field_type="Generic",
                                             grid_type="LatLong10min")
        #Copy present day Caspian
        landsea_mask.data[756:842,272:328] = landsea_mask_present.data[756:842,272:328]
        dynamic_hd.write_field(filename=self.generated_ls_mask_filepath +
                               file_label + '.nc',
                               field=landsea_mask,file_type=".nc")

    def upscale_ETOPO2v2_to_10minute_grid(self):
        """Upscale ETOPO2v2 data to a 10 minutes grid by averaging"""
        file_label = self._generate_file_label()
        etopo2v2_file=path.join(self.orography_path,"ETOPO2v2c_f4.nc")
        intermediary_file=self.generated_orography_filepath + "intermediary_" +file_label + ".nc"
        output_file=self.generated_orography_filepath + file_label + ".nc"
        utilities.upscale_field_driver(input_filename=etopo2v2_file,
                                       output_filename=intermediary_file,
                                       input_grid_type="LatLong2min",
                                       output_grid_type="LatLong10min",
                                       method="Sum",
                                       scalenumbers=True)
        self._apply_transforms_to_field(input_filename=intermediary_file,
                                        output_filename=output_file,flip_ud=True,
                                        rotate180lr=False,invert_data=False,
                                        timeslice=None,
                                        griddescfile=self.ten_minute_grid_filepath,
                                        grid_type='LatLong10min')

    def create_10min_present_day_lsmask_from_model_gaussian_mask(self):
        """Create a 10 minute present day land-sea mask from a gaussian mask from the model"""
        file_label = self._generate_file_label()
        input_filename = path.join(self.ls_masks_path,
                                   "lsmask_from_restart_rid0002_jsbach_70091231.nc")
        intermediary_filename = self.generated_ls_mask_filepath + "intermediary_" + file_label + ".nc"
        outfile = self.generated_ls_mask_filepath + file_label + ".nc"
        utilities.generate_regular_landsea_mask_from_gaussian_landsea_mask(input_filename,
                                                                           intermediary_filename,
                                                                           self.ten_minute_grid_filepath)
        self._apply_transforms_to_field(input_filename=intermediary_filename,
                                        output_filename=outfile,
                                        flip_ud=True, rotate180lr=True, invert_data=True,
                                        grid_type='LatLong10min')

    def create_10min_present_day_lsmask_from_model_ocean_mask(self):
        """Create a 10 minute present day land-sea mask from an ocean mask from the model"""
        file_label = self._generate_file_label()
        input_filename = path.join(self.ls_masks_path,
                                   "hdpara_lsmask_standardGR30s.nc")
        intermediary_filename = self.generated_ls_mask_filepath + "intermediary_" + file_label + ".nc"
        outfile = self.generated_ls_mask_filepath + file_label + ".nc"
        utilities.generate_regular_landsea_mask_from_gaussian_landsea_mask(input_filename,
                                                                           intermediary_filename,
                                                                           self.ten_minute_grid_filepath)
        self._apply_transforms_to_field(input_filename=intermediary_filename,
                                        output_filename=outfile,
                                        flip_ud=True, rotate180lr=True, invert_data=True,
                                        grid_type='LatLong10min')

class Original_HD_Model_RFD_Drivers(Dynamic_HD_Drivers):
    """Drive processes using the present day manually corrected river directions currently in JSBACH"""

    def __init__(self):
        """Class constructor. Set path to various files specific to this set of river directions"""
        super(Original_HD_Model_RFD_Drivers,self).__init__()
        self.corrected_RFD_filepath = path.join(self.rdir_path,"rivdir_vs_1_9_data_from_stefan.nc")
        self.corrected_HD_orography_filepath = path.join(self.orography_path,"topo_hd_vs1_9_data_from_stefan.nc")
        self.current_model_HDparas_filepath = path.join(self.hd_file_path,"hdpara_file_from_current_model.nc")
        self.RFD_from_current_HDparas_filepath = path.join(self.rdir_path,"rdirs_from_current_hdparas.nc")

    def corrected_HD_rdirs_post_processing(self):
        """Run post processing on the present day manually corrected river directions"""
        file_label = self._generate_file_label()
        self._run_postprocessing(self.corrected_RFD_filepath,
                                 output_file_label=file_label,
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 grid_type='HD')

    def extract_ls_mask_from_corrected_HD_rdirs(self):
        """Extract a landsea mask from the present day manually corrected river directions"""
        file_label = self._generate_file_label()
        utilities.extract_ls_mask_from_rdirs(rdirs_filename=self.corrected_RFD_filepath,
                                             lsmask_filename=self.generated_ls_mask_filepath +\
                                             file_label + '.nc',
                                             grid_type='HD')

    def extract_true_sinks_from_corrected_HD_rdirs(self):
        """Extact a field of true sinks from the present day manually corrected river directions"""
        file_label = self._generate_file_label()
        utilities.extract_true_sinks_from_rdirs(rdirs_filename=self.corrected_RFD_filepath,
                                                 truesinks_filename=self.generated_truesinks_path +\
                                                 file_label + '.nc',
                                                 grid_type='HD')

    def extract_current_HD_rdirs_from_hdparas_file(self):
        """Extact the river direction field from the current JSBACH hdparas file"""
        rdirs = dynamic_hd.load_field(self.current_model_HDparas_filepath,
                                      file_type=dynamic_hd.get_file_extension(self.current_model_HDparas_filepath),
                                      field_type="RiverDirections",
                                      unmask=True,
                                      fieldname='FDIR',
                                      grid_type='HD')
        dynamic_hd.write_field(self.RFD_from_current_HDparas_filepath,rdirs,
                               dynamic_hd.get_file_extension(self.RFD_from_current_HDparas_filepath))

    def regenerate_hd_file_without_lakes_and_wetlands(self):
        """Regenerate the current hdparas file without any lakes or wetlands"""
        file_label = self._generate_file_label()
        extracted_ls_mask_path = self.generated_ls_mask_filepath + file_label + '.nc'
        utilities.extract_ls_mask_from_rdirs(rdirs_filename=self.corrected_RFD_filepath,
                                             lsmask_filename=extracted_ls_mask_path,
                                             grid_type='HD')
        transformed_rdirs_filename = self.generated_rdir_filepath + file_label + '_transf.nc'
        transformed_extracted_ls_mask_path = path.splitext(extracted_ls_mask_path)[0] + '_transf' +\
                                            path.splitext(extracted_ls_mask_path)[1]
        transformed_extracted_inverted_ls_mask_path= path.splitext(extracted_ls_mask_path)[0] +\
                                                     '_transf_inv' +\
                                                     path.splitext(extracted_ls_mask_path)[1]
        transformed_orography_filename = self.generated_orography_filepath + file_label + '_transf.nc'
        self._apply_transforms_to_field(input_filename=self.corrected_RFD_filepath,
                                        output_filename=transformed_rdirs_filename,
                                        flip_ud=False, rotate180lr=False, invert_data=False,
                                        timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                        grid_type='HD')
        self._apply_transforms_to_field(input_filename=extracted_ls_mask_path,
                                        output_filename=transformed_extracted_inverted_ls_mask_path,
                                        flip_ud=False, rotate180lr=False, invert_data=True,
                                        timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                        grid_type='HD')
        self._apply_transforms_to_field(input_filename=extracted_ls_mask_path,
                                        output_filename=transformed_extracted_ls_mask_path,
                                        flip_ud=False, rotate180lr=False, invert_data=False,
                                        timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                        grid_type='HD')
        self._apply_transforms_to_field(input_filename=self.corrected_HD_orography_filepath,
                                        output_filename=transformed_orography_filename,
                                        flip_ud=False, rotate180lr=False, invert_data=False,
                                        timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                        grid_type='HD')
        self._generate_flow_parameters(rdir_file=transformed_rdirs_filename,
                                       topography_file=transformed_orography_filename,
                                       inner_slope_file=\
                                       path.join(self.orography_path,'bin_innerslope.dat'),
                                       lsmask_file=transformed_extracted_inverted_ls_mask_path,
                                       null_file=\
                                       path.join(self.null_fields_filepath,'null.dat'),
                                       area_spacing_file=\
                                       path.join(self.grid_areas_and_spacings_filepath,
                                                 'fl_dp_dl.dat'),
                                       orography_variance_file=\
                                       path.join(self.orography_path,'bin_toposig.dat'),
                                       output_dir=path.join(self.flow_params_dirs_path,
                                                            'hd_flow_params' + file_label))
        self._generate_hd_file(rdir_file=path.splitext(transformed_rdirs_filename)[0] + ".dat",
                               lsmask_file=transformed_extracted_ls_mask_path,
                               null_file=\
                               path.join(self.null_fields_filepath,'null.dat'),
                               area_spacing_file=\
                               path.join(self.grid_areas_and_spacings_filepath,
                                         'fl_dp_dl.dat'),
                               hd_grid_specs_file=self.half_degree_grid_filepath,
                               output_file=self.generated_hd_file_path + file_label + '.nc',
                               paras_dir=path.join(self.flow_params_dirs_path,
                                                   'hd_flow_params' + file_label))
        utilities.prepare_hdrestart_file_driver(base_hdrestart_filename=self.base_hd_restart_file,
                                                output_hdrestart_filename=self.generated_hd_restart_file_path +
                                                    file_label + '.nc',
                                                hdparas_filename=self.generated_hd_file_path + file_label + '.nc',
                                                ref_hdparas_filename=self.ref_hd_paras_file,
                                                timeslice=None,
                                                res_num_data_rotate180lr=False,
                                                res_num_data_flipup=False,
                                                res_num_ref_rotate180lr=False,
                                                res_num_ref_flipud=False, grid_type='HD')
        raise UserWarning("This function will only produce the expected results if paragen.f is"
                          " manually returned to its original setup")

class ETOPO1_Data_Drivers(Dynamic_HD_Drivers):
    """Drivers for working on the ETOPO1 orography dataset"""

    def __init__(self):
        """Class constructor. Setup path to the ETOPO1 dataset"""
        super(ETOPO1_Data_Drivers,self).__init__()
        self.etopo1_data_filepath = path.join(self.orography_path,'ETOPO1_Ice_c_gmt4.nc')

    def etopo1_data_all_points(self):
        """Generate the naive river direction from the ETOPO data without sink filling"""
        file_label = self._generate_file_label()
        dynamic_hd.main(new_orography_file=self.etopo1_data_filepath,
                        grid_type='LatLong1min',
                        updated_RFD_file=self.generated_rdir_filepath + file_label + '.nc',
                        recompute_changed_orography_only=False,
                        recompute_significant_gradient_changes_only=False)
        self._run_postprocessing(rdirs_filename=self.generated_rdir_filepath + file_label + '.nc',
                                 output_file_label=file_label,
                                 grid_type='LatLong1min')

    def etopo1_data_ALG4_sinkless(self):
        """Generate sinkless river directions from the ETOPO data using algorithm 4 of Barnes et al 2014"""
        file_label = self._generate_file_label()
        orography_filename = path.join(self.etopo1_data_filepath)
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        ls_mask_filename = self.generated_ls_mask_filepath + file_label + '.nc'
        connected_ls_mask_filename = self.generated_ls_mask_filepath + 'connected_' +\
            file_label + '.nc'
        unsorted_catchments_filename = self.generated_catchments_path + 'unsorted_' +\
            file_label + '.nc'
        truesinks_filename = self.generated_truesinks_path + file_label + '.nc'
        ls_seedpoints_filename = path.\
            join(self.ls_seed_points_path,
                 'lsseedpoints_downscale_HD_ls_seed_points_to_1min_lat_lon_20160530_160506.txt')
        utilities.generate_ls_mask(orography_filename=orography_filename,
                                   ls_mask_filename=ls_mask_filename,
                                   sea_level=0.0,
                                   grid_type='LatLong1min')
        cc_lsmask_driver.drive_connected_lsmask_creation(input_lsmask_filename=ls_mask_filename,
                                                         output_lsmask_filename=\
                                                            connected_ls_mask_filename,
                                                         input_ls_seed_points_filename=None,
                                                         input_ls_seed_points_list_filename=\
                                                         ls_seedpoints_filename,
                                                         use_diagonals_in=True,
                                                         rotate_seeds_about_polar_axis=False,
                                                         grid_type='LatLong1min')
        utilities.downscale_true_sink_points_driver(input_fine_orography_filename=\
                                                        orography_filename,
                                                    input_course_truesinks_filename=\
                                                        self.hd_truesinks_filepath,
                                                    output_fine_truesinks_filename=\
                                                        truesinks_filename,
                                                    input_fine_orography_grid_type='LatLong1min',
                                                    input_course_truesinks_grid_type='HD',
                                                    flip_course_grid_ud=True,
                                                    rotate_course_true_sink_about_polar_axis=False)
        fill_sinks_driver.generate_sinkless_flow_directions(filename=orography_filename,
                                                           output_filename=rdirs_filename,
                                                           ls_mask_filename=\
                                                           connected_ls_mask_filename,
                                                           truesinks_filename=truesinks_filename,
                                                           catchment_nums_filename=\
                                                           unsorted_catchments_filename,
                                                           grid_type='LatLong1min')
        self._run_postprocessing(rdirs_filename=rdirs_filename,
                                 output_file_label=file_label,
                                 ls_mask_filename=connected_ls_mask_filename,
                                 compute_catchments=False,
                                 grid_type='LatLong1min')
        self._etopo1_data_ALG4_sinkless_upscale_riverflows_and_river_mouth_flows(file_label)

    def _etopo1_data_ALG4_sinkless_upscale_riverflows_and_river_mouth_flows(self,original_data_file_label,
                                                                            new_label=True):
        """Upscale the results of sinkless river direction generation

        Arguments:
        original_data_file_label: string; label of the original data to be upscaled
        new_label: generate a new label (true) or continue to use label input via original_data_file_label
        Returns:nothing
        """

        if new_label:
            upscaled_file_label = self._generate_file_label()
        else:
            upscaled_file_label = original_data_file_label
        utilities.upscale_field_driver(input_filename=self.generated_flowmaps_filepath
                                       + original_data_file_label + '.nc',
                                       output_filename=self.upscaled_flowmaps_filepath
                                       + upscaled_file_label + '.nc',
                                       input_grid_type='LatLong1min',
                                       output_grid_type='HD',
                                       method='Max',
                                       scalenumbers=True)
        utilities.upscale_field_driver(input_filename=self.generated_rmouth_cumulative_flow_path
                                       + original_data_file_label + '.nc',
                                       output_filename=self.upscaled_rmouth_cumulative_flow_path
                                       + upscaled_file_label + '.nc',
                                       input_grid_type='LatLong1min',
                                       output_grid_type='HD',
                                       method='Sum',
                                       scalenumbers=True)

class ICE5G_Data_Drivers(Dynamic_HD_Drivers):
    """Drivers for working on the ICE5G orography dataset"""

    def __init__(self):
        """Class constructor. Setup various filepaths specific to work with this dataset"""
        super(ICE5G_Data_Drivers,self).__init__()
        self.remap_10min_to_HD_grid_weights_filepath = path.join(self.weights_path,
                                                                "weights10mintoHDgrid.nc")
        self.ice5g_orography_corrections_master_filepath = path.join(self.orography_corrections_path,
                                                                     'ice5g_10min_orog_corrs_master.txt')
        self.tarasov_style_upscaled_srtm30_extra_corrections_master_filepath = \
            path.join(self.orography_corrections_path,'tarasov_style_upscaled_srtm30_orog_corrs_master.txt')
        self.ice5g_intelligent_burning_regions_list_master_filepath = path.\
            join(self.intelligent_burning_regions_path,'ice5g_10min_int_burning_regions_master.txt')
        self.hd_data_helper_run = False

    def _ICE5G_as_HD_data_21k_0k_Helper(self):
        """Run various preparatory process common to several other methods.

        Uses the boolean variable hd_data_helper_run to show that it has been run already
        """

        self.hd_data_helper_run = True
        file_label = self._generate_file_label()
        self.ice5g_0k_HD_filepath = self.generated_orography_filepath + '0k_HD' + file_label + ".nc"
        self._prepare_topography(orog_nc_file=path.join(self.orography_path,
                                                        "ice5g_v1_2_00_0k_10min.nc"),
                                 grid_file=self.hd_grid_filepath,
                                 weights_file=self.remap_10min_to_HD_grid_weights_filepath,
                                 output_file=self.ice5g_0k_HD_filepath,
                                 lsmask_file=self.hd_grid_ls_mask_filepath)
        self.ice5g_0k_HD_lsmaskpath = self.generated_ls_mask_filepath + '0k_HD' \
                                    + file_label + '.nc'
        utilities.generate_ls_mask(orography_filename=self.ice5g_0k_HD_filepath,
                                   ls_mask_filename=self.ice5g_0k_HD_lsmaskpath,
                                   sea_level=0.0,
                                   grid_type='HD')
        self.ice5g_21k_HD_filepath = self.generated_orography_filepath + '21k_HD' + file_label + ".nc"
        self._prepare_topography(orog_nc_file=path.join(self.orography_path,
                                                        "ice5g_v1_2_21_0k_10min.nc"),
                                 grid_file=self.hd_grid_filepath,
                                 weights_file=self.remap_10min_to_HD_grid_weights_filepath,
                                 output_file=self.ice5g_21k_HD_filepath,
                                 lsmask_file=self.hd_grid_ls_mask_filepath)
        self.ice5g_21k_HD_lsmaskpath = self.generated_ls_mask_filepath + '21k_HD' \
                                    + file_label + '.nc'
        utilities.generate_ls_mask(orography_filename=self.ice5g_21k_HD_filepath,
                                   ls_mask_filename=self.ice5g_21k_HD_lsmaskpath,
                                   sea_level=0.0,
                                   grid_type='HD')

    def ICE5G_as_HD_data_21k_0k_sig_grad_only_all_neighbours_driver(self):
        """Generate river directions for LGM only where significant gradient changes have occurred"""
        file_label = self._generate_file_label()
        if not self.hd_data_helper_run:
            self._ICE5G_as_HD_data_21k_0k_Helper()
        dynamic_hd.main(new_orography_file=self.ice5g_0k_HD_filepath,
                        grid_type = 'HD',
                        base_orography_file=self.ice5g_21k_HD_filepath,
                        base_RFD_file=self.base_RFD_filepath,
                        updated_RFD_file=self.generated_rdir_filepath+file_label+'.nc',
                        update_mask_file=self.generated_update_masks_filepath+file_label+'.nc',
                        recompute_changed_orography_only=False,
                        recompute_significant_gradient_changes_only=True,
                        gc_absolute_tol=3)
        self._run_postprocessing(rdirs_filename=self.generated_rdir_filepath+
                                 file_label+'.nc',
                                 output_file_label=file_label,grid_type='HD')

    def ICE5G_as_HD_data_all_points_21k(self):
        """Generate naive river directions for all points at LGM"""
        file_label = self._generate_file_label()
        if not self.hd_data_helper_run:
            self._ICE5G_as_HD_data_21k_0k_Helper()
        dynamic_hd.main(new_orography_file=self.ice5g_21k_HD_filepath,
                        grid_type='HD',
                        updated_RFD_file=self.generated_rdir_filepath + file_label+'.nc',
                        recompute_changed_orography_only=False,
                        recompute_significant_gradient_changes_only=False)
        self._run_postprocessing(rdirs_filename=self.generated_rdir_filepath+
                                 file_label+'.nc',
                                 output_file_label=file_label,
                                 ls_mask_filename=self.ice5g_21k_HD_lsmaskpath,
                                 grid_type='HD')

    def ICE5G_as_HD_data_all_points_0k(self):
        """Generate naive river direction for all points at the present day"""
        file_label = self._generate_file_label()
        if not self.hd_data_helper_run:
            self._ICE5G_as_HD_data_21k_0k_Helper()
        dynamic_hd.main(new_orography_file=self.ice5g_0k_HD_filepath,
                        grid_type='HD',
                        updated_RFD_file=self.generated_rdir_filepath + file_label+'.nc',
                        recompute_changed_orography_only=False,
                        recompute_significant_gradient_changes_only=False)
        self._run_postprocessing(rdirs_filename=self.generated_rdir_filepath+
                                 file_label+'.nc',
                                 output_file_label=file_label,
                                 ls_mask_filename=self.ice5g_0k_HD_lsmaskpath,
                                 grid_type='HD')

    def ICE5G_as_HD_data_ALG4_sinkless_all_points_0k(self):
        """Generate sinkless river direction for all points at the present day after upscaling ICE5G data to the HD grid"""
        file_label = self._generate_file_label()
        if not self.hd_data_helper_run:
            self._ICE5G_as_HD_data_21k_0k_Helper()
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        connected_lsmask = self.generated_ls_mask_filepath + 'connected_' + file_label + '.nc'
        truesinks_filename = path.join(self.truesinks_path,
                                       "truesinks_extract_true_sinks_from_corrected_HD_rdirs"
                                       "_20160527_105218.nc")
        ls_seedpoints_filename = path.join(self.ls_seed_points_path,
                                           'lsseedpoints_HD_160530_0001900.txt')
        cc_lsmask_driver.drive_connected_lsmask_creation(self.ice5g_0k_HD_lsmaskpath,
                                                         output_lsmask_filename=connected_lsmask,
                                                         input_ls_seed_points_list_filename=\
                                                         ls_seedpoints_filename,
                                                         use_diagonals_in=True,
                                                         rotate_seeds_about_polar_axis=True,
                                                         grid_type='HD')
        fill_sinks_driver.generate_sinkless_flow_directions(filename=self.ice5g_0k_HD_filepath,
                                                           output_filename=rdirs_filename,
                                                           ls_mask_filename=connected_lsmask,
                                                           truesinks_filename=truesinks_filename,
                                                           grid_type='HD')
        self._run_postprocessing(rdirs_filename=self.generated_rdir_filepath+
                                 file_label+'.nc',
                                 output_file_label=file_label,
                                 ls_mask_filename=self.ice5g_0k_HD_lsmaskpath,
                                 grid_type='HD')

    def ICE5G_data_all_points_0k(self):
        """Generate naive river directions using the ICE5G data for the present day"""
        file_label = self._generate_file_label()
        dynamic_hd.main(new_orography_file=path.join(self.orography_path,
                                                     "ice5g_v1_2_00_0k_10min.nc"),
                        grid_type='LatLong10min',
                        updated_RFD_file=self.generated_rdir_filepath + file_label+'.nc',
                        recompute_changed_orography_only=False,
                        recompute_significant_gradient_changes_only=False)
        self._run_postprocessing(rdirs_filename=self.generated_rdir_filepath+
                                 file_label+'.nc',
                                 output_file_label=file_label,grid_type='LatLong10min')

    def ICE5G_data_all_points_21k(self):
        """Generate naive river directions using the ICE5G data at the LGM"""
        file_label = self._generate_file_label()
        dynamic_hd.main(new_orography_file=path.join(self.orography_path,
                                                     "ice5g_v1_2_21_0k_10min.nc"),
                        grid_type='LatLong10min',
                        updated_RFD_file=self.generated_rdir_filepath + file_label+'.nc',
                        recompute_changed_orography_only=False,
                        recompute_significant_gradient_changes_only=False)
        self._run_postprocessing(rdirs_filename=self.generated_rdir_filepath+
                                 file_label+'.nc',
                                 output_file_label=file_label,grid_type='LatLong10min')

    def ICE5G_data_ALG4_sinkless_0k(self):
        """Generate sinkless river directions for the present day using the ICE5G data and a corrected orography"""
        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        orography_filename = self.corrected_orography_filepath + file_label + '.nc'
        self._correct_orography(input_orography_filename=original_orography_filename,
                                input_corrections_list_filename=\
                                self.ice5g_orography_corrections_master_filepath,
                                output_orography_filename=orography_filename,
                                output_file_label=file_label, grid_type='LatLong10min')
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        ls_mask_filename = self.generated_ls_mask_filepath + file_label + '.nc'
        connected_ls_mask_filename = self.generated_ls_mask_filepath + 'connected_' +\
            file_label + '.nc'
        unsorted_catchments_filename = self.generated_catchments_path + 'unsorted_' +\
            file_label + '.nc'
        truesinks_filename = self.generated_truesinks_path + file_label + '.nc'
        ls_seedpoints_filename = path.\
            join(self.ls_seed_points_path,
                 'lsseedpoints_downscale_HD_ls_seed_points_to_10min_lat_lon_20160531_155753.txt')
        #True sinks modifications are no longer used
        truesinks_mods_10min_filename = None
        truesinks_mods_HD_filename = None
        utilities.generate_ls_mask(orography_filename=orography_filename,
                                   ls_mask_filename=ls_mask_filename,
                                   sea_level=0.0,
                                   grid_type='LatLong10min')
        cc_lsmask_driver.drive_connected_lsmask_creation(input_lsmask_filename=ls_mask_filename,
                                                         output_lsmask_filename=\
                                                            connected_ls_mask_filename,
                                                            input_ls_seed_points_filename=None,
                                                            input_ls_seed_points_list_filename=\
                                                                ls_seedpoints_filename,
                                                            use_diagonals_in=True,
                                                            rotate_seeds_about_polar_axis=True,
                                                            grid_type='LatLong10min')
        utilities.downscale_true_sink_points_driver(input_fine_orography_filename=\
                                                        orography_filename,
                                                    input_course_truesinks_filename=\
                                                        self.hd_truesinks_filepath,
                                                    output_fine_truesinks_filename=\
                                                        truesinks_filename,
                                                    input_fine_orography_grid_type=\
                                                        'LatLong10min',
                                                    input_course_truesinks_grid_type='HD',
                                                    flip_course_grid_ud=True,
                                                    rotate_course_true_sink_about_polar_axis=True,
                                                    downscaled_true_sink_modifications_filename=\
                                                        truesinks_mods_10min_filename,
                                                    course_true_sinks_modifications_filename=\
                                                        truesinks_mods_HD_filename)
        fill_sinks_driver.generate_sinkless_flow_directions(filename=orography_filename,
                                                            output_filename=rdirs_filename,
                                                            ls_mask_filename=\
                                                            connected_ls_mask_filename,
                                                            truesinks_filename=truesinks_filename,
                                                            catchment_nums_filename=\
                                                            unsorted_catchments_filename,
                                                            grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=rdirs_filename,
                                 output_file_label=file_label,
                                 ls_mask_filename=connected_ls_mask_filename,
                                 compute_catchments=False,
                                 grid_type='LatLong10min')
        self._ICE5G_data_ALG4_sinkless_0k_upscale_riverflows_and_river_mouth_flows(file_label,new_label=False)

    def ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k(self):
        """Generate sinkless river directions for the present day using a corrected orography and a downscaled HD lsmask"""
        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        super_fine_orography_filename = path.join(self.orography_path,"ETOPO1_Ice_c_gmt4.nc")
        super_fine_flowmap_filename = path.join(self.flowmaps_path,
                                                "flowmap_etopo1_data_ALG4_sinkless_20160603_112520.nc")
        intermediary_orography_filename = self.corrected_orography_filepath +\
                                          "intermediary_" + file_label + '.nc'
        orography_filename = self.corrected_orography_filepath + file_label + '.nc'
        orography_corrections_field_filename = self.generated_orography_corrections_fields_path +\
                                                file_label + '.nc'
        self._correct_orography(input_orography_filename=original_orography_filename,
                                input_corrections_list_filename=\
                                self.ice5g_orography_corrections_master_filepath,
                                output_orography_filename=intermediary_orography_filename,
                                output_file_label=file_label, grid_type='LatLong10min')
        self._apply_intelligent_burning(input_orography_filename=\
                                        intermediary_orography_filename,
                                        input_superfine_orography_filename=\
                                        super_fine_orography_filename,
                                        input_superfine_flowmap_filename=\
                                        super_fine_flowmap_filename,
                                        input_intelligent_burning_regions_list=\
                                        self.ice5g_intelligent_burning_regions_list_master_filepath,
                                        output_orography_filename=orography_filename,
                                        output_file_label=file_label,
                                        grid_type='LatLong10min',
                                        super_fine_grid_type='LatLong1min')
        utilities.generate_orog_correction_field(original_orography_filename=\
                                                 original_orography_filename,
                                                 corrected_orography_filename=\
                                                 orography_filename,
                                                 orography_corrections_filename=\
                                                 orography_corrections_field_filename,
                                                 grid_type='LatLong10min')
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        connected_ls_mask_filename = self.generated_ls_mask_filepath + 'connected_' +\
            file_label + '.nc'
        unsorted_catchments_filename = self.generated_catchments_path + 'unsorted_' +\
            file_label + '.nc'
        truesinks_filename = self.generated_truesinks_path + file_label + '.nc'
        HD_ls_mask_filename = self.generated_ls_mask_filepath +\
                              "extract_ls_mask_from_corrected_HD_rdirs_20160504_142435.nc"
        #True sinks modifications are no longer used
        truesinks_mods_10min_filename = None
        truesinks_mods_HD_filename = None
        utilities.downscale_ls_mask_driver(input_course_ls_mask_filename=\
                                           HD_ls_mask_filename,
                                           output_fine_ls_mask_filename=\
                                           connected_ls_mask_filename,
                                           input_flipud=True,
                                           input_rotate180lr=True,
                                           course_grid_type='HD',
                                           fine_grid_type='LatLong10min')
        utilities.downscale_true_sink_points_driver(input_fine_orography_filename=\
                                                        orography_filename,
                                                    input_course_truesinks_filename=\
                                                        self.hd_truesinks_filepath,
                                                    output_fine_truesinks_filename=\
                                                        truesinks_filename,
                                                    input_fine_orography_grid_type=\
                                                        'LatLong10min',
                                                    input_course_truesinks_grid_type='HD',
                                                    flip_course_grid_ud=True,
                                                    rotate_course_true_sink_about_polar_axis=True,
                                                    downscaled_true_sink_modifications_filename=\
                                                        truesinks_mods_10min_filename,
                                                    course_true_sinks_modifications_filename=\
                                                        truesinks_mods_HD_filename)
        fill_sinks_driver.generate_sinkless_flow_directions(filename=orography_filename,
                                                            output_filename=rdirs_filename,
                                                            ls_mask_filename=\
                                                            connected_ls_mask_filename,
                                                            truesinks_filename=truesinks_filename,
                                                            catchment_nums_filename=\
                                                            unsorted_catchments_filename,
                                                            grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=rdirs_filename,
                                 output_file_label=file_label,
                                 ls_mask_filename=connected_ls_mask_filename,
                                 compute_catchments=False,
                                 flip_mask_ud=True,
                                 grid_type='LatLong10min')
        self._ICE5G_data_ALG4_sinkless_0k_upscale_riverflows_and_river_mouth_flows(file_label,new_label=False)

    def ICE5G_and_tarasov_upscaled_srtm30plus_data_ALG4_sinkless_downscaled_ls_mask_0k(self):
        """Generate sinkless flow direction from a tarasov-style upscaled srtm30plus orogoraphy then upscale to HD grid

        The actual river direction come from the tarasov-style upscaled srtm30plus but the correction field produced is
        relative to the ICE5G orography
        """

        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        original_tarasov_upscaled_orography_filename = path.join(self.orography_path,"tarasov_upscaled",
                                                                 "upscaled_orog_upscale_srtm30_plus_orog_"
                                                                 "to_10min_no_lsmask_half_cell_upscaling_"
                                                                 "params_20170507_214815.nc")
        original_tarasov_upscaled_orography_flipped_ud_filename = self.generated_orography_filepath +\
                                                                  "original_tarasov_orog_flipped_" +\
                                                                  file_label + '.nc'
        super_fine_orography_filename = path.join(self.orography_path,"ETOPO1_Ice_c_gmt4.nc")
        super_fine_flowmap_filename = path.join(self.flowmaps_path,
                                                "flowmap_etopo1_data_ALG4_sinkless_20160603_112520.nc")
        intermediary_orography_filename = self.corrected_orography_filepath +\
                                          "intermediary_" + file_label + '.nc'
        second_intermediary_orography_filename = self.corrected_orography_filepath +\
                                          "2nd_intermediary_" + file_label + '.nc'
        third_intermediary_orography_filename = self.corrected_orography_filepath +\
                                          "3rd_intermediary_" + file_label + '.nc'
        orography_filename = self.corrected_orography_filepath + file_label + '.nc'
        orography_corrections_field_filename = self.generated_orography_corrections_fields_path +\
                                                file_label + '.nc'
        self._apply_transforms_to_field(input_filename=original_tarasov_upscaled_orography_filename,
                                        output_filename=original_tarasov_upscaled_orography_flipped_ud_filename,
                                        flip_ud=True, rotate180lr=True, invert_data=False,griddescfile=None,
                                        grid_type="LatLong10min")
        self._correct_orography(input_orography_filename=original_orography_filename,
                                input_corrections_list_filename=\
                                self.ice5g_orography_corrections_master_filepath,
                                output_orography_filename=intermediary_orography_filename,
                                output_file_label=file_label, grid_type='LatLong10min')
        self._apply_intelligent_burning(input_orography_filename=\
                                        intermediary_orography_filename,
                                        input_superfine_orography_filename=\
                                        super_fine_orography_filename,
                                        input_superfine_flowmap_filename=\
                                        super_fine_flowmap_filename,
                                        input_intelligent_burning_regions_list=\
                                        self.ice5g_intelligent_burning_regions_list_master_filepath,
                                        output_orography_filename=second_intermediary_orography_filename,
                                        output_file_label=file_label,
                                        grid_type='LatLong10min',
                                        super_fine_grid_type='LatLong1min')
        utilities.merge_corrected_and_tarasov_upscaled_orography(input_corrected_orography_file=\
                                                                 second_intermediary_orography_filename,
                                                                 input_tarasov_upscaled_orography_file=\
                                                                 original_tarasov_upscaled_orography_flipped_ud_filename,
                                                                 output_merged_orography_file=\
                                                                 third_intermediary_orography_filename,
                                                                 grid_type='LatLong10min')
        self._correct_orography(input_orography_filename=third_intermediary_orography_filename,
                                input_corrections_list_filename=\
                                self.tarasov_style_upscaled_srtm30_extra_corrections_master_filepath,
                                output_orography_filename=orography_filename,
                                output_file_label=file_label,
                                grid_type="LatLong10min")
        utilities.generate_orog_correction_field(original_orography_filename=\
                                                 original_orography_filename,
                                                 corrected_orography_filename=\
                                                 orography_filename,
                                                 orography_corrections_filename=\
                                                 orography_corrections_field_filename,
                                                 grid_type='LatLong10min')
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        connected_ls_mask_filename = self.generated_ls_mask_filepath + 'connected_' +\
            file_label + '.nc'
        unsorted_catchments_filename = self.generated_catchments_path + 'unsorted_' +\
            file_label + '.nc'
        truesinks_filename = self.generated_truesinks_path + file_label + '.nc'
        HD_ls_mask_filename = self.generated_ls_mask_filepath +\
                              "extract_ls_mask_from_corrected_HD_rdirs_20160504_142435.nc"
        #True sinks modifications are no longer used
        truesinks_mods_10min_filename = None
        truesinks_mods_HD_filename = None
        utilities.downscale_ls_mask_driver(input_course_ls_mask_filename=\
                                           HD_ls_mask_filename,
                                           output_fine_ls_mask_filename=\
                                           connected_ls_mask_filename,
                                           input_flipud=True,
                                           input_rotate180lr=True,
                                           course_grid_type='HD',
                                           fine_grid_type='LatLong10min')
        utilities.downscale_true_sink_points_driver(input_fine_orography_filename=\
                                                        orography_filename,
                                                    input_course_truesinks_filename=\
                                                        self.hd_truesinks_filepath,
                                                    output_fine_truesinks_filename=\
                                                        truesinks_filename,
                                                    input_fine_orography_grid_type=\
                                                        'LatLong10min',
                                                    input_course_truesinks_grid_type='HD',
                                                    flip_course_grid_ud=True,
                                                    rotate_course_true_sink_about_polar_axis=True,
                                                    downscaled_true_sink_modifications_filename=\
                                                        truesinks_mods_10min_filename,
                                                    course_true_sinks_modifications_filename=\
                                                        truesinks_mods_HD_filename)
        fill_sinks_driver.generate_sinkless_flow_directions(filename=orography_filename,
                                                            output_filename=rdirs_filename,
                                                            ls_mask_filename=\
                                                            connected_ls_mask_filename,
                                                            truesinks_filename=truesinks_filename,
                                                            catchment_nums_filename=\
                                                            unsorted_catchments_filename,
                                                            grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=rdirs_filename,
                                 output_file_label=file_label,
                                 ls_mask_filename=connected_ls_mask_filename,
                                 compute_catchments=False,
                                 flip_mask_ud=True,
                                 grid_type='LatLong10min')
        self._ICE5G_data_ALG4_sinkless_0k_upscale_riverflows_and_river_mouth_flows(file_label,new_label=False)

    def ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_downscaled_ls_mask_0k(self):
        """Generate sinkless flow direction from a tarasov-style upscaled srtm30plus orogoraphy then upscale to HD grid

        The actual river direction come from the tarasov-style upscaled srtm30plus but the correction field produced is
        relative to the ICE5G orography
        """

        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        original_tarasov_upscaled_orography_filename = path.join(self.orography_path,"tarasov_upscaled",
                                                                 "upscaled_orog_upscale_srtm30_plus_orog_"
                                                                 "to_10min_no_lsmask_half_cell_upscaling_"
                                                                 "params_20170507_214815.nc")
        original_tarasov_upscaled_orography_flipped_ud_filename = self.generated_orography_filepath +\
                                                                  "original_tarasov_orog_flipped_" +\
                                                                  file_label + '.nc'
        super_fine_orography_filename = path.join(self.orography_path,"ETOPO1_Ice_c_gmt4.nc")
        super_fine_flowmap_filename = path.join(self.flowmaps_path,
                                                "flowmap_etopo1_data_ALG4_sinkless_20160603_112520.nc")
        intermediary_orography_filename = self.corrected_orography_filepath +\
                                          "intermediary_" + file_label + '.nc'
        second_intermediary_orography_filename = self.corrected_orography_filepath +\
                                          "2nd_intermediary_" + file_label + '.nc'
        third_intermediary_orography_filename = self.corrected_orography_filepath +\
                                          "3rd_intermediary_" + file_label + '.nc'
        orography_filename = self.corrected_orography_filepath + file_label + '.nc'
        orography_corrections_field_filename = self.generated_orography_corrections_fields_path +\
                                                file_label + '.nc'
        self._apply_transforms_to_field(input_filename=original_tarasov_upscaled_orography_filename,
                                        output_filename=original_tarasov_upscaled_orography_flipped_ud_filename,
                                        flip_ud=True, rotate180lr=True, invert_data=False,griddescfile=None,
                                        grid_type="LatLong10min")
        self._correct_orography(input_orography_filename=original_orography_filename,
                                input_corrections_list_filename=\
                                self.ice5g_orography_corrections_master_filepath,
                                output_orography_filename=intermediary_orography_filename,
                                output_file_label=file_label, grid_type='LatLong10min')
        self._apply_intelligent_burning(input_orography_filename=\
                                        intermediary_orography_filename,
                                        input_superfine_orography_filename=\
                                        super_fine_orography_filename,
                                        input_superfine_flowmap_filename=\
                                        super_fine_flowmap_filename,
                                        input_intelligent_burning_regions_list=\
                                        self.ice5g_intelligent_burning_regions_list_master_filepath,
                                        output_orography_filename=second_intermediary_orography_filename,
                                        output_file_label=file_label,
                                        grid_type='LatLong10min',
                                        super_fine_grid_type='LatLong1min')
        utilities.merge_corrected_and_tarasov_upscaled_orography(input_corrected_orography_file=\
                                                                 second_intermediary_orography_filename,
                                                                 input_tarasov_upscaled_orography_file=\
                                                                 original_tarasov_upscaled_orography_flipped_ud_filename,
                                                                 output_merged_orography_file=\
                                                                 third_intermediary_orography_filename,
                                                                 use_upscaled_orography_only_in_region="North America",
                                                                 grid_type='LatLong10min')
        self._correct_orography(input_orography_filename=third_intermediary_orography_filename,
                                input_corrections_list_filename=\
                                self.tarasov_style_upscaled_srtm30_extra_corrections_master_filepath,
                                output_orography_filename=orography_filename,
                                output_file_label=file_label,
                                grid_type="LatLong10min")
        utilities.generate_orog_correction_field(original_orography_filename=\
                                                 original_orography_filename,
                                                 corrected_orography_filename=\
                                                 orography_filename,
                                                 orography_corrections_filename=\
                                                 orography_corrections_field_filename,
                                                 grid_type='LatLong10min')
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        connected_ls_mask_filename = self.generated_ls_mask_filepath + 'connected_' +\
            file_label + '.nc'
        unsorted_catchments_filename = self.generated_catchments_path + 'unsorted_' +\
            file_label + '.nc'
        truesinks_filename = self.generated_truesinks_path + file_label + '.nc'
        HD_ls_mask_filename = self.generated_ls_mask_filepath +\
                              "extract_ls_mask_from_corrected_HD_rdirs_20160504_142435.nc"
        #True sinks modifications are no longer used
        truesinks_mods_10min_filename = None
        truesinks_mods_HD_filename = None
        utilities.downscale_ls_mask_driver(input_course_ls_mask_filename=\
                                           HD_ls_mask_filename,
                                           output_fine_ls_mask_filename=\
                                           connected_ls_mask_filename,
                                           input_flipud=True,
                                           input_rotate180lr=True,
                                           course_grid_type='HD',
                                           fine_grid_type='LatLong10min')
        utilities.downscale_true_sink_points_driver(input_fine_orography_filename=\
                                                        orography_filename,
                                                    input_course_truesinks_filename=\
                                                        self.hd_truesinks_filepath,
                                                    output_fine_truesinks_filename=\
                                                        truesinks_filename,
                                                    input_fine_orography_grid_type=\
                                                        'LatLong10min',
                                                    input_course_truesinks_grid_type='HD',
                                                    flip_course_grid_ud=True,
                                                    rotate_course_true_sink_about_polar_axis=True,
                                                    downscaled_true_sink_modifications_filename=\
                                                        truesinks_mods_10min_filename,
                                                    course_true_sinks_modifications_filename=\
                                                        truesinks_mods_HD_filename)
        fill_sinks_driver.generate_sinkless_flow_directions(filename=orography_filename,
                                                            output_filename=rdirs_filename,
                                                            ls_mask_filename=\
                                                            connected_ls_mask_filename,
                                                            truesinks_filename=truesinks_filename,
                                                            catchment_nums_filename=\
                                                            unsorted_catchments_filename,
                                                            grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=rdirs_filename,
                                 output_file_label=file_label,
                                 ls_mask_filename=connected_ls_mask_filename,
                                 compute_catchments=False,
                                 flip_mask_ud=True,
                                 grid_type='LatLong10min')
        self._ICE5G_data_ALG4_sinkless_0k_upscale_riverflows_and_river_mouth_flows(file_label,new_label=False)

    def ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson_lsmask_0k(self):
        """Generate sinkless flow direction from a tarasov-style upscaled srtm30plus orogoraphy then upscale to HD grid

        The actual river direction come from the tarasov-style upscaled srtm30plus but the correction field produced is
        relative to the ICE5G orography
        """

        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        original_tarasov_upscaled_orography_filename = path.join(self.orography_path,"tarasov_upscaled",
                                                                 "upscaled_orog_upscale_srtm30_plus_orog_"
                                                                 "to_10min_no_lsmask_half_cell_upscaling_"
                                                                 "params_20170507_214815.nc")
        original_tarasov_upscaled_orography_flipped_ud_filename = self.generated_orography_filepath +\
                                                                  "original_tarasov_orog_flipped_" +\
                                                                  file_label + '.nc'
        super_fine_orography_filename = path.join(self.orography_path,"ETOPO1_Ice_c_gmt4.nc")
        super_fine_flowmap_filename = path.join(self.flowmaps_path,
                                                "flowmap_etopo1_data_ALG4_sinkless_20160603_112520.nc")
        intermediary_orography_filename = self.corrected_orography_filepath +\
                                          "intermediary_" + file_label + '.nc'
        second_intermediary_orography_filename = self.corrected_orography_filepath +\
                                          "2nd_intermediary_" + file_label + '.nc'
        third_intermediary_orography_filename = self.corrected_orography_filepath +\
                                          "3rd_intermediary_" + file_label + '.nc'
        orography_filename = self.corrected_orography_filepath + file_label + '.nc'
        orography_corrections_field_filename = self.generated_orography_corrections_fields_path +\
                                                file_label + '.nc'
        self._apply_transforms_to_field(input_filename=original_tarasov_upscaled_orography_filename,
                                        output_filename=original_tarasov_upscaled_orography_flipped_ud_filename,
                                        flip_ud=True, rotate180lr=True, invert_data=False,griddescfile=None,
                                        grid_type="LatLong10min")
        self._correct_orography(input_orography_filename=original_orography_filename,
                                input_corrections_list_filename=\
                                self.ice5g_orography_corrections_master_filepath,
                                output_orography_filename=intermediary_orography_filename,
                                output_file_label=file_label, grid_type='LatLong10min')
        self._apply_intelligent_burning(input_orography_filename=\
                                        intermediary_orography_filename,
                                        input_superfine_orography_filename=\
                                        super_fine_orography_filename,
                                        input_superfine_flowmap_filename=\
                                        super_fine_flowmap_filename,
                                        input_intelligent_burning_regions_list=\
                                        self.ice5g_intelligent_burning_regions_list_master_filepath,
                                        output_orography_filename=second_intermediary_orography_filename,
                                        output_file_label=file_label,
                                        grid_type='LatLong10min',
                                        super_fine_grid_type='LatLong1min')
        utilities.merge_corrected_and_tarasov_upscaled_orography(input_corrected_orography_file=\
                                                                 second_intermediary_orography_filename,
                                                                 input_tarasov_upscaled_orography_file=\
                                                                 original_tarasov_upscaled_orography_flipped_ud_filename,
                                                                 output_merged_orography_file=\
                                                                 third_intermediary_orography_filename,
                                                                 use_upscaled_orography_only_in_region="North America",
                                                                 grid_type='LatLong10min')
        self._correct_orography(input_orography_filename=third_intermediary_orography_filename,
                                input_corrections_list_filename=\
                                self.tarasov_style_upscaled_srtm30_extra_corrections_master_filepath,
                                output_orography_filename=orography_filename,
                                output_file_label=file_label,
                                grid_type="LatLong10min")
        utilities.generate_orog_correction_field(original_orography_filename=\
                                                 original_orography_filename,
                                                 corrected_orography_filename=\
                                                 orography_filename,
                                                 orography_corrections_filename=\
                                                 orography_corrections_field_filename,
                                                 grid_type='LatLong10min')
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        original_connected_ls_mask_filename = path.join(self.ls_masks_path,'generated',
                                                        "ls_mask_recreate_connected_10min_"
                                                        "lsmask_from_glcc_olson_data_"
                                                        "20170513_195421.nc")
        connected_ls_mask_filename = self.generated_ls_mask_filepath +\
                                     file_label + "_flipped.nc"
        self._apply_transforms_to_field(input_filename=original_connected_ls_mask_filename,
                                        output_filename=connected_ls_mask_filename,
                                        flip_ud=True, rotate180lr=False, invert_data=False,griddescfile=None,
                                        grid_type="LatLong10min")
        unsorted_catchments_filename = self.generated_catchments_path + 'unsorted_' +\
            file_label + '.nc'
        truesinks_filename = self.generated_truesinks_path + file_label + '.nc'
        truesinks_mods_10min_filename = path.join(self.truesinks_modifications_filepath,
                                                  "truesinks_mods_for_HD_downscaled_to_"
                                                  "10min_add_in_aral_sea_and_lake_chad.txt")
        truesinks_mods_HD_filename = None
        utilities.downscale_true_sink_points_driver(input_fine_orography_filename=\
                                                        orography_filename,
                                                    input_course_truesinks_filename=\
                                                        self.hd_truesinks_filepath,
                                                    output_fine_truesinks_filename=\
                                                        truesinks_filename,
                                                    input_fine_orography_grid_type=\
                                                        'LatLong10min',
                                                    input_course_truesinks_grid_type='HD',
                                                    flip_course_grid_ud=True,
                                                    rotate_course_true_sink_about_polar_axis=True,
                                                    downscaled_true_sink_modifications_filename=\
                                                        truesinks_mods_10min_filename,
                                                    course_true_sinks_modifications_filename=\
                                                        truesinks_mods_HD_filename)
        fill_sinks_driver.generate_sinkless_flow_directions(filename=orography_filename,
                                                            output_filename=rdirs_filename,
                                                            ls_mask_filename=\
                                                            connected_ls_mask_filename,
                                                            truesinks_filename=truesinks_filename,
                                                            catchment_nums_filename=\
                                                            unsorted_catchments_filename,
                                                            grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=rdirs_filename,
                                 output_file_label=file_label,
                                 ls_mask_filename=connected_ls_mask_filename,
                                 compute_catchments=False,
                                 flip_mask_ud=True,
                                 grid_type='LatLong10min')
        self._ICE5G_data_ALG4_sinkless_0k_upscale_riverflows_and_river_mouth_flows(file_label,new_label=False)

    def ICE5G_data_ALG4_sinkless_no_true_sinks_0k(self):
        """Generate sinkless river directions using a connected landsea mask and no true sinks"""
        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        orography_filename = self.corrected_orography_filepath + file_label + '.nc'
        self._correct_orography(input_orography_filename=original_orography_filename,
                                input_corrections_list_filename=\
                                self.ice5g_orography_corrections_master_filepath,
                                output_orography_filename=orography_filename,
                                output_file_label=file_label, grid_type='LatLong10min')
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        ls_mask_filename = self.generated_ls_mask_filepath + file_label + '.nc'
        connected_ls_mask_filename = self.generated_ls_mask_filepath + 'connected_' +\
            file_label + '.nc'
        unsorted_catchments_filename = self.generated_catchments_path + 'unsorted_' +\
            file_label + '.nc'
        ls_seedpoints_filename = path.\
            join(self.ls_seed_points_path,
                 "lsseedpoints_downscale_HD_ls_seed_points_to_10min"
                 "_lat_lon_true_seas_inc_casp_only_20160718_114402.txt")
        utilities.generate_ls_mask(orography_filename=orography_filename,
                                   ls_mask_filename=ls_mask_filename,
                                   sea_level=0.0,
                                   grid_type='LatLong10min')
        cc_lsmask_driver.drive_connected_lsmask_creation(input_lsmask_filename=ls_mask_filename,
                                                         output_lsmask_filename=\
                                                            connected_ls_mask_filename,
                                                            input_ls_seed_points_filename=None,
                                                            input_ls_seed_points_list_filename=\
                                                                ls_seedpoints_filename,
                                                            use_diagonals_in=True,
                                                            rotate_seeds_about_polar_axis=True,
                                                            grid_type='LatLong10min')
        fill_sinks_driver.generate_sinkless_flow_directions(filename=orography_filename,
                                                            output_filename=rdirs_filename,
                                                            ls_mask_filename=\
                                                            connected_ls_mask_filename,
                                                            truesinks_filename=None,
                                                            catchment_nums_filename=\
                                                            unsorted_catchments_filename,
                                                            grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=rdirs_filename,
                                 output_file_label=file_label,
                                 ls_mask_filename=connected_ls_mask_filename,
                                 compute_catchments=False,
                                 grid_type='LatLong10min')
        self._ICE5G_data_ALG4_sinkless_0k_upscale_riverflows_and_river_mouth_flows(file_label,new_label=False)

    def ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs(self):
        """Generate sinkless flow direction from a downscaled HD lsmask then upscale them to the HD grid"""
        file_label = self._generate_file_label()
        fine_fields_filelabel = "ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_20170514_104220"
        fine_rdirs_filename = self.generated_rdir_with_outflows_marked_filepath + fine_fields_filelabel + ".nc"
        fine_cumulative_flow_filename = self.generated_flowmaps_filepath + fine_fields_filelabel + ".nc"
        output_course_rdirs_filename = self.upscaled_generated_rdir_filepath + file_label + '.nc'
        cotat_plus_parameters_filename = path.join(self.cotat_plus_parameters_path,'cotat_plus_standard_params.nl')
        self._run_cotat_plus_upscaling(input_fine_rdirs_filename=fine_rdirs_filename,
                                       input_fine_cumulative_flow_filename=fine_cumulative_flow_filename,
                                       cotat_plus_parameters_filename=cotat_plus_parameters_filename,
                                       output_course_rdirs_filename=output_course_rdirs_filename,
                                       output_file_label=file_label,
                                       fine_grid_type='LatLong10min',
                                       course_grid_type='HD')
        self._run_postprocessing(rdirs_filename=output_course_rdirs_filename,
                                 output_file_label=file_label,
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 compute_catchments=True, grid_type='HD')
        original_course_cumulative_flow_filename = self.generated_flowmaps_filepath + file_label + '.nc'
        original_course_catchments_filename = self.generated_catchments_path + file_label + '.nc'
        loops_nums_list_filename = self.generated_catchments_path + file_label + '_loops.log'
        updated_file_label = file_label + "_updated"
        updated_course_rdirs_filename = self.upscaled_generated_rdir_filepath + updated_file_label + '.nc'
        loop_breaker_driver.loop_breaker_driver(input_course_rdirs_filepath=output_course_rdirs_filename,
                                                input_course_cumulative_flow_filepath=\
                                                original_course_cumulative_flow_filename,
                                                input_course_catchments_filepath=\
                                                original_course_catchments_filename,
                                                input_fine_rdirs_filepath=\
                                                fine_rdirs_filename,
                                                input_fine_cumulative_flow_filepath=\
                                                fine_cumulative_flow_filename,
                                                output_updated_course_rdirs_filepath=\
                                                updated_course_rdirs_filename,
                                                loop_nums_list_filepath=\
                                                loops_nums_list_filename,
                                                course_grid_type='HD',
                                                fine_grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=updated_course_rdirs_filename,
                                 output_file_label=updated_file_label,
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 compute_catchments=True, grid_type='HD')


    def ICE5G_and_tarasov_upscaled_srtm30plus_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs(self):
        """Generate sinkless flow direction from a downscaled HD lsmask then upscale them to the HD grid"""
        file_label = self._generate_file_label()
        fine_fields_filelabel = ("ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only"
                                 "_data_ALG4_sinkless_downscaled_ls_mask_0k_20170513_213910")
        fine_rdirs_filename = self.generated_rdir_with_outflows_marked_filepath + fine_fields_filelabel + ".nc"
        fine_cumulative_flow_filename = self.generated_flowmaps_filepath + fine_fields_filelabel + ".nc"
        output_course_rdirs_filename = self.upscaled_generated_rdir_filepath + file_label + '.nc'
        cotat_plus_parameters_filename = path.join(self.cotat_plus_parameters_path,'cotat_plus_standard_params.nl')
        self._run_cotat_plus_upscaling(input_fine_rdirs_filename=fine_rdirs_filename,
                                       input_fine_cumulative_flow_filename=fine_cumulative_flow_filename,
                                       cotat_plus_parameters_filename=cotat_plus_parameters_filename,
                                       output_course_rdirs_filename=output_course_rdirs_filename,
                                       output_file_label=file_label,
                                       fine_grid_type='LatLong10min',
                                       course_grid_type='HD')
        self._run_postprocessing(rdirs_filename=output_course_rdirs_filename,
                                 output_file_label=file_label,
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 compute_catchments=True, grid_type='HD')
        original_course_cumulative_flow_filename = self.generated_flowmaps_filepath + file_label + '.nc'
        original_course_catchments_filename = self.generated_catchments_path + file_label + '.nc'
        loops_nums_list_filename = self.generated_catchments_path + file_label + '_loops.log'
        updated_file_label = file_label + "_updated"
        updated_course_rdirs_filename = self.upscaled_generated_rdir_filepath + updated_file_label + '.nc'
        loop_breaker_driver.loop_breaker_driver(input_course_rdirs_filepath=output_course_rdirs_filename,
                                                input_course_cumulative_flow_filepath=\
                                                original_course_cumulative_flow_filename,
                                                input_course_catchments_filepath=\
                                                original_course_catchments_filename,
                                                input_fine_rdirs_filepath=\
                                                fine_rdirs_filename,
                                                input_fine_cumulative_flow_filepath=\
                                                fine_cumulative_flow_filename,
                                                output_updated_course_rdirs_filepath=\
                                                updated_course_rdirs_filename,
                                                loop_nums_list_filepath=\
                                                loops_nums_list_filename,
                                                course_grid_type='HD',
                                                fine_grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=updated_course_rdirs_filename,
                                 output_file_label=updated_file_label,
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 compute_catchments=True, grid_type='HD')

    def ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs(self):
        """Generate sinkless flow direction from a downscaled HD lsmask then upscale them to the HD grid"""
        file_label = self._generate_file_label()
        fine_fields_filelabel = ("ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only"
                                 "_data_ALG4_sinkless_downscaled_ls_mask_0k_20170511_224938")
        fine_rdirs_filename = self.generated_rdir_with_outflows_marked_filepath + fine_fields_filelabel + ".nc"
        fine_cumulative_flow_filename = self.generated_flowmaps_filepath + fine_fields_filelabel + ".nc"
        output_course_rdirs_filename = self.upscaled_generated_rdir_filepath + file_label + '.nc'
        cotat_plus_parameters_filename = path.join(self.cotat_plus_parameters_path,'cotat_plus_standard_params.nl')
        self._run_cotat_plus_upscaling(input_fine_rdirs_filename=fine_rdirs_filename,
                                       input_fine_cumulative_flow_filename=fine_cumulative_flow_filename,
                                       cotat_plus_parameters_filename=cotat_plus_parameters_filename,
                                       output_course_rdirs_filename=output_course_rdirs_filename,
                                       output_file_label=file_label,
                                       fine_grid_type='LatLong10min',
                                       course_grid_type='HD')
        self._run_postprocessing(rdirs_filename=output_course_rdirs_filename,
                                 output_file_label=file_label,
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 compute_catchments=True, grid_type='HD')
        original_course_cumulative_flow_filename = self.generated_flowmaps_filepath + file_label + '.nc'
        original_course_catchments_filename = self.generated_catchments_path + file_label + '.nc'
        loops_nums_list_filename = self.generated_catchments_path + file_label + '_loops.log'
        updated_file_label = file_label + "_updated"
        updated_course_rdirs_filename = self.upscaled_generated_rdir_filepath + updated_file_label + '.nc'
        loop_breaker_driver.loop_breaker_driver(input_course_rdirs_filepath=output_course_rdirs_filename,
                                                input_course_cumulative_flow_filepath=\
                                                original_course_cumulative_flow_filename,
                                                input_course_catchments_filepath=\
                                                original_course_catchments_filename,
                                                input_fine_rdirs_filepath=\
                                                fine_rdirs_filename,
                                                input_fine_cumulative_flow_filepath=\
                                                fine_cumulative_flow_filename,
                                                output_updated_course_rdirs_filepath=\
                                                updated_course_rdirs_filename,
                                                loop_nums_list_filepath=\
                                                loops_nums_list_filename,
                                                course_grid_type='HD',
                                                fine_grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=updated_course_rdirs_filename,
                                 output_file_label=updated_file_label,
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 compute_catchments=True, grid_type='HD')


    def ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson_lsmask_0k_upscale_rdirs(self):
        """Generate sinkless flow direction from a downscaled HD lsmask then upscale them to the HD grid"""
        file_label = self._generate_file_label()
        fine_fields_filelabel = ("ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_"
                                 "sinkless_glcc_olson_lsmask_0k_20170517_003802")
        fine_rdirs_filename = self.generated_rdir_with_outflows_marked_filepath + fine_fields_filelabel + ".nc"
        fine_cumulative_flow_filename = self.generated_flowmaps_filepath + fine_fields_filelabel + ".nc"
        output_course_rdirs_filename = self.upscaled_generated_rdir_filepath + file_label + '.nc'
        cotat_plus_parameters_filename = path.join(self.cotat_plus_parameters_path,'cotat_plus_standard_params.nl')
        self._run_cotat_plus_upscaling(input_fine_rdirs_filename=fine_rdirs_filename,
                                       input_fine_cumulative_flow_filename=fine_cumulative_flow_filename,
                                       cotat_plus_parameters_filename=cotat_plus_parameters_filename,
                                       output_course_rdirs_filename=output_course_rdirs_filename,
                                       output_file_label=file_label,
                                       fine_grid_type='LatLong10min',
                                       course_grid_type='HD')
        self._run_postprocessing(rdirs_filename=output_course_rdirs_filename,
                                 output_file_label=file_label,
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 compute_catchments=True, grid_type='HD')
        original_course_cumulative_flow_filename = self.generated_flowmaps_filepath + file_label + '.nc'
        original_course_catchments_filename = self.generated_catchments_path + file_label + '.nc'
        loops_nums_list_filename = self.generated_catchments_path + file_label + '_loops.log'
        updated_file_label = file_label + "_updated"
        updated_course_rdirs_filename = self.upscaled_generated_rdir_filepath + updated_file_label + '.nc'
        loop_breaker_driver.loop_breaker_driver(input_course_rdirs_filepath=output_course_rdirs_filename,
                                                input_course_cumulative_flow_filepath=\
                                                original_course_cumulative_flow_filename,
                                                input_course_catchments_filepath=\
                                                original_course_catchments_filename,
                                                input_fine_rdirs_filepath=\
                                                fine_rdirs_filename,
                                                input_fine_cumulative_flow_filepath=\
                                                fine_cumulative_flow_filename,
                                                output_updated_course_rdirs_filepath=\
                                                updated_course_rdirs_filename,
                                                loop_nums_list_filepath=\
                                                loops_nums_list_filename,
                                                course_grid_type='HD',
                                                fine_grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=updated_course_rdirs_filename,
                                 output_file_label=updated_file_label,
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 compute_catchments=True, grid_type='HD')

    def ICE5G_data_ALG4_sinkless_21k(self):
        """Generate sinkless river directions at LGM using a fully connected ls mask"""
        file_label = self._generate_file_label()
        orography_filename = path.join(self.orography_path,"ice5g_v1_2_21_0k_10min.nc")
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        ls_mask_filename = self.generated_ls_mask_filepath + file_label + '.nc'
        connected_ls_mask_filename = self.generated_ls_mask_filepath + 'connected_' +\
            file_label + '.nc'
        unsorted_catchments_filename = self.generated_catchments_path + 'unsorted_' +\
            file_label + '.nc'
        truesinks_filename = self.generated_truesinks_path + file_label + '.nc'
        ls_seedpoints_filename = path.\
            join(self.ls_seed_points_path,
                 'lsseedpoints_downscale_HD_ls_seed_points_to_10min_lat_lon_20160531_155753.txt')
        utilities.generate_ls_mask(orography_filename=orography_filename,
                                   ls_mask_filename=ls_mask_filename,
                                   sea_level=0.0,
                                   grid_type='LatLong10min')
        cc_lsmask_driver.drive_connected_lsmask_creation(input_lsmask_filename=ls_mask_filename,
                                                         output_lsmask_filename=\
                                                            connected_ls_mask_filename,
                                                            input_ls_seed_points_filename=None,
                                                            input_ls_seed_points_list_filename=\
                                                                ls_seedpoints_filename,
                                                            use_diagonals_in=True,
                                                            rotate_seeds_about_polar_axis=True,
                                                            grid_type='LatLong10min')
        utilities.downscale_true_sink_points_driver(input_fine_orography_filename=\
                                                        orography_filename,
                                                    input_course_truesinks_filename=\
                                                        self.hd_truesinks_filepath,
                                                    output_fine_truesinks_filename=\
                                                        truesinks_filename,
                                                    input_fine_orography_grid_type=\
                                                        'LatLong10min',
                                                    input_course_truesinks_grid_type='HD',
                                                    flip_course_grid_ud=True,
                                                    rotate_course_true_sink_about_polar_axis=True)
        fill_sinks_driver.generate_sinkless_flow_directions(filename=orography_filename,
                                                            output_filename=rdirs_filename,
                                                            ls_mask_filename=\
                                                            connected_ls_mask_filename,
                                                            truesinks_filename=truesinks_filename,
                                                            catchment_nums_filename=\
                                                            unsorted_catchments_filename,
                                                            grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=rdirs_filename,
                                 output_file_label=file_label,
                                 ls_mask_filename=connected_ls_mask_filename,
                                 compute_catchments=False,
                                 grid_type='LatLong10min')
        self._ICE5G_data_ALG4_sinkless_21k_upscale_riverflows_and_river_mouth_flows(file_label)

    def _ICE5G_data_ALG4_sinkless_21k_upscale_riverflows_and_river_mouth_flows(self,original_data_file_label,
                                                                            new_label=True):
        """Upscale the cumulative flow and river mouth flow of sinkless river directions at the LGM

        Arguments:
        original_data_file_label: string; label of the original data to be upscaled
        new_label: generate a new label (true) or continue to use label input via original_data_file_label
        Returns:nothing
        """

        if new_label:
            upscaled_file_label = self._generate_file_label()
        else:
            upscaled_file_label = original_data_file_label
        utilities.upscale_field_driver(input_filename=self.generated_flowmaps_filepath
                                       + original_data_file_label + '.nc',
                                       output_filename=self.upscaled_flowmaps_filepath
                                       + upscaled_file_label + '.nc',
                                       input_grid_type='LatLong10min',
                                       output_grid_type='HD',
                                       method='Max',
                                       scalenumbers=True)
        utilities.upscale_field_driver(input_filename=self.generated_rmouth_cumulative_flow_path
                                       + original_data_file_label + '.nc',
                                       output_filename=self.upscaled_rmouth_cumulative_flow_path
                                       + upscaled_file_label + '.nc',
                                       input_grid_type='LatLong10min',
                                       output_grid_type='HD',
                                       method='Sum',
                                       scalenumbers=True)

    def _ICE5G_data_ALG4_sinkless_0k_upscale_riverflows_and_river_mouth_flows(self,original_data_file_label,
                                                                                   new_label=True):
        """Upscale the cumulative flow and river mouth flow of sinkless river directions for the present day

        Arguments:
        original_data_file_label: string; label of the original data to be upscaled
        new_label: generate a new label (true) or continue to use label input via original_data_file_label
        Returns:nothing
        """

        if new_label:
            upscaled_file_label = self._generate_file_label()
        else:
            upscaled_file_label = original_data_file_label
        utilities.upscale_field_driver(input_filename=self.generated_flowmaps_filepath
                                       + original_data_file_label + '.nc',
                                       output_filename=self.upscaled_flowmaps_filepath
                                       + upscaled_file_label + '.nc',
                                       input_grid_type='LatLong10min',
                                       output_grid_type='HD',
                                       method='Max',
                                       scalenumbers=True)
        utilities.upscale_field_driver(input_filename=self.generated_rmouth_cumulative_flow_path
                                       + original_data_file_label + '.nc',
                                       output_filename=self.upscaled_rmouth_cumulative_flow_path
                                       + upscaled_file_label + '.nc',
                                       input_grid_type='LatLong10min',
                                       output_grid_type='HD',
                                       method='Sum',
                                       scalenumbers=True)
        utilities.upscale_field_driver(input_filename=self.generated_catchments_path
                                       + "unsorted_"
                                       + original_data_file_label + '.nc',
                                       output_filename=self.upscaled_catchments_path
                                       + "unsorted_"
                                       + upscaled_file_label + '.nc',
                                       input_grid_type='LatLong10min',
                                       output_grid_type='HD',
                                       method='Mode',
                                       scalenumbers=False)

    def ICE_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_from_orog_corrs_field(self):
        """Generate sinkless river directions from the 5minute ICE5G data provided by Virna at a selected timeslice"""
        timeslice=260
        orog_corrections_filename = path.join(self.orography_corrections_fields_path,
                                              "orog_corrs_field_ICE5G_data_ALG4_sink"
                                              "less_downscaled_ls_mask_0k_20160930_001057.nc")
        file_label = "timeslice{0}_".format(timeslice) + self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        orography_filename = self.generated_orography_filepath + file_label + '.nc'
        HD_orography_filename = self.upscaled_orography_filepath + file_label + '_HD' + '.nc'
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        original_ls_mask_filename = path.join(self.ls_masks_path,"mask-final-OR-from-virna.nc")
        upscaled_ls_mask_filename = self.generated_ls_mask_filepath + 'upscaled_' +\
            file_label + '.nc'
        HD_ls_mask_filename = self.generated_ls_mask_filepath + file_label + '_HD' + '.nc'
        unsorted_catchments_filename = self.generated_catchments_path + 'unsorted_' +\
            file_label + '.nc'
        utilities.upscale_field_driver(input_filename=original_ls_mask_filename,
                                       output_filename=upscaled_ls_mask_filename,
                                       input_grid_type='LatLong5min',
                                       output_grid_type='LatLong10min',
                                       method='Max', timeslice=timeslice,
                                       scalenumbers=False)
        utilities.change_dtype(input_filename=upscaled_ls_mask_filename,
                               output_filename=upscaled_ls_mask_filename,
                               new_dtype=np.int32,grid_type='LatLong10min')
        utilities.apply_orog_correction_field(original_orography_filename=original_orography_filename,
                                              orography_corrections_filename=orog_corrections_filename,
                                              corrected_orography_filename=orography_filename,
                                              grid_type="LatLong10min")
        fill_sinks_driver.generate_sinkless_flow_directions(filename=orography_filename,
                                                            output_filename=rdirs_filename,
                                                            ls_mask_filename=\
                                                            upscaled_ls_mask_filename,
                                                            truesinks_filename=None,
                                                            catchment_nums_filename=\
                                                            unsorted_catchments_filename,
                                                            grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=rdirs_filename,
                                 output_file_label=file_label,
                                 ls_mask_filename=upscaled_ls_mask_filename,
                                 compute_catchments=False,
                                 grid_type='LatLong10min')
#         utilities.upscale_field_driver(input_filename=self.generated_flowmaps_filepath
#                                        + original_data_file_label + '.nc',
#                                        output_filename=self.upscaled_flowmaps_filepath
#                                        + upscaled_file_label + '.nc',
#                                        input_grid_type='LatLong10min',
#                                        output_grid_type='HD',
#                                        method='Max',
#                                        scalenumbers=True)
#         utilities.upscale_field_driver(input_filename=self.generated_rmouth_cumulative_flow_path
#                                        + original_data_file_label + '.nc',
#                                        output_filename=self.upscaled_rmouth_cumulative_flow_path
#                                        + upscaled_file_label + '.nc',
#                                        input_grid_type='LatLong10min',
#                                        output_grid_type='HD',
#                                        method='Sum',
#                                        scalenumbers=True)
#         utilities.upscale_field_driver(input_filename=self.generated_catchments_path
#                                        + "unsorted_"
#                                        + original_data_file_label + '.nc',
#                                        output_filename=self.upscaled_catchments_path
#                                        + "unsorted_"
#                                        + upscaled_file_label + '.nc',
#                                        input_grid_type='LatLong10min',
#                                        output_grid_type='HD',
#                                        method='Mode',
#                                        scalenumbers=False)
        fine_cumulative_flow = self.generated_flowmaps_filepath + file_label + '.nc'
        output_course_rdirs_filename = self.upscaled_generated_rdir_filepath + file_label + '.nc'
        cotat_plus_parameters_filename = path.join(self.cotat_plus_parameters_path,'cotat_plus_standard_params.nl')
        self._run_cotat_plus_upscaling(input_fine_rdirs_filename=rdirs_filename,
                                       input_fine_cumulative_flow_filename=fine_cumulative_flow,
                                       cotat_plus_parameters_filename=cotat_plus_parameters_filename,
                                       output_course_rdirs_filename=output_course_rdirs_filename,
                                       output_file_label=file_label,
                                       fine_grid_type='LatLong10min',
                                       course_grid_type='HD')
        upscaled_file_label = file_label + '_upscaled'
        self._run_postprocessing(rdirs_filename=output_course_rdirs_filename,
                                 output_file_label=upscaled_file_label,
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 compute_catchments=True, grid_type='HD')
        original_course_cumulative_flow_filename = self.generated_flowmaps_filepath + upscaled_file_label + '.nc'
        original_course_catchments_filename = self.generated_catchments_path + upscaled_file_label + '.nc'
        loops_nums_list_filename = self.generated_catchments_path + upscaled_file_label + '_loops.log'
        updated_file_label = upscaled_file_label + "_updated"
        updated_course_rdirs_filename = self.upscaled_generated_rdir_filepath + updated_file_label + '.nc'
        loop_breaker_driver.loop_breaker_driver(input_course_rdirs_filepath=output_course_rdirs_filename,
                                                input_course_cumulative_flow_filepath=\
                                                original_course_cumulative_flow_filename,
                                                input_course_catchments_filepath=\
                                                original_course_catchments_filename,
                                                input_fine_rdirs_filepath=\
                                                rdirs_filename,
                                                input_fine_cumulative_flow_filepath=\
                                                fine_cumulative_flow,
                                                output_updated_course_rdirs_filepath=\
                                                updated_course_rdirs_filename,
                                                loop_nums_list_filepath=\
                                                loops_nums_list_filename,
                                                course_grid_type='HD',
                                                fine_grid_type='LatLong10min')
        utilities.upscale_field_driver(input_filename=orography_filename,
                                       output_filename=HD_orography_filename,
                                       input_grid_type='LatLong10min',
                                       output_grid_type='HD',
                                       method='Sum', timeslice=None,
                                       scalenumbers=True)
        utilities.extract_ls_mask_from_rdirs(rdirs_filename=updated_course_rdirs_filename,
                                             lsmask_filename=HD_ls_mask_filename,
                                             grid_type='HD')
        transformed_course_rdirs_filename = path.splitext(updated_course_rdirs_filename)[0] + '_transf' +\
                                            path.splitext(updated_course_rdirs_filename)[1]
        transformed_HD_orography_filename = path.splitext(HD_orography_filename)[0] + '_transf' +\
                                          path.splitext(HD_orography_filename)[1]
        transformed_HD_ls_mask_filename = path.splitext(HD_ls_mask_filename)[0] + '_transf' +\
                                            path.splitext(HD_ls_mask_filename)[1]
        self._apply_transforms_to_field(input_filename=updated_course_rdirs_filename,
                                        output_filename=transformed_course_rdirs_filename,
                                        flip_ud=True, rotate180lr=True, invert_data=False,
                                        timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                        grid_type='HD')
        self._apply_transforms_to_field(input_filename=HD_orography_filename,
                                        output_filename=transformed_HD_orography_filename,
                                        flip_ud=True, rotate180lr=True, invert_data=False,
                                        timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                        grid_type='HD')
        self._apply_transforms_to_field(input_filename=HD_ls_mask_filename,
                                        output_filename=transformed_HD_ls_mask_filename,
                                        flip_ud=True, rotate180lr=True, invert_data=True,
                                        timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                        grid_type='HD')
        self._generate_flow_parameters(rdir_file=transformed_course_rdirs_filename,
                                       topography_file=transformed_HD_orography_filename,
                                       inner_slope_file=\
                                       path.join(self.orography_path,'bin_innerslope.dat'),
                                       lsmask_file=transformed_HD_ls_mask_filename,
                                       null_file=\
                                       path.join(self.null_fields_filepath,'null.dat'),
                                       area_spacing_file=\
                                       path.join(self.grid_areas_and_spacings_filepath,
                                                 'fl_dp_dl.dat'),
                                       orography_variance_file=\
                                       path.join(self.orography_path,'bin_toposig.dat'),
                                       output_dir=path.join(self.flow_params_dirs_path,
                                                            'hd_flow_params' + file_label))
        self._generate_hd_file(rdir_file=path.splitext(transformed_course_rdirs_filename)[0] + ".dat",
                               lsmask_file=path.splitext(transformed_HD_ls_mask_filename)[0] + ".dat",
                               null_file=\
                               path.join(self.null_fields_filepath,'null.dat'),
                               area_spacing_file=\
                               path.join(self.grid_areas_and_spacings_filepath,
                                         'fl_dp_dl.dat'),
                               hd_grid_specs_file=self.half_degree_grid_filepath,
                               output_file=self.generated_hd_file_path + file_label + '.nc',
                               paras_dir=path.join(self.flow_params_dirs_path,
                                                   'hd_flow_params' + file_label))
        self._run_postprocessing(rdirs_filename=updated_course_rdirs_filename,
                                 output_file_label=updated_file_label,
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 compute_catchments=True, grid_type='HD')

    def ICE5G_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs_generation_and_upscaling(self):
        """Generate and upscale sinkless river directions for the present-day"""
        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        original_ls_mask_filename = path.join(self.ls_masks_path,
                                              "generated",
                                              "ls_mask_ICE5G_and_tarasov_upscaled_srtm30plus_"
                                              "north_america_only_data_ALG4_sinkless_glcc_olson"
                                              "_lsmask_0k_20170517_003802_flipped.nc")
        ice5g_glacial_mask_file = path.join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        ten_minute_data_from_virna_driver_instance = Ten_Minute_Data_From_Virna_Driver()
        ten_minute_data_from_virna_driver_instance.\
        _ten_minute_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_helper(file_label,
                                                                                             original_orography_filename,
                                                                                             original_ls_mask_filename,
                                                                                             tarasov_based_orog_correction=\
                                                                                             True,
                                                                                             glacial_mask=\
                                                                                             ice5g_glacial_mask_file)

    def ICE5G_21k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs_generation_and_upscaling(self):
        """Generate and upscale sinkless river directions for the LGM"""
        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"ice5g_v1_2_21_0k_10min.nc")
        present_day_base_orography_filename = path.join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        original_ls_mask_filename = path.join(self.ls_masks_path,
                                              "10min_ice6g_lsmask_with_disconnected_point_removed_21k.nc")
        ice5g_glacial_mask_file = path.join(self.orography_path,"ice5g_v1_2_21_0k_10min.nc")
        ten_minute_data_from_virna_driver_instance = Ten_Minute_Data_From_Virna_Driver()
        ten_minute_data_from_virna_driver_instance.\
        _ten_minute_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_helper(file_label,
                                                                                             original_orography_filename,
                                                                                             original_ls_mask_filename,
                                                                                             tarasov_based_orog_correction=\
                                                                                             True,
                                                                                             present_day_base_orography_filename=\
                                                                                             present_day_base_orography_filename,
                                                                                             glacial_mask=\
                                                                                             ice5g_glacial_mask_file)

class GLAC_Data_Drivers(ICE5G_Data_Drivers):
    """Driver runs on the GLAC orography data provided by Virna"""

    def GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_timeslice0(self):
        """Run sinkless river direction generation for timeslice zero"""
        self._GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs(timeslice=0)

    def GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_27timeslices_merge_timeslices_only(self):
        """Merge previously generated sinkless river directions for twenty seven evenly spaced slices

        To generate and then merge in a single step use the method below.
        """

        base_file_label="GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_27timeslices_20161128_170639"
        file_label = self._generate_file_label()
        combined_dataset_filename = self.generated_hd_file_path + "combined_" + file_label + '.nc'
        for i in range(260,-10,-10):
            print "Adding slice {0}".format(i)
            timeslice_hdfile_label = self.generated_hd_file_path + "timeslice{0}_".format(i) + base_file_label + '.nc'
            self._add_timeslice_to_combined_dataset(first_timeslice=(i==260),
                                                    slicetime=-26000 + i*100,
                                                    timeslice_hdfile_label=timeslice_hdfile_label,
                                                    combined_dataset_filename=combined_dataset_filename)

    def GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_27timeslices(self):
        """Generate and merge sinkless river directions for twenty seven evenly spaced slices"""
        base_file_label = self._generate_file_label()
        combined_dataset_filename = self.generated_hd_file_path + "combined_" + base_file_label + '.nc'
        combined_restart_filename= self.generated_hd_restart_file_path + "combined_" + base_file_label + '.nc'
        for i in range(260,-10,-10):
            print "Processing timeslice: {0}".format(i)
            self._GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs(timeslice=i,
                                                                                         base_file_label=base_file_label)
            timeslice_hdfile_label = self.generated_hd_file_path + "timeslice{0}_".format(i) + base_file_label + '.nc'
            timeslice_hdrestart_file_label = self.generated_hd_restart_file_path + "timeslice{0}_".format(i) +\
                base_file_label + '.nc'
            self._add_timeslice_to_combined_dataset(self,first_timeslice=(i==260),
                                                    slicetime=-26000 + i*100,
                                                    timeslice_hdfile_label=timeslice_hdfile_label,
                                                    combined_dataset_filename=combined_dataset_filename)
            self._add_timeslice_to_combined_dataset(self,first_timeslice=(i==260),
                                                    slicetime=-26000 + i*100,
                                                    timeslice_hdfile_label=timeslice_hdrestart_file_label,
                                                    combined_dataset_filename=combined_restart_filename)

    def _GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs(self,timeslice,base_file_label=None):
        orog_corrections_filename = path.join(self.orography_corrections_fields_path,
                                              "orog_corrs_field_ICE5G_data_ALG4_sink"
                                              "less_downscaled_ls_mask_0k_20160930_001057.nc")
        """Generate sinkless river direction and upscale them for a given timeslice.

        timeslice: integer; which timeslice to select
        base_file_label: string or None; if none then this will generate its own file label, if not none then
            use the given label as the base file label.

        Also attaches a timesliceX label to the file label for clarity as to which timeslice was processed.
        """

        if base_file_label is None:
            file_label = "timeslice{0}_".format(timeslice) + self._generate_file_label()
        else:
            file_label = "timeslice{0}_".format(timeslice) + base_file_label
        original_orography_filename = path.join(self.orography_path,"topo-final-OR-from-virna.nc")
        upscaled_orography_filename = self.upscaled_orography_filepath + file_label + '.nc'
        orography_filename = self.generated_orography_filepath + file_label + '.nc'
        HD_orography_filename = self.upscaled_orography_filepath + file_label + '_HD' + '.nc'
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        original_ls_mask_filename = path.join(self.ls_masks_path,"mask-final-OR-from-virna.nc")
        upscaled_ls_mask_filename = self.generated_ls_mask_filepath + 'upscaled_' +\
            file_label + '.nc'
        HD_ls_mask_filename = self.generated_ls_mask_filepath + file_label + '_HD' + '.nc'
        unsorted_catchments_filename = self.generated_catchments_path + 'unsorted_' +\
            file_label + '.nc'
        utilities.upscale_field_driver(input_filename=original_orography_filename,
                                       output_filename=upscaled_orography_filename,
                                       input_grid_type='LatLong5min',
                                       output_grid_type='LatLong10min',
                                       method='Sum', timeslice=timeslice,
                                       scalenumbers=True)
        utilities.upscale_field_driver(input_filename=original_ls_mask_filename,
                                       output_filename=upscaled_ls_mask_filename,
                                       input_grid_type='LatLong5min',
                                       output_grid_type='LatLong10min',
                                       method='Max', timeslice=timeslice,
                                       scalenumbers=False)
        utilities.change_dtype(input_filename=upscaled_ls_mask_filename,
                               output_filename=upscaled_ls_mask_filename,
                               new_dtype=np.int32,grid_type='LatLong10min')
        utilities.apply_orog_correction_field(original_orography_filename=upscaled_orography_filename,
                                              orography_corrections_filename=orog_corrections_filename,
                                              corrected_orography_filename=orography_filename,
                                              grid_type="LatLong10min")
        fill_sinks_driver.generate_sinkless_flow_directions(filename=orography_filename,
                                                            output_filename=rdirs_filename,
                                                            ls_mask_filename=\
                                                            upscaled_ls_mask_filename,
                                                            truesinks_filename=None,
                                                            catchment_nums_filename=\
                                                            unsorted_catchments_filename,
                                                            grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=rdirs_filename,
                                 output_file_label=file_label,
                                 ls_mask_filename=upscaled_ls_mask_filename,
                                 compute_catchments=False,
                                 grid_type='LatLong10min')
#         utilities.upscale_field_driver(input_filename=self.generated_flowmaps_filepath
#                                        + original_data_file_label + '.nc',
#                                        output_filename=self.upscaled_flowmaps_filepath
#                                        + upscaled_file_label + '.nc',
#                                        input_grid_type='LatLong10min',
#                                        output_grid_type='HD',
#                                        method='Max',
#                                        scalenumbers=True)
#         utilities.upscale_field_driver(input_filename=self.generated_rmouth_cumulative_flow_path
#                                        + original_data_file_label + '.nc',
#                                        output_filename=self.upscaled_rmouth_cumulative_flow_path
#                                        + upscaled_file_label + '.nc',
#                                        input_grid_type='LatLong10min',
#                                        output_grid_type='HD',
#                                        method='Sum',
#                                        scalenumbers=True)
#         utilities.upscale_field_driver(input_filename=self.generated_catchments_path
#                                        + "unsorted_"
#                                        + original_data_file_label + '.nc',
#                                        output_filename=self.upscaled_catchments_path
#                                        + "unsorted_"
#                                        + upscaled_file_label + '.nc',
#                                        input_grid_type='LatLong10min',
#                                        output_grid_type='HD',
#                                        method='Mode',
#                                        scalenumbers=False)
        fine_cumulative_flow = self.generated_flowmaps_filepath + file_label + '.nc'
        output_course_rdirs_filename = self.upscaled_generated_rdir_filepath + file_label + '.nc'
        cotat_plus_parameters_filename = path.join(self.cotat_plus_parameters_path,'cotat_plus_standard_params.nl')
        self._run_cotat_plus_upscaling(input_fine_rdirs_filename=rdirs_filename,
                                       input_fine_cumulative_flow_filename=fine_cumulative_flow,
                                       cotat_plus_parameters_filename=cotat_plus_parameters_filename,
                                       output_course_rdirs_filename=output_course_rdirs_filename,
                                       output_file_label=file_label,
                                       fine_grid_type='LatLong10min',
                                       course_grid_type='HD')
        upscaled_file_label = file_label + '_upscaled'
        self._run_postprocessing(rdirs_filename=output_course_rdirs_filename,
                                 output_file_label=upscaled_file_label,
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 compute_catchments=True, grid_type='HD')
        original_course_cumulative_flow_filename = self.generated_flowmaps_filepath + upscaled_file_label + '.nc'
        original_course_catchments_filename = self.generated_catchments_path + upscaled_file_label + '.nc'
        loops_nums_list_filename = self.generated_catchments_path + upscaled_file_label + '_loops.log'
        updated_file_label = upscaled_file_label + "_updated"
        updated_course_rdirs_filename = self.upscaled_generated_rdir_filepath + updated_file_label + '.nc'
        loop_breaker_driver.loop_breaker_driver(input_course_rdirs_filepath=output_course_rdirs_filename,
                                                input_course_cumulative_flow_filepath=\
                                                original_course_cumulative_flow_filename,
                                                input_course_catchments_filepath=\
                                                original_course_catchments_filename,
                                                input_fine_rdirs_filepath=\
                                                rdirs_filename,
                                                input_fine_cumulative_flow_filepath=\
                                                fine_cumulative_flow,
                                                output_updated_course_rdirs_filepath=\
                                                updated_course_rdirs_filename,
                                                loop_nums_list_filepath=\
                                                loops_nums_list_filename,
                                                course_grid_type='HD',
                                                fine_grid_type='LatLong10min')
        utilities.upscale_field_driver(input_filename=orography_filename,
                                       output_filename=HD_orography_filename,
                                       input_grid_type='LatLong10min',
                                       output_grid_type='HD',
                                       method='Sum', timeslice=None,
                                       scalenumbers=True)
        utilities.extract_ls_mask_from_rdirs(rdirs_filename=updated_course_rdirs_filename,
                                             lsmask_filename=HD_ls_mask_filename,
                                             grid_type='HD')
        transformed_course_rdirs_filename = path.splitext(updated_course_rdirs_filename)[0] + '_transf' +\
                                            path.splitext(updated_course_rdirs_filename)[1]
        transformed_HD_orography_filename = path.splitext(HD_orography_filename)[0] + '_transf' +\
                                          path.splitext(HD_orography_filename)[1]
        transformed_HD_ls_mask_filename = path.splitext(HD_ls_mask_filename)[0] + '_transf' +\
                                            path.splitext(HD_ls_mask_filename)[1]
        self._apply_transforms_to_field(input_filename=updated_course_rdirs_filename,
                                        output_filename=transformed_course_rdirs_filename,
                                        flip_ud=True, rotate180lr=True, invert_data=False,
                                        timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                        grid_type='HD')
        self._apply_transforms_to_field(input_filename=HD_orography_filename,
                                        output_filename=transformed_HD_orography_filename,
                                        flip_ud=True, rotate180lr=True, invert_data=False,
                                        timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                        grid_type='HD')
        self._apply_transforms_to_field(input_filename=HD_ls_mask_filename,
                                        output_filename=transformed_HD_ls_mask_filename,
                                        flip_ud=True, rotate180lr=True, invert_data=True,
                                        timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                        grid_type='HD')
        self._generate_flow_parameters(rdir_file=transformed_course_rdirs_filename,
                                       topography_file=transformed_HD_orography_filename,
                                       inner_slope_file=\
                                       path.join(self.orography_path,'bin_innerslope.dat'),
                                       lsmask_file=transformed_HD_ls_mask_filename,
                                       null_file=\
                                       path.join(self.null_fields_filepath,'null.dat'),
                                       area_spacing_file=\
                                       path.join(self.grid_areas_and_spacings_filepath,
                                                 'fl_dp_dl.dat'),
                                       orography_variance_file=\
                                       path.join(self.orography_path,'bin_toposig.dat'),
                                       output_dir=path.join(self.flow_params_dirs_path,
                                                            'hd_flow_params' + file_label))
        self._generate_hd_file(rdir_file=path.splitext(transformed_course_rdirs_filename)[0] + ".dat",
                               lsmask_file=path.splitext(transformed_HD_ls_mask_filename)[0] + ".dat",
                               null_file=\
                               path.join(self.null_fields_filepath,'null.dat'),
                               area_spacing_file=\
                               path.join(self.grid_areas_and_spacings_filepath,
                                         'fl_dp_dl.dat'),
                               hd_grid_specs_file=self.half_degree_grid_filepath,
                               output_file=self.generated_hd_file_path + file_label + '.nc',
                               paras_dir=path.join(self.flow_params_dirs_path,
                                                   'hd_flow_params' + file_label))
        utilities.prepare_hdrestart_file_driver(base_hdrestart_filename=self.base_hd_restart_file,
                                                output_hdrestart_filename=self.generated_hd_restart_file_path +
                                                    file_label + '.nc',
                                                hdparas_filename=self.generated_hd_file_path + file_label + '.nc',
                                                ref_hdparas_filename=self.ref_hd_paras_file,
                                                timeslice=None,
                                                res_num_data_rotate180lr=False,
                                                res_num_data_flipup=False,
                                                res_num_ref_rotate180lr=False,
                                                res_num_ref_flipud=False, grid_type='HD')
        self._run_postprocessing(rdirs_filename=updated_course_rdirs_filename,
                                 output_file_label=updated_file_label,
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 compute_catchments=True, grid_type='HD')


    def test_paragen_on_GLAC_data(self):
        """Test paragen code on GLAC data without having to rerun sinkless river direction generation and river direction upscaling"""
        file_label = self._generate_file_label() + "_test"
        transformed_course_rdirs_filename = "/Users/thomasriddick/Documents/data/HDdata/rdirs/generated/upscaled/upscaled_rdirs_timeslice0__GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20161124_141503_upscaled_updated_transf.dat"
        transformed_HD_orography_filename = "/Users/thomasriddick/Documents/data/HDdata/orographys/upscaled/upscaled_orog_timeslice0__GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20161124_143139_HD_transf.dat"
        transformed_HD_ls_mask_filename = "/Users/thomasriddick/Documents/data/HDdata/lsmasks/generated/ls_mask_timeslice0__GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20161124_141503_HD_transf.dat"
        self._generate_flow_parameters(rdir_file=transformed_course_rdirs_filename,
                                       topography_file=transformed_HD_orography_filename,
                                       inner_slope_file=\
                                       path.join(self.orography_path,'bin_innerslope.dat'),
                                       lsmask_file=transformed_HD_ls_mask_filename,
                                       null_file=\
                                       path.join(self.null_fields_filepath,'null.dat'),
                                       area_spacing_file=\
                                       path.join(self.grid_areas_and_spacings_filepath,
                                                 'fl_dp_dl.dat'),
                                       orography_variance_file=\
                                       path.join(self.orography_path,'bin_toposig.dat'),
                                       output_dir=path.join(self.flow_params_dirs_path,
                                                           'hd_flow_params' + file_label))
        self._generate_hd_file(rdir_file=path.splitext(transformed_course_rdirs_filename)[0] + ".dat",
                               lsmask_file=path.splitext(transformed_HD_ls_mask_filename)[0] + ".dat",
                               null_file=\
                               path.join(self.null_fields_filepath,'null.dat'),
                               area_spacing_file=\
                               path.join(self.grid_areas_and_spacings_filepath,
                                         'fl_dp_dl.dat'),
                               hd_grid_specs_file=self.half_degree_grid_filepath,
                               output_file=self.generated_hd_file_path + file_label + '.nc',
                               paras_dir=path.join(self.flow_params_dirs_path,
                                                   'hd_flow_params' + file_label))
        utilities.prepare_hdrestart_file_driver(base_hdrestart_filename=self.base_hd_restart_file,
                                                output_hdrestart_filename=self.generated_hd_restart_file_path +
                                                    file_label + '.nc',
                                                hdparas_filename=self.generated_hd_file_path + file_label + '.nc',
                                                ref_hdparas_filename=self.ref_hd_paras_file,
                                                timeslice=None,
                                                res_num_data_rotate180lr=False,
                                                res_num_data_flipup=False,
                                                res_num_ref_rotate180lr=False,
                                                res_num_ref_flipud=False, grid_type='HD')

class Ten_Minute_Data_From_Virna_Driver(ICE5G_Data_Drivers):
    """Drivers for the new 10 minute resolution data from Virna"""

    def ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs(self):
        """Generate and upscale sinkless river directions for the present day"""
        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"10min-topo-present-from-virna.nc")
        original_ls_mask_filename = path.join(self.ls_masks_path,"10min-mask-present-from-virna.nc")
        self._ten_minute_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_helper(file_label,
                                                                                                  original_orography_filename,
                                                                                                  original_ls_mask_filename)

    def ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs(self):
        """Generate and upscale sinkless river directions for the present day"""
        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"10min-topo-present-from-virna.nc")
        original_ls_mask_filename = path.join(self.ls_masks_path,"10min-mask-present-from-virna.nc")
        ice5g_glacial_mask_file = path.join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        self._ten_minute_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_helper(file_label,
                                                                                                  original_orography_filename,
                                                                                                  original_ls_mask_filename,
                                                                                                  tarasov_based_orog_correction=\
                                                                                                  True,
                                                                                                  glacial_mask=\
                                                                                                  ice5g_glacial_mask_file)

    def ten_minute_data_from_virna_0k_2017v_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs(self):
        """Generate and upscale sinkless river directions for the present day"""
        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"OR-topography-present_data_from_virna_2017.nc")
        original_ls_mask_filename = path.join(self.ls_masks_path,"OR-remapped-mask-present_data_from_virna_2017.nc")
        self._ten_minute_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_helper(file_label,
                                                                                                  original_orography_filename,
                                                                                                  original_ls_mask_filename)

    def ten_minute_data_from_virna_0k_13_04_2017v_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs(self):
        """Generate and upscale sinkless river directions for the present day"""
        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"OR-topography-present_data_from_virna_13_04_17.nc")
        original_ls_mask_filename = path.join(self.ls_masks_path,"OR-remapped-mask-present_data_from_virna_13_04_17.nc")
        self._ten_minute_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_helper(file_label,
                                                                                                  original_orography_filename,
                                                                                                  original_ls_mask_filename)

    def _ten_minute_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_helper(self,file_label,
                                                                                             original_orography_filename,
                                                                                             original_ls_mask_filename,
                                                                                             tarasov_based_orog_correction=\
                                                                                             False,
                                                                                             glacial_mask=None,
                                                                                             original_orography_fieldname=None,
                                                                                             present_day_base_orography_filename=None):
        """Helper for generating and upscaling sinkless river direction for a given 10 minute orography and landsea mask

        Arguments:
        file_label: string; file label to use
        original_orography_filename: string; full path to 10 minute orography to start from
        original_ls_mask_filename: string; full path to 10 minute landsea mask to start from
        present_day_base_orography_filename: string; full path to the present day orography
        the supplied orography is based upon
        Returns: nothing
        """

        if present_day_base_orography_filename:
            present_day_reference_orography_filename = path.join(self.orography_path,
                                                        "ice5g_v1_2_00_0k_10min.nc")
            original_orography_filename_before_base_change = original_orography_filename
            original_orography_filename =  self.generated_orography_filepath +\
                                           "rebased_original_" + file_label + '.nc'
        if tarasov_based_orog_correction:
            orog_corrections_filename = path.join(self.orography_corrections_fields_path,
                                                  "orog_corrs_field_ICE5G_and_tarasov_upscaled_"
                                                  "srtm30plus_north_america_only_data_ALG4_sinkless"
                                                  "_glcc_olson_lsmask_0k_20170517_003802.nc")
        else:
            orog_corrections_filename = path.join(self.orography_corrections_fields_path,
                                                  "orog_corrs_field_ICE5G_data_ALG4_sink"
                                                  "less_downscaled_ls_mask_0k_20160930_001057.nc")
        if glacial_mask is not None:
            intermediary_orography_filename = self.generated_orography_filepath +\
                                                "intermediary_" + file_label + '.nc'
        if present_day_base_orography_filename:
            utilities.rebase_orography_driver(orography_filename=\
                                              original_orography_filename_before_base_change,
                                              present_day_base_orography_filename=\
                                              present_day_base_orography_filename,
                                              present_day_reference_orography_filename=\
                                              present_day_reference_orography_filename,
                                              rebased_orography_filename=original_orography_filename,
                                              orography_fieldname=original_orography_fieldname,
                                              grid_type="LatLong10min")
            original_orography_fieldname="field_value"
        orography_filename = self.generated_orography_filepath + file_label + '.nc'
        HD_orography_filename = self.upscaled_orography_filepath + file_label + '_HD' + '.nc'
        HD_filled_orography_filename = self.upscaled_orography_filepath + file_label + '_HD_filled' + '.nc'
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        HD_ls_mask_filename = self.generated_ls_mask_filepath + file_label + '_HD' + '.nc'
        original_ls_mask_with_new_dtype_filename = self.generated_ls_mask_filepath + file_label + '_orig' + '.nc'
        unsorted_catchments_filename = self.generated_catchments_path + 'unsorted_' +\
            file_label + '.nc'
        utilities.change_dtype(input_filename=original_ls_mask_filename,
                               output_filename=original_ls_mask_with_new_dtype_filename,
                               new_dtype=np.int32,grid_type='LatLong10min')
        utilities.apply_orog_correction_field(original_orography_filename=original_orography_filename,
                                              orography_corrections_filename=orog_corrections_filename,
                                              corrected_orography_filename=
                                              orography_filename if glacial_mask is None else
                                              intermediary_orography_filename,
                                              original_orography_fieldname=\
                                              original_orography_fieldname,
                                              grid_type="LatLong10min")
        if glacial_mask is not None:
            utilities.\
            replace_corrected_orography_with_original_for_glaciated_grid_points_drivers(
                input_corrected_orography_file=intermediary_orography_filename,
                input_original_orography_file=original_orography_filename,
                input_glacier_mask_file=glacial_mask,
                out_orography_file=orography_filename,
                grid_type="LatLong10min")
        fill_sinks_driver.generate_sinkless_flow_directions(filename=orography_filename,
                                                            output_filename=rdirs_filename,
                                                            ls_mask_filename=\
                                                            original_ls_mask_with_new_dtype_filename,
                                                            truesinks_filename=None,
                                                            catchment_nums_filename=\
                                                            unsorted_catchments_filename,
                                                            grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=rdirs_filename,
                                 output_file_label=file_label,
                                 ls_mask_filename=original_ls_mask_with_new_dtype_filename,
                                 compute_catchments=False,
                                 flip_mask_ud=True,
                                 grid_type='LatLong10min')
#         utilities.upscale_field_driver(input_filename=self.generated_flowmaps_filepath
#                                        + original_data_file_label + '.nc',
#                                        output_filename=self.upscaled_flowmaps_filepath
#                                        + upscaled_file_label + '.nc',
#                                        input_grid_type='LatLong10min',
#                                        output_grid_type='HD',
#                                        method='Max',
#                                        scalenumbers=True)
#         utilities.upscale_field_driver(input_filename=self.generated_rmouth_cumulative_flow_path
#                                        + original_data_file_label + '.nc',
#                                        output_filename=self.upscaled_rmouth_cumulative_flow_path
#                                        + upscaled_file_label + '.nc',
#                                        input_grid_type='LatLong10min',
#                                        output_grid_type='HD',
#                                        method='Sum',
#                                        scalenumbers=True)
#         utilities.upscale_field_driver(input_filename=self.generated_catchments_path
#                                        + "unsorted_"
#                                        + original_data_file_label + '.nc',
#                                        output_filename=self.upscaled_catchments_path
#                                        + "unsorted_"
#                                        + upscaled_file_label + '.nc',
#                                        input_grid_type='LatLong10min',
#                                        output_grid_type='HD',
#                                        method='Mode',
#                                        scalenumbers=False)
        fine_rdirs_with_outflows_marked = self.generated_rdir_with_outflows_marked_filepath + file_label + '.nc'
        fine_cumulative_flow = self.generated_flowmaps_filepath + file_label + '.nc'
        output_course_rdirs_filename = self.upscaled_generated_rdir_filepath + file_label + '.nc'
        cotat_plus_parameters_filename = path.join(self.cotat_plus_parameters_path,'cotat_plus_standard_params.nl')
        self._run_cotat_plus_upscaling(input_fine_rdirs_filename=fine_rdirs_with_outflows_marked,
                                       input_fine_cumulative_flow_filename=fine_cumulative_flow,
                                       cotat_plus_parameters_filename=cotat_plus_parameters_filename,
                                       output_course_rdirs_filename=output_course_rdirs_filename,
                                       output_file_label=file_label,
                                       fine_grid_type='LatLong10min',
                                       course_grid_type='HD')
        upscaled_file_label = file_label + '_upscaled'
        self._run_postprocessing(rdirs_filename=output_course_rdirs_filename,
                                 output_file_label=upscaled_file_label,
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 compute_catchments=True, grid_type='HD')
        original_course_cumulative_flow_filename = self.generated_flowmaps_filepath + upscaled_file_label + '.nc'
        original_course_catchments_filename = self.generated_catchments_path + upscaled_file_label + '.nc'
        loops_nums_list_filename = self.generated_catchments_path + upscaled_file_label + '_loops.log'
        updated_file_label = upscaled_file_label + "_updated"
        updated_course_rdirs_filename = self.upscaled_generated_rdir_filepath + updated_file_label + '.nc'
        loop_breaker_driver.loop_breaker_driver(input_course_rdirs_filepath=output_course_rdirs_filename,
                                                input_course_cumulative_flow_filepath=\
                                                original_course_cumulative_flow_filename,
                                                input_course_catchments_filepath=\
                                                original_course_catchments_filename,
                                                input_fine_rdirs_filepath=\
                                                fine_rdirs_with_outflows_marked,
                                                input_fine_cumulative_flow_filepath=\
                                                fine_cumulative_flow,
                                                output_updated_course_rdirs_filepath=\
                                                updated_course_rdirs_filename,
                                                loop_nums_list_filepath=\
                                                loops_nums_list_filename,
                                                course_grid_type='HD',
                                                fine_grid_type='LatLong10min')
        utilities.upscale_field_driver(input_filename=orography_filename,
                                       output_filename=HD_orography_filename,
                                       input_grid_type='LatLong10min',
                                       output_grid_type='HD',
                                       method='Sum', timeslice=None,
                                       scalenumbers=True)
        utilities.extract_ls_mask_from_rdirs(rdirs_filename=updated_course_rdirs_filename,
                                             lsmask_filename=HD_ls_mask_filename,
                                             grid_type='HD')
        fill_sinks_driver.generate_orography_with_sinks_filled(HD_orography_filename,
                                                               HD_filled_orography_filename,
                                                               ls_mask_filename=HD_ls_mask_filename,
                                                               truesinks_filename=None,
                                                               flip_ud=False,
                                                               flip_lsmask_ud=True,
                                                               grid_type='HD',
                                                               add_slight_slope_when_filling_sinks=False,
                                                               slope_param=0.0)
        transformed_course_rdirs_filename = path.splitext(updated_course_rdirs_filename)[0] + '_transf' +\
                                            path.splitext(updated_course_rdirs_filename)[1]
        transformed_HD_filled_orography_filename = path.splitext(HD_filled_orography_filename)[0] + '_transf' +\
                                          path.splitext(HD_filled_orography_filename)[1]
        transformed_HD_ls_mask_filename = path.splitext(HD_ls_mask_filename)[0] + '_transf' +\
                                            path.splitext(HD_ls_mask_filename)[1]
        self._apply_transforms_to_field(input_filename=updated_course_rdirs_filename,
                                        output_filename=transformed_course_rdirs_filename,
                                        flip_ud=False, rotate180lr=True, invert_data=False,
                                        timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                        grid_type='HD')
        self._apply_transforms_to_field(input_filename=HD_filled_orography_filename,
                                        output_filename=transformed_HD_filled_orography_filename,
                                        flip_ud=True, rotate180lr=True, invert_data=False,
                                        timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                        grid_type='HD')
        self._apply_transforms_to_field(input_filename=HD_ls_mask_filename,
                                        output_filename=transformed_HD_ls_mask_filename,
                                        flip_ud=False, rotate180lr=True, invert_data=True,
                                        timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                        grid_type='HD')
        self._generate_flow_parameters(rdir_file=transformed_course_rdirs_filename,
                                       topography_file=transformed_HD_filled_orography_filename,
                                       inner_slope_file=\
                                       path.join(self.orography_path,'bin_innerslope.dat'),
                                       lsmask_file=transformed_HD_ls_mask_filename,
                                       null_file=\
                                       path.join(self.null_fields_filepath,'null.dat'),
                                       area_spacing_file=\
                                       path.join(self.grid_areas_and_spacings_filepath,
                                                 'fl_dp_dl.dat'),
                                       orography_variance_file=\
                                       path.join(self.orography_path,'bin_toposig.dat'),
                                       output_dir=path.join(self.flow_params_dirs_path,
                                                            'hd_flow_params' + file_label))
        self._generate_hd_file(rdir_file=path.splitext(transformed_course_rdirs_filename)[0] + ".dat",
                               lsmask_file=path.splitext(transformed_HD_ls_mask_filename)[0] + ".dat",
                               null_file=\
                               path.join(self.null_fields_filepath,'null.dat'),
                               area_spacing_file=\
                               path.join(self.grid_areas_and_spacings_filepath,
                                         'fl_dp_dl.dat'),
                               hd_grid_specs_file=self.half_degree_grid_filepath,
                               output_file=self.generated_hd_file_path + file_label + '.nc',
                               paras_dir=path.join(self.flow_params_dirs_path,
                                                   'hd_flow_params' + file_label))
        utilities.prepare_hdrestart_file_driver(base_hdrestart_filename=self.base_hd_restart_file,
                                                output_hdrestart_filename=self.generated_hd_restart_file_path +
                                                    file_label + '.nc',
                                                hdparas_filename=self.generated_hd_file_path + file_label + '.nc',
                                                ref_hdparas_filename=self.ref_hd_paras_file,
                                                timeslice=None,
                                                res_num_data_rotate180lr=False,
                                                res_num_data_flipup=False,
                                                res_num_ref_rotate180lr=False,
                                                res_num_ref_flipud=False, grid_type='HD')
        utilities.generate_gaussian_landsea_mask(input_lsmask_filename=transformed_HD_ls_mask_filename,
                                                 output_gaussian_latlon_mask_filename=\
                                                 self.generated_gaussian_ls_mask_filepath +'80_' + file_label +
                                                 '.nc',
                                                 gaussian_grid_spacing=80)
        utilities.insert_new_landsea_mask_into_jsbach_restart_file(input_landsea_mask_filename=\
                                                                   self.generated_gaussian_ls_mask_filepath +
                                                                   '80_' + file_label + '.nc',
                                                                   input_js_bach_filename=\
                                                                   self.base_js_bach_restart_file_T106,
                                                                   output_modified_js_bach_filename=\
                                                                   self.generated_js_bach_restart_filepath +
                                                                   "jsbach_T106_11tiles_5layers_1976_"
                                                                   + file_label + '.nc',
                                                                   modify_fractional_lsm=True,
                                                                   modify_lake_mask=True)
        utilities.generate_gaussian_landsea_mask(input_lsmask_filename=transformed_HD_ls_mask_filename,
                                                 output_gaussian_latlon_mask_filename=\
                                                 self.generated_gaussian_ls_mask_filepath + '48_' + file_label +
                                                 '.nc',
                                                 gaussian_grid_spacing=48)
        utilities.insert_new_landsea_mask_into_jsbach_restart_file(input_landsea_mask_filename=\
                                                                   self.generated_gaussian_ls_mask_filepath +
                                                                   "48_" + file_label + '.nc',
                                                                   input_js_bach_filename=\
                                                                   self.base_js_bach_restart_file_T63,
                                                                   output_modified_js_bach_filename=\
                                                                   self.generated_js_bach_restart_filepath +
                                                                   "jsbach_T63_11tiles_5layers_1976_"
                                                                   + file_label + '.nc',
                                                                   modify_fractional_lsm=True,
                                                                   modify_lake_mask=True)

        self._run_postprocessing(rdirs_filename=updated_course_rdirs_filename,
                                 output_file_label=updated_file_label,
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 compute_catchments=True, grid_type='HD')

    def ten_minute_data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs(self):
        """Generate and upscale sinkless river directions for the LGM"""
        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"10min-topo-lgm-from-virna.nc")
        original_ls_mask_filename = path.join(self.ls_masks_path,"10min-mask-lgm-from-virna.nc")
        self._ten_minute_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_helper(file_label,
                                                                                                  original_orography_filename,
                                                                                                  original_ls_mask_filename)

    def ten_minute_data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs(self):
        """Generate and upscale sinkless river directions for the LGM"""
        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"10min-topo-lgm-from-virna.nc")
        original_ls_mask_filename = path.join(self.ls_masks_path,"10min-mask-lgm-from-virna.nc")
        ice5g_glacial_mask_file = path.join(self.orography_path,"ice5g_v1_2_21_0k_10min.nc")
        self._ten_minute_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_helper(file_label,
                                                                                                  original_orography_filename,
                                                                                                  original_ls_mask_filename,
                                                                                                  tarasov_based_orog_correction=\
                                                                                                  True,
                                                                                                  glacial_mask=\
                                                                                                  ice5g_glacial_mask_file)


class ICE6g_Data_Drivers(Ten_Minute_Data_From_Virna_Driver):

    def ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs(self):
        """Generate and upscale sinkless river directions for the LGM"""
        file_label = self._generate_file_label()
        ice6g_orography_lgm_filename = path.join(self.orography_path,"Ice6g_c_VM5a_10min_21k.nc")
        ice6g_ls_mask_lgm_filename = path.join(self.ls_masks_path,
                                               "10min_ice6g_lsmask_with_disconnected_point_removed_21k.nc")
        ice6g_glacial_mask_file = path.join(self.orography_path,"Ice6g_c_VM5a_10min_21k.nc")
        ice6g_orography_0k_filename = path.join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        self._ten_minute_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_helper(file_label,
                                                                                                  ice6g_orography_lgm_filename,
                                                                                                  ice6g_ls_mask_lgm_filename,
                                                                                                  tarasov_based_orog_correction=\
                                                                                                  True,
                                                                                                  glacial_mask=\
                                                                                                  ice6g_glacial_mask_file,
                                                                                                  original_orography_fieldname=\
                                                                                                  'Topo',
                                                                                                  present_day_base_orography_filename=\
                                                                                                  ice6g_orography_0k_filename)

    def ICE6g_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs(self):
        """Generate and upscale sinkless river directions for the present day"""
        file_label = self._generate_file_label()
        ice6g_orography_0k_filename = path.join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        ice6g_ls_mask_0k_filename = path.join(self.ls_masks_path,
                                              "10min_ice6g_lsmask_with_disconnected_point_removed_0k.nc")
        ice6g_glacial_mask_file = path.join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        self._ten_minute_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_helper(file_label,
                                                                                                  ice6g_orography_0k_filename,
                                                                                                  ice6g_ls_mask_0k_filename,
                                                                                                  tarasov_based_orog_correction=\
                                                                                                  True,
                                                                                                  glacial_mask=\
                                                                                                  ice6g_glacial_mask_file,
                                                                                                  original_orography_fieldname=\
                                                                                                  'Topo',
                                                                                                  present_day_base_orography_filename=\
                                                                                                  ice6g_orography_0k_filename)

#     def ICE6g_lgm_ALG4_sinkless_no_true_sinks_jsbach_lsmask_plus_upscale_rdirs_tarasov_orog_corrs(self):
#         """Generate and upscale sinkless river directions for the LGM"""
#         file_label = self._generate_file_label()
#         ice6g_orography_lgm_filename = path.join(self.orography_path,"Ice6g_c_VM5a_10min_21k.nc")
#         ice6g_ls_mask_lgm_filename = path.join(self.ls_masks_path,
#                                                "10min_ice6g_lsmask_with_disconnected_point_removed_21k.nc")
#         ice6g_glacial_mask_file = path.join(self.orography_path,"Ice6g_c_VM5a_10min_21k.nc")
#         ice6g_orography_0k_filename = path.join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
#         self._ten_minute_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_helper(file_label,
#                                                                                                   ice6g_orography_lgm_filename,
#                                                                                                   ice6g_ls_mask_lgm_filename,
#                                                                                                   tarasov_based_orog_correction=\
#                                                                                                   True,
#                                                                                                   glacial_mask=\
#                                                                                                   ice6g_glacial_mask_file,
#                                                                                                   original_orography_fieldname=\
#                                                                                                   'Topo',
#                                                                                                   present_day_base_orography_filename=\
#                                                                                                   ice6g_orography_0k_filename)

    def ICE6g_0k_ALG4_sinkless_no_true_sinks_jsbach_lsmask_plus_upscale_rdirs_tarasov_orog_corrs(self):
        """Generate and upscale sinkless river directions for the present day"""
        file_label = self._generate_file_label()
        ice6g_orography_0k_filename = path.join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        ice6g_ls_mask_0k_filename = path.join(self.ls_masks_path,"generated",
                                              "ls_mask_create_10min_present_day_lsmask_from_model"
                                              "_gaussian_mask_20170620_211713.nc")
        ice6g_glacial_mask_file = path.join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        self._ten_minute_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_helper(file_label,
                                                                                                  ice6g_orography_0k_filename,
                                                                                                  ice6g_ls_mask_0k_filename,
                                                                                                  tarasov_based_orog_correction=\
                                                                                                  True,
                                                                                                  glacial_mask=\
                                                                                                  ice6g_glacial_mask_file,
                                                                                                  original_orography_fieldname=\
                                                                                                  'Topo',
                                                                                                  present_day_base_orography_filename=\
                                                                                                  ice6g_orography_0k_filename)

    def ICE6g_0k_ALG4_sinkless_no_true_sinks_mpiom_lsmask_plus_upscale_rdirs_tarasov_orog_corrs(self):
        """Generate and upscale sinkless river directions for the present day"""
        file_label = self._generate_file_label()
        ice6g_orography_0k_filename = path.join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        ice6g_ls_mask_0k_filename = path.join(self.ls_masks_path,"generated",
                                              "ls_mask_create_10min_present_day_lsmask_from_model"
                                              "_ocean_mask_20170621_130700.nc")
        ice6g_glacial_mask_file = path.join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        self._ten_minute_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_helper(file_label,
                                                                                                  ice6g_orography_0k_filename,
                                                                                                  ice6g_ls_mask_0k_filename,
                                                                                                  tarasov_based_orog_correction=\
                                                                                                  True,
                                                                                                  glacial_mask=\
                                                                                                  ice6g_glacial_mask_file,
                                                                                                  original_orography_fieldname=\
                                                                                                  'Topo',
                                                                                                  present_day_base_orography_filename=\
                                                                                                  ice6g_orography_0k_filename)

class ETOPO2v2DataDrivers(Ten_Minute_Data_From_Virna_Driver):

    def ETOPO2v2_upscaled_to_10min_grid(self):
        """Generate and upscale sinkless river directions for the present day"""
        file_label = self._generate_file_label()
        etopo2v2_orography_0k_filename = path.join(self.orography_path,"generated",
                                                "updated_orog_upscale_ETOPO2v2_to_10minute_grid_20170608_183659.nc")
        etopo2v2_orography_0k_rotated_filename = self.generated_orography_filepath + "rotated_" + file_label + ".nc"
        self._apply_transforms_to_field(input_filename=etopo2v2_orography_0k_filename,
                                        output_filename=etopo2v2_orography_0k_rotated_filename,
                                        flip_ud=True, rotate180lr=True, invert_data=False,
                                        grid_type="LatLong10min")
        ice6g_ls_mask_0k_filename = path.join(self.ls_masks_path,
                                              "10min_ice6g_lsmask_with_disconnected_point_removed_0k.nc")
        ice6g_glacial_mask_file = path.join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        self._ten_minute_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_helper(file_label,
                                                                                                  etopo2v2_orography_0k_rotated_filename,
                                                                                                  ice6g_ls_mask_0k_filename,
                                                                                                  tarasov_based_orog_correction=\
                                                                                                  True,
                                                                                                  glacial_mask=\
                                                                                                  ice6g_glacial_mask_file,
                                                                                                  original_orography_fieldname=\
                                                                                                  'field_value')

def main():
    """Select the revelant runs to make

    Select runs by uncommenting them and also the revelant object instantation.
    """

    #ice5g_data_drivers = ICE5G_Data_Drivers()
    #ice5g_data_drivers.ICE5G_as_HD_data_21k_0k_sig_grad_only_all_neighbours_driver()
    #ice5g_data_drivers.ICE5G_as_HD_data_all_points_21k()
    #ice5g_data_drivers.ICE5G_as_HD_data_all_points_0k()
    #ice5g_data_drivers.ICE5G_as_HD_data_ALG4_sinkless_all_points_0k()
    #ice5g_data_drivers.ICE5G_data_all_points_0k()
    #ice5g_data_drivers.ICE5G_data_all_points_21k()
    #ice5g_data_drivers.ICE5G_data_ALG4_sinkless_0k()
    #ice5g_data_drivers.ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k()
    #ice5g_data_drivers.ICE5G_data_ALG4_sinkless_no_true_sinks_0k()
    #ice5g_data_drivers.ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs()
    #ice5g_data_drivers.ICE5G_data_ALG4_sinkless_21k()
    #ice5g_data_drivers.ICE_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_from_orog_corrs_field()
    #ice5g_data_drivers.ICE5G_and_tarasov_upscaled_srtm30plus_data_ALG4_sinkless_downscaled_ls_mask_0k()
    #ice5g_data_drivers.ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_downscaled_ls_mask_0k()
    #ice5g_data_drivers.ICE5G_and_tarasov_upscaled_srtm30plus_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs()
    #ice5g_data_drivers.ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_downscaled_ls_mask_0k_upscale_rdirs()
    #ice5g_data_drivers.ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson_lsmask_0k()
    #ice5g_data_drivers.ICE5G_and_tarasov_upscaled_srtm30plus_north_america_only_data_ALG4_sinkless_glcc_olson_lsmask_0k_upscale_rdirs()
    #ice5g_data_drivers.\
    #ICE5G_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs_generation_and_upscaling()
    #ice5g_data_drivers.\
    #ICE5G_21k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs_generation_and_upscaling()
    #etopo1_data_drivers = ETOPO1_Data_Drivers()
    #etopo1_data_drivers.etopo1_data_all_points()
    #etopo1_data_drivers.etopo1_data_ALG4_sinkless()
    utilities_drivers = Utilities_Drivers()
    #utilities_drivers.convert_corrected_HD_hydrology_dat_files_to_nc()
    #utilities_drivers.recreate_connected_HD_lsmask()
    #utilities_drivers.recreate_connected_HD_lsmask_true_seas_inc_casp_only()
    #utilities_drivers.downscale_HD_ls_seed_points_to_1min_lat_lon()
    #utilities_drivers.downscale_HD_ls_seed_points_to_10min_lat_lon()
    #utilities_drivers.downscale_HD_ls_seed_points_to_10min_lat_lon_true_seas_inc_casp_only()
    #utilities_drivers.recreate_connected_lsmask_for_black_azov_and_caspian_seas_from_glcc_olson_data()
    #utilities_drivers.recreate_connected_HD_lsmask_from_glcc_olson_data()
    #utilities_drivers.recreate_connected_10min_lsmask_from_glcc_olson_data()
    #utilities_drivers.upscale_srtm30_plus_orog_to_10min()
    #utilities_drivers.upscale_srtm30_plus_orog_to_10min_no_lsmask()
    #utilities_drivers.upscale_srtm30_plus_orog_to_10min_no_lsmask_tarasov_style_params()
    #utilities_drivers.upscale_srtm30_plus_orog_to_10min_no_lsmask_reduced_back_looping()
    #utilities_drivers.upscale_1min_orography_to_30min()
    #utilities_drivers.upscale_srtm30_plus_orog_to_10min_no_lsmask_half_cell_upscaling_params()
    #utilities_drivers.downscale_ICE6G_21k_landsea_mask_and_remove_disconnected_points()
    #utilities_drivers.remove_disconnected_points_from_ICE6G_21k_landsea_mask_and_add_caspian()
    #utilities_drivers.remove_disconnected_points_from_ICE6G_0k_landsea_mask_and_add_caspian()
    #utilities_drivers.upscale_ETOPO2v2_to_10minute_grid()
    #utilities_drivers.create_10min_present_day_lsmask_from_model_gaussian_mask()
    #utilities_drivers.create_10min_present_day_lsmask_from_model_ocean_mask()
    #utilities_drivers.create_catchments_from_hdpara_file_from_swati()
    utilities_drivers.generate_rdirs_from_srtm30_plus()
    #utilities_drivers.generate_rdirs_from_ice5g_21k()
    #original_hd_model_rfd_drivers = Original_HD_Model_RFD_Drivers()
    #original_hd_model_rfd_drivers.corrected_HD_rdirs_post_processing()
    #original_hd_model_rfd_drivers.extract_ls_mask_from_corrected_HD_rdirs()
    #original_hd_model_rfd_drivers.extract_true_sinks_from_corrected_HD_rdirs()
    #original_hd_model_rfd_drivers.regenerate_hd_file_without_lakes_and_wetlands()
    #original_hd_model_rfd_drivers.extract_current_HD_rdirs_from_hdparas_file()
    #glac_data_drivers = GLAC_Data_Drivers()
    #glac_data_drivers.GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_timeslice0()
    #glac_data_drivers.test_paragen_on_GLAC_data()
    #glac_data_drivers.GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_27timeslices()
    #glac_data_drivers.GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_27timeslices_merge_timeslices_only()
    #ten_minute_data_from_virna_driver = Ten_Minute_Data_From_Virna_Driver()
    #ten_minute_data_from_virna_driver.ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs()
    #ten_minute_data_from_virna_driver.ten_minute_data_from_virna_0k_2017v_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs()
    #ten_minute_data_from_virna_driver.ten_minute_data_from_virna_0k_13_04_2017v_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs()
    #ten_minute_data_from_virna_driver.ten_minute_data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_2017_data()
    #ten_minute_data_from_virna_driver.ten_minute_data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs()
    #ten_minute_data_from_virna_driver.ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs()
    #ten_minute_data_from_virna_driver.ten_minute_data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs()
    #ice6g_data_drivers = ICE6g_Data_Drivers()
    #ice6g_data_drivers.ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs()
    #ice6g_data_drivers.ICE6g_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs()
    #ice6g_data_drivers.ICE6g_0k_ALG4_sinkless_no_true_sinks_jsbach_lsmask_plus_upscale_rdirs_tarasov_orog_corrs()
    #ice6g_data_drivers.ICE6g_0k_ALG4_sinkless_no_true_sinks_mpiom_lsmask_plus_upscale_rdirs_tarasov_orog_corrs()
    #etopo2v2_data_drivers = ETOPO2v2DataDrivers()
    #etopo2v2_data_drivers.ETOPO2v2_upscaled_to_10min_grid()

if __name__ == '__main__':
    main()
