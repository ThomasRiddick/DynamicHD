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
import utilities
import river_mouth_marking_driver
import create_connected_lsmask_driver as cc_lsmask_driver
import shutil
import cotat_plus_driver
import loop_breaker_driver
import numpy as np
import iohelper
import netCDF4

class Dynamic_HD_Drivers(object):
    
    bash_scripts_path = "/Users/thomasriddick/Documents/workspace/Dynamic_HD_bash_scripts"
   
    def __init__(self):
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
        cotat_plus_parameters_path_extension = path.join(parameter_path_extension,'cotat_plus')
        self.base_RFD_filepath = path.join(data_dir,rdirs_path_extension,
                                           base_RFD_filename)
        self.orography_path = path.join(data_dir,orog_path_extension)
        self.upscaled_orography_filepath = path.join(self.orography_path,'upscaled','upscaled_orog_')
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
        self.hd_grid_filepath = path.join(self.grids_path,"hdmodel2d_griddes")
        self.half_degree_grid_filepath = path.join(self.grids_path,"grid_0_5.txt")
        self.hd_grid_ls_mask_filepath = path.join(self.ls_masks_path,
                                                  "lsmmaskvonGR30.srv")
        self.hd_truesinks_filepath = path.join(self.truesinks_path,
                                               "truesinks_extract_true_sinks_from_"
                                               "corrected_HD_rdirs_20160527_105218.nc")
        self.base_hd_restart_file = path.join(self.hd_restart_file_path,"hd_restart_file_from_current_model.nc")
        self.ref_hd_paras_file = path.join(self.hd_file_path,"hdpara_file_from_current_model.nc")
        
    @staticmethod 
    def _generate_file_label():
        return "_".join([str(inspect.stack()[1][3]),datetime.datetime.now().strftime("%Y%m%d_%H%M%S")])

    def _prepare_topography(self,orog_nc_file,grid_file,weights_file,output_file,lsmask_file):
        try:
            print subprocess.check_output([path.join(self.bash_scripts_path,
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
                          hd_grid_specs_file,output_file,paras_dir):
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
            print subprocess.check_output([path.join(self.bash_scripts_path,
                                                     "generate_output_file.sh"),
                                           path.join(self.bash_scripts_path,
                                                     "bin"),
                                           path.join(self.bash_scripts_path,
                                                     "fortran"),
                                           path.splitext(rdir_file)[0] + ".dat",
                                           path.splitext(lsmask_file)[0] + ".dat",
                                           path.splitext(null_file)[0] + ".dat",
                                           path.splitext(area_spacing_file)[0] + ".dat",
                                           hd_grid_specs_file,output_file,paras_dir])
        except CalledProcessError as cperror:
            raise RuntimeError("Failure in called process {0}; return code {1}; output:\n{2}".format(cperror.cmd,
                                                                                                     cperror.returncode,
                                                                                                     cperror.output))

    def _generate_flow_parameters(self,rdir_file,topography_file,inner_slope_file,lsmask_file,
                                  null_file,area_spacing_file,output_dir):
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
        try:
            print subprocess.check_output([path.join(self.bash_scripts_path,
                                                     "parameter_generation_driver.sh"),
                                           path.join(self.bash_scripts_path,
                                                     "bin"),
                                           path.join(self.bash_scripts_path,
                                                     "fortran"),
                                           path.splitext(rdir_file)[0] + ".dat",
                                           path.splitext(topography_file)[0] + ".dat",
                                           path.splitext(inner_slope_file)[0] + ".dat",
                                           path.splitext(lsmask_file)[0] + ".dat",
                                           path.splitext(null_file)[0] + ".dat",
                                           path.splitext(area_spacing_file)[0] + ".dat",
                                           output_dir],
                                          stderr=subprocess.STDOUT)
        except CalledProcessError as cperror:
            raise RuntimeError("Failure in called process {0}; return code {1}; output:\n{2}".format(cperror.cmd,
                                                                                                     cperror.returncode,
                                                                                                     cperror.output))

    def _run_postprocessing(self,rdirs_filename,output_file_label,ls_mask_filename = None,
                            skip_marking_mouths=False,compute_catchments=True,
                            grid_type='HD',**grid_kwargs):
        self._run_flow_to_grid_cell(rdirs_filename,output_file_label,grid_type,**grid_kwargs)
        if compute_catchments:
            self._run_compute_catchments(rdirs_filename, output_file_label, 
                                         grid_type,**grid_kwargs)
        self._run_river_mouth_marking(rdirs_filename, output_file_label, ls_mask_filename, 
                                      flowtocell_filename=self.generated_flowmaps_filepath
                                      + output_file_label + '.nc',
                                      skip_marking_mouths=skip_marking_mouths, 
                                      grid_type=grid_type,**grid_kwargs)
       
    def _run_compute_catchments(self,rdirs_filename,output_file_label,grid_type,**grid_kwargs):
        compute_catchments.main(filename=rdirs_filename,
                                output_filename=self.generated_catchments_path +
                                output_file_label + '.nc',
                                loop_logfile=self.generated_catchments_path +
                                output_file_label + '_loops.log',
                                grid_type=grid_type,**grid_kwargs)
        
    def _run_flow_to_grid_cell(self,rdirs_filename,output_file_label,grid_type,**grid_kwargs):
        flow_to_grid_cell.main(rdirs_filename=rdirs_filename, 
                               output_filename=self.generated_flowmaps_filepath
                               + output_file_label + '.nc', 
                               grid_type=grid_type,**grid_kwargs)
    
    def _run_river_mouth_marking(self,rdirs_filename,output_file_label,ls_mask_filename,
                                 flowtocell_filename,skip_marking_mouths,grid_type,
                                 **grid_kwargs):
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
                                        grid_type=grid_type,**grid_kwargs)
        
    def _convert_data_file_type(self,filename,new_file_type,grid_type,**grid_kwargs):
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
        
    def _run_cotat_plus_upscaling(self,input_fine_rdirs_filename,input_fine_cumulative_flow_filename,
                                  cotat_plus_parameters_filename,output_course_rdirs_filename,
                                  output_file_label,fine_grid_type,fine_grid_kwargs={},
                                  course_grid_type='HD',**course_grid_kwargs):
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
        
    def _apply_transforms_to_field(self,input_filename,output_filename,flip_up=False,
                                   rotate180lr=False,invert_data=False,
                                   timeslice=None,grid_type='HD',**grid_kwargs):
        if (not flip_up) and (not rotate180lr) and (not invert_data):
            raise UserWarning("No transform specified")
        field = dynamic_hd.load_field(input_filename,
                                      file_type=dynamic_hd.get_file_extension(input_filename),
                                      field_type='Generic',unmask=False,timeslice=timeslice,
                                      grid_type=grid_type,**grid_kwargs)
        if flip_up:
            field.flip_data_ud()
        if rotate180lr:
            field.rotate_field_by_a_hundred_and_eighty_degrees()
        if invert_data:
            field.invert_data()
        dynamic_hd.write_field(output_filename,
                               field=field,
                               file_type=dynamic_hd.get_file_extension(output_filename))
        
        
    def _add_timeslice_to_combined_dataset(self,first_timeslice,slicetime,
                                           timeslice_hdfile_label,combined_dataset_filename):
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
    
    def convert_corrected_HD_hydrology_dat_files_to_nc(self):
        corrected_RFD_filepath = path.join(self.rdir_path,'rivdir_vs_1_9_data_from_stefan.dat')
        corrected_orography_filepath = path.join(self.orography_path,'topo_hd_vs1_9_data_from_stefan.dat')
        for filename in [corrected_RFD_filepath,corrected_orography_filepath]:
            self._convert_data_file_type(filename, new_file_type='.nc', grid_type='HD')
            
    def recreate_connected_HD_lsmask(self):
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
        
    def recreate_connected_HD_lsmask_true_seas_inc_casp_only(self):
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
        
    def downscale_HD_ls_seed_points_to_1min_lat_lon(self):
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
        
class Original_HD_Model_RFD_Drivers(Dynamic_HD_Drivers):
   
    def __init__(self):
        super(Original_HD_Model_RFD_Drivers,self).__init__()
        self.corrected_RFD_filepath = path.join(self.rdir_path,"rivdir_vs_1_9_data_from_stefan.nc")
    
    def corrected_HD_rdirs_post_processing(self):
        file_label = self._generate_file_label()
        self._run_postprocessing(self.corrected_RFD_filepath, 
                                 output_file_label=file_label, 
                                 ls_mask_filename=None,
                                 skip_marking_mouths=True,
                                 grid_type='HD')
        
    def extract_ls_mask_from_corrected_HD_rdirs(self):
        file_label = self._generate_file_label()
        utilities.extract_ls_mask_from_rdirs(rdirs_filename=self.corrected_RFD_filepath, 
                                             lsmask_filename=self.generated_ls_mask_filepath +\
                                             file_label + '.nc', 
                                             grid_type='HD')
        
    def extract_true_sinks_from_corrected_HD_rdirs(self):
        file_label = self._generate_file_label()
        utilities.extract_true_sinks_from_rdirs(rdirs_filename=self.corrected_RFD_filepath,
                                                 truesinks_filename=self.generated_truesinks_path +\
                                                 file_label + '.nc',
                                                 grid_type='HD')
            
class ETOPO1_Data_Drivers(Dynamic_HD_Drivers):
   
    def __init__(self):
        super(ETOPO1_Data_Drivers,self).__init__()
        self.etopo1_data_filepath = path.join(self.orography_path,'ETOPO1_Ice_c_gmt4.nc')
    
    def etopo1_data_all_points(self):
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

    def __init__(self):
        super(ICE5G_Data_Drivers,self).__init__()
        self.remap_10min_to_HD_grid_weights_filepath = path.join(self.weights_path,
                                                                "weights10mintoHDgrid.nc")
        self.ice5g_orography_corrections_master_filepath = path.join(self.orography_corrections_path, 
                                                                     'ice5g_10min_orog_corrs_master.txt')
        self.ice5g_intelligent_burning_regions_list_master_filepath = path.\
            join(self.intelligent_burning_regions_path,'ice5g_10min_int_burning_regions_master.txt')
        self.hd_data_helper_run = False

    def _ICE5G_as_HD_data_21k_0k_Helper(self):
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
                                 grid_type='LatLong10min')
        self._ICE5G_data_ALG4_sinkless_0k_upscale_riverflows_and_river_mouth_flows(file_label,new_label=False)        
        
    def ICE5G_data_ALG4_sinkless_no_true_sinks_0k(self):
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
        file_label = self._generate_file_label()
        fine_fields_filelabel = "ICE5G_data_ALG4_sinkless_downscaled_ls_mask_0k_20160930_001057" 
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
                                        flip_up=True, rotate180lr=True, invert_data=False,
                                        timeslice=None, grid_type='HD')
        self._apply_transforms_to_field(input_filename=HD_orography_filename,
                                        output_filename=transformed_HD_orography_filename, 
                                        flip_up=True, rotate180lr=True, invert_data=False,
                                        timeslice=None, grid_type='HD')
        self._apply_transforms_to_field(input_filename=HD_ls_mask_filename,
                                        output_filename=transformed_HD_ls_mask_filename,
                                        flip_up=True, rotate180lr=True, invert_data=True,
                                        timeslice=None, grid_type='HD')
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
        
class GLAC_Data_Drivers(ICE5G_Data_Drivers):

    def GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_timeslice0(self):
        self._GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs(timeslice=0)
        
    def GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_27timeslices_merge_timeslices_only(self):
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
                                        flip_up=True, rotate180lr=True, invert_data=False,
                                        timeslice=None, grid_type='HD')
        self._apply_transforms_to_field(input_filename=HD_orography_filename,
                                        output_filename=transformed_HD_orography_filename, 
                                        flip_up=True, rotate180lr=True, invert_data=False,
                                        timeslice=None, grid_type='HD')
        self._apply_transforms_to_field(input_filename=HD_ls_mask_filename,
                                        output_filename=transformed_HD_ls_mask_filename,
                                        flip_up=True, rotate180lr=True, invert_data=True,
                                        timeslice=None, grid_type='HD')
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
    
    def ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs(self):
        orog_corrections_filename = path.join(self.orography_corrections_fields_path,
                                              "orog_corrs_field_ICE5G_data_ALG4_sink"
                                              "less_downscaled_ls_mask_0k_20160930_001057.nc")
        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"10min-topo-present-from-virna.nc")
        orography_filename = self.generated_orography_filepath + file_label + '.nc'
        HD_orography_filename = self.upscaled_orography_filepath + file_label + '_HD' + '.nc'
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        original_ls_mask_filename = path.join(self.ls_masks_path,"10min-mask-present-from-virna.nc")
        HD_ls_mask_filename = self.generated_ls_mask_filepath + file_label + '_HD' + '.nc'
        unsorted_catchments_filename = self.generated_catchments_path + 'unsorted_' +\
            file_label + '.nc'
        utilities.change_dtype(input_filename=original_ls_mask_filename, 
                               output_filename=original_ls_mask_filename,
                               new_dtype=np.int32,grid_type='LatLong10min')
        utilities.apply_orog_correction_field(original_orography_filename=original_orography_filename,
                                              orography_corrections_filename=orog_corrections_filename,
                                              corrected_orography_filename=orography_filename,
                                              grid_type="LatLong10min")
        fill_sinks_driver.generate_sinkless_flow_directions(filename=orography_filename,
                                                            output_filename=rdirs_filename,
                                                            ls_mask_filename=\
                                                            original_ls_mask_filename,
                                                            truesinks_filename=None,
                                                            catchment_nums_filename=\
                                                            unsorted_catchments_filename,
                                                            grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=rdirs_filename, 
                                 output_file_label=file_label,
                                 ls_mask_filename=original_ls_mask_filename,
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
                                        flip_up=True, rotate180lr=True, invert_data=False,
                                        timeslice=None, grid_type='HD')
        self._apply_transforms_to_field(input_filename=HD_orography_filename,
                                        output_filename=transformed_HD_orography_filename, 
                                        flip_up=True, rotate180lr=True, invert_data=False,
                                        timeslice=None, grid_type='HD')
        self._apply_transforms_to_field(input_filename=HD_ls_mask_filename,
                                        output_filename=transformed_HD_ls_mask_filename,
                                        flip_up=True, rotate180lr=True, invert_data=True,
                                        timeslice=None, grid_type='HD')
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
        
    def ten_minute_data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs(self):
        orog_corrections_filename = path.join(self.orography_corrections_fields_path,
                                              "orog_corrs_field_ICE5G_data_ALG4_sink"
                                              "less_downscaled_ls_mask_0k_20160930_001057.nc")
        file_label = self._generate_file_label()
        original_orography_filename = path.join(self.orography_path,"10min-topo-lgm-from-virna.nc")
        orography_filename = self.generated_orography_filepath + file_label + '.nc'
        HD_orography_filename = self.upscaled_orography_filepath + file_label + '_HD' + '.nc'
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        original_ls_mask_filename = path.join(self.ls_masks_path,"10min-mask-lgm-from-virna.nc")
        HD_ls_mask_filename = self.generated_ls_mask_filepath + file_label + '_HD' + '.nc'
        unsorted_catchments_filename = self.generated_catchments_path + 'unsorted_' +\
            file_label + '.nc'
        utilities.change_dtype(input_filename=original_ls_mask_filename, 
                               output_filename=original_ls_mask_filename,
                               new_dtype=np.int32,grid_type='LatLong10min')
        utilities.apply_orog_correction_field(original_orography_filename=original_orography_filename,
                                              orography_corrections_filename=orog_corrections_filename,
                                              corrected_orography_filename=orography_filename,
                                              grid_type="LatLong10min")
        fill_sinks_driver.generate_sinkless_flow_directions(filename=orography_filename,
                                                            output_filename=rdirs_filename,
                                                            ls_mask_filename=\
                                                            original_ls_mask_filename,
                                                            truesinks_filename=None,
                                                            catchment_nums_filename=\
                                                            unsorted_catchments_filename,
                                                            grid_type='LatLong10min')
        self._run_postprocessing(rdirs_filename=rdirs_filename, 
                                 output_file_label=file_label,
                                 ls_mask_filename=original_ls_mask_filename,
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
                                        flip_up=True, rotate180lr=True, invert_data=False,
                                        timeslice=None, grid_type='HD')
        self._apply_transforms_to_field(input_filename=HD_orography_filename,
                                        output_filename=transformed_HD_orography_filename, 
                                        flip_up=True, rotate180lr=True, invert_data=False,
                                        timeslice=None, grid_type='HD')
        self._apply_transforms_to_field(input_filename=HD_ls_mask_filename,
                                        output_filename=transformed_HD_ls_mask_filename,
                                        flip_up=True, rotate180lr=True, invert_data=True,
                                        timeslice=None, grid_type='HD')
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

def main():
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
    #etopo1_data_drivers = ETOPO1_Data_Drivers()
    #etopo1_data_drivers.etopo1_data_all_points()
    #etopo1_data_drivers.etopo1_data_ALG4_sinkless()
    #utilties_drivers = Utilities_Drivers()
    #utilties_drivers.convert_corrected_HD_hydrology_dat_files_to_nc()
    #utilties_drivers.recreate_connected_HD_lsmask()
    #utilties_drivers.recreate_connected_HD_lsmask_true_seas_inc_casp_only()
    #utilties_drivers.downscale_HD_ls_seed_points_to_1min_lat_lon()
    #utilties_drivers.downscale_HD_ls_seed_points_to_10min_lat_lon()
    #utilties_drivers.downscale_HD_ls_seed_points_to_10min_lat_lon_true_seas_inc_casp_only()
    #original_hd_model_rfd_drivers = Original_HD_Model_RFD_Drivers()
    #original_hd_model_rfd_drivers.corrected_HD_rdirs_post_processing()
    #original_hd_model_rfd_drivers.extract_ls_mask_from_corrected_HD_rdirs()
    #original_hd_model_rfd_drivers.extract_true_sinks_from_corrected_HD_rdirs()
    #glac_data_drivers = GLAC_Data_Drivers()
    #glac_data_drivers.GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_timeslice0()        
    #glac_data_drivers.test_paragen_on_GLAC_data()
    #glac_data_drivers.GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_27timeslices()
    #glac_data_drivers.GLAC_data_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_27timeslices_merge_timeslices_only()
    ten_minute_data_from_virna_driver = Ten_Minute_Data_From_Virna_Driver()
    #ten_minute_data_from_virna_driver.ten_minute_data_from_virna_0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs()
    ten_minute_data_from_virna_driver.ten_minute_data_from_virna_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs()

if __name__ == '__main__':
    main()