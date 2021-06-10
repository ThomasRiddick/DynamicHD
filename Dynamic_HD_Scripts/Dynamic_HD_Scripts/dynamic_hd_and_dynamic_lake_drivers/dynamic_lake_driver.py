'''
Created on Dec 4, 2017

@author: thomasriddick
'''
from os.path import join
import os.path as path
import numpy as np
import time
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.base import field
from Dynamic_HD_Scripts.tools import determine_river_directions
from Dynamic_HD_Scripts.tools import extract_lake_volumes
from Dynamic_HD_Scripts.tools import compute_catchments as cc
from Dynamic_HD_Scripts.tools import flow_to_grid_cell as ftgc
from Dynamic_HD_Scripts.tools import connect_coarse_lake_catchments as cclc
from Dynamic_HD_Scripts.tools import dynamic_lake_operators
from Dynamic_HD_Scripts.utilities import utilities
from Dynamic_HD_Scripts.dynamic_hd_and_dynamic_lake_drivers import fill_sinks_driver
from Dynamic_HD_Scripts.dynamic_hd_and_dynamic_lake_drivers import dynamic_hd_driver

class Dynamic_Lake_Drivers(dynamic_hd_driver.Dynamic_HD_Drivers):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super(Dynamic_Lake_Drivers,self).__init__()

    def prepare_orography_ICE5G_0k_uncorrected(self):
        file_label = self._generate_file_label()
        ICE5G_0k_file = join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        ICE5G_0k_orography_fieldname = "orog"
        ICE5G_minima_filename = self.generated_minima_filepath+file_label+".nc"
        ICE5G_minima_reduced_filename = self.generated_minima_filepath+file_label+"_reduced.nc"
        ICE5G_minima_fieldname = "minima"
        ICE5G_flowdirs_filename = self.generated_rdir_filepath+file_label+'.nc'
        ICE5G_flowdirs_fieldname = "FDIR"
        ICE5G_output_orog_filename= self.generated_orography_filepath + file_label + '.nc'
        ICE5G_lakemask_filename= join(self.lakemask_filepath,"empty_lakemask.nc")
        ICE5G_lakemask_fieldname="lakemask"
        ICE5G_landsea_mask_filename=join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        ICE5G_landsea_mask_fieldname="sftlf"
        lakemask = field.makeEmptyField(field_type='Generic',dtype=np.int32,grid_type="LatLong10min")
        orog = iodriver.advanced_field_loader(ICE5G_0k_file,fieldname=ICE5G_0k_orography_fieldname)
        lakemask.grid = orog.get_grid()
        iodriver.advanced_field_writer(ICE5G_lakemask_filename,lakemask,
                                       ICE5G_lakemask_fieldname,clobber=True)

        dynamic_lake_operators.advanced_local_minima_finding_driver(ICE5G_0k_file,
                                                                    ICE5G_0k_orography_fieldname,
                                                                    ICE5G_minima_filename,
                                                                    ICE5G_minima_fieldname)
        dynamic_lake_operators.reduce_connected_areas_to_points(ICE5G_minima_filename,
                                                                ICE5G_minima_fieldname,
                                                                ICE5G_minima_reduced_filename,
                                                                ICE5G_minima_fieldname)
        fill_sinks_driver.advanced_sinkless_flow_directions_generator(filename=ICE5G_0k_file,
                                                                      output_filename=ICE5G_flowdirs_filename,
                                                                      ls_mask_filename=
                                                                      ICE5G_landsea_mask_filename,
                                                                      fieldname=ICE5G_0k_orography_fieldname,
                                                                      output_fieldname=
                                                                      ICE5G_flowdirs_fieldname,
                                                                      ls_mask_fieldname=
                                                                      ICE5G_landsea_mask_fieldname)
        dynamic_lake_operators.advanced_burn_carved_rivers_driver(input_orography_file=
                                                                  ICE5G_0k_file,
                                                                  input_orography_fieldname=
                                                                  ICE5G_0k_orography_fieldname,
                                                                  input_rdirs_file=
                                                                  ICE5G_flowdirs_filename,
                                                                  input_rdirs_fieldname=
                                                                  ICE5G_flowdirs_fieldname,
                                                                  input_minima_file=
                                                                  ICE5G_minima_reduced_filename,
                                                                  input_minima_fieldname=
                                                                  ICE5G_minima_fieldname,
                                                                  input_lakemask_file=
                                                                  ICE5G_lakemask_filename,
                                                                  input_lakemask_fieldname=
                                                                  ICE5G_lakemask_fieldname,
                                                                  output_orography_file=
                                                                  ICE5G_output_orog_filename,
                                                                  output_orography_fieldname=
                                                                  ICE5G_0k_orography_fieldname)

    def prepare_orography_ICE5G_0k_corrected(self):
        file_label = self._generate_file_label()
        original_orography_filename = join(self.orography_path,
                                                "ice5g_v1_2_00_0k_10min.nc")
        orog_corrections_filename = join(self.orography_corrections_fields_path,
                                                  "orog_corrs_field_ICE5G_and_tarasov_upscaled_"
                                                  "srtm30plus_north_america_only_data_ALG4_sinkless"
                                                  "_glcc_olson_lsmask_0k_20170517_003802.nc")
        intermediary_orography_filename = self.generated_orography_filepath +\
                                                "intermediary_" + file_label + '.nc'
        second_intermediary_orography_filename = self.generated_orography_filepath +\
                                                "second_intermediary_" + file_label + '.nc'
        orography_filename = self.generated_orography_filepath + file_label + '.nc'
        output_orog_filename = self.generated_orography_filepath + "lake_" + file_label + '.nc'
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        original_ls_mask_filename=join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        original_landsea_mask_fieldname="sftlf"
        original_ls_mask_with_new_dtype_filename = (self.generated_ls_mask_filepath +
                                                    file_label + '_orig' + '.nc')
        original_ls_mask_with_grid_filename= (self.generated_ls_mask_filepath +
                                                    file_label + '_grid' + '.nc')
        minima_filename = self.generated_minima_filepath+file_label+".nc"
        minima_reduced_filename = self.generated_minima_filepath+file_label+"_reduced.nc"
        minima_fieldname = "minima"
        lakemask_filename= self.lakemask_filepath+"empty_lakemask.nc"
        lakemask_fieldname="lakemask"
        glacial_mask_file = join(self.orography_path,"ice5g_v1_2_21_0k_10min.nc")
        utilities.change_dtype(input_filename=original_ls_mask_filename,
                               output_filename=original_ls_mask_with_new_dtype_filename,
                               input_fieldname=original_landsea_mask_fieldname,
                               output_fieldname="lsmask",
                               new_dtype=np.int32,grid_type='LatLong10min')
        utilities.apply_orog_correction_field(original_orography_filename=original_orography_filename,
                                              orography_corrections_filename=orog_corrections_filename,
                                              corrected_orography_filename=
                                              intermediary_orography_filename,
                                              original_orography_fieldname=\
                                              "orog",
                                              grid_type="LatLong10min")
        utilities.replace_corrected_orography_with_original_for_glaciated_grid_points_drivers(
          input_corrected_orography_file=intermediary_orography_filename,
          input_original_orography_file=original_orography_filename,
          input_glacier_mask_file=glacial_mask_file,
          out_orography_file=second_intermediary_orography_filename,
          grid_type="LatLong10min")
        iodriver.add_grid_information_to_field(target_filename=
                                               orography_filename,
                                               original_filename=
                                               second_intermediary_orography_filename,
                                               target_fieldname="field_value",
                                               original_fieldname="field_value",
                                               flip_ud_raw=True,rotate180lr_raw=True,
                                               grid_desc_file=self.ten_minute_grid_filepath)
        iodriver.add_grid_information_to_field(target_filename=
                                               original_ls_mask_with_grid_filename,
                                               original_filename=
                                               original_ls_mask_with_new_dtype_filename,
                                               target_fieldname="lsmask",
                                               original_fieldname="lsmask",
                                               flip_ud_raw=True,rotate180lr_raw=True,
                                               grid_desc_file=self.ten_minute_grid_filepath)
        dynamic_lake_operators.advanced_local_minima_finding_driver(orography_filename,
                                                                    "field_value",
                                                                    minima_filename,
                                                                    minima_fieldname)
        dynamic_lake_operators.reduce_connected_areas_to_points(minima_filename,
                                                                minima_fieldname,
                                                                minima_reduced_filename,
                                                                minima_fieldname)
        fill_sinks_driver.advanced_sinkless_flow_directions_generator(filename=orography_filename,
                                                                      output_filename=rdirs_filename,
                                                                      ls_mask_filename=
                                                                      original_ls_mask_with_grid_filename,
                                                                      fieldname="field_value",
                                                                      output_fieldname=
                                                                      "rdir",
                                                                      ls_mask_fieldname=
                                                                      "lsmask")
        dynamic_lake_operators.advanced_burn_carved_rivers_driver(input_orography_file=
                                                                  orography_filename,
                                                                  input_orography_fieldname=
                                                                  "field_value",
                                                                  input_rdirs_file=
                                                                  rdirs_filename,
                                                                  input_rdirs_fieldname=
                                                                  "rdir",
                                                                  input_minima_file=
                                                                  minima_filename,
                                                                  input_minima_fieldname=
                                                                  minima_fieldname,
                                                                  input_lakemask_file=
                                                                  lakemask_filename,
                                                                  input_lakemask_fieldname=
                                                                  lakemask_fieldname,
                                                                  output_orography_file=
                                                                  output_orog_filename,
                                                                  output_orography_fieldname=
                                                                  "field_value")


    def prepare_orography_ICE6G_21k_corrected(self):
        file_label = self._generate_file_label()
        original_orography_filename = join(self.orography_path,
                                                "ice5g_v1_2_00_0k_10min.nc")
        ice6g_0k_filename = join(self.orography_path,
                                 "Ice6g_c_VM5a_10min_0k.nc")
        ice6g_21k_filename = join(self.orography_path,
                                 "Ice6g_c_VM5a_10min_21k.nc")
        orog_corrections_filename = join(self.orography_corrections_fields_path,
                                                  "orog_corrs_field_ICE5G_and_tarasov_upscaled_"
                                                  "srtm30plus_north_america_only_data_ALG4_sinkless"
                                                  "_glcc_olson_lsmask_0k_20170517_003802.nc")
        intermediary_orography_filename = self.generated_orography_filepath +\
                                                "intermediary_" + file_label + '.nc'
        second_intermediary_orography_filename = self.generated_orography_filepath +\
                                                "second_intermediary_" + file_label + '.nc'
        orography_filename = self.generated_orography_filepath + file_label + '.nc'
        output_0k_ice5g_orog_filename = self.generated_orography_filepath + "0k_ice5g_lake_" + file_label + '.nc'
        output_21k_ice6g_orog_filename = self.generated_orography_filepath + "21k_ice6g_lake_" + file_label + '.nc'
        output_21k_ice6g_orog_sinkless_filename = (self.generated_orography_filepath +
                                                   "21k_ice6g_lake_sinkless_" + file_label + '.nc')
        output_21k_ice6g_orog_sinkless_improved_filename = (self.generated_orography_filepath +
                                                            "21k_ice6g_lake_sinkless_improved_" + file_label + '.nc')
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        original_ls_mask_filename=join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        original_landsea_mask_fieldname="sftlf"
        original_ls_mask_with_new_dtype_filename = (self.generated_ls_mask_filepath +
                                                    file_label + '_orig' + '.nc')
        original_ls_mask_with_grid_filename= (self.generated_ls_mask_filepath +
                                                    file_label + '_grid' + '.nc')
        minima_filename = self.generated_minima_filepath+file_label+".nc"
        minima_filename_21k = self.generated_minima_filepath+file_label+"_21k.nc"
        minima_reduced_filename = self.generated_minima_filepath+file_label+"_reduced.nc"
        minima_reduced_filename_21k = self.generated_minima_filepath+file_label+"_reduced_21k.nc"
        minima_fieldname = "minima"
        lakemask_filename= self.lakemask_filepath+"/empty_lakemask.nc"
        lakemask_fieldname="lakemask"
        glacial_mask_file = join(self.orography_path,"ice5g_v1_2_21_0k_10min.nc")
        utilities.change_dtype(input_filename=original_ls_mask_filename,
                               output_filename=original_ls_mask_with_new_dtype_filename,
                               input_fieldname=original_landsea_mask_fieldname,
                               output_fieldname="lsmask",
                               new_dtype=np.int32,grid_type='LatLong10min')
        utilities.apply_orog_correction_field(original_orography_filename=original_orography_filename,
                                              orography_corrections_filename=orog_corrections_filename,
                                              corrected_orography_filename=
                                              intermediary_orography_filename,
                                              original_orography_fieldname=\
                                              "orog",
                                              grid_type="LatLong10min")
        utilities.replace_corrected_orography_with_original_for_glaciated_grid_points_drivers(
          input_corrected_orography_file=intermediary_orography_filename,
          input_original_orography_file=original_orography_filename,
          input_glacier_mask_file=glacial_mask_file,
          out_orography_file=second_intermediary_orography_filename,
          grid_type="LatLong10min")
        iodriver.add_grid_information_to_field(target_filename=
                                               orography_filename,
                                               original_filename=
                                               second_intermediary_orography_filename,
                                               target_fieldname="field_value",
                                               original_fieldname="field_value",
                                               flip_ud_raw=True,rotate180lr_raw=True,
                                               grid_desc_file=self.ten_minute_grid_filepath)
        iodriver.add_grid_information_to_field(target_filename=
                                               original_ls_mask_with_grid_filename,
                                               original_filename=
                                               original_ls_mask_with_new_dtype_filename,
                                               target_fieldname="lsmask",
                                               original_fieldname="lsmask",
                                               flip_ud_raw=True,rotate180lr_raw=True,
                                               grid_desc_file=self.ten_minute_grid_filepath)
        dynamic_lake_operators.advanced_local_minima_finding_driver(orography_filename,
                                                                    "field_value",
                                                                    minima_filename,
                                                                    minima_fieldname)
        dynamic_lake_operators.reduce_connected_areas_to_points(minima_filename,
                                                                minima_fieldname,
                                                                minima_reduced_filename,
                                                                minima_fieldname)
        fill_sinks_driver.advanced_sinkless_flow_directions_generator(filename=orography_filename,
                                                                      output_filename=rdirs_filename,
                                                                      ls_mask_filename=
                                                                      original_ls_mask_with_grid_filename,
                                                                      fieldname="field_value",
                                                                      output_fieldname=
                                                                      "rdir",
                                                                      ls_mask_fieldname=
                                                                      "lsmask")
        dynamic_lake_operators.advanced_burn_carved_rivers_driver(input_orography_file=
                                                                  orography_filename,
                                                                  input_orography_fieldname=
                                                                  "field_value",
                                                                  input_rdirs_file=
                                                                  rdirs_filename,
                                                                  input_rdirs_fieldname=
                                                                  "rdir",
                                                                  input_minima_file=
                                                                  minima_filename,
                                                                  input_minima_fieldname=
                                                                  minima_fieldname,
                                                                  input_lakemask_file=
                                                                  lakemask_filename,
                                                                  input_lakemask_fieldname=
                                                                  lakemask_fieldname,
                                                                  output_orography_file=
                                                                  output_0k_ice5g_orog_filename,
                                                                  output_orography_fieldname=
                                                                  "Topo")
        utilities.advanced_rebase_orography_driver(orography_filename=
                                                   ice6g_21k_filename,
                                                   present_day_base_orography_filename=
                                                   ice6g_0k_filename,
                                                   present_day_reference_orography_filename=
                                                   output_0k_ice5g_orog_filename,
                                                   rebased_orography_filename=
                                                   output_21k_ice6g_orog_filename,
                                                   orography_fieldname="Topo",
                                                   present_day_base_orography_fieldname="Topo",
                                                   present_day_reference_orography_fieldname="Topo",
                                                   rebased_orography_fieldname="Topo")
        fill_sinks_driver.\
        generate_orography_with_sinks_filled_advanced_driver(output_21k_ice6g_orog_filename,
                                                             output_21k_ice6g_orog_sinkless_filename,
                                                             "Topo",
                                                             "Topo",
                                                             ls_mask_filename=None,
                                                             truesinks_filename=None,
                                                             ls_mask_fieldname=None,
                                                             truesinks_fieldname=None,
                                                             add_slight_slope_when_filling_sinks=False,
                                                             slope_param=0.1)
        ice6g_sinkless_field = iodriver.advanced_field_loader(output_21k_ice6g_orog_sinkless_filename,
                                                              fieldname="Topo",
                                                              adjust_orientation=True)
        ice6g_field = iodriver.advanced_field_loader(output_21k_ice6g_orog_filename,fieldname="Topo",
                                                     adjust_orientation=True)
        ice6g_21k_icemask = iodriver.advanced_field_loader(ice6g_21k_filename,
                                                           fieldname="sftgif",
                                                           adjust_orientation=True)
        ice6g_21k_lsmask = iodriver.advanced_field_loader(ice6g_21k_filename,
                                                          fieldname="sftlf",
                                                          adjust_orientation=True)
        ice6g_21k_icemask.invert_data()
        ice6g_field.mask_field_with_external_mask(ice6g_21k_icemask.get_data())
        ice6g_sinkless_field.update_field_with_partially_masked_data(ice6g_field)
        ice6g_field.mask_field_with_external_mask(ice6g_21k_lsmask.get_data())
        ice6g_sinkless_field.update_field_with_partially_masked_data(ice6g_field)
        iodriver.advanced_field_writer(output_21k_ice6g_orog_sinkless_improved_filename,
                                       ice6g_sinkless_field,fieldname="Topo",clobber=True)
        dynamic_lake_operators.advanced_local_minima_finding_driver(output_21k_ice6g_orog_filename,
                                                                    "Topo",
                                                                    minima_filename_21k,
                                                                    minima_fieldname)
        dynamic_lake_operators.reduce_connected_areas_to_points(minima_filename_21k,
                                                                minima_fieldname,
                                                                minima_reduced_filename_21k,
                                                                minima_fieldname)
        print("minima filename: " + minima_reduced_filename_21k)
        print("minima fieldname" + minima_fieldname)
        print("ice6g_21k_filename" + ice6g_21k_filename)
        print("output21k_orog_filename" + output_21k_ice6g_orog_filename)

    def prepare_orography(self,orography_filename,orography_fieldname,timestep,
                          orography_0k_filename,orography_0k_fieldname,timestep_0k,
                          glacier_mask_filename,glacier_mask_fieldname,
                          glacier_mask_timestep,
                          ls_mask_filename,ls_mask_fieldname,ls_mask_timestep,
                          ls_mask_0k_filename,ls_mask_0k_fieldname,ls_mask_timestep_0k,
                          file_label):
        tstart = time.time()
        flip_ls_mask_0k   = False
        invert_ls_mask    = True
        rotate_lsmask_180_lr = True
        rotate_lsmask_180_lr_0k = False
        original_orography_filename = join(self.orography_path,
                                                "ice5g_v1_2_00_0k_10min.nc")
        true_sinks_filename = join(self.truesinks_path,
                                   "truesinks_ICE5G_and_tarasov_upscaled_srtm30plus_"
                                   "north_america_only_data_ALG4_sinkless_glcc_olson"
                                   "_lsmask_0k_20191014_173825_with_grid.nc")
        if timestep_0k is not None:
          orography_0k = iodriver.advanced_field_loader(orography_0k_filename,
                                                        time_slice=timestep_0k,
                                                        fieldname=orography_0k_fieldname)
          working_orog_0k_filename = self.generated_orography_filepath + \
                                     "_extracted_for_0k_" + \
                                     file_label + ".nc"
          iodriver.advanced_field_writer(working_orog_0k_filename,orography_0k,"Topo")
        else:
          working_orog_0k_filename = orography_filename
        if timestep is not None:
          orography = iodriver.advanced_field_loader(orography_filename,time_slice=timestep,
                                                     fieldname=orography_fieldname)
          working_orog_filename =  self.generated_orography_filepath + \
                                   "_extracted_for_{}".format(timestep) + \
                                   file_label + ".nc"
          iodriver.advanced_field_writer(working_orog_filename,orography,"Topo")
        else:
          working_orog_filename = orography_filename
        orog_corrections_filename = join(self.orography_corrections_fields_path,
                                                  "orog_corrs_field_ICE5G_and_tarasov_upscaled_"
                                                  "srtm30plus_north_america_only_data_ALG4_sinkless"
                                                  "_glcc_olson_lsmask_0k_20170517_003802_g.nc")
        intermediary_orography_filename = self.generated_orography_filepath +\
                                                "intermediary_" + file_label + '.nc'
        second_intermediary_orography_filename = self.generated_orography_filepath +\
                                                "second_intermediary_" + file_label + '.nc'
        #orography_filename = self.generated_orography_filepath + file_label + '.nc'
        output_0k_ice5g_orog_filename = self.generated_orography_filepath + "0k_ice5g_lake_" + file_label + '.nc'
        output_working_orog_filename = self.generated_orography_filepath + "{}_ice6g_lake_".format(timestep) + file_label + '.nc'
        output_intermediary_filtered_working_orog_filename = self.generated_orography_filepath +\
                                                             "{}_ice6g_lake_filtered_int_".format(timestep) +\
                                                file_label + '.nc'
        output_filtered_working_orog_filename = self.generated_orography_filepath +\
                                                "{}_ice6g_lake_filtered_".format(timestep) +\
                                                file_label + '.nc'
        output_working_orog_sinkless_filename = (self.generated_orography_filepath +
                                                   "{}_ice6g_lake_sinkless_".format(timestep) + file_label + '.nc')
        output_working_orog_sinkless_improved_filename = (self.generated_orography_filepath +
                                                            "{}_ice6g_lake_sinkless_improved_".format(timestep) + file_label + '.nc')
        orog_diff_filename = (self.generated_orography_filepath + "{}_lake_basins_".format(timestep) +
                              file_label + '.nc')
        rdirs_filename = self.generated_rdir_filepath + file_label + '.nc'
        if ls_mask_timestep_0k is not None:
          ls_mask_0k = iodriver.advanced_field_loader(ls_mask_0k_filename,
                                                      time_slice=ls_mask_timestep_0k,
                                                      fieldname=ls_mask_0k_fieldname,
                                                      grid_desc_file="/Users/thomasriddick/Documents/data/HDdata/grids/grid_10min.txt")
          original_ls_mask_filename= self.generated_ls_mask_filepath + \
                                     "_extracted_for_{}".format(timestep) + \
                                     file_label + ".nc"
          iodriver.advanced_field_writer(original_ls_mask_filename,ls_mask_0k,
                                         fieldname=ls_mask_0k_fieldname)
        else:
          original_ls_mask_filename=ls_mask_0k_filename
        original_landsea_mask_fieldname=ls_mask_0k_fieldname
        original_ls_mask_with_new_dtype_filename = (self.generated_ls_mask_filepath +
                                                    file_label + '_orig' + '.nc')
        original_ls_mask_with_grid_filename= (self.generated_ls_mask_filepath +
                                                    file_label + '_grid' + '.nc')
        minima_filename = self.generated_minima_filepath+file_label+".nc"
        minima_working_orog_filename = self.generated_minima_filepath+file_label+"_{}.nc".format(timestep)
        minima_reduced_filename = self.generated_minima_filepath+file_label+"_reduced.nc"
        minima_reduced_filename_working_orog = \
          self.generated_minima_filepath+file_label+"_reduced_{}.nc".format(timestep)
        minima_fieldname = "minima"
        lakemask_filename= self.lakemask_filepath+"/empty_lakemask.nc"
        lakemask_fieldname="lakemask"
        glacial_mask_file = join(self.orography_path,"ice5g_v1_2_00_0k_10min.nc")
        utilities.change_dtype(input_filename=original_ls_mask_filename,
                               output_filename=original_ls_mask_with_new_dtype_filename,
                               input_fieldname=original_landsea_mask_fieldname,
                               output_fieldname="lsmask",
                               new_dtype=np.int32,grid_type='LatLong10min')
        utilities.advanced_apply_orog_correction_field(original_orography_filename=
                                                       original_orography_filename,
                                                       orography_corrections_filename=
                                                       orog_corrections_filename,
                                                       corrected_orography_filename=
                                                       intermediary_orography_filename,
                                                       original_orography_fieldname=
                                                       "orog")
        utilities.advanced_replace_corrected_orog_with_orig_for_glcted_grid_points_drivers(
          input_corrected_orography_file=intermediary_orography_filename,
          input_original_orography_file=original_orography_filename,
          input_glacier_mask_file=glacial_mask_file,
          out_orography_file=second_intermediary_orography_filename,
          input_corrected_orography_fieldname=None,
          input_original_orography_fieldname=None,
          input_glacier_mask_fieldname=None,
          out_orography_fieldname=None)
        orography_filename = second_intermediary_orography_filename
        iodriver.add_grid_information_to_field(target_filename=
                                               original_ls_mask_with_grid_filename,
                                               original_filename=
                                               original_ls_mask_with_new_dtype_filename,
                                               target_fieldname="lsmask",
                                               original_fieldname="lsmask",
                                               flip_ud_raw=flip_ls_mask_0k,
                                               rotate180lr_raw=rotate_lsmask_180_lr_0k,
                                               grid_desc_file=self.ten_minute_grid_filepath)
        dynamic_lake_operators.advanced_local_minima_finding_driver(orography_filename,
                                                                    "field_value",
                                                                    minima_filename,
                                                                    minima_fieldname)
        dynamic_lake_operators.reduce_connected_areas_to_points(minima_filename,
                                                                minima_fieldname,
                                                                minima_reduced_filename,
                                                                minima_fieldname)
        fill_sinks_driver.advanced_sinkless_flow_directions_generator(filename=orography_filename,
                                                                      output_filename=rdirs_filename,
                                                                      ls_mask_filename=
                                                                      original_ls_mask_with_grid_filename,
                                                                      truesinks_filename=
                                                                      true_sinks_filename,
                                                                      fieldname="field_value",
                                                                      output_fieldname=
                                                                      "rdir",
                                                                      ls_mask_fieldname=
                                                                      "lsmask",
                                                                      truesinks_fieldname=
                                                                      "true_sinks")
        dynamic_lake_operators.advanced_burn_carved_rivers_driver(input_orography_file=
                                                                  orography_filename,
                                                                  input_orography_fieldname=
                                                                  "field_value",
                                                                  input_rdirs_file=
                                                                  rdirs_filename,
                                                                  input_rdirs_fieldname=
                                                                  "rdir",
                                                                  input_minima_file=
                                                                  minima_filename,
                                                                  input_minima_fieldname=
                                                                  minima_fieldname,
                                                                  input_lakemask_file=
                                                                  lakemask_filename,
                                                                  input_lakemask_fieldname=
                                                                  lakemask_fieldname,
                                                                  output_orography_file=
                                                                  output_0k_ice5g_orog_filename,
                                                                  output_orography_fieldname=
                                                                  "Topo",
                                                                  add_slope = True,
                                                                  max_exploration_range = 10,
                                                                  minimum_height_change_threshold = 5.0,
                                                                  short_path_threshold = 6,
                                                                  short_minimum_height_change_threshold = 0.25)
        new_orography_corrections_filename = path.join(self.orography_corrections_fields_path,
                                                       "ice5g_0k_lake_corrs_" + file_label + ".nc")
        utilities.advanced_orog_correction_field_generator(original_orography_filename=
                                                           original_orography_filename,
                                                           corrected_orography_filename=
                                                           output_0k_ice5g_orog_filename,
                                                           orography_corrections_filename=
                                                           new_orography_corrections_filename,
                                                           original_orography_fieldname=
                                                           "orog",
                                                           corrected_orography_fieldname=
                                                           "Topo",
                                                           orography_corrections_fieldname=
                                                           "orog")
        print("Time for initial setup: " + str(time.time() - tstart))
        utilities.advanced_rebase_orography_driver(orography_filename=
                                                   working_orog_filename,
                                                   present_day_base_orography_filename=
                                                   working_orog_0k_filename,
                                                   present_day_reference_orography_filename=
                                                   output_0k_ice5g_orog_filename,
                                                   rebased_orography_filename=
                                                   output_working_orog_filename,
                                                   orography_fieldname="Topo",
                                                   present_day_base_orography_fieldname="Topo",
                                                   present_day_reference_orography_fieldname="Topo",
                                                   rebased_orography_fieldname="Topo")
        fill_sinks_driver.\
        generate_orography_with_sinks_filled_advanced_driver(output_working_orog_filename,
                                                             output_working_orog_sinkless_filename,
                                                             "Topo",
                                                             "Topo",
                                                             ls_mask_filename=None,
                                                             truesinks_filename=None,
                                                             ls_mask_fieldname=None,
                                                             truesinks_fieldname=None,
                                                             add_slight_slope_when_filling_sinks=False,
                                                             slope_param=0.1)

        dynamic_lake_operators.\
        advanced_shallow_lake_filtering_driver(input_unfilled_orography_file=
                                               output_working_orog_filename,
                                               input_unfilled_orography_fieldname="Topo",
                                               input_filled_orography_file=
                                               output_working_orog_sinkless_filename,
                                               input_filled_orography_fieldname="Topo",
                                               output_unfilled_orography_file=
                                               output_intermediary_filtered_working_orog_filename,
                                               output_unfilled_orography_fieldname="Topo",
                                               minimum_depth_threshold=5.0)
        dynamic_lake_operators.\
        advanced_narrow_lake_filtering_driver(input_unfilled_orography_file=
                                              output_intermediary_filtered_working_orog_filename,
                                              input_unfilled_orography_fieldname=
                                              "Topo",
                                              input_filled_orography_file=
                                              output_working_orog_sinkless_filename,
                                              input_filled_orography_fieldname=
                                              "Topo",
                                              output_unfilled_orography_file=
                                              output_filtered_working_orog_filename,
                                              output_unfilled_orography_fieldname=
                                              "Topo",
                                              interior_cell_min_masked_neighbors=5,
                                              edge_cell_max_masked_neighbors=4,
                                              max_range=5,
                                              iterations=5)
        working_orog_sinkless_field = iodriver.advanced_field_loader(output_working_orog_sinkless_filename,
                                                              fieldname="Topo",
                                                              adjust_orientation=True)
        working_orog_field = iodriver.advanced_field_loader(output_filtered_working_orog_filename,fieldname="Topo",
                                                     adjust_orientation=True)
        working_orog_icemask = iodriver.advanced_field_loader(glacier_mask_filename,
                                                         time_slice=glacier_mask_timestep,
                                                         fieldname=glacier_mask_fieldname,
                                                         adjust_orientation=True)
        working_orog_lsmask = iodriver.advanced_field_loader(ls_mask_filename,
                                                        time_slice=ls_mask_timestep,
                                                        fieldname=ls_mask_fieldname,
                                                        adjust_orientation=False,
                                                        grid_desc_file="/Users/thomasriddick/Documents/data/HDdata/grids/grid_10min.txt")
        working_orog_icemask.invert_data()
        if (invert_ls_mask):
          working_orog_lsmask.invert_data()
        if (rotate_lsmask_180_lr):
          working_orog_lsmask.rotate_field_by_a_hundred_and_eighty_degrees()
        working_orog_field.mask_field_with_external_mask(working_orog_icemask.get_data())
        working_orog_sinkless_field.update_field_with_partially_masked_data(working_orog_field)
        working_orog_field.mask_field_with_external_mask(working_orog_lsmask.get_data())
        working_orog_sinkless_field.update_field_with_partially_masked_data(working_orog_field)
        iodriver.advanced_field_writer(output_working_orog_sinkless_improved_filename,
                                       working_orog_sinkless_field,fieldname="Topo",clobber=True)
        dynamic_lake_operators.advanced_local_minima_finding_driver(output_filtered_working_orog_filename,
                                                                    "Topo",
                                                                    minima_working_orog_filename,
                                                                    minima_fieldname)
        dynamic_lake_operators.reduce_connected_areas_to_points(minima_working_orog_filename,
                                                                minima_fieldname,
                                                                minima_reduced_filename_working_orog,
                                                                minima_fieldname)
        print("minima filename: " + minima_reduced_filename_working_orog)
        print("minima fieldname: " + minima_fieldname)
        print("timestep{}_filename: ".format(timestep) + working_orog_filename)
        print("timestep{}_orog_filename:".format(timestep) + output_working_orog_filename)
        improved_sinkless_orog = iodriver.advanced_field_loader(output_working_orog_sinkless_improved_filename,
                                                                fieldname="Topo",adjust_orientation=True)
        lake_orog = iodriver.advanced_field_loader(output_filtered_working_orog_filename,
                                                   fieldname="Topo",adjust_orientation=True)
        improved_sinkless_orog.subtract(lake_orog)
        iodriver.advanced_field_writer(orog_diff_filename,improved_sinkless_orog,
                                       fieldname="depth",clobber=True)

    def extract_lake_volumes_from_glac1D_basins(self):
      overarching_file_label = self._generate_file_label()
      transient_data_folder_path = "/Users/thomasriddick/Documents/data/transient_sim_data/1"
      timesteps_to_use = [ 950,1000,1050,1100,1150,1200,1250,1300,1350,
                          1400,1450,1500,1550,1600,1650,1700,1750,1800]
      for timestep in timesteps_to_use:
        file_label = self._generate_file_label() + "_" + str(timestep)
        lake_parameters_filepath = join(transient_data_folder_path,
                                        "lakeparas_prepare_basins_from_glac1D_{}.nc".format(timestep))
        basin_catchment_numbers_filepath = join(transient_data_folder_path,
                                                "basin_catchment_numbers_prepare_"
                                                "basins_from_glac1D_{}.nc".format(timestep))
        lake_volumes_out_filepath = ("/Users/thomasriddick/Documents/data/temp/"
                                     "lake_volumes_out_{}.nc".format(timestep))
        extract_lake_volumes.lake_volume_extraction_driver(lake_parameters_filepath,
                                                           basin_catchment_numbers_filepath,
                                                           lake_volumes_out_filepath)

    def prepare_basins_from_glac1D(self):
      overarching_file_label = self._generate_file_label()
      # timesteps_to_use = [ 950,1000,1050,1100,1150,1200,1250,1300,1350,
      #                     1400,1450,1500,1550,1600,1650,1700,1750,1800]
      timesteps_to_use = [1250]
      timestep_for_0k = 2600
      glac_1d_topo_filename = join(self.orography_path,
                                   "GLAC1D_Top01_surf.nc")
      ls_mask_filename = join(self.ls_masks_path,
                             "10min_lsmask_pmu0178_merged.nc")
      ls_mask_0k_filename = join(self.ls_masks_path,"generated",
                                 "ls_mask_make_1000m_depth_contour_mask_from_ICE6G_20200721_144332.nc")
      glacier_mask_filename = join(self.orography_path,
                                   "GLAC1D_ICEM_10min.nc")
      cell_areas_filename_10min = join(self.grid_areas_and_spacings_filepath,
                                       "10min_grid_area_default_R.nc")
      for timestep in timesteps_to_use:
        file_label = self._generate_file_label() + "_" + str(timestep)
        self.prepare_orography(orography_filename=glac_1d_topo_filename,
                               orography_fieldname="HDCB",
                               timestep=timestep,
                               orography_0k_filename=glac_1d_topo_filename,
                               orography_0k_fieldname="HDCB",
                               timestep_0k=timestep_for_0k,
                               glacier_mask_filename=glacier_mask_filename,
                               glacier_mask_fieldname="ICEM",
                               glacier_mask_timestep=timestep,
                               ls_mask_filename=ls_mask_filename,
                               ls_mask_fieldname="field_value",
                               ls_mask_timestep=timestep,
                               #ls_mask_0k_filename=ls_mask_filename,
                               ls_mask_0k_filename=ls_mask_0k_filename,
                               #ls_mask_0kfieldname="field_value",
                               ls_mask_0k_fieldname="lsm",
                               #ls_mask_timestep_0k=timestep_for_0k,
                               ls_mask_timestep_0k=None,
                               file_label=file_label)
        working_orography_filename = "/Users/thomasriddick/Documents/data/HDdata/orographys/generated/updated_orog_" + str(timestep) + \
                                     "_ice6g_lake_filtered_" + file_label + ".nc"
        lsmask_filename = "/Users/thomasriddick/Documents/data/HDdata/lsmasks/generated/ls_mask_" + file_label + "_grid.nc"
        self.prepare_river_directions_with_depressions(working_orography_filename=
                                                       working_orography_filename,
                                                       lsmask_filename=
                                                       lsmask_filename,
                                                       orography_fieldname="Topo",
                                                       lsmask_fieldname="lsmask",
                                                       file_label=file_label)
        minima_from_rdirs_filename = ("/Users/thomasriddick/Documents/data/HDdata/minima/"
                                      "minima_" + file_label + "_reduced"
                                      "_" + str(timestep) + "_landonly_from_rdirs.nc")
        utilities.advanced_extract_true_sinks_from_rdirs(rdirs_filename=
                                                         "/Users/thomasriddick/Documents/data/HDdata/rdirs/generated/"
                                                         "updated_RFDs_" + file_label + "_10min_with_depressions.nc",
                                                         truesinks_filename=
                                                         minima_from_rdirs_filename,
                                                         rdirs_fieldname="FDIR",
                                                         truesinks_fieldname="minima")
        dynamic_lake_operators.\
          advanced_basin_evaluation_driver(input_minima_file=
                                           minima_from_rdirs_filename,
                                           input_minima_fieldname="minima",
                                           input_raw_orography_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/orographys/"
                                           "generated/updated_orog_" + str(timestep) +
                                           "_ice6g_lake_filtered_" + file_label + ".nc",
                                           input_raw_orography_fieldname="Topo",
                                           input_corrected_orography_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/orographys/"
                                           "generated/updated_orog_" + str(timestep) +
                                           "_ice6g_lake_filtered_" + file_label + ".nc",
                                           input_corrected_orography_fieldname="Topo",
                                           input_cell_areas_file= cell_areas_filename_10min,
                                           input_cell_areas_fieldname="cell_area",
                                           input_prior_fine_rdirs_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/rdirs/generated/"
                                           "updated_RFDs_" + file_label + "_10min_with_depressions.nc",
                                           input_prior_fine_rdirs_fieldname="FDIR",
                                           input_prior_fine_catchments_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/catchmentmaps/"
                                           "catchmentmap_" + file_label + "_10mins.nc",
                                           input_prior_fine_catchments_fieldname="catchments",
                                           input_coarse_catchment_nums_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/catchmentmaps/"
                                           "catchmentmap_" + file_label + "_30mins.nc",
                                           input_coarse_catchment_nums_fieldname="catchments",
                                           input_coarse_rdirs_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/rdirs/generated/"
                                           "updated_RFDs_" + file_label + "_30min_with_depressions.nc",
                                           input_coarse_rdirs_fieldname="FDIR",
                                           combined_output_filename=
                                            join(self.lake_parameter_file_path,
                                                 "lakeparas_" + file_label + ".nc"),
                                           output_filepath=self.lake_parameter_file_path,
                                           output_filelabel=file_label,
                                           output_basin_catchment_nums_filepath=
                                           join(self.basin_catchment_numbers_path,
                                                "basin_catchment_numbers_" + file_label + ".nc"))
        with open(self.generated_lake_and_hd_params_log_path +
                  overarching_file_label + ".log",'a') as f:
          f.write("Timestep=" + str(timestep) + '\n')
          f.write(self.generated_hd_file_path + file_label + ".nc\n")
          f.write(join(self.lake_parameter_file_path,
                       "lakeparas_" + file_label + ".nc\n"))
          f.write(join(self.basin_catchment_numbers_path,
                       "basin_catchment_numbers_" + file_label + ".nc\n"))

    def prepare_river_directions_with_depressions(self,
                                                  working_orography_filename,
                                                  lsmask_filename,
                                                  orography_fieldname,
                                                  lsmask_fieldname,
                                                  file_label):
      rdirs_filename_10min = \
        self.generated_rdir_filepath + file_label + "_10min_with_depressions.nc"
      rdirs_filename_30min = \
        self.generated_rdir_filepath + file_label + "_30min_with_depressions.nc"
      determine_river_directions.\
        advanced_river_direction_determination_driver(rdirs_filename_10min,
                                                      working_orography_filename,
                                                      lsmask_filename,
                                                      truesinks_filename=None,
                                                      rdirs_fieldname="FDIR",
                                                      orography_fieldname=orography_fieldname,
                                                      lsmask_fieldname=lsmask_fieldname,
                                                      truesinks_fieldname=None,
                                                      always_flow_to_sea=True,
                                                      use_diagonal_nbrs=True,
                                                      mark_pits_as_true_sinks=True)
      fine_cumulative_flow_filename = (self.generated_flowmaps_filepath + file_label
                                       + '_10mins.nc')
      fine_catchments_filename = (self.generated_catchments_path + file_label
                                  + '_10mins.nc')
      cc.advanced_main(rdirs_filename_10min,"FDIR",
                       fine_catchments_filename,"catchments",
                       loop_logfile='/Users/thomasriddick/Documents/data/temp/loop_log.txt',
                       use_cpp_alg=True)
      ftgc.advanced_main(rdirs_filename=rdirs_filename_10min,
                         output_filename=fine_cumulative_flow_filename,
                         rdirs_fieldname='FDIR',
                         output_fieldname='cflow')
      cotat_plus_parameters_filename = join(self.cotat_plus_parameters_path,'cotat_plus_standard_params.nl')
      self._run_advanced_cotat_plus_upscaling(input_fine_rdirs_filename=
                                              rdirs_filename_10min,
                                              input_fine_cumulative_flow_filename=
                                              fine_cumulative_flow_filename,
                                              output_course_rdirs_filename=
                                              rdirs_filename_30min,
                                              input_fine_rdirs_fieldname="FDIR",
                                              input_fine_cumulative_flow_fieldname="cflow",
                                              output_course_rdirs_fieldname="FDIR",
                                              cotat_plus_parameters_filename=
                                              cotat_plus_parameters_filename,
                                              output_file_label=file_label,
                                              scaling_factor=3)
      coarse_cumulative_flow_filename = (self.generated_flowmaps_filepath + file_label
                                       + '_30mins.nc')
      coarse_catchments_filename = (self.generated_catchments_path + file_label
                                  + '_30mins.nc')
      cc.advanced_main(rdirs_filename_30min,"FDIR",
                       coarse_catchments_filename,"catchments",
                       loop_logfile='/Users/thomasriddick/Documents/data/temp/loop_log.txt',
                       use_cpp_alg=True)
      ftgc.advanced_main(rdirs_filename=rdirs_filename_30min,
                         output_filename=coarse_cumulative_flow_filename,
                         rdirs_fieldname='FDIR',
                         output_fieldname='cflow')
      coarse_orography_filename = (self.generated_orography_filepath + file_label + "with_depressions"
                                   + '_30mins.nc')
      coarse_lsmask_filename = (self.generated_ls_mask_filepath + file_label + "_30mins.nc")
      utilities.upscale_field_driver(input_filename=working_orography_filename,
                                     output_filename=coarse_orography_filename,
                                     input_grid_type='LatLong10min',
                                     output_grid_type='HD',
                                     method='Sum', timeslice=None,
                                     scalenumbers=True)
      utilities.upscale_field_driver(input_filename=lsmask_filename,
                                     output_filename=coarse_lsmask_filename,
                                     input_grid_type='LatLong10min',
                                     output_grid_type='HD',
                                     method='Mode', timeslice=None,
                                     scalenumbers=True)
      transformed_course_rdirs_filename = path.splitext(rdirs_filename_30min)[0] + '_transf' +\
                                          path.splitext(rdirs_filename_30min)[1]
      transformed_HD_filled_orography_filename = path.splitext(coarse_orography_filename)[0] + '_transf' +\
                                        path.splitext(coarse_orography_filename)[1]
      transformed_HD_ls_mask_filename = path.splitext(coarse_lsmask_filename)[0] + '_transf' +\
                                          path.splitext(coarse_lsmask_filename)[1]
      self._apply_transforms_to_field(input_filename=rdirs_filename_30min,
                                      output_filename=transformed_course_rdirs_filename,
                                      flip_ud=False, rotate180lr=True, invert_data=False,
                                      timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                      grid_type='HD')
      self._apply_transforms_to_field(input_filename=coarse_orography_filename,
                                      output_filename=transformed_HD_filled_orography_filename,
                                      flip_ud=True, rotate180lr=True, invert_data=False,
                                      timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                      grid_type='HD')
      self._apply_transforms_to_field(input_filename=coarse_lsmask_filename,
                                      output_filename=transformed_HD_ls_mask_filename,
                                      flip_ud=False, rotate180lr=False, invert_data=True,
                                      timeslice=None, griddescfile=self.half_degree_grid_filepath,
                                      grid_type='HD')

    def prepare_river_directions_with_depressions_from_glac1D(self):
      working_orography_filename = "/Users/thomasriddick/Documents/data/HDdata/orographys/generated/updated_orog_1900_ice6g_lake_prepare_orography_20190211_131605.nc"
      lsmask_filename = "/Users/thomasriddick/Documents/data/HDdata/lsmasks/generated/ls_mask_prepare_orography_20190211_131605_grid.nc"
      orography_fieldname = "Topo"
      lsmask_fieldname = "lsmask"
      file_label = self._generate_file_label()
      self.prepare_river_directions_with_depressions(working_orography_filename,
                                                     lsmask_filename,
                                                     orography_fieldname,
                                                     lsmask_fieldname,
                                                     file_label)

    def prepare_flow_parameters_from_rdirs(self,rdirs_filepath,orography_filepath,lsmask_filepath,
                                           file_label):
      self._generate_flow_parameters(rdir_file=rdirs_filepath,
                                     topography_file=orography_filepath,
                                     inner_slope_file=\
                                     path.join(self.orography_path,'bin_innerslope.dat'),
                                     lsmask_file=lsmask_filepath,
                                     null_file=\
                                     path.join(self.null_fields_filepath,'null.dat'),
                                     area_spacing_file=\
                                     path.join(self.grid_areas_and_spacings_filepath,
                                               'fl_dp_dl.dat'),
                                     orography_variance_file=\
                                     path.join(self.orography_path,'bin_toposig.dat'),
                                     output_dir=path.join(self.flow_params_dirs_path,
                                                         'hd_flow_params' + file_label),
                                     paragen_source_label=None,production_run=False,
                                     grid_type="HD")
      self._generate_hd_file(rdir_file=path.splitext(rdirs_filepath)[0] + ".dat",
                             lsmask_file=lsmask_filepath,
                             null_file=\
                             path.join(self.null_fields_filepath,'null.dat'),
                             area_spacing_file=\
                             path.join(self.grid_areas_and_spacings_filepath,
                                       'fl_dp_dl.dat'),
                             hd_grid_specs_file=self.half_degree_grid_filepath,
                             output_file=self.generated_hd_file_path + file_label + '.nc',
                             paras_dir=path.join(self.flow_params_dirs_path,
                                                 'hd_flow_params' + file_label))

    def evaluate_glac1D_ts1900_basins(self):
        file_label = self._generate_file_label()
        dynamic_lake_operators.\
          advanced_basin_evaluation_driver(input_minima_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/minima/"
                                           "minima_prepare_orography_20190401_115141_reduced"
                                           "_1900_landonly_from_rdirs.nc",
                                           input_minima_fieldname="minima",
                                           input_raw_orography_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/orographys/"
                                           "generated/updated_orog_1900_ice6g_lake_prepare_orography"
                                           "_20190211_131605.nc",
                                           input_raw_orography_fieldname="Topo",
                                           input_corrected_orography_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/orographys/"
                                           "generated/updated_orog_1900_ice6g_lake_prepare_orography"
                                           "_20190211_131605.nc",
                                           input_corrected_orography_fieldname="Topo",
                                           input_cell_areas_file="/Users/thomasriddick/Documents/"
                                           "data/HDdata/10min_grid_area_default_R.nc",
                                           input_cell_areas_fieldname="cell_area",
                                           input_prior_fine_rdirs_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/rdirs/generated/"
                                           "updated_RFDs_prepare_river_directions_with_depressions_"
                                           "20190401_115141_10min_with_depressions.nc",
                                           input_prior_fine_rdirs_fieldname="FDIR",
                                           input_prior_fine_catchments_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/catchmentmaps/"
                                           "catchmentmap_prepare_river_directions_with_depressions_"
                                           "20190401_115141_10mins.nc",
                                           input_prior_fine_catchments_fieldname="catchments",
                                           input_coarse_catchment_nums_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/catchmentmaps/"
                                           "catchmentmap_prepare_river_directions_with_depressions_"
                                           "20190401_115141_30mins.nc",
                                           input_coarse_catchment_nums_fieldname="catchments",
                                           input_coarse_rdirs_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/rdirs/generated/"
                                           "updated_RFDs_prepare_river_directions_with_depressions_"
                                           "20190401_115141_30min_with_depressions.nc",
                                           input_coarse_rdirs_fieldname="FDIR",
                                           combined_output_filename=
                                            join(self.lake_parameter_file_path,
                                                 "lakeparas" + file_label + ".nc"),
                                           output_filepath=self.lake_parameter_file_path,
                                           output_filelabel=file_label,
                                           output_basin_catchment_nums_filepath=
                                           join(self.basin_catchment_numbers_path,
                                                "basin_catchment_numbers" + file_label + ".nc"))

    def evaluate_ICE6G_lgm_basins(self):
        file_label = self._generate_file_label()
        dynamic_lake_operators.\
          advanced_basin_evaluation_driver(input_minima_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/minima/"
                                           "minima_prepare_orography_ICE6G_21k_corrected_20180921"
                                           "_155937_reduced_21k_landonly.nc",
                                           input_minima_fieldname="minima",
                                           input_raw_orography_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/orographys/"
                                           "generated/updated_orog_21k_ice6g_lake_prepare_orography"
                                           "_ICE6G_21k_corrected_20180921_155937.nc",
                                           input_raw_orography_fieldname="Topo",
                                           input_corrected_orography_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/orographys/"
                                           "generated/updated_orog_21k_ice6g_lake_prepare_orography"
                                           "_ICE6G_21k_corrected_20180921_155937.nc",
                                           input_corrected_orography_fieldname="Topo",
                                           input_cell_areas_file="/Users/thomasriddick/Documents/"
                                           "data/HDdata/10min_grid_area_default_R.nc",
                                           input_cell_areas_fieldname="cell_area",
                                           input_prior_fine_rdirs_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/rdirs/generated/"
                                           "updated_RFDs_ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_"
                                           "lsmask_plus_upscale_""rdirs_tarasov_orog_corrs_20171015_031541.nc",
                                           input_prior_fine_rdirs_fieldname="field_value",
                                           input_prior_fine_catchments_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/catchmentmaps/"
                                           "catchmentmap_unsorted_ICE6g_lgm_ALG4_sinkless_no_true_sinks"
                                           "_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs_20171015_031541.nc",
                                           input_prior_fine_catchments_fieldname="field_value",
                                           input_coarse_catchment_nums_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/catchmentmaps/"
                                           "catchmentmap_ICE6g_lgm_ALG4_sinkless_no_true_sinks"
                                           "_oceans_lsmask_plus_upscale_rdirs_tarasov_orog_corrs"
                                           "_20171015_031541_upscaled_updated.nc",
                                           input_coarse_catchment_nums_fieldname="field_value",
                                           input_coarse_rdirs_file=
                                           "/Users/thomasriddick/Documents/data/HDdata/rdirs/generated/"
                                           "updated_RFDs_ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_"
                                           "lsmask_plus_upscale_rdirs_tarasov_orog_corrs_20171015_"
                                           "031541.nc",
                                           input_coarse_rdirs_fieldname="FDIR",
                                           combined_output_filename=
                                            join(self.lake_parameter_file_path,
                                                 "lakeparas" + file_label + ".nc"),
                                           output_filepath=self.lake_parameter_file_path,
                                           output_filelabel=file_label)

    def connect_catchments_for_glac1D(self):
      coarse_catchments_filepath = ("/Users/thomasriddick/Documents/data/HDdata/catchmentmaps/"
                                    "catchmentmap_prepare_basins_from_glac1D_"
                                    "20210205_151552_1250_30mins.nc")
      lake_parameters_filepath = ("/Users/thomasriddick/Documents/data/HDdata/lakeparafiles/"
                                  "lakeparas_prepare_basins_from_glac1D_20210205_151552_1250.nc")
      river_directions_filepath = ("/Users/thomasriddick/Documents/data/HDdata/rdirs/generated/"
                                   "updated_RFDs_prepare_basins_from_glac1D_20210205_151552_1250_30min_with_depressions.nc")
      basin_numbers_filepath = ("/Users/thomasriddick/Documents/data/HDdata/basin_catchment_numbers/"
                                "basin_catchment_numbers_prepare_basins_from_glac1D"
                                "_20210205_151552_1250.nc")
      connected_coarse_catchments_out_filename = ("/Users/thomasriddick/Documents/data/temp/"
                                                  "catchment_1250.nc")
      coarse_catchments_fieldname = "catchments"
      connected_coarse_catchments_out_fieldname = "catchments"
      basin_catchment_numbers_fieldname = "basin_catchment_numbers"
      river_directions_fieldname = "FDIR"
      cclc.connect_coarse_lake_catchments_driver(coarse_catchments_filepath,
                                                 lake_parameters_filepath,
                                                 basin_numbers_filepath,
                                                 river_directions_filepath,
                                                 connected_coarse_catchments_out_filename,
                                                 coarse_catchments_fieldname,
                                                 connected_coarse_catchments_out_fieldname,
                                                 basin_catchment_numbers_fieldname,
                                                 river_directions_fieldname)

    def connect_catchments_for_transient_run(self):
      base_filepath = "/Users/thomasriddick/Documents/data/lake_transient_data/run_1"
      dates = range(15990,10950,-10)
      for date in dates:
        river_directions_filepath = ("{0}/hdpara_{1}k.nc".format(base_filepath,date))
        coarse_catchments_filepath = ("{0}/catchments_{1}.nc".format(base_filepath,date))
        coarse_catchments_fieldname = "catchments"
        cc.advanced_main(filename=river_directions_filepath,
                         fieldname="FDIR",
                         output_filename=coarse_catchments_filepath,
                         output_fieldname="catchments",
                         loop_logfile=("{0}/loops_log_{1}.txt".format(base_filepath,date)),
                         use_cpp_alg=True)
        lake_parameters_filepath = ("{0}/lakepara_{1}k.nc".format(base_filepath,date))
        basin_numbers_filepath = ("{0}/lake_numbers_{1}.nc".format(base_filepath,date))
        connected_coarse_catchments_out_filename = ("{0}/connected_catchments_{1}.nc".\
                                                    format(base_filepath,date))
        connected_coarse_catchments_out_fieldname = "catchments"
        basin_catchment_numbers_fieldname = "lake_number"
        river_directions_fieldname = "FDIR"
        cclc.connect_coarse_lake_catchments_driver(coarse_catchments_filepath,
                                                   lake_parameters_filepath,
                                                   basin_numbers_filepath,
                                                   river_directions_filepath,
                                                   connected_coarse_catchments_out_filename,
                                                   coarse_catchments_fieldname,
                                                   connected_coarse_catchments_out_fieldname,
                                                   basin_catchment_numbers_fieldname,
                                                   river_directions_fieldname)

def main():
    """Select the revelant runs to make

    Select runs by uncommenting them and also the revelant object instantation.
    """
    lake_drivers = Dynamic_Lake_Drivers()
    #lake_drivers.prepare_orography_ICE5G_0k_uncorrected()
    #lake_drivers.prepare_orography_ICE5G_0k_corrected()
    #lake_drivers.prepare_orography_ICE6G_21k_corrected()
    #lake_drivers.prepare_river_directions_with_depressions_from_glac1D()
    #lake_drivers.evaluate_glac1D_ts1900_basins()
    #import time
    # start = time.time()
    #lake_drivers.evaluate_ICE6G_lgm_basins()
    # end = time.time()
    # print(end - start)
    #lake_drivers.prepare_basins_from_glac1D()
    #lake_drivers.extract_lake_volumes_from_glac1D_basins()
    #lake_drivers.connect_catchments_for_glac1D()
    lake_drivers.connect_catchments_for_transient_run()

if __name__ == '__main__':
    main()
