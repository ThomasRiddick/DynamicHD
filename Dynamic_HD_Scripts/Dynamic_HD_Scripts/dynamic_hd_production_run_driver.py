'''
Driving routines for production dynamic HD file generation runs
Created on Mar 18, 2017

@author: thomasriddick
'''

import re
import argparse
import os.path as path
import numpy as np
import utilities
import dynamic_hd
import field
import ConfigParser
import shutil
import libs.fill_sinks_wrapper as fill_sinks_wrapper
import dynamic_hd_driver as dyn_hd_dr
import compute_catchments as comp_catchs
from Dynamic_HD_Scripts.field import Field, RiverDirections
from flow_to_grid_cell import create_hypothetical_river_paths_map
from cotat_plus_driver import run_cotat_plus
from loop_breaker_driver import run_loop_breaker

class Dynamic_HD_Production_Run_Drivers(dyn_hd_dr.Dynamic_HD_Drivers):
    """A class with methods used for running a production run of the dynamic HD generation code"""

    def __init__(self,input_orography_filepath=None,input_ls_mask_filepath=None,output_hdparas_filepath=None,
                 ancillary_data_directory=None,working_directory=None,output_hdstart_filepath=None,
                 present_day_base_orography_filepath=None,glacier_mask_filepath=None):
        """Class constructor.

        Deliberately does NOT call constructor of Dynamic_HD_Drivers so the many paths
        within the data directory structure used for offline runs is not initialized here
        """

        self.original_orography_filename=input_orography_filepath
        self.original_ls_mask_filename=input_ls_mask_filepath
        self.output_hdparas_filepath=output_hdparas_filepath
        self.ancillary_data_path=ancillary_data_directory
        self.working_directory_path=working_directory
        self.output_hdstart_filepath=output_hdstart_filepath
        self.present_day_base_orography_filename=present_day_base_orography_filepath
        self.glacier_mask_filename=glacier_mask_filepath
        self.tarasov_based_orog_correction=True
        if self.ancillary_data_path is not None:
            self.python_config_filename=path.join(self.ancillary_data_path,
                                                  "dynamic_hd_production_driver.cfg")

    def trial_run_using_data_from_new_data_from_virna_2016_version(self):
        """Run a full trial using the 2016 version of the new data from Virna"""
        super(Dynamic_HD_Production_Run_Drivers,self).__init__()
        file_label = self._generate_file_label()
        self.original_orography_filename=path.join(self.orography_path,"10min-topo-present-from-virna.nc")
        self.original_ls_mask_filename=path.join(self.ls_masks_path,"10min-mask-present-from-virna.nc")
        self.output_hdparas_filepath="/Users/thomasriddick/Documents/data/temp/hdpara_{0}.nc".format(file_label)
        self.ancillary_data_path="/Users/thomasriddick/Documents/data/HDancillarydata_original_version"
        self.working_directory_path="/Users/thomasriddick/Documents/data/temp/temp_workdir_2016_data"
        self.output_hdstart_filepath="/Users/thomasriddick/Documents/data/temp/hdstart_{0}.nc".format(file_label)
        self.python_config_filename=path.join("/Users/thomasriddick/Documents/data/HDancillarydata/"
                                              "dynamic_hd_production_driver.cfg")
        self.tarasov_based_orog_correction=False
        self.compile_paragen_and_hdfile()
        self.no_intermediaries_ten_minute_data_ALG4_no_true_sinks_plus_upscale_rdirs_driver()
        return self.output_hdparas_filepath,self.output_hdstart_filepath

    def trial_run_using_data_from_new_data_from_virna_2017_version(self):
        """Run a full trial using the 2017 version of the new data from Virna"""
        super(Dynamic_HD_Production_Run_Drivers,self).__init__()
        file_label = self._generate_file_label()
        self.original_orography_filename=path.join(self.orography_path,"OR-topography-present_data_from_virna_2017.nc")
        self.original_ls_mask_filename=path.join(self.ls_masks_path,"OR-remapped-mask-present_data_from_virna_2017.nc")
        self.output_hdparas_filepath="/Users/thomasriddick/Documents/data/temp/hdpara_{0}.nc".format(file_label)
        self.ancillary_data_path="/Users/thomasriddick/Documents/data/HDancillarydata_original_version"
        self.working_directory_path="/Users/thomasriddick/Documents/data/temp/temp_workdir_2017_data"
        self.output_hdstart_filepath="/Users/thomasriddick/Documents/data/temp/hdstart_{0}.nc".format(file_label)
        self.python_config_filename=path.join("/Users/thomasriddick/Documents/data/HDancillarydata/"
                                              "dynamic_hd_production_driver.cfg")
        self.tarasov_based_orog_correction=False
        self.compile_paragen_and_hdfile()
        self.no_intermediaries_ten_minute_data_ALG4_no_true_sinks_plus_upscale_rdirs_driver()
        return self.output_hdparas_filepath,self.output_hdstart_filepath

    def trial_run_using_data_from_new_data_from_virna_2017_version_2(self):
        """Run a full trial using the second 2017 version of the new data from Virna"""
        super(Dynamic_HD_Production_Run_Drivers,self).__init__()
        file_label = self._generate_file_label()
        self.original_orography_filename=path.join(self.orography_path,
                                                   "OR-topography-present_data_from_virna_13_04_17.nc")
        self.original_ls_mask_filename=path.join(self.ls_masks_path,
                                                 "OR-remapped-mask-present_data_from_virna_13_04_17.nc")
        self.output_hdparas_filepath="/Users/thomasriddick/Documents/data/temp/hdpara_{0}.nc".format(file_label)
        self.ancillary_data_path="/Users/thomasriddick/Documents/data/HDancillarydata_original_version"
        self.working_directory_path="/Users/thomasriddick/Documents/data/temp/temp_workdir_2017_data"
        self.output_hdstart_filepath="/Users/thomasriddick/Documents/data/temp/hdstart_{0}.nc".format(file_label)
        self.python_config_filename=path.join("/Users/thomasriddick/Documents/data/HDancillarydata/"
                                              "dynamic_hd_production_driver.cfg")
        self.tarasov_based_orog_correction=False
        self.compile_paragen_and_hdfile()
        self.no_intermediaries_ten_minute_data_ALG4_no_true_sinks_plus_upscale_rdirs_driver()
        return self.output_hdparas_filepath,self.output_hdstart_filepath

    def trial_run_using_ice6g_present_day_data(self):
        """Run a full trial using present day ICE6G data"""
        super(Dynamic_HD_Production_Run_Drivers,self).__init__()
        file_label = self._generate_file_label()
        self.original_orography_filename=path.join(self.orography_path,
                                                   "Ice6g_c_VM5a_10min_0k.nc")
        self.original_ls_mask_filename=path.join(self.ls_masks_path,
                                                 "10min_ice6g_lsmask_with_disconnected_point_removed_0k.nc")
        self.present_day_base_orography_filename=path.join(self.orography_path,
                                                           "Ice6g_c_VM5a_10min_0k.nc")
        self.glacier_mask_filename=path.join(self.orography_path,
                                             "Ice6g_c_VM5a_10min_0k.nc")
        self.output_hdparas_filepath="/Users/thomasriddick/Documents/data/temp/hdpara_{0}.nc".format(file_label)
        self.ancillary_data_path="/Users/thomasriddick/Documents/data/HDancillarydata"
        self.working_directory_path="/Users/thomasriddick/Documents/data/temp/temp_workdir_2017_data"
        self.output_hdstart_filepath="/Users/thomasriddick/Documents/data/temp/hdstart_{0}.nc".format(file_label)
        self.python_config_filename=path.join("/Users/thomasriddick/Documents/data/HDancillarydata/"
                                              "dynamic_hd_production_driver.cfg")
        self.tarasov_based_orog_correction=True
        self.compile_paragen_and_hdfile()
        self.no_intermediaries_ten_minute_data_ALG4_no_true_sinks_plus_upscale_rdirs_driver()
        return self.output_hdparas_filepath,self.output_hdstart_filepath

    def trial_run_using_ice6g_lgm_day_data(self):
        """Run a full trial using LGM ICE6G data"""
        super(Dynamic_HD_Production_Run_Drivers,self).__init__()
        file_label = self._generate_file_label()
        self.original_orography_filename=path.join(self.orography_path,
                                                   "Ice6g_c_VM5a_10min_21k.nc")
        self.original_ls_mask_filename=path.join(self.ls_masks_path,
                                                 "10min_ice6g_lsmask_with_disconnected_point_removed_21k.nc")
        self.present_day_base_orography_filename=path.join(self.orography_path,
                                                           "Ice6g_c_VM5a_10min_0k.nc")
        self.glacier_mask_filename=path.join(self.orography_path,
                                             "Ice6g_c_VM5a_10min_21k.nc")
        self.output_hdparas_filepath="/Users/thomasriddick/Documents/data/temp/hdpara_{0}.nc".format(file_label)
        self.ancillary_data_path="/Users/thomasriddick/Documents/data/HDancillarydata"
        self.working_directory_path="/Users/thomasriddick/Documents/data/temp/temp_workdir_2017_data"
        self.output_hdstart_filepath="/Users/thomasriddick/Documents/data/temp/hdstart_{0}.nc".format(file_label)
        self.python_config_filename=path.join("/Users/thomasriddick/Documents/data/HDancillarydata/"
                                              "dynamic_hd_production_driver.cfg")
        self.tarasov_based_orog_correction=True
        self.compile_paragen_and_hdfile()
        self.no_intermediaries_ten_minute_data_ALG4_no_true_sinks_plus_upscale_rdirs_driver()
        return self.output_hdparas_filepath,self.output_hdstart_filepath

    def trial_run_using_ice6g_present_day_data_with_specified_fieldnames(self):
        """Run a full trial using present day ICE6G data and specified fieldnames"""
        super(Dynamic_HD_Production_Run_Drivers,self).__init__()
        file_label = self._generate_file_label()
        self.original_orography_filename=path.join(self.orography_path,
                                                   "Ice6g_c_VM5a_10min_0k.nc")
        self.original_ls_mask_filename=path.join(self.ls_masks_path,
                                                 "10min_ice6g_lsmask_with_disconnected_point_removed_0k.nc")
        self.present_day_base_orography_filename=path.join(self.orography_path,
                                                           "Ice6g_c_VM5a_10min_0k.nc")
        self.glacier_mask_filename=path.join(self.orography_path,
                                             "Ice6g_c_VM5a_10min_0k.nc")
        self.output_hdparas_filepath="/Users/thomasriddick/Documents/data/temp/hdpara_{0}.nc".format(file_label)
        self.ancillary_data_path="/Users/thomasriddick/Documents/data/HDancillarydata"
        self.working_directory_path="/Users/thomasriddick/Documents/data/temp/temp_workdir_2017_data"
        self.output_hdstart_filepath="/Users/thomasriddick/Documents/data/temp/hdstart_{0}.nc".format(file_label)
        self.python_config_filename=path.join("/Users/thomasriddick/Documents/data/"
                                              "HDancillarydata_specified_fieldnames/"
                                              "dynamic_hd_production_driver.cfg")
        self.tarasov_based_orog_correction=True
        self.compile_paragen_and_hdfile()
        self.no_intermediaries_ten_minute_data_ALG4_no_true_sinks_plus_upscale_rdirs_driver()
        return self.output_hdparas_filepath,self.output_hdstart_filepath

    def trial_run_using_ice6g_lgm_day_data_with_specified_fieldnames(self):
        """Run a full trial using LGM ICE6G data and specified fieldnames"""
        super(Dynamic_HD_Production_Run_Drivers,self).__init__()
        file_label = self._generate_file_label()
        self.original_orography_filename=path.join(self.orography_path,
                                                   "Ice6g_c_VM5a_10min_21k.nc")
        self.original_ls_mask_filename=path.join(self.ls_masks_path,
                                                 "10min_ice6g_lsmask_with_disconnected_point_removed_21k.nc")
        self.present_day_base_orography_filename=path.join(self.orography_path,
                                                           "Ice6g_c_VM5a_10min_0k.nc")
        self.glacier_mask_filename=path.join(self.orography_path,
                                             "Ice6g_c_VM5a_10min_21k.nc")
        self.output_hdparas_filepath="/Users/thomasriddick/Documents/data/temp/hdpara_{0}.nc".format(file_label)
        self.ancillary_data_path="/Users/thomasriddick/Documents/data/HDancillarydata"
        self.working_directory_path="/Users/thomasriddick/Documents/data/temp/temp_workdir_2017_data"
        self.output_hdstart_filepath="/Users/thomasriddick/Documents/data/temp/hdstart_{0}.nc".format(file_label)
        self.python_config_filename=path.join("/Users/thomasriddick/Documents/data/"
                                              "HDancillarydata_specified_fieldnames/"
                                              "dynamic_hd_production_driver.cfg")
        self.tarasov_based_orog_correction=True
        self.compile_paragen_and_hdfile()
        self.no_intermediaries_ten_minute_data_ALG4_no_true_sinks_plus_upscale_rdirs_driver()
        return self.output_hdparas_filepath,self.output_hdstart_filepath


    def _read_and_validate_config(self):
        """Reads and checks format of config file

        Arguments: None
        Returns: ConfigParser object; the read and checked configuration
        """

        valid_config = True
        config = ConfigParser.ConfigParser()
        print "Read python driver options from file {0}".format(self.python_config_filename)
        config.read(self.python_config_filename)
        valid_config = valid_config \
            if config.has_section("output_options") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_corrected_orog") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_fine_rdirs") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_fine_catchments") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_fine_flowtocell") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_fine_flowtorivermouths") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_pre_loop_removal_course_rdirs") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_pre_loop_removal_course_flowtocell") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_pre_loop_removal_course_flowtorivermouths") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_pre_loop_removal_course_catchments") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_course_rdirs") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_course_unfilled_orog") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_course_filled_orog") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_course_flowtocell") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_course_flowtorivermouths") else False
        valid_config = valid_config \
            if config.has_option("output_options","output_course_catchments") else False
        if not valid_config:
            raise RuntimeError("Invalid configuration file supplied")
        if not config.has_section("input_fieldname_options"):
            config.add_section("input_fieldname_options")
        if not config.has_option("input_fieldname_options","input_orography_fieldname"):
            config.set("input_fieldname_options","input_orography_fieldname","")
        if not config.has_option("input_fieldname_options","input_landsea_mask_fieldname"):
            config.set("input_fieldname_options","input_landsea_mask_fieldname","")
        if not config.has_option("input_fieldname_options","input_glacier_mask_fieldname"):
            config.set("input_fieldname_options","input_glacier_mask_fieldname","")
        if not config.has_option("input_fieldname_options","input_base_present_day_orography_fieldname"):
            config.set("input_fieldname_options","input_base_present_day_orography_fieldname","")
        if not config.has_section("general_options"):
            config.add_section("general_options")
        if not config.has_option("general_options","generate_flow_parameters"):
            config.set("general_options","generate_flow_parameters","True")
        return config

    def no_intermediaries_ten_minute_data_ALG4_no_true_sinks_plus_upscale_rdirs_driver(self):
        """Generates and upscales sinkless river direction from a initial 10 minute orography and landsea mask

        Arguments: None
        Returns: nothing
        """

        config = self._read_and_validate_config()
        base_hd_restart_file = path.join(self.ancillary_data_path,"hd_restart_from_hd_file_ten_minute_data_from_virna_"
                                        "0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_"
                                        "135934_after_one_year_running.nc")
        ref_hd_paras_file = path.join(self.ancillary_data_path,"hd_file_ten_minute_data_from_virna_0k_ALG4_sinkless_no_"
                                      "true_sinks_oceans_lsmask_plus_upscale_rdirs_20170116_123858_to_use_as_"
                                      "hdparas_ref.nc")
        if self.present_day_base_orography_filename:
            present_day_reference_orography_filename = path.join(self.ancillary_data_path,
                                                                 "ice5g_v1_2_00_0k_10min.nc")
        if self.tarasov_based_orog_correction:
            orography_corrections_filename = path.join(self.ancillary_data_path,
                                                       "orog_corrs_field_ICE5G_and_tarasov_upscaled_"
                                                       "srtm30plus_north_america_only_data_ALG4_sinkless"
                                                       "_glcc_olson_lsmask_0k_20170517_003802.nc")
        else:
            #This is deprecated; don't use except for in unit testing. This file will NOT be present
            #in the folder provided for online running
            orography_corrections_filename = path.join(self.ancillary_data_path,
                                                       "orog_corrs_field_ICE5G_data_ALG4_sink"
                                                       "less_downscaled_ls_mask_0k_20160930_001057.nc")
        #Change ls mask to correct type
        ls_mask_10min_fieldname = config.get("input_fieldname_options",
                                             "input_landsea_mask_fieldname")
        ls_mask_10min_fieldname = ls_mask_10min_fieldname if ls_mask_10min_fieldname else None
        ls_mask_10min = dynamic_hd.load_field(self.original_ls_mask_filename,
                                              file_type=\
                                              dynamic_hd.get_file_extension(self.original_ls_mask_filename),
                                              field_type='Generic',
                                              fieldname=ls_mask_10min_fieldname,
                                              unmask=False,
                                              timeslice=None,
                                              grid_type='LatLong10min')
        ls_mask_10min.change_dtype(np.int32)
        #Add corrections to orography
        orography_10min_fieldname = config.get("input_fieldname_options",
                                               "input_orography_fieldname")
        orography_10min_fieldname = orography_10min_fieldname if orography_10min_fieldname else None
        orography_10min = dynamic_hd.load_field(self.original_orography_filename,
                                                file_type=dynamic_hd.\
                                                get_file_extension(self.original_orography_filename),
                                                fieldname=orography_10min_fieldname,
                                                field_type='Orography', grid_type="LatLong10min")
        if self.present_day_base_orography_filename:
            present_day_base_orography_fieldname = config.get("input_fieldname_options",
                                                              "input_base_present_day_orography_fieldname")
            present_day_base_orography_fieldname = present_day_base_orography_fieldname if \
                                                   present_day_base_orography_fieldname else None
            present_day_base_orography = \
            dynamic_hd.load_field(self.present_day_base_orography_filename,
                                  file_type=dynamic_hd.\
                                  get_file_extension(self.present_day_base_orography_filename),
                                  field_type='Orography',
                                  fieldname=present_day_base_orography_fieldname,
                                  grid_type="LatLong10min")
            present_day_reference_orography = \
            dynamic_hd.load_field(present_day_reference_orography_filename,
                                  file_type=dynamic_hd.\
                                  get_file_extension(present_day_reference_orography_filename),
                                  field_type='Orography',
                                  grid_type="LatLong10min")
            orography_10min = utilities.rebase_orography(orography=orography_10min,
                                                         present_day_base_orography=\
                                                         present_day_base_orography,
                                                         present_day_reference_orography=\
                                                         present_day_reference_orography)
        orography_corrections_10min =  dynamic_hd.load_field(orography_corrections_filename,
                                                             file_type=dynamic_hd.\
                                                             get_file_extension(orography_corrections_filename),
                                                             field_type='Orography', grid_type="LatLong10min")
        orography_uncorrected_10min = orography_10min.copy()
        orography_10min.add(orography_corrections_10min)
        if self.glacier_mask_filename:
            glacier_mask_10min_fieldname = config.get("input_fieldname_options",
                                                "input_glacier_mask_fieldname")
            glacier_mask_10min_fieldname = glacier_mask_10min_fieldname if glacier_mask_10min_fieldname else 'sftgif'
            glacier_mask_10min = dynamic_hd.load_field(self.glacier_mask_filename,
                                                       file_type=dynamic_hd.\
                                                       get_file_extension(self.glacier_mask_filename),
                                                       fieldname=glacier_mask_10min_fieldname,
                                                       field_type='Orography',
                                                       unmask=True,grid_type="LatLong10min")
            orography_10min = utilities.\
            replace_corrected_orography_with_original_for_glaciated_grid_points(input_corrected_orography=\
                                                                                orography_10min,
                                                                                input_original_orography=\
                                                                                orography_uncorrected_10min,
                                                                                input_glacier_mask=
                                                                                glacier_mask_10min)
        if config.getboolean("output_options","output_corrected_orog"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "10min_corrected_orog.nc"),
                                   orography_10min,
                                   file_type=".nc")
        #Fill sinks
        grid_dims_10min=orography_10min.get_grid().get_grid_dimensions()
        rdirs_10min = RiverDirections(np.zeros(grid_dims_10min,dtype=np.float64,order='C'),grid='LatLong10min')
        truesinks = Field(np.empty((1,1),dtype=np.int32),grid='HD')
        catchment_10min = Field(np.zeros(grid_dims_10min,dtype=np.int32,order='C'),grid='LatLong10min')
        next_cell_lat_index_in_10min = np.zeros(grid_dims_10min,dtype=np.int32,order='C')
        next_cell_lon_index_in_10min = np.zeros(grid_dims_10min,dtype=np.int32,order='C')
        ls_mask_10min.flip_data_ud()
        orography_10min.flip_data_ud()
        fill_sinks_wrapper.fill_sinks_cpp_func(orography_array=np.ascontiguousarray(orography_10min.get_data(), #@UndefinedVariable
                                                                                    dtype=np.float64),
                                               method = 4,
                                               use_ls_mask = True,
                                               landsea_in = np.ascontiguousarray(ls_mask_10min.get_data(),
                                                                                 dtype=np.int32),
                                               set_ls_as_no_data_flag = False,
                                               use_true_sinks = False,
                                               true_sinks_in = np.ascontiguousarray(truesinks.get_data(),
                                                                                    dtype=np.int32),
                                               next_cell_lat_index_in = next_cell_lat_index_in_10min,
                                               next_cell_lon_index_in = next_cell_lon_index_in_10min,
                                               rdirs_in = rdirs_10min.get_data(),
                                               catchment_nums_in = catchment_10min.get_data(),
                                               prefer_non_diagonal_initial_dirs = False)
        if config.getboolean("output_options","output_fine_rdirs"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "10min_rdirs.nc"),
                                   rdirs_10min,
                                   file_type=".nc")
        if config.getboolean("output_options","output_fine_catchments"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "10min_catchments.nc"),
                                   catchment_10min,
                                   file_type=".nc")
        #Run post processing
        nlat10,nlong10 = grid_dims_10min
        flowtocell_10min = field.CumulativeFlow(create_hypothetical_river_paths_map(riv_dirs=rdirs_10min.get_data(),
                                                                                    lsmask=ls_mask_10min.get_data(),
                                                                                    use_f2py_func=True,
                                                                                    use_f2py_sparse_iterator=True,
                                                                                    nlat=nlat10,
                                                                                    nlong=nlong10),
                                                                                    grid='LatLong10min')
        rdirs_10min.mark_river_mouths(ls_mask_10min.get_data())
        rivermouths_10min = field.makeField(rdirs_10min.get_river_mouths(),'Generic','LatLong10min')
        flowtorivermouths_10min = field.makeField(flowtocell_10min.\
                                                  find_cumulative_flow_at_outlets(rivermouths_10min.\
                                                                                  get_data()),
                                                  'Generic','LatLong10min')
        if config.getboolean("output_options","output_fine_flowtocell"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "10min_flowtocell.nc"),
                                   flowtocell_10min,
                                   file_type=".nc")
        if config.getboolean("output_options","output_fine_flowtorivermouths"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "10min_flowtorivermouths.nc"),
                                   flowtorivermouths_10min,
                                   file_type=".nc")
        #Run Upscaling
        loops_log_filename = path.join(self.working_directory_path,"loops.log")
        catchments_log_filename= path.join(self.working_directory_path,"catchments.log")
        cotat_plus_parameters_filename = path.join(self.ancillary_data_path,'cotat_plus_standard_params.nl')
        rdirs_30min = run_cotat_plus(rdirs_10min, flowtocell_10min,
                                      cotat_plus_parameters_filename,'HD')
        if config.getboolean("output_options","output_pre_loop_removal_course_rdirs"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_pre_loop_removal_rdirs.nc"),
                                   rdirs_30min,
                                   file_type=".nc")
        #Post processing
        nlat30,nlong30 = rdirs_30min.get_grid().get_grid_dimensions()
        flowtocell_30min = field.CumulativeFlow(create_hypothetical_river_paths_map(riv_dirs=rdirs_30min.get_data(),
                                                                                    lsmask=None,
                                                                                    use_f2py_func=True,
                                                                                    use_f2py_sparse_iterator=True,
                                                                                    nlat=nlat30,
                                                                                    nlong=nlong30),
                                                                                    grid='HD')
        catchment_types_30min, catchments_30min = comp_catchs.compute_catchments(rdirs_30min.get_data(),
                                                                                 loops_log_filename)
        comp_catchs.check_catchment_types(catchment_types_30min,
                                          logfile=catchments_log_filename)
        catchments_30min = field.Field(comp_catchs.renumber_catchments_by_size(catchments_30min,
                                                                               loops_log_filename),
                                       grid="HD")
        rivermouths_30min = field.makeField(rdirs_30min.get_river_mouths(),'Generic','HD')
        flowtorivermouths_30min = field.makeField(flowtocell_30min.\
                                                  find_cumulative_flow_at_outlets(rivermouths_30min.\
                                                                                  get_data()),
                                                  'Generic','HD')
        if config.getboolean("output_options","output_pre_loop_removal_course_flowtocell"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_pre_loop_removal_flowtocell.nc"),
                                   flowtocell_30min,
                                   file_type=".nc")
        if config.getboolean("output_options","output_pre_loop_removal_course_flowtorivermouths"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_pre_loop_removal_flowtorivermouths.nc"),
                                   flowtorivermouths_30min,
                                   file_type=".nc")
        if config.getboolean("output_options","output_pre_loop_removal_course_catchments"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_pre_loop_removal_catchments.nc"),
                                   catchments_30min,
                                   file_type=".nc")
        #Run Loop Breaker
        loop_nums_list = []
        first_line_pattern = re.compile(r"^Loops found in catchments:$")
        with open(loops_log_filename,'r') as f:
            if not first_line_pattern.match(f.readline().strip()):
                raise RuntimeError("Format of the file with list of catchments to remove loops from"
                                   " is invalid")
            for line in f:
                loop_nums_list.append(int(line.strip()))
        print 'Removing loops from catchments: ' + ", ".join(str(value) for value in loop_nums_list)
        rdirs_30min = run_loop_breaker(rdirs_30min,flowtocell_30min,
                                       catchments_30min,rdirs_10min,
                                       flowtocell_10min,loop_nums_list,
                                       course_grid_type="HD")
        if config.getboolean("output_options","output_course_rdirs"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_rdirs.nc"),
                                   rdirs_30min,
                                   file_type=".nc")
        #Upscale the orography to the HD grid for calculating the flow parameters
        orography_30min= utilities.upscale_field(orography_10min,"HD","Sum",{},
                                                 scalenumbers=True)
        if config.getboolean("output_options","output_course_unfilled_orog"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_unfilled_orog.nc"),
                                   orography_30min,
                                   file_type=".nc")
        #Extract HD ls mask from river directions
        ls_mask_30min = field.RiverDirections(rdirs_30min.get_lsmask(),grid='HD')
        #Fill HD orography for parameter generation
        truesinks = Field(np.empty((1,1),dtype=np.int32),grid='HD')
        fill_sinks_wrapper.fill_sinks_cpp_func(orography_array=np.ascontiguousarray(orography_30min.get_data(), #@UndefinedVariable
                                                                                    dtype=np.float64),
                                               method = 1,
                                               use_ls_mask = True,
                                               landsea_in = np.ascontiguousarray(ls_mask_30min.get_data(),
                                                                             dtype=np.int32),
                                               set_ls_as_no_data_flag = False,
                                               use_true_sinks = False,
                                               true_sinks_in = np.ascontiguousarray(truesinks.get_data(),
                                                                                dtype=np.int32),
                                               add_slope = False,
                                               epsilon = 0.0)
        if config.getboolean("output_options","output_course_filled_orog"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_filled_orog.nc"),
                                   orography_30min,
                                   file_type=".nc")
        #Transform any necessary field into the necessary format and save ready for parameter generation
        transformed_course_rdirs_filename = path.join(self.working_directory_path,"30minute_river_dirs_temp.nc")
        transformed_HD_filled_orography_filename = path.join(self.working_directory_path,"30minute_filled_orog_temp.nc")
        transformed_HD_ls_mask_filename = path.join(self.working_directory_path,"30minute_ls_mask_temp.nc")
        half_degree_grid_filepath = path.join(self.ancillary_data_path,"grid_0_5.txt")
        rdirs_30min.rotate_field_by_a_hundred_and_eighty_degrees()
        dynamic_hd.write_field(filename=transformed_course_rdirs_filename,
                               field=rdirs_30min,
                               file_type=\
                               dynamic_hd.get_file_extension(transformed_course_rdirs_filename),
                               griddescfile=half_degree_grid_filepath)
        orography_30min.rotate_field_by_a_hundred_and_eighty_degrees()
        dynamic_hd.write_field(filename=transformed_HD_filled_orography_filename,
                               field=orography_30min,
                               file_type=dynamic_hd.\
                               get_file_extension(transformed_HD_filled_orography_filename),
                               griddescfile=half_degree_grid_filepath)
        ls_mask_30min.rotate_field_by_a_hundred_and_eighty_degrees()
        ls_mask_30min.invert_data()
        dynamic_hd.write_field(filename=transformed_HD_ls_mask_filename,
                               field=ls_mask_30min,
                               file_type=dynamic_hd.\
                               get_file_extension(transformed_HD_ls_mask_filename),
                               griddescfile=half_degree_grid_filepath)
        #If required generate flow parameters and create a hdparas.file otherwise
        #use a river direction file with coordinates as the hdparas.file
        if config.getboolean("general_options","generate_flow_parameters"):
            #Generate parameters
            self._generate_flow_parameters(rdir_file=transformed_course_rdirs_filename,
                                           topography_file=transformed_HD_filled_orography_filename,
                                           inner_slope_file=\
                                           path.join(self.ancillary_data_path,'bin_innerslope.dat'),
                                           lsmask_file=transformed_HD_ls_mask_filename,
                                           null_file=\
                                           path.join(self.ancillary_data_path,'null.dat'),
                                           area_spacing_file=\
                                           path.join(self.ancillary_data_path,
                                                     'fl_dp_dl.dat'),
                                           orography_variance_file=\
                                           path.join(self.ancillary_data_path,'bin_toposig.dat'),
                                           output_dir=path.join(self.working_directory_path,'paragen'),
                                           production_run=True)
            #Place parameters and rdirs into a hdparas.file
            self._generate_hd_file(rdir_file=path.splitext(transformed_course_rdirs_filename)[0] + ".dat",
                                   lsmask_file=path.splitext(transformed_HD_ls_mask_filename)[0] + ".dat",
                                   null_file=\
                                   path.join(self.ancillary_data_path,'null.dat'),
                                   area_spacing_file=\
                                   path.join(self.ancillary_data_path,
                                             'fl_dp_dl.dat'),
                                   hd_grid_specs_file=half_degree_grid_filepath,
                                   output_file=self.output_hdparas_filepath,
                                   paras_dir=path.join(self.working_directory_path,'paragen'),
                                   production_run=True)
        else:
            #Use a river direction file including coordinates as the hdparas file
            shutil.copy2(transformed_course_rdirs_filename,self.output_hdparas_filepath)
        if self.output_hdstart_filepath is not None:
            utilities.prepare_hdrestart_file_driver(base_hdrestart_filename=base_hd_restart_file,
                                                    output_hdrestart_filename=\
                                                    self.output_hdstart_filepath,
                                                    hdparas_filename=self.output_hdparas_filepath,
                                                    ref_hdparas_filename=ref_hd_paras_file,
                                                    timeslice=None,
                                                    res_num_data_rotate180lr=False,
                                                    res_num_data_flipup=False,
                                                    res_num_ref_rotate180lr=False,
                                                    res_num_ref_flipud=False, grid_type='HD')
        #Post processing
        flowtocell_30min = field.CumulativeFlow(create_hypothetical_river_paths_map(riv_dirs=rdirs_30min.get_data(),
                                                                                    lsmask=None,
                                                                                    use_f2py_func=True,
                                                                                    use_f2py_sparse_iterator=True,
                                                                                    nlat=nlat30,
                                                                                    nlong=nlong30),
                                                                                    grid='HD')
        catchment_types_30min, catchments_30min = comp_catchs.compute_catchments(rdirs_30min.get_data(),
                                                                                 loops_log_filename)
        comp_catchs.check_catchment_types(catchment_types_30min,
                                          logfile=catchments_log_filename)
        catchments_30min = field.Field(comp_catchs.renumber_catchments_by_size(catchments_30min,
                                                                               loops_log_filename),
                                       grid="HD")
        rivermouths_30min = field.makeField(rdirs_30min.get_river_mouths(),'Generic','HD')
        flowtorivermouths_30min = field.makeField(flowtocell_30min.\
                                                  find_cumulative_flow_at_outlets(rivermouths_30min.\
                                                                                  get_data()),
                                                  'Generic','HD')
        if config.getboolean("output_options","output_course_flowtocell"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_flowtocell.nc"),
                                   flowtocell_30min,
                                   file_type=".nc")
        if config.getboolean("output_options","output_course_flowtorivermouths"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_flowtorivermouths.nc"),
                                   flowtorivermouths_30min,
                                   file_type=".nc")
        if config.getboolean("output_options","output_course_catchments"):
            dynamic_hd.write_field(path.join(self.working_directory_path,
                                             "30min_catchments.nc"),
                                   catchments_30min,
                                   file_type=".nc")

def setup_and_run_dynamic_hd_para_gen_from_command_line_arguments(args):
    """Setup and run a dynamic hd production run from the command line arguments passed in by main"""
    driver_object = Dynamic_HD_Production_Run_Drivers(**vars(args))
    driver_object.no_intermediaries_ten_minute_data_ALG4_no_true_sinks_plus_upscale_rdirs_driver()

class Arguments(object):
    """An empty class used to pass namelist arguments into the main routine as keyword arguments."""

    pass

def parse_arguments():
    """Parse the command line arguments using the argparse module.

    Returns:
    An Arguments object containing the comannd line arguments.
    """

    args = Arguments()
    parser = argparse.ArgumentParser("Update river flow directions")
    parser.add_argument('input_orography_filepath',
                        metavar='input-orography-filepath',
                        help='Full path to input orography to use',
                        type=str)
    parser.add_argument('input_ls_mask_filepath',
                        metavar='input-ls-mask-filepath',
                        help='Full path to input land sea mask to use',
                        type=str)
    parser.add_argument('present_day_base_orography_filepath',
                        metavar='present-day-base-orography-filepath',
                        help='Full path to present day orography input orography is based on',
                        type=str)
    parser.add_argument('glacier_mask_filepath',
                        metavar='glacier-mask-filepath',
                        help='Full path to input glacier mask file',
                        type=str)
    parser.add_argument('output_hdparas_filepath',
                        metavar='output-hdparas-filepath',
                        help='Full path to target destination for output hdparas file',
                        type=str)
    parser.add_argument('ancillary_data_directory',
                        metavar='ancillary-data-directory',
                        help='Full path to directory containing ancillary data',
                        type=str)
    parser.add_argument('working_directory',
                        metavar='working-directory',
                        help='Full path to target working directory',
                        type=str)
    parser.add_argument('-s','--output-hdstart-filepath',
                        help='Full path to target destination for output hdstart file',
                        type=str,
                        default=None)
    #Adding the variables to a namespace other than that of the parser keeps the namespace clean
    #and allows us to pass it directly to main
    parser.parse_args(namespace=args)
    return args

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_dynamic_hd_para_gen_from_command_line_arguments(args)
