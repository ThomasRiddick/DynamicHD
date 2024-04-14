'''
Created on April 1, 2020

@author: thomasriddick
'''
import re
import os
import os.path as path
import configparser
import numpy as np
import cdo
import argparse
import shutil
from timeit import default_timer as timer
from Dynamic_HD_Scripts.base import field
from Dynamic_HD_Scripts.base import grid
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.tools import determine_river_directions
from Dynamic_HD_Scripts.tools.loop_breaker_driver import run_loop_breaker
from Dynamic_HD_Scripts.tools.cotat_plus_driver import run_cotat_plus
from Dynamic_HD_Scripts.tools import dynamic_lake_operators
from Dynamic_HD_Scripts.tools import compute_catchments as comp_catchs
from Dynamic_HD_Scripts.tools.flow_to_grid_cell import create_hypothetical_river_paths_map
from Dynamic_HD_Scripts.tools import extract_lake_volumes
from Dynamic_HD_Scripts.tools import connect_coarse_lake_catchments as cclc
from Dynamic_HD_Scripts.tools import river_mouth_marking_driver
from Dynamic_HD_Scripts.utilities import utilities
from Dynamic_HD_Scripts.dynamic_hd_and_dynamic_lake_drivers \
    import dynamic_hd_driver as dyn_hd_dr
import fill_sinks_wrapper
import lake_operators_wrapper
import evaluate_basins_wrapper

class Dynamic_Lake_Production_Run_Drivers(dyn_hd_dr.Dynamic_HD_Drivers):
    """A class with methods used for running a production run of the dynamic HD and Lake generation code"""

    #To prevent strange edge effects occuring this should be set to a large
    #negative value (larger than any plausible negative height in the model)
    ocean_floor_depth = -1.0E9

    def __init__(self,input_orography_filepath=None,input_ls_mask_filepath=None,
                 input_water_to_redistribute_filepath=None,
                 output_hdparas_filepath=None,output_lakeparas_filepath=None,
                 output_lakestart_filepath=None,ancillary_data_directory=None,
                 working_directory=None,output_hdstart_filepath=None,
                 present_day_base_orography_filepath=None,glacier_mask_filepath=None,
                 non_standard_orog_correction_filename=None,
                 date_based_sill_height_corrections_list_filename=None,
                 current_date=None,
                 additional_orography_corrections_filepath=None):
        """Class constructor.

        Deliberately does NOT call constructor of Dynamic_HD_Drivers so the many paths
        within the data directory structure used for offline runs is not initialized here
        """

        self.original_orography_filename=input_orography_filepath
        self.original_ls_mask_filename=input_ls_mask_filepath
        self.input_water_to_redistribute_filepath=input_water_to_redistribute_filepath
        self.output_hdparas_filepath=output_hdparas_filepath
        self.ancillary_data_path=ancillary_data_directory
        self.working_directory_path=working_directory
        self.output_hdstart_filepath=output_hdstart_filepath
        self.output_lakeparas_filepath = output_lakeparas_filepath
        self.output_lakestart_filepath = output_lakestart_filepath
        self.present_day_base_orography_filename=present_day_base_orography_filepath
        self.glacier_mask_filename=glacier_mask_filepath
        self.tarasov_based_orog_correction=True
        self.non_standard_orog_correction_filename=non_standard_orog_correction_filename
        self.date_based_sill_height_corrections_list_filename = \
             date_based_sill_height_corrections_list_filename
        self.current_date = current_date
        self.additional_orography_corrections_filename = \
             additional_orography_corrections_filepath
        if self.ancillary_data_path is not None:
            self.python_config_filename=path.join(self.ancillary_data_path,
                                                  "dynamic_lake_production_driver.cfg")

    def store_diagnostics(self,dest):
        shutil.move(path.join(self.working_directory_path,"10min_corrected_orog.nc"),
                    path.join(dest,"10min_corrected_orog.nc"))
        shutil.move(path.join(self.working_directory_path,"10min_rdirs.nc"),
                    path.join(dest,"10min_rdirs.nc"))
        shutil.move(path.join(self.working_directory_path,"10min_filled_orog.nc"),
                    path.join(dest,"10min_filled_orog.nc"))
        shutil.move(path.join(self.working_directory_path,"10min_catchments.nc"),
                    path.join(dest,"10min_catchments.nc"))
        shutil.move(path.join(self.working_directory_path,"10min_flowtorivermouths.nc"),
                    path.join(dest,"10min_flowtorivermouths.nc"))
        shutil.move(path.join(self.working_directory_path,"10min_flowtocell.nc"),
                    path.join(dest,"10min_flowtocell.nc"))
        shutil.move(path.join(self.working_directory_path,"30min_unfilled_orog.nc"),
                    path.join(dest,"30min_unfilled_orog.nc"))
        shutil.move(path.join(self.working_directory_path,"30min_rdirs.nc"),
                    path.join(dest,"30min_rdirs.nc"))
        shutil.move(path.join(self.working_directory_path,
                            "30min_pre_loop_removal_rdirs.nc"),
                    path.join(dest,"30min_pre_loop_removal_rdirs.nc"))
        shutil.move(path.join(self.working_directory_path,
                            "30min_pre_loop_removal_flowtorivermouths.nc"),
                    path.join(dest,"30min_pre_loop_removal_flowtorivermouths.nc"))
        shutil.move(path.join(self.working_directory_path,
                            "30min_pre_loop_removal_flowtocell.nc"),
                    path.join(dest,"30min_pre_loop_removal_flowtocell.nc"))
        shutil.move(path.join(self.working_directory_path,
                            "30min_pre_loop_removal_catchments.nc"),
                    path.join(dest,"30min_pre_loop_removal_catchments.nc"))
        shutil.move(path.join(self.working_directory_path,
                            "30min_filled_orog.nc"),
                    path.join(dest,"30min_filled_orog.nc"))
        shutil.move(path.join(self.working_directory_path,
                            "30min_flowtorivermouths.nc"),
                    path.join(dest,"30min_flowtorivermouths.nc"))
        shutil.move(path.join(self.working_directory_path,
                            "30min_flowtocell.nc"),
                    path.join(dest,"30min_flowtocell.nc"))
        shutil.move(path.join(self.working_directory_path,
                            "30min_catchments.nc"),
                    path.join(dest,"30min_catchments.nc"))
        shutil.move(path.join(self.working_directory_path,
                             "10min_basin_catchment_numbers.nc"),
                    path.join(dest,"10min_basin_catchment_numbers.nc"))
        shutil.move(path.join(self.working_directory_path,
                              "10min_sinkless_rdirs.nc"),
                    path.join(dest,"10min_sinkless_rdirs.nc"))
        shutil.move(path.join(self.working_directory_path,
                             "10min_lake_volumes.nc"),
                    path.join(dest,"10min_lake_volumes.nc"))
        shutil.move(path.join(self.working_directory_path,
                            "30min_flowtocell_connected.nc"),
                   path.join(dest,"30min_flowtocell_connected.nc"))
        shutil.move(path.join(self.working_directory_path,
                             "30min_connected_catchments.nc"),
                    path.join(dest,"30min_connected_catchments.nc"))
        shutil.move(path.join(self.working_directory_path,
                             "30min_flowtorivermouths_connected.nc"),
                    path.join(dest,"30min_flowtorivermouths_connected.nc"))

    def clean_work_dir(self):
        os.remove(path.join(self.working_directory_path,"30minute_river_dirs_temp.nc"))
        os.remove(path.join(self.working_directory_path,"30minute_filled_orog_temp.nc"))
        os.remove(path.join(self.working_directory_path,"30minute_river_dirs_temp.dat"))
        os.remove(path.join(self.working_directory_path,"30minute_ls_mask_temp.nc"))
        os.remove(path.join(self.working_directory_path,"30minute_ls_mask_temp.dat"))
        os.remove(path.join(self.working_directory_path,"30minute_filled_orog_temp.dat"))
        os.remove(path.join(self.working_directory_path,"loops_10min.log"))
        os.remove(path.join(self.working_directory_path,"loops_30min.log"))
        os.remove(path.join(self.working_directory_path,"paragen",
                            "soil_partab.txt"))
        os.remove(path.join(self.working_directory_path,"paragen",
                            "slope.dat"))
        os.remove(path.join(self.working_directory_path,"paragen",
                            "riv_vel.dat"))
        os.remove(path.join(self.working_directory_path,"paragen",
                            "riv_n.dat"))
        os.remove(path.join(self.working_directory_path,"paragen",
                            "riv_k.dat"))
        os.remove(path.join(self.working_directory_path,"paragen",
                            "paragen.inp"))
        os.remove(path.join(self.working_directory_path,"paragen",
                            "over_vel.dat"))
        os.remove(path.join(self.working_directory_path,"paragen",
                            "over_n.dat"))
        os.remove(path.join(self.working_directory_path,"paragen",
                            "over_k.dat"))
        os.remove(path.join(self.working_directory_path,"paragen",
                            "hdpara.srv"))
        os.remove(path.join(self.working_directory_path,"paragen",
                            "global.inp"))
        os.remove(path.join(self.working_directory_path,"paragen",
                            "ddir.inp"))
        os.remove(path.join(self.working_directory_path,"paragen",
                            "bas_k.dat"))
        os.rmdir(path.join(self.working_directory_path,"paragen"))

    def _check_config_section_is_valid(self,config,section_name):
        if not config.has_section(section_name):
            raise RuntimeError("Invalid configuration file supplied"
                               " - section {} missing".format(section_name))

    def _check_config_option_is_valid(self,config,section_name,option_name):
        if not config.has_option(section_name,option_name):
            raise RuntimeError("Invalid configuration file supplied"
                               " - option {}:{} missing".format(section_name,
                                                                option_name))

    def _read_and_validate_config(self):
        """Reads and checks format of config file

        Arguments: None
        Returns: ConfigParser object; the read and checked configuration
        """

        config = configparser.ConfigParser()
        print("Read python driver options from file {0}".format(self.python_config_filename))
        config.read(self.python_config_filename)
        self._check_config_section_is_valid(config,"input_fieldname_options")
        self._check_config_option_is_valid(config,
                                          "input_fieldname_options",
                                          "input_orography_fieldname")
        self._check_config_option_is_valid(config,
                                          "input_fieldname_options",
                                          "input_landsea_mask_fieldname")
        self._check_config_option_is_valid(config,
                                          "input_fieldname_options",
                                          "input_glacier_mask_fieldname")
        self._check_config_option_is_valid(config,
                                          "input_fieldname_options",
                                          "input_base_present_day_orography_fieldname")
        self._check_config_option_is_valid(config,
                                          "input_fieldname_options",
                                          "input_reference_present_day_fieldname")
        self._check_config_option_is_valid(config,
                                          "input_fieldname_options",
                                          "input_orography_corrections_fieldname")
        self._check_config_option_is_valid(config,
                                          "input_fieldname_options",
                                          "input_surface_lat_index_fieldname")
        self._check_config_option_is_valid(config,
                                          "input_fieldname_options",
                                          "input_surface_lon_index_fieldname")
        if not config.has_section("general_options"):
            config.add_section("general_options")
        if not config.has_option("general_options","generate_flow_parameters"):
            config.set("general_options","generate_flow_parameters","True")
        if not config.has_option("general_options","print_timing_information"):
            config.set("general_options","print_timing_information","False")
        if not config.has_option("general_options","use_gradual_transitions"):
            config.set("general_options","use_gradual_transitions","True")

        output_options = \
            { "output_10min_corrected_orog_fieldname":"corrected_orog",
              "output_10min_filled_orog_fieldname":"filled_orog",
              "output_10min_rdirs_fieldname":"rdirs",
              "output_10min_flow_to_cell":"cumulative_flow",
              "output_10min_flow_to_river_mouths":"cumulative_flow_to_ocean",
              "output_10min_catchments":"catchments",
              "output_30min_pre_loop_removal_rdirs":"rdirs",
              "output_30min_pre_loop_removal_flow_to_cell":"cumulative_flow",
              "output_30min_pre_loop_removal_flow_to_river_mouth":"cumulative_flow_to_ocean",
              "output_30min_pre_loop_removal_catchments":"catchments",
              "output_30min_rdirs":"rdirs",
              "output_30min_unfilled_orog":"unfilled_orog",
              "output_30min_filled_orog":"filled_orog",
              "output_30min_ls_mask":"lsmask",
              "output_30min_flow_to_cell":"cumulative_flow",
              "output_30min_flow_to_river_mouths":"cumulative_flow_to_ocean",
              "output_30min_catchments":"catchments",
              "output_10min_sinkless_rdirs":"rdirs",
              "output_10min_basin_catchment_numbers":"basin_catchment_numbers" }
        self._check_config_section_is_valid(config,"output_options")
        if not config.has_section("output_fieldname_options"):
            config.add_section("output_fieldname_options")
        #This is exception as it doesn't take a fieldname
        if not config.has_option("output_options","output_10min_lake_volumes"):
            config.set("output_options","output_10min_lake_volumes","False")
        for key,value in output_options.items():
            if not config.has_option("output_options",key):
                config.set("output_options",key,"False")
            if not config.has_option("output_fieldname_options",key):
                config.set("output_fieldname_options",key,value)
        return config

    def trial_run_for_present_day(self):
        super(Dynamic_Lake_Production_Run_Drivers,self).__init__()
        file_label = self._generate_file_label()
        self.original_orography_filename=path.join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        self.original_ls_mask_filename=("/Users/thomasriddick/Documents/data/unit_test_data/"
                                        "ls_mask_no_intermediaries_lake_corrections_"
                                        "driver_20200726_181304_grid.nc")
        self.output_hdparas_filepath="/Users/thomasriddick/Documents/data/temp/hdpara_{0}_pd.nc".format(file_label)
        self.output_hdstart_filepath="/Users/thomasriddick/Documents/data/temp/hdstart_{0}_pd.nc".format(file_label)
        self.ancillary_data_path="/Users/thomasriddick/Documents/data/hd_ancillary_data_dirs/HDancillarydata_lakes"
        self.working_directory_path="/Users/thomasriddick/Documents/data/temp/temp_workdir_pd"
        self.output_lakeparas_filepath = "/Users/thomasriddick/Documents/data/temp/lakeparas_{0}_pd.nc".format(file_label)
        self.output_lakestart_filepath = "/Users/thomasriddick/Documents/data/temp/lakestart_{0}_pd.nc".format(file_label)
        self.input_water_to_redistribute_filepath=("/Users/thomasriddick/Documents/data/"
                                                   "simulation_data/laketestdata/lake_volumes_pmt0531_Tom_41091231.nc")
        self.python_config_filename=path.join(self.ancillary_data_path,
                                              "dynamic_lake_production_driver.cfg")
        self.tarasov_based_orog_correction=True
        self.glacier_mask_filename==path.join(self.orography_path,"Ice6g_c_VM5a_10min_0k.nc")
        self.present_day_base_orography_filename=path.join(self.orography_path,
                                                           "Ice6g_c_VM5a_10min_0k.nc")
        self.non_standard_orog_correction_filename=path.join(self.ancillary_data_path,
                                                             "ice5g_0k_lake_corrs_no_intermediaries_lake_corrections_driver_20200726_181304.nc")
        self.current_date = 0
        self.date_based_sill_height_corrections_list_filename = path.join(self.ancillary_data_path,
                                                                          "southern_exit_agassiz_corrections.txt")
        self.compile_paragen_and_hdfile()
        self.no_intermediaries_dynamic_lake_driver()
        return (self.output_hdparas_filepath,self.output_hdstart_filepath,
                self.output_lakeparas_filepath,self.output_lakestart_filepath)

    def trial_run_for_14440(self):
        """Trial run using data from main topo scripting for 14440k"""
        super(Dynamic_Lake_Production_Run_Drivers,self).__init__()
        file_label = self._generate_file_label()
        data_directory = "/Users/thomasriddick/Documents/data/simulation_data/laketestdata/files_for_14440_timeslice"
        self.original_orography_filename=data_directory+ "/GLAC1D_Top01_surf_14440.nc"
        self.original_ls_mask_filename=data_directory+"/landsea_14440k_inv.nc"
        self.output_hdparas_filepath=("/Users/thomasriddick/Documents/data/temp/"
                                      "temp_workdir_lake_14440/hdpara_{0}_14440.nc".format(file_label))
        self.output_hdstart_filepath=("/Users/thomasriddick/Documents/data/temp/"
                                      "temp_workdir_lake_14440/hdstart_{0}_14440.nc".format(file_label))
        self.ancillary_data_path=("/Users/thomasriddick/Documents/data/hd_ancillary_data_dirs/HDancillarydata_lakes_v2")
        self.working_directory_path="/Users/thomasriddick/Documents/data/temp/temp_workdir_lake_14440"
        self.output_lakeparas_filepath = "/Users/thomasriddick/Documents/data/temp/temp_workdir_lake_14440/lakeparas_{0}_14440.nc".format(file_label)
        self.output_lakestart_filepath = "/Users/thomasriddick/Documents/data/temp/temp_workdir_lake_14440/lakestart_{0}_14440.nc".format(file_label)
        self.input_water_to_redistribute_filepath=("/Users/thomasriddick/Documents/data/simulation_data"
                                                   "/laketestdata/lake_volumes_pmt0531_Tom_41091231.nc")
        self.python_config_filename=path.join(self.ancillary_data_path,
                                              "dynamic_lake_production_driver.cfg")
        self.tarasov_based_orog_correction=True
        self.glacier_mask_filename=data_directory+"/glac01_14440.nc"
        self.present_day_base_orography_filename=data_directory+"/GLAC1D_Top01_surf_0000.nc"
        self.non_standard_orog_correction_filename=path.join(self.ancillary_data_path,
                                                             "lake_analysis_two_26_Mar_2022_correction_field_version_34.nc")
        self.current_date = 14440
        self.date_based_sill_height_corrections_list_filename = path.join(self.ancillary_data_path,
                                                                          "southern_exit_agassiz_corrections.txt")
        self.compile_paragen_and_hdfile()
        self.no_intermediaries_dynamic_lake_driver()
        return (self.output_hdparas_filepath,self.output_hdstart_filepath,
                self.output_lakeparas_filepath,self.output_lakestart_filepath)

    def trial_run_for_mid_deglaciation(self):
        super(Dynamic_Lake_Production_Run_Drivers,self).__init__()
        file_label = self._generate_file_label()
        self.original_orography_filename=("/Users/thomasriddick/Documents/data/unit_test_data/"
                                          "updated_orog__extracted_for_0k_prepare_basins_from_glac1D_20210205_135817_1250.nc")
        self.original_ls_mask_filename=("/Users/thomasriddick/Documents/data/simulation_data/laketestdata/"
                                        "ls_mask_prepare_basins_from_glac1D_20201123_200519_1250_grid.nc")
        self.output_hdparas_filepath="/Users/thomasriddick/Documents/data/temp/hdpara_{0}_deglac.nc".format(file_label)
        self.output_hdstart_filepath="/Users/thomasriddick/Documents/data/temp/hdstart_{0}_deglac.nc".format(file_label)
        self.ancillary_data_path="/Users/thomasriddick/Documents/data/hd_ancillary_data_dirs/HDancillarydata_lakes"
        self.working_directory_path="/Users/thomasriddick/Documents/data/temp/temp_workdir_lake_deglac"
        self.output_lakeparas_filepath = "/Users/thomasriddick/Documents/data/temp/lakeparas_{0}_deglac.nc".format(file_label)
        self.output_lakestart_filepath = "/Users/thomasriddick/Documents/data/temp/lakestart_{0}_deglac.nc".format(file_label)
        self.input_water_to_redistribute_filepath=("/Users/thomasriddick/Documents/data/simulation_data/"
                                                   "laketestdata/lake_volumes_pmt0531_Tom_41091231.nc")
        self.python_config_filename=path.join(self.ancillary_data_path,
                                              "dynamic_lake_production_driver.cfg")
        self.tarasov_based_orog_correction=True
        self.glacier_mask_filename=("/Users/thomasriddick/Documents/"
                                    "data/simulation_data/transient_sim_data/1/GLAC1D_ICEM_10min_1250.nc")
        self.present_day_base_orography_filename=("/Users/thomasriddick/Documents/data/unit_test_data/"
                                                  "updated_orog__extracted_for_1250prepare_basins_from_glac1D_20210205_135817_1250.nc")
        self.non_standard_orog_correction_filename=path.join(self.ancillary_data_path,
                                                             "ice5g_0k_lake_corrs_no_intermediaries_lake_corrections_driver_20200726_181304.nc")
        self.current_date = 12500
        self.date_based_sill_height_corrections_list_filename = path.join(self.ancillary_data_path,
                                                                          "southern_exit_agassiz_corrections.txt")
        self.compile_paragen_and_hdfile()
        self.no_intermediaries_dynamic_lake_driver()
        return (self.output_hdparas_filepath,self.output_hdstart_filepath,
                self.output_lakeparas_filepath,self.output_lakestart_filepath)

    def no_intermediaries_dynamic_lake_driver(self):
        """Generates necessary files for running a dynamic lake model

        Arguments: None
        Returns: nothing
        """

        raise RuntimeError("Writing Jumps not implemented!")
        config = self._read_and_validate_config()
        print_timing_info = config.getboolean("general_options","print_timing_information")
        if print_timing_info:
            start_time = timer()
        base_hd_restart_file = path.join(self.ancillary_data_path,"hd_restart_from_hd_file_ten_minute_data_from_virna_"
                                        "0k_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170113_"
                                        "135934_after_one_year_running.nc")
        ref_hd_paras_file = path.join(self.ancillary_data_path,"hd_file_ten_minute_data_from_virna_0k_ALG4_sinkless_no_"
                                      "true_sinks_oceans_lsmask_plus_upscale_rdirs_20170116_123858_to_use_as_"
                                      "hdparas_ref.nc")
        surface_to_10min_map_filename = path.join(self.ancillary_data_path,
                                                  "t31_to_ten_min_map.nc")
        if self.present_day_base_orography_filename:
            present_day_reference_orography_filename = path.join(self.ancillary_data_path,
                                                                 "ice5g_v1_2_00_0k_10min.nc")
        if self.non_standard_orog_correction_filename is not None:
            orography_corrections_filename = self.non_standard_orog_correction_filename
        else:
            orography_corrections_filename = path.join(self.ancillary_data_path,
                                                       "lake_analysis_two_26_Mar_2022_correction_field_version_34.nc")
        if self.date_based_sill_height_corrections_list_filename is None:
            if config.has_option("general_options",
                                 "date_based_sill_height_corrections_list_filename"):
                self.date_based_sill_height_corrections_list_filename =\
                    config.get("general_options",
                               "date_based_sill_height_corrections_list_filename")
        elif config.has_option("general_options",
                                 "date_based_sill_height_corrections_list_filename"):
            raise RuntimeError("More than one date based sill height correction file specified")
        #Change ls mask to correct type
        ls_mask_10min = iodriver.advanced_field_loader(self.original_ls_mask_filename,
                                                       field_type='Generic',
                                                       fieldname=config.get("input_fieldname_options",
                                                                            "input_landsea_mask_fieldname"),
                                                       adjust_orientation=True)
        ls_mask_10min.change_dtype(np.int32)
        #Add corrections to orography
        orography_10min = iodriver.advanced_field_loader(self.original_orography_filename,
                                                         fieldname=config.get("input_fieldname_options",
                                                                              "input_orography_fieldname"),
                                                         field_type='Orography')
        if self.present_day_base_orography_filename:
            present_day_base_orography = \
            iodriver.advanced_field_loader(self.present_day_base_orography_filename,
                                           field_type='Orography',
                                           fieldname=config.get("input_fieldname_options",
                                                                "input_base_present_day_orography_fieldname"))
            present_day_reference_orography = \
            iodriver.advanced_field_loader(present_day_reference_orography_filename,
                                           field_type='Orography',
                                           fieldname=config.get("input_fieldname_options",
                                                                "input_reference_present_day_fieldname"))
            orography_10min = utilities.rebase_orography(orography=orography_10min,
                                                         present_day_base_orography=\
                                                         present_day_base_orography,
                                                         present_day_reference_orography=\
                                                         present_day_reference_orography)

        orography_corrections_10min =  iodriver.advanced_field_loader(orography_corrections_filename,
                                                                      fieldname=config.get("input_fieldname_options",
                                                                                           "input_orography_corrections_fieldname"),
                                                                      field_type='Orography')
        orography_uncorrected_10min = orography_10min.copy()
        orography_10min.add(orography_corrections_10min)
        if self.date_based_sill_height_corrections_list_filename is not None:
            utilities.apply_date_based_sill_height_corrections(orography_10min,
                self.date_based_sill_height_corrections_list_filename,int(self.current_date))
        truesinks = field.Field(np.empty((1,1),dtype=np.int32),grid='HD')
        if print_timing_info:
            time_before_glacier_mask_application = timer()
        if self.glacier_mask_filename:
            glacier_mask_10min = iodriver.advanced_field_loader(self.glacier_mask_filename,
                                                                fieldname=config.get("input_fieldname_options",
                                                                                     "input_glacier_mask_fieldname"),
                                                                field_type='Orography')
            if config.getboolean("general_options",
                                 "use_gradual_transitions"):
                print("Using gradual transition from icesheet to land")
                glacier_mask_as_bool_10min = glacier_mask_10min.copy()
                glacier_mask_as_bool_10min.change_dtype(bool)
                orography_10min = utilities.\
                    replace_corrected_orography_with_original_for_glaciated_points_with_gradual_transition(
                    input_corrected_orography=orography_10min,
                    input_original_orography=orography_uncorrected_10min,
                    input_base_orography=present_day_base_orography,
                    input_glacier_mask=glacier_mask_as_bool_10min,
                    blend_to_threshold=75.0,blend_from_threshold=25.0)
            else:
                orography_10min = utilities.\
                replace_corrected_orography_with_original_for_glaciated_grid_points(input_corrected_orography=\
                                                                                    orography_10min,
                                                                                    input_original_orography=\
                                                                                    orography_uncorrected_10min,
                                                                                    input_glacier_mask=
                                                                                    glacier_mask_10min)
            if self.additional_orography_corrections_filename is not None:
                additional_corrections_10min = \
                    iodriver.advanced_field_loader(
                        self.additional_orography_corrections_filename,
                        fieldname=config.get("input_fieldname_options",
                                         "input_additional_orography_corrections_fieldname"),
                        field_type='Orography')
                orography_10min.add(additional_corrections_10min)
            orography_10min.change_dtype(np.float64)
            orography_10min.make_contiguous()
            inverted_glacier_mask_10min = glacier_mask_10min.copy()
            inverted_glacier_mask_10min.invert_data()
            fill_sinks_wrapper.fill_sinks_cpp_func(orography_array=
                                                   orography_10min.get_data(),
                                                   method = 1,
                                                   use_ls_mask = True,
                                                   landsea_in =
                                                   np.ascontiguousarray(inverted_glacier_mask_10min.get_data(),
                                                                        dtype=np.int32),
                                                   set_ls_as_no_data_flag = False,
                                                   use_true_sinks = False,
                                                   true_sinks_in = np.ascontiguousarray(truesinks.get_data(),
                                                                                        dtype=np.int32),
                                                   add_slope = False,epsilon = 0.0)
        orography_10min.mask_field_with_external_mask(ls_mask_10min.get_data())
        orography_10min.fill_mask(self.ocean_floor_depth)
        if config.getboolean("output_options","output_corrected_orog"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "10min_corrected_orog.nc"),
                                           orography_10min,
                                           fieldname=config.get("output_fieldname_options",
                                                                "output_10min_corrected_orog_fieldname"))
        #Generate orography with filled sinks
        if print_timing_info:
            time_before_sink_filling = timer()
        grid_dims_10min=orography_10min.get_grid().get_grid_dimensions()
        orography_10min_filled = orography_10min.copy()
        fill_sinks_wrapper.fill_sinks_cpp_func(orography_array=
                                               np.ascontiguousarray(orography_10min_filled.get_data(),
                                                                                    dtype=np.float64),
                                               method = 1,
                                               use_ls_mask = True,
                                               landsea_in = np.ascontiguousarray(ls_mask_10min.get_data(),
                                                                                 dtype=np.int32),
                                               set_ls_as_no_data_flag = False,
                                               use_true_sinks = False,
                                               true_sinks_in = np.ascontiguousarray(truesinks.get_data(),
                                                                                    dtype=np.int32),
                                               add_slope = False,epsilon = 0.0)
        if config.getboolean("output_options","output_fine_filled_orog"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                     "10min_filled_orog.nc"),
                                           orography_10min_filled,
                                           fieldname=config.get("output_fieldname_options",
                                                                "output_10min_filled_orog_fieldname"))
        #Filter unfilled orography
        if print_timing_info:
            time_before_filtering = timer()
        orography_10min = \
          field.Field(np.ascontiguousarray(orography_10min.get_data(),
                                           dtype=np.float64),
                      grid=orography_10min.get_grid())
        lake_operators_wrapper.filter_out_shallow_lakes(orography_10min.get_data(),
                                                        np.ascontiguousarray(orography_10min_filled.\
                                                                             get_data(),
                                                                             dtype=np.float64),
                                                        minimum_depth_threshold=5.0)
        orography_10min = dynamic_lake_operators.\
            filter_narrow_lakes(input_unfilled_orography=orography_10min,
                                input_filled_orography=orography_10min_filled,
                                interior_cell_min_masked_neighbors=5,
                                edge_cell_max_masked_neighbors=4,
                                max_range=5,
                                iterations=5)
        #Generate River Directions
        if print_timing_info:
            time_before_rdir_generation = timer()
        rdirs_10min = determine_river_directions.determine_river_directions(orography=orography_10min,
                                                                            lsmask=ls_mask_10min,
                                                                            truesinks=None,
                                                                            always_flow_to_sea=True,
                                                                            use_diagonal_nbrs=True,
                                                                            mark_pits_as_true_sinks=True)
        if config.getboolean("output_options","output_fine_rdirs"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                     "10min_rdirs.nc"),
                                           rdirs_10min,
                                           fieldname=config.get("output_fieldname_options",
                                                                "output_10min_rdirs_fieldname"))
        #Run post processing
        if print_timing_info:
            time_before_10min_post_processing = timer()
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
        loops_log_10min_filename = path.join(self.working_directory_path,"loops_10min.log")
        catchments_10min = comp_catchs.compute_catchments_cpp(rdirs_10min.get_data(),
                                                              loops_log_10min_filename)
        catchments_10min = field.Field(comp_catchs.renumber_catchments_by_size(catchments_10min,
                                                                               loops_log_10min_filename),
                                       grid=rdirs_10min.get_grid())
        if config.getboolean("output_options","output_fine_flowtocell"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "10min_flowtocell.nc"),
                                           flowtocell_10min,
                                           fieldname=config.get("output_fieldname_options",
                                                                "output_10min_flow_to_cell"))
        if config.getboolean("output_options","output_fine_flowtorivermouths"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "10min_flowtorivermouths.nc"),
                                           flowtorivermouths_10min,
                                           fieldname=config.get("output_fieldname_options",
                                                                "output_10min_flow_to_river_mouths"))
        if config.getboolean("output_options","output_fine_catchments"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "10min_catchments.nc"),
                                           catchments_10min,
                                           fieldname=config.get("output_fieldname_options",
                                                                "output_10min_catchments"))
        #Run Upscaling
        if print_timing_info:
            time_before_upscaling = timer()
        loops_log_30min_filename = path.join(self.working_directory_path,"loops_30min.log")
        catchments_log_filename= path.join(self.working_directory_path,"catchments.log")
        cotat_plus_parameters_filename = path.join(self.ancillary_data_path,'cotat_plus_standard_params.nl')
        rdirs_30min = run_cotat_plus(rdirs_10min, flowtocell_10min,
                                      cotat_plus_parameters_filename,'HD')
        if config.getboolean("output_options","output_pre_loop_removal_coarse_rdirs"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_pre_loop_removal_rdirs.nc"),
                                           rdirs_30min,
                                           fieldname=config.get("output_fieldname_options",
                                                                "output_30min_pre_loop_removal_rdirs"))
        #Post processing
        if print_timing_info:
            time_before_30min_post_processing_one = timer()
        nlat30,nlong30 = rdirs_30min.get_grid().get_grid_dimensions()
        flowtocell_30min = field.CumulativeFlow(create_hypothetical_river_paths_map(riv_dirs=rdirs_30min.get_data(),
                                                                                    lsmask=None,
                                                                                    use_f2py_func=True,
                                                                                    use_f2py_sparse_iterator=True,
                                                                                    nlat=nlat30,
                                                                                    nlong=nlong30),
                                                                                    grid=rdirs_30min.get_grid())
        catchments_30min = comp_catchs.compute_catchments_cpp(rdirs_30min.get_data(),
                                                              loops_log_30min_filename)
        catchments_30min = field.Field(comp_catchs.renumber_catchments_by_size(catchments_30min,
                                                                               loops_log_30min_filename),
                                       grid=rdirs_30min.get_grid())
        rivermouths_30min = field.makeField(rdirs_30min.get_river_mouths(),'Generic',
                                            rdirs_30min.get_grid())
        flowtorivermouths_30min = field.makeField(flowtocell_30min.\
                                                  find_cumulative_flow_at_outlets(rivermouths_30min.\
                                                                                  get_data()),
                                                  'Generic',flowtocell_30min.get_grid())
        if config.getboolean("output_options","output_pre_loop_removal_coarse_flowtocell"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                     "30min_pre_loop_removal_flowtocell.nc"),
                                           flowtocell_30min,
                                           fieldname=config.get("output_fieldname_options",
                                                                "output_30min_pre_loop_removal_flow_to_cell"))
        if config.getboolean("output_options","output_pre_loop_removal_coarse_flowtorivermouths"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_pre_loop_removal_flowtorivermouths.nc"),
                                             flowtorivermouths_30min,
                                             fieldname=config.get("output_fieldname_options",
                                                                  "output_30min_pre_loop_removal_flow_to_river_mouth"))
        if config.getboolean("output_options","output_pre_loop_removal_coarse_catchments"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_pre_loop_removal_catchments.nc"),
                                             catchments_30min,
                                             fieldname=config.get("output_fieldname_options",
                                                                  "output_30min_pre_loop_removal_catchments"))
        #Run Loop Breaker
        if print_timing_info:
            time_before_loop_breaker = timer()
        loop_nums_list = []
        first_line_pattern = re.compile(r"^Loops found in catchments:$")
        with open(loops_log_30min_filename,'r') as f:
            if not first_line_pattern.match(f.readline().strip()):
                raise RuntimeError("Format of the file with list of catchments to remove loops from"
                                   " is invalid")
            for line in f:
                loop_nums_list.append(int(line.strip()))
        print('Removing loops from catchments: ' + ", ".join(str(value) for value in loop_nums_list))
        rdirs_30min = run_loop_breaker(rdirs_30min,flowtocell_30min,
                                       catchments_30min,rdirs_10min,
                                       flowtocell_10min,loop_nums_list,
                                       coarse_grid_type="HD")
        if config.getboolean("output_options","output_coarse_rdirs"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_rdirs.nc"),
                                             rdirs_30min,
                                             fieldname=config.get("output_fieldname_options",
                                                                  "output_30min_rdirs"))
        #Upscale the orography to the HD grid for calculating the flow parameters
        orography_30min= utilities.upscale_field(orography_10min,"HD","Sum",{},
                                                 scalenumbers=True)
        if config.getboolean("output_options","output_coarse_unfilled_orog"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_unfilled_orog.nc"),
                                             orography_30min,
                                             fieldname=config.get("output_fieldname_options",
                                                                  "output_30min_unfilled_orog"))
        #Extract HD ls mask from river directions
        ls_mask_30min = field.RiverDirections(rdirs_30min.get_lsmask(),
                                              grid=rdirs_30min.get_grid())
        #Fill HD orography for parameter generation
        if print_timing_info:
            time_before_coarse_sink_filling = timer()
        truesinks = field.Field(np.empty((1,1),dtype=np.int32),grid='HD')
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
        if config.getboolean("output_options","output_coarse_filled_orog"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_filled_orog.nc"),
                                             orography_30min,
                                             fieldname=config.get("output_fieldname_options",
                                                                  "output_30min_filled_orog"))
        #Transform any necessary field into the necessary format and save ready for parameter generation
        if print_timing_info:
            time_before_parameter_generation = timer()
        transformed_coarse_rdirs_filename = path.join(self.working_directory_path,"30minute_river_dirs_temp.nc")
        transformed_HD_filled_orography_filename = path.join(self.working_directory_path,"30minute_filled_orog_temp.nc")
        transformed_HD_ls_mask_filename = path.join(self.working_directory_path,"30minute_ls_mask_temp.nc")
        half_degree_grid_filepath = path.join(self.ancillary_data_path,"grid_0_5.txt")
        iodriver.advanced_field_writer(transformed_coarse_rdirs_filename,
                                         rdirs_30min,
                                         fieldname="field_value")
        iodriver.advanced_field_writer(transformed_HD_filled_orography_filename,
                                         orography_30min,
                                         fieldname="field_value")
        ls_mask_30min.invert_data()
        iodriver.advanced_field_writer(transformed_HD_ls_mask_filename,
                                       ls_mask_30min,
                                       fieldname=config.get("output_fieldname_options",
                                                            "output_30min_ls_mask"))
        #If required generate flow parameters and create a hdparas.file otherwise
        #use a river direction file with coordinates as the hdparas.file
        if config.getboolean("general_options","generate_flow_parameters"):
            #Generate parameters
            self._generate_flow_parameters(rdir_file=transformed_coarse_rdirs_filename,
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
            self._generate_hd_file(rdir_file=path.splitext(transformed_coarse_rdirs_filename)[0] + ".dat",
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
            shutil.copy2(transformed_coarse_rdirs_filename,self.output_hdparas_filepath)
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
        if print_timing_info:
            time_before_30min_post_processing_two = timer()
        flowtocell_30min = field.CumulativeFlow(create_hypothetical_river_paths_map(riv_dirs=rdirs_30min.get_data(),
                                                                                    lsmask=None,
                                                                                    use_f2py_func=True,
                                                                                    use_f2py_sparse_iterator=True,
                                                                                    nlat=nlat30,
                                                                                    nlong=nlong30),
                                                                                    grid=rdirs_30min.get_grid())
        catchments_30min = comp_catchs.compute_catchments_cpp(rdirs_30min.get_data(),
                                                              loops_log_30min_filename)
        catchments_30min = field.Field(comp_catchs.renumber_catchments_by_size(catchments_30min,
                                                                               loops_log_30min_filename),
                                       grid=rdirs_30min.get_grid())
        rivermouths_30min = field.makeField(rdirs_30min.get_river_mouths(),'Generic',
                                            rdirs_30min.get_grid())
        flowtorivermouths_30min = field.makeField(flowtocell_30min.\
                                                  find_cumulative_flow_at_outlets(rivermouths_30min.\
                                                                                  get_data()),
                                                  'Generic',
                                                  flowtocell_30min.get_grid())
        if config.getboolean("output_options","output_coarse_flowtocell"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_flowtocell.nc"),
                                           flowtocell_30min,
                                           fieldname=config.get("output_fieldname_options",
                                                                "output_30min_flow_to_cell"))
        if config.getboolean("output_options","output_coarse_flowtorivermouths"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                     "30min_flowtorivermouths.nc"),
                                           flowtorivermouths_30min,
                                           fieldname=config.get("output_fieldname_options",
                                                                "output_30min_flow_to_river_mouths"))
        if config.getboolean("output_options","output_coarse_catchments"):
            iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                       "30min_catchments.nc"),
                                           catchments_30min,
                                           fieldname=config.get("output_fieldname_options",
                                                                "output_30min_catchments"))
        cell_areas_filename_10min = path.join(self.ancillary_data_path,
                                              "10min_grid_area_default_R.nc")
        output_lakestart_dirname = path.dirname(self.output_lakestart_filepath)
        output_water_redistributed_to_lakes_file = path.join(self.working_directory_path,
                                                             "water_to_lakes.nc")
        output_water_redistributed_to_rivers_file = path.join(self.working_directory_path,
                                                             "water_to_rivers.nc")
        if print_timing_info:
            time_before_basin_evaluation = timer()
        #Generate Minima
        minima = field.Field(rdirs_10min.extract_truesinks(),grid=rdirs_10min.get_grid())
        #Load 10 minute to surface grid mapping
        corresponding_surface_cell_lat_index = \
            iodriver.advanced_field_loader(surface_to_10min_map_filename,
                                           field_type='Generic',
                                           fieldname=config.get("input_fieldname_options",
                                                                "input_surface_lat_index_fieldname"))
        corresponding_surface_cell_lon_index = \
            iodriver.advanced_field_loader(surface_to_10min_map_filename,
                                           field_type='Generic',
                                           fieldname=config.get("input_fieldname_options",
                                                                "input_surface_lon_index_fieldname"))
        #Run Basin Evaluation
        input_cell_areas = iodriver.advanced_field_loader(cell_areas_filename_10min,
                                                          field_type='Generic',
                                                          fieldname="cell_area")
        fine_grid = orography_10min.get_grid()
        fine_shape = orography_10min.get_data().shape
        connection_volume_thresholds = field.Field(np.zeros(fine_shape,dtype=np.float64,order='C'),fine_grid)
        flood_volume_thresholds = field.Field(np.zeros(fine_shape,dtype=np.float64,order='C'),fine_grid)
        flood_next_cell_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        flood_next_cell_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        connect_next_cell_lat_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        connect_next_cell_lon_index = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        connection_heights = field.Field(np.zeros(fine_shape,dtype=np.float64,order='C'),fine_grid)
        flood_heights = field.Field(np.zeros(fine_shape,dtype=np.float64,order='C'),fine_grid)
        connect_merge_and_redirect_indices_index = \
            field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        flood_merge_and_redirect_indices_index = \
            field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        basin_catchment_numbers = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        sinkless_rdirs_10min = field.Field(np.zeros(fine_shape,dtype=np.int32,order='C'),fine_grid)
        merges_filename =  path.join(self.working_directory_path,"merges_and_redirects_temp.nc")
        fields_filename =  path.join(self.working_directory_path,"fields_temp.nc")
        evaluate_basins_wrapper.evaluate_basins(minima_in_int=
                                                np.ascontiguousarray(minima.get_data(),dtype=np.int32),
                                                raw_orography_in=
                                                np.ascontiguousarray(orography_10min.get_data(),
                                                                     dtype=np.float64),
                                                corrected_orography_in=
                                                np.ascontiguousarray(orography_10min.get_data(),
                                                                     dtype=np.float64),
                                                cell_areas_in=
                                                np.ascontiguousarray(input_cell_areas.get_data(),
                                                                     dtype=np.float64),
                                                connection_volume_thresholds_in=
                                                connection_volume_thresholds.get_data(),
                                                flood_volume_thresholds_in=
                                                flood_volume_thresholds.get_data(),
                                                connection_heights_in=
                                                connection_heights.get_data(),
                                                flood_heights_in=
                                                flood_heights.get_data(),
                                                prior_fine_rdirs_in=
                                                np.ascontiguousarray(rdirs_10min.get_data(),
                                                                     dtype=np.float64),
                                                prior_coarse_rdirs_in=
                                                np.ascontiguousarray(rdirs_30min.get_data(),
                                                                     dtype=np.float64),
                                                prior_fine_catchments_in=
                                                np.ascontiguousarray(catchments_10min.get_data(),
                                                                     dtype=np.int32),
                                                coarse_catchment_nums_in=
                                                np.ascontiguousarray(catchments_30min.get_data(),
                                                                     dtype=np.int32),
                                                flood_next_cell_lat_index_in=
                                                flood_next_cell_lat_index.get_data(),
                                                flood_next_cell_lon_index_in=
                                                flood_next_cell_lon_index.get_data(),
                                                connect_next_cell_lat_index_in=
                                                connect_next_cell_lat_index.get_data(),
                                                connect_next_cell_lon_index_in=
                                                connect_next_cell_lon_index.get_data(),
                                                connect_merge_and_redirect_indices_index_in=
                                                connect_merge_and_redirect_indices_index.get_data(),
                                                flood_merge_and_redirect_indices_index_in=
                                                flood_merge_and_redirect_indices_index.get_data(),
                                                merges_filepath=merges_filename,
                                                basin_catchment_numbers_in=
                                                basin_catchment_numbers.get_data(),
                                                sinkless_rdirs_in=
                                                sinkless_rdirs_10min.get_data())
        fields_to_write = [connection_volume_thresholds,
                           flood_volume_thresholds,
                           connection_heights,
                           flood_heights,
                           orography_10min,
                           orography_10min,
                           input_cell_areas,
                           flood_next_cell_lat_index,
                           flood_next_cell_lon_index,
                           connect_next_cell_lat_index,
                           connect_next_cell_lon_index,
                           corresponding_surface_cell_lat_index,
                           corresponding_surface_cell_lon_index,
                           connect_merge_and_redirect_indices_index,
                           flood_merge_and_redirect_indices_index,
                           minima]
        fieldnames_for_fields_to_write = ['connection_volume_thresholds',
                                          'flood_volume_thresholds',
                                          'connection_heights',
                                          'flood_heights',
                                          'corrected_heights',
                                          'raw_heights',
                                          'cell_areas',
                                          'flood_next_cell_lat_index',
                                          'flood_next_cell_lon_index',
                                          'connect_next_cell_lat_index',
                                          'connect_next_cell_lon_index',
                                          'corresponding_surface_cell_lat_index',
                                          'corresponding_surface_cell_lon_index',
                                          'connect_merge_and_redirect_indices_index',
                                          'flood_merge_and_redirect_indices_index',
                                          'lake_centers']
        iodriver.advanced_field_writer(fields_filename,
                                       fields_to_write,
                                       fieldname=fieldnames_for_fields_to_write)
        cdo_inst = cdo.Cdo()
        cdo_inst.merge(input=" ".join([merges_filename,fields_filename]),
                      output=self.output_lakeparas_filepath)
        for file_to_remove in [merges_filename,fields_filename]:
            os.remove(file_to_remove)
        #Write out basins maps
        basin_catchment_numbers_filename = path.join(self.working_directory_path,
                                                     "10min_basin_catchment_numbers.nc")
        iodriver.advanced_field_writer(basin_catchment_numbers_filename,
                                       basin_catchment_numbers,
                                       fieldname=config.get("output_fieldname_options",
                                                            "output_10min_basin_catchment_numbers"))
        #Calculate and write lake volumes
        if config.getboolean("output_options","output_10min_lake_volumes"):
            extract_lake_volumes.\
                lake_volume_extraction_driver(self.output_lakeparas_filepath,
                                              basin_catchment_numbers_filename,
                                              path.join(self.working_directory_path,
                                                        "10min_lake_volumes.nc"))
        #Write out sinkless rdirs
        if config.getboolean("output_options","output_10min_sinkless_rdirs"):
            sinkless_rdirs_10min_filename = \
                iodriver.advanced_field_writer(path.join(self.working_directory_path,
                                                         "10min_sinkless_rdirs.nc"),
                                               sinkless_rdirs_10min,
                                               fieldname=config.get("output_fieldname_options",
                                                                    "output_10min_sinkless_rdirs"))

        river_directions_filepath = path.join(self.working_directory_path,"30min_rdirs.nc")
        coarse_catchments_filepath = path.join(self.working_directory_path,"30min_catchments.nc")
        coarse_catchments_fieldname = "catchments"
        connected_coarse_catchments_out_filename = path.join(self.working_directory_path,
                                                             "30min_connected_catchments.nc")
        connected_coarse_catchments_out_fieldname = "catchments"
        river_directions_fieldname = "rdirs"
        cumulative_flow_filename=path.join(self.working_directory_path,
                                           "30min_flowtocell.nc")
        cumulative_flow_fieldname="cumulative_flow"
        cumulative_flow_out_filename=path.join(self.working_directory_path,
                                               "30min_flowtocell_connected.nc")
        cumulative_flow_out_fieldname="cumulative_flow"
        cumulative_river_mouth_flow_out_filename=path.join(self.working_directory_path,
                                                           "30min_flowtorivermouths_connected.nc")
        cumulative_river_mouth_flow_out_fieldname="cumulative_flow_to_ocean"
        cclc.connect_coarse_lake_catchments_driver(coarse_catchments_filepath,
                                                   self.output_lakeparas_filepath,
                                                   basin_catchment_numbers_filename,
                                                   river_directions_filepath,
                                                   connected_coarse_catchments_out_filename,
                                                   coarse_catchments_fieldname,
                                                   connected_coarse_catchments_out_fieldname,
                                                   "basin_catchment_numbers",
                                                   river_directions_fieldname,
                                                   cumulative_flow_filename,
                                                   cumulative_flow_out_filename,
                                                   cumulative_flow_fieldname,
                                                   cumulative_flow_out_fieldname)
        river_mouth_marking_driver.\
        advanced_flow_to_rivermouth_calculation_driver(input_river_directions_filename=
                                                       river_directions_filepath,
                                                       input_flow_to_cell_filename=
                                                       cumulative_flow_out_filename,
                                                       output_flow_to_river_mouths_filename=
                                                       cumulative_river_mouth_flow_out_filename,
                                                       input_river_directions_fieldname=
                                                       river_directions_fieldname,
                                                       input_flow_to_cell_fieldname=
                                                       cumulative_flow_out_fieldname,
                                                       output_flow_to_river_mouths_fieldname=
                                                       cumulative_river_mouth_flow_out_fieldname)
        #Redistribute water
        if print_timing_info:
            time_before_water_redistribution = timer()
        water_to_redistribute = \
        iodriver.advanced_field_loader(self.input_water_to_redistribute_filepath,
                                       field_type='Generic',
                                       fieldname="lake_field")
        fine_grid = basin_catchment_numbers.get_grid()
        fine_shape = basin_catchment_numbers.get_data().shape
        coarse_grid = grid.makeGrid('HD')
        water_redistributed_to_lakes = field.Field(np.zeros(fine_shape,dtype=np.float64,order='C'),
                                                 fine_grid)
        water_redistributed_to_rivers = field.Field(coarse_grid.create_empty_field(np.float64),coarse_grid)
        lake_operators_wrapper.redistribute_water(np.ascontiguousarray(basin_catchment_numbers.get_data(),
                                                dtype=np.int32),
                                                np.ascontiguousarray(minima.get_data(),
                                                dtype=np.int32),
                                                np.ascontiguousarray(water_to_redistribute.get_data(),
                                                dtype=np.float64),
                                                water_redistributed_to_lakes.get_data(),
                                                water_redistributed_to_rivers.get_data())
        iodriver.advanced_field_writer(output_water_redistributed_to_lakes_file,
                                       water_redistributed_to_lakes,
                                       fieldname="water_redistributed_to_lakes")
        iodriver.advanced_field_writer(output_water_redistributed_to_rivers_file,
                                       water_redistributed_to_rivers,
                                       fieldname="water_redistributed_to_rivers")
        cdo_inst = cdo.Cdo()
        cdo_inst.merge(input=" ".join([output_water_redistributed_to_lakes_file,
                                       output_water_redistributed_to_rivers_file]),
                       output=self.output_lakestart_filepath)
        os.remove(output_water_redistributed_to_lakes_file)
        os.remove(output_water_redistributed_to_rivers_file)
        if not config.getboolean("output_options","output_10min_basin_catchment_numbers"):
            os.remove(basin_catchment_numbers_filename)
        if print_timing_info:
            end_time = timer()
            print("---- Timing info ----")
            print("Initial setup:         {: 6.2f}s".\
                format(time_before_glacier_mask_application - start_time))
            print("Glacier Mask Addition: {: 6.2f}s".\
                format(time_before_sink_filling - time_before_glacier_mask_application))
            print("Sink Filling:          {: 6.2f}s".\
                format(time_before_filtering - time_before_sink_filling))
            print("Filtering:             {: 6.2f}s".\
                format(time_before_rdir_generation - time_before_filtering))
            print("River Direction Gen.:  {: 6.2f}s".\
                format(time_before_10min_post_processing - time_before_rdir_generation))
            print("Post Processing:       {: 6.2f}s".\
                format(time_before_upscaling - time_before_10min_post_processing))
            print("Upscaling:             {: 6.2f}s".\
                format(time_before_30min_post_processing_one - time_before_upscaling))
            print("Post Processing (II):  {: 6.2f}s".\
                format(time_before_loop_breaker -
                        time_before_30min_post_processing_one))
            print("Loop Breaker:          {: 6.2f}s".\
                format(time_before_coarse_sink_filling - time_before_loop_breaker))
            print("Sink Filling (II):     {: 6.2f}s".\
                format(time_before_parameter_generation - time_before_coarse_sink_filling))
            print("Parameter Generation:  {: 6.2f}s".\
                format(time_before_30min_post_processing_two -
                        time_before_parameter_generation))
            print("Post Processing:       {: 6.2f}s".\
                format(time_before_basin_evaluation - time_before_30min_post_processing_two))
            print("Basin Evaluation:      {: 6.2f}s".\
                format(time_before_water_redistribution - time_before_basin_evaluation))
            print("Water Redistribution:  {: 6.2f}s".\
                format(end_time-time_before_water_redistribution))
            print("Total:                 {: 6.2f}s".\
                format(end_time-start_time))

def setup_and_run_dynamic_hd_para_and_lake_gen_from_command_line_arguments(args):
    """Setup and run a dynamic hd production run from the command line arguments passed in by main"""
    driver_object = Dynamic_Lake_Production_Run_Drivers(**vars(args))
    driver_object.no_intermediaries_dynamic_lake_driver()

class Arguments:
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
    parser.add_argument('input_water_to_redistribute_filepath',
                        metavar='input-water-to-redistribute-filepath',
                        help='Full path to file containing water from lakes in previous run',
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
    parser.add_argument('output_lakeparas_filepath',
                        metavar='output-lakeparas-filepath',
                        help='Full path to target destination for output lakeparas file',
                        type=str)
    parser.add_argument('ancillary_data_directory',
                        metavar='ancillary-data-directory',
                        help='Full path to directory containing ancillary data',
                        type=str)
    parser.add_argument('working_directory',
                        metavar='working-directory',
                        help='Full path to target working directory',
                        type=str)
    parser.add_argument('output_lakestart_filepath',
                        metavar='output-lakestart-filepath',
                        help='Full path to target destination for output lakestart file ',
                        type=str)
    parser.add_argument('-s','--output-hdstart-filepath',
                        help='Full path to target destination for output hdstart file',
                        type=str,
                        default=None)
    parser.add_argument('-d','--current-date',
                        help='Current date for date based orography corrections',
                        type=str,
                        default=None)
    parser.add_argument('-a','--additional-orography-corrections-filepath',
                        help='Field with custom relative orography corrections to apply',
                        type=str,
                        default=None)
    #Adding the variables to a namespace other than that of the parser keeps the namespace clean
    #and allows us to pass it directly to main
    parser.parse_args(namespace=args)
    return args

if __name__ == '__main__':
    #Parse arguments and then run
    args = parse_arguments()
    setup_and_run_dynamic_hd_para_and_lake_gen_from_command_line_arguments(args)
