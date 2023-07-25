'''
Created on Dec 4, 2017

@author: thomasriddick
'''
from os.path import join
import os.path as path
import numpy as np
import time
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.base.iodriver import advanced_field_loader
from Dynamic_HD_Scripts.base.iodriver import advanced_field_writer
from Dynamic_HD_Scripts.base import field
from Dynamic_HD_Scripts.tools import determine_river_directions
from Dynamic_HD_Scripts.tools import extract_lake_volumes
from Dynamic_HD_Scripts.tools import compute_catchments as cc
from Dynamic_HD_Scripts.tools import flow_to_grid_cell as ftgc
from Dynamic_HD_Scripts.tools import connect_coarse_lake_catchments as cclc
from Dynamic_HD_Scripts.tools import dynamic_lake_operators
from Dynamic_HD_Scripts.tools import fill_sinks_driver
from Dynamic_HD_Scripts.tools import river_mouth_marking_driver
from Dynamic_HD_Scripts.tools import create_connected_lsmask_driver as ccld
from Dynamic_HD_Scripts.utilities import utilities
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

    def connect_catchments_for_transient_run(self):
      #base_filepath = "/Users/thomasriddick/Documents/data/lake_analysis_runs/lake_analysis_one_21_Jun_2021/lakes/results"
      #base_filepath = "/Users/thomasriddick/Documents/data/lake_analysis_runs/lake_analysis_one_21_Jun_2021/lakes/results"
      base_filepath = "/Users/thomasriddick/Documents/data/lake_analysis_runs/lake_analysis_two_26_Mar_2022/lakes/results"
      #dates = range(15990,15980,-10)
      dates = range(14800,11000,-100)
      #dates = [0]
      for date in dates:
        #river_directions_filepath = ("{0}/diag_version_13_date_{1}/30min_rdirs.nc".format(base_filepath,date))
        #coarse_catchments_filepath = ("{0}/diag_version_13_date_{1}/30min_catchments.nc".format(base_filepath,date))
        river_directions_filepath = ("{0}/diag_version_35_date_{1}/30min_rdirs.nc".format(base_filepath,date))
        coarse_catchments_filepath = ("{0}/diag_version_35_date_{1}/30min_catchments.nc".format(base_filepath,date))
        river_directions_fieldname = "rdirs"
        coarse_catchments_fieldname = "catchments"
        # cc.advanced_main(filename=river_directions_filepath,
        #                  fieldname="rdirs",
        #                  output_filename=coarse_catchments_filepath,
        #                  output_fieldname="catchments",
        #                  loop_logfile=("{0}/loops_log_{1}.txt".format(base_filepath,date)),
        #                  use_cpp_alg=True)
        lake_parameters_filepath = ("{0}/lakeparas_version_35_date_{1}.nc".format(base_filepath,date))
        basin_numbers_filepath = ("{0}/diag_version_35_date_{1}/basin_catchment_numbers.nc".format(base_filepath,date))
        connected_coarse_catchments_out_filename = ("{0}/diag_version_35_date_{1}/30min_connected_catchments.nc".\
                                                    format(base_filepath,date))
        connected_coarse_catchments_out_fieldname = "catchments"
        basin_catchment_numbers_fieldname = "basin_catchment_numbers"
        cumulative_flow_filename=("{0}/diag_version_35_date_{1}/30min_flowtocell.nc".format(base_filepath,date))
        cumulative_flow_fieldname="cumulative_flow"
        cumulative_flow_out_filename=("{0}/diag_version_35_date_{1}/30min"
                                      "_flowtocell_connected.nc".format(base_filepath,date))
        cumulative_flow_out_fieldname="cumulative_flow"
        cumulative_river_mouth_flow_out_filename=("{0}/diag_version_35_date_{1}/"
                                                  "30min_flowtorivermouths_connectedv2.nc".format(base_filepath,date))
        cumulative_river_mouth_flow_out_fieldname="cumulative_flow_to_ocean"
        cclc.connect_coarse_lake_catchments_driver(coarse_catchments_filepath,
                                                   lake_parameters_filepath,
                                                   basin_numbers_filepath,
                                                   river_directions_filepath,
                                                   connected_coarse_catchments_out_filename,
                                                   coarse_catchments_fieldname,
                                                   connected_coarse_catchments_out_fieldname,
                                                   basin_catchment_numbers_fieldname,
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

    def extract_volumes_for_transient_run(self):
      base_filepath = "/Users/thomasriddick/Documents/data/lake_analysis_runs/lake_analysis_one_21_Jun_2021/lakes/results"
      dates = range(15000,11000,-100)
      #dates = range(15990,15980,-10)
      #dates = [0]
      for date in dates:
        lake_parameters_filepath = ("{0}/lakeparas_version_33_date_{1}.nc".format(base_filepath,date))
        basin_catchment_numbers_filepath = ("{0}/diag_version_33_date_{1}/"
                                            "basin_catchment_numbers.nc".format(base_filepath,date))
        lake_volumes_out_filepath = ("{0}/diag_version_33_date_{1}/"
                                     "10min_lake_volumes.nc".format(base_filepath,date))
        extract_lake_volumes.\
            lake_volume_extraction_driver(lake_parameters_filepath,
                                          basin_catchment_numbers_filepath,
                                          lake_volumes_out_filepath)

    def add_10min_rmouth_to_transient_data(self):
            basename_analysis_two = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                                      "lake_analysis_one_21_Jun_2021/rivers/results/default_orog_corrs/"
                                      "diag_version_34_date_0_with_truesinks/")
            rdirs_list = [basename_analysis_one+"10min_rdirs.nc"]
                          # basename_analysis_two+"10min_rdirs.nc"]
            flow_to_cell_list = [basename_analysis_one+"10min_flowtocell.nc"]
                                 # basename_analysis_two+"10min_flowtocell.nc"]
            output_list = [basename_analysis_one+"10min_rmouth_flowtocell.nc"]
                           # basename_analysis_two+"10min_rmouth_flowtocell.nc"]
            for rdirs_file,flow_to_cell_file,output_file in \
                 zip(rdirs_list,flow_to_cell_list,output_list):
                river_mouth_marking_driver.\
                    advanced_flow_to_rivermouth_calculation_driver(input_river_directions_filename=rdirs_file,
                                                                   input_flow_to_cell_filename=flow_to_cell_file,
                                                                   output_flow_to_river_mouths_filename=output_file,
                                                                   input_river_directions_fieldname="rdir",
                                                                   input_flow_to_cell_fieldname="acc",
                                                                   output_flow_to_river_mouths_fieldname="acc")

    def expand_transient_data_catchments_to_include_rmouth(self):
        dates = range(15000,11000,-100)
        #dates = range(15990,15980,-10)
        #dates = [0]
        for date in dates:
            # basename_analysis_one = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
            #                          "lake_analysis_two_26_Mar_2022/rivers/results/"
            #                          "diag_version_33_date_{}_with_truesinks/".format(date))
            basename_analysis_one = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
                                     "lake_analysis_two_26_Mar_2022/rivers/results/"
                                     "diag_version_34_date_{}/".format(date))
            # basename_analysis_two = ("/Users/thomasriddick/Documents/data/lake_analysis_runs/"
            #                          "lake_analysis_two_26_Mar_2022/"
            #                          "rivers/results/diag_version_0_date_0/")
            ref_catchment_filename = join(basename_analysis_one,"10min_catchments.nc")
            # data_catchment_filename = join(basename_analysis_two,"10min_catchments_grid.nc")
            ref_rdirs_filename = join(basename_analysis_one,"10min_rdirs.nc")
            # data_rdirs_filename = join(basename_analysis_two,"10min_rdirs_grid.nc")
            ref_expanded_catchment_filename = join(basename_analysis_one,
                                                   "10min_catchments_ext.nc")
            # data_expanded_catchment_filename = join(basename_analysis_two,
            #                                         "10min_catchments_grid_ext.nc")
            ref_catchment_field = advanced_field_loader(ref_catchment_filename,
                                                        time_slice=None,
                                                        fieldname="catchments",
                                                        adjust_orientation=True)
            # data_catchment_field = advanced_field_loader(data_catchment_filename,
            #                                              time_slice=None,
            #                                              fieldname="catch",
            #                                              adjust_orientation=True)
            ref_rdirs_field = advanced_field_loader(ref_rdirs_filename,
                                                    time_slice=None,
                                                    fieldname="rdir",
                                                    adjust_orientation=True)
            # data_rdirs_field = advanced_field_loader(data_rdirs_filename,
            #                                          time_slice=None,
            #                                          fieldname="rdir",
            #                                          adjust_orientation=True)
            for coords in zip(*(np.nonzero(ref_rdirs_field.get_data() == 0.0))):
                utilities.expand_catchment_to_include_rivermouths(ref_rdirs_field.get_data(),
                                                                  ref_catchment_field.get_data(),
                                                                  coords)
            # for coords in zip(*(np.nonzero(data_rdirs_field.get_data() == 0.0))):
            #     utilities.expand_catchment_to_include_rivermouths(data_rdirs_field.get_data(),
            #                                                       data_catchment_field.get_data(),
            #                                                       coords)
            iodriver.advanced_field_writer(ref_expanded_catchment_filename,
                                           ref_catchment_field,
                                           fieldname="catch")
            # iodriver.advanced_field_writer(data_expanded_catchment_filename,
            #                                data_catchment_field,
            #                                fieldname="catch")

    def remove_no_data_values_from_upscaled_MERIT_correction_set(self):
        input_upscaled_correction_set_filename = ("/Users/thomasriddick/Documents/data/analysis_data/"
                                                  "orogcorrsfields/orog_corrs_field_generate_"
                                                  "corrections_for_upscaled_MeritHydro_0k_20220326_121152_rn.nc")
        output_upscaled_correction_set_filename = ("/Users/thomasriddick/Documents/data/analysis_data/"
                                                   "orogcorrsfields/orog_corrs_field_generate_"
                                                   "corrections_for_upscaled_MeritHydro_0k_20220326_121152_rn_adjusted.nc")
        upscaled_orography_filename = ("/Users/thomasriddick/Documents/data/HDdata/"
                                       "orographys/tarasov_upscaled/"
                                       "MERITdem_hydroupscaled_to_10min_global_g.nc")
        upscaled_correction_set = advanced_field_loader(input_upscaled_correction_set_filename,
                                                        time_slice=None,
                                                        fieldname="orog",
                                                        adjust_orientation=True)
        upscaled_orography = advanced_field_loader(upscaled_orography_filename,
                                                   time_slice=None,
                                                   fieldname="z",
                                                   adjust_orientation=True)
        upscaled_correction_set.get_data()[upscaled_orography.get_data() == -9999.0] = 0.0
        iodriver.advanced_field_writer(output_upscaled_correction_set_filename,
                                       upscaled_correction_set,
                                       fieldname="orog")

    def remove_disconnected_points_from_slm(self):
        input_lsmask_filename = ("/Users/thomasriddick/Documents/data/"
                                 "simulation_data/lake_transient_data/run_1/"
                                 "10min_slm_0_old.nc")
        output_lsmask_filename =  ("/Users/thomasriddick/Documents/data/"
                                   "simulation_data/lake_transient_data/run_1/"
                                   "10min_slm_0_disconnected_points_removed.nc")
        input_ls_seed_points_list_filename = ("/Users/thomasriddick/Documents/data/HDdata/"
                                              "lsseedpoints/lsseedpoints_downscale_HD_ls_seed"
                                              "_points_to_10min_lat_lon_true_seas_inc_casp_only_20160718_114402.txt")
        ccld.advanced_connected_lsmask_creation_driver(input_lsmask_filename,
                                                       output_lsmask_filename,
                                                       input_lsmask_fieldname="slm",
                                                       output_lsmask_fieldname="slm",
                                                       input_ls_seed_points_list_filename =
                                                       input_ls_seed_points_list_filename,
                                                       use_diagonals_in=True,
                                                       rotate_seeds_about_polar_axis=True,
                                                       flip_seeds_ud=False,
                                                       adjust_lsmask_orientation=False)


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
    #lake_drivers.extract_volumes_for_transient_run()
    #lake_drivers.add_10min_rmouth_to_transient_data()
    #lake_drivers.expand_transient_data_catchments_to_include_rmouth()
    #lake_drivers.remove_no_data_values_from_upscaled_MERIT_correction_set()
    #lake_drivers.remove_disconnected_points_from_slm()

if __name__ == '__main__':
    main()
