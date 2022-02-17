'''
Unit test of the production version of the dynamic_hd driver
Created on Mar 24, 2017

@author: thomasriddick
'''

import cdo
import unittest
import os
from Dynamic_HD_Scripts.dynamic_hd_and_dynamic_lake_drivers.dynamic_hd_production_run_driver \
    import Dynamic_HD_Production_Run_Drivers
from Dynamic_HD_Script_Tests.context import data_dir
from Dynamic_HD_Scripts.process_manager import ProcessManager
from Dynamic_HD_Scripts.process_manager import using_mpi
from Dynamic_HD_Scripts.process_manager import MPICommands
from mpi4py import MPI


class Test_Dynamic_HD_Production_Run_Drivers(unittest.TestCase):
    """Test creating hdpara and hdrestart files for production runs"""

    hdpara_offline_run_result_for_comparison_2016_data  = os.path.join(data_dir,
        "HDdata/hdfiles/generated/hd_file_ten_minute_data_from_virna_0k_ALG4"
        "_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170419_125745.nc")
    hdstart_offline_run_result_for_comparison_2016_data = os.path.join(data_dir,
        "HDdata/hdrestartfiles/generated/hd_restart_file_ten_minute_data_from_virna_0k_"
        "ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170419_125745.nc")
    hdpara_offline_run_result_for_comparison_2017_data  = os.path.join(data_dir,
        "HDdata/hdfiles/generated/hd_file_ten_minute_data_from_virna_0k_2017v_ALG4_sinkless_no_true_sinks_"
        "oceans_lsmask_plus_upscale_rdirs_20170420_114224.nc")
    hdstart_offline_run_result_for_comparison_2017_data = os.path.join(data_dir,
        "HDdata/hdrestartfiles/generated/hd_restart_file_ten_minute_data_from_virna_0k_2017v_ALG4_sinkless_no_true"
        "_sinks_oceans_lsmask_plus_upscale_rdirs_20170420_114224.nc")
    hdpara_offline_run_result_for_comparison_13_04_2017_data  = os.path.join(data_dir,
        "HDdata/hdfiles/generated/hd_file_ten_minute_data_from_virna_0k_13_04_2017v_ALG4_sinkless_no_true_sinks"
        "_oceans_lsmask_plus_upscale_rdirs_20170420_115401.nc")
    hdstart_offline_run_result_for_comparison_13_04_2017_data = os.path.join(data_dir,
        "HDdata/hdrestartfiles/generated/hd_restart_file_ten_minute_data_from_virna_0k_13_04_2017v_"
        "ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_rdirs_20170420_115401.nc")
    hdpara_offline_run_result_for_comparison_ice6g_0k_data = os.path.join(data_dir,
        "HDdata/hdfiles/generated/hd_file_ICE6g_0k_ALG4_sinkless_no_true_sinks_oceans"
        "_lsmask_plus_upscale_rdirs_tarasov_orog_corrs_20170612_202721.nc")
    hdstart_offline_run_result_for_comparison_ice6g_0k_data = os.path.join(data_dir,
        "HDdata/hdrestartfiles/generated/hd_restart_file_ICE6g_0k_ALG4_sinkless_no_true_sinks_oceans_"
        "lsmask_plus_upscale_rdirs_tarasov_orog_corrs_20170612_202721.nc")
    hdpara_offline_run_result_for_comparison_ice6g_21k_data = os.path.join(data_dir,
        "HDdata/hdfiles/generated/hd_file_ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask_plus_upscale_"
        "rdirs_tarasov_orog_corrs_20170612_202559.nc")
    hdstart_offline_run_result_for_comparison_ice6g_21k_data = os.path.join(data_dir,
        "HDdata/hdrestartfiles/generated/hd_restart_file_ICE6g_lgm_ALG4_sinkless_no_true_sinks_oceans_lsmask"
        "_plus_upscale_rdirs_tarasov_orog_corrs_20170612_202559.nc")

    def setUp(self):
        """Class constructor. Create a Dynamic_HD_Production_Run_Driver object."""
        self.driver = Dynamic_HD_Production_Run_Drivers()
        self.cdo_instance = cdo.Cdo()
        self.temp_dir = os.path.join(data_dir,'temp','temp_workdir')
        try:
            os.stat(self.temp_dir)
        except:
            os.mkdir(self.temp_dir)

    def tearDown(self):
        """Unit test tear down function"""
        files_to_remove = ["bas_k.dat","global.inp","over_k.dat","over_vel.dat","riv_k.dat",
                           "riv_vel.dat","soil_partab.txt","ddir.inp","hdpara.srv","over_n.dat",
                           "paragen.inp","riv_n.dat","slope.dat",os.path.join(self.temp_dir),"paragen"]
        temp_dir = "/Users/thomasriddick/Documents/data/temp/temp_workdir_2017_data/"
        further_files_to_remove = ["10min_catchments.nc","10min_corrected_orog.nc",
                                   "10min_flow_to_river_mouths.nc","10min_flowtocell.nc",
                                   "10min_flowtorivermouths.nc","10min_rdirs.nc",
                                   "30min_catchments.nc","30min_filled_orog.nc",
                                   "30min_flowtocell.nc","30min_flowtorivermouths.nc",
                                   "30min_pre_loop_removal_catchments.nc",
                                   "30min_pre_loop_removal_flowtocell.nc",
                                   "30min_pre_loop_removal_flowtorivermouths.nc",
                                   "30min_pre_loop_removal_rdirs.nc",
                                   "30min_rdirs.nc","30min_unfilled_orog.nc",
                                   "30minute_filled_orog_temp.dat",
                                   "30minute_filled_orog_temp.nc",
                                   "30minute_ls_mask_temp.dat","30minute_ls_mask_temp.nc",
                                   "30minute_river_dirs_temp.dat","30minute_river_dirs_temp.nc",
                                   "loops.log"]
        files_to_remove.extend([os.path.join(temp_dir,file) for
                                file in further_files_to_remove])
        for filename in files_to_remove:
            try:
                os.remove(filename)
            except:
                pass

    def testTrialUsingNewDataFromVirna2016Version(self):
        """Run a trial using the 2016 version of the new data from Virna"""
        output_hdparas_filepath,output_hdstart_filepath =\
            self.driver.trial_run_using_data_from_new_data_from_virna_2016_version()
        hdpara_diff_out = self.cdo_instance.diff(input=[output_hdparas_filepath,
                                                        self.hdpara_offline_run_result_for_comparison_2016_data])
        hdstart_diff_out = self.cdo_instance.diff(input=[output_hdstart_filepath,
                                                         self.hdstart_offline_run_result_for_comparison_2016_data])
        self.assertTrue(not hdpara_diff_out and hdpara_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdpara file for 2016 data")
        self.assertTrue(not hdstart_diff_out and hdstart_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdpara file for 2016 data")

    def testTrialUsingNewDataFromVirna2017Version(self):
        """Run a trial using the 2017 version of the new data from Virna"""
        output_hdparas_filepath,output_hdstart_filepath =\
            self.driver.trial_run_using_data_from_new_data_from_virna_2017_version()
        hdpara_diff_out = self.cdo_instance.diff(input=[output_hdparas_filepath,
                                                        self.hdpara_offline_run_result_for_comparison_2017_data])
        hdstart_diff_out  = self.cdo_instance.diff(input=[output_hdstart_filepath,
                                                          self.hdstart_offline_run_result_for_comparison_2017_data])
        self.assertTrue(not hdpara_diff_out and hdpara_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdpara file for 2017 data")
        self.assertTrue(not hdstart_diff_out and hdstart_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdstart file for 2017 data")

    def testTrialUsingNewDataFromVirna2017Version2(self):
        """Run a trial using the second 2017 version of the new data from Virna"""
        output_hdparas_filepath,output_hdstart_filepath =\
            self.driver.trial_run_using_data_from_new_data_from_virna_2017_version_2()
        hdpara_diff_out = self.cdo_instance.diff(input=[output_hdparas_filepath,
                                                        self.hdpara_offline_run_result_for_comparison_13_04_2017_data])
        hdstart_diff_out = self.cdo_instance.diff(input=[output_hdstart_filepath,
                                                         self.hdstart_offline_run_result_for_comparison_13_04_2017_data])
        self.assertTrue(not hdpara_diff_out and hdpara_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        "hdpara file for 2017 data version 2")
        self.assertTrue(not hdstart_diff_out and hdstart_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdstart file for 2017 data version 2")

    def testTrialUsingICE6GPresentData(self):
        """Run a trial using ICE6G present day data"""
        output_hdparas_filepath,output_hdstart_filepath =\
            self.driver.trial_run_using_ice6g_present_day_data()
        hdpara_diff_out = self.cdo_instance.diff(input=[output_hdparas_filepath,
                                                        self.hdpara_offline_run_result_for_comparison_ice6g_0k_data])
        hdstart_diff_out  = self.cdo_instance.diff(input=[output_hdstart_filepath,
                                                          self.hdstart_offline_run_result_for_comparison_ice6g_0k_data])
        self.assertTrue(not hdpara_diff_out and hdpara_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdpara file for ICE6G 0k")
        self.assertTrue(not hdstart_diff_out and hdstart_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdstart file for ICE6G 0k")

    def testTrialUsingICE6GLGMData(self):
        """Run a trial using ICE6G LGM day data"""
        output_hdparas_filepath,output_hdstart_filepath =\
            self.driver.trial_run_using_ice6g_lgm_day_data()
        hdpara_diff_out = self.cdo_instance.diff(input=[output_hdparas_filepath,
                                                        self.hdpara_offline_run_result_for_comparison_ice6g_21k_data])
        hdstart_diff_out  = self.cdo_instance.diff(input=[output_hdstart_filepath,
                                                          self.hdstart_offline_run_result_for_comparison_ice6g_21k_data])
        self.assertTrue(not hdpara_diff_out and hdpara_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdpara file for ICE6G 21k")
        self.assertTrue(not hdstart_diff_out and hdstart_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdstart file for ICE6G 21k")

    def testTrialUsingICE6GPresentDataWithSpecifiedFieldnames(self):
        """Run a trial using ICE6G present day data with specified fieldnames"""
        output_hdparas_filepath,output_hdstart_filepath =\
            self.driver.trial_run_using_ice6g_present_day_data_with_specified_fieldnames()
        hdpara_diff_out = self.cdo_instance.diff(input=[output_hdparas_filepath,
                                                        self.hdpara_offline_run_result_for_comparison_ice6g_0k_data])
        hdstart_diff_out  = self.cdo_instance.diff(input=[output_hdstart_filepath,
                                                          self.hdstart_offline_run_result_for_comparison_ice6g_0k_data])
        self.assertTrue(not hdpara_diff_out and hdpara_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdpara file for ICE6G 0k")
        self.assertTrue(not hdstart_diff_out and hdstart_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdstart file for ICE6G 0k")

    def testTrialUsingICE6GLGMDataWithSpecifiedFieldnames(self):
        """Run a trial using ICE6G LGM data with specified fieldnames"""
        output_hdparas_filepath,output_hdstart_filepath =\
            self.driver.trial_run_using_ice6g_lgm_day_data_with_specified_fieldnames()
        hdpara_diff_out = self.cdo_instance.diff(input=[output_hdparas_filepath,
                                                        self.hdpara_offline_run_result_for_comparison_ice6g_21k_data])
        hdstart_diff_out  = self.cdo_instance.diff(input=[output_hdstart_filepath,
                                                          self.hdstart_offline_run_result_for_comparison_ice6g_21k_data])
        self.assertTrue(not hdpara_diff_out and hdpara_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdpara file for ICE6G 21k")
        self.assertTrue(not hdstart_diff_out and hdstart_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdstart file for ICE6G 21k")

    def testTrialUsingICE6GPresentDataAndGridInfo(self):
        """Run a trial using ICE6G present day data"""
        output_hdparas_filepath,output_hdstart_filepath =\
            self.driver.trial_run_using_ice6g_present_day_data_and_grid_info()
        hdpara_diff_out = self.cdo_instance.diff(input=[output_hdparas_filepath,
                                                        self.hdpara_offline_run_result_for_comparison_ice6g_0k_data])
        hdstart_diff_out  = self.cdo_instance.diff(input=[output_hdstart_filepath,
                                                          self.hdstart_offline_run_result_for_comparison_ice6g_0k_data])
        self.assertTrue(not hdpara_diff_out and hdpara_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdpara file for ICE6G 0k")
        self.assertTrue(not hdstart_diff_out and hdstart_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdstart file for ICE6G 0k")

    def testTrialUsingICE6GLGMDataAndGridInfo(self):
        """Run a trial using ICE6G LGM day data"""
        output_hdparas_filepath,output_hdstart_filepath =\
            self.driver.trial_run_using_ice6g_lgm_day_data_and_grid_info()
        hdpara_diff_out = self.cdo_instance.diff(input=[output_hdparas_filepath,
                                                        self.hdpara_offline_run_result_for_comparison_ice6g_21k_data])
        hdstart_diff_out  = self.cdo_instance.diff(input=[output_hdstart_filepath,
                                                          self.hdstart_offline_run_result_for_comparison_ice6g_21k_data])
        self.assertTrue(not hdpara_diff_out and hdpara_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdpara file for ICE6G 21k")
        self.assertTrue(not hdstart_diff_out and hdstart_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdstart file for ICE6G 21k")

    def testTrialUsingICE6GPresentDataWithSpecifiedFieldnamesAndGridInfo(self):
        """Run a trial using ICE6G present day data with specified fieldnames"""
        output_hdparas_filepath,output_hdstart_filepath =\
            self.driver.trial_run_using_ice6g_present_day_data_with_specified_fieldnames_and_grid_info()
        hdpara_diff_out = self.cdo_instance.diff(input=[output_hdparas_filepath,
                                                        self.hdpara_offline_run_result_for_comparison_ice6g_0k_data])
        hdstart_diff_out  = self.cdo_instance.diff(input=[output_hdstart_filepath,
                                                          self.hdstart_offline_run_result_for_comparison_ice6g_0k_data])
        self.assertTrue(not hdpara_diff_out and hdpara_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdpara file for ICE6G 0k")
        self.assertTrue(not hdstart_diff_out and hdstart_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdstart file for ICE6G 0k")

    def testTrialUsingICE6GLGMDataWithSpecifiedFieldnamesAndGridInfo(self):
        """Run a trial using ICE6G LGM data with specified fieldnames"""
        output_hdparas_filepath,output_hdstart_filepath =\
            self.driver.trial_run_using_ice6g_lgm_day_data_with_specified_fieldnames_and_grid_info()
        hdpara_diff_out = self.cdo_instance.diff(input=[output_hdparas_filepath,
                                                        self.hdpara_offline_run_result_for_comparison_ice6g_21k_data])
        hdstart_diff_out  = self.cdo_instance.diff(input=[output_hdstart_filepath,
                                                          self.hdstart_offline_run_result_for_comparison_ice6g_21k_data])
        self.assertTrue(not hdpara_diff_out and hdpara_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdpara file for ICE6G 21k")
        self.assertTrue(not hdstart_diff_out and hdstart_diff_out is not None,
                        "Trial of production script doesn't produce expected"
                        " hdstart file for ICE6G 21k")
						
if __name__ == "__main__":
    if using_mpi():
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            unittest.main(exit=False)
            command = MPICommands.EXIT
            comm.bcast(command, root=0)
        else:
            process_manager = ProcessManager(comm)
            process_manager.wait_for_commands()
    else:
        unittest.main()
