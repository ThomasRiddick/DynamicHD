import cdo
import unittest
import os
from Dynamic_HD_Scripts.dynamic_hd_and_dynamic_lake_drivers.dynamic_lake_production_run_driver \
    import Dynamic_Lake_Production_Run_Drivers
from tests.context import data_dir

class Test_Dynamic_HD_Production_Run_Drivers(unittest.TestCase):
    """Test creating hdpara and hdrestart files for production runs"""

    lakepara_offline_run_result_for_comparison_deglac  = os.path.join(data_dir,"unit_test_data",
                                                                      "lakeparas_trial_run_for_mid_deglaciation_20230218_010439_deglac.nc")
    lakestart_offline_run_result_for_comparison_deglac = os.path.join(data_dir,"unit_test_data",
                                                                      "lakestart_trial_run_for_mid_deglaciation_20230326_152521_deglac.nc")
    hdpara_offline_run_result_for_comparison_deglac  = os.path.join(data_dir,"unit_test_data",
                                                                    "hdpara_trial_run_for_mid_deglaciation_20221116_113050_deglac.nc")
    hdstart_offline_run_result_for_comparison_deglac = os.path.join(data_dir,"unit_test_data",
                                                                    "hdstart_trial_run_for_mid_deglaciation_20221116_113050_deglac.nc")
    lakepara_offline_run_result_for_comparison_pd  = os.path.join(data_dir,"unit_test_data",
                                                                  "lakeparas_trial_run_for_present_day_20230218_010605_pd.nc")
    lakestart_offline_run_result_for_comparison_pd = os.path.join(data_dir,"unit_test_data",
                                                                  "lakestart_trial_run_for_present_day_20221115_190605_pd.nc")
    hdpara_offline_run_result_for_comparison_pd  = os.path.join(data_dir,"unit_test_data",
                                                                "hdpara_trial_run_for_present_day_20221115_190605_pd.nc")
    hdstart_offline_run_result_for_comparison_pd = os.path.join(data_dir,"unit_test_data",
                                                                "hdstart_trial_run_for_present_day_20221115_190605_pd.nc")

    def setUp(self):
        """Class constructor. Create a Dynamic_HD_Production_Run_Driver object."""
        self.driver = Dynamic_Lake_Production_Run_Drivers()
        self.cdo_instance = cdo.Cdo()
        self.temp_dirs = [os.path.join(data_dir,'temp','temp_workdir_lake_deglac'),
                          os.path.join(data_dir,'temp','temp_workdir_pd')]
        self.clean_up()
        for temp_dir in self.temp_dirs:
            try:
                os.stat(temp_dir)
            except:
                os.mkdir(temp_dir)

    def tearDown(self):
        """Unit test tear down function"""
        self.clean_up()

    def clean_up(self):
        files_to_remove = ["paragen/bas_k.dat","paragen/global.inp","paragen/over_k.dat",
                           "paragen/over_vel.dat","paragen/riv_k.dat",
                           "paragen/riv_vel.dat","paragen/soil_partab.txt",
                           "paragen/ddir.inp","paragen/hdpara.srv","paragen/over_n.dat",
                           "paragen/paragen.inp","paragen/riv_n.dat","paragen/slope.dat",
                           "10min_catchments.nc","10min_corrected_orog.nc",
                           "10min_flowtocell.nc","10min_flowtorivermouths.nc",
                           "10min_rdirs.nc","10min_lake_volumes.nc",
                           "30min_catchments.nc","30min_connected_catchments.nc",
                           "30min_filled_orog.nc","30min_flowtocell.nc",
                           "30min_flowtorivermouths.nc","30min_pre_loop_removal_catchments.nc",
                           "30min_pre_loop_removal_flowtocell.nc",
                           "30min_pre_loop_removal_flowtorivermouths.nc",
                           "30min_pre_loop_removal_rdirs.nc","30min_rdirs.nc",
                           "30min_unfilled_orog.nc","30minute_filled_orog_temp.dat",
                           "30minute_filled_orog_temp.nc","30minute_ls_mask_temp.dat",
                           "30minute_ls_mask_temp.nc","30minute_river_dirs_temp.dat",
                           "30minute_river_dirs_temp.nc","30min_flowtocell_connected.nc",
                           "30min_flowtorivermouths_connected.nc",
                           "merges_and_redirects_temp.nc",
                           "fields_temp.nc",
                           "basin_catchment_numbers_temp.nc",
                           "loops_10min.log","loops_30min.log"]
        directories_to_remove = self.temp_dirs
        for directory in directories_to_remove:
            for filename in files_to_remove:
                try:
                    os.remove(os.path.join(directory,filename))
                except:
                    pass
            try:
                os.rmdir(os.path.join(directory,"paragen"))
            except:
                pass
            try:
                os.rmdir(directory)
            except:
                pass

    def testTrialUsingPresentDayData(self):
        """Run a trial using ICE6G present data"""
        (output_hdparas_filepath,output_hdstart_filepath,
         output_lakeparas_filepath,output_lakestart_filepath) =\
            self.driver.trial_run_for_present_day()
        #sinfon tests for file existance and validity ... diff does not output an error
        #if file doesn't exist!
        self.cdo_instance.sinfon(input=[output_hdparas_filepath,
                          self.hdpara_offline_run_result_for_comparison_pd])
        hdpara_diff_out = self.cdo_instance.diff(input=[output_hdparas_filepath,
                          self.hdpara_offline_run_result_for_comparison_pd])
        self.cdo_instance.sinfon(input=[output_hdstart_filepath,
                            self.hdstart_offline_run_result_for_comparison_pd])
        hdstart_diff_out  = self.cdo_instance.diff(input=[output_hdstart_filepath,
                            self.hdstart_offline_run_result_for_comparison_pd])
        self.cdo_instance.sinfon(input=[output_lakeparas_filepath,
                                        self.lakepara_offline_run_result_for_comparison_pd])
        lakepara_diff_out = self.cdo_instance.diff(input=[output_lakeparas_filepath,
                                 self.lakepara_offline_run_result_for_comparison_pd])
        self.cdo_instance.sinfon(input=[output_lakestart_filepath,
                                 self.lakestart_offline_run_result_for_comparison_pd])
        lakestart_diff_out  = self.cdo_instance.diff(input=[output_lakestart_filepath,
                              self.lakestart_offline_run_result_for_comparison_pd])
        self.assertTrue(not hdpara_diff_out and hdpara_diff_out is not None,
                        "hdpara file for present day doesn't match expected result")
        self.assertTrue(not hdstart_diff_out and hdstart_diff_out is not None,
                        "hdstart file for present day doesn't match expected result")
        self.assertTrue(not lakepara_diff_out and lakepara_diff_out is not None,
                        "lakepara file for present day doesn't match expected result")
        self.assertTrue(not lakestart_diff_out and lakestart_diff_out is not None,
                        "lakestart file for present day doesn't match expected result")

    def testTrialUsingMidDeglaciationData(self):
        """Run a trial using ICE6G mid deglaciation data"""
        (output_hdparas_filepath,output_hdstart_filepath,
         output_lakeparas_filepath,output_lakestart_filepath) =\
            self.driver.trial_run_for_mid_deglaciation()
        #sinfon tests for file existance and validity ... diff does not output an error
        #if file doesn't exist!
        self.cdo_instance.sinfon(input=[output_hdparas_filepath,
                                 self.hdpara_offline_run_result_for_comparison_deglac])
        hdpara_diff_out = self.cdo_instance.diff(input=[output_hdparas_filepath,
                          self.hdpara_offline_run_result_for_comparison_deglac])
        self.cdo_instance.sinfon(input=[output_hdstart_filepath,
                                 self.hdstart_offline_run_result_for_comparison_deglac ])
        hdstart_diff_out  = self.cdo_instance.diff(input=[output_hdstart_filepath,
                            self.hdstart_offline_run_result_for_comparison_deglac ])
        self.cdo_instance.sinfon(input=[output_lakeparas_filepath,
                                 self.lakepara_offline_run_result_for_comparison_deglac])
        lakepara_diff_out = self.cdo_instance.diff(input=[output_lakeparas_filepath,
                            self.lakepara_offline_run_result_for_comparison_deglac])
        self.cdo_instance.sinfon(input=[output_lakestart_filepath,
                                        self.lakestart_offline_run_result_for_comparison_deglac])
        lakestart_diff_out  = self.cdo_instance.diff(input=[output_lakestart_filepath,
                              self.lakestart_offline_run_result_for_comparison_deglac])
        self.assertTrue(not hdpara_diff_out and hdpara_diff_out is not None,
                        "hdpara file for mid-deglaciation doesn't match expected result")
        self.assertTrue(not hdstart_diff_out and hdstart_diff_out is not None,
                        "hdstart file for mid-deglaciation doesn't match expected result")
        self.assertTrue(not lakepara_diff_out and lakepara_diff_out is not None,
                        "lakepara file for mid-deglaciation doesn't match expected result")
        self.assertTrue(not lakestart_diff_out and lakestart_diff_out is not None,
                        "lakestart file for mid-deglaciation doesn't match expected result")

if __name__ == "__main__":
    unittest.main()
