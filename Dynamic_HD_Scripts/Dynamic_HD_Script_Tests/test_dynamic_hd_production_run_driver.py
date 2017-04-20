'''
Unit test of the production version of the dynamic_hd driver
Created on Mar 24, 2017

@author: thomasriddick
'''
import cdo
import unittest
import os
from Dynamic_HD_Scripts.dynamic_hd_production_run_driver import Dynamic_HD_Production_Run_Drivers
from context import data_dir


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
        files_to_remove = ["bas_k.dat","global.inp","over_k.dat","over_vel.dat","riv_k.dat",
                           "riv_vel.dat","soil_partab.txt","ddir.inp","hdpara.srv","over_n.dat",
                           "paragen.inp","riv_n.dat","slope.dat",os.path.join(self.temp_dir),"paragen"]
        for filename in files_to_remove:
            try:
                os.remove(filename)
            except:
                pass

    def testTrialUsingNewDataFromVirna2016Version(self):
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

if __name__ == "__main__":
    unittest.main()