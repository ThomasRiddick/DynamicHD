'''
Unit test of the production version of the dynamic_hd driver
Created on Mar 24, 2017

@author: thomasriddick
'''
import unittest
import os
from Dynamic_HD_Scripts.dynamic_hd_production_run_driver import Dynamic_HD_Production_Run_Drivers
from context import data_dir


class Test_Dynamic_HD_Production_Run_Drivers(unittest.TestCase):
    """Test creating hdpara and hdrestart files for production runs"""

    def setUp(self):
        """Class constructor. Create a Dynamic_HD_Production_Run_Driver object."""
        self.driver = Dynamic_HD_Production_Run_Drivers()
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

    def testName(self):
        self.driver.trial_run_using_data_from_new_data_from_virna_2016_version()

if __name__ == "__main__":
    unittest.main()