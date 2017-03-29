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
        temp_dir = os.path.join(data_dir,'temp','temp_workdir')
        try:
            os.stat(temp_dir)
        except:
            os.mkdir(temp_dir)

    def testName(self):
        self.driver.trial_run_using_data_from_new_data_from_virna_2017_version()

if __name__ == "__main__":
    unittest.main()