'''
Driving routines for production dynamic HD file generation runs
Created on Mar 18, 2017

@author: thomasriddick
'''
import dynamic_hd_driver as dyn_hd_dr

class Dynamic_HD_Production_Run_Drivers(dyn_hd_dr.Dynamic_HD_Drivers):
    """A class with methods used for running a production run of the dynamic HD generation code"""

    def __init__(self):
        """Class constructor. 
        
        Deliberately does NOT call constructor of Dynamic_HD_Drivers so the many paths
        within the data directory structure used for offline runs is not initialized here
        """
        pass

if __name__ == '__main__':
    pass