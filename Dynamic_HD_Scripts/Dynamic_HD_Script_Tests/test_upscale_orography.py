'''
Created on Apr 13, 2017

@author: thomasriddick
'''
import unittest
import numpy as np
import Dynamic_HD_Scripts.libs.upscale_orography_wrapper as upscale_orography_wrapper #@UnresolvedImport
from __builtin__ import False

class Test(unittest.TestCase):

    def setUp(self):
        """Unit test setup function"""
        orography_input_data_flat = np.asarray([
            100.0, 90.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 90.0,100.0,100.0,  51.0,100.0, 65.0,100.0,100.0,
            100.0, 91.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 91.0,100.0,100.0,  52.0,100.0, 64.0,100.0,100.0,
            100.0, 92.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 92.0,100.0,100.0,  53.0,100.0, 63.0,100.0,100.0,
            100.0, 93.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 93.0,100.0,100.0,  54.0,100.0, 62.0,100.0,100.0,
            100.0, 94.0,100.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 94.0,100.0,100.0,  55.0,100.0, 61.0,100.0,100.0,
            100.0, 95.0,100.0,100.0,100.0, 100.0,100.0, 54.0,100.0, 60.0,  100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 95.0,100.0,100.0, 100.0, 58.0,100.0,100.0,100.0,
            100.0, 96.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,  100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0, 78.0,  100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0, 78.0, 100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
            100.0, 97.0,100.0,100.0,100.0, 100.0, 52.0,100.0, 53.0,100.0,  100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0, 77.0,100.0,  100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0, 77.0,100.0, 100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
            100.0, 98.0,100.0,100.0,100.0, 100.0, 51.0,100.0, 53.0,100.0,  100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0, 76.0,100.0,  100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0, 76.0,100.0, 100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
            100.0, 99.0,100.0,100.0,100.0, 100.0, 50.0,100.0, 51.0,100.0,  100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0, 75.0,  100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0, 75.0,100.0, 100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
    
            100.0,100.0, 90.0,100.0,100.0,  51.0,100.0, 65.0,100.0,100.0,  100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 90.0,100.0,100.0, 100.0, 55.0,100.0,100.0,100.0,  100.0,100.0, 90.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
            100.0,100.0, 91.0,100.0,100.0,  52.0,100.0, 64.0,100.0,100.0,  100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 91.0,100.0,100.0, 100.0, 54.0,100.0,100.0,100.0,  100.0,100.0, 91.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
            100.0,100.0, 92.0,100.0,100.0,  53.0,100.0, 63.0,100.0,100.0,  100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 92.0,100.0,100.0, 100.0, 53.0,100.0,100.0,100.0,  100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
            100.0,100.0, 93.0,100.0,100.0,  54.0,100.0, 62.0,100.0,100.0,  100.0,100.0, 93.0,100.0,100.0,  48.0, 49.0, 50.0, 51.0, 52.0, 100.0,100.0, 93.0,100.0,100.0, 100.0,100.0, 52.0, 51.0, 50.0,  100.0,100.0, 93.0,100.0,100.0, 100.0, 49.0, 50.0, 51.0, 52.0,
            100.0,100.0, 94.0,100.0,100.0, 100.0, 58.0,100.0,100.0,100.0,  100.0,100.0, 94.0,100.0, 47.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 94.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 94.0,100.0,100.0,  48.0,100.0,100.0,100.0,100.0,
            100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 95.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
            100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 96.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
            100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 97.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
            100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 98.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
            100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 99.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
    
            100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 77.0,100.0,100.0, 100.0,100.0, 92.0,100.0,100.0, 100.0,100.0, 81.0,100.0,100.0, 100.0, 70.0,100.0,100.0,100.0,  100.0,100.0, 77.0,100.0,100.0, 100.0,100.0, 92.0,100.0,100.0,
            100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 78.0, 79.0, 79.0,  79.0, 79.0, 80.0,100.0,100.0, 100.0,100.0, 87.0,100.0,100.0, 100.0, 88.0,100.0,100.0,100.0,  100.0,100.0, 78.0, 79.0, 79.0,  79.0, 79.0, 80.0,100.0,100.0,
            100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 49.0,100.0,100.0,  100.0,100.0,100.0,100.0,100.0, 100.0, 88.0, 82.0,100.0,100.0, 100.0,100.0, 83.0,100.0,100.0, 100.0, 89.0,100.0,100.0,100.0,  100.0,100.0,100.0,100.0,100.0, 100.0, 88.0, 82.0,100.0,100.0,
            100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,  100.0,100.0,100.0,100.0,100.0, 100.0, 91.0,100.0,100.0,100.0, 100.0,100.0, 86.0,100.0,100.0, 100.0, 84.0,100.0,100.0,100.0,  100.0,100.0,100.0,100.0,100.0, 100.0, 91.0,100.0,100.0,100.0,
            100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 60.0,100.0,100.0,   78.0, 78.0,100.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0, 100.0,100.0, 85.0,100.0,100.0, 100.0, 83.0,100.0,100.0,100.0,   78.0, 78.0,100.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0,
            100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 65.0,100.0,100.0,  100.0, 91.5,100.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0, 100.0,100.0, 85.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0,  100.0, 78.0,100.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0,
            100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 55.0,100.0,100.0,  100.0, 78.0,100.0,100.0,100.0, 100.0, 81.0,100.0,100.0,100.0, 100.0,100.0, 85.0,100.0,100.0, 100.0, 81.0,100.0,100.0,100.0,  100.0, 78.0,100.0,100.0,100.0, 100.0, 81.0,100.0,100.0,100.0,
            100.0,100.0,100.0,100.0, 80.0, 100.0,100.0, 50.0,100.0,100.0,  100.0, 78.0,100.0, 79.0, 79.0,  80.0,100.0,100.0,100.0,100.0, 100.0,100.0, 85.0,100.0,100.0, 100.0, 82.0,100.0,100.0,100.0,  100.0, 78.0,100.0, 79.0, 79.0,  80.0,100.0,100.0,100.0,100.0,
            100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 78.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0, 100.0, 84.0, 85.0,100.0,100.0,  80.0,100.0,100.0,100.0,100.0,  100.0,100.0, 78.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
            100.0,100.0,100.0,100.0, 80.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 93.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,  83.0,100.0, 90.0,100.0, 79.0, 100.0,100.0,100.0,100.0,100.0,  100.0,100.0, 92.0,100.0,100.0, 100.0,100.0,100.0,100.0,100.0,
    
            100.0,100.0,100.0,100.0, 59.0, 100.0,100.0,100.0, 90.0,100.0,  100.0,100.0,100.0,100.0, 53.0, 100.0,100.0,100.0, 90.0,100.0, 100.0,100.0,100.0,100.0, 53.0, 100.0,100.0,100.0, 90.0,100.0,  100.0,100.0,100.0,100.0, 53.0, 100.0,100.0,100.0, 90.0,100.0,
            100.0,100.0,100.0, 53.0,100.0, 100.0, 92.0, 91.0,100.0,100.0,  100.0,100.0,100.0, 53.0,100.0, 100.0, 92.0, 91.0,100.0,100.0, 100.0,100.0,100.0, 63.0,100.0, 100.0, 92.0, 91.0,100.0,100.0,  100.0,100.0,100.0, 63.0,100.0, 100.0, 92.0, 91.0,100.0,100.0,
            100.0,100.0, 52.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0,  100.0,100.0, 52.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0, 100.0,100.0, 52.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0,  100.0,100.0, 52.0,100.0,100.0, 100.0, 93.0,100.0,100.0,100.0,
            100.0, 51.0,100.0,100.0,100.0, 100.0,100.0, 93.0, 94.0,100.0,  100.0, 51.0,100.0,100.0,100.0, 100.0,100.0, 93.0, 94.0,100.0, 100.0, 51.0,100.0,100.0,100.0, 100.0,100.0, 93.0, 94.0,100.0,  100.0, 51.0,100.0,100.0,100.0, 100.0,100.0, 93.0, 94.0,100.0,
             50.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,   50.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,  50.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,   50.0,100.0,100.0,100.0,100.0, 100.0,100.0,100.0, 94.0,100.0,
            100.0, 51.0, 52.0, 53.0, 54.0, 100.0,100.0, 95.0,100.0,100.0,  100.0, 51.0, 52.0, 53.0, 54.0, 100.0,100.0, 95.0,100.0,100.0, 100.0, 51.0, 52.0, 53.0, 54.0, 100.0,100.0, 95.0,100.0,100.0,  100.0, 51.0, 52.0, 53.0, 54.0, 100.0,100.0, 95.0,100.0,100.0,
            100.0, 52.0,100.0,100.0, 55.0, 100.0,100.0, 97.0, 96.0,100.0,  100.0, 52.0,100.0,100.0, 55.0, 100.0,100.0, 97.0, 96.0,100.0, 100.0, 52.0,100.0,100.0, 55.0, 100.0,100.0, 97.0, 96.0,100.0,  100.0, 52.0,100.0,100.0, 55.0, 100.0,100.0, 97.0, 96.0,100.0,
            100.0, 53.0,100.0,100.0, 56.0, 100.0, 98.0,100.0,100.0,100.0,  100.0, 53.0,100.0,100.0, 56.0, 100.0, 98.0,100.0,100.0,100.0, 100.0, 53.0,100.0,100.0, 56.0, 100.0, 98.0,100.0,100.0,100.0,  100.0, 53.0,100.0,100.0, 56.0, 100.0, 98.0,100.0,100.0,100.0,
            100.0, 54.0,100.0,100.0, 57.0, 100.0, 98.0,100.0,100.0,100.0,  100.0, 54.0,100.0,100.0, 57.0, 100.0, 98.0,100.0,100.0,100.0, 100.0, 54.0,100.0,100.0, 57.0, 100.0, 98.0,100.0,100.0,100.0,  100.0, 54.0,100.0,100.0, 57.0, 100.0, 98.0,100.0,100.0,100.0,
            100.0, 58.0,100.0,100.0, 60.0, 100.0,100.0, 99.0,100.0,100.0,  100.0, 58.0,100.0,100.0, 60.0, 100.0,100.0, 99.0,100.0,100.0, 100.0, 61.0,100.0,100.0, 60.0, 100.0,100.0, 99.0,100.0,100.0,  100.0, 60.0,100.0,100.0, 60.0, 100.0,100.0, 99.0,100.0,100.0], 
            np.float64,order='C')
        self.orography_input_data = np.reshape(orography_input_data_flat,(40,40),order='C')

        landsea_input_data_flat = np.asarray([
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
    
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,1,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,
    
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,1,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
    
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0],
            np.int32,order='C')
        self.landsea_input_data = np.reshape(landsea_input_data_flat,(40,40),order='C')
        
        true_sinks_data_land = np.asarray([
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
    
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,1,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,1,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
    
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,1,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
    
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0,   0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0],
            np.int32,order='C')
        self.true_sinks_data = np.reshape(true_sinks_data_land,(40,40),order='C')
        self.orography_expected_output = np.asarray([[60.0,78.0,78.0,65.0],
                                                     [99.0,52.0,55.0,52.0],
                                                     [80.0,91.5,87.0,91.0],
                                                     [58.0,53.0,60.0,60.0]])
        self.extreme_case_test_orography = np.asarray([[100.0,100.0,81.0,100.0,100.0, 100.0,100.0,81.0,100.0,100.0],
                                                       [100.0,100.0,87.0, 82.0, 81.0, 100.0,100.0,87.0, 82.0, 81.0],
                                                       [100.0,100.0,83.0,100.0,100.0, 100.0,100.0,83.0,100.0,100.0],
                                                       [100.0,100.0,86.0,100.0,100.0, 100.0,100.0,86.0,100.0,100.0],
                                                       [100.0,100.0,85.0,100.0,100.0, 100.0,100.0,85.0,100.0,100.0],
                                                       
                                                       [100.0,100.0,81.0,100.0,100.0, 100.0,100.0,81.0,100.0,100.0],
                                                       [100.0,100.0,87.0, 82.0, 81.0, 100.0,100.0,87.0, 82.0, 81.0],
                                                       [100.0,100.0,83.0,100.0,100.0, 100.0,100.0,83.0,100.0,100.0],
                                                       [100.0,100.0,86.0,100.0,100.0, 100.0,100.0,86.0,100.0,100.0],
                                                       [100.0,100.0,85.0,100.0,100.0, 100.0,100.0,85.0,100.0,100.0]],
                                                      np.float64,order='C')
    
        self.extreme_case_test_lsmask = np.asarray([[False,False,False,False,False, True,True,True,True,True],
                                                    [False,False,False,False,False, True,True,True,True,True],
                                                    [False,False,False,False,False, True,True,True,True,True],
                                                    [False,False,False,False,False, True,True,True,True,True],
                                                    [False,False,False,False,False, True,True,True,True,True],

                                                    [False,False,False,False,False, True,False,False,False,True],
                                                    [False,False,False,False,False, True,False,False,False,True],
                                                    [False,False,False,False,False, True,False,False,True,True],
                                                    [False,False,False,False,False, True,False,False,True,True],
                                                    [False,False,False,False,False, True,False,False,True,True]],
                                                   np.int32,order='C')
        
        self.extreme_case_test_truesinks = np.asarray([[False,False,False,False,False, False,False,False,False,False],
                                                       [False,False,False,False,False, False,False,False,False,False],
                                                       [False,False,False,False,False, False,False,False,False,False],
                                                       [False,False,False,False,False, False,False,False,False,False],
                                                       [False,False,False,False,False, False,False,False,False,False],

                                                       [True,True,True,True,True, False,True,True,True,False],
                                                       [True,True,True,True,True, False,True,True,True,False],
                                                       [True,True,True,True,True, False,True,True,False,False],
                                                       [True,True,True,True,True, False,True,True,False,False],
                                                       [True,True,True,True,True, False,True,True,False,False]],
                                                      np.int32,order='C')
    
        self.extreme_case_expected_result = np.asarray([[82,-1.79769313e+308],
                                                        [82,83]])
        
        self.second_extreme_case_test_orography = np.asarray([[100.0,100.0,81.0,100.0,100.0, 100.0,100.0,81.0,100.0,100.0],
                                                              [100.0,100.0,87.0, 82.0, 81.0, 100.0,100.0,87.0, 82.0, 81.0],
                                                              [100.0,100.0,83.0,100.0,100.0, 100.0,100.0,83.0,100.0,100.0],
                                                              [100.0,100.0,86.0,100.0,100.0, 100.0,100.0,86.0,100.0,100.0],
                                                              [100.0,100.0,85.0,100.0,100.0, 100.0,100.0,85.0,100.0,100.0],
                                                       
                                                              [100.0,100.0,100.0,100.0,100.0, 100.0,100.0,81.0,100.0,100.0],
                                                              [100.0, 82.0,82.0, 82.0,100.0, 100.0,100.0,87.0, 82.0, 81.0],
                                                              [100.0, 82.0,80.0, 82.0,100.0, 100.0,100.0,83.0,100.0,100.0],
                                                              [100.0, 82.0,82.0, 82.0,100.0, 100.0,100.0,86.0,100.0,100.0],
                                                              [100.0,100.0,100.0,100.0,100.0, 100.0,100.0,85.0,100.0,100.0]],
                                                             np.float64,order='C')
    
        self.second_extreme_case_test_lsmask = np.asarray([[False,False,False,False,False, True,True,True,False,False],
                                                           [False,False,False,False,False, True,True,True,False,False],
                                                           [False,False,False,False,False, True,True,True,False,False],
                                                           [False,False,False,False,False, True,True,True,False,False],
                                                           [False,False,False,False,False, True,True,True,False,False],

                                                           [False,False,False,False,False, True,True,False,False,True],
                                                           [False,False,False,False,False, True,True,True,True,True],
                                                           [False,False,False,False,False, True,True,True,True,True],
                                                           [False,False,False,False,False, True,True,True,True,True],
                                                           [False,False,False,False,False, True,False,False,True,True]],
                                                          np.int32,order='C')
        
        self.second_extreme_case_test_truesinks = np.asarray([[False,False,False,False,False, True,True,True,True,True],
                                                              [False,False,False,False,False, True,True,True,True,True],
                                                              [False,False,False,False,False, True,True,True,True,True],
                                                              [False,False,False,False,False, True,True,True,True,True],
                                                              [False,False,False,False,False, True,True,True,True,True],

                                                              [True,True,True,True,True, False,True,True,True,False],
                                                              [True,True,True,True,True, False,False,True,True,False],
                                                              [True,True,True,True,True, False,False,True,False,False],
                                                              [True,True,True,True,True, False,False,True,False,False],
                                                              [True,True,True,True,True, False,True,True,False,False]],
                                                             np.int32,order='C')
    
        self.second_extreme_case_expected_result = np.asarray([[82.0,100],
                                                               [100,-1.79769313e+308]])

    def testOrographyUpscalingWithFourbyFourCellGrid(self):
        """Test orography upscaling with a 4 by 4 grid of cells"""
        add_slope_in = 0
        epsilon_in = 0.1
        tarasov_separation_threshold_for_returning_to_same_edge_in = 5
        tarasov_min_path_length_in = 3.0
        tarasov_include_corners_in_same_edge_criteria_in = 0
        orography_output = np.zeros((4,4),dtype=np.float64)
        upscale_orography_wrapper.upscale_orography(orography_in=self.orography_input_data, 
                                                    orography_out=orography_output,method=1, 
                                                    landsea_in=self.landsea_input_data,
                                                    true_sinks_in=self.true_sinks_data,
                                                    add_slope_in=add_slope_in, epsilon_in=epsilon_in,
                                                    tarasov_separation_threshold_for_returning_to_same_edge_in=\
                                                    tarasov_separation_threshold_for_returning_to_same_edge_in,
                                                    tarasov_min_path_length_in=tarasov_min_path_length_in,
                                                    tarasov_include_corners_in_same_edge_criteria_in=\
                                                    tarasov_include_corners_in_same_edge_criteria_in)
        np.testing.assert_array_equal(orography_output,self.orography_expected_output,
                                      "Test of orography upscaling code doesn't produce expected result")
        
    def testOrographyUpscalingWithTwobyTwoCellGridExtremeCases(self):
        """Test orography upscaling on an extreme case with a 2 by 2 grid of cells"""
        add_slope_in = 0
        epsilon_in = 0.1
        tarasov_separation_threshold_for_returning_to_same_edge_in = 5
        tarasov_min_path_length_in = 3.0
        tarasov_include_corners_in_same_edge_criteria_in = 0
        orography_output = np.zeros((2,2),dtype=np.float64)
        upscale_orography_wrapper.upscale_orography(orography_in=self.extreme_case_test_orography, 
                                                    orography_out=orography_output,method=1, 
                                                    landsea_in=self.extreme_case_test_lsmask,
                                                    true_sinks_in=self.extreme_case_test_truesinks,
                                                    add_slope_in=add_slope_in, epsilon_in=epsilon_in,
                                                    tarasov_separation_threshold_for_returning_to_same_edge_in=\
                                                    tarasov_separation_threshold_for_returning_to_same_edge_in,
                                                    tarasov_min_path_length_in=tarasov_min_path_length_in,
                                                    tarasov_include_corners_in_same_edge_criteria_in=\
                                                    tarasov_include_corners_in_same_edge_criteria_in)
        np.testing.assert_allclose(orography_output,self.extreme_case_expected_result,0.0001) 
        
    def testOrographyUpscalingWithTwobyTwoCellGridExtremeCasesTwo(self):
        """Test orography upscaling on a second extreme case with a 2 by 2 grid of cells"""
        add_slope_in = 0
        epsilon_in = 0.1
        tarasov_separation_threshold_for_returning_to_same_edge_in = 5
        tarasov_min_path_length_in = 3.0
        tarasov_include_corners_in_same_edge_criteria_in = 0
        orography_output = np.zeros((2,2),dtype=np.float64)
        upscale_orography_wrapper.upscale_orography(orography_in=self.second_extreme_case_test_orography, 
                                                    orography_out=orography_output,method=1, 
                                                    landsea_in=self.second_extreme_case_test_lsmask,
                                                    true_sinks_in=self.second_extreme_case_test_truesinks,
                                                    add_slope_in=add_slope_in, epsilon_in=epsilon_in,
                                                    tarasov_separation_threshold_for_returning_to_same_edge_in=\
                                                    tarasov_separation_threshold_for_returning_to_same_edge_in,
                                                    tarasov_min_path_length_in=tarasov_min_path_length_in,
                                                    tarasov_include_corners_in_same_edge_criteria_in=\
                                                    tarasov_include_corners_in_same_edge_criteria_in)
        np.testing.assert_allclose(orography_output,self.second_extreme_case_expected_result,0.0001) 
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()