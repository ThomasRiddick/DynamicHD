'''
Provides local context for Dynamic HD testing scripts when using a laptop
Created on May 12, 2016

@author: thomasriddick
'''

import Dynamic_HD_Scripts.f2py_manager as f2py_man
import inspect
import os.path as path
import os

fortran_source_path = path.join(path.dirname(inspect.getfile(f2py_man)),'fortran')
valgrind_path = "/usr/local/bin/valgrind"
if path.exists(path.expanduser("~/Documents/data")):
    data_dir = path.expanduser("~/Documents/data") 
elif path.exists(path.expanduser("~/data")):
    data_dir = path.expanduser("~/data")
else:
    os.makedirs(path.join(path.expanduser("~"),"data"))