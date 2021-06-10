'''
Provides local context for Dynamic HD testing scripts when using a laptop
Created on May 12, 2016

@author: thomasriddick
'''

from Dynamic_HD_Scripts import context as scripts_context
import inspect
import os.path as path
import os

fortran_source_path = scripts_context.fortran_source_path
valgrind_path = "/usr/local/bin/valgrind"
if path.exists(path.expanduser("~/Documents/data")):
    data_dir = path.expanduser("~/Documents/data")
elif path.exists(path.expanduser("~/data")):
    data_dir = path.expanduser("~/data")
else:
    os.makedirs(path.join(path.expanduser("~"),"data"))
