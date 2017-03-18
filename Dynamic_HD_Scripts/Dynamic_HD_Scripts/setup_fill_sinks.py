'''
Created on Mar 14, 2016

Compiles the C++ program fill_sinks and its cython wrapper fill_sinks_wrapper.pyx

@author: thomasriddick
'''

import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from context import module_dir
import os.path as path
import os
from sys import platform

if platform == "linux" or platform == "linux2":
    extra_compile_args = ['-std=gnu++11']
    extra_links_args = []
elif platform == "darwin":
    #specify gcc as the compiler (which is despite the name actually a version of clang on mac)
    os.environ["CC"] = "/usr/bin/gcc"
    extra_compile_args=['-std=gnu++11','-stdlib=libc++','-mmacosx-version-min=10.7']
    extra_links_args = ['-mmacosx-version-min=10.7']
else:
    raise RuntimeError("Invalid platform detected")

cpp_base = path.normpath(path.join(module_dir,"../../Dynamic_HD_Cpp_Code"))
src = path.join(cpp_base,"src")
include = path.join(cpp_base,"include")

extensions=[Extension("libs.fill_sinks_wrapper",[path.join(module_dir,"fill_sinks_wrapper.pyx"),
                                                 path.join(src,"grid.cpp"),
                                                 path.join(src,"cell.cpp"),
                                                 path.join(src,"sink_filling_algorithm.cpp")],
                      include_dirs=[src,include,np.get_include()],
                      language="c++",
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_links_args),
            Extension("libs.create_connected_lsmask_wrapper",[path.join(module_dir,"create_connected_lsmask_wrapper.pyx"),
                                                              path.join(src,"grid.cpp"),
                                                              path.join(src,"cell.cpp"),
                                                              path.join(src,"connected_lsmask_generation_algorithm.cpp")],
                      include_dirs=[src,include,np.get_include()],
                      language="c++",
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_links_args)]

setup(
      ext_modules = cythonize(extensions)
     )