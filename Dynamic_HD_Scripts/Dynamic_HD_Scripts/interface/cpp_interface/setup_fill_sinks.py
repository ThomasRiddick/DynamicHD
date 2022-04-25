'''
Created on Mar 14, 2016

Compiles the C++ program fill_sinks and its cython wrapper fill_sinks_wrapper.pyx

@author: thomasriddick
'''

import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Dynamic_HD_Scripts.context import module_dir
import os.path as path
import os
import socket
from sys import platform

if platform == "linux" or platform == "linux2":
    netcdfc = "/sw/stretch-x64/netcdf/netcdf_c-4.6.1"
    netcdfcxx = "/sw/stretch-x64/netcdf/netcdf_cxx-4.3.0-gccsys"
    extra_compile_args = ['-std=gnu++11',
                          "-isystem" + netcdfcxx + "/include",
                          "-isystem" + netcdfc +"/include"]
    if socket.getfqdn().endswith("atos.local"):
        extra_links_args = ["-shared",
                            "-isystem" + netcdfcxx + "/include",
                            "-isystem" + netcdfc +"/include"]
    else:
        extra_links_args = ['-static-libstdc++',"-shared",
                            "-isystem" + netcdfcxx + "/include",
                            "-isystem" + netcdfc +"/include"]
    if socket.getfqdn().endswith(".hpc.dkrz.de"):
      os.environ["LDSHARED"] = "/sw/rhel6-x64/gcc/binutils-2.26-gccsys/bin/ld"
elif platform == "darwin":
    #specify gcc as the compiler (which is despite the name actually a version of clang on mac)
    os.environ["CC"] = "/usr/bin/gcc"
    os.environ["CFLAGS"]= "-stdlib=libc++"
    extra_compile_args=['-DPROCESSED_CELL_COUNTER','-std=gnu++11','-stdlib=libc++','-mmacosx-version-min=10.7']
    extra_links_args = ['-mmacosx-version-min=10.7']
else:
    raise RuntimeError("Invalid platform detected")

cpp_base = path.normpath(path.join(module_dir,"../../Dynamic_HD_Cpp_Code"))
src = path.join(cpp_base,"src")
include = path.join(cpp_base,"include")

extensions=[Extension("libs.fill_sinks_wrapper",
                      [path.join(module_dir,
                                 "interface","cpp_interface",
                                 "fill_sinks_wrapper.pyx"),
                       path.join(src,"base/grid.cpp"),
                       path.join(src,"base/cell.cpp"),
                       path.join(src,"algorithms/"
                                     "sink_filling_algorithm.cpp")],
                      include_dirs=[src,include,np.get_include()],
                      language="c++",
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_links_args),
            Extension("libs.fill_sinks_wrapper_low_mem",
                      [path.join(module_dir,
                                 "interface","cpp_interface",
                                 "fill_sinks_wrapper_low_mem.pyx"),
                       path.join(src,"base/grid.cpp"),
                       path.join(src,"base/cell.cpp"),
                       path.join(src,"algorithms/"
                                     "sink_filling_algorithm.cpp")],
                      include_dirs=[src,include,np.get_include()],
                      language="c++",
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_links_args),
            Extension("libs.create_connected_lsmask_wrapper",
                      [path.join(module_dir,
                                 "interface","cpp_interface",
                                 "create_connected_lsmask_wrapper.pyx"),
                       path.join(src,"base/grid.cpp"),
                       path.join(src,"base/cell.cpp"),
                       path.join(src,"algorithms/"
                                     "connected_lsmask_generation_algorithm.cpp")],
                      include_dirs=[src,include,np.get_include()],
                      language="c++",
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_links_args),
            Extension("libs.upscale_orography_wrapper",
                      [path.join(module_dir,
                                 "interface","cpp_interface",
                                 "upscale_orography_wrapper.pyx"),
                       path.join(src,"base/grid.cpp"),
                       path.join(src,"base/cell.cpp"),
                       path.join(src,"algorithms/"
                                 "sink_filling_algorithm.cpp")],
                      include_dirs=[src,include,np.get_include()],
                      language="c++",
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_links_args),
            Extension("libs.lake_operators_wrapper",
                      [path.join(module_dir,
                                 "interface","cpp_interface",
                                 "lake_operators_wrapper.pyx"),
                       path.join(src,"base/grid.cpp"),
                       path.join(src,"base/cell.cpp"),
                       path.join(src,"algorithms/"
                                 "lake_filling_algorithm.cpp"),
                       path.join(src,"algorithms/"
                                 "reduce_connected_areas_to_points_algorithm.cpp"),
                       path.join(src,"algorithms/"
                                 "carved_river_direction_burning_algorithm.cpp"),
                       path.join(src,"algorithms/"
                                 "water_redistribution_algorithm.cpp"),
                       path.join(src,"algorithms/"
                                 "connected_lsmask_generation_algorithm.cpp"),
                       path.join(src,"drivers/"
                                 "create_connected_lsmask.cpp")],
                      include_dirs=[src,include,np.get_include()],
                      language="c++",
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_links_args),
            Extension("libs.compute_catchments_wrapper",
                      [path.join(module_dir,
                                 "interface","cpp_interface",
                                 "compute_catchments_wrapper.pyx"),
                       path.join(src,"base/grid.cpp"),
                       path.join(src,"base/cell.cpp"),
                       path.join(src,"algorithms/"
                                     "catchment_computation_algorithm.cpp")],
                      include_dirs=[src,include,np.get_include()],
                      language="c++",
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_links_args),
            Extension("libs.evaluate_basins_wrapper",
                      [path.join(module_dir,
                                 "interface","cpp_interface",
                                 "evaluate_basins_wrapper.pyx"),
                       path.join(src,"base/grid.cpp"),
                       path.join(src,"base/cell.cpp"),
                       path.join(src,"algorithms/"
                                     "sink_filling_algorithm.cpp"),
                       path.join(src,"algorithms/"
                                     "basin_evaluation_algorithm.cpp")],
                      include_dirs=[src,include,np.get_include()],
                      language="c++",
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_links_args),
            Extension("libs.determine_river_directions_wrapper",
                      [path.join(module_dir,
                                 "interface","cpp_interface",
                                 "determine_river_directions_wrapper.pyx"),
                       path.join(src,"base/grid.cpp"),
                       path.join(src,"base/cell.cpp"),
                       path.join(src,"algorithms/"
                                     "river_direction_determination_algorithm.cpp"),],
                      include_dirs=[src,include,np.get_include()],
                      language="c++",
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_links_args),
            Extension("libs.follow_streams_wrapper",
                      [path.join(module_dir,
                                 "interface","cpp_interface",
                                 "follow_streams_wrapper.pyx"),
                       path.join(src,"base/grid.cpp"),
                       path.join(src,"algorithms/"
                                     "stream_following_algorithm.cpp"),],
                      include_dirs=[src,include,np.get_include()],
                      language="c++",
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_links_args),
            Extension("libs.create_orography_wrapper",
                      [path.join(module_dir,
                                 "interface","cpp_interface",
                                 "create_orography_wrapper.pyx"),
                       path.join(src,"base/grid.cpp"),
                       path.join(src,"base/cell.cpp"),
                       path.join(src,"algorithms/"
                                     "orography_creation_algorithm.cpp"),],
                      include_dirs=[src,include,np.get_include()],
                      language="c++",
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_links_args)]

setup(
      ext_modules = cythonize(extensions)
     )
