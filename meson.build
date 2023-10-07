project('dynamic HD scripts',['c','cpp','fortran','cython'],
        version: '3.12',
        default_options:['cpp_std=gnu++11','warning_level=everything',
                         'optimization=0','default_library=static',
                         'cython_language=cpp'])
add_project_arguments('-DUSE_NETCDFCPP','-DPROCESSED_CELL_COUNTER','-DDARWIN',language:'cpp')
add_project_arguments('-cpp','-c','-fmessage-length=0','-fPIC',language:'fortran')
cpp_src = []
cpp_test_src = []
fortran_src = ['/Users/thomasriddick/Documents/code/fruit_3.4.1/src_copy/fruit.f90']
fortran_test_src = []
cpp_src_path = 'Dynamic_HD_Cpp_Code/src/'
fortran_src_path = 'Dynamic_HD_Fortran_Code/src/'
subdir(cpp_src_path + 'algorithms')
subdir(cpp_src_path + 'base')
subdir(cpp_src_path + 'drivers')
subdir(cpp_src_path + 'gtest')
subdir(cpp_src_path + 'testing')
subdir(fortran_src_path + 'algorithms')
subdir(fortran_src_path + 'base')
subdir(fortran_src_path + 'drivers')
subdir(fortran_src_path + 'testing')
subdir(fortran_src_path + 'icosohedral_hd_and_lake_model')
subdir(fortran_src_path + 'latlon_hd_and_lake_model')
cpp_inc = include_directories(['Dynamic_HD_Cpp_Code/include',
                               'Dynamic_HD_Cpp_Code/src',
                               'Dynamic_HD_Cpp_Code'])
fortran_inc = include_directories(['Dynamic_HD_Fortran_Code/include'])
cpp = meson.get_compiler('cpp')
netcdf_c_lib = dependency('netcdf', language: 'c')
netcdf_cxx_lib = cpp.find_library('netcdf-cxx4.1.1.0',
                                  dirs : ['/usr/local/Cellar/netcdf-cxx/4.3.1_1/lib'])
netcdf_fortran_lib = dependency('netcdf', language: 'fortran')
lib_cpp = library('dyhd_cpp',cpp_src,
                  include_directories: cpp_inc,
                  dependencies: [netcdf_cxx_lib,netcdf_c_lib])
if get_option('build_unit_tests')
  e = executable('cpp_unit_tests',
                 cpp_test_src,
                 include_directories: cpp_inc,
                 dependencies: [netcdf_cxx_lib,netcdf_c_lib],
                 link_with: lib_cpp,
                 install: false)
  test('Cpp unit tests',e)
endif
if get_option('build_tools')
  cpp_command_line_drivers = {'bifurcate_rivers_basic_icon_simple_interface':
                              'bifurcate_rivers_basic_icon_simple_interface.cpp',
                              'compute_catchments_icon_simple_interface':
                              'compute_catchments_icon_simple_interface.cpp',
                              'determine_river_directions_icon_simple_interface':
                              'determine_river_directions_icon_simple_interface.cpp'}
  foreach driver_name,source : cpp_command_line_drivers
    executable(driver_name,
               [cpp_src_path + 'command_line_drivers/'+source],
               include_directories: cpp_inc,
               dependencies: [netcdf_cxx_lib,netcdf_c_lib],
               link_with: lib_cpp,
               install: false)
  endforeach
endif
py = import('python').find_installation('/Users/thomasriddick/anaconda3/bin/python')
numpy_inc = run_command(py,['-c',
  'import numpy as np; print(np.get_include())'],check : true).stdout().strip()
py.extension_module('fill_sinks_wrapper',
                    'Dynamic_HD_Scripts/Dynamic_HD_Scripts/'+
                    'interface/cpp_interface/'+
                    'fill_sinks_wrapper.pyx',
                    include_directories: [numpy_inc,cpp_inc],
                    link_with: lib_cpp)
lib_fortran = library('dyhd_fortran',fortran_src,
                      include_directories: fortran_inc,
                      dependencies: netcdf_fortran_lib,)
if get_option('build_unit_tests')
  fe = executable('fortran_unit_tests',
                  fortran_test_src,
                  include_directories: fortran_inc,
                  dependencies: netcdf_fortran_lib,
                  link_with: lib_fortran,
                  install: false)
  test('Fortran unit tests',fe)
endif
if get_option('build_tools')
  fortran_command_line_drivers = {'accumulate_flow_icon_simple_interface':
                                  'accumulate_flow_icon_simple_interface.f90',
                                  'cotat_plus_latlon_to_icon_simple_interface':
                                  'cotat_plus_latlon_to_icon_simple_interface.f90',
                                  'cross_grid_mapper_latlon_to_icon_simple_interface':
                                  'cross_grid_mapper_latlon_to_icon_simple_interface.f90',
                                  'icon_to_latlon_landsea_downscaler_simple_interface':
                                  'icon_to_latlon_landsea_downscaler_simple_interface.f90',
                                  'latlon_to_icon_loop_breaker_simple_interface':
                                  'latlon_to_icon_loop_breaker_simple_interface.f90'}
  foreach driver_name,source : fortran_command_line_drivers
    executable(driver_name,
               [fortran_src_path + 'command_line_drivers/'+source],
               include_directories: fortran_inc,
               dependencies: netcdf_fortran_lib,
               link_with: lib_fortran,
               install: false)
  endforeach
endif
