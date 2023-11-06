project('dynamic HD scripts',['c','cpp','fortran','cython'],
        version: '3.12',
        default_options:['cpp_std=gnu++11','warning_level=0',
                         'optimization=0','default_library=static',
                         'cython_language=cpp'])
add_project_arguments('-DUSE_NETCDFCPP','-DPROCESSED_CELL_COUNTER','-DDARWIN',language:'cpp')
add_project_arguments('-cpp','-c','-fmessage-length=0','-fPIC',language:'fortran')
cpp_src = []
cpp_test_src = []
fortran_src = []
fortran_test_src = ['/Users/thomasriddick/Documents/code/fruit_3.4.1/src_copy/fruit.f90']
cpp_src_path = 'Dynamic_HD_Cpp_Code/src'
fortran_src_path = 'Dynamic_HD_Fortran_Code/src'
cython_src_path = 'Dynamic_HD_Scripts/Dynamic_HD_Scripts/interface/cpp_interface'
cpp_subdirs = ['algorithms','base','drivers','gtest','testing']
foreach cpp_subdir : cpp_subdirs
  subdir(cpp_src_path / cpp_subdir)
endforeach
fortran_subdirs = ['algorithms','base','drivers',
                   'testing','icosohedral_hd_and_lake_model',
                   'latlon_hd_and_lake_model']
foreach fortran_subdir : fortran_subdirs
  subdir(fortran_src_path / fortran_subdir)
endforeach
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
if get_option('build_cython')
  py = import('python').find_installation('/Users/thomasriddick/anaconda3/bin/python')
  numpy_inc = run_command(py,['-c',
    'import numpy as np; print(np.get_include())'],check : true).stdout().strip()
  cython_libraries = ['compute_catchments_wrapper',
                      'create_connected_lsmask_wrapper',
                      'create_orography_wrapper',
                      'determine_river_directions_wrapper',
                      'evaluate_basins_wrapper',
                      'fill_sinks_wrapper',
                      'fill_sinks_wrapper_low_mem',
                      'follow_streams_wrapper',
                      'lake_operators_wrapper',
                      'upscale_orography_wrapper',
		                  'bifurcate_rivers_basic_icon_si_wrapper',
		                  'compute_catchments_icon_si_wrapper',
                      'determine_river_directions_icon_si_wrapper',
                      'sink_filling_icon_si_wrapper']
  foreach cython_library : cython_libraries
    py.extension_module(cython_library,
                        cython_src_path /
                        cython_library + '.pyx',
                        include_directories: [numpy_inc,cpp_inc],
                        link_with: lib_cpp)
  endforeach
endif
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
  fortran_command_line_drivers = ['accumulate_flow_icon_simple_interface',
                                  'cotat_plus_latlon_to_icon_simple_interface',
                                  'cross_grid_mapper_latlon_to_icon_simple_interface',
                                  'icon_to_latlon_landsea_downscaler_simple_interface',
                                  'latlon_to_icon_loop_breaker_simple_interface']
  foreach driver : fortran_command_line_drivers
    executable(driver,
               [fortran_src_path / 'command_line_drivers' / driver + '.f90'],
               include_directories: fortran_inc,
               dependencies: netcdf_fortran_lib,
               link_with: lib_fortran,
               install: false)
  endforeach
endif
