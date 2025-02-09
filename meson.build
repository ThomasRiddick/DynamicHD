project('dynamic HD scripts',['c','cpp','fortran','cython'],
        version: '3.13',
        default_options:['cpp_std=gnu++11','warning_level=0',
                         'optimization=0','default_library=static',
                         'cython_language=cpp'])
add_project_arguments('-DPROCESSED_CELL_COUNTER','-DDARWIN',language:'cpp')
add_project_arguments('-cpp','-c','-fmessage-length=0','-fPIC',language:'fortran')
bin_dir = meson.project_source_root() / 'bin'
lib_dir = meson.project_source_root() / 'lib'
cpp_src = []
cpp_test_src = []
fortran_src = []
fortran_l2_lake_model_src = []
fortran_test_src = ['/Users/thomasriddick/Documents/code/fruit_3.4.1/src_copy/fruit.f90']
cpp_src_path = 'Dynamic_HD_Cpp_Code/src'
fortran_src_path = 'Dynamic_HD_Fortran_Code/src'
cython_src_path = 'Dynamic_HD_Scripts/Dynamic_HD_Scripts/interface/cpp_interface'
cpp_subdirs = ['algorithms','base','drivers']
if get_option('build_unit_tests')
  cpp_subdirs += ['gtest','testing']
endif
foreach cpp_subdir : cpp_subdirs
  subdir(cpp_src_path / cpp_subdir)
endforeach
fortran_subdirs = ['algorithms','base','drivers',
                   'testing','icosohedral_hd_and_lake_model',
                   'latlon_hd_and_lake_model','l2_lake_model']
foreach fortran_subdir : fortran_subdirs
  subdir(fortran_src_path / fortran_subdir)
endforeach
cpp_inc = include_directories(['Dynamic_HD_Cpp_Code/include',
                               'Dynamic_HD_Cpp_Code/src',
                               'Dynamic_HD_Cpp_Code'])
fortran_inc = include_directories(['Dynamic_HD_Fortran_Code/include'])
cpp = meson.get_compiler('cpp')
netcdf_fortran_lib = dependency('netcdf', language: 'fortran')
lib_cpp = library('dyhd_cpp',cpp_src,
                  include_directories: cpp_inc)
if get_option('build_unit_tests')
  e = executable('cpp_unit_tests',
                 cpp_test_src,
                 include_directories: cpp_inc,
                 link_with: lib_cpp,
                 install: true,
                 install_dir: bin_dir)
  test('Cpp unit tests',e)
endif
if get_option('build_cython')
  py = import('python').find_installation(
          meson.get_external_property('python_installation_path'))
  py_dependency = py.dependency()
  numpy_inc = run_command(py,['-c',
    'import numpy as np; print(np.get_include())'],check : true).stdout().strip()
  cython_libraries = []
  subdir(cython_src_path)
  foreach cython_library : cython_libraries
    py.extension_module(cython_library,
                        cython_src_path /
                        cython_library + '.pyx',
                        include_directories: [numpy_inc,cpp_inc],
                        dependencies: py_dependency,
                        link_with: lib_cpp,
                        install: true,
                        install_dir: lib_dir)
  endforeach
endif
l2_lake_gen = generator(py,
                        output: '@BASENAME@.f90',
                        arguments: ['@SOURCE_DIR@/utils/preprocessor_' +
                                    'utilities/l2_lake_model_preprocessor.py',
                                    '@INPUT@','@OUTPUT@'])
fortran_src += l2_lake_gen.process(fortran_l2_lake_model_src)
lib_fortran = library('dyhd_fortran',fortran_src,
                      include_directories: fortran_inc,
                      dependencies: netcdf_fortran_lib)
if get_option('build_unit_tests')
  fe = executable('fortran_unit_tests',
                  fortran_test_src,
                  include_directories: fortran_inc,
                  dependencies: netcdf_fortran_lib,
                  link_with: lib_fortran,
                  install: true,
                  install_dir: bin_dir)
  test('Fortran unit tests',fe)
endif
