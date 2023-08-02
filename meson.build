project('dynamic HD scripts',['cpp','fortran'])
add_project_arguments('-stdlib=libc++','-std=gnu++11',language:'cpp')
add_project_arguments('-cpp','-O0','-g','-Wall','-c','-fmessage-length=0','-fPIC',language:'fortran')
cpp_src = []
fortran_src = ['/Users/thomasriddick/Documents/code/fruit_3.4.1/src/fruit.f90']
subdir('Dynamic_HD_Cpp_Code/src/algorithms')
subdir('Dynamic_HD_Cpp_Code/src/base')
subdir('Dynamic_HD_Cpp_Code/src/drivers')
subdir('Dynamic_HD_Cpp_Code/src/gtest')
subdir('Dynamic_HD_Cpp_Code/src/testing')
subdir('Dynamic_HD_Fortran_Code/src/algorithms')
subdir('Dynamic_HD_Fortran_Code/src/base')
subdir('Dynamic_HD_Fortran_Code/src/drivers')
subdir('Dynamic_HD_Fortran_Code/src/testing')
subdir('Dynamic_HD_Fortran_Code/src/icosohedral_hd_and_lake_model')
subdir('Dynamic_HD_Fortran_Code/src/latlon_hd_and_lake_model')
executable('Dynamic_HD_Cpp_Exec',
           cpp_src,
           include_directories: include_directories(['Dynamic_HD_Cpp_Code/include',
                                                     'Dynamic_HD_Cpp_Code/src',
                                                     'Dynamic_HD_Cpp_Code']),
           install: false)
executable('Dynamic_HD_Fortran_Exec',
           fortran_src,
           install: false)
