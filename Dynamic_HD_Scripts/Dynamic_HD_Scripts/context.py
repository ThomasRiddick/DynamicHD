'''
Provides local context for Dynamic HD testing scripts when using a laptop
Created on May 17, 2016

@author: thomasriddick
'''

import os
module_path = os.path.abspath(__file__)
module_dir = os.path.dirname(module_path)
workspace_dir = os.path.dirname(os.path.dirname(module_dir))

fortran_source_path = os.path.join(module_dir,"fortran")
shared_object_path = os.path.join(workspace_dir,'lib')
build_path = os.path.join(module_dir,'build')
bin_path = os.path.join(module_dir,'bin')
fortran_project_source_path = os.path.join(workspace_dir,"Dynamic_HD_Fortran_Code","src")
fortran_project_object_path = os.path.join(workspace_dir,"build","libdyhd_fortran.a.p")
fortran_project_include_path = os.path.join(workspace_dir,"build","libdyhd_fortran.a.p")
fortran_project_executable_path = os.path.join(workspace_dir,"Dynamic_HD_Fortran_Code","Release","Dynamic_HD_Fortran_Exec")
bash_scripts_path = os.path.join(workspace_dir,"Dynamic_HD_bash_scripts")
private_bash_scripts_path = os.path.join(bash_scripts_path, "parameter_generation_scripts")
