'''
Contains a class that manages f2py fortran modules
Created on Jan 15, 2016

@author: thomasriddick
'''

import imp
import os
import subprocess
import textwrap
import stat
import socket
import re
import shutil
from numpy.distutils import core as distutils_core
from subprocess import CalledProcessError
from context import shared_object_path, bin_path,build_path
from sys import platform

class f2py_manager(object):
    """Manages a f2py fortran module.

    Public methods:
    set_function_or_subroutine_name
    run_current_fuction_or_subroutine
    run_named_function_or_subroutine
    get_module_signature
    get_named_function_or_subroutine_signature
    get_current_function_or_subroutine_signature

    This module compiles a fortran module using f2py and allows the user to
    run functions or subroutines from it and to get the module, function and
    subroutine signatures.
    """

    mod=None
    func=None
    fortran_module_name=None
    fortran_file_name=None
    shared_object_file_path=None
    remove_wrapper=None
    additional_fortran_files = []
    include_path = []
    wrapper_path=os.path.join(bin_path,'f2py_setup_wrapper.py')

    def __init__(self, fortran_file_name,func_name=None,remove_wrapper=False,
                 additional_fortran_files=None,include_path=None):
        """Class Constructor. Ensure that the fortran module is compiled and then import it.

        Arguments:
        fortran_file_name: string; the full path to the file contain the fortran module
        func_name (optional): string; the name of the function or subroutine to set as the
            current subroutine or module (default None)
        remove_wrapper (optional): boolean; a flag to select if any the wrapper script is cleaned
            up after use or not (default False)
        additional_fortran_files: list of strings, list of names of any additional
            fortran files that need to be included in compilation

        Perform standard setup; checks fortran file exists; extracts module name from filename supplied
        and then checks if fortran module has changed. If so recompile it; if not simply load it (if it
        is compiled the compile method handles the loading). If a function/subroutine name has been
        given then call method to set it as current function.
        """

        self.fortran_file_name = fortran_file_name
        self.remove_wrapper = remove_wrapper
        if not os.path.isfile(self.fortran_file_name):
            raise RuntimeError("Fortran file {0} does not exist".format(self.fortran_file_name))
        self.fortran_module_name = os.path.basename(self.fortran_file_name).split('.')[0]
        self.shared_object_file_path = os.path.join(shared_object_path,
                                                    self.fortran_module_name+'.so')
        self.sources = [self.fortran_file_name]
        if additional_fortran_files is not None:
            self.additional_fortran_files = additional_fortran_files
        if include_path:
            self.include_path = [include_path]
        self.wrapper_text = textwrap.dedent("""\
            import numpy.distutils.core as npdistutils_core
            import os
            import sys
            import shutil
            import argparse
            import ast
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument("--build-path",required=True)
            parser.add_argument("--module-name",required=True)
            parser.add_argument("--sources",required=True)
            parser.add_argument("--include-path",required=True)
            parser.add_argument("--objects",required=True)
            params, unknown_args = parser.parse_known_args()
            sys.argv = [sys.argv[0]] + unknown_args
            path_to_build_dir = os.path.join(params.build_path,params.module_name,
                                             "temp")
            owd = os.getcwd()
            os.chdir(path_to_build_dir)
            if sys.platform == "darwin":
                extra_link_args=['-Wl,-rpath,/usr/local/Cellar/gcc@6/6.5.0_2/lib/gcc/6/']
            else:
                extra_link_args=[]
            ext = npdistutils_core.Extension(name=params.module_name,
                                             sources=ast.literal_eval(params.sources),
                                             include_dirs=ast.literal_eval(params.include_path),
                                             extra_objects=ast.literal_eval(params.objects),
                                             extra_link_args=extra_link_args,
                                             language="f90")
            npdistutils_core.setup(name=params.module_name,
                                   ext_modules=[ext])
            os.chdir(owd)""")
        if self.check_for_changes():
            self.run_f2py_compilation()
        else:
            self.load_module()
        if func_name is not None:
            self.set_function_or_subroutine_name(func_name)

    def __del__(self):
        """Class destructor. If remove_wrapper is set then remove the wrapper script"""
        if os.path.isfile(self.wrapper_path) and self.remove_wrapper:
            os.remove(self.wrapper_path)

    def set_function_or_subroutine_name(self,func_name):
        """Set the name of the function or subroutine to use."""
        fortranmod=getattr(self.mod,self.fortran_module_name.lower())
        self.func = getattr(fortranmod,func_name.lower())

    def run_current_function_or_subroutine(self,*args):
        """Run the current function or subroutine with given arguments."""
        return self.func(*args)

    def run_named_function_or_subroutine(self,func_name,*args):
        """Run a given function or subroutine with given arguments"""
        self.set_function_or_subroutine_name(func_name)
        return self.func(*args)

    def get_module_signature(self):
        """Get the signture of the current fortran module"""
        fortranmod=getattr(self.mod,self.fortran_module_name.lower())
        return fortranmod.__doc__

    def get_named_function_or_subroutine_signature(self,func_name):
        """Get the signature of a named function or subroutine"""
        self.set_function_or_subroutine_name(func_name)
        return self.func.__doc__

    def get_current_function_or_subroutine_signature(self):
        """Get the signature of the current function or subroutine"""
        return self.func.__doc__

    def run_f2py_compilation(self):
        """Compile fortran module using f2py and then load it

        Raises (on error):
        RuntimeError

        First check the compilation wrapper exists and if not create it. Then run
        the compilation within a try-except block, printing the output and any errors.
        Finally load the module.
        """

        if not os.path.isfile(self.wrapper_path):
            self.create_wrapper()
        path_to_build_dir = os.path.join(build_path,self.fortran_module_name,"temp")
        if os.path.isdir(path_to_build_dir):
            shutil.rmtree(path_to_build_dir)
        os.makedirs(path_to_build_dir)
        distutils_core.run_setup(self.wrapper_path,["build_ext",
                                                    "--build-lib={0}".format(shared_object_path),
                                                    "--build-temp={0}".format(os.path.join(build_path,
                                                                                      self.fortran_module_name,
                                                                                      "temp")),
                                                    "--build-path={0}".format(build_path),
                                                    "--module-name={0}".format(self.fortran_module_name),
                                                    "--sources={0}".format(str(self.sources)),
                                                    "--include-path={0}".format(str(self.include_path)),
                                                    "--objects={0}".
                                                    format(str(self.additional_fortran_files))],
                                 stop_after="run")
        self.load_module()

    def check_for_changes(self):
        """Check if fortran source code has changed since shared library was created or not

        Returns:
        A boolean flag - if true then fortran source code has changed since the python shared
        library created from it was compiled; if not then it hasn't
        """

        if not os.path.isfile(self.shared_object_file_path):
            return True
        if (os.path.getmtime(self.fortran_file_name) >
            os.path.getmtime(self.shared_object_file_path)):
            return True
        if self.additional_fortran_files is not None:
            for additional_file in self.additional_fortran_files:
                if(os.path.getmtime(additional_file) >
                   os.path.getmtime(self.shared_object_file_path)):
                    return True
        else:
            return False

    def create_wrapper(self):
        """Create a f2py compilation wrapper and set the correct permissions for it"""
        with open(self.wrapper_path,'w') as f:
            f.write(self.wrapper_text)
        filestatinfo = os.stat(self.wrapper_path)
        os.chmod(self.wrapper_path, filestatinfo.st_mode | stat.S_IEXEC)

    def load_module(self):
        """Load a shared library created for the current fortran module"""
        #There does not appear to any way to correctly reload a shared libary and get
        #changes that have been compiled in to appear without restarting the interpreter
        self.mod = imp.load_dynamic(self.fortran_module_name,self.shared_object_file_path)
