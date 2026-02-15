'''
Manage MPI processes
Created on Thu 1, 2020

@author: thomasriddick
'''
from mpi4py import MPI
import os
from Dynamic_HD_Scripts.interface.fortran_interface import f2py_manager
import os.path as path
from Dynamic_HD_Scripts.context import fortran_project_source_path,fortran_project_object_path,fortran_project_include_path

def using_mpi():
    use_mpi_in_python = os.environ.get('USE_MPI_IN_PYTHON')
    if use_mpi_in_python is not None:
        return (use_mpi_in_python.lower() == "true" or
                use_mpi_in_python.lower() == "t")
    else:
        return False

class MPICommands:
    EXIT = 0
    RUNCOTATPLUS = 1

class ProcessManager:

    def __init__(self,comm):
        self.comm = comm
        self.commands = {MPICommands.RUNCOTATPLUS:self.run_cotat_plus}

    def wait_for_commands(self):
        command = None
        command  = self.comm.bcast(command, root=0)
        if command == MPICommands.EXIT:
            return
        self.commands[command]()
        self.wait_for_commands()

    def run_cotat_plus(self):
        f2py_mngr = f2py_manager.f2py_manager(path.join(fortran_project_source_path,
                                                        "drivers",
                                                        "cotat_plus_driver_mod.f90"),
                                              func_name="cotat_plus_latlon_f2py_worker_wrapper",
                                              no_compile=True)
        f2py_mngr.\
            run_current_function_or_subroutine()


