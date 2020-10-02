'''
Manage MPI processes
Created on Thu 1, 2020

@author: thomasriddick
'''
from mpi4py import MPI

def using_mpi():
    return (os.environ.get('USE_MPI_IN_PYTHON').lower() == "true" or
            os.environ.get('USE_MPI_IN_PYTHON').lower() == "t")

class MPICommands(Enum):
    RUNCOTATPLUS = 1

class ProcessManager(object):

    def __init__(self):
        commands = {MPICommands.RUNCOTATPLUS:self.run_cotat_plus}

    def wait_for_commands(self):
        command = None
        command  = comm.bcast(command, root=0)
        self.commands[command]()
        self.wait_for_commands()

    def run_cotat_plus(self):
        f2py_mngr = f2py_manager.f2py_manager(path.join(fortran_project_source_path,
                                                        "cotat_plus_driver_mod.f90"),
                                              no_compile=True)
        f2py_mngr.\
            run_current_function_or_subroutine()


