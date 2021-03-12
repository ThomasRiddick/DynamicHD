'''
A number of functions related to computing catchments from a field of flow directions.
The function 'main' is intended to be the function called externally

Created on Feb 9, 2016

@author: thomasriddick
'''

import f2py_manager
import dynamic_hd
import numpy as np
import field
import warnings
import os.path as path
import iodriver
import libs.compute_catchments_wrapper as cc_ccp_wrap
from context import fortran_source_path

def compute_catchments_cpp(field,loop_logfile):
    """Compute the unordered catchments on all grid points using c++ code

    Input:
    field: numpy-like; field of river flow directions
    loop_logfile: string; the path to a file in which to store any loops found
    Output:
    A numpy array containing the unordered catchments calculated

    Use a c++ module to calculate the unordered catchements (labelling them by
    order of discovery) on all the grid points in the supplied array of river flow
    direction. Store any loops found in the named logfile. (The axis swap is required
    to ensure the data is processed in the correct orientation). The algorithm used
    is based on a queue structure
    """

    print "Writing circular flow log file to {0}".format(loop_logfile)
    catchments = np.empty(shape=field.shape,dtype=np.int32)
    cc_ccp_wrap.compute_catchments_cpp(catchments,
                                       np.ascontiguousarray(field,dtype=np.float64),
                                       loop_logfile)
    return catchments

def compute_catchments(field,loop_logfile,circ_flow_check_period=1000):
    """Compute the unordered catchments on all grid points

    Input:
    field: numpy-like; field of river flow directions
    loop_logfile: string; the path to a file in which to store any loops found
    circ_flow_check_period (optional): integer; how often (in terms of step
        along a river) to check if the river is going in a loop
    Output:
    An tuple consisting of an numpy array of the number of each of the six catchment
    types found and a second numpy array containing the unordered catchments calculated
    themselves.

    Use a fortran module to calculate the unordered catchements (labelling them by
    order of discovery) on all the grid points in the supplied array of river flow
    direction. Store any loops found in the named logfile. (The axis swap is required
    to ensure the data is processed in the correct orientation).
    """

    f2py_mngr = f2py_manager.f2py_manager(path.join(fortran_source_path,
                                                     "mod_compute_catchments.f90"),
                                          func_name="compute_catchments")
    field = np.swapaxes(np.asarray(field,dtype=np.int64),0,1)
    print "Writing circular flow log file to {0}".format(loop_logfile)
    catchment_types,catchments = \
        f2py_mngr.run_current_function_or_subroutine(field,
                                                     circ_flow_check_period,
                                                     loop_logfile)
    catchments = np.ma.array(np.swapaxes(catchments,0,1))
    return catchment_types,catchments

def check_catchment_types(catchment_types,logfile=None):
    """Check the catchment types returned for potential problems

    Input:
    catchment_types: numpy-like array; the number of each catchment type found
    logfile(optional): string; full path to a logfile to record catchment_type information
    Return: None

    The output is both printed to screen and saved to a file if one is specified. Warnings
    are given for unknown flow directions, flows over the pole and circular flows.
    """

    catchment_type_names = ["coast","ocean","local","unknown"]
    other_type_names     = ["flow over pole","circular flow"]
    output = ""
    for name, count in zip(catchment_type_names,catchment_types[0:4].tolist()):
        output += "Number of {0} type sinks found: {1} \n".format(name,count)
        if name == "flow over pole" and count > 0:
            warnings.warn("Unknown flow direction detected!")
    for name, count in zip(other_type_names,catchment_types[4:6]):
        output += "Number of {0}s found: {1} \n".format(name,count)
        if count > 0:
            warnings.warn("{0} detected!".format(name))
    output = output.rstrip('\n')
    print output
    if logfile:
        with open(logfile,'w') as f:
            print "Logging catchment type counts in file {0}".format(logfile)
            f.write(output)

def renumber_catchments_by_size(catchments,loop_logfile=None):
    """Renumber catchments according to there size in terms of number of cells

    Input:
    catchments: numpy-like array; catchments to be relabelled catchments
    loop_logfile: string; the full path to the existing loop_logfile (optional)
    Returns: Relabelled catchements

    Label catchments in order of desceding size using a fortran function
    to assist a time critical part of the procedure that can't be done in
    numpy effectively. Also opens the existing loop logfile and changes the
    catchment numbers with loops to reflect the new catchment labelling.
    """

    f2py_mngr = f2py_manager.f2py_manager(path.join(fortran_source_path,
                                                    "mod_compute_catchments.f90"),
                                          func_name="relabel_catchments")
    catch_nums = np.arange(np.amax(catchments)+1)
    counts = np.bincount(catchments.flatten())
    catchments_sizes = np.empty(len(catch_nums),
                                dtype=[('catch_nums',int),
                                       ('new_catch_nums',int),
                                       ('counts',int)])
    catchments_sizes['catch_nums'] = catch_nums
    catchments_sizes['counts']    = counts
    catchments_sizes.sort(order='counts')
    catchments_sizes['new_catch_nums'] = np.arange(len(catchments_sizes['catch_nums']),
                                                       0,-1)
    catchments = np.asfortranarray(catchments,np.int32)
    old_to_new_label_map = np.asfortranarray(np.copy(np.sort(catchments_sizes,
                                             order='catch_nums'))['new_catch_nums'],np.int32)
    f2py_mngr.run_current_function_or_subroutine(catchments,
                                                 old_to_new_label_map)
    if loop_logfile is not None:
        with open(loop_logfile,'r') as f:
            f.next()
            loops = [int(line.strip()) for line in f]
        #-1 to account for differing array offset between Fortran and python
        loops = [str(old_to_new_label_map[old_loop_num])+'\n' for old_loop_num in loops]
        with open(loop_logfile,'w') as f:
            f.write('Loops found in catchments:\n')
            f.writelines(loops)
    return catchments

def advanced_main(filename,fieldname,output_filename,output_fieldname,
                  loop_logfile,use_cpp_alg=True):
    rdirs = iodriver.advanced_field_loader(filename,
                                           field_type='Generic',
                                           fieldname=fieldname)
    nlat,nlon = rdirs.get_grid_dimensions()
    if use_cpp_alg:
        catchments = compute_catchments_cpp(rdirs.get_data(),
                                            loop_logfile)
    else:
        catchment_types, catchments = compute_catchments(rdirs.get_data(),loop_logfile)
        check_catchment_types(catchment_types,logfile=path.splitext(output_filename)[0]+".log")
    numbered_catchments = field.Field(renumber_catchments_by_size(catchments,loop_logfile),
                                      grid=rdirs.get_grid())
    iodriver.advanced_field_writer(target_filename=output_filename,field=numbered_catchments,
                                   fieldname=output_fieldname)

def main(filename,output_filename,loop_logfile,use_cpp_code=True,grid_type='HD',**grid_kwargs):
    """Generates a file with numbered catchments from a given river flow direction file

    Inputs:
    filename: string; the input file of river directions
    output_filename: string; the target file for the output numbered catchments
    loop_logfile: string; an input file of catchments with loop to be updated
    use_cpp_alg: bool; use the Cpp code if True otherwise use the Fortran code
    grid_type: string; a keyword giving the type of grid being used
    **grid_kwargs(optional): keyword dictionary; the parameter of the grid to
    be used (if required)
    Returns: Nothing

    Produces the numbered catchments where the numbering is in descending order of size;
    also update the loop log file to reflect the relabelling of catchements and runs a
    check on the type of catchments generated (which are placed in a log file with the
    same basename as the output catchments but with the extension '.log').
    """

    rdirs = dynamic_hd.load_field(filename,
                                  file_type=dynamic_hd.get_file_extension(filename),
                                  field_type='Generic',grid_type=grid_type,**grid_kwargs)
    if use_cpp_code:
        catchments = compute_catchments_cpp(rdirs.get_data(),
                                            loop_logfile)
    else:
        catchment_types, catchments = compute_catchments(rdirs.get_data(),loop_logfile)
        check_catchment_types(catchment_types,logfile=path.splitext(output_filename)[0]+".log")
    numbered_catchments = field.Field(renumber_catchments_by_size(catchments,loop_logfile),
                                      grid=grid_type,
                                      **grid_kwargs)
    dynamic_hd.write_field(filename=output_filename,field=numbered_catchments,
                           file_type=dynamic_hd.get_file_extension(output_filename))
