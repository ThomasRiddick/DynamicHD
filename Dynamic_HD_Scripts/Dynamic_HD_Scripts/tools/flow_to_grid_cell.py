'''
Module for calculating the cumulative flow to cell values from a field of
river directions.

Created on Jan 20, 2016

@author: thomasriddick
'''

import os.path as path
import numpy as np
from Dynamic_HD_Scripts.interface.fortran_interface \
    import f2py_manager as f2py_mg
from Dynamic_HD_Scripts.base import field
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.context import fortran_source_path
from Dynamic_HD_Scripts.context import fortran_project_object_path
from Dynamic_HD_Scripts.context import fortran_project_source_path
from Dynamic_HD_Scripts.context import fortran_project_include_path

def accumulate_flow_icon_single_index(cell_neighbors,
                                      input_river_directions,
                                      input_bifurcated_river_directions=None):
    additional_fortran_filenames = ["Dynamic_HD_Fortran_Code_src_algorithms_accumulate_flow_mod.f90.o",
                                    "Dynamic_HD_Fortran_Code_src_base_coords_mod.f90.o",
                                    "Dynamic_HD_Fortran_Code_src_algorithms_flow_accumulation_algorithm_mod.f90.o",
                                    "Dynamic_HD_Fortran_Code_src_base_convert_rdirs_to_indices.f90.o",
                                    "Dynamic_HD_Fortran_Code_src_base_doubly_linked_list_mod.f90.o",
                                    "Dynamic_HD_Fortran_Code_src_base_doubly_linked_list_link_mod.f90.o",
                                    "Dynamic_HD_Fortran_Code_src_base_subfield_mod.f90.o",
                                    "Dynamic_HD_Fortran_Code_src_base_unstructured_grid_mod.f90.o",
                                    "Dynamic_HD_Fortran_Code_src_base_precision_mod.f90.o"]
    additional_fortran_filepaths = [path.join(fortran_project_object_path,filename) for filename in\
                                    additional_fortran_filenames]
    f2py_mngr = f2py_mg.f2py_manager(path.join(fortran_project_source_path,
                                               "drivers",
                                               "accumulate_flow_driver_mod.f90"),
                                          func_name=
                                          "bifurcated_accumulate_flow_icon_f2py_wrapper" if
                                          input_bifurcated_river_directions is not None else
                                          "accumulate_flow_icon_f2py_wrapper",
                                          additional_fortran_files=additional_fortran_filepaths,
                                          include_path=fortran_project_include_path)
    if input_bifurcated_river_directions is not None:
        paths_map = f2py_mngr.\
        run_current_function_or_subroutine(np.asfortranarray(cell_neighbors),
                                           np.asfortranarray(input_river_directions),
                                           np.asfortranarray(input_bifurcated_river_directions),
                                           *input_river_directions.shape)
    else:
        paths_map = f2py_mngr.\
        run_current_function_or_subroutine(np.asfortranarray(cell_neighbors),
                                           np.asfortranarray(input_river_directions),
                                           *input_river_directions.shape)
    #Make a minor postprocessing correction
    paths_map[np.logical_and(np.logical_or(input_river_directions == 5,
                                           input_river_directions == 0),
                             paths_map == 0)] = 1
    return paths_map

def create_hypothetical_river_paths_map(riv_dirs,lsmask=None,use_f2py_func=True,
                                        use_f2py_sparse_iterator=False,nlat=360,nlong=720,
                                        sparse_fraction=0.5,use_new_method=False):
    """Create map of cumulative flow to cell from a field of river directions

    Inputs:
    riv_dirs: numpy array; an array of river flow directions
    lsmask(optional): numpy array; a landsea mask (with sea points masked)
    use_f2py_func (optional): boolean; Whether to use an iterator written in
        Fortran (True) or in python (False)
    use_f2py_sparse_iterator (optional): boolean; Whether to use a sparse iterator
        (that works on a vector containing the remaining points to process
        rather than the whole river direction array) written in Fortran.
        Independent of use_f2py_func.
    nlat: integer; number of latitude points in the input arrays
    nlong:integer; number of longitude points in the input arrays
    sparse_fraction (optional): float; threshold (as a fraction of the
        points in the array remaining to be processed) for switching
        to the sparse iterator (if that option is selected)
    Returns: Generated flow to cell path map as a numpy array

    Perform a certain amount of preperation of the input arrays (adding a
    one cell border across the top and bottom of the edge of the river
    direction array and the land sea mask and also applying the landsea
    mask if that option is selected). Then select an iterator and iterate
    over the initially empty flow map till it is completely filled; switching
    to the the sparse iterator when the given threshold is reached if that
    option is selected.
    """

    riv_dirs = np.insert(riv_dirs,obj=0,values=np.zeros(nlong), axis=0)
    #nlat+1 because the array is now already nlat+1 elements wide so you want to place
    #the new row after the last row
    riv_dirs = np.insert(riv_dirs,obj=nlat+1,values=np.zeros(nlong), axis=0)
    if lsmask is not None:
        lsmask = np.insert(lsmask,obj=0,values=np.ones(nlong,dtype=bool), axis=0)
        #nlat+1 because the array is now already nlat+1 elements wide so you want to place
        #the new row after the last row
        lsmask = np.insert(lsmask,obj=nlat+1,values=np.ones(nlong,dtype=bool), axis=0)
        riv_dirs = np.ma.array(riv_dirs,mask=lsmask,copy=True,dtype=int).filled(0)
    else:
        riv_dirs = np.array(riv_dirs,copy=True,dtype=int)
    paths_map = np.zeros((nlat+2,nlong),dtype=np.int32,order='F')
    if use_f2py_func and use_new_method:
        additional_fortran_filenames = ["algorithms/accumulate_flow_mod.o",
                                        "base/coords_mod.o",
                                        "algorithms/flow_accumulation_algorithm_mod.o",
                                        "base/convert_rdirs_to_indices.o",
                                        "base/doubly_linked_list_mod.o",
                                        "base/doubly_linked_list_link_mod.o",
                                        "base/subfield_mod.o",
                                        "base/unstructured_grid_mod.o",
                                        "base/precision_mod.o"]
        additional_fortran_filepaths = [path.join(fortran_project_object_path,filename) for filename in\
                                        additional_fortran_filenames]
        f2py_mngr = f2py_mg.f2py_manager(path.join(fortran_project_source_path,
                                                   "drivers",
                                                   "accumulate_flow_driver_mod.f90"),
                                              func_name="accumulate_flow_latlon_f2py_wrapper",
                                              additional_fortran_files=additional_fortran_filepaths,
                                              include_path=fortran_project_include_path)
        paths_map = f2py_mngr.\
        run_current_function_or_subroutine(np.asfortranarray(riv_dirs),
                                           *riv_dirs.shape)
        #Make a minor postprocessing correction
        paths_map[np.logical_and(np.logical_or(riv_dirs == 5,
                                               riv_dirs == 0),
                                 paths_map == 0)] = 1
    else:
        if use_f2py_func:
            f2py_kernel = f2py_mg.f2py_manager(path.join(fortran_source_path,
                                                         'mod_iterate_paths_map.f90'),
                                               func_name='iterate_paths_map')
            iterate_paths_map_function = f2py_kernel.run_current_function_or_subroutine
        else:
            iterate_paths_map_function = iterate_paths_map
        while iterate_paths_map_function(riv_dirs,paths_map,nlat,nlong):
            remaining_points = paths_map.size - np.count_nonzero(paths_map)
            if use_f2py_sparse_iterator and remaining_points/float(paths_map.size) < sparse_fraction:
                f2py_sparse_iterator = f2py_mg.f2py_manager(path.join(fortran_source_path,
                                                                      'mod_iterate_paths_map.f90'),
                                                            func_name='sparse_iterator')
                f2py_sparse_iterator.run_current_function_or_subroutine(riv_dirs,paths_map,nlat,nlong)
                break
    return paths_map[1:-1,:]

def iterate_paths_map(riv_dirs,paths_map,nlat=360,nlong=720):
    """Iterate the process of calculating the flow to cell

    Input:
    riv_dirs: numpy array; array of river flow directions
    paths_map: numpy array; array of values of flow to cell

    Returns: Boolean; return True if there are no more points to
    calculate; otherwise return False. The input paths_map array
    is also changed as result of this function

    Test if there are still cells to be calculated in the input
    flow to cell (paths_map). If so then iterates across the points
    input river directions fields and passes them to a function to
    calculate the inflow to a given cell if necessary. Treat edge
    cases seperately; the input river direction array should have
    an extra 1 cell border of zeros across its top and bottom edge
    and will wrap around the side edges
    """

    if np.count_nonzero(paths_map) == paths_map.size:
        return False
    for i in range(nlat+2):
        for j in range(nlong):
            if i == 0 or i == nlat+1:
                paths_map[i,j] = 1
            elif j == 0:
                paths_map[i,j] = count_accumulated_inflow(np.append(riv_dirs[i-1:i+2,nlong-1:nlong],riv_dirs[i-1:i+2,j:j+2],axis=1),
                                                          np.append(paths_map[i-1:i+2,nlong-1:nlong],paths_map[i-1:i+2,j:j+2],axis=1))
            elif j == nlong-1:
                paths_map[i,j] = count_accumulated_inflow(np.append(riv_dirs[i-1:i+2,j-1:j+1],riv_dirs[i-1:i+2,0:1],axis=1),
                                                          np.append(paths_map[i-1:i+2,j-1:j+1],paths_map[i-1:i+2,0:1],axis=1))
            else:
                paths_map[i,j] = count_accumulated_inflow(riv_dirs[i-1:i+2,j-1:j+2],
                                                          paths_map[i-1:i+2,j-1:j+2])
    return True

def count_accumulated_inflow(riv_dirs_section,paths_map_section):
    """Count the accumulated inflow to a cell if possible

    Input:
    riv_dirs_section: numpy array; 3 by 3 section of river direction data
    centered on the cell in question
    paths_map_section: numpy array; 3 by 3 section of partial complete paths
        maps (i.e. flow to cell) data centered on the cell in question
    Returns: the flow to the central cell if it can be calculated or zero
        if it cannot

    Find if any of the surrounding cell flow into this cell and if so
    there flow to cell has already been defined already. If the flow
    to cell of all the neighbours flowing to this cell has been
    calculated already then calculate the flow to cell for this
    cell (including one to account for this cell itself) return it
    otherwise return zero. Note if this has no neighbours flowing
    to it it is assigned a value of one.
    """

    flow_to_cell = 0
    #Exact opposite across the keypad of the direction values
    inflow_values = np.array([[3, 2, 1],
                             [6, 5, 4],
                             [9, 8, 7]])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                flow_to_cell += 1
                #skip this iteration as flow to self is already counted
                continue
            if inflow_values[i,j] == riv_dirs_section[i,j]:
                if paths_map_section[i,j] != 0:
                    flow_to_cell += paths_map_section[i,j]
                else:
                    return 0
    if flow_to_cell < 1:
        raise RuntimeError('In flow less than 1')
    return flow_to_cell

def advanced_main(rdirs_filename,output_filename,rdirs_fieldname,output_fieldname):
    rdirs = iodriver.advanced_field_loader(rdirs_filename,
                                           field_type="Generic",
                                           fieldname=rdirs_fieldname)
    nlat,nlong = rdirs.get_grid().get_grid_dimensions()
    paths_map = field.Field(create_hypothetical_river_paths_map(riv_dirs=rdirs.get_data(),
                                                                lsmask=None,
                                                                use_f2py_func=True,
                                                                use_f2py_sparse_iterator=True,
                                                                nlat=nlat,
                                                                nlong=nlong),
                            grid=rdirs.get_grid())
    iodriver.advanced_field_writer(target_filename=output_filename,field=paths_map,
                                   fieldname=output_fieldname)

def main(rdirs_filename,output_filename,grid_type,**grid_kwargs):
    """Top level function for cumulative flow to cell flow map generation

    Inputs:
    rdir_filename: string; full path to the file contain the input river
        direction field
    output_filename: string; full path of the target file to write the
        generated cumulative flow to cell field to
    grid_type: string; a keyword specifying the grid type of the input
        and output fields
    **grid_kwargs (optional): keyword dictionary; any parameters of the
        input and output grid that are required
    Returns: Nothing
    """

    rdirs = iodriver.load_field(rdirs_filename,
                                iodriver.get_file_extension(rdirs_filename),
                                "Generic", grid_type=grid_type,**grid_kwargs)
    nlat,nlong = rdirs.get_grid().get_grid_dimensions()
    paths_map = field.Field(create_hypothetical_river_paths_map(riv_dirs=rdirs.get_data(),
                                                                lsmask=None,
                                                                use_f2py_func=True,
                                                                use_f2py_sparse_iterator=True,
                                                                nlat=nlat,
                                                                nlong=nlong),
                            grid=grid_type,
                            **grid_kwargs)
    iodriver.write_field(output_filename,paths_map,
                         iodriver.get_file_extension(output_filename))
