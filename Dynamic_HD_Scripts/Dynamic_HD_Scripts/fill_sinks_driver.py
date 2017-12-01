'''
This function acts as driver for the Cython fill_sinks_wrapper.pyx module that interfaces to the
fill_sinks C++ routines/program

Created on Mar 14, 2016

@author: thomasriddick
'''

import dynamic_hd
import numpy as np
import libs.fill_sinks_wrapper as fill_sinks_wrapper #@UnresolvedImportError
from Dynamic_HD_Scripts.field import Field

def generate_orography_with_sinks_filled(input_orography_filename,output_orography_filename,
                                         ls_mask_filename=None,truesinks_filename=None,
                                         flip_ud=True,flip_lsmask_ud=True,
                                         flip_truesinks_ud=True,grid_type='HD',
                                         add_slight_slope_when_filling_sinks=True,
                                         slope_param=0.1,**grid_kwargs):
    """Generate an orography with the sinks filled using a priority flood type technique
    Input: 
    input_orography_filename: string; the full path to the file containing the input orography
    output_orography_filename: string; the full path to the target file to write the output orography to
    ls_mask_filename: string; the full path to the file containing the input landsea mask.None means ignore.
    truesinks_filename: string; the full path to the file containing the true sinks to use. None means ignore.
    flip_ud: boolean; Flip the orography upside down before processing
    flip_lsmask_ud: boolean; Flip the landsea mask upside down before processing
    flip_truesinks_ud: boolean; Flip the true sinks upside down before processing
    grid_type: string; keyword for the grid type being used
    add_slight_slope_when_filling_sinks: boolean; a small increment to the height of each successive cell when 
        filling a sink to give sink a slight slope
    slope_param: double; the small increment to add if adding a slight slope when fillings sinks
    **grid_kwargs: keyword dictionary; paramaters of the grid being used (if necessary) 
    returns: nothing
    
    Load and manipulates necessary fields (making dummy field where necessary) then calls C++ sink 
    filling algorithm through cython to perform actual sink filling and finally write out the result.
    """

    orography = dynamic_hd.load_field(input_orography_filename,
                                      file_type=dynamic_hd.get_file_extension(input_orography_filename),
                                      field_type='Orography', grid_type=grid_type,**grid_kwargs)
    if flip_ud:
        orography.flip_data_ud()
    grid_dims=orography.get_grid().get_grid_dimensions()
    if not truesinks_filename:
        truesinks = Field(np.empty((1,1),dtype=np.int32),grid='HD')
        use_true_sinks = False;
    else:
        use_true_sinks = True;
        truesinks = dynamic_hd.load_field(truesinks_filename,
                                          file_type=dynamic_hd.\
                                          get_file_extension(truesinks_filename),
                                          field_type='Generic', grid_type=grid_type,
                                          **grid_kwargs)
        if flip_truesinks_ud:
            truesinks.flip_data_ud()
    if ls_mask_filename is None:
        use_ls_mask = False
        ls_mask = np.zeros(grid_dims,dtype=np.int32,order='C')
    else:
        use_ls_mask = True
        ls_mask = dynamic_hd.load_field(ls_mask_filename,
                                        file_type=dynamic_hd.get_file_extension(ls_mask_filename),
                                        field_type='Generic',grid_type=grid_type,**grid_kwargs)
        if flip_lsmask_ud:
            ls_mask.flip_data_ud()
    fill_sinks_wrapper.fill_sinks_cpp_func(orography_array=np.ascontiguousarray(orography.get_data(), #@UndefinedVariable
                                                                                dtype=np.float64),
                                           method = 1, 
                                           use_ls_mask = use_ls_mask,
                                           landsea_in = np.ascontiguousarray(ls_mask.get_data(),
                                                                             dtype=np.int32), 
                                           set_ls_as_no_data_flag = False, 
                                           use_true_sinks = use_true_sinks,
                                           true_sinks_in = np.ascontiguousarray(truesinks.get_data(),
                                                                                dtype=np.int32),
                                           add_slope =add_slight_slope_when_filling_sinks,
                                           epsilon = slope_param)
    dynamic_hd.write_field(output_orography_filename,orography,
                           file_type=dynamic_hd.get_file_extension(output_orography_filename))

def generate_sinkless_flow_directions(filename,output_filename,ls_mask_filename=None,
                                      truesinks_filename=None,catchment_nums_filename=None,
                                      flip_ud=True,grid_type='HD',**grid_kwargs):
    """Generate sinkless flow directions from a given orography
   
    Input: 
    filename: string; the full path to the file containing the input orography
    output_filename: string; the full target path for the output file the generated flow direction will
        be written to
    ls_mask_filename: string, the full path to the file containing the input landsea mask. None means ignore.
    truesinks_filename: string, the full path to the file containing the true sinks to use. None means ignore.
    flip_ud: boolean; If true flip the input fields upside down before processing... 
        this affects the directions you get as it affects whether 7 8 9 is up and 1 2 3 is
        down or visa versa
    grid_type: string; keyword for the grid type being used
    **grid_kwargs: keyword dictionary; paramaters of the grid being used (if necessary) 
    returns: nothing
    
    Loads the orography, creates an empty array for the river flow direction of the same size; either loads
    the landsea mask or makes a dummy field for this of the appropriate size (required for the interface to
    work correctly) and then passes this input to fill_sinks_cpp_func of Cython fill_sinks_wrapper to interface
    with the C++ code that actual generates the flow directions according to Algorithm 4 of Barnes et al (2014).
    Finally save the output river direction field.
    """

    orography = dynamic_hd.load_field(filename,
                                      file_type=dynamic_hd.get_file_extension(filename),
                                      field_type='Orography', grid_type=grid_type,**grid_kwargs)
    if flip_ud:
        orography.flip_data_ud()
    grid_dims=orography.get_grid().get_grid_dimensions()
    rdirs = np.zeros(grid_dims,dtype=np.float64,order='C')
    if not truesinks_filename:
        truesinks = Field(np.empty((1,1),dtype=np.int32),grid='HD')
        use_true_sinks = False;
    else:
        use_true_sinks = True;
        truesinks = dynamic_hd.load_field(truesinks_filename,
                                          file_type=dynamic_hd.\
                                          get_file_extension(truesinks_filename),
                                          field_type='Generic', grid_type=grid_type,
                                          **grid_kwargs)
        if flip_ud:
            truesinks.flip_data_ud()
    if ls_mask_filename is None:
        use_ls_mask = False
        ls_mask = np.zeros(grid_dims,dtype=np.int32,order='C')
    else:
        use_ls_mask = True
        ls_mask = dynamic_hd.load_field(ls_mask_filename,
                                        file_type=dynamic_hd.get_file_extension(ls_mask_filename),
                                        field_type='Generic',grid_type=grid_type,**grid_kwargs)
        if flip_ud:
            ls_mask.flip_data_ud()
    catchment_nums = np.zeros(grid_dims,dtype=np.int32,order='C')
    next_cell_lat_index_in = np.zeros(grid_dims,dtype=np.int32,order='C')
    next_cell_lon_index_in = np.zeros(grid_dims,dtype=np.int32,order='C')
    fill_sinks_wrapper.fill_sinks_cpp_func(orography_array=np.ascontiguousarray(orography.get_data(), #@UndefinedVariable
                                                                                dtype=np.float64),
                                           method = 4, 
                                           use_ls_mask = use_ls_mask,
                                           landsea_in = np.ascontiguousarray(ls_mask.get_data(),
                                                                             dtype=np.int32), 
                                           set_ls_as_no_data_flag = False, 
                                           use_true_sinks = use_true_sinks,
                                           true_sinks_in = np.ascontiguousarray(truesinks.get_data(),
                                                                                dtype=np.int32),
                                           next_cell_lat_index_in = next_cell_lat_index_in,
                                           next_cell_lon_index_in = next_cell_lon_index_in,
                                           rdirs_in = rdirs,
                                           catchment_nums_in = catchment_nums,
                                           prefer_non_diagonal_initial_dirs = False) 
    dynamic_hd.write_field(output_filename,Field(rdirs,grid_type,**grid_kwargs),
                           file_type=dynamic_hd.get_file_extension(output_filename))
    if catchment_nums_filename:
        dynamic_hd.write_field(catchment_nums_filename, 
                              field=Field(catchment_nums,grid_type,**grid_kwargs), 
                              file_type=dynamic_hd.get_file_extension(catchment_nums_filename)) 