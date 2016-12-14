'''
Created on Apr 5, 2016

@author: thomasriddick
'''
import dynamic_hd
import numpy as np
import field
import grid
import f2py_manager
import os.path as path
import re
import copy
import netCDF4
import shutil
from context import fortran_source_path
from Dynamic_HD_Scripts.field import makeField

def prepare_hdrestart_field(input_field,resnum_riv_field,ref_resnum_field,is_river_res_field=True):
    """Create a hd restart reservoir content field by adapting an existing hd restart field
    
    Arguments:
    input_field: Field object; the input reservoir content field to adapt
    resnum_riv_field: Field object; the reservoir number field for the new river directions
    ref_resnum_field: Field object; the reservior number field for the reference river 
        directions
    is_river_res_field: boolean; if true process this as a river reservoir content field 
        otherwise process it as a ground flow or overland flow reservoir content field.
    Return: Field object of the generated reservoir content
    
    An hd restart field is prepared using the field from a hd restart files for the present day
    (assuming the reference is field is the present day field) and the reservoir number from the 
    reference field (which indicates whether the cell is a lake (or sea) or if it is a river and
    the reservoir number field for the timeslice this hd restart file is being prepared for.
    The hd restart reservoir field is prepared by
    1) setting any negative values to zero
    2) setting any points that the reference reservoir numbers supplied indicate are really
        zero temporarily to minus one
    3) replacing any lake points in the reference with zero
    4) replacing any sea points in the timeslice in question with zero
    5) replacing all zeros with the value of the highest value neighbor (including 
        the zero itself if that is highest) where possible
    6) replacing all remaining zeros with the average value of postive sized reservoirs
    7) setting the points set to minus one back to zero (thus perserving points that are
       simply zero because they are really dry) 
    8) returning the results
    """
    input_field_copy = input_field.copy()
    input_field.mask_field_with_external_mask(input_field.get_data() < 0.0)
    input_field.fill_mask(0.0)
    true_zero_threshold = 2.0 if is_river_res_field else 0.5
    input_field.mask_field_with_external_mask(np.logical_and(ref_resnum_field.get_data() > true_zero_threshold,
                                                             input_field_copy.get_data() == 0))
    input_field.fill_mask(-1.0)
    if is_river_res_field:
        input_field.mask_field_with_external_mask(resnum_riv_field.get_data() < 2.0 )
        input_field.fill_mask(0.0)
    input_field.mask_field_with_external_mask(resnum_riv_field.get_data() < 0.5)
    input_field.fill_mask(-1.0)
    input_field.replace_zeros_with_highest_valued_neighbor()
    input_field.replace_zeros_with_global_postive_value_average()
    input_field.mask_field_with_external_mask(resnum_riv_field.get_data() < 0.5)
    input_field.fill_mask(0.0)
    input_field.mask_field_with_external_mask(np.logical_and(ref_resnum_field.get_data() > true_zero_threshold,
                                                             input_field_copy.get_data() == 0))
    input_field.fill_mask(0.0)
    return input_field

def prepare_hdrestart_file(dataset_inout,rflow_res_num,ref_rflow_res_num,
                           overland_flow_res_num,ref_overland_flow_res_num,
                           base_flow_res_num,ref_base_flow_res_num,
                           grid_type,**grid_kwargs):
    """Creates an hd restart file by adapting an existing one to a different timeslice
   
    Arguments:
    dataset_inout: netCDF4 Dataset object; the object containing the data to
        adapt and then return
    rflow_res_num: Field object, the river flow reservoir number field
    ref_rflow_res_num: Field object, the reference river flow reservoir number field
    overland_flow_res_num: Field object, the overland flow reservoir number field
    ref_overland_flow_res_num: Field object, the reference overland flow reservoir 
        number field
    base_flow_res_num: Field object, the base flow reservoir number field 
    ref_base_flow_res_num: Field object, the reference base flow reservoir number
        field
    grid_type: string; the code for this grid type 
    grid_kwargs: dictionary; key word arguments specifying parameters of the grid
    Returns: netCDF4 Dataset
    
    Prepares each field in the dataset using the method detailed in the doc string 
    for prepare_hdrestart_field. Ignore lat and lon field which obviously remain 
    unchanged.
    """
    for var_name,var_obj in dataset_inout.variables.iteritems():
        if var_name == 'lat' or var_name == 'lon':
            continue
        reservoir_field = makeField(np.array(var_obj),'ReservoirSize',grid_type,**grid_kwargs)
        #baseflow
        if var_name == 'FGMEM':
            reservoir_field=prepare_hdrestart_field(input_field=reservoir_field,
                                            resnum_riv_field=base_flow_res_num,
                                            ref_resnum_field=ref_base_flow_res_num,
                                            is_river_res_field=False)
        #Grid box in flow
        elif var_name == 'FINFL':
            reservoir_field=prepare_hdrestart_field(input_field=reservoir_field,
                                                    resnum_riv_field=overland_flow_res_num,
                                                    ref_resnum_field=ref_overland_flow_res_num,
                                                    is_river_res_field=False)
        #overland flow
        elif var_name == 'FLFMEM':
            reservoir_field=prepare_hdrestart_field(input_field=reservoir_field,
                                                    resnum_riv_field=overland_flow_res_num,
                                                    ref_resnum_field=ref_overland_flow_res_num,
                                                    is_river_res_field=False)
        #river flow
        elif var_name.startswith('FRFMEM'):
            reservoir_field=prepare_hdrestart_field(input_field=reservoir_field,
                                                    resnum_riv_field=rflow_res_num,
                                                    ref_resnum_field=ref_rflow_res_num,
                                                    is_river_res_field=True)
        else:
            raise RuntimeError("Unknown field in hd restart file to transform")
        #Slicing of whole array is critical, otherwise var_obj becomes an numpy array instead of 
        #a netcdf4 variable
        var_obj[:,:] = reservoir_field.get_data()
    return dataset_inout
        
        
def prepare_hdrestart_file_driver(base_hdrestart_filename,output_hdrestart_filename,
                                  hdparas_filename,ref_hdparas_filename,timeslice=None,
                                  res_num_data_rotate180lr=False,res_num_data_flipup=False,
                                  res_num_ref_rotate180lr=False,res_num_ref_flipud=False,
                                  netcdf_dataset_format='NETCDF3_CLASSIC',
                                  grid_type='HD',**grid_kwargs):
    """Drive the hd restart file creation code
   
    Arguments:
    base_hdrestart_filename: string; the full path to the hdrestart file to copy and adapt 
    output_hdrestart_filename: string; the full path to write the adapted hd restart file 
        to
    hdparas_filename: string; the full path to a hdparameter file for this timeslice (or
        many timeslices if a timeslice arguments is given)
    ref_hdparas_filename: string; the full path to the reference hd parameters file
    timeslice(optional): integer; timeslice to use from hd parameters fields
    res_num_data_rotate180lr: boolean; rotate the reservoir number field by 
    180 degrees about the pole
    res_num_data_flipup: flip the reservoir number field about the equator
    res_num_ref_rotate180lr: boolean; rotate the reference reservoir number field by 
    180 degrees about the pole
    res_num_ref_flipud: boolean; flip the reference reservoir number field about the
        equator
    netcdf_dataset_format: string; code for the type of NETCDF dataset format to give
        to the netCDF4-dataset method when opening the base hd restart file
    grid_type: string; the code for this grid type 
    grid_kwargs: dictionary; key word arguments specifying parameters of the grid
    Returns: nothing

    Further details of codes working given in docstrings for prepare_hdrestart_file and
    prepare_hdrestart_field. This routine makes of copy of the base hd restart file 
    to work with, handles filehandling and calls prepare_hdrestart_file.
    """

    rflow_res_num= dynamic_hd.load_field(filename=hdparas_filename,
                                         file_type=dynamic_hd.get_file_extension(hdparas_filename),
                                         field_type='Generic', unmask=True, timeslice=timeslice,
                                         fieldname='ARF_N',grid_type=grid_type,**grid_kwargs)
    ref_rflow_res_num= dynamic_hd.load_field(filename=ref_hdparas_filename,
                                             file_type=dynamic_hd.get_file_extension(ref_hdparas_filename),
                                             field_type='Generic', unmask=True, timeslice=None,
                                             fieldname='ARF_N',grid_type=grid_type,**grid_kwargs)
    overland_flow_res_num= dynamic_hd.load_field(filename=hdparas_filename,
                                                 file_type=dynamic_hd.get_file_extension(hdparas_filename),
                                                 field_type='Generic', unmask=True, timeslice=timeslice,
                                                 fieldname='ALF_N',grid_type=grid_type,**grid_kwargs)
    ref_overland_flow_res_num= dynamic_hd.load_field(filename=ref_hdparas_filename,
                                                     file_type=dynamic_hd.get_file_extension(ref_hdparas_filename),
                                                     field_type='Generic', unmask=True, timeslice=None,
                                                     fieldname='ALF_N',grid_type=grid_type,**grid_kwargs)
    #Use overland flow reservoir number for baseflow (as this parameter doesn't exist for baseflow)
    base_flow_res_num = dynamic_hd.load_field(filename=hdparas_filename,
                                              file_type=dynamic_hd.get_file_extension(hdparas_filename),
                                              field_type='Generic', unmask=True, timeslice=timeslice,
                                              fieldname='ALF_N',grid_type=grid_type,**grid_kwargs)
    ref_base_flow_res_num = dynamic_hd.load_field(filename=ref_hdparas_filename,
                                                  file_type=dynamic_hd.get_file_extension(ref_hdparas_filename),
                                                  field_type='Generic', unmask=True, timeslice=None,
                                                  fieldname='ALF_N',grid_type=grid_type,**grid_kwargs)
    if res_num_data_rotate180lr:
        rflow_res_num.rotate_field_by_a_hundred_and_eighty_degrees()
        overland_flow_res_num.rotate_field_by_a_hundred_and_eighty_degrees()
        base_flow_res_num.rotate_field_by_a_hundred_and_eighty_degrees()
    if res_num_data_flipup:
        rflow_res_num.flip_data_ud()
        overland_flow_res_num.flip_data_ud()
        base_flow_res_num.flip_data_ud()
    if res_num_ref_rotate180lr:
        ref_rflow_res_num.rotate_field_by_a_hundred_and_eighty_degrees()
        ref_overland_flow_res_num.rotate_field_by_a_hundred_and_eighty_degrees()
        ref_base_flow_res_num.rotate_field_by_a_hundred_and_eighty_degrees()
    if res_num_ref_flipud:
        ref_rflow_res_num.flip_data_ud()
        ref_overland_flow_res_num.flip_data_ud()
        ref_base_flow_res_num.flip_data_ud()
    #Deliberately avoid copying metadata
    shutil.copy(base_hdrestart_filename,output_hdrestart_filename)
    print 'Writing output to: {0}'.format(output_hdrestart_filename)
    with netCDF4.Dataset(output_hdrestart_filename,mode='a',format=netcdf_dataset_format) as dataset:
        dataset=prepare_hdrestart_file(dataset,rflow_res_num,ref_rflow_res_num,overland_flow_res_num,
                                       ref_overland_flow_res_num,base_flow_res_num,ref_base_flow_res_num,
                                       grid_type,**grid_kwargs)

def change_dtype(input_filename,output_filename,new_dtype,grid_type,**grid_kwargs):
    """Change the data type of a field in a file
    
    Arguments:
    input_filename: string; the filename of the input file to read the field from
    output_filename: string; the filename of hte output file to write the field with
        its new datatype to
    new_dtype: numpy datatype; the datatype to change the field to 
    grid_type: string; the code for this grid type
    grid_kwargs: dictionary; key word arguments specifying parameters of the grid
    Returns: nothing
    
    It is possible to use a new dtype that is identical to the old one; this function
    then effectively does nothing but copy the file.
    """

    field = dynamic_hd.load_field(input_filename, 
                                  file_type=\
                                  dynamic_hd.get_file_extension(input_filename),
                                  field_type='Generic',
                                  unmask=False,
                                  timeslice=None,
                                  grid_type=grid_type,
                                  **grid_kwargs)
    field.change_dtype(new_dtype)
    dynamic_hd.write_field(output_filename,field,
                           file_type=dynamic_hd.get_file_extension(output_filename))

def invert_ls_mask(original_ls_mask_filename,
                   inverted_ls_mask_filename,
                   timeslice=None,
                   grid_type='HD',**grid_kwargs):
    """Invert a landsea mask, i.e. change 1s to zeros and visa-versa
    
    Arguments:
    original_ls_mask_filename: file to load input field from 
    inverted_ls_mask_filename: file to write field with change data type to
    timeslice(optional): timeslice of the input file load the field from 
    grid_type: string; the code for this grid type 
    grid_kwargs: dictionary; key word arguments specifying parameters of the grid
    Returns"""
    ls_mask = dynamic_hd.load_field(original_ls_mask_filename, 
                                    file_type=\
                                    dynamic_hd.get_file_extension(original_ls_mask_filename),
                                    field_type='Generic',
                                    unmask=False,
                                    timeslice=timeslice,
                                    grid_type=grid_type,
                                    **grid_kwargs)
    ls_mask.invert_data()
    dynamic_hd.write_field(inverted_ls_mask_filename,field=ls_mask,
                           file_type=dynamic_hd.get_file_extension(inverted_ls_mask_filename))

def generate_orog_correction_field(original_orography_filename,
                                   corrected_orography_filename,
                                   orography_corrections_filename,
                                   grid_type,**grid_kwargs):
    """Compare an original and corrected orography to create a field of relative orography corrections
    
    Arguments:
    original_orography_filename: string; the full path to file with the original base orography
    corrected_orography_filename: string; the full path to the file with the absolute corrected
        orography
    orography_corrections_filename: string; full path to write the generated field of relative 
        corrections to
    grid_type: string; the code for this grid type 
    grid_kwargs: dictionary; key word arguments specifying parameters of the grid
    Returns: nothing
    """
    original_orography_field = dynamic_hd.load_field(original_orography_filename, 
                                                     file_type=dynamic_hd.\
                                                     get_file_extension(original_orography_filename), 
                                                     field_type='Orography', grid_type=grid_type,
                                                     **grid_kwargs)
    corrected_orography_field = dynamic_hd.load_field(corrected_orography_filename, 
                                                      file_type=dynamic_hd.\
                                                      get_file_extension(corrected_orography_filename), 
                                                      field_type='Orography', grid_type=grid_type,
                                                      **grid_kwargs)
    corrected_orography_field.subtract(original_orography_field)
    dynamic_hd.write_field(orography_corrections_filename, 
                           corrected_orography_field, 
                           file_type=dynamic_hd.get_file_extension(orography_corrections_filename))
    
def apply_orog_correction_field(original_orography_filename,
                                orography_corrections_filename,
                                corrected_orography_filename,
                                grid_type,**grid_kwargs):
    """Apply a field of relative orography corrections to a base orography
    
    Arguments:
    original_orography_filename: string; full file path to the orography to apply the corrections to
    orography_corrections_filename: string; full file path of the relative orography to apply
    corrected_orography_filename: string; full file path to write the generated absolute corrected
        orography to
    grid_type: string; the code for this grid type 
    grid_kwargs: dictionary; key word arguments specifying parameters of the grid
    Returns: nothing
    """
    original_orography_field = dynamic_hd.load_field(original_orography_filename, 
                                                     file_type=dynamic_hd.\
                                                     get_file_extension(original_orography_filename), 
                                                     field_type='Orography', grid_type=grid_type,
                                                     **grid_kwargs)
    orography_corrections_field =  dynamic_hd.load_field(orography_corrections_filename, 
                                                         file_type=dynamic_hd.\
                                                         get_file_extension(orography_corrections_filename), 
                                                         field_type='Orography', grid_type=grid_type,
                                                         **grid_kwargs)
    original_orography_field.add(orography_corrections_field)
    dynamic_hd.write_field(corrected_orography_filename,
                           original_orography_field,
                           file_type=dynamic_hd.\
                           get_file_extension(corrected_orography_filename))

def generate_ls_mask(orography_filename,ls_mask_filename,sea_level=0.0,
                     grid_type='HD',**grid_kwargs):
    """Generate a land-sea mask from an orography given a sea level
    
    Arguments:
    orography_filename: string; full path to file containing the orography to use
    ls_mask_filename: string; full path to write the generated land sea mask to
    sea_level: float; sea level height to generated landsea mask for
    grid_type: string; the code for this grid type 
    grid_kwargs: dictionary; key word arguments specifying parameters of the grid
    Returns: nothing
    
    Crudely generates a land-sea mask by assumming all points below the specified 
    sea level are sea points, even if they are disconnected from all other sea points
    """
    orography = dynamic_hd.load_field(filename=orography_filename, 
                                      file_type=dynamic_hd.get_file_extension(orography_filename), 
                                      field_type='Orography',
                                      grid_type=grid_type,**grid_kwargs)
    ls_mask = orography.generate_ls_mask(sea_level).astype(dtype=np.int32,order='C')
    dynamic_hd.write_field(filename=ls_mask_filename, 
                           field=field.Field(ls_mask,grid_type,**grid_kwargs), 
                           file_type=dynamic_hd.get_file_extension(ls_mask_filename))
    
def extract_ls_mask_from_rdirs(rdirs_filename,lsmask_filename,grid_type='HD',**grid_kwargs):
    """Extract an land sea mask from river directions with coast and sea cells marked
    
    Arguments:
    rdirs_filename: string, full path to the river directions file to extract the land-sea mask 
        from
    lsmask_filename: string, full path to the target file to the write the extracted land-sea mask
        to
    grid_type: string; the code for this grid type 
    grid_kwargs: dictionary; key word arguments specifying parameters of the grid
    Returns: nothing
    """
    rdirs_field = dynamic_hd.load_field(rdirs_filename, 
                                        file_type=dynamic_hd.get_file_extension(rdirs_filename), 
                                        field_type='RiverDirections', 
                                        grid_type=grid_type,
                                        **grid_kwargs)
    lsmask = field.RiverDirections(rdirs_field.get_lsmask(),grid=grid_type,**grid_kwargs) 
    dynamic_hd.write_field(lsmask_filename, lsmask, file_type=dynamic_hd.get_file_extension(lsmask_filename))

def extract_true_sinks_from_rdirs(rdirs_filename,truesinks_filename,grid_type='HD',**grid_kwargs):
    """Extract a set of true sinks from an rdirs file
    
    Arguments:
    rdirs_filename: string; full path to river directions file containing true sink points
    truesinks_filename: string; full path to write the field of extracted true sinks to
    grid_type: string; the code for this grid type 
    grid_kwargs: dictionary; key word arguments specifying parameters of the grid
    Returns: nothing
    
    Extracts all points that have the code for a sink (which is 5) from the river directions
    field
    """
    rdirs_field = dynamic_hd.load_field(rdirs_filename, 
                                        file_type=dynamic_hd.get_file_extension(rdirs_filename), 
                                        field_type='RiverDirections', 
                                        grid_type=grid_type,
                                        **grid_kwargs)
    truesinks = field.Field(rdirs_field.extract_truesinks(),grid=grid_type,**grid_kwargs)
    dynamic_hd.write_field(truesinks_filename, truesinks,
                           file_type=dynamic_hd.get_file_extension(truesinks_filename))
    
def upscale_field_driver(input_filename,output_filename,input_grid_type,output_grid_type,
                  method,timeslice=None,input_grid_kwargs={},output_grid_kwargs={},scalenumbers=False):
    """Load input, drive the process of upscaling a field using a specified method, and write output
    
    Arguments:
    input_filename: string; full path to file with input fine scale field to upscale
    output_filename: string; full path to file with output course scale upscaled field
    input_grid_type: string; the code for the type of the input fine grid 
    output_grid_type: string; the code for the type of the input course grid 
    method: string; upscaling method to use - see upscale_field for valid methods 
    timeslice(optional): integer; timeslice to upscale if input file contains multiple timeslices 
    input_grid_kwargs: dictionary; key word arguments specifying parameters of the fine input grid
        (if required)
    output_grid_kwargs: dictionary; key word arguments specifying parameters of the course output grid
        (if required)
    scalenumbers: scale numbers according to difference in size between the two grids 
        (assuming a density-like number is being upscaled)
    Returns: nothing
    
    Perform a crude upscaling using the basic named method. This doesn't upscale river direction; 
    this requires much more sophisticated code and is done by a seperate function.
    """
    input_field = dynamic_hd.load_field(input_filename, 
                                        file_type=dynamic_hd.get_file_extension(input_filename), 
                                        field_type='Generic', 
                                        timeslice=timeslice,
                                        grid_type=input_grid_type,
                                        **input_grid_kwargs)
    output_field = upscale_field(input_field,output_grid_type, method,output_grid_kwargs,scalenumbers)
    dynamic_hd.write_field(output_filename,output_field, 
                           file_type=dynamic_hd.get_file_extension(output_filename)) 
    
def upscale_field(input_field,output_grid_type,method,output_grid_kwargs,scalenumbers=False):
    """Upscale a field using a specified method
    
    Arguments:
    input_field: Field object, input field to upscale
    output_grid_type: string; the code for the grid type of the course 
        grid to upscale to
    method: string, upscaling method to use - see defined method below
    output_grid_kwargs: dictionary; key word arguments specifying parameters of
        the course grid type to upscale to (if required)
    scalenumbers: scale numbers according to difference in size between the two grids 
        (assuming a density-like number is being upscaled) 
    Returns: Field object, the upscaled field
    
    Works by manipulating field into a particular 4 dimensional shape then 
    reducing two of those dimensions via a specified reduction operator using 
    wrapped fortran code. 
    """

    try:
        reduction_op = {'Sum':np.sum,'Max':np.amax,
                        'Mode':mode_wrapper,
                        'CheckValue':check_for_value_wrapper}[method]
    except KeyError:
        raise RuntimeError('Invalid upscaling method')
    output_grid = grid.makeGrid(output_grid_type,**output_grid_kwargs)
    nlat_in,nlon_in = input_field.grid.get_grid_dimensions()
    nlat_out,nlon_out = output_grid.get_grid_dimensions()
    if type(input_field.get_grid()) is not grid.LatLongGrid or\
        type(output_grid) is not grid.LatLongGrid:
        raise RuntimeError('Invalid input or output grid; only lat lon grids'
                           ' can be upscaled')
    if nlat_in % nlat_out != 0 or nlon_in % nlon_out != 0:
        raise RuntimeError('Incompatible input and output grid dimensions')
    scalingfactor = (nlat_out*1.0/nlat_in)*(nlon_out*1.0/nlon_in) if scalenumbers else 1 
    reshaped_data_array = input_field.get_data().reshape(nlat_out,nlat_in/nlat_out,
                                                         nlon_out,nlon_in/nlon_out)
    return field.Field(reduction_op(reduction_op(reshaped_data_array,
                                                 axis=3),axis=1)*scalingfactor,output_grid) 

def mode_wrapper(array,axis):
    """Wrapper for the mode finding FORTRAN kernel
    
    Arguments:
    array: ndarray, 3 or 4 dimensional array to be reduced in order to upscale dimension 
    axis: axis to reduce
    Returns: ndarray, reduced 2 or 3 dimensional array
    
    Runs fortran code using f2py. Find modal value of each cell in the given dimension
    """

    array = np.asfortranarray(np.rollaxis(array,axis=axis,start=0))
    axis_lengths = array.shape
    dims = len(axis_lengths)
    f2py_mngr = f2py_manager.f2py_manager(path.join(fortran_source_path,
                                                    "mod_find_mode.f90"),
                                          func_name="find_mode_{0}d".format(dims))
    return f2py_mngr.run_current_function_or_subroutine(array,*axis_lengths)

def check_for_value_wrapper(array,axis,value=5):
    """Wrapper for the check for value FORTRAN kernel
    
    Arguments:
    array: ndarray, 3 or 4 dimensional array to be reduced in order to upscale dimension 
    axis: axis to reduce
    value: integer, value to check for
    Returns: ndarray, reduced 2 or 3 dimensional array
    
    Runs fortran code using f2py. Check to see if a value is present in the reduced dimension,
        if so reduce to that value if not reduce to zero 
    """

    array = np.asfortranarray(np.rollaxis(array,axis=axis,start=0))
    axis_lengths = array.shape
    dims = len(axis_lengths)
    f2py_mngr = f2py_manager.f2py_manager(path.join(fortran_source_path,
                                                    "mod_check_for_value.f90"),
                                          func_name="check_for_value_{0}d".format(dims))
    return f2py_mngr.run_current_function_or_subroutine(array,value,*axis_lengths)
    
def downscale_true_sink_points_driver(input_fine_orography_filename,input_course_truesinks_filename,
                                      output_fine_truesinks_filename,input_fine_orography_grid_type,
                                      input_course_truesinks_grid_type,input_fine_orography_grid_kwargs={},
                                      input_course_truesinks_grid_kwargs={},flip_course_grid_ud=False,
                                      rotate_course_true_sink_about_polar_axis=False,
                                      downscaled_true_sink_modifications_filename=None,
                                      course_true_sinks_modifications_filename=None):
    """Load input, drive the process of downscale a true sinks a field and write output
    
    Argument:
    input_fine_orography_filename: string; full path to input fine orography
    input_course_truesinks_filename: string; full path to input true sinks array file
    output_fine_truesinks_filename: string; full path to target output true sinks array 
        file to write to
    input_fine_orography_grid_type: string; the code for the grid type of the input 
        orography to downscale the true sinks to
    input_course_truesinks_grid_type: string; the code for the grid type of the input
        course grid that the input true sinks are on
    input_fine_orography_grid_kwargs: dictionary; key word arguments specifying 
        parameters of the fine input grid (if required)
    input_course_truesinks_grid_kwargs: dictionary; key word arguments specifying 
        parameters of the course input grid (if required)
    flip_course_grid_ud: boolean; flip the course truesinks field about the
        equator
    rotate_course_true_sink_about_polar_axis: boolean; rotate the course truesinks
    field by 180 degrees about the pole
    downscaled_true_sink_modifications_filename: string; full path to text file (see
        apply_modifications_to_truesinks_field for the correct format) with the 
        modification to apply to the downscaled truesinks
    course_true_sinks_modifications_filename: string; full path to text file (see
        apply_modifications_to_truesinks_field for the correct format) with the
        modification
        to apply to the course truesinks before downscaling
    Returns: nothing
    
    Place the true sinks at the lowest point in the set of fine cells covered by the
    course sell. Can make modifications both to the course true sinks field before 
    processing and the fine true sinks field after processing.
    """

    input_fine_orography_field = dynamic_hd.load_field(input_fine_orography_filename,
                                                       file_type=dynamic_hd.get_file_extension(input_fine_orography_filename), 
                                                       field_type='Orography', 
                                                       grid_type=input_fine_orography_grid_type,
                                                       **input_fine_orography_grid_kwargs)
    input_course_truesinks_field = dynamic_hd.load_field(input_course_truesinks_filename,
                                                         file_type=dynamic_hd.get_file_extension(input_course_truesinks_filename),
                                                         field_type='Generic',
                                                         grid_type=input_course_truesinks_grid_type,
                                                         **input_course_truesinks_grid_kwargs)
    if course_true_sinks_modifications_filename:
        input_course_truesinks_field =\
            field.Field(apply_modifications_to_truesinks_field(truesinks_field=\
                                                               input_course_truesinks_field.get_data(), 
                                                               true_sink_modifications_filename=\
                                                               course_true_sinks_modifications_filename),
                        grid=input_course_truesinks_grid_type,
                        **input_course_truesinks_grid_kwargs)
    if flip_course_grid_ud:
        input_course_truesinks_field.flip_data_ud()
    if rotate_course_true_sink_about_polar_axis:
        input_course_truesinks_field.rotate_field_by_a_hundred_and_eighty_degrees()
    output_fine_truesinks_field = field.Field(downscale_true_sink_points(input_fine_orography_field, 
                                                                         input_course_truesinks_field),
                                              input_fine_orography_grid_type,
                                              **input_fine_orography_grid_kwargs) 
                                              
    if downscaled_true_sink_modifications_filename:
        #Assume that same orientation of grid for course and fine grid is used in mods files 
        #and it the course grid orientation - thus in this case flip fine grid and flip back
        #after
        if flip_course_grid_ud:
            output_fine_truesinks_field.flip_data_ud()
        if rotate_course_true_sink_about_polar_axis:
            output_fine_truesinks_field.rotate_field_by_a_hundred_and_eighty_degrees()
        output_fine_truesinks_field =\
            field.Field(apply_modifications_to_truesinks_field(output_fine_truesinks_field.get_data(), 
                                                               downscaled_true_sink_modifications_filename),
                        input_fine_orography_grid_type,
                        **input_fine_orography_grid_kwargs)
        if rotate_course_true_sink_about_polar_axis:
            output_fine_truesinks_field.rotate_field_by_a_hundred_and_eighty_degrees()
        if flip_course_grid_ud:
            output_fine_truesinks_field.flip_data_ud()
    dynamic_hd.write_field(output_fine_truesinks_filename,
                           output_fine_truesinks_field,
                           file_type=dynamic_hd.get_file_extension(output_fine_truesinks_filename))

def apply_modifications_to_truesinks_field(truesinks_field,
                                           true_sink_modifications_filename):
    """Remove and/or add points to a logical field of true sinks points
    
    Arguments:
    truesinks_field: Field object, field of true sinks to modify
    true_sink_modifications_filename: string; full path to file containing
        list of modifications, see inside method for format.
    Returns: Field object, the modified field of true sinks
    """

    points_to_remove = []
    points_to_add = []
    first_line_pattern  = re.compile(r"^lat *, *lon$")
    second_line_pattern = re.compile(r"^Points to Remove$")
    third_line_pattern  = re.compile(r"^Points to Add$")
    with open(true_sink_modifications_filename) as f:
        if not first_line_pattern.match(f.readline().strip()):
            raise RuntimeError("List of sink point modifications format is incorrect on line 1")
        if not second_line_pattern.match(f.readline().strip()):
            raise RuntimeError("List of sink point modifications format is incorrect on line 2")
        for line in f:
            if third_line_pattern.match(line):
                break
            points_to_remove.append(tuple(int(coord) 
                                          for coord in line.strip().split(",")))
        for line in f:
            points_to_add.append(tuple(int(coord) 
                                       for coord in line.strip().split(",")))
    for point in points_to_remove:
        print 'Removing true sink point at {0},{1}'.format(point[0],point[1])
        truesinks_field[point[0],point[1]] = False
    for point in points_to_add:
        print 'Adding true sink point at {0},{1}'.format(point[0],point[1])
        truesinks_field[point[0],point[1]] = True
    return truesinks_field
   
def downscale_true_sink_points(input_fine_orography_field,input_course_truesinks_field):
    """Downscale a field of true sink points flags
    
    Arguments:
    input_fine_orography_field: Field object, the input fine orography field to place the
        downscaled true sinks in
    input_course_truesinks_field: Field object, the input course true sinks field (as 
        a logical field)
    Returns: Field object contain the logical fine true sinks field
    
    Downscale true sinks by placing each course true sink at the mimima (of height) of
        the set of fine orography pixels covered by the course cell. 
    """

    nlat_fine,nlon_fine = input_fine_orography_field.grid.get_grid_dimensions()
    nlat_course,nlon_course = input_course_truesinks_field.grid.get_grid_dimensions()
    if nlat_course > nlat_fine or nlon_course > nlon_fine:
        raise RuntimeError('Cannot use the downscale true sink points function to perform an upscaling')
    if nlat_fine % nlat_course != 0 or nlon_fine % nlon_course !=0 :
        raise RuntimeError('Incompatible input and output grid dimensions')
    scalingfactor_lat = nlat_fine / nlat_course
    scalingfactor_lon = nlon_fine / nlon_course
    flagged_points_coords = input_course_truesinks_field.get_flagged_points_coords()
    flagged_points_coords_scaled=[]
    for coord_pair in flagged_points_coords:
        lat_scaled = coord_pair[0]*scalingfactor_lat
        lon_scaled = coord_pair[1]*scalingfactor_lon 
        flagged_points_coords_scaled.append((lat_scaled,lon_scaled))
    return input_fine_orography_field.find_area_minima(flagged_points_coords_scaled,
                                                       (scalingfactor_lat,
                                                        scalingfactor_lon))

def downscale_ls_seed_points_list_driver(input_ls_seed_points_list_filename,
                                         output_ls_seed_points_list_filename,
                                         factor, nlat_fine, nlon_fine, 
                                         input_grid_type,
                                         output_grid_type):
    """Downscale a list of land sea mask ocean seeding points
    
    Arguments:
    input_ls_seed_points_list_filename: string, input course land sea point list
    output_ls_seed_points_list_filename: string, output fine land sea point list
    factor: integer, difference in scale between course and fine grid
    nlat_fine: integer, total number of latitude points
    nlon_fine: integer, total number of longitude points
    input_grid_type: string; the code for the type of the input fine grid
    output_grid_type: string; the code for the type of the output course grid
    Return: nothing
    
    Checks that the grid type of the course and fine grid match then downscale
    using downscale_ls_seed_points_list.
    """

    input_points_list = []
    comment_line_pattern = re.compile(r"^ *#.*$")
    with open(input_ls_seed_points_list_filename) as f:
        if f.readline().strip() != input_grid_type:
                raise RuntimeError("List of landsea points being loaded is not for correct grid-type")
        for line in f:
            if comment_line_pattern.match(line):
                continue
            input_points_list.append(tuple(int(coord) for coord in line.strip().split(",")))
    output_points_list = downscale_ls_seed_points_list(input_course_list=input_points_list, 
                                                       downscale_factor=factor, 
                                                       nlat_fine=nlat_fine, 
                                                       nlon_fine=nlon_fine)
    with open(output_ls_seed_points_list_filename,'w') as f:
        f.write(output_grid_type + '\n')
        for entry in output_points_list:
            f.write("{0},{1}\n".format(entry[0],entry[1]))
    
def downscale_ls_seed_points_list(input_course_list,downscale_factor,nlat_fine,nlon_fine):
    """Downscale a list of land sea seed points by scaling their coordinates
    
    Arguments:
    input_course_list: list of tuples, a list of course latitude longitude coordinates
    (latitude first) giving the position of the course land-sea seed points
    downscale_factor: integer, factor to downscale by
    nlat_fine: integer, total number of fine latitude points on the fine grid
    nlon_fine: integer, total number of fine longitude points on the fine grid
    Returns:  list of tuple, a list of fine latitude longitude coordinates (latitude
        first) created by downscaling the course grid points 
    
    Each downscale course point is turned into a course cell size block of 
    fine points. Longitude is simply downscaled. Latitude is also inverted (the
    implied field of points flipped up down) during the process. 
    """

    output_fine_list = []
    for coords in input_course_list:
        for i in range(downscale_factor):
            for j in range(downscale_factor):
                output_fine_list.append((nlat_fine - i - 1 - coords[0]*downscale_factor,
                                         j  + coords[1]*downscale_factor))
    return output_fine_list

def apply_orography_corrections(input_orography_filename,
                                input_corrections_list_filename,
                                output_orography_filename,
                                grid_type,**grid_kwargs):
    """Apply a specified list of corrections to an orography
    
    Arguments
    input_orography_filename: string, full path to the orography to apply the corrections to
    input_corrections_list_filename: string, full path to the file with the list of corrections
        to apply, see inside function for format of header and comment lines
    output_orography_filename: string, full path of target file to write the corrected orography
        to
    grid_type: string; the code for the type of the grid used
    grid_kwargs: dictionary; key word arguments specifying parameters of
        the grid type used
    Returns: nothing
    
    Any manipulations of the field required are specified in the correction list file itself in the
    header and are then read and applied by this method. However the output orography that is written
    out is restored to the same orientation as the input orography.
    """
    orography_field = dynamic_hd.load_field(input_orography_filename,
                                            file_type=dynamic_hd.\
                                            get_file_extension(input_orography_filename), 
                                            field_type='Orography', 
                                            grid_type=grid_type,
                                            **grid_kwargs)
    first_line_pattern = re.compile(r"^grid_type *= *" + grid_type + r"$")
    second_line_pattern = re.compile(r"^flipud *= *(True|true|TRUE|False|false|FALSE)$")
    third_line_pattern = re.compile(r"^rotate180lr *= *(True|true|TRUE|False|false|FALSE)$")
    fourth_line_pattern = re.compile(r"^lat, *lon, *height$")
    comment_line_pattern = re.compile(r"^ *#.*$")
    correction_list = []
    with open(input_corrections_list_filename) as f:
        if not first_line_pattern.match(f.readline().strip()):
                raise RuntimeError("List of corrections being loaded is not for correct grid-type")
        second_line_pattern_match = second_line_pattern.match(f.readline().strip())
        if not second_line_pattern_match:
                raise RuntimeError("List of corrections being loaded has incorrect format on line 2")
        flipud = True if second_line_pattern_match.group(1).lower() == 'true' else False
        third_line_pattern_match = third_line_pattern.match(f.readline().strip())
        if not third_line_pattern_match:
                raise RuntimeError("List of corrections being loaded has incorrect format on line 3")
        rotate180lr = True if third_line_pattern_match.group(1).lower() == 'true' else False
        if not fourth_line_pattern.match(f.readline().strip()):
                raise RuntimeError("List of corrections being loaded has incorrect format on line 4")
        for line in f:
            if comment_line_pattern.match(line):
                continue
            correction_list.append(tuple(int(coord) if i < 2 else float(coord) \
                                         for i,coord in enumerate(line.strip().split(","))))
    if flipud:
        orography_field.flip_data_ud()
    if rotate180lr:
        orography_field.rotate_field_by_a_hundred_and_eighty_degrees()
    for lat,lon,height in correction_list:
        print "Correcting height of lat={0},lon={1} to {2} m".format(lat,lon,height)
        orography_field.get_data()[lat,lon] = height
    #restore to original format so as not to disrupt downstream processing chain
    if flipud:
        orography_field.flip_data_ud()
    if rotate180lr:
        orography_field.rotate_field_by_a_hundred_and_eighty_degrees()
    dynamic_hd.write_field(output_orography_filename,
                           orography_field, 
                           file_type=dynamic_hd.get_file_extension(output_orography_filename))

def downscale_ls_mask_driver(input_course_ls_mask_filename,
                             output_fine_ls_mask_filename,
                             input_flipud=False,
                             input_rotate180lr=False,
                             course_grid_type='HD',fine_grid_type='LatLong10min',
                             course_grid_kwargs={},**fine_grid_kwargs):
    """Drive process of downscaling a land-sea mask
    
    Arguments:
    input_course_ls_mask_filename: string; full path to input course land sea mask file
    output_fine_ls_mask_filename: string; full path to target fine land sea mask file
    input_flipud: boolean, flip the input land sea mask about the equator
    input_rotate180lr: boolean; rotate the input land sea mask by 180 degrees about 
        the pole 
    course_grid_type: string; code for the course grid type of the input field
    fine_grid_type: string;  code for the fine grid type of the output field
    course_grid_kwargs: dictionary; key word argument dictionary for the course grid type
        of the input field (if required)
    fine_grid_kwargs: dictionary; key word arguments dictionary for the fine grid type of
        the output field (if required)
    Returns: nothing
    
    Outflow field orientations is the same as input field orientation.
    """

    input_course_ls_mask_field = dynamic_hd.load_field(input_course_ls_mask_filename,
                                                       file_type=dynamic_hd.\
                                                       get_file_extension(input_course_ls_mask_filename), 
                                                       field_type='Generic', 
                                                       grid_type=course_grid_type,
                                                       **course_grid_kwargs)
    if input_flipud:
        input_course_ls_mask_field.flip_data_ud() 
    if input_rotate180lr:
        input_course_ls_mask_field.rotate_field_by_a_hundred_and_eighty_degrees()
    output_fine_ls_mask_field = downscale_ls_mask(input_course_ls_mask_field, 
                                                  fine_grid_type,**fine_grid_kwargs)
    dynamic_hd.write_field(output_fine_ls_mask_filename,output_fine_ls_mask_field,
                           file_type=dynamic_hd.get_file_extension(output_fine_ls_mask_filename))
    
def downscale_ls_mask(input_course_ls_mask_field,fine_grid_type,**fine_grid_kwargs):
    """Downscale a land-sea mask
    
    Arguments:
    input_course_ls_mask_field: Field object; the input course field to downscale
    fine_grid_type: string; the code of the grid type to downscale to
    fine_grid_kwargs: dictionary; key word dictionary for the grid type to be
        downscaled to (if required)
    Returns: The downscaled field in a Field object
    
    The downscaling is done crudely by assuming that all fine pixels/cells covered by
    a course cell have the same land-sea value (1 or 0) as the course cell itself; thus
    this will produce a blocky land-sea mask on the fine grid with a granular size equal 
    to that of the course grid.
    """

    nlat_fine,nlon_fine = grid.makeGrid(fine_grid_type,**fine_grid_kwargs).get_grid_dimensions()
    nlat_course,nlon_course = input_course_ls_mask_field.grid.get_grid_dimensions()
    if nlat_course > nlat_fine or nlon_course > nlon_fine:
        raise RuntimeError('Cannot use the downscale ls mask function to perform an upscaling')
    if nlat_fine % nlat_course != 0 or nlon_fine % nlon_course !=0 :
        raise RuntimeError('Incompatible input and output grid dimensions')
    scalingfactor_lat = nlat_fine / nlat_course
    scalingfactor_lon = nlon_fine / nlon_course
    #this series of manipulations produces the required 'binary' upscaling
    #outer is the outer product
    output_fine_ls_mask_flatted = np.outer(input_course_ls_mask_field.get_data().flatten(order='F'),
                                           np.ones(scalingfactor_lat,dtype=np.int32)).flatten(order='C')
    output_fine_ls_mask_flatted = np.outer(output_fine_ls_mask_flatted,
                                           np.ones(scalingfactor_lon,dtype=np.int32)).flatten(order='C')
    output_fine_ls_mask = output_fine_ls_mask_flatted.reshape((scalingfactor_lon,nlat_fine,nlon_course),
                                                               order='F')
    output_fine_ls_mask = output_fine_ls_mask.swapaxes(0,1)
    output_fine_ls_mask = output_fine_ls_mask.reshape(nlat_fine,nlon_fine,order='F')
    return field.Field(np.ascontiguousarray(output_fine_ls_mask,dtype=np.int32),grid=fine_grid_type,
                       **fine_grid_kwargs)

def intelligently_burn_orography(input_fine_orography_field,course_orography_field,
                                 input_fine_fmap,threshold,region,
                                 course_grid_type,**course_grid_kwargs):
    """Intelligently burn an orography in a given reason using a given threshold
    
    Arguments:
    input_fine_orography_field: Field object; the input fine orography to take intelligent
        burning height values
    course_orography_field: Field object; the input course orography to applying the burning
        too
    input_fine_fmap: Field object; fine culumalative flow to cell to determine which fine cell are
        river cells (using the threshold)
    threshold: integer; the threshold using to decide if cumulative flow to a cell is high enough for 
        it to be eligible for intelligent burning
    region: dictionary: a dictionary specifying the coordinate of a region on this grid type 
    course_grid_type: string; code for the the course grid type
    course_grid_kwarg: dictionary; keyword parameter dictionary specifying parameters of the
        course grid type (if required)
    Returns: Orography with intelligent burning applied in a Field object
    
    Intelligent burn course field by masking the pixels of a fine orography field inside a course cell
    where the flow is less than a given threshold in the comparable fine total cumulative flow field
    then taking the highest height of the fine orography inside the course cell where it is not masked.
    If this height is greater than that of the course cell itself in the course orography field then
    replace the course orography field value with this height. So a river flowing through a course cell
    must pass over the maximum height of the river flowing through the pixels of a fine field equivalent 
    to the course field.
    """

    fmap_below_threshold_mask = \
        input_fine_fmap.generate_cumulative_flow_threshold_mask(threshold)
    input_fine_orography_field.mask_field_with_external_mask(fmap_below_threshold_mask)
    input_fine_orography_field.fill_mask(input_fine_orography_field.get_no_data_value())
    intelligent_height_field = field.Orography(upscale_field(input_field=input_fine_orography_field, 
                                                             output_grid_type=course_grid_type,
                                                             method='Max',
                                                             output_grid_kwargs=course_grid_kwargs,
                                                             scalenumbers=False).get_data(),
                                               grid=course_grid_type,**course_grid_kwargs)
    intelligent_height_field.mask_no_data_points()
    intelligent_height_field.mask_where_greater_than(course_orography_field)
    intelligent_height_field.mask_outside_region(region)
    course_orography_field.update_field_with_partially_masked_data(intelligent_height_field)
    return course_orography_field

def intelligent_orography_burning_driver(input_fine_orography_filename,
                                         input_course_orography_filename,
                                         input_fine_fmap_filename,
                                         output_course_orography_filename,
                                         regions_to_burn_list_filename,
                                         change_print_out_limit = 200,
                                         fine_grid_type=None,course_grid_type=None,
                                         fine_grid_kwargs={},**course_grid_kwargs):
    """Drive intelligent burning of an orography
    
    Arguments:
    input_fine_orography_filename: string; full path to the fine orography field file to use as a reference
    input_course_orography_filename: string; full path to the course orography field file to intelligently burn
    input_fine_fmap_filename: string; full path to the fine cumulative flow field file to use as reference
    output_course_orography_filename: string; full path to target file to write the output intelligently burned
        orogrpahy to
    regions_to_burn_list_filename: string; full path of list of regions to burn and the burning thershold to use
        for each region. See inside function the necessary format for the header and the necessary format to 
        specifying each region to burn 
    change_print_out_limit: integer; limit on the number of changes to the orography to individually print out
    fine_grid_type: string; code for the grid type of the fine grid
    course_grid_type: string; code for teh grid type of the course grid
    fine_grid_kwargs: dictionary; key word dictionary specifying parameters of the fine grid (if required)
    course_grid_kwargs: dictionary; key word dictionary specifying parameters of the course grid (if required)
    Returns: nothing
    
    Reads input file, orientates field according to information on orientation supplied in corrections region file,
    uses intelligent burning function to burn each specified region then writes out results.
    """

    input_fine_orography_field = dynamic_hd.load_field(input_fine_orography_filename, 
                                                       file_type=\
                                                       dynamic_hd.get_file_extension(input_fine_orography_filename),
                                                       field_type='Orography', 
                                                       grid_type=fine_grid_type,
                                                       **fine_grid_kwargs)
    input_course_orography_field = dynamic_hd.load_field(input_course_orography_filename,
                                                         file_type=\
                                                         dynamic_hd.get_file_extension(input_course_orography_filename),
                                                         field_type='Orography',
                                                         grid_type=course_grid_type,
                                                         **course_grid_kwargs)
    input_fine_fmap_field = dynamic_hd.load_field(input_fine_fmap_filename,
                                                  file_type=\
                                                  dynamic_hd.get_file_extension(input_fine_fmap_filename),
                                                  field_type='CumulativeFlow',
                                                  grid_type=fine_grid_type,
                                                  **fine_grid_kwargs)
    regions = []
    thresholds = []
    first_line_pattern = re.compile(r"^course_grid_type *= *" + course_grid_type + r"$")
    second_line_pattern = re.compile(r"^fine_grid_type *= *" + fine_grid_type + r"$")
    third_line_pattern = re.compile(r"^course_grid_flipud *= *(True|true|TRUE|False|false|FALSE)$")
    fourth_line_pattern = re.compile(r"^course_grid_rotate180lr *= *(True|true|TRUE|False|false|FALSE)$")
    fifth_line_pattern = re.compile(r"^fine_grid_flipud *= *(True|true|TRUE|False|false|FALSE)$")
    sixth_line_pattern = re.compile(r"^fine_grid_rotate180lr *= *(True|true|TRUE|False|false|FALSE)$")
    comment_line_pattern = re.compile(r"^ *#.*$")
    threshold_pattern = re.compile(r"^ *threshold *= *([0-9]*) *$")
    region_bound_pattern = re.compile(r"^ *([a-zA-Z0-9_]*) *= *([0-9]*) *$")
    with open(regions_to_burn_list_filename) as f:
        if not first_line_pattern.match(f.readline().strip()):
            raise RuntimeError("List of regions to burn is not for correct course grid-type")
        if not second_line_pattern.match(f.readline().strip()):
            raise RuntimeError("List of regions to burn is not for correct fine grid-type")
        third_line_pattern_match = third_line_pattern.match(f.readline().strip())
        if not third_line_pattern_match:
            raise RuntimeError("List of regions to burn being loaded has incorrect format on line 3")
        flipud = True if third_line_pattern_match.group(1).lower() == 'true' else False
        fourth_line_pattern_match = fourth_line_pattern.match(f.readline().strip())
        if not fourth_line_pattern_match:
            raise RuntimeError("List of regions to burn being loaded has incorrect format on line 4")
        rotate180lr = True if fourth_line_pattern_match.group(1).lower() == 'true' else False
        fifth_line_pattern_match = fifth_line_pattern.match(f.readline().strip())
        if not fifth_line_pattern_match:
            raise RuntimeError("List of regions to burn being loaded has incorrect format on line 5")
        fine_grid_flipud = True if fifth_line_pattern_match.group(1).lower() == 'true' else False
        sixth_line_pattern_match = sixth_line_pattern.match(f.readline().strip())
        if not sixth_line_pattern_match:
            raise RuntimeError("List of regions to burn being loaded has incorrect format on line 6")
        fine_grid_rotate180lr =  True if sixth_line_pattern_match.group(1).lower() == 'true' else False
        for line_num,line in enumerate(f,start=5):
            if comment_line_pattern.match(line):
                continue 
            entries = line.strip().split(",")
            threshold = None
            region = {} 
            for entry in entries:
                threshold_pattern_match = threshold_pattern.match(entry)
                region_bound_pattern_match = region_bound_pattern.match(entry)
                if threshold_pattern_match:
                    threshold = int(threshold_pattern_match.group(1))
                elif region_bound_pattern_match:
                    region[region_bound_pattern_match.group(1)] = int(region_bound_pattern_match.group(2))
                else:
                    raise RuntimeError("Entry specifying region bounds in regions to burn file line {0} is"
                                       " in incorrect format".format(line_num))
            if threshold is None:
                raise RuntimeError("No threshold specified on line {0}".format(line_num))
            regions.append(region)
            thresholds.append(threshold)
    if flipud:
        input_course_orography_field.flip_data_ud()
    if rotate180lr:
        input_course_orography_field.rotate_field_by_a_hundred_and_eighty_degrees()
    if fine_grid_flipud:
        input_fine_orography_field.flip_data_ud()
        input_fine_fmap_field.flip_data_ud()
    if fine_grid_rotate180lr:
        input_fine_orography_field.rotate_field_by_a_hundred_and_eighty_degrees()
        input_fine_fmap_field.rotate_field_by_a_hundred_and_eighty_degrees
    output_course_orography = copy.deepcopy(input_course_orography_field)
    for region,threshold in zip(regions,thresholds):
        print "Intelligently burning region: {0} \n using the threshold {1}".format(region,threshold)
        #This is modified by the intelligently burn orography field so need to make a
        #copy to pass in each time
        working_fine_orography_field = copy.deepcopy(input_fine_orography_field)
        output_course_orography = intelligently_burn_orography(working_fine_orography_field, 
                                                               course_orography_field=output_course_orography,
                                                               input_fine_fmap=input_fine_fmap_field, 
                                                               threshold=threshold, region=region,
                                                               course_grid_type=course_grid_type,
                                                               **course_grid_kwargs)
        difference_in_orography_field = output_course_orography.get_data() - \
                                        input_course_orography_field.get_data()
        if np.count_nonzero(difference_in_orography_field) > change_print_out_limit:
            print "Intelligent burning makes more than {0} changes to orography".\
                format(change_print_out_limit)
        else:
            changes_in_orography_field = np.transpose(np.nonzero(difference_in_orography_field))
            for change in changes_in_orography_field:
                print "Changing the height of cell nlat={0},nlon={1} from {2}m to {3}m".\
                    format(change[0],change[1],
                           input_course_orography_field.get_data()[tuple(change.tolist())],
                           output_course_orography.get_data()[tuple(change.tolist())])
        #re-use the input orography field as a intermediary value holder
        input_course_orography_field = copy.deepcopy(output_course_orography)
    if flipud:
        output_course_orography.flip_data_ud()
    if rotate180lr:
        output_course_orography.rotate_field_by_a_hundred_and_eighty_degrees()
    dynamic_hd.write_field(output_course_orography_filename, 
                           field=output_course_orography, 
                           file_type=dynamic_hd.get_file_extension(output_course_orography_filename))