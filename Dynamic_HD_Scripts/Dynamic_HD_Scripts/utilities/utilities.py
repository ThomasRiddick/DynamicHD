'''
Created on Apr 5, 2016

@author: thomasriddick
'''

import numpy as np
import os.path as path
import re
import copy
import shutil
import cdo
import netCDF4
import enum
import warnings
from enum import Enum
from Dynamic_HD_Scripts.base import field
from Dynamic_HD_Scripts.base import grid
from Dynamic_HD_Scripts.base import iodriver
from Dynamic_HD_Scripts.base.field import makeField
from Dynamic_HD_Scripts.tools import follow_streams_driver
from Dynamic_HD_Scripts.interface.fortran_interface \
    import f2py_manager
from Dynamic_HD_Scripts.context import fortran_source_path

def create_30sec_lgm_orography_from_highres_present_day_and_low_res_pair(input_lgm_low_res_orog,
                                                                         input_present_day_low_res_orog,
                                                                         input_present_day_high_res_orog):
  input_lgm_low_res_orog.subtract(input_present_day_low_res_orog)
  #This is not a landsea mask but this function should also work to downscale a field of anomalies
  high_res_anomalies = downscale_ls_mask(input_lgm_low_res_orog,fine_grid_type='LatLong30sec')
  input_present_day_high_res_orog.add(high_res_anomalies)
  return input_present_day_high_res_orog

def create_30s_lgm_orog_from_hr_present_day_and_lr_pair_driver(input_lgm_low_res_orog_filename,
                                                               input_present_day_low_res_orog_filename,
                                                               input_present_day_high_res_orog_filename,
                                                               output_lgm_high_res_orog_filename,
                                                               input_lgm_low_res_orog_fieldname,
                                                               input_present_day_low_res_orog_fieldname,
                                                               input_present_day_high_res_orog_fieldname,
                                                               output_lgm_high_res_orog_fieldname):
  lgm_low_res_orog = iodriver.advanced_field_loader(input_lgm_low_res_orog_filename,
                                                    field_type='Orography',
                                                    fieldname=input_lgm_low_res_orog_fieldname)
  present_day_low_res_orog = iodriver.advanced_field_loader(input_present_day_low_res_orog_filename,
                                                            field_type='Orography',
                                                            fieldname=
                                                            input_present_day_low_res_orog_fieldname)
  present_day_high_res_orog = iodriver.advanced_field_loader(input_present_day_high_res_orog_filename,
                                                             field_type='Orography',
                                                             fieldname=
                                                             input_present_day_high_res_orog_fieldname)
  lgm_high_res_orog = \
    create_30sec_lgm_orography_from_highres_present_day_and_low_res_pair(lgm_low_res_orog,
                                                                         present_day_low_res_orog,
                                                                         present_day_high_res_orog)
  iodriver.advanced_field_writer(output_lgm_high_res_orog_filename,lgm_high_res_orog,
                                 fieldname=output_lgm_high_res_orog_fieldname)


def replace_corrected_orography_with_original_for_glaciated_grid_points(input_corrected_orography,
                                                                        input_original_orography,
                                                                        input_glacier_mask):
    """Replace a corrected orography with the original orography at points that are glaciated

    Arguments:
    input_corrected_orography: Field object; the corrected orography where correction are applied
        to both glaciated and unglaciated points
    input_original_orography: Field object; the original orography to use for glaciated points
    input_glacier_mask: Field object; a binary glacial mask where glacier is 1/True and non-glacier
        is 0/False
    Returns: An orography using the corrected orography for non-glaciated points and the original
        orography for glaciated points.
    """

    input_corrected_orography.mask_field_with_external_mask(input_glacier_mask.get_data())
    input_original_orography.update_field_with_partially_masked_data(input_corrected_orography)
    return input_original_orography

def replace_corrected_orography_with_original_for_glaciated_points_with_gradual_transition(input_corrected_orography,
                                                                                           input_original_orography,
                                                                                           input_base_orography,
                                                                                           input_glacier_mask,
                                                                                           blend_to_threshold,
                                                                                           blend_from_threshold):
    """Replace a corrected orography with the original orography at points that are glaciated

    Arguments:
    input_corrected_orography: Field object; the corrected orography where correction are applied
        to both glaciated and unglaciated points
    input_original_orography: Field object; the original orography to use for glaciated points
    input_reference_orography: Field object; the base present day orography the original orography
                               was produced from
    input_glacier_mask: Field object; a binary glacial mask where glacier is 1/True and non-glacier
        is 0/False
    Returns: An orography using the corrected orography for non-glaciated points and the original
        orography for glaciated points with blended edges
    """
    input_corrected_orography_copy = input_corrected_orography.copy()
    input_corrected_orography_copy.replace_glaciated_points_gradual_transition(input_glacier_mask,
                                                                               input_original_orography,
                                                                               input_base_orography,
                                                                               blend_to_threshold,
                                                                               blend_from_threshold)
    return input_corrected_orography_copy

def replace_corrected_orography_with_original_for_glaciated_grid_points_drivers(input_corrected_orography_file,
                                                                                input_original_orography_file,
                                                                                input_glacier_mask_file,
                                                                                out_orography_file,
                                                                                grid_type='HD',**grid_kwargs):
    """Drive replacing a corrected orography with the original orography at points that are glaciated

    Arguments:
    input_corrected_orography_file: string; Full path to the file containing the corrected orography
        where corrections are applied to both glaciated and unglaciated points
    input_original_orography_file: string; Full path to the original uncorrected orography file
    input_glacier_mask_file: string; Full path to the file containing the glacier mask with 1/True as
        glacier and 0/False as non-glacier
    out_orography_file: string; full path to target file to write corrected orography with original
        orography used for glacial points to
    grid_type: string; the code for this grid type
    grid_kwargs: dictionary; key word arguments specifying parameters of the grid
    Returns: Nothing
    """

    input_corrected_orography = iodriver.load_field(input_corrected_orography_file,
                                                    file_type=iodriver.\
                                                    get_file_extension(input_corrected_orography_file),
                                                    field_type='Orography',
                                                    unmask=True,grid_type=grid_type,
                                                    **grid_kwargs)
    input_original_orography = iodriver.load_field(input_original_orography_file,
                                                   file_type=iodriver.\
                                                   get_file_extension(input_original_orography_file),
                                                   field_type='Orography',
                                                   unmask=True,grid_type=grid_type,
                                                   **grid_kwargs)
    input_glacier_mask = iodriver.load_field(input_glacier_mask_file,
                                             file_type=iodriver.\
                                             get_file_extension(input_glacier_mask_file),
                                             fieldname='sftgif',
                                             field_type='Orography',
                                             unmask=True,grid_type=grid_type,
                                             **grid_kwargs)
    output_orography = replace_corrected_orography_with_original_for_glaciated_grid_points(input_corrected_orography,
                                                                                           input_original_orography,
                                                                                           input_glacier_mask)
    iodriver.write_field(filename=out_orography_file,
                         field=output_orography,
                         file_type=iodriver.\
                         get_file_extension(out_orography_file))


def advanced_replace_corrected_orog_with_orig_for_glcted_grid_points_drivers(input_corrected_orography_file,
                                                                             input_original_orography_file,
                                                                             input_glacier_mask_file,
                                                                             out_orography_file,
                                                                             input_corrected_orography_fieldname,
                                                                             input_original_orography_fieldname,
                                                                             input_glacier_mask_fieldname,
                                                                             out_orography_fieldname):
    input_corrected_orography = iodriver.advanced_field_loader(input_corrected_orography_file,
                                                               field_type='Orography',
                                                               fieldname=\
                                                               input_corrected_orography_fieldname)
    input_original_orography = iodriver.advanced_field_loader(input_original_orography_file,
                                                              field_type='Orography',
                                                              fieldname=\
                                                              input_original_orography_fieldname)
    input_glacier_mask = iodriver.advanced_field_loader(input_glacier_mask_file,
                                                        field_type='Orography',
                                                        fieldname=\
                                                        input_glacier_mask_fieldname)
    output_orography = replace_corrected_orography_with_original_for_glaciated_grid_points(input_corrected_orography,
                                                                                           input_original_orography,
                                                                                           input_glacier_mask)
    iodriver.advanced_field_writer(out_orography_file,
                                   field=output_orography,
                                   fieldname=out_orography_fieldname)

def merge_corrected_and_tarasov_upscaled_orography_main_routine(corrected_orography_field,
                                                                tarasov_upscaled_orography_field,
                                                                use_upscaled_orography_only_in_region=None,
                                                                grid_type='HD'):
    corrected_orography_field.mask_where_greater_than(tarasov_upscaled_orography_field)
    tarasov_upscaled_orography_field.update_field_with_partially_masked_data(corrected_orography_field)
    if use_upscaled_orography_only_in_region is not None:
        _grid = grid.makeGrid(grid_type)
        not_in_region_mask = np.zeros(_grid.get_grid_dimensions(),dtype=np.bool_)
        if use_upscaled_orography_only_in_region == "North America":
            if grid_type == 'LatLong10min':
                not_in_region_mask[620:1080,1296:1925] = True
                not_in_region_mask[572:620, 1296:1682] = True
            else:
                raise RuntimeError('Not definition for specified region on specified grid type')
        else:
            raise RuntimeError("Specified region not recognised by merging orographies")
        corrected_orography_field.mask_field_with_external_mask(not_in_region_mask)
        tarasov_upscaled_orography_field.update_field_with_partially_masked_data(corrected_orography_field)
    return tarasov_upscaled_orography_field

def merge_corrected_and_tarasov_upscaled_orography(input_corrected_orography_file,
                                                   input_tarasov_upscaled_orography_file,
                                                   output_merged_orography_file,
                                                   use_upscaled_orography_only_in_region=None,
                                                   grid_type='HD',**grid_kwargs):
    """Merge a normal corrected orography with a tarasov upscaled orography

    Argument:
    input_corrected_orography_file: string; Full path to the the normal corrected orography file
    input_tarasov_upscaled_orography_file: string; Full path to the tarasov upscaled orography
        file
    output_merged_orography_file: string; Full path to the target merged output orography file
    use_upscaled_orography_only_in_region: string; Either None (in which case the upscaled
        orography is used everywhere) or the name of a region (see below in fuction for region
        names and definitions) in which to use the upscaled orography in combination with the
        corrected orography; outside of this the upscaled orography is not used and only the
        corrected orography is used.
    grid_type: string; the code for this grid type
    grid_kwargs: dictionary; key word arguments specifying parameters of the grid
    """

    corrected_orography_field = iodriver.load_field(input_corrected_orography_file,
                                                    file_type=iodriver.get_file_extension(input_corrected_orography_file),
                                                    field_type='Orography',
                                                    unmask=True,grid_type=grid_type,
                                                    **grid_kwargs)
    tarasov_upscaled_orography_field = iodriver.load_field(input_tarasov_upscaled_orography_file,
                                                           file_type=iodriver.get_file_extension(input_corrected_orography_file),
                                                           field_type='Orography',
                                                           unmask=True,grid_type=grid_type,
                                                           **grid_kwargs)
    tarasov_upscaled_orography_field =\
    merge_corrected_and_tarasov_upscaled_orography_main_routine(corrected_orography_field,
                                                                tarasov_upscaled_orography_field,
                                                                use_upscaled_orography_only_in_region,
                                                                grid_type)
    iodriver.write_field(output_merged_orography_file,
                           tarasov_upscaled_orography_field,
                           file_type=iodriver.get_file_extension(output_merged_orography_file),
                           griddescfile=None)

def advanced_merge_corrected_and_tarasov_upscaled_orography(input_corrected_orography_file,
                                                            input_tarasov_upscaled_orography_file,
                                                            output_merged_orography_file,
                                                            input_corrected_orography_fieldname,
                                                            input_tarasov_upscaled_orography_fieldname,
                                                            output_merged_orography_fieldname,
                                                            use_upscaled_orography_only_in_region=None):

    corrected_orography_field = iodriver.advanced_field_loader(input_corrected_orography_file,
                                                               field_type='Orography',
                                                               fieldname=input_corrected_orography_fieldname)
    tarasov_upscaled_orography_field = iodriver.load_field(input_tarasov_upscaled_orography_file,
                                                           field_type='Orography',
                                                           fieldname=input_tarasov_upscaled_orography_fieldname)
    nlat,nlon = corrected_orography_field.get_grid_dimensions()
    if nlat == 1080 and nlon == 2160:
        grid_type="LatLong10min"
    else:
        grid_type=None
    tarasov_upscaled_orography_field =\
    merge_corrected_and_tarasov_upscaled_orography_main_routine(corrected_orography_field,
                                                                tarasov_upscaled_orography_field,
                                                                use_upscaled_orography_only_in_region,
                                                                grid_type)
    iodriver.write_field(output_merged_orography_file,
                           tarasov_upscaled_orography_field,
                           fieldname=output_merged_orography_fieldname)

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

    for var_name,var_obj in list(dataset_inout.variables.items()):
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

    rflow_res_num= iodriver.load_field(filename=hdparas_filename,
                                       file_type=iodriver.get_file_extension(hdparas_filename),
                                       field_type='Generic', unmask=True, timeslice=timeslice,
                                       fieldname='ARF_N',grid_type=grid_type,**grid_kwargs)
    ref_rflow_res_num= iodriver.load_field(filename=ref_hdparas_filename,
                                           file_type=iodriver.get_file_extension(ref_hdparas_filename),
                                           field_type='Generic', unmask=True, timeslice=None,
                                           fieldname='ARF_N',grid_type=grid_type,**grid_kwargs)
    overland_flow_res_num= iodriver.load_field(filename=hdparas_filename,
                                               file_type=iodriver.get_file_extension(hdparas_filename),
                                               field_type='Generic', unmask=True, timeslice=timeslice,
                                               fieldname='ALF_N',grid_type=grid_type,**grid_kwargs)
    ref_overland_flow_res_num= iodriver.load_field(filename=ref_hdparas_filename,
                                                   file_type=iodriver.get_file_extension(ref_hdparas_filename),
                                                   field_type='Generic', unmask=True, timeslice=None,
                                                   fieldname='ALF_N',grid_type=grid_type,**grid_kwargs)
    #Use overland flow reservoir number for baseflow (as this parameter doesn't exist for baseflow)
    base_flow_res_num = iodriver.load_field(filename=hdparas_filename,
                                            file_type=iodriver.get_file_extension(hdparas_filename),
                                            field_type='Generic', unmask=True, timeslice=timeslice,
                                            fieldname='ALF_N',grid_type=grid_type,**grid_kwargs)
    ref_base_flow_res_num = iodriver.load_field(filename=ref_hdparas_filename,
                                                file_type=iodriver.get_file_extension(ref_hdparas_filename),
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
    print('Writing output to: {0}'.format(output_hdrestart_filename))
    with netCDF4.Dataset(output_hdrestart_filename,mode='a',format=netcdf_dataset_format) as dataset:
        dataset=prepare_hdrestart_file(dataset,rflow_res_num,ref_rflow_res_num,overland_flow_res_num,
                                       ref_overland_flow_res_num,base_flow_res_num,ref_base_flow_res_num,
                                       grid_type,**grid_kwargs)

def change_dtype(input_filename,output_filename,input_fieldname,
                 output_fieldname,new_dtype,grid_type,**grid_kwargs):
    """Change the data type of a field in a file

    Arguments:
    input_filename: string; the filename of the input file to read the field from
    output_filename: string; the filename of hte output file to write the field with
        its new datatype to
    input_fieldname: string; fieldname within the input cdf file of the input field
    output_fieldname: string; fieldname for the output field in the output cdf file
    new_dtype: numpy datatype; the datatype to change the field to
    grid_type: string; the code for this grid type
    grid_kwargs: dictionary; key word arguments specifying parameters of the grid
    Returns: nothing

    It is possible to use a new dtype that is identical to the old one; this function
    then effectively does nothing but copy the file.
    """

    field = iodriver.load_field(input_filename,
                                file_type=\
                                iodriver.get_file_extension(input_filename),
                                field_type='Generic',
                                fieldname=input_fieldname,
                                unmask=False,
                                timeslice=None,
                                grid_type=grid_type,
                                **grid_kwargs)
    field.change_dtype(new_dtype)
    iodriver.write_field(output_filename,field,
                         file_type=iodriver.get_file_extension(output_filename),
                         fieldname=output_fieldname)

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
    Returns: nothing
    """

    ls_mask = iodriver.load_field(original_ls_mask_filename,
                                  file_type=\
                                  iodriver.get_file_extension(original_ls_mask_filename),
                                  field_type='Generic',
                                  unmask=False,
                                  timeslice=timeslice,
                                  grid_type=grid_type,
                                  **grid_kwargs)
    ls_mask.invert_data()
    iodriver.write_field(inverted_ls_mask_filename,field=ls_mask,
                         file_type=iodriver.get_file_extension(inverted_ls_mask_filename))

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

    original_orography_field = iodriver.load_field(original_orography_filename,
                                                   file_type=iodriver.\
                                                   get_file_extension(original_orography_filename),
                                                   field_type='Orography', grid_type=grid_type,
                                                   **grid_kwargs)
    corrected_orography_field = iodriver.load_field(corrected_orography_filename,
                                                    file_type=iodriver.\
                                                    get_file_extension(corrected_orography_filename),
                                                    field_type='Orography', grid_type=grid_type,
                                                    **grid_kwargs)
    corrected_orography_field.subtract(original_orography_field)
    iodriver.write_field(orography_corrections_filename,
                         corrected_orography_field,
                         file_type=iodriver.get_file_extension(orography_corrections_filename))

def advanced_orog_correction_field_generator(original_orography_filename,
                                             corrected_orography_filename,
                                             orography_corrections_filename,
                                             original_orography_fieldname,
                                             corrected_orography_fieldname,
                                             orography_corrections_fieldname):

    original_orography_field = iodriver.advanced_field_loader(original_orography_filename,
                                                              field_type='Orography',
                                                              fieldname=\
                                                              original_orography_fieldname)
    corrected_orography_field = iodriver.advanced_field_loader(corrected_orography_filename,
                                                               field_type='Orography',
                                                               fieldname=\
                                                               corrected_orography_fieldname)
    corrected_orography_field.subtract(original_orography_field)
    iodriver.advanced_field_writer(orography_corrections_filename,
                                   corrected_orography_field,
                                   fieldname=orography_corrections_fieldname)

def apply_orog_correction_field(original_orography_filename,
                                orography_corrections_filename,
                                corrected_orography_filename,
                                original_orography_fieldname=None,
                                grid_type='HD',**grid_kwargs):
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

    original_orography_field = iodriver.load_field(original_orography_filename,
                                                     file_type=iodriver.\
                                                     get_file_extension(original_orography_filename),
                                                     field_type='Orography', grid_type=grid_type,
                                                     fieldname=original_orography_fieldname,
                                                     **grid_kwargs)
    orography_corrections_field =  iodriver.load_field(orography_corrections_filename,
                                                       file_type=iodriver.\
                                                       get_file_extension(orography_corrections_filename),
                                                       field_type='Orography', grid_type=grid_type,
                                                       **grid_kwargs)
    original_orography_field.add(orography_corrections_field)
    iodriver.write_field(corrected_orography_filename,
                         original_orography_field,
                         file_type=iodriver.\
                         get_file_extension(corrected_orography_filename))

def advanced_apply_orog_correction_field(original_orography_filename,
                                         orography_corrections_filename,
                                         corrected_orography_filename,
                                         original_orography_fieldname=None,
                                         orography_corrections_fieldname=None,
                                         corrected_orography_fieldname=None):
    original_orography_field = iodriver.advanced_field_loader(original_orography_filename,
                                                              field_type='Orography',
                                                              fieldname=original_orography_fieldname)
    orography_corrections_field = iodriver.advanced_field_loader(orography_corrections_filename,
                                                                 field_type='Orography',
                                                                 fieldname=\
                                                                 orography_corrections_fieldname)
    original_orography_field.add(orography_corrections_field)
    iodriver.advanced_field_writer(corrected_orography_filename,
                                   original_orography_field,
                                   fieldname=corrected_orography_fieldname)

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

    orography = iodriver.load_field(filename=orography_filename,
                                    file_type=iodriver.get_file_extension(orography_filename),
                                    field_type='Orography',
                                    grid_type=grid_type,**grid_kwargs)
    ls_mask = orography.generate_ls_mask(sea_level).astype(dtype=np.int32,order='C')
    iodriver.write_field(filename=ls_mask_filename,
                         field=field.Field(ls_mask,grid_type,**grid_kwargs),
                         file_type=iodriver.get_file_extension(ls_mask_filename))

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

    rdirs_field = iodriver.load_field(rdirs_filename,
                                      file_type=iodriver.get_file_extension(rdirs_filename),
                                      field_type='RiverDirections',
                                      grid_type=grid_type,
                                      **grid_kwargs)
    lsmask = field.RiverDirections(rdirs_field.get_lsmask(),grid=grid_type,**grid_kwargs)
    iodriver.write_field(lsmask_filename, lsmask, file_type=iodriver.get_file_extension(lsmask_filename))

def advanced_extract_ls_mask_from_rdirs(rdirs_filename,lsmask_filename,
                                        rdirs_fieldname,lsmask_fieldname):
    """Extract an land sea mask from river directions with coast and sea cells marked

    Arguments:
    rdirs_filename: string, full path to the river directions file to extract the land-sea mask
        from
    lsmask_filename: string, full path to the target file to the write the extracted land-sea mask
        to
    rdirs_fieldname: string, name of rdirs field within file
    lsmask_fieldname: string, name of lsmask to use in file
    Returns: nothing
    """
    rdirs_field = iodriver.advanced_field_loader(rdirs_filename,field_type='RiverDirections',
                                                 fieldname=rdirs_fieldname)
    lsmask = field.Field(rdirs_field.get_lsmask(),grid=rdirs_field.get_grid())
    iodriver.advanced_field_writer(lsmask_filename,lsmask,
                                   fieldname=lsmask_fieldname)

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

    rdirs_field = iodriver.load_field(rdirs_filename,
                                      file_type=iodriver.get_file_extension(rdirs_filename),
                                      field_type='RiverDirections',
                                      grid_type=grid_type,
                                      **grid_kwargs)
    truesinks = field.Field(rdirs_field.extract_truesinks(),grid=grid_type,**grid_kwargs)
    iodriver.write_field(truesinks_filename, truesinks,
                         file_type=iodriver.get_file_extension(truesinks_filename))

def advanced_extract_true_sinks_from_rdirs(rdirs_filename,truesinks_filename,
                                           rdirs_fieldname,truesinks_fieldname):
    """Extract a set of true sinks from an rdirs file

    Arguments:
    rdirs_filename: string; full path to river directions file containing true sink points
    truesinks_filename: string; full path to write the field of extracted true sinks to
    rdirs_fieldname: string, name of rdirs field within file
    lsmask_fieldname: string, name of truesinks field to use in file
    Returns: nothing

    Extracts all points that have the code for a sink (which is 5) from the river directions
    field
    """

    rdirs_field = iodriver.advanced_field_loader(rdirs_filename,field_type='RiverDirections',
                                                 fieldname=rdirs_fieldname)
    truesinks = field.Field(rdirs_field.extract_truesinks(),grid=rdirs_field.get_grid())
    iodriver.advanced_field_writer(truesinks_filename,truesinks,
                                   fieldname=truesinks_fieldname)

def upscale_field_driver(input_filename,output_filename,input_grid_type,output_grid_type,
                  method,timeslice=None,input_grid_kwargs={},output_grid_kwargs={},scalenumbers=False):
    """Load input, drive the process of upscaling a field using a specified method, and write output

    Arguments:
    input_filename: string; full path to file with input fine scale field to upscale
    output_filename: string; full path to file with output coarse scale upscaled field
    input_grid_type: string; the code for the type of the input fine grid
    output_grid_type: string; the code for the type of the input coarse grid
    method: string; upscaling method to use - see upscale_field for valid methods
    timeslice(optional): integer; timeslice to upscale if input file contains multiple timeslices
    input_grid_kwargs: dictionary; key word arguments specifying parameters of the fine input grid
        (if required)
    output_grid_kwargs: dictionary; key word arguments specifying parameters of the coarse output grid
        (if required)
    scalenumbers: scale numbers according to difference in size between the two grids
        (assuming a density-like number is being upscaled)
    Returns: nothing

    Perform a crude upscaling using the basic named method. This doesn't upscale river direction;
    this requires much more sophisticated code and is done by a seperate function.
    """

    input_field = iodriver.load_field(input_filename,
                                      file_type=iodriver.get_file_extension(input_filename),
                                      field_type='Generic',
                                      timeslice=timeslice,
                                      grid_type=input_grid_type,
                                      **input_grid_kwargs)
    output_field = upscale_field(input_field,output_grid_type, method,output_grid_kwargs,scalenumbers)
    iodriver.write_field(output_filename,output_field,
                         file_type=iodriver.get_file_extension(output_filename))

def upscale_field(input_field,output_grid_type,method,output_grid_kwargs,scalenumbers=False):
    """Upscale a field using a specified method

    Arguments:
    input_field: Field object, input field to upscale
    output_grid_type: string; the code for the grid type of the coarse
        grid to upscale to
    method: string, upscaling method to use - see defined method below
    output_grid_kwargs: dictionary; key word arguments specifying parameters of
        the coarse grid type to upscale to (if required)
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
    reshaped_data_array = input_field.get_data().reshape(nlat_out,nlat_in//nlat_out,
                                                         nlon_out,nlon_in//nlon_out)
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

def downscale_true_sink_points_driver(input_fine_orography_filename,input_coarse_truesinks_filename,
                                      output_fine_truesinks_filename,input_fine_orography_grid_type,
                                      input_coarse_truesinks_grid_type,input_fine_orography_grid_kwargs={},
                                      input_coarse_truesinks_grid_kwargs={},flip_coarse_grid_ud=False,
                                      rotate_coarse_true_sink_about_polar_axis=False,
                                      downscaled_true_sink_modifications_filename=None,
                                      coarse_true_sinks_modifications_filename=None):
    """Load input, drive the process of downscale a true sinks a field and write output

    Argument:
    input_fine_orography_filename: string; full path to input fine orography
    input_coarse_truesinks_filename: string; full path to input true sinks array file
    output_fine_truesinks_filename: string; full path to target output true sinks array
        file to write to
    input_fine_orography_grid_type: string; the code for the grid type of the input
        orography to downscale the true sinks to
    input_coarse_truesinks_grid_type: string; the code for the grid type of the input
        coarse grid that the input true sinks are on
    input_fine_orography_grid_kwargs: dictionary; key word arguments specifying
        parameters of the fine input grid (if required)
    input_coarse_truesinks_grid_kwargs: dictionary; key word arguments specifying
        parameters of the coarse input grid (if required)
    flip_coarse_grid_ud: boolean; flip the coarse truesinks field about the
        equator
    rotate_coarse_true_sink_about_polar_axis: boolean; rotate the coarse truesinks
    field by 180 degrees about the pole
    downscaled_true_sink_modifications_filename: string; full path to text file (see
        apply_modifications_to_truesinks_field for the correct format) with the
        modification to apply to the downscaled truesinks
    coarse_true_sinks_modifications_filename: string; full path to text file (see
        apply_modifications_to_truesinks_field for the correct format) with the
        modification
        to apply to the coarse truesinks before downscaling
    Returns: nothing

    Place the true sinks at the lowest point in the set of fine cells covered by the
    coarse cell. Can make modifications both to the coarse true sinks field before
    processing and the fine true sinks field after processing.
    """

    input_fine_orography_field = iodriver.load_field(input_fine_orography_filename,
                                                     file_type=iodriver.get_file_extension(input_fine_orography_filename),
                                                     field_type='Orography',
                                                     grid_type=input_fine_orography_grid_type,
                                                     **input_fine_orography_grid_kwargs)
    input_coarse_truesinks_field = iodriver.load_field(input_coarse_truesinks_filename,
                                                       file_type=iodriver.get_file_extension(input_coarse_truesinks_filename),
                                                       field_type='Generic',
                                                       grid_type=input_coarse_truesinks_grid_type,
                                                       **input_coarse_truesinks_grid_kwargs)
    if coarse_true_sinks_modifications_filename:
        input_coarse_truesinks_field =\
            field.Field(apply_modifications_to_truesinks_field(truesinks_field=\
                                                               input_coarse_truesinks_field.get_data(),
                                                               true_sink_modifications_filename=\
                                                               coarse_true_sinks_modifications_filename),
                        grid=input_coarse_truesinks_grid_type,
                        **input_coarse_truesinks_grid_kwargs)
    if flip_coarse_grid_ud:
        input_coarse_truesinks_field.flip_data_ud()
    if rotate_coarse_true_sink_about_polar_axis:
        input_coarse_truesinks_field.rotate_field_by_a_hundred_and_eighty_degrees()
    output_fine_truesinks_field = field.Field(downscale_true_sink_points(input_fine_orography_field,
                                                                         input_coarse_truesinks_field),
                                              input_fine_orography_grid_type,
                                              **input_fine_orography_grid_kwargs)

    if downscaled_true_sink_modifications_filename:
        #Assume that same orientation of grid for coarse and fine grid is used in mods files
        #and it the coarse grid orientation - thus in this case flip fine grid and flip back
        #after
        if flip_coarse_grid_ud:
            output_fine_truesinks_field.flip_data_ud()
        if rotate_coarse_true_sink_about_polar_axis:
            output_fine_truesinks_field.rotate_field_by_a_hundred_and_eighty_degrees()
        output_fine_truesinks_field =\
            field.Field(apply_modifications_to_truesinks_field(output_fine_truesinks_field.get_data(),
                                                               downscaled_true_sink_modifications_filename),
                        input_fine_orography_grid_type,
                        **input_fine_orography_grid_kwargs)
        if rotate_coarse_true_sink_about_polar_axis:
            output_fine_truesinks_field.rotate_field_by_a_hundred_and_eighty_degrees()
        if flip_coarse_grid_ud:
            output_fine_truesinks_field.flip_data_ud()
    iodriver.write_field(output_fine_truesinks_filename,
                         output_fine_truesinks_field,
                         file_type=iodriver.get_file_extension(output_fine_truesinks_filename))

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
        print('Removing true sink point at {0},{1}'.format(point[0],point[1]))
        truesinks_field[point[0],point[1]] = False
    for point in points_to_add:
        print('Adding true sink point at {0},{1}'.format(point[0],point[1]))
        truesinks_field[point[0],point[1]] = True
    return truesinks_field

def downscale_true_sink_points(input_fine_orography_field,input_coarse_truesinks_field):
    """Downscale a field of true sink points flags

    Arguments:
    input_fine_orography_field: Field object, the input fine orography field to place the
        downscaled true sinks in
    input_coarse_truesinks_field: Field object, the input coarse true sinks field (as
        a logical field)
    Returns: Field object contain the logical fine true sinks field

    Downscale true sinks by placing each coarse true sink at the mimima (of height) of
        the set of fine orography pixels covered by the coarse cell.
    """

    nlat_fine,nlon_fine = input_fine_orography_field.grid.get_grid_dimensions()
    nlat_coarse,nlon_coarse = input_coarse_truesinks_field.grid.get_grid_dimensions()
    if nlat_coarse > nlat_fine or nlon_coarse > nlon_fine:
        raise RuntimeError('Cannot use the downscale true sink points function to perform an upscaling')
    if nlat_fine % nlat_coarse != 0 or nlon_fine % nlon_coarse !=0 :
        raise RuntimeError('Incompatible input and output grid dimensions')
    scalingfactor_lat = nlat_fine // nlat_coarse
    scalingfactor_lon = nlon_fine // nlon_coarse
    flagged_points_coords = input_coarse_truesinks_field.get_flagged_points_coords()
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
    input_ls_seed_points_list_filename: string, input coarse land sea point list
    output_ls_seed_points_list_filename: string, output fine land sea point list
    factor: integer, difference in scale between coarse and fine grid
    nlat_fine: integer, total number of latitude points
    nlon_fine: integer, total number of longitude points
    input_grid_type: string; the code for the type of the input fine grid
    output_grid_type: string; the code for the type of the output coarse grid
    Return: nothing

    Checks that the grid type of the coarse and fine grid match then downscale
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
    output_points_list = downscale_ls_seed_points_list(input_coarse_list=input_points_list,
                                                       downscale_factor=factor,
                                                       nlat_fine=nlat_fine,
                                                       nlon_fine=nlon_fine)
    with open(output_ls_seed_points_list_filename,'w') as f:
        f.write(output_grid_type + '\n')
        for entry in output_points_list:
            f.write("{0},{1}\n".format(entry[0],entry[1]))

def downscale_ls_seed_points_list(input_coarse_list,downscale_factor,nlat_fine,nlon_fine):
    """Downscale a list of land sea seed points by scaling their coordinates

    Arguments:
    input_coarse_list: list of tuples, a list of coarse latitude longitude coordinates
    (latitude first) giving the position of the coarse land-sea seed points
    downscale_factor: integer, factor to downscale by
    nlat_fine: integer, total number of fine latitude points on the fine grid
    nlon_fine: integer, total number of fine longitude points on the fine grid
    Returns:  list of tuple, a list of fine latitude longitude coordinates (latitude
        first) created by downscaling the coarse grid points

    Each downscale coarse point is turned into a coarse cell size block of
    fine points. Longitude is simply downscaled. Latitude is also inverted (the
    implied field of points flipped up down) during the process.
    """

    output_fine_list = []
    for coords in input_coarse_list:
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

    orography_field = iodriver.load_field(input_orography_filename,
                                          file_type=iodriver.\
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
        print("Correcting height of lat={0},lon={1} to {2} m".format(lat,lon,height))
        orography_field.get_data()[lat,lon] = height
    #restore to original format so as not to disrupt downstream processing chain
    if flipud:
        orography_field.flip_data_ud()
    if rotate180lr:
        orography_field.rotate_field_by_a_hundred_and_eighty_degrees()
    iodriver.write_field(output_orography_filename,
                         orography_field,
                         file_type=iodriver.get_file_extension(output_orography_filename))

def downscale_ls_mask_driver(input_coarse_ls_mask_filename,
                             output_fine_ls_mask_filename,
                             input_flipud=False,
                             input_rotate180lr=False,
                             coarse_grid_type='HD',fine_grid_type='LatLong10min',
                             coarse_grid_kwargs={},**fine_grid_kwargs):
    """Drive process of downscaling a land-sea mask

    Arguments:
    input_coarse_ls_mask_filename: string; full path to input coarse land sea mask file
    output_fine_ls_mask_filename: string; full path to target fine land sea mask file
    input_flipud: boolean, flip the input land sea mask about the equator
    input_rotate180lr: boolean; rotate the input land sea mask by 180 degrees about
        the pole
    coarse_grid_type: string; code for the coarse grid type of the input field
    fine_grid_type: string;  code for the fine grid type of the output field
    coarse_grid_kwargs: dictionary; key word argument dictionary for the coarse grid type
        of the input field (if required)
    fine_grid_kwargs: dictionary; key word arguments dictionary for the fine grid type of
        the output field (if required)
    Returns: nothing

    Outflow field orientations is the same as input field orientation.
    """

    input_coarse_ls_mask_field = iodriver.load_field(input_coarse_ls_mask_filename,
                                                     file_type=iodriver.\
                                                     get_file_extension(input_coarse_ls_mask_filename),
                                                     field_type='Generic',
                                                     grid_type=coarse_grid_type,
                                                     **coarse_grid_kwargs)
    if input_flipud:
        input_coarse_ls_mask_field.flip_data_ud()
    if input_rotate180lr:
        input_coarse_ls_mask_field.rotate_field_by_a_hundred_and_eighty_degrees()
    output_fine_ls_mask_field = downscale_ls_mask(input_coarse_ls_mask_field,
                                                  fine_grid_type,**fine_grid_kwargs)
    iodriver.write_field(output_fine_ls_mask_filename,output_fine_ls_mask_field,
                         file_type=iodriver.get_file_extension(output_fine_ls_mask_filename))

def advanced_downscale_ls_mask_driver(input_coarse_ls_mask_filename,
                                      output_fine_ls_mask_filename,
                                      input_coarse_ls_mask_fieldname,
                                      output_fine_ls_mask_fieldname,
                                      fine_grid_type='LatLong10min',
                                      **fine_grid_kwargs):
    """Drive process of downscaling a land-sea mask using advanced loader/writer

    Arguments:
    input_coarse_ls_mask_filename: string; full path to input coarse land sea mask file
    output_fine_ls_mask_filename: string; full path to target fine land sea mask file
    fine_grid_type: string;  code for the fine grid type of the output field
    fine_grid_kwargs: dictionary; key word arguments dictionary for the fine grid type of
        the output field (if required)
    Returns: nothing

    Outflow field orientations is the same as input field orientation.
    """

    input_coarse_ls_mask_field = iodriver.advanced_field_loader(input_coarse_ls_mask_filename,
                                                                fieldname=
                                                                input_coarse_ls_mask_fieldname)
    output_fine_ls_mask_field = downscale_ls_mask(input_coarse_ls_mask_field,
                                                  fine_grid_type,**fine_grid_kwargs)
    #scale_factor = (output_fine_ls_mask_field.get_grid().nlong/
    #               input_coarse_ls_mask_field.get_grid().nlong)
    #coarse_coords = input_coarse_ls_mask_field.get_grid().get_coordinates()
    #fine_nlat = scale_factor*input_coarse_ls_mask_field.get_grid().nlat
    #fine_nlong = scale_factor*input_coarse_ls_mask_field.get_grid().nlong
    #output_fine_ls_mask.get_grid().set_coordinates()
    warnings.warn("Coordinates not set for fine grid!")
    iodriver.advanced_field_writer(output_fine_ls_mask_filename,output_fine_ls_mask_field,
                                   fieldname=output_fine_ls_mask_fieldname)

def downscale_ls_mask(input_coarse_ls_mask_field,fine_grid_type,**fine_grid_kwargs):
    """Downscale a land-sea mask

    Arguments:
    input_coarse_ls_mask_field: Field object; the input coarse field to downscale
    fine_grid_type: string; the code of the grid type to downscale to
    fine_grid_kwargs: dictionary; key word dictionary for the grid type to be
        downscaled to (if required)
    Returns: The downscaled field in a Field object

    The downscaling is done crudely by assuming that all fine pixels/cells covered by
    a coarse cell have the same land-sea value (1 or 0) as the coarse cell itself; thus
    this will produce a blocky land-sea mask on the fine grid with a granular size equal
    to that of the coarse grid.
    """

    nlat_fine,nlon_fine = grid.makeGrid(fine_grid_type,**fine_grid_kwargs).get_grid_dimensions()
    nlat_coarse,nlon_coarse = input_coarse_ls_mask_field.grid.get_grid_dimensions()
    if nlat_coarse > nlat_fine or nlon_coarse > nlon_fine:
        raise RuntimeError('Cannot use the downscale ls mask function to perform an upscaling')
    if nlat_fine % nlat_coarse != 0 or nlon_fine % nlon_coarse !=0 :
        raise RuntimeError('Incompatible input and output grid dimensions')
    scalingfactor_lat = nlat_fine // nlat_coarse
    scalingfactor_lon = nlon_fine // nlon_coarse
    #this series of manipulations produces the required 'binary' upscaling
    #outer is the outer product
    output_fine_ls_mask_flatted = np.outer(input_coarse_ls_mask_field.get_data().flatten(order='F'),
                                           np.ones(scalingfactor_lat,dtype=np.int32)).flatten(order='C')
    output_fine_ls_mask_flatted = np.outer(output_fine_ls_mask_flatted,
                                           np.ones(scalingfactor_lon,dtype=np.int32)).flatten(order='C')
    output_fine_ls_mask = output_fine_ls_mask_flatted.reshape((scalingfactor_lon,nlat_fine,nlon_coarse),
                                                               order='F')
    output_fine_ls_mask = output_fine_ls_mask.swapaxes(0,1)
    output_fine_ls_mask = output_fine_ls_mask.reshape(nlat_fine,nlon_fine,order='F')
    return field.Field(np.ascontiguousarray(output_fine_ls_mask,dtype=np.int32),grid=fine_grid_type,
                       **fine_grid_kwargs)

def intelligently_burn_orography(input_fine_orography_field,coarse_orography_field,
                                 input_fine_fmap,threshold,region,
                                 coarse_grid_type,**coarse_grid_kwargs):
    """Intelligently burn an orography in a given reason using a given threshold

    Arguments:
    input_fine_orography_field: Field object; the input fine orography to take intelligent
        burning height values
    coarse_orography_field: Field object; the input coarse orography to applying the burning
        too
    input_fine_fmap: Field object; fine culumalative flow to cell to determine which fine cell are
        river cells (using the threshold)
    threshold: integer; the threshold using to decide if cumulative flow to a cell is high enough for
        it to be eligible for intelligent burning
    region: dictionary: a dictionary specifying the coordinate of a region on this grid type
    coarse_grid_type: string; code for the the coarse grid type
    coarse_grid_kwarg: dictionary; keyword parameter dictionary specifying parameters of the
        coarse grid type (if required)
    Returns: Orography with intelligent burning applied in a Field object

    Intelligent burn coarse field by masking the pixels of a fine orography field inside a coarse cell
    where the flow is less than a given threshold in the comparable fine total cumulative flow field
    then taking the highest height of the fine orography inside the coarse cell where it is not masked.
    If this height is greater than that of the coarse cell itself in the coarse orography field then
    replace the coarse orography field value with this height. So a river flowing through a coarse cell
    must pass over the maximum height of the river flowing through the pixels of a fine field equivalent
    to the coarse field.
    """

    fmap_below_threshold_mask = \
        input_fine_fmap.generate_cumulative_flow_threshold_mask(threshold)
    input_fine_orography_field.mask_field_with_external_mask(fmap_below_threshold_mask)
    input_fine_orography_field.fill_mask(input_fine_orography_field.get_no_data_value())
    intelligent_height_field = field.Orography(upscale_field(input_field=input_fine_orography_field,
                                                             output_grid_type=coarse_grid_type,
                                                             method='Max',
                                                             output_grid_kwargs=coarse_grid_kwargs,
                                                             scalenumbers=False).get_data(),
                                               grid=coarse_grid_type,**coarse_grid_kwargs)
    intelligent_height_field.mask_no_data_points()
    intelligent_height_field.mask_where_greater_than(coarse_orography_field)
    intelligent_height_field.mask_outside_region(region)
    coarse_orography_field.update_field_with_partially_masked_data(intelligent_height_field)
    return coarse_orography_field

def intelligent_orography_burning_driver(input_fine_orography_filename,
                                         input_coarse_orography_filename,
                                         input_fine_fmap_filename,
                                         output_coarse_orography_filename,
                                         regions_to_burn_list_filename,
                                         change_print_out_limit = 200,
                                         fine_grid_type=None,coarse_grid_type=None,
                                         fine_grid_kwargs={},**coarse_grid_kwargs):
    """Drive intelligent burning of an orography

    Arguments:
    input_fine_orography_filename: string; full path to the fine orography field file to use as a reference
    input_coarse_orography_filename: string; full path to the coarse orography field file to intelligently burn
    input_fine_fmap_filename: string; full path to the fine cumulative flow field file to use as reference
    output_coarse_orography_filename: string; full path to target file to write the output intelligently burned
        orogrpahy to
    regions_to_burn_list_filename: string; full path of list of regions to burn and the burning thershold to use
        for each region. See inside function the necessary format for the header and the necessary format to
        specifying each region to burn
    change_print_out_limit: integer; limit on the number of changes to the orography to individually print out
    fine_grid_type: string; code for the grid type of the fine grid
    coarse_grid_type: string; code for teh grid type of the coarse grid
    fine_grid_kwargs: dictionary; key word dictionary specifying parameters of the fine grid (if required)
    coarse_grid_kwargs: dictionary; key word dictionary specifying parameters of the coarse grid (if required)
    Returns: nothing

    Reads input file, orientates field according to information on orientation supplied in corrections region file,
    uses intelligent burning function to burn each specified region then writes out results.
    """

    input_fine_orography_field = iodriver.load_field(input_fine_orography_filename,
                                                     file_type=\
                                                     iodriver.get_file_extension(input_fine_orography_filename),
                                                     field_type='Orography',
                                                     grid_type=fine_grid_type,
                                                     **fine_grid_kwargs)
    input_coarse_orography_field = iodriver.load_field(input_coarse_orography_filename,
                                                       file_type=\
                                                       iodriver.get_file_extension(input_coarse_orography_filename),
                                                       field_type='Orography',
                                                       grid_type=coarse_grid_type,
                                                       **coarse_grid_kwargs)
    input_fine_fmap_field = iodriver.load_field(input_fine_fmap_filename,
                                                file_type=\
                                                iodriver.get_file_extension(input_fine_fmap_filename),
                                                field_type='CumulativeFlow',
                                                grid_type=fine_grid_type,
                                                **fine_grid_kwargs)
    regions = []
    thresholds = []
    first_line_pattern = re.compile(r"^coarse_grid_type *= *" + coarse_grid_type + r"$")
    second_line_pattern = re.compile(r"^fine_grid_type *= *" + fine_grid_type + r"$")
    third_line_pattern = re.compile(r"^coarse_grid_flipud *= *(True|true|TRUE|False|false|FALSE)$")
    fourth_line_pattern = re.compile(r"^coarse_grid_rotate180lr *= *(True|true|TRUE|False|false|FALSE)$")
    fifth_line_pattern = re.compile(r"^fine_grid_flipud *= *(True|true|TRUE|False|false|FALSE)$")
    sixth_line_pattern = re.compile(r"^fine_grid_rotate180lr *= *(True|true|TRUE|False|false|FALSE)$")
    comment_line_pattern = re.compile(r"^ *#.*$")
    threshold_pattern = re.compile(r"^ *threshold *= *([0-9]*) *$")
    region_bound_pattern = re.compile(r"^ *([a-zA-Z0-9_]*) *= *([0-9]*) *$")
    with open(regions_to_burn_list_filename) as f:
        if not first_line_pattern.match(f.readline().strip()):
            raise RuntimeError("List of regions to burn is not for correct coarse grid-type")
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
        input_coarse_orography_field.flip_data_ud()
    if rotate180lr:
        input_coarse_orography_field.rotate_field_by_a_hundred_and_eighty_degrees()
    if fine_grid_flipud:
        input_fine_orography_field.flip_data_ud()
        input_fine_fmap_field.flip_data_ud()
    if fine_grid_rotate180lr:
        input_fine_orography_field.rotate_field_by_a_hundred_and_eighty_degrees()
        input_fine_fmap_field.rotate_field_by_a_hundred_and_eighty_degrees
    output_coarse_orography = copy.deepcopy(input_coarse_orography_field)
    for region,threshold in zip(regions,thresholds):
        print("Intelligently burning region: {0} \n using the threshold {1}".format(region,threshold))
        #This is modified by the intelligently burn orography field so need to make a
        #copy to pass in each time
        working_fine_orography_field = copy.deepcopy(input_fine_orography_field)
        output_coarse_orography = intelligently_burn_orography(working_fine_orography_field,
                                                               coarse_orography_field=output_coarse_orography,
                                                               input_fine_fmap=input_fine_fmap_field,
                                                               threshold=threshold, region=region,
                                                               coarse_grid_type=coarse_grid_type,
                                                               **coarse_grid_kwargs)
        difference_in_orography_field = output_coarse_orography.get_data() - \
                                        input_coarse_orography_field.get_data()
        if np.count_nonzero(difference_in_orography_field) > change_print_out_limit:
            print("Intelligent burning makes more than {0} changes to orography".\
                format(change_print_out_limit))
        else:
            changes_in_orography_field = np.transpose(np.nonzero(difference_in_orography_field))
            for change in changes_in_orography_field:
                print("Changing the height of cell nlat={0},nlon={1} from {2}m to {3}m".\
                    format(change[0],change[1],
                           input_coarse_orography_field.get_data()[tuple(change.tolist())],
                           output_coarse_orography.get_data()[tuple(change.tolist())]))
        #re-use the input orography field as a intermediary value holder
        input_coarse_orography_field = copy.deepcopy(output_coarse_orography)
    if flipud:
        output_coarse_orography.flip_data_ud()
    if rotate180lr:
        output_coarse_orography.rotate_field_by_a_hundred_and_eighty_degrees()
    iodriver.write_field(output_coarse_orography_filename,
                         field=output_coarse_orography,
                         file_type=iodriver.get_file_extension(output_coarse_orography_filename))

def generate_regular_landsea_mask_from_gaussian_landsea_mask(input_gaussian_latlon_lsmask_filename,
                                                             output_regular_latlon_mask_filename,
                                                             regular_grid_spacing_file):
    """Generate a regular landsea mask from a lat-lon gaussian landsea mask using cdos

    Arguments:
    input_gaussian_latlon_lsmask_filename: string; full path to the input gaussian land-sea mask file
        to generate a regular land-sea mask from
    output_regular_latlon_mask_filename: string; full path to the target file for  the output regular
        landsea mask; this mask can be in any format the cdos support and is specified by the
        regular_grid_spacing_file (see below)
    regular_grid_spacing_file: string; full path to the file containing the regular grid spacing desired
        for the output file
    """

    cdo_instance = cdo.Cdo()
    print("Generating regular land-sea mask from input gaussian mask: {0}"\
    .format(input_gaussian_latlon_lsmask_filename))
    print("Writing output to: {0}".format(output_regular_latlon_mask_filename))
    cdo_instance.setname("field_value",
                         input=cdo_instance.remaplaf(regular_grid_spacing_file,
                                                     input=input_gaussian_latlon_lsmask_filename),
                         output=output_regular_latlon_mask_filename)

def generate_gaussian_landsea_mask(input_lsmask_filename,output_gaussian_latlon_mask_filename,
                                   gaussian_grid_spacing):
    """Generate a lat-lon gaussian landsea mask from a regular mask (lat-lon or otherwise) using cdos

    Arguments:
    input_lsmask_filename: string; full path to the input land-sea mask file to generate a gaussian land-sea
        mask from; this can be any format that the cdos will recognize.
    output_gaussian_latlon_mask_filename: string; full path to the target file for the output gaussian land-sea
        mask
    gaussian_grid_spacing: integer, the number of latitude lines between the pole and equator.
    Returns: nothing

    Uses the python wrapper to cdos provided by developers. Input file can any input grid allowed by cdos, output
    grid is global lat-lon gaussian as defined by cdos.
    """

    cdo_instance = cdo.Cdo()
    print("Generate gaussian land-sea mask from input mask: {0}".format(input_lsmask_filename))
    print("Writing output to: {0}".format(output_gaussian_latlon_mask_filename))
    cdo_instance.remapnn('n{0}'.format(gaussian_grid_spacing),input=input_lsmask_filename,
                         output=output_gaussian_latlon_mask_filename)

def insert_new_landsea_mask_into_jsbach_restart_file(input_landsea_mask_filename,input_js_bach_filename,
                                                     output_modified_js_bach_filename,
                                                     modify_fractional_lsm=False,
                                                     modify_lake_mask=False):
    """Insert a new landsea mask into a jsbach restart file

    Arguments:
    input_landsea_mask_filename: string; full path to new input landsea mask to insert
        into jsbach file
    input_js_bach_filename: string; full path to jsbach file to insert landsea mask into
    output_modified_js_bach_filename: string; full target path to write the new jsbach file to
    modify_fractional_lsm: boolean; also modify the fractional land sea mask (replace it the
        input update landsea mask that may not be fractional!?)?
    modify_lake_mask: boolean, set the lake mask to zero?
    Returns: nothing

    Uses python cdos. Always modifies the bindary landsea mask, will also modify the fractional
    land sea mask and/or set the lake mask to zero if boolean options are set accordingly.
    """

    cdo_instance = cdo.Cdo()
    temp_lsm_file = cdo_instance.chname("field_value,slm",input=input_landsea_mask_filename)
    if modify_fractional_lsm or modify_lake_mask:
        temp_output_file = cdo_instance.replace(input=" ".join([input_js_bach_filename,temp_lsm_file]))
    else:
        cdo_instance.replace(input=" ".join([input_js_bach_filename,temp_lsm_file]),
                             output=output_modified_js_bach_filename)
        return
    if modify_fractional_lsm:
        temp_lsm_file = cdo_instance.chname("field_value,slf",input=input_landsea_mask_filename)
        if modify_lake_mask:
            temp_output_file2 = cdo_instance.replace(input=" ".join([temp_output_file,temp_lsm_file]))
        else:
            cdo_instance.replace(input=" ".join([temp_output_file,temp_lsm_file]),
                                 output=output_modified_js_bach_filename)
            return
    #This if test is technically unnecessary but added for clarity/future proofing
    if modify_lake_mask:
        temp_lsm_file = cdo_instance.chname("field_value,lake",input=input_landsea_mask_filename)
        temp_lsm_file2 = cdo_instance.setclonlatbox("0.0,180.0,-180.0,90.0,-90.0",input=temp_lsm_file)
        cdo_instance.replace(input=" ".join([temp_output_file2 if modify_lake_mask else temp_output_file,
                                             temp_lsm_file2]),
                             output=output_modified_js_bach_filename)


def rebase_orography(orography,present_day_base_orography,present_day_reference_orography):
    """Change the present day basis orography of an orography for any timeslice

    Arguments:
    orography: field; an orography for a particular timeslice
    present_day_base_orography: field; the present day base that the orography comes from
    present_day_reference_orography: field, the present day reference orography to switch
    the input orography's base orography to
    """

    orography.subtract(present_day_base_orography)
    orography.add(present_day_reference_orography)
    return orography

def rebase_orography_driver(orography_filename,present_day_base_orography_filename,
                            present_day_reference_orography_filename,
                            rebased_orography_filename,orography_fieldname,
                            grid_type="HD",**grid_kwargs):
    """Driver changing the present day basis orography of an orography for any timeslice

    Arguments:
    orography_filename: string; the full path to the orography for a particular timeslice
    present_day_base_orography_filename: string; the full path to the present day base that
    the orography comes from
    present_day_reference_orography: string; the full path to the present day reference orography to switch
    the input orography's base orography to
    rebased_orography_filename: string; the full target path to save the rebased orography to
    orography_fieldname: string; name of the orography field in the orography and base orography files
    if it is non-standard(optional)
    """

    orography = iodriver.load_field(orography_filename,
                                    file_type=iodriver.get_file_extension(orography_filename),
                                    field_type="Orography", unmask=True,fieldname=orography_fieldname,
                                    grid_type=grid_type,**grid_kwargs)
    present_day_base_orography = iodriver.load_field(present_day_base_orography_filename,
                                                     file_type=iodriver.\
                                                     get_file_extension(present_day_base_orography_filename),
                                                     field_type="Orography", unmask=True,
                                                     fieldname=orography_fieldname,
                                                     grid_type=grid_type,**grid_kwargs)
    present_day_reference_orography = iodriver.load_field(present_day_reference_orography_filename,
                                                          file_type=iodriver.\
                                                          get_file_extension(present_day_reference_orography_filename),
                                                          field_type="Orography", unmask=True,grid_type=grid_type,
                                                           **grid_kwargs)
    rebased_orography = rebase_orography(orography, present_day_base_orography, present_day_reference_orography)
    iodriver.write_field(rebased_orography_filename,
                         field=rebased_orography,
                         file_type=iodriver.get_file_extension(rebased_orography_filename))

def advanced_rebase_orography_driver(orography_filename,present_day_base_orography_filename,
                                     present_day_reference_orography_filename,
                                     rebased_orography_filename,orography_fieldname,
                                     present_day_base_orography_fieldname,
                                     present_day_reference_orography_fieldname,
                                     rebased_orography_fieldname):
    orography = iodriver.advanced_field_loader(orography_filename,
                                               field_type="Orography",
                                               fieldname=orography_fieldname)
    present_day_base_orography = iodriver.advanced_field_loader(present_day_base_orography_filename,
                                                                field_type="Orography",
                                                                fieldname=\
                                                                present_day_base_orography_fieldname)
    present_day_reference_orography = iodriver.\
        advanced_field_loader(present_day_reference_orography_filename,
                              field_type="Orography",
                              fieldname=\
                              present_day_reference_orography_fieldname)
    rebased_orography = rebase_orography(orography, present_day_base_orography, present_day_reference_orography)
    iodriver.advanced_field_writer(rebased_orography_filename,
                                   field=rebased_orography,
                                   fieldname=rebased_orography_fieldname)

def convert_hydrosheds_river_directions(input_rdirs_field):
  output_rdirs_field = input_rdirs_field.convert_hydrosheds_rdirs()
  return output_rdirs_field

def advanced_convert_hydrosheds_river_directions_driver(input_river_directions_filename,
                                                        output_river_directions_filename,
                                                        input_river_directions_fieldname,
                                                        output_river_directions_fieldname):
  input_rdirs = iodriver.advanced_field_loader(input_river_directions_filename,
                                               field_type="RiverDirections",
                                               fieldname=input_river_directions_fieldname)
  output_rdirs = convert_hydrosheds_river_directions(input_rdirs)
  iodriver.advanced_field_writer(output_river_directions_filename,
                                 field=output_rdirs,
                                 fieldname=output_river_directions_fieldname)

def splice_rdirs(rdirs_matching_ls_mask,ls_mask,
                 other_rdirs):
  other_rdirs.remove_river_mouths()
  rdirs_matching_ls_mask.remove_river_mouths()
  other_rdirs.mark_ocean_points(ls_mask)
  other_rdirs.fill_land_without_rdirs(rdirs_matching_ls_mask)
  other_rdirs.mark_river_mouths()
  return other_rdirs

def advanced_splice_rdirs_driver(rdirs_matching_ls_mask_filename,
                                 ls_mask_filename,
                                 other_rdirs_filename,
                                 output_river_directions_filename,
                                 rdirs_matching_ls_mask_fieldname,
                                 ls_mask_fieldname,
                                 other_rdirs_fieldname,
                                 output_river_directions_fieldname):
  rdirs_matching_ls_mask =  iodriver.advanced_field_loader(rdirs_matching_ls_mask_filename,
                                                           field_type="RiverDirections",
                                                           fieldname=rdirs_matching_ls_mask_fieldname)
  ls_mask = iodriver.advanced_field_loader(ls_mask_filename,
                                           field_type="Generic",
                                           fieldname=ls_mask_fieldname)
  other_rdirs = iodriver.advanced_field_loader(other_rdirs_filename,
                                               field_type="RiverDirections",
                                               fieldname=other_rdirs_fieldname)
  spliced_rdirs = splice_rdirs(rdirs_matching_ls_mask,ls_mask,other_rdirs)
  iodriver.advanced_field_writer(output_river_directions_filename,
                                 field=spliced_rdirs,
                                 fieldname=output_river_directions_fieldname)

def remove_endorheic_basins(rdirs,catchments,rdirs_without_endorheic_basins,
                            replace_only_catchments=[],exclude_catchments=[]):
  if replace_only_catchments:
    print("Only replacing catchments:")
    print(replace_only_catchments)
  if exclude_catchments:
    print("Not replacing catchments")
    print(exclude_catchments)
  rdirs.remove_river_mouths()
  rdirs_without_endorheic_basins.remove_river_mouths()
  if replace_only_catchments:
    catchments_to_replace = replace_only_catchments
  else:
    catchments_to_replace = rdirs.find_endorheic_catchments(catchments)
  catchments_to_replace = [catchment for catchment in catchments_to_replace if catchment not in exclude_catchments]
  rdirs.replace_specified_catchments(catchments,catchments_to_replace,
                                     rdirs_without_endorheic_basins)
  rdirs.mark_river_mouths()
  return rdirs

def remove_endorheic_basins_driver(rdirs_filename,catchments_filename,
                                   rdirs_without_endorheic_basins_filename,
                                   output_rdirs_filename,
                                   rdirs_fieldname,catchment_fieldname,
                                   rdirs_without_endorheic_basins_fieldname,
                                   output_rdirs_fieldname,
                                   replace_only_catchments=[],
                                   exclude_catchments=[]):
  rdirs = iodriver.advanced_field_loader(rdirs_filename,
                                         field_type="RiverDirections",
                                         fieldname=rdirs_fieldname)
  catchments = iodriver.advanced_field_loader(catchments_filename,
                                              field_type="Generic",
                                              fieldname=catchment_fieldname)
  rdirs_without_endorheic_basins = \
    iodriver.advanced_field_loader(rdirs_without_endorheic_basins_filename,
                                   field_type="RiverDirections",
                                   fieldname=rdirs_without_endorheic_basins_fieldname)
  sinkless_rdirs = remove_endorheic_basins(rdirs,catchments,rdirs_without_endorheic_basins,
                                           replace_only_catchments=replace_only_catchments,
                                           exclude_catchments=exclude_catchments)
  iodriver.advanced_field_writer(output_rdirs_filename,field=sinkless_rdirs,
                                 fieldname=output_rdirs_fieldname)

def replace_streams_downstream_from_loop(rdirs,cumulative_flow,other_rdirs):
  rdirs.remove_river_mouths()
  other_rdirs.remove_river_mouths()
  mask = follow_streams_driver.follow_streams(other_rdirs,cumulative_flow)
  rdirs.replace_areas_in_mask(mask=mask.get_data(),other_rdirs=other_rdirs)
  rdirs.mark_river_mouths()
  return rdirs

def replace_streams_downstream_from_loop_driver(rdirs_filename,
                                                cumulative_flow_filename,
                                                other_rdirs_filename,
                                                output_rdirs_filename,
                                                rdirs_fieldname,
                                                cumulative_flow_fieldname,
                                                other_rdirs_fieldname,
                                                output_rdirs_fieldname):
  rdirs = iodriver.advanced_field_loader(rdirs_filename,
                                         field_type="RiverDirections",
                                         fieldname=rdirs_fieldname)
  cumulative_flow = iodriver.advanced_field_loader(cumulative_flow_filename,
                                                   field_type="CumulativeFlow",
                                                   fieldname=cumulative_flow_fieldname)
  other_rdirs = iodriver.advanced_field_loader(other_rdirs_filename,
                                               field_type="RiverDirections",
                                               fieldname=other_rdirs_fieldname)
  new_rdirs = replace_streams_downstream_from_loop(rdirs,cumulative_flow,other_rdirs)
  iodriver.advanced_field_writer(output_rdirs_filename,field=new_rdirs,
                                 fieldname=output_rdirs_fieldname)


class LineTypes(Enum):
    NEWENTRY = enum.auto()
    LATLONDEF = enum.auto()
    LATLONVALUES = enum.auto()
    DATEANDHEIGHTDEF = enum.auto()
    DATEANDHEIGHTVALUES = enum.auto()
    NOTSET = enum.auto()

def apply_date_based_sill_height_corrections(orography_field,
                                             date_based_sill_height_corrections_list_filename,
                                             current_date):
    """
    """
    version_number_pattern = re.compile(r"# Version")
    most_recent_only_full_pattern = re.compile(r"# *most_recent_only_full")
    additive_condensed_pattern = re.compile(r"# *additive_condensed")
    new_entry_line_pattern = re.compile(r"-*new *entry*-")
    latlon_line_pattern = re.compile(r"^ *lat *, *lon")
    latlon_line_values_pattern = re.compile(r"^ *[0-9]+ *, *[0-9]+")
    date_and_height_line_pattern= re.compile(r"^ *end *date, *height")
    date_and_height_line_values_pattern = re.compile(r"^ *-?[0-9]+ *, *[0-9]+\.[0-9]*")
    comment_line_pattern = re.compile(r"^ *#.*$")
    blank_line_pattern = re.compile(r"^ *$")
    correction_list = []
    dates_and_heights= None
    coords = None
    with open(date_based_sill_height_corrections_list_filename) as f:
        first_line = f.readline().strip('\n')
        if version_number_pattern.match(first_line):
            first_line = f.readline().strip('\n')
        if most_recent_only_full_pattern.match(first_line):
            most_recent_only = True
            previous_line_type = LineTypes.NOTSET
            for line in f:
                if comment_line_pattern.match(line):
                    continue
                elif new_entry_line_pattern.match(line):
                    if (previous_line_type == LineTypes.NOTSET or
                        previous_line_type == LineTypes.DATEANDHEIGHTVALUES):
                        previous_line_type = LineTypes.NEWENTRY
                        if previous_line_type == LineTypes.DATEANDHEIGHTVALUES:
                            correction_list.append([coords,dates_and_heights])
                        coords = None
                        dates_and_heights= {}
                    else:
                        raise RunTimeError("Invalid sill correction file format")
                elif previous_line_type == LineTypes.NEWENTRY:
                    if latlon_line_pattern.match(line):
                        previous_line_type == LineTypes.LATLONDEF
                    else:
                        raise RunTimeError("Invalid sill correction file format")
                elif previous_line_type == LineTypes.LATLONDEF:
                    if latlon_line_values_pattern.match(line):
                        coords = tuple(int(coord) for coord in line.strip().split(","))
                        previous_line_type == LineTypes.LATLONVALUES
                    else:
                        raise RuntimeError("Invalid sill correction file format")
                elif previous_line_type == LineTypes.LATLONVALUES:
                    if date_and_height_line_pattern.match(line):
                        previous_line_type == LineTypes.DATEANDHEIGHTDEF
                    else:
                        raise RuntimeError("Invalid sill correction file format")
                elif (previous_line_type == LineTypes.DATEANDHEIGHTDEF or
                         previous_line_type == LineTypes.DATEANDHEIGHTVALUES):
                    if date_and_height_line_values_pattern.match(line):
                        date_str,height_str = line.strip().split(",")
                        date = int(date_str)
                        height_str = float(height_str)
                        dates_and_heights[date] = height
                        previous_line_type == LineTypes.DATEANDHEIGHTVALUES
                    else:
                        raise RuntimeError("Invalid sill correction file format")
                else:
                    raise RuntimeError("Invalid sill correction file format")
            if (previous_line_type != LineTypes.NOTSET and
                previous_line_type != LineTypes.DATEANDHEIGHTVALUES):
                raise RuntimeError("Invalid sill correction file format")
            if previous_line_type == LineTypes.DATEANDHEIGHTVALUES:
                correction_list.append([coords,dates_and_heights])
        elif additive_condensed_pattern.match(first_line):
            most_recent_only = False
            corrections = []
            for line in f:
                if (comment_line_pattern.match(line) or
                    blank_line_pattern.match(line)):
                    continue
                data = line.strip().split(",")
                correction_list.append({"lat":int(data[0]),
                                        "lon":int(data[1]),
                                        "date":int(data[2]),
                                        "height_change":float(data[3])})
        else:
            raise RuntimeError("Corrections file format not recognised")
    if most_recent_only:
        for coords,heights_and_dates in correction_list:
            oldest_date_less_than_current_date = \
                min([date for date in list(heights_and_dates.keys()) if date >= current_date])
            height = heights_and_dates[oldest_date_less_than_current_date]
            print("Correcting height of lat={0},lon={1} to {2} m at date {3}".format(*coords,height,
                                                                                     current_date))
            orography_field.get_data()[coords] += height
    else:
        for correction in correction_list:
            if current_date > correction["date"]:
                print("Correcting height of lat={0},lon={1}"
                      " by {2} m at date {3}".format(correction["lat"],correction["lon"],
                                                     correction["height_change"],
                                                     current_date))
                orography_field.get_data()[correction["lat"],correction["lon"]] += correction["height_change"]

def expand_catchment_to_include_rivermouths(rdirs,catchments,mouth_coords):
    dir_to_centre = [[3,2,1],
                     [6,5,4],
                     [9,8,7]]
    assigned_catchment_number = 0
    mouth_neighbors = rdirs[mouth_coords[0]-1:mouth_coords[0]+2,
                            mouth_coords[1]-1:mouth_coords[1]+2]
    catchments_slice = catchments[mouth_coords[0]-1:mouth_coords[0]+2,
                                  mouth_coords[1]-1:mouth_coords[1]+2]
    for nbr_coords in np.argwhere(mouth_neighbors == dir_to_centre):
        if assigned_catchment_number != 0:
            nbr_catchment = catchments_slice[tuple(nbr_coords)]
            catchments[catchments == nbr_catchment] = assigned_catchment_number
        else:
            assigned_catchment_number = catchments_slice[tuple(nbr_coords)]
            catchments[tuple(mouth_coords)] = assigned_catchment_number

def write_fields_for_debug(fields,workdir):
    for fieldname,field in fields.items():
        iodriver.advanced_field_writer(path.join(workdir,f"{fieldname}_temp.nc"),
                                       field,
                                       fieldname=fieldname)

