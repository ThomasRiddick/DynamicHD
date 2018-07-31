'''
High level routines of dynamic hydrology script.
Created on Dec 15, 2015

@author: triddick
'''

import numpy as np
import argparse
import os
import iohelper
import field as fld

def generate_flow_directions_from_orography(orography):
    """Generates flow direction from a given orography field object.

    Arguments:
    orography: an Orography object; the orography to generate flow direction for
    Returns:
    A field object containing the flow directions

    Retreives the a grid object from the orography field object given. Uses
    the compute_flow_directions function of the orography to compute the flow
    directions using the data from the orography field object. If the orography was
    masked return the flow direction as a masked field object with the same mask
    as the orography otherwise simply return the flow direction as a field object
    without any mask. In either case the retrieved grid is passed to the flow
    directions field object that is returned.
    """

    grid = orography.get_grid()
    flow_directions = orography.compute_flow_directions()
    if type(orography.get_data() is np.ma.MaskedArray):
        return fld.Field(np.ma.array(flow_directions,
                                     mask=np.ma.getmaskarray(orography.get_data())),
                         grid)
    else:
        return fld.Field(flow_directions,grid)

def load_field(filename,file_type,field_type,unmask=True,timeslice=None,
               fieldname=None,check_for_grid_info=False,grid_type='HD',**grid_kwargs):
    """Inteface that loads a field as a raw array of data and returns it as a field object.

    Arguments:
    filename: string; the full path to the file to opened
    file_type: string; the code for this file type
    field_type: string; the code for this field type
    unmask (optional): boolean; flag to specifying whether the field is returned
            with any mask removed (True) or not (False)
    timeslice: integer; which timeslice to select from a (netCDF4) file with multiple time
        slices
    fieldname: string; which field to select from a (netCDF4) file with multiple fields
    check_for_grid_info: boolean; Search to see if file has grid info and use this
            to replace grid type specified if found
    grid_type: string; the code for this grid type
    **grid_kwargs: dictionary; key word arguments specifying parameters of the grid
    Returns:
    A field of object or an object of a field subclass

    Uses the getFileHelper pseudo-factory function to get the appropriate IOHelper subclass
    to load this file type and then uses this to loads the file; passing in any grid_kwargs
    supplied. Uses makeField factory function to create a field of the requested type with
    the loaded data and returns this object. Use a list to retrieve any grid information
    found by the loading function and add it to the field.
    """

    grid_info=[]
    raw_field = iohelper.getFileHelper(file_type).load_field(filename,unmask,timeslice,fieldname,
                                                             check_for_grid_info,grid_info,
                                                             grid_type,**grid_kwargs)
    if len(grid_info) == 0:
        return fld.makeField(raw_field,field_type,grid_type,**grid_kwargs)
    else:
        return fld.makeField(raw_field,field_type,grid_info[0])

def write_field(filename,field,file_type,griddescfile=None,
                fieldname=None):
    """Writes the given field object to the given file type.

    Arguments:
    filename: string; the full path of the target file
    field: field object; the field object containing the data to be written
    file_type: string; the code of the type of file that is to be written
    griddescfile (optional): string; full path to the grid description metadata to add to
            file written out (if possible). Nothing is added if this is set to
            None
    fieldname(optional): string; fieldname to use in a (netCDF4) file

    Uses the getFileHeper pseudo-factory function to get the appropriate IOHelper subclass
    to write this file type and then uses this to write the field object.
    """

    iohelper.getFileHelper(file_type).write_field(filename,field,griddescfile=griddescfile,
                                                  fieldname=fieldname)

def get_file_extension(filename):
    """Return the extension of a given filename"""
    return os.path.splitext(filename)[1]

def get_field_mask(field,grid_type,**grid_kwargs):
    """Get the mask from a field and return it cast as 1's and 0's inside a Field instance"""
    return fld.Field(field.get_mask().astype(np.int64),grid_type,
                     **grid_kwargs)

def filter_kwargs(kwargs):
    """Seperate keyword arguments between grad_change_kwargs and grid_kwargs"""
    grad_change_kwargs_prefix='gc_'
    grad_change_kwargs = {keyword: kwargs[keyword] for keyword in kwargs.keys()
                          if grad_change_kwargs_prefix in keyword}
    grid_kwargs = {keyword: kwargs[keyword] for keyword in kwargs.keys()
                          if not grad_change_kwargs_prefix in keyword}
    return grid_kwargs, grad_change_kwargs

def main(new_orography_file,
         grid_type,
         base_orography_file=None,
         corrected_base_orography_file=None,
         updated_orography_file=None,
         base_RFD_file=None,
         updated_RFD_file=None,
         update_mask_file=None,
         recompute_changed_orography_only=True,
         recompute_significant_gradient_changes_only=False,
         **kwargs):
    """The main high-level routine of the dynamic hd script.

    Arguments:
    new_orography_file: string; the full path of the new orography
    grid_type: string; the code for the type of grid to be used
    base_orography_file (optional): string; the full path of the base present day
        orography - this is required if the recompute changed orography only flag
        is set to True
    corrected_base_orography_file (optional): string; the full path of the present day
        orography including the modern day river direction corrections - this is required
        if the recompute changed orography only flag is set to True
    updated_orography_file (optional): string; the target full path to the write the
        updated orography to
    base_RFD_file (optional): string; the full path to the file contain the base modern
    day river directions (including manual corrections)
    updated_RFD_file: string; the target full path to write the updated river direction to
    recompute_changed_orography_only: boolean; a flag that select whether the orography
        is recomputed everywhere or only where the orography has changed
    **kwargs: dictionary; keyword arguments containing grid parameters and parameters for
        determining significant gradient changes (if that option has been set). The later
        should all start with gc_ to allow them to be seperated easily from the former.

    A description of the actions of this routine is given by imbedded comments.
    """

    #Check all files required are defined if recompute_changed_orography_only is True
    if recompute_changed_orography_only and (base_orography_file is None or
                                             corrected_base_orography_file is None or
                                             updated_orography_file is None or
                                             base_RFD_file is None or
                                             updated_RFD_file is None):
        raise RuntimeError('Not all files required for --recomputed-changed-orography-only option are present')
    if recompute_significant_gradient_changes_only and (base_orography_file is None or
                                                        base_RFD_file is None or
                                                        updated_RFD_file is None):
        raise RuntimeError('Not all files required for --recompute-significant-gradient-changes-only option are present')
    #seperate the parameters for determining signficant gradient changes (if any) from the
    #grid parameters
    grid_kwargs, grad_changes_kwargs = filter_kwargs(kwargs)
    #load the new orography
    new_orography =  load_field(new_orography_file,
                                file_type=get_file_extension(new_orography_file),
                                field_type='Orography',
                                grid_type=grid_type,
                                **grid_kwargs)
    #The step in this block are only if we are recomputing flow directions either only for changed orography or only
    #only for orography where the gradient has significantly changed
    if recompute_changed_orography_only or recompute_significant_gradient_changes_only:
        #load the original orography
        base_orography = load_field(base_orography_file,
                                    file_type=get_file_extension(base_orography_file),
                                    field_type='Orography',
                                    grid_type=grid_type,
                                    **grid_kwargs)

        #load the original flow directions
        flow_directions = load_field(base_RFD_file,
                                    file_type=get_file_extension(base_RFD_file)+'int',
                                    field_type='Generic',
                                    grid_type=grid_type,
                                    **grid_kwargs)
        #mask the new orography where it is unchanged from the original orography. If the
        #option to only recalculate the flow direction where significant changes in
        #gradient occur pass that to the function using a keyword argument
        new_orography.mask_new_orography_using_base_orography(base_orography,
                                                              use_gradient=\
                                                              recompute_significant_gradient_changes_only,
                                                              **grad_changes_kwargs)
        #Creating a hybrid orography doesn't work if we are  recomputing only significant
        #gradient changes... the corrected orography and the new orography could be very
        #different and a hybrid would include lots of unphysical features. We also don't
        #extend the mask for recompute_significant_gradient_changes_only
        if not recompute_significant_gradient_changes_only:
            #load the corrected base orography file. (Note this will differ substantially
            #from both the new orography and also the base orography - it will be the
            #base orography with a series of correction applied; it will nearly correspond
            #with the original flow directions)
            corrected_base_orography  = load_field(corrected_base_orography_file,
                                                   file_type=get_file_extension(corrected_base_orography_file),
                                                   field_type='Orography',
                                                   grid_type=grid_type,
                                                   **grid_kwargs)
            #Before extending mask (note the order is important) we update the corrected base
            #orography with the masked new orography to get a hybrid orography comprising the
            #new orography where changes have occurred and the corrected base orography where
            #there are no changes
            corrected_base_orography.update_field_with_partially_masked_data(new_orography)
            #extend the unmasked region to include the neighbour of changed grid cells
            new_orography.extend_mask()
            #Write out updated orography
            write_field(updated_orography_file,corrected_base_orography,
                    file_type=get_file_extension(updated_orography_file))
        #generate new flow directions and mask them with the same mask as the new orography
        new_flow_directions = generate_flow_directions_from_orography(new_orography)
        #update the old flow directions with the new flow directions where they are not masked
        flow_directions.update_field_with_partially_masked_data(new_flow_directions)
        #Write out updated flow directions
        write_field(updated_RFD_file, flow_directions, file_type=get_file_extension(updated_RFD_file))
        #If a update mask file has been specified get the mask from the new orography (not the flow
        #direction which have by this point filled their mask) and write it to this file
        if update_mask_file is not None:
            #Get the mask and place it in field object.
            update_mask = get_field_mask(new_orography,grid_type,**grid_kwargs)
            #Write out the update mask
            write_field(update_mask_file,update_mask,
                        file_type=get_file_extension(update_mask_file))
    #This else block is for if we are recomputing flow direction everywhere
    else:
        #generate new flow directions and mask them with the same mask as the new orography
        new_flow_directions = generate_flow_directions_from_orography(new_orography)
        #Write out updated flow directions
        write_field(updated_RFD_file, new_flow_directions,
                    file_type=get_file_extension(updated_RFD_file))

class Arguments(object):
    """An empty class used to pass namelist arguments into the main routine as keyword arguments."""

    pass

def parse_arguments():
    """Parse the command line arguments using the argparse module.

    Returns:
    An Arguments object containing the comannd line arguments.
    """

    args = Arguments()
    parser = argparse.ArgumentParser("Update river flow directions")
    parser.add_argument('new_orography_file',
                        metavar='new-orography-file',
                        help='Update orography file',
                        type=str)
    parser.add_argument('grid_type',
                        metavar='grid-type',
                        help='Specify a grid type',
                        type=str)
    parser.add_argument('-b','--base-orography-file',
                        help='Original base orography file',
                        type=str,
                        default='base_orography.dat')
    parser.add_argument('-c','--corrected-base-orography-file',
                        help='Original corrected orography file',
                        type=str,
                        default='original_corrected_orography.dat')
    parser.add_argument('-r','--base-RFD-file',
                        help='Original river flow directions file',
                        type=str,
                        default='base_RFD_file.dat')
    parser.add_argument('-p','--updated-orography-file',
                        help='Target location for updated orography',
                        type=str,
                        default='orography_updated.dat')
    parser.add_argument('-u','--updated-RFD-file',
                        help='Target location for updated river flow directions',
                        type=str,
                        default='rivdir_updated.dat')
    parser.add_argument('-m','--update-mask-file',
                        help='Target location for update mask',
                        type=str,
                        default='update_mask.dat')
    parser.add_argument('-o','--recompute-changed-orography-only',
                        help='Recompute only regions where is a change in orography',
                        action='store_true')
    parser.add_argument('-g','--recompute-significant-gradient-changes-only',
                        help='Recompute only regions where the gradient of the orography is significantly changed',
                        type=str,
                        default='store_false')
    parser.add_argument('--gc-absolute-tol',
                        help='Absolute tolerance to use if --recompute-significant-gradient-changes-only is selected',
                        type=float,
                        default=5.0)
    parser.add_argument('--gc-relative-tol',
                        help='Relative tolerance to use if --recompute-significant-gradient-changes-only is selected',
                        type=float,
                        default=None)
    #Adding the variables to a namespace other than that of the parser keeps the namespace clean
    #and allows us to pass it directly to main
    parser.parse_args(namespace=args)
    return args

if __name__ == '__main__':
    #Parse arguments and then run main
    args = parse_arguments()
    main(**vars(args))
