'''
A module containing top level functions (currently only one) related to finding and mark the river mouths 
in a flow direction field 

Created on Apr 18, 2016

@author: thomasriddick
'''
import dynamic_hd
import field

def main(rdirs_filepath,updatedrdirs_filepath,lsmask_filepath=None,
         flowtocell_filepath=None,rivermouths_filepath=None,
         flowtorivermouths_filepath=None,skip_marking_mouths=False,
         flip_mask_ud=False,grid_type='HD',**grid_kwargs):
    """Top level function to drive the river mouth marking process
    
    Arguments:
    rdirs_filepath: string; full path to input river directions file
    updatedrdirs_filepath: string; full path to target output river directions file
    lsmask_filepath(optional): string; full path to land sea mask filepath
    flowtocell_filepath(optional): string; full path to cumulative flow input file path. Requires
        flowtorivermouths_filepath to be defined for this option to be meaningful; if it is not
        will raise a warning
    rivermouths_filepath(optional): string; full path to optional target river mouths output
        file - if used this will create a file where the position of river mouths is marked 
        True and all other points are marked false
    flowtorivermouths_filepath(optional): string; full path to optional target flow to river mouth
        output file that will contain 0 everywhere expect at river mouths where it will contain the
        cumulative flow to that river mouth
    skip_marking_mouths(optional): boolean; if this flag is set to True then don't mark river mouths
        and perform only the additional task otherwise (if it is False) proceed as normal
    flip_mask_ud: boolean; flip the landsea mask (if any) upside down before processing?
    grid_type: string; keyword specifying the grid type
    **grid_kwargs: dictionary of keyword arguments; parameters for the specified grid type if required
    Return: Nothing
    
    Additional tasks are producing a river mouths output file and producing a flow to river mouth output 
    file.
    """

    #Load files
    rdirs_field = dynamic_hd.load_field(filename=rdirs_filepath,
                                        file_type=dynamic_hd.get_file_extension(rdirs_filepath),
                                        field_type="RiverDirections",
                                        grid_type=grid_type,**grid_kwargs)
    if lsmask_filepath:
        lsmask = dynamic_hd.load_field(filename=lsmask_filepath,
                                        file_type=dynamic_hd.get_file_extension(lsmask_filepath),
                                        field_type="Generic",
                                        grid_type=grid_type,**grid_kwargs)
        if flip_mask_ud:
            lsmask.flip_data_ud()
    else:
        lsmask = None
        
    if flowtocell_filepath:
        flowtocell_field = dynamic_hd.load_field(filename=flowtocell_filepath,
                                        file_type=dynamic_hd.get_file_extension(flowtocell_filepath),
                                        field_type="CumulativeFlow",
                                        grid_type=grid_type,**grid_kwargs)
    else:
        flowtocell_field = None

    if not skip_marking_mouths:
        #Perform the actual marking of river mouths
        rdirs_field.mark_river_mouths(lsmask.get_data() if lsmask is not None else None)
       
        #Write out the updated river directions field 
        dynamic_hd.write_field(filename=updatedrdirs_filepath, 
                               field=rdirs_field,
                               file_type=dynamic_hd.get_file_extension(updatedrdirs_filepath))

    #Perform any additional tasks if any: either making a seperate river mouths mask file or
    #generating a file with the cumulative flow to each river mouth or both (or potentially neither)
    if rivermouths_filepath or (flowtocell_field is not None):
        rivermouth_field = field.makeField(rdirs_field.get_river_mouths(),'Generic',grid_type,
                                           **grid_kwargs)
        if flowtocell_field is not None:
            if flowtorivermouths_filepath:
                flowtorivermouths_field = field.makeField(flowtocell_field.\
                                                          find_cumulative_flow_at_outlets(rivermouth_field.\
                                                                                          get_data()),
                                                          'Generic',grid_type,
                                                          **grid_kwargs)
                                                          
                dynamic_hd.write_field(flowtorivermouths_filepath,flowtorivermouths_field,
                                       file_type=dynamic_hd.get_file_extension(flowtorivermouths_filepath))
            else:
                raise UserWarning("Flow to cell file specified but no target flow to river"
                                  " mouth target file defined; not processing the flow to"
                                  " cell field")
        if rivermouths_filepath:
            dynamic_hd.write_field(rivermouths_filepath,rivermouth_field,
                                   file_type=dynamic_hd.get_file_extension(rivermouths_filepath))